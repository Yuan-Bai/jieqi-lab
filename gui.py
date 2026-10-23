"""揭棋 GUI（Tkinter）。

人机/AI 对战、树形棋谱回放和变例、明/暗子摆棋。
提示/胜率可随时开关，后台分析默认 200 模拟，按节点缓存并取消过时任务。
胜率由根价值换算（和棋折半）；分析使用行动方信息，棋谱保留原观看视角。

音效为程序化生成的 WAV（assets/sounds/），首次运行自动生成，可直接替换同名文件。
"""

from __future__ import annotations

import array
import copy
import json
import math
import os
import random
import sys
import time
import tempfile
import tkinter as tk
import wave
from tkinter import filedialog, messagebox, ttk

import ismcts as im
import jieqi as jq
from game_record import GameRecord
from gui_tasks import LatestTaskRunner, SearchBelief
from position_editor import PositionEditor

CELL = 60
BOARD_W = jq.BOARD_W * CELL
BOARD_H = jq.BOARD_H * CELL
MAIN_GEOMETRY = (580, 660)
GAME_GEOMETRY = (BOARD_W + 460, BOARD_H + 210)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")
SOUND_DIR = os.path.join(PROJECT_ROOT, "assets", "sounds")
REPLAY_DIR = os.path.join(PROJECT_ROOT, "data", "replays")

AGENT_RIS = "ris"
AGENT_AZ = "az"
AGENT_LABELS = {AGENT_RIS: "RIS-ISMCTS", AGENT_AZ: "AlphaZero"}
AGENT_BY_LABEL = {label: key for key, label in AGENT_LABELS.items()}
RIS_SIM_CHOICES = ("400", "800", "1600", "3200")
AZ_SIM_CHOICES = ("100", "200", "400", "800", "1600", "3200")
DEVICE_CHOICES = ("auto", "cuda", "cpu")
SPEED_CHOICES = ("快", "正常", "慢")
SPEED_DELAYS = {"快": 0, "正常": 450, "慢": 1200}
ANALYSIS_SIMS = 200
AZ_LEAF_BATCH_SIZE = 64
LEAF_BATCH_CHOICES = ("1", "8", "16", "32", "64", "128")
AZ_CHANCE_NODES = True
AZ_CHANCE_K = 0  # 0=全量枚举揭示候选

MODE_HUMAN = "human"
MODE_AIAI = "aiai"
MODE_STUDY = "study"
STUDY_TIP = "回放可走子添加变例；未固定的暗子在新变例中重新抽样，已有走法复用原结果；分叉处选择编号。"


def parse_leaf_batch_size(value) -> int:
    text = str(value).strip()
    if not text.isdecimal() or not 1 <= int(text) <= 100000:
        raise ValueError("叶批大小必须是 1～100000 的整数")
    return int(text)

APP_BG = "#24211d"
CARD_BG = "#332e27"
CARD_EDGE = "#4a4234"
FIELD_BG = "#413a30"
TEXT = "#f2e9d8"
MUTED = "#b3a68e"
ACCENT = "#d9a84e"
ACCENT_ACTIVE = "#eac272"
ACCENT_TEXT = "#241b0c"
BTN_BG = "#4a4437"
BTN_ACTIVE = "#5d5544"
RED_SIDE = "#e2685f"
BLACK_SIDE = "#9db4c8"

BOARD_TOP = "#eedcb4"
BOARD_BOTTOM = "#d8bc84"
BOARD_LINE = "#7a5836"
BOARD_LINE_DARK = "#5c3f24"

LAST_FROM = "#e2685f"
LAST_TO = "#f4c95d"
HINT_COLOR = "#3fc7d4"

PIECE_NAMES = {
    jq.RED: {
        jq.TYPE_KING: "帅",
        jq.TYPE_ROOK: "车",
        jq.TYPE_KNIGHT: "马",
        jq.TYPE_ELEPHANT: "相",
        jq.TYPE_ADVISOR: "仕",
        jq.TYPE_CANNON: "炮",
        jq.TYPE_PAWN: "兵",
    },
    jq.BLACK: {
        jq.TYPE_KING: "将",
        jq.TYPE_ROOK: "车",
        jq.TYPE_KNIGHT: "马",
        jq.TYPE_ELEPHANT: "象",
        jq.TYPE_ADVISOR: "士",
        jq.TYPE_CANNON: "炮",
        jq.TYPE_PAWN: "卒",
    },
}

SAMPLE_RATE = 22050


def piece_text(piece: int) -> str:
    label = PIECE_NAMES[jq.piece_color(piece)][jq.piece_type(piece)]
    if jq.is_dark(piece):
        return f"暗{label}"
    return label


def default_model_path() -> str:
    for name in ("latest.pt", "best.pt"):
        path = os.path.join(MODEL_DIR, name)
        if os.path.isfile(path):
            return path
    return ""


def ensure_replay_dir() -> str:
    """确保回放默认目录存在，返回文件对话框可用的初始目录。"""
    try:
        os.makedirs(REPLAY_DIR, exist_ok=True)
        return REPLAY_DIR
    except OSError:
        return PROJECT_ROOT


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(v))) for v in rgb)


def _blend(color_a: str, color_b: str, t: float) -> str:
    a = _hex_to_rgb(color_a)
    b = _hex_to_rgb(color_b)
    return _rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def _tone(freq: float, duration: float, amp: float = 0.5, decay: float = 10.0) -> list[float]:
    count = int(SAMPLE_RATE * duration)
    return [amp * math.sin(2 * math.pi * freq * i / SAMPLE_RATE) * math.exp(-decay * i / SAMPLE_RATE) for i in range(count)]


def _noise(duration: float, amp: float = 0.3, decay: float = 60.0, seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    count = int(SAMPLE_RATE * duration)
    return [amp * (rng.random() * 2 - 1) * math.exp(-decay * i / SAMPLE_RATE) for i in range(count)]


def _mix(*tracks: list[float]) -> list[float]:
    length = max(len(track) for track in tracks)
    out = [0.0] * length
    for track in tracks:
        for i, value in enumerate(track):
            out[i] += value
    return out


def _seq(*tracks: list[float]) -> list[float]:
    out: list[float] = []
    for track in tracks:
        out.extend(track)
    return out


def _build_select() -> list[float]:
    return _tone(1650, 0.045, 0.22, 60.0)


def _build_move() -> list[float]:
    return _mix(_tone(430, 0.10, 0.55, 32.0), _noise(0.03, 0.22, 90.0))


def _build_capture() -> list[float]:
    return _mix(_tone(196, 0.16, 0.7, 18.0), _tone(330, 0.11, 0.35, 24.0), _noise(0.05, 0.3, 60.0))


def _build_check() -> list[float]:
    return _seq(_tone(1046, 0.09, 0.42, 9.0), _tone(1397, 0.17, 0.38, 7.0))


def _build_win() -> list[float]:
    return _seq(*[_tone(freq, 0.14, 0.45, 6.0) for freq in (523, 659, 784, 1046)])


def _build_lose() -> list[float]:
    return _seq(*[_tone(freq, 0.17, 0.4, 5.0) for freq in (392, 330, 262)])


def _build_draw() -> list[float]:
    return _seq(_tone(440, 0.14, 0.38, 6.0), _tone(440, 0.2, 0.32, 5.0))


def _write_wav(path: str, samples: list[float]) -> None:
    data = array.array("h", (max(-32767, min(32767, int(value * 32767))) for value in samples))
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(data.tobytes())


class SoundManager:
    """程序化生成并播放音效；非 Windows 或生成失败时静默降级。"""

    BUILDERS = {
        "select": _build_select,
        "move": _build_move,
        "capture": _build_capture,
        "check": _build_check,
        "win": _build_win,
        "lose": _build_lose,
        "draw": _build_draw,
    }

    def __init__(self) -> None:
        self.module = None
        if sys.platform == "win32":
            try:
                import winsound

                self.module = winsound
            except Exception:
                self.module = None
        if self.module is not None:
            try:
                self._ensure_files()
            except Exception:
                self.module = None

    def _ensure_files(self) -> None:
        os.makedirs(SOUND_DIR, exist_ok=True)
        for name, builder in self.BUILDERS.items():
            path = os.path.join(SOUND_DIR, f"{name}.wav")
            if not os.path.isfile(path):
                _write_wav(path, builder())

    def play(self, name: str) -> None:
        if self.module is None:
            return
        path = os.path.join(SOUND_DIR, f"{name}.wav")
        if not os.path.isfile(path):
            return
        try:
            self.module.PlaySound(
                path,
                self.module.SND_FILENAME | self.module.SND_ASYNC | self.module.SND_NODEFAULT,
            )
        except Exception:
            pass


class RisAgent:
    """RIS-ISMCTS 对手：纯搜索，无需神经网络。"""

    def __init__(self, simulations: int, rng: random.Random) -> None:
        self.simulations = simulations
        self.agent = im.ISMCTSAgent(im.ISMCTSConfig(iterations=simulations), rng)

    def select_move(self, state, belief, history):
        move, _ = self.agent.select_move(state, belief)
        return move


class AzAgent:
    """AlphaZero 对手：PUCT + 根信息采样 + 合成行动者信息集。"""

    def __init__(self, model_path: str, simulations: int, device_str: str, rng: random.Random) -> None:
        import torch

        import azero

        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.net = azero.PolicyValueNet()
        if not azero.load_model_if_exists(self.net, model_path, self.device):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
        self.net.to(self.device)
        self.net.eval()
        self.config = azero.AZConfig(
            mcts_simulations=simulations,
            chance_nodes=AZ_CHANCE_NODES,
            chance_k=AZ_CHANCE_K,
        )
        self.mcts = azero.NeuralMCTS(self.net, self.config, self.device, rng)

    def select_move(self, state, belief, history):
        _, move = self.mcts.search(
            state,
            belief,
            history,
            temperature=1e-6,
            simulations=self.config.mcts_simulations,
            add_dirichlet=False,
        )
        return move

    def analyse(self, state, belief, history, simulations: int | None = None) -> tuple[tuple[int, int] | None, float]:
        """返回 (推荐着法, 根玩家视角价值)：价值取搜索根节点均值，范围 [-1, 1]。"""
        _, move, root = self.mcts.search(
            state,
            belief,
            history,
            temperature=1e-6,
            simulations=simulations or self.config.mcts_simulations,
            add_dirichlet=False,
            return_root=True,
        )
        visits = root.visits
        value = root.value_sum / visits if visits > 0 else 0.0
        return move, value


class AgentPool:
    """后台对象不持有 Tk 控件，避免搜索结束后在工作线程析构 Tk 对象。"""
    def __init__(self):
        self.cache = {}

    @staticmethod
    def settings_key(settings):
        key = tuple(sorted(settings.items()))
        path = settings.get("model_path")
        if path:
            try:
                stat = os.stat(path)
                key += (("model_stamp", (stat.st_mtime_ns, stat.st_size)),)
            except OSError:
                pass
        return key

    def get(self, settings, lane, rng):
        # 两个通道各自串行执行；在工作线程搜索开始前更新参数，不重载权重。
        is_az = settings["agent"] == AGENT_AZ
        leaf_batch = parse_leaf_batch_size(settings.get("leaf_batch_size", AZ_LEAF_BATCH_SIZE)) if is_az else None
        engine_settings = {k: v for k, v in settings.items()
                           if not is_az or k not in ("sims", "leaf_batch_size")}
        key = (lane, self.settings_key(engine_settings))
        agent = self.cache.get(key)
        if agent is None:
            if settings["agent"] == AGENT_RIS:
                agent = RisAgent(settings["sims"], rng)
            else:
                agent = AzAgent(settings["model_path"], settings["sims"], settings.get("device", "auto"), rng)
            for old in list(self.cache):
                if old[0] == lane:
                    self.cache.pop(old, None)
            self.cache[key] = agent
        if is_az:
            agent.config.mcts_simulations = settings["sims"]
            agent.config.leaf_batch_size = leaf_batch
        return agent


class JieQiGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Jieqi Lab · 揭棋")
        self.root.resizable(False, False)
        self.root.configure(bg=APP_BG)
        self._setup_style()

        self.rng = random.Random()
        self.sounds = SoundManager()
        self.belief = None
        self.human_side = None
        self.ai_side = None
        self.side_settings: dict[int, dict | None] = {}
        self.game_kind = MODE_HUMAN
        self.ai_delay_ms = 0
        self.paused = False
        self.ai_thinking = False
        self._ai_token = 0
        self._agents = AgentPool()
        self._ai_runner = LatestTaskRunner()
        self._analysis_runner = LatestTaskRunner()
        self._analysis_cache = {}
        self._analysis_key = None
        self._analysis_after = None
        self._dirty = False
        self.record: GameRecord | None = None
        self.setup_open = False

        self.state = None
        self.history = []
        self.selected_idx = None
        self.selected_targets = set()
        self.selected_captures = set()
        self.last_move = None
        self.move_count = 0
        self.captured = {jq.RED: [], jq.BLACK: []}

        self.flipped = False
        self.god_view = True
        self.last_move_note = ""
        self.hint_move = None
        self.analysis_settings: dict | None = None
        self.hint_on = False
        self.winrate_on = False
        self.analysis_sims = ANALYSIS_SIMS
        self.analysis_leaf_batch_size = AZ_LEAF_BATCH_SIZE
        self.analysis_thinking = False
        self._analysis_token = 0

        self.move_log: list[dict] = []
        self.replay_mode = False
        self.replay_meta: dict | None = None
        self.replay_index = 0
        self._branch_visible = False
        self._tree_sync = False
        self._tree_record = None

        self.main_frame = tk.Frame(root, bg=APP_BG, padx=26, pady=20)
        self.game_host = tk.Frame(root, bg=APP_BG)
        self.game_host.rowconfigure(0, weight=1)
        self.game_host.columnconfigure(0, weight=1)
        self.game_viewport = tk.Canvas(self.game_host, bg=APP_BG, highlightthickness=0)
        self.game_viewport.grid(row=0, column=0, sticky="nsew")
        self.game_frame = tk.Frame(self.game_viewport, bg=APP_BG, padx=14, pady=14)
        self.game_viewport.create_window(0, 0, window=self.game_frame, anchor="nw")
        self.game_frame.bind("<Configure>", self._content_layout_changed)
        self.game_vscroll = ttk.Scrollbar(self.game_host, orient="vertical", command=self.game_viewport.yview)
        self.game_hscroll = ttk.Scrollbar(self.game_host, orient="horizontal", command=self.game_viewport.xview)
        self.game_viewport.configure(yscrollcommand=self.game_vscroll.set, xscrollcommand=self.game_hscroll.set)
        self.game_viewport.bind("<MouseWheel>", lambda e: self.game_viewport.yview_scroll(-int(e.delta / 120), "units"))
        self._build_menu()
        self._build_game_panel()
        self.root.bind("<Left>", lambda _event: self._replay_step(-1))
        self.root.bind("<Right>", lambda _event: self._replay_step(1))
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        for number in range(1, 10):
            self.root.bind(str(number), lambda _event, n=number: self._choose_branch(n))

        self.main_frame.pack()
        self._set_geometry(*MAIN_GEOMETRY)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "App.TRadiobutton",
            background=APP_BG,
            foreground=TEXT,
            focuscolor=APP_BG,
            font=("Microsoft YaHei", 11),
        )
        style.map("App.TRadiobutton", background=[("active", APP_BG)], foreground=[("active", ACCENT)])
        style.configure(
            "App.TCombobox",
            fieldbackground=FIELD_BG,
            background=BTN_BG,
            foreground=TEXT,
            arrowcolor=TEXT,
            bordercolor=CARD_EDGE,
            lightcolor=CARD_EDGE,
            darkcolor=CARD_EDGE,
        )
        style.map(
            "App.TCombobox",
            fieldbackground=[("readonly", FIELD_BG), ("disabled", FIELD_BG)],
            foreground=[("disabled", MUTED)],
            selectbackground=[("readonly", FIELD_BG)],
            selectforeground=[("readonly", TEXT)],
        )
        self.root.option_add("*TCombobox*Listbox.background", FIELD_BG)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", ACCENT_TEXT)
        style.configure("Record.Treeview", background=FIELD_BG, fieldbackground=FIELD_BG,
                        foreground=TEXT, rowheight=24, font=("Microsoft YaHei", 9), borderwidth=0)
        style.map("Record.Treeview", background=[("selected", BTN_ACTIVE)])
        style.configure("App.TButton", background=BTN_BG, foreground=TEXT, padding=(10, 7),
                        font=("Microsoft YaHei", 10), borderwidth=0)
        style.map("App.TButton", background=[("active", BTN_ACTIVE)], foreground=[("active", ACCENT)])

    def _button(self, parent, text: str, command, primary: bool = False, width: int | None = None) -> tk.Button:
        bg = ACCENT if primary else BTN_BG
        fg = ACCENT_TEXT if primary else TEXT
        active = ACCENT_ACTIVE if primary else BTN_ACTIVE
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active,
            activeforeground=fg,
            disabledforeground=MUTED,
            relief="flat",
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2",
            font=("Microsoft YaHei", 10),
        )
        if width is not None:
            button.configure(width=width)
        return button

    def _checkbutton(self, parent, text: str, var: tk.BooleanVar, bg: str = CARD_BG, command=None) -> tk.Checkbutton:
        return tk.Checkbutton(
            parent,
            text=text,
            variable=var,
            command=command,
            bg=bg,
            fg=MUTED,
            activebackground=bg,
            activeforeground=ACCENT,
            selectcolor=FIELD_BG,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            font=("Microsoft YaHei", 10),
        )

    def _card(self, parent, title: str) -> tuple[tk.Frame, tk.Frame]:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=CARD_EDGE, highlightthickness=1)
        tk.Label(frame, text=title, bg=CARD_BG, fg=ACCENT, font=("Microsoft YaHei", 11, "bold")).pack(
            anchor="w", padx=12, pady=(8, 2)
        )
        body = tk.Frame(frame, bg=CARD_BG)
        body.pack(fill="x", padx=12, pady=(2, 10))
        return frame, body

    def _row_label(self, parent, text: str) -> tk.Label:
        return tk.Label(parent, text=text, bg=CARD_BG, fg=MUTED, font=("Microsoft YaHei", 10))

    def _combo(self, parent, var, values, width: int = 12) -> ttk.Combobox:
        return ttk.Combobox(parent, textvariable=var, values=values, width=width, style="App.TCombobox")

    def _build_menu(self) -> None:
        header = tk.Frame(self.main_frame, bg=APP_BG)
        header.pack(fill="x", pady=(0, 14))
        tk.Label(header, text="揭  棋", bg=APP_BG, fg=ACCENT, font=("Microsoft YaHei", 26, "bold")).pack()
        tk.Label(header, text="暗棋对弈 · RIS-ISMCTS / AlphaZero", bg=APP_BG, fg=MUTED, font=("Microsoft YaHei", 10)).pack(
            pady=(2, 0)
        )

        mode_row = tk.Frame(self.main_frame, bg=APP_BG)
        mode_row.pack(fill="x", pady=(0, 10))
        tk.Label(mode_row, text="对局模式", bg=APP_BG, fg=TEXT, font=("Microsoft YaHei", 11, "bold")).pack(side="left")
        self.mode_var = tk.StringVar(value=MODE_HUMAN)
        ttk.Radiobutton(
            mode_row,
            text="人机对战",
            variable=self.mode_var,
            value=MODE_HUMAN,
            command=self._on_mode_change,
            style="App.TRadiobutton",
        ).pack(side="left", padx=(14, 8))
        ttk.Radiobutton(
            mode_row,
            text="AI 对战",
            variable=self.mode_var,
            value=MODE_AIAI,
            command=self._on_mode_change,
            style="App.TRadiobutton",
        ).pack(side="left")

        self.card_area = tk.Frame(self.main_frame, bg=APP_BG)
        self.card_area.pack(fill="x")

        self.human_card, human_body = self._card(self.card_area, "人机参数")
        self.human_agent_var = tk.StringVar(value=AGENT_LABELS[AGENT_RIS])
        self.human_sims_var = tk.StringVar(value=RIS_SIM_CHOICES[1])
        human_grid = tk.Frame(human_body, bg=CARD_BG)
        human_grid.pack(fill="x")
        human_grid.columnconfigure(3, weight=1)
        self._row_label(human_grid, "对手").grid(row=0, column=0, sticky="w")
        self.human_agent_box = self._combo(human_grid, self.human_agent_var, tuple(AGENT_LABELS.values()), width=11)
        self.human_agent_box.grid(row=0, column=1, sticky="w", padx=(6, 14))
        self._row_label(human_grid, "模拟").grid(row=0, column=2, sticky="w")
        self.human_sims_box = self._combo(human_grid, self.human_sims_var, RIS_SIM_CHOICES, width=8)
        self.human_sims_box.grid(row=0, column=3, sticky="w", padx=(6, 0))
        self.human_agent_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._sync_agent_row(self.human_agent_var, self.human_sims_var, self.human_sims_box),
        )
        self.human_az_frame = tk.Frame(human_body, bg=CARD_BG)
        self.human_model_row, self.human_model_box = self._model_row(self.human_az_frame)
        self.human_device_row, self.human_device_box = self._device_row(self.human_az_frame)
        self.human_model_row.pack(fill="x")
        self.human_device_row.pack(fill="x", pady=(6, 0))
        self.human_leaf_batch_var = tk.StringVar(value=str(AZ_LEAF_BATCH_SIZE))
        self._leaf_batch_row(self.human_az_frame, self.human_leaf_batch_var).pack(fill="x", pady=(6, 0))

        self.human_analysis_opts = tk.Frame(human_body, bg=CARD_BG)
        self.human_analysis_opts.pack(fill="x", pady=(6, 0))
        self.hint_var = tk.BooleanVar(value=False)
        self._checkbutton(
            self.human_analysis_opts, "AI 提示", self.hint_var, command=self._refresh_az_visibility
        ).pack(side="left")
        self.winrate_var = tk.BooleanVar(value=False)
        self._checkbutton(
            self.human_analysis_opts, "胜率（AlphaZero）", self.winrate_var, command=self._refresh_az_visibility
        ).pack(side="left", padx=(12, 0))
        self.human_analysis_frame = tk.Frame(human_body, bg=CARD_BG)
        self.analysis_model_row, self.analysis_model_box = self._model_row(self.human_analysis_frame, label="分析模型")
        self.analysis_model_row.pack(fill="x", pady=(4, 0))

        self.ai_card, ai_body = self._card(self.card_area, "AI 对战参数")
        self.red_agent_var = tk.StringVar(value=AGENT_LABELS[AGENT_RIS])
        self.red_sims_var = tk.StringVar(value=RIS_SIM_CHOICES[1])
        self.black_agent_var = tk.StringVar(value=AGENT_LABELS[AGENT_AZ])
        self.black_sims_var = tk.StringVar(value=AZ_SIM_CHOICES[2])
        ai_grid = tk.Frame(ai_body, bg=CARD_BG)
        ai_grid.pack(fill="x")
        ai_grid.columnconfigure(3, weight=1)
        self._row_label(ai_grid, "红方").grid(row=0, column=0, sticky="w", pady=2)
        self.red_agent_box = self._combo(ai_grid, self.red_agent_var, tuple(AGENT_LABELS.values()), width=11)
        self.red_agent_box.grid(row=0, column=1, sticky="w", padx=(6, 14), pady=2)
        self._row_label(ai_grid, "模拟").grid(row=0, column=2, sticky="w", pady=2)
        self.red_sims_box = self._combo(ai_grid, self.red_sims_var, RIS_SIM_CHOICES, width=8)
        self.red_sims_box.grid(row=0, column=3, sticky="w", padx=(6, 0), pady=2)
        self._row_label(ai_grid, "黑方").grid(row=1, column=0, sticky="w", pady=2)
        self.black_agent_box = self._combo(ai_grid, self.black_agent_var, tuple(AGENT_LABELS.values()), width=11)
        self.black_agent_box.grid(row=1, column=1, sticky="w", padx=(6, 14), pady=2)
        self._row_label(ai_grid, "模拟").grid(row=1, column=2, sticky="w", pady=2)
        self.black_sims_box = self._combo(ai_grid, self.black_sims_var, AZ_SIM_CHOICES, width=8)
        self.black_sims_box.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=2)
        self.red_agent_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._sync_agent_row(self.red_agent_var, self.red_sims_var, self.red_sims_box),
        )
        self.black_agent_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._sync_agent_row(self.black_agent_var, self.black_sims_var, self.black_sims_box),
        )
        self.ai_az_frame = tk.Frame(ai_body, bg=CARD_BG)
        self.ai_model_row, self.ai_model_box = self._model_row(self.ai_az_frame)
        self.ai_device_row, self.ai_device_box = self._device_row(self.ai_az_frame)
        self.ai_model_row.pack(fill="x")
        self.ai_device_row.pack(fill="x", pady=(6, 0))
        self.red_leaf_batch_var = tk.StringVar(value=str(AZ_LEAF_BATCH_SIZE))
        self.black_leaf_batch_var = tk.StringVar(value=str(AZ_LEAF_BATCH_SIZE))
        batch_row = tk.Frame(self.ai_az_frame, bg=CARD_BG)
        batch_row.pack(fill="x", pady=(6, 0))
        self._leaf_batch_row(batch_row, self.red_leaf_batch_var, "红方叶批").pack(side="left", padx=(0, 16))
        self._leaf_batch_row(batch_row, self.black_leaf_batch_var, "黑方叶批").pack(side="left")
        self.speed_label = tk.Label(ai_body, text="速度", bg=CARD_BG, fg=MUTED, font=("Microsoft YaHei", 10))
        self.speed_label.pack(anchor="w", pady=(6, 0))
        self.speed_var = tk.StringVar(value="正常")
        self.speed_box = self._combo(ai_body, self.speed_var, SPEED_CHOICES, width=8)
        self.speed_box.pack(anchor="w", pady=(2, 0))
        self.ai_winrate_var = tk.BooleanVar(value=False)
        self._checkbutton(
            ai_body, "胜率（AlphaZero）", self.ai_winrate_var, command=self._refresh_az_visibility
        ).pack(anchor="w", pady=(6, 0))
        self.god_var = tk.BooleanVar(value=True)
        self._checkbutton(ai_body, "上帝视角（AI 对战显示全部信息）", self.god_var).pack(anchor="w", pady=(6, 0))

        self.start_area = tk.Frame(self.main_frame, bg=APP_BG)
        self.start_area.pack(fill="x")
        self.human_start_row = tk.Frame(self.start_area, bg=APP_BG)
        self._button(self.human_start_row, "执红先手", lambda: self.start_human_game(jq.RED), primary=True).pack(
            side="left", padx=(0, 10)
        )
        self._button(self.human_start_row, "执黑后手", lambda: self.start_human_game(jq.BLACK), primary=True).pack(
            side="left"
        )
        self.ai_start_row = tk.Frame(self.start_area, bg=APP_BG)
        self._button(self.ai_start_row, "开始 AI 对战", self.start_ai_game, primary=True).pack()

        footer = tk.Frame(self.main_frame, bg=APP_BG)
        footer.pack(fill="x", pady=(16, 0))
        self.sound_var = tk.BooleanVar(value=True)
        self._checkbutton(footer, "音效", self.sound_var, bg=APP_BG).pack(side="left")
        self._button(footer, "退出", self._close, width=6).pack(side="right")
        self._button(footer, "加载历史对局", self.load_game_file, width=13).pack(side="right", padx=(0, 10))
        self._button(footer, "摆棋", self.open_position_editor, width=6).pack(side="right", padx=(0, 8))

        self._on_mode_change()

    def _model_row(self, parent, label: str = "模型") -> tuple[tk.Frame, ttk.Combobox]:
        row = tk.Frame(parent, bg=CARD_BG)
        self._row_label(row, label).pack(side="left")
        var = tk.StringVar(value=default_model_path())
        box = self._combo(
            row,
            var,
            (os.path.join(MODEL_DIR, "latest.pt"), os.path.join(MODEL_DIR, "best.pt")),
            width=24,
        )
        box.pack(side="left", fill="x", expand=True, padx=(6, 6))
        box.var = var
        self._button(row, "浏览", lambda: self._browse_model(box)).pack(side="left")
        return row, box

    def _leaf_batch_row(self, parent, var, label="叶批大小"):
        row = tk.Frame(parent, bg=CARD_BG)
        self._row_label(row, label).pack(side="left", padx=(0, 6))
        self._combo(row, var, LEAF_BATCH_CHOICES, width=8).pack(side="left")
        return row

    def _device_row(self, parent) -> tuple[tk.Frame, ttk.Combobox]:
        row = tk.Frame(parent, bg=CARD_BG)
        self._row_label(row, "设备").pack(side="left")
        var = tk.StringVar(value=DEVICE_CHOICES[0])
        box = self._combo(row, var, DEVICE_CHOICES, width=8)
        box.configure(state="readonly")
        box.pack(side="left", padx=(6, 0))
        box.var = var
        return row, box

    def _sync_agent_row(self, agent_var: tk.StringVar, sims_var: tk.StringVar, sims_box: ttk.Combobox) -> None:
        agent = AGENT_BY_LABEL.get(agent_var.get(), AGENT_RIS)
        choices = AZ_SIM_CHOICES if agent == AGENT_AZ else RIS_SIM_CHOICES
        sims_box.configure(values=choices)
        if sims_var.get() not in choices:
            sims_var.set(AZ_SIM_CHOICES[2] if agent == AGENT_AZ else RIS_SIM_CHOICES[1])
        self._refresh_az_visibility()

    def _browse_model(self, box: ttk.Combobox) -> None:
        initial = MODEL_DIR if os.path.isdir(MODEL_DIR) else PROJECT_ROOT
        path = filedialog.askopenfilename(
            title="选择模型文件",
            initialdir=initial,
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")],
        )
        if path:
            box.var.set(path)

    def _active_agent_vars(self) -> list[tk.StringVar]:
        if self.mode_var.get() == MODE_AIAI:
            return [self.red_agent_var, self.black_agent_var]
        return [self.human_agent_var]

    def _refresh_az_visibility(self) -> None:
        if self.mode_var.get() == MODE_AIAI:
            self.human_card.pack_forget()
            self.human_start_row.pack_forget()
            self.ai_card.pack(fill="x", pady=(0, 4))
            self.ai_start_row.pack(pady=(12, 0))
        else:
            self.ai_card.pack_forget()
            self.ai_start_row.pack_forget()
            self.human_card.pack(fill="x", pady=(0, 4))
            self.human_start_row.pack(pady=(12, 0))
        needs_model = (any(var.get() == AGENT_LABELS[AGENT_AZ] for var in self._active_agent_vars())
                       or (self.mode_var.get() == MODE_AIAI and self.ai_winrate_var.get()))
        if needs_model:
            self.human_az_frame.pack(fill="x", pady=(6, 0))
            self.ai_az_frame.pack(fill="x", pady=(6, 0), before=self.speed_label)
        else:
            self.human_az_frame.pack_forget()
            self.ai_az_frame.pack_forget()
        if self.mode_var.get() == MODE_HUMAN and (self.hint_var.get() or self.winrate_var.get()):
            self.human_analysis_frame.pack(fill="x", pady=(6, 0))
        else:
            self.human_analysis_frame.pack_forget()

    def _on_mode_change(self) -> None:
        self._refresh_az_visibility()

    def _build_game_panel(self) -> None:
        self.left_col = tk.Frame(self.game_frame, bg=APP_BG)
        self.left_col.grid(row=0, column=0, padx=(0, 12), sticky="nw")
        self.canvas = tk.Canvas(
            self.left_col,
            width=BOARD_W,
            height=BOARD_H,
            bg=BOARD_TOP,
            highlightbackground=CARD_EDGE,
            highlightthickness=2,
        )
        self.canvas.pack(anchor="nw")
        self.canvas.bind("<Button-1>", self.on_click)

        panel = tk.Frame(self.game_frame, bg=CARD_BG, padx=16, pady=14, highlightbackground=CARD_EDGE, highlightthickness=1)
        panel.grid(row=0, column=1, sticky="nw")

        self.header_var = tk.StringVar(value="")
        tk.Label(panel, textvariable=self.header_var, bg=CARD_BG, fg=MUTED, font=("Microsoft YaHei", 10), wraplength=380, justify="left").pack(anchor="w")

        self.status_var = tk.StringVar(value="红方走子")
        self.status_label = tk.Label(panel, textvariable=self.status_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei", 17, "bold"))
        self.status_label.pack(anchor="w", pady=(6, 2))

        self.turn_var = tk.StringVar(value="回合: 1")
        tk.Label(panel, textvariable=self.turn_var, bg=CARD_BG, fg=MUTED, font=("Microsoft YaHei", 10)).pack(anchor="w")

        self.last_move_var = tk.StringVar(value="上一步：-")
        tk.Label(panel, textvariable=self.last_move_var, bg=CARD_BG, fg=LAST_TO, font=("Microsoft YaHei", 11, "bold"), wraplength=380, justify="left").pack(anchor="w", pady=(4, 0))

        self.hint_text_var = tk.StringVar(value="")
        tk.Label(panel, textvariable=self.hint_text_var, bg=CARD_BG, fg=HINT_COLOR, font=("Microsoft YaHei", 10), wraplength=380, justify="left").pack(anchor="w")

        self.ai_info_var = tk.StringVar(value="")
        tk.Label(panel, textvariable=self.ai_info_var, bg=CARD_BG, fg=MUTED, font=("Microsoft YaHei", 9), wraplength=380, justify="left").pack(anchor="w", pady=(2, 0))

        self._panel_separator(panel)
        tk.Label(panel, text="被吃棋子", bg=CARD_BG, fg=ACCENT, font=("Microsoft YaHei", 11, "bold")).pack(anchor="w")
        self.captured_left_var = tk.StringVar(value="-")
        self.captured_right_var = tk.StringVar(value="-")
        captured_cols = tk.Frame(panel, bg=CARD_BG)
        captured_cols.pack(fill="x", pady=(4, 0))
        captured_left = tk.Frame(captured_cols, bg=CARD_BG, width=186)
        captured_right = tk.Frame(captured_cols, bg=CARD_BG, width=186)
        captured_left.pack(side="left", anchor="n", fill="x", expand=True)
        captured_right.pack(side="left", anchor="n", fill="x", expand=True)
        self.captured_left_title = tk.Label(captured_left, text="红方吃子", bg=CARD_BG, fg=RED_SIDE, font=("Microsoft YaHei", 10))
        self.captured_left_title.pack(anchor="w")
        tk.Label(captured_left, textvariable=self.captured_left_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei", 9), wraplength=180, justify="left").pack(anchor="w")
        self.captured_right_title = tk.Label(captured_right, text="黑方吃子", bg=CARD_BG, fg=BLACK_SIDE, font=("Microsoft YaHei", 10))
        self.captured_right_title.pack(anchor="w")
        tk.Label(captured_right, textvariable=self.captured_right_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei", 9), wraplength=180, justify="left").pack(anchor="w")

        self._panel_separator(panel)
        self.pool_title_var = tk.StringVar(value="暗盒信息")
        tk.Label(panel, textvariable=self.pool_title_var, bg=CARD_BG, fg=ACCENT, font=("Microsoft YaHei", 11, "bold")).pack(anchor="w")
        self.pool_red_var = tk.StringVar(value="-")
        self.pool_black_var = tk.StringVar(value="-")
        pool_cols = tk.Frame(panel, bg=CARD_BG)
        pool_cols.pack(fill="x", pady=(4, 0))
        pool_left = tk.Frame(pool_cols, bg=CARD_BG, width=186)
        pool_right = tk.Frame(pool_cols, bg=CARD_BG, width=186)
        pool_left.pack(side="left", anchor="n")
        pool_right.pack(side="left", anchor="n")
        tk.Label(pool_left, text="红方暗子", bg=CARD_BG, fg=RED_SIDE, font=("Microsoft YaHei", 10)).pack(anchor="w")
        tk.Label(pool_left, textvariable=self.pool_red_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei", 9), wraplength=180, justify="left").pack(anchor="w")
        tk.Label(pool_right, text="黑方暗子", bg=CARD_BG, fg=BLACK_SIDE, font=("Microsoft YaHei", 10)).pack(anchor="w")
        tk.Label(pool_right, textvariable=self.pool_black_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei", 9), wraplength=180, justify="left").pack(anchor="w")

        self._panel_separator(panel)
        tk.Label(panel, text="AI 评估", bg=CARD_BG, fg=ACCENT, font=("Microsoft YaHei", 11, "bold")).pack(anchor="w")
        self.eval_var = tk.StringVar(value="未启用（在菜单勾选「胜率」后显示）")
        tk.Label(panel, textvariable=self.eval_var, bg=CARD_BG, fg=TEXT, font=("Microsoft YaHei", 10), wraplength=380, justify="left").pack(anchor="w", pady=(4, 0))
        self.eval_canvas = tk.Canvas(panel, width=380, height=12, bg=FIELD_BG, highlightthickness=0)
        self.eval_canvas.pack(anchor="w", pady=(4, 0))
        self._update_eval_bar(None)
        analysis_controls = tk.Frame(panel, bg=CARD_BG)
        analysis_controls.pack(fill="x", pady=(4, 0))
        self.live_hint_var = tk.BooleanVar(value=False)
        self.live_winrate_var = tk.BooleanVar(value=False)
        self._checkbutton(analysis_controls, "提示", self.live_hint_var, command=self._analysis_options_changed).pack(side="left")
        self._checkbutton(analysis_controls, "胜率", self.live_winrate_var, command=self._analysis_options_changed).pack(side="left")
        self._button(analysis_controls, "分析设置", self._analysis_settings_dialog).pack(side="right")
        self.analysis_status_var = tk.StringVar(value="")
        tk.Label(panel, textvariable=self.analysis_status_var, bg=CARD_BG, fg=MUTED,
                 font=("Microsoft YaHei", 9), wraplength=380, justify="left").pack(anchor="w")
        tk.Label(panel, text="分析按行动方已知信息；上帝视角仅影响界面显示。", bg=CARD_BG, fg=MUTED,
                 font=("Microsoft YaHei", 9), wraplength=380, justify="left").pack(anchor="w", pady=(2, 0))
        self._panel_separator(panel)
        record_header = tk.Frame(panel, bg=CARD_BG)
        record_header.pack(fill="x")
        tk.Label(record_header, text="棋谱 · 点击走子回看", bg=CARD_BG, fg=ACCENT,
                 font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        self.continue_btn = self._button(record_header, "从此对弈", self.continue_game)
        self.continue_btn.pack(side="right")
        tree_frame = tk.Frame(panel, bg=CARD_BG)
        tree_frame.pack(fill="x", pady=(4, 0))
        self.record_tree = ttk.Treeview(tree_frame, show="tree", height=5, style="Record.Treeview", selectmode="browse")
        self.record_tree.column("#0", width=350, stretch=True)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.record_tree.yview)
        self.record_tree.configure(yscrollcommand=tree_scroll.set)
        self.record_tree.pack(side="left", fill="x", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.record_tree.bind("<<TreeviewSelect>>", self._tree_selected)

        controls = tk.Frame(self.left_col, bg=APP_BG)
        controls.pack(anchor="w", pady=(10, 0))
        self.pause_btn = self._button(controls, "暂停", self.toggle_pause, width=8)
        self.restart_btn = self._button(controls, "重来", self.reset_game, width=8)
        self.flip_btn = self._button(controls, "翻转棋盘", self.toggle_flip, width=10)
        self.save_btn = self._button(controls, "保存对局", self.save_game_file, width=10)
        self.pause_btn.pack(side="left", padx=(0, 8))
        self.restart_btn.pack(side="left", padx=(0, 8))
        self.flip_btn.pack(side="left", padx=(0, 8))
        self.save_btn.pack(side="left", padx=(0, 8))

        controls2 = tk.Frame(self.left_col, bg=APP_BG)
        controls2.pack(anchor="w", pady=(8, 0))
        self.menu_btn = self._button(controls2, "返回菜单", self.back_to_menu, width=10)
        self.menu_btn.pack(side="left", padx=(0, 8))
        self.replay_btn = self._button(controls2, "回放本局", self.enter_replay_from_game, width=10)
        self.replay_btn.pack(side="left", padx=(0, 8))
        self._button(controls2, "摆棋", self.open_position_editor, width=6).pack(side="left", padx=(0, 6))
        self._checkbutton(controls2, "音效", self.sound_var, bg=APP_BG).pack(side="left", padx=6)
        self.play_batch_btn = self._button(controls2, "对弈叶批量", self._play_batch_settings_dialog, width=12)
        self.play_batch_btn.pack(side="left", padx=(6, 0))

        self.replay_row = tk.Frame(self.left_col, bg=APP_BG)
        self._button(self.replay_row, "开头", lambda: self._replay_goto(0), width=8).pack(side="left", padx=(0, 8))
        self._button(self.replay_row, "上一步", lambda: self._replay_step(-1), width=10).pack(side="left", padx=(0, 8))
        self.next_btn = self._button(self.replay_row, "下一步", lambda: self._replay_step(1), width=10)
        self.next_btn.pack(side="left", padx=(0, 8))
        self._button(self.replay_row, "结尾", self._replay_to_end, width=8).pack(side="left", padx=(0, 8))
        self.branch_row = tk.Frame(self.left_col, bg=APP_BG)
        self.study_tip_var = tk.StringVar(value=STUDY_TIP)
        self.study_tip = tk.Label(self.left_col, textvariable=self.study_tip_var, bg=APP_BG, fg=MUTED,
                                  font=("Microsoft YaHei", 9), wraplength=BOARD_W, justify="left")

        tk.Frame(panel, bg=CARD_BG, width=380, height=0).pack(anchor="w")

    def _panel_separator(self, parent) -> None:
        tk.Frame(parent, bg=CARD_EDGE, height=1).pack(fill="x", pady=6)

    def _collect_agent_settings(self, agent_var: tk.StringVar, sims_var: tk.StringVar, model_box: ttk.Combobox,
                                device_box: ttk.Combobox, leaf_batch_var=None) -> dict:
        agent = AGENT_BY_LABEL.get(agent_var.get())
        if agent is None:
            raise ValueError("未知的 AI 类型")
        sims_text = sims_var.get().strip()
        if not sims_text.isdigit() or int(sims_text) <= 0:
            raise ValueError("模拟次数必须是正整数")
        settings: dict = {"agent": agent, "sims": int(sims_text)}
        if agent == AGENT_AZ:
            model_path = model_box.var.get().strip()
            if not model_path:
                raise ValueError("请选择 AlphaZero 模型文件")
            if not os.path.isabs(model_path):
                model_path = os.path.join(PROJECT_ROOT, model_path)
            if not os.path.isfile(model_path):
                raise ValueError(f"模型文件不存在: {model_path}")
            settings["model_path"] = model_path
            settings["device"] = device_box.var.get()
            settings["leaf_batch_size"] = parse_leaf_batch_size(
                leaf_batch_var.get() if leaf_batch_var is not None else AZ_LEAF_BATCH_SIZE)
        return settings

    def _enter_game(self) -> None:
        self.main_frame.pack_forget()
        self.game_host.pack(fill="both", expand=True)
        self._fit_game_geometry()
        if self.game_kind == MODE_AIAI:
            self.pause_btn.configure(state="normal")
            self.pool_title_var.set("暗盒信息（上帝视角）" if self.god_view else "暗盒信息（观众视角）")
        else:
            self.pause_btn.configure(state="disabled")
            self.pool_title_var.set("暗盒信息")
        self.reset_game()
        self.live_hint_var.set(self.hint_on)
        self.live_winrate_var.set(self.winrate_on)

    def _collect_analysis_settings(self, mode: str) -> dict | None:
        """构造后台分析（提示/胜率）用的引擎设置；未启用返回 None。"""
        if mode == MODE_HUMAN:
            enabled = self.hint_var.get() or self.winrate_var.get()
            model_var = self.analysis_model_box.var
            allow_ris_fallback = self.hint_var.get() and not self.winrate_var.get()
        else:
            enabled = self.ai_winrate_var.get()
            model_var = self.ai_model_box.var
            allow_ris_fallback = False
        if not enabled:
            return None
        model_path = model_var.get().strip() or default_model_path()
        if not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)
        if not os.path.isfile(model_path):
            if allow_ris_fallback:
                return {"agent": AGENT_RIS, "sims": self.analysis_sims}
            raise ValueError(f"分析模型不存在: {model_path}\n请选择有效的 AlphaZero 模型文件。")
        return {
            "agent": AGENT_AZ,
            "sims": self.analysis_sims,
            "model_path": model_path,
            "device": "auto",
            "leaf_batch_size": self.analysis_leaf_batch_size,
        }

    def start_human_game(self, human_side: int) -> None:
        try:
            settings = self._collect_agent_settings(
                self.human_agent_var, self.human_sims_var, self.human_model_box, self.human_device_box,
                self.human_leaf_batch_var
            )
            analysis = self._collect_analysis_settings(MODE_HUMAN)
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.game_kind = MODE_HUMAN
        self.human_side = human_side
        self.ai_side = 1 - human_side
        self.side_settings = {human_side: None, self.ai_side: settings}
        self.analysis_settings = analysis
        self.hint_on = bool(self.hint_var.get())
        self.winrate_on = bool(self.winrate_var.get())
        self.god_view = False
        self.flipped = human_side == jq.BLACK
        self.ai_delay_ms = 0
        self.header_var.set(
            f"你执{'红方' if human_side == jq.RED else '黑方'} · 对手 {self._agent_label(settings)}"
        )
        self._enter_game()

    def start_ai_game(self) -> None:
        try:
            red_settings = self._collect_agent_settings(
                self.red_agent_var, self.red_sims_var, self.ai_model_box, self.ai_device_box, self.red_leaf_batch_var
            )
            black_settings = self._collect_agent_settings(
                self.black_agent_var, self.black_sims_var, self.ai_model_box, self.ai_device_box, self.black_leaf_batch_var
            )
            analysis = self._collect_analysis_settings(MODE_AIAI)
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.game_kind = MODE_AIAI
        self.human_side = None
        self.ai_side = None
        self.side_settings = {jq.RED: red_settings, jq.BLACK: black_settings}
        self.analysis_settings = analysis
        self.hint_on = False
        self.winrate_on = bool(self.ai_winrate_var.get())
        self.god_view = bool(self.god_var.get())
        self.flipped = False
        self.ai_delay_ms = SPEED_DELAYS.get(self.speed_var.get(), 450)
        red_label = self._agent_label(red_settings)
        black_label = self._agent_label(black_settings)
        self.header_var.set(f"红方 {red_label}   vs   黑方 {black_label}")
        self._enter_game()

    def _set_geometry(self, width: int, height: int) -> None:
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 2, 0)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _fit_game_geometry(self) -> None:
        """按内容布局；小屏幕提供滚动，避免底部操作被裁切。"""
        self.root.update_idletasks()
        width = max(GAME_GEOMETRY[0], self.game_frame.winfo_reqwidth())
        height = max(GAME_GEOMETRY[1], self.game_frame.winfo_reqheight())
        view_w = min(width, self.root.winfo_screenwidth() - 50)
        view_h = min(height, self.root.winfo_screenheight() - 100)
        self.game_viewport.configure(width=view_w, height=view_h, scrollregion=(0, 0, width, height))
        self.game_vscroll.grid_remove()
        self.game_hscroll.grid_remove()
        extra_w = extra_h = 0
        if view_h < height:
            self.game_vscroll.grid(row=0, column=1, sticky="ns")
            extra_w = self.game_vscroll.winfo_reqwidth()
        if view_w < width:
            self.game_hscroll.grid(row=1, column=0, sticky="ew")
            extra_h = self.game_hscroll.winfo_reqheight()
        self._set_geometry(view_w + extra_w, view_h + extra_h)

    def _content_layout_changed(self, _event):
        if not self.game_host.winfo_manager():
            return
        self.game_viewport.configure(scrollregion=(0, 0, self.game_frame.winfo_reqwidth(), self.game_frame.winfo_reqheight()))
        if self.game_frame.winfo_reqheight() > self.game_viewport.winfo_height() + 2:
            self.game_vscroll.grid(row=0, column=1, sticky="ns")
        if self.game_frame.winfo_reqwidth() > self.game_viewport.winfo_width() + 2:
            self.game_hscroll.grid(row=1, column=0, sticky="ew")

    def back_to_menu(self) -> None:
        if not self._confirm_leave():
            return
        self._cancel_searches()
        self.game_host.pack_forget()
        self.main_frame.pack()
        self._set_geometry(*MAIN_GEOMETRY)
        self.human_side = None
        self.ai_side = None
        self.belief = None
        self.side_settings = {}
        self.analysis_settings = None
        self.hint_on = False
        self.winrate_on = False
        self.hint_move = None
        self.analysis_thinking = False
        self.paused = False
        self.ai_thinking = False
        self.replay_mode = False
        self.replay_meta = None
        self.replay_index = 0
        self.replay_row.pack_forget()
        self.branch_row.pack_forget()
        self.study_tip.pack_forget()
        self.move_log = []
        self.state = None
        self.record = None
        self._dirty = False
        self._analysis_cache.clear()

    def reset_game(self) -> None:
        if self.record is not None and not self._confirm_leave():
            return
        self._cancel_searches()
        self.paused = False
        self.pause_btn.configure(text="暂停")
        self.replay_mode = False
        self.replay_meta = None
        self.replay_index = 0
        self.move_log = []
        self.replay_row.pack_forget()
        self.branch_row.pack_forget()
        self.study_tip.pack_forget()
        self.hint_move = None
        self.state = jq.GameState.initial(rng=self.rng)
        self.belief = im.init_belief_for_game(self.state)
        self.record = GameRecord(self.state, self.belief)
        self._dirty = False
        self._analysis_cache.clear()
        self.history = []
        self.selected_idx = None
        self.selected_targets = set()
        self.selected_captures = set()
        self.last_move = None
        self.last_move_note = ""
        self.move_count = 0
        self.captured = {jq.RED: [], jq.BLACK: []}
        self.ai_info_var.set("")
        self.last_move_var.set("上一步：-")
        self.hint_text_var.set("")
        self.eval_var.set(
            f"评估中…（{AGENT_LABELS[AGENT_AZ]}-{self.analysis_sims}）"
            if self.winrate_on
            else "未启用（在菜单勾选「胜率」后显示）"
        )
        self._update_eval_bar(None)
        self.update_captured_labels()
        self.update_pool_labels()
        self.update_turn_label()
        self.update_status()
        self.draw_board()
        self._maybe_ai_move()
        self._update_control_states()
        self._sync_record_tree()
        self._schedule_analysis()

    def _settings_for_side(self, side: int) -> dict | None:
        return self.side_settings.get(side)

    def _maybe_ai_move(self) -> None:
        if self.state is None or self.belief is None or self.replay_mode:
            return
        if self.paused or self.ai_thinking or self.setup_open:
            return
        if self.state.game_outcome()[0] is not None:
            return
        settings = self._settings_for_side(self.state.side)
        if settings is None:
            return
        self.ai_thinking = True
        delay = self.ai_delay_ms if self.game_kind == MODE_AIAI else 10
        self.update_status()
        # 记录对局代次：返回菜单/重开/进回放都会递增 _ai_token，
        # 延迟回调若发现代次变化则作废，避免旧局设置在新局落子。
        game_token = self._ai_token
        self.root.after(delay, lambda: self._start_ai_search(settings, game_token))

    def _start_ai_search(self, settings: dict, game_token: int) -> None:
        if game_token != self._ai_token or not self.ai_thinking or self.state is None:
            return
        if self.paused or self.replay_mode or self.setup_open:
            self.ai_thinking = False
            self.update_status()
            return
        state, belief, history = self.state.clone(), self.belief.clone(), list(self.history)
        get_agent = self._agents.get

        def work(rng):
            agent = get_agent(settings, "play", rng)
            rng.check()
            view = SearchBelief(belief, rng)
            if settings["agent"] == AGENT_AZ:
                return agent.analyse(state, view, history, settings["sims"])
            return agent.select_move(state, view, history), None

        token = self._ai_runner.submit(work)
        self._ai_token = token
        self.root.after(50, lambda: self._poll_ai_search(token))

    @staticmethod
    def _settings_key(settings):
        return AgentPool.settings_key(settings)

    def _poll_ai_search(self, token: int) -> None:
        if token != self._ai_token:
            return
        result = self._ai_runner.poll(token)
        if result is None:
            self.root.after(50, lambda: self._poll_ai_search(token))
            return
        self.ai_thinking = False
        if self.paused or self.replay_mode or self.setup_open or self.state is None:
            return
        if result.error:
            self.ai_info_var.set(f"AI 搜索失败: {result.error}")
            self.update_status()
            messagebox.showerror("AI 搜索失败", result.error)
            return
        move, value = result.value
        settings = self._settings_for_side(self.state.side)
        if settings:
            self.ai_info_var.set(f"对手 {self._agent_label(settings)} · {result.elapsed:.1f}s")
            if value is not None and self.analysis_settings and self._settings_key(settings) == self._settings_key(self.analysis_settings):
                key = self._current_analysis_key()
                self._analysis_cache[key] = (move, value, result.elapsed)
        if move is not None and self.state.game_outcome()[0] is None:
            if move not in self.state.legal_moves():
                self.ai_info_var.set("AI 返回了非法着法，请检查模型/搜索版本。")
                self.update_status()
                return
            self._apply_move(move)
            self._maybe_ai_move()
        else:
            self.update_status()

    def _cancel_analysis(self):
        self._analysis_token = self._analysis_runner.invalidate()
        self.analysis_thinking = False
        self._analysis_key = None
        if self._analysis_after is not None:
            self.root.after_cancel(self._analysis_after)
            self._analysis_after = None

    def _cancel_searches(self):
        self._ai_token = self._ai_runner.invalidate()
        self.ai_thinking = False
        self._cancel_analysis()

    def _schedule_analysis(self):
        # 快速翻页时只分析最后停留的局面；人类走子不等待分析完成。
        if self._analysis_after is not None:
            self.root.after_cancel(self._analysis_after)
        self._analysis_after = self.root.after(180, self._maybe_start_analysis)

    def _current_analysis_key(self):
        return (id(self.record), self.record.cursor, self._settings_key(self.analysis_settings))

    def _maybe_start_analysis(self) -> None:
        self._analysis_after = None
        if self.state is None or self.belief is None or self.setup_open:
            return
        want_hint = self.hint_on and (self.replay_mode or self.state.side == self.human_side)
        if not (want_hint or self.winrate_on):
            self.analysis_status_var.set("提示和胜率可随时开启")
            return
        outcome = self.state.game_outcome()[0]
        if outcome is not None:
            self.hint_text_var.set("终局没有推荐走法")
            if self.winrate_on:
                prob = .5 if outcome == jq.RESULT_DRAW else float(outcome == jq.RESULT_RED_WIN)
                self.eval_var.set("终局：和棋" if outcome == jq.RESULT_DRAW else ("终局：红方胜" if prob else "终局：黑方胜"))
                self._update_eval_bar(prob)
            self.analysis_status_var.set("终局结果由规则判定")
            return
        if self.analysis_settings is None:
            self.analysis_status_var.set("请选择分析模型；无模型时仅支持 RIS 提示")
            if self.winrate_on:
                self.eval_var.set("选择模型后显示胜率")
            return
        key = self._current_analysis_key()
        if key in self._analysis_cache:
            self._show_analysis(self._analysis_cache[key], want_hint)
            return
        if self.analysis_thinking and key == self._analysis_key:
            return
        self._cancel_analysis()
        self._analysis_key = key
        self.analysis_thinking = True
        settings = dict(self.analysis_settings)
        state, belief, history = self.state.clone(), self.belief.clone(), list(self.history)
        get_agent = self._agents.get
        side = state.side
        self.analysis_status_var.set(f"正在分析{'红' if side == jq.RED else '黑'}方视角 · {settings['sims']} 次模拟")
        if self.winrate_on:
            self.eval_var.set("当前局面评估中…")
            self._update_eval_bar(None)
        if want_hint:
            self.hint_text_var.set("正在生成当前局面的提示…")

        def work(rng):
            agent = get_agent(settings, "analysis", rng)
            rng.check()
            view = SearchBelief(belief, rng)
            if settings["agent"] == AGENT_RIS:
                return agent.select_move(state, view, history), None
            return agent.analyse(state, view, history, settings["sims"])

        token = self._analysis_runner.submit(work)
        self._analysis_token = token
        self.root.after(60, lambda: self._poll_analysis(token, key))

    def _poll_analysis(self, token: int, key) -> None:
        if token != self._analysis_token:
            return
        result = self._analysis_runner.poll(token)
        if result is None:
            self.root.after(60, lambda: self._poll_analysis(token, key))
            return
        self.analysis_thinking = False
        if self.state is None or self.analysis_settings is None or key != self._current_analysis_key():
            return
        if result.error:
            self.hint_text_var.set("")
            self.analysis_status_var.set(f"分析失败：{result.error}")
            if self.winrate_on:
                self.eval_var.set("评估失败，请检查分析设置")
            return
        move, value = result.value
        cached = (move, value, result.elapsed)
        self._analysis_cache[key] = cached
        want_hint = self.hint_on and (self.replay_mode or self.state.side == self.human_side)
        self._show_analysis(cached, want_hint)

    def _show_analysis(self, result, want_hint):
        move, value, elapsed = result
        engine = self._agent_label(self.analysis_settings)
        if want_hint and move is not None and move in self.state.legal_moves():
            self.hint_move = move
            self.hint_text_var.set(f"推荐：{self._format_move_text(self.state.board[move[0]], move)}")
        if self.winrate_on:
            if value is None:
                self.eval_var.set("RIS 仅提供走法提示；胜率需要 AlphaZero 模型")
            else:
                value = max(-1., min(1., value))
                red_prob = (1 + value) / 2 if self.state.side == jq.RED else (1 - value) / 2
                self.eval_var.set(f"红方 {red_prob:.1%}   ·   黑方 {1 - red_prob:.1%}")
                self._update_eval_bar(red_prob)
        self.analysis_status_var.set(f"{engine} · {elapsed:.1f}s · 行动方视角（胜率含和棋折半）")
        self.draw_board()

    def _analysis_options_changed(self):
        self.hint_on = self.live_hint_var.get()
        self.winrate_on = self.live_winrate_var.get()
        self._cancel_analysis()
        self.hint_move = None
        self.hint_text_var.set("")
        self._update_eval_bar(None)
        self.eval_var.set("当前局面评估中…" if self.winrate_on else "胜率未开启")
        if self.hint_on or self.winrate_on:
            if self.analysis_settings is None or (self.winrate_on and self.analysis_settings['agent'] != AGENT_AZ):
                model = default_model_path() or self.analysis_model_box.var.get().strip()
                if model and os.path.isfile(model):
                    self.analysis_settings = {"agent": AGENT_AZ, "sims": self.analysis_sims,
                                              "model_path": os.path.abspath(model), "device": "auto",
                                              "leaf_batch_size": self.analysis_leaf_batch_size}
                elif self.winrate_on:
                    self.analysis_status_var.set("胜率需要模型，请在分析设置中选择")
                    self._analysis_settings_dialog()
                else:
                    self.analysis_settings = {"agent": AGENT_RIS, "sims": self.analysis_sims}
        self.draw_board()
        self._schedule_analysis()

    def _analysis_settings_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("分析设置")
        dialog.configure(bg=CARD_BG, padx=16, pady=16)
        dialog.transient(self.root)
        settings = self.analysis_settings or {}
        path = tk.StringVar(value=settings.get("model_path", default_model_path()))
        sims = tk.StringVar(value=str(settings.get("sims", self.analysis_sims)))
        leaf_batch = tk.StringVar(value=str(settings.get("leaf_batch_size", self.analysis_leaf_batch_size)))
        device = tk.StringVar(value=settings.get("device", "auto"))
        tk.Label(dialog, text="AlphaZero 模型（留空使用 RIS 提示）", bg=CARD_BG, fg=TEXT).pack(anchor="w")
        row = tk.Frame(dialog, bg=CARD_BG)
        row.pack(fill="x", pady=6)
        box = self._combo(row, path, (), width=42)
        box.var = path
        box.pack(side="left")
        self._button(row, "浏览", lambda: self._browse_model(box)).pack(side="left", padx=(6, 0))
        tk.Label(dialog, text="模拟次数：较少模拟可更快显示；200 为默认", bg=CARD_BG, fg=MUTED).pack(anchor="w")
        self._combo(dialog, sims, AZ_SIM_CHOICES, width=12).pack(anchor="w", pady=6)
        self._leaf_batch_row(dialog, leaf_batch).pack(fill="x", pady=6)
        tk.Label(dialog, text="AZ 搜索参数，默认 64；可选 16/32。与训练批量无关，实际不超过模拟次数。",
                 bg=CARD_BG, fg=MUTED, wraplength=380, justify="left").pack(anchor="w")
        tk.Label(dialog, text="设备", bg=CARD_BG, fg=MUTED).pack(anchor="w")
        device_box = self._combo(dialog, device, DEVICE_CHOICES, width=12)
        device_box.configure(state="readonly")
        device_box.pack(anchor="w", pady=6)

        def apply():
            try:
                if not sims.get().isdigit() or not 1 <= int(sims.get()) <= 100000:
                    raise ValueError("模拟次数必须是 1～100000 的整数")
                model = path.get().strip()
                if model and not os.path.isabs(model):
                    model = os.path.join(PROJECT_ROOT, model)
                if model and not os.path.isfile(model):
                    raise ValueError("模型文件不存在")
                if not model and self.winrate_on:
                    raise ValueError("胜率需要 AlphaZero 模型；可关闭胜率后使用 RIS 提示")
                batch_value = parse_leaf_batch_size(leaf_batch.get())
                self.analysis_sims = int(sims.get())
                self.analysis_leaf_batch_size = batch_value
                self.analysis_settings = {"agent": AGENT_AZ if model else AGENT_RIS, "sims": self.analysis_sims}
                if model:
                    self.analysis_settings.update(model_path=os.path.abspath(model), device=device.get(),
                                                  leaf_batch_size=batch_value)
                self._cancel_analysis()
                self.hint_move = None
                self.hint_text_var.set("")
                self._update_eval_bar(None)
                self.eval_var.set("当前局面评估中…" if self.winrate_on else "胜率未开启")
                dialog.destroy()
                self._schedule_analysis()
                self.draw_board()
            except ValueError as exc:
                messagebox.showerror("分析设置", str(exc), parent=dialog)

        self._button(dialog, "应用", apply, primary=True).pack(anchor="e", pady=(8, 0))

    def _play_batch_settings_dialog(self):
        sides = [side for side, settings in self.side_settings.items()
                 if settings and settings["agent"] == AGENT_AZ]
        if self.replay_mode or not sides or self.state is None:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("对弈叶批量")
        dialog.configure(bg=CARD_BG, padx=16, pady=16)
        dialog.transient(self.root)
        values = {}
        for side in sides:
            values[side] = tk.StringVar(value=str(self.side_settings[side].get("leaf_batch_size", AZ_LEAF_BATCH_SIZE)))
            self._leaf_batch_row(dialog, values[side], "红方 AZ" if side == jq.RED else "黑方 AZ").pack(fill="x", pady=6)
        tk.Label(dialog, text="应用后取消当前对弈搜索并用新批量重算。分析叶批量在“分析设置”中单独调整。",
                 bg=CARD_BG, fg=MUTED, wraplength=380, justify="left").pack(anchor="w", pady=6)

        def apply():
            try:
                batches = {side: parse_leaf_batch_size(var.get()) for side, var in values.items()}
                self._set_play_leaf_batches(batches)
            except ValueError as exc:
                messagebox.showerror("对弈叶批量", str(exc), parent=dialog)
                return
            dialog.destroy()

        self._button(dialog, "应用", apply, primary=True).pack(anchor="e", pady=(8, 0))

    def _set_play_leaf_batches(self, batches):
        if self.replay_mode or self.state is None:
            return
        updated = dict(self.side_settings)
        for side, value in batches.items():
            settings = updated.get(side)
            if not settings or settings["agent"] != AGENT_AZ:
                raise ValueError("只能设置正在对弈的 AlphaZero 一方")
            updated[side] = dict(settings, leaf_batch_size=parse_leaf_batch_size(value))
        # 只改快照；旧任务返回后不得落子。引擎参数在下一任务的工作线程中修改。
        self._ai_token = self._ai_runner.invalidate()
        self.ai_thinking = False
        self.side_settings = updated
        for side in batches:
            var = self.red_leaf_batch_var if side == jq.RED else self.black_leaf_batch_var
            var.set(str(updated[side]["leaf_batch_size"]))
            if side == self.ai_side:
                self.human_leaf_batch_var.set(var.get())
        meta = self._live_metadata()
        self.header_var.set(f"红方 {meta['red_agent']} / 黑方 {meta['black_agent']}")
        self.update_status()
        self._maybe_ai_move()

    def _update_eval_bar(self, red_prob: float | None) -> None:
        self.eval_canvas.delete("all")
        if red_prob is None:
            self.eval_canvas.create_rectangle(0, 0, 380, 12, fill=FIELD_BG, outline="")
            return
        split = int(380 * red_prob)
        self.eval_canvas.create_rectangle(0, 0, split, 12, fill=RED_SIDE, outline="")
        self.eval_canvas.create_rectangle(split, 0, 380, 12, fill=BLACK_SIDE, outline="")

    @staticmethod
    def _format_move_text(piece: int, move: tuple[int, int]) -> str:
        color = jq.piece_color(piece)
        name = PIECE_NAMES[color][jq.piece_type(piece)]
        prefix = "暗" if jq.is_dark(piece) else ""
        side_label = "红" if color == jq.RED else "黑"
        return f"{side_label}方 {prefix}{name} {jq.square_to_name(move[0])}→{jq.square_to_name(move[1])}"

    def _apply_move(self, move: tuple[int, int]) -> None:
        self._cancel_analysis()
        captured_before = self.state.board[move[1]]
        old_count = len(self.record.nodes)
        try:
            node = self.record.add_move(move, rng=self.rng)
        except ValueError as exc:
            messagebox.showerror("无法走子", str(exc))
            return
        if len(self.record.nodes) != old_count:
            self._dirty = True
        self._branch_visible = False
        self._display_node(node.id)
        self._play("capture" if captured_before else "move")
        outcome = self.state.game_outcome()[0]
        if outcome is not None and not self.replay_mode:
            self._play_end_sound(outcome)
        elif self.state.is_in_check(self.state.side):
            self.root.after(160, lambda: self._play("check"))

    def _play(self, name: str) -> None:
        if self.sound_var.get():
            self.sounds.play(name)

    def _play_end_sound(self, outcome: int) -> None:
        if outcome == jq.RESULT_DRAW:
            name = "draw"
        elif self.game_kind == MODE_AIAI:
            name = "win"
        elif (outcome == jq.RESULT_RED_WIN) == (self.human_side == jq.RED):
            name = "win"
        else:
            name = "lose"
        self.root.after(200, lambda: self._play(name))

    def _reveal_captured(self, capturer_side: int) -> bool:
        """吃方是否向我方公开被吃暗子的类型（终局 / 上帝视角 / 本人吃子时公开）。"""
        if self.state is not None and self.state.game_outcome()[0] is not None:
            return True
        if self.game_kind in (MODE_AIAI, MODE_STUDY):
            return self.god_view
        return self.human_side == capturer_side

    def update_status(self) -> None:
        if self.state is None:
            return
        outcome, reason = self.state.game_outcome()
        if outcome is None:
            if self.replay_mode:
                self.status_var.set("研究 · 红方走子" if self.state.side == jq.RED else "研究 · 黑方走子")
                self.status_label.configure(fg=RED_SIDE if self.state.side == jq.RED else BLACK_SIDE)
            elif self.paused:
                self.status_var.set("已暂停")
                self.status_label.configure(fg=ACCENT)
            elif self.ai_thinking:
                side = "红方" if self.state.side == jq.RED else "黑方"
                text = f"{side} AI 思考中..." if self.game_kind == MODE_AIAI else "AI 思考中..."
                self.status_var.set(text)
                self.status_label.configure(fg=RED_SIDE if self.state.side == jq.RED else BLACK_SIDE)
            else:
                self.status_var.set("红方走子" if self.state.side == jq.RED else "黑方走子")
                self.status_label.configure(fg=RED_SIDE if self.state.side == jq.RED else BLACK_SIDE)
            return
        self.status_var.set(self._outcome_text(outcome, reason))
        self.status_label.configure(fg=ACCENT)

    @staticmethod
    def _outcome_text(outcome: int, reason: str) -> str:
        reason_text = {
            jq.REASON_LONG_CHECK: "长将禁着判负",
            jq.REASON_LONG_CHASE: "长捉禁着判负",
            jq.REASON_REPETITION: "重复局面和棋",
            jq.REASON_NO_CAPTURE: "无吃子和局",
            jq.REASON_CHECKMATE: "将死",
            jq.REASON_STALEMATE: "困毙",
        }.get(reason, "")
        if outcome == jq.RESULT_DRAW:
            label = "和局"
        else:
            label = "红方胜" if outcome == jq.RESULT_RED_WIN else "黑方胜"
        if reason_text:
            label = f"{label}（{reason_text}）"
        return label

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self._ai_token = self._ai_runner.invalidate()
            self.ai_thinking = False
        self.pause_btn.configure(text="继续" if self.paused else "暂停")
        self.update_status()
        if not self.paused:
            self._maybe_ai_move()
        self._maybe_start_analysis()

    def toggle_flip(self) -> None:
        self.flipped = not self.flipped
        self.draw_board()

    def update_turn_label(self) -> None:
        round_no = self.move_count // 2 + 1
        self.turn_var.set(f"回合: {round_no}")

    def update_captured_labels(self) -> None:
        if self.game_kind in (MODE_AIAI, MODE_STUDY):
            left_side, right_side = jq.RED, jq.BLACK
            left_title, right_title = "红方吃子", "黑方吃子"
        else:
            left_side = self.human_side if self.human_side is not None else jq.RED
            right_side = self.ai_side if self.ai_side is not None else jq.BLACK
            left_title, right_title = "我方吃子", "对方吃子"
        self.captured_left_title.configure(
            text=left_title, fg=RED_SIDE if left_side == jq.RED else BLACK_SIDE
        )
        self.captured_right_title.configure(
            text=right_title, fg=RED_SIDE if right_side == jq.RED else BLACK_SIDE
        )
        self.captured_left_var.set(self._captured_text(left_side))
        self.captured_right_var.set(self._captured_text(right_side))

    def _captured_text(self, capturer_side: int) -> str:
        entries = self.captured[capturer_side]
        if not entries:
            return "-"
        reveal = self._reveal_captured(capturer_side)
        parts = []
        for captured_side, was_dark, true_type in entries:
            if not was_dark:
                parts.append(PIECE_NAMES[captured_side][true_type])
            elif reveal and true_type is not None:
                parts.append(f"[{PIECE_NAMES[captured_side][true_type]}]")
            else:
                parts.append("[暗]")
        return " ".join(parts)

    def update_pool_labels(self) -> None:
        if self.state is None:
            return
        self.pool_red_var.set(self.format_pool_text(jq.RED))
        self.pool_black_var.set(self.format_pool_text(jq.BLACK))

    def format_pool_text(self, side: int) -> str:
        if self.state is None:
            return "-"
        if self.game_kind in (MODE_AIAI, MODE_STUDY):
            total = self.state.pool_total[side]
            if total <= 0:
                return "无"
            if not self.god_view and self.state.game_outcome()[0] is None:
                return f"总数{total}（组成未知）"
            return self._format_pool_counts(side, self.state.pool[side], total)
        if (
            self.belief is None
            or self.human_side is None
            or self.state.game_outcome()[0] is not None
        ):
            return self._format_pool_counts(side, self.state.pool[side], self.state.pool_total[side])
        observer = self.human_side
        counts = self.belief.pool_counts[observer][side]
        total = self.belief.pool_total[observer][side]
        if total <= 0:
            return "无"
        unknown = self.belief.unknown_removals[observer][side]
        if unknown > 0:
            return f"剩余暗子{total}（{unknown} 枚被吃类型未知）"
        return self._format_pool_counts(side, counts, total)

    def _format_pool_counts(self, side: int, counts, total: int) -> str:
        if total <= 0:
            return "无"
        parts = [f"总数{total}"]
        for p_type in jq.POOL_TYPES:
            count = counts[p_type]
            pct = int(round(count * 100 / total))
            label = PIECE_NAMES[side][p_type]
            parts.append(f"{label}{count}({pct}%)")
        return " ".join(parts)

    def _view_rc(self, idx: int) -> tuple[int, int]:
        row, col = jq.ROWS[idx], jq.COLS[idx]
        if self.flipped:
            row = jq.BOARD_H - 1 - row
            col = jq.BOARD_W - 1 - col
        return row, col

    def _point(self, idx: int) -> tuple[float, float]:
        row, col = self._view_rc(idx)
        return CELL / 2 + col * CELL, CELL / 2 + row * CELL

    def _index_at(self, x: float, y: float) -> int | None:
        col = int(x // CELL)
        row = int(y // CELL)
        if self.flipped:
            row = jq.BOARD_H - 1 - row
            col = jq.BOARD_W - 1 - col
        if not jq.in_bounds(row, col):
            return None
        return jq.index(row, col)

    def _draw_board_background(self) -> None:
        bands = 30
        height = BOARD_H / bands
        for i in range(bands):
            color = _blend(BOARD_TOP, BOARD_BOTTOM, i / (bands - 1))
            self.canvas.create_rectangle(0, i * height, BOARD_W, (i + 1) * height + 1, fill=color, outline="")
        left = CELL / 2
        top = CELL / 2
        right = BOARD_W - CELL / 2
        bottom = BOARD_H - CELL / 2
        self.canvas.create_rectangle(left - 14, top - 14, right + 14, bottom + 14, outline=BOARD_LINE_DARK, width=3)
        self.canvas.create_rectangle(left - 7, top - 7, right + 7, bottom + 7, outline=BOARD_LINE, width=1)
        for r in range(jq.BOARD_H):
            y = top + r * CELL
            self.canvas.create_line(left, y, right, y, fill=BOARD_LINE, width=1)
        for c in range(jq.BOARD_W):
            x = left + c * CELL
            if c in (0, jq.BOARD_W - 1):
                self.canvas.create_line(x, top, x, bottom, fill=BOARD_LINE, width=1)
            else:
                self.canvas.create_line(x, top, x, top + 4 * CELL, fill=BOARD_LINE, width=1)
                self.canvas.create_line(x, top + 5 * CELL, x, bottom, fill=BOARD_LINE, width=1)
        for r0, r1 in ((0, 2), (7, 9)):
            self.canvas.create_line(left + 3 * CELL, top + r0 * CELL, left + 5 * CELL, top + r1 * CELL, fill=BOARD_LINE)
            self.canvas.create_line(left + 5 * CELL, top + r0 * CELL, left + 3 * CELL, top + r1 * CELL, fill=BOARD_LINE)
        river_y = top + 4.5 * CELL
        self.canvas.create_text(left + 2 * CELL, river_y, text="楚 河", fill="#8a6a42", font=("KaiTi", 18))
        self.canvas.create_text(right - 2 * CELL, river_y, text="漢 界", fill="#8a6a42", font=("KaiTi", 18))
        self._draw_star_marks()

    def _draw_star_marks(self) -> None:
        def mark(row: int, col: int) -> None:
            x, y = self._point(jq.index(row, col))
            _srow, scol = self._view_rc(jq.index(row, col))
            d = 5
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                if scol == 0 and dx < 0:
                    continue
                if scol == jq.BOARD_W - 1 and dx > 0:
                    continue
                px, py = x + dx * d, y + dy * d
                self.canvas.create_line(px, py, px + dx * 6, py, fill=BOARD_LINE, width=1)
                self.canvas.create_line(px, py, px, py + dy * 6, fill=BOARD_LINE, width=1)

        for col in (0, 2, 4, 6, 8):
            mark(3, col)
            mark(6, col)
        for col in (1, 7):
            mark(2, col)
            mark(7, col)

    def _draw_piece(self, idx: int, piece: int) -> None:
        x, y = self._point(idx)
        radius = 24
        self.canvas.create_oval(
            x - radius + 2, y - radius + 4, x + radius + 2, y + radius + 4,
            fill="#7d6742", outline="", stipple="gray50",
        )
        color = jq.piece_color(piece)
        if jq.is_dark(piece):
            fill = "#57687c"
            ring = "#d7dee7"
            text_color = "#f6f7f9"
        else:
            fill = "#fdf6e6"
            ring = "#b8352b" if color == jq.RED else "#33302c"
            text_color = "#b8352b" if color == jq.RED else "#2c2a27"
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=ring, width=3)
        self.canvas.create_oval(
            x - radius + 4, y - radius + 4, x + radius - 4, y + radius - 4,
            outline="#d9c9a8" if not jq.is_dark(piece) else "#8fa0b3",
            width=1,
        )
        label = PIECE_NAMES[color][jq.piece_type(piece)]
        if jq.is_dark(piece):
            self.canvas.create_text(x, y + 1, text=label, fill=text_color, font=("Microsoft YaHei", 17, "bold"))
            bx, by = x + 15, y - 15
            self.canvas.create_oval(bx - 8, by - 8, bx + 8, by + 8, fill="#e2b45a", outline="#8a6520")
            self.canvas.create_text(bx, by, text="暗", fill="#4a3410", font=("Microsoft YaHei", 8, "bold"))
        else:
            self.canvas.create_text(x, y + 1, text=label, fill=text_color, font=("Microsoft YaHei", 19, "bold"))

    def draw_board(self) -> None:
        self.canvas.delete("all")
        self._draw_board_background()

        if self.last_move:
            for idx in self.last_move:
                x, y = self._point(idx)
                self.canvas.create_rectangle(
                    x - CELL / 2 + 2, y - CELL / 2 + 2, x + CELL / 2 - 2, y + CELL / 2 - 2,
                    fill=LAST_TO, outline="", stipple="gray25",
                )

        for idx, piece in enumerate(self.state.board):
            if piece:
                self._draw_piece(idx, piece)

        if self.last_move:
            src, dst = self.last_move
            sx, sy = self._point(src)
            dx, dy = self._point(dst)
            self.canvas.create_rectangle(sx - 28, sy - 28, sx + 28, sy + 28, outline=LAST_FROM, width=2, dash=(6, 4))
            self.canvas.create_rectangle(dx - 29, dy - 29, dx + 29, dy + 29, outline=LAST_TO, width=3)
            self.canvas.create_line(
                sx, sy, dx, dy, fill=LAST_TO, width=4, arrow="last", arrowshape=(14, 18, 7), stipple="gray75"
            )

        outcome = self.state.game_outcome()[0]
        if outcome is None:
            king_idx = self.state.king_pos[self.state.side]
            if king_idx >= 0 and self.state.is_in_check(self.state.side):
                x, y = self._point(king_idx)
                self.canvas.create_oval(x - 27, y - 27, x + 27, y + 27, outline="#e74c3c", width=3)

        if self.hint_move:
            hs, hd = self.hint_move
            hx0, hy0 = self._point(hs)
            hx1, hy1 = self._point(hd)
            self.canvas.create_oval(hx0 - 28, hy0 - 28, hx0 + 28, hy0 + 28, outline=HINT_COLOR, width=2, dash=(4, 3))
            self.canvas.create_line(
                hx0, hy0, hx1, hy1, fill=HINT_COLOR, width=4, arrow="last", arrowshape=(12, 16, 6), stipple="gray75"
            )

        if self.selected_idx is not None:
            x, y = self._point(self.selected_idx)
            self.canvas.create_oval(x - 27, y - 27, x + 27, y + 27, outline=ACCENT, width=3)

        for idx in self.selected_targets - self.selected_captures:
            x, y = self._point(idx)
            self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#3f8f4f", outline="#f2f6ee", width=1)

        for idx in self.selected_captures:
            x, y = self._point(idx)
            self.canvas.create_oval(x - 26, y - 26, x + 26, y + 26, outline="#cf3f34", width=3)
        if self.replay_mode and self._branch_visible:
            colors = (HINT_COLOR, "#d979bd", "#78b851", "#ed9046")
            occupied = {}
            for number, child_id in enumerate(self.record.current.children, 1):
                record = self.record.nodes[child_id].record
                sx, sy = self._point(record['from'])
                dx, dy = self._point(record['to'])
                color = colors[(number - 1) % len(colors)]
                self.canvas.create_line(sx, sy, dx, dy, fill=color, width=3, arrow="last", dash=(6, 3))
                x, y = sx + (dx - sx) * .6, sy + (dy - sy) * .6
                key = (round(x), round(y))
                offset = occupied.get(key, 0)
                occupied[key] = offset + 1
                x = max(16, min(BOARD_W - 16, x + offset * 28))
                y = max(16, min(BOARD_H - 16, y))
                tag = f"branch_{number}"
                self.canvas.create_oval(x - 13, y - 13, x + 13, y + 13, fill=color, outline=ACCENT_TEXT, width=2, tags=(tag,))
                self.canvas.create_text(x, y, text=str(number), fill=ACCENT_TEXT, font=("Microsoft YaHei", 11, "bold"), tags=(tag,))

    def on_click(self, event) -> None:
        if self.state is None or self.belief is None or self.setup_open:
            return
        if self.replay_mode and self._branch_visible:
            for item in reversed(self.canvas.find_overlapping(event.x, event.y, event.x, event.y)):
                for tag in self.canvas.gettags(item):
                    if tag.startswith("branch_"):
                        self._choose_branch(int(tag.split('_')[1]))
                        return
        if self.state.game_outcome()[0] is not None:
            return
        if self.ai_thinking or self.paused:
            return
        if not self.replay_mode and (self.human_side is None or self.state.side != self.human_side):
            return
        idx = self._index_at(event.x, event.y)
        if idx is None:
            return
        piece = self.state.board[idx]

        if self.selected_idx is None:
            if piece and jq.piece_color(piece) == self.state.side:
                self.selected_idx = idx
                self.selected_targets, self.selected_captures = self._legal_targets(idx)
                self._play("select")
                self.draw_board()
            return

        if idx == self.selected_idx:
            self.selected_idx = None
            self.selected_targets = set()
            self.selected_captures = set()
            self.draw_board()
            return

        if idx in self.selected_targets:
            self._apply_move((self.selected_idx, idx))
            self._maybe_ai_move()
        else:
            if piece and jq.piece_color(piece) == self.state.side:
                self.selected_idx = idx
                self.selected_targets, self.selected_captures = self._legal_targets(idx)
                self._play("select")
                self.draw_board()

    def _legal_targets(self, from_idx: int) -> tuple[set[int], set[int]]:
        moves = self.state.legal_moves()
        targets = set()
        captures = set()
        for src, to_idx in moves:
            if src != from_idx:
                continue
            targets.add(to_idx)
            if self.state.board[to_idx]:
                captures.add(to_idx)
        return targets, captures


    def _update_control_states(self) -> None:
        available = self.state is not None and self.record is not None
        live = available and not self.replay_mode
        self.restart_btn.configure(state="normal" if live else "disabled")
        self.save_btn.configure(state="normal" if available else "disabled")
        self.replay_btn.configure(state="normal" if live else "disabled")
        self.pause_btn.configure(state="normal" if live and self.game_kind == MODE_AIAI else "disabled")
        self.next_btn.configure(state="normal" if available and self.record.current.children else "disabled")
        self.continue_btn.configure(state="normal" if available and self.replay_mode and self.state.game_outcome()[0] is None else "disabled")
        self.play_batch_btn.configure(state="normal" if live and any(
            settings and settings["agent"] == AGENT_AZ for settings in self.side_settings.values()) else "disabled")

    @staticmethod
    def _agent_label(settings: dict | None) -> str:
        if not settings:
            return "?"
        label = f"{AGENT_LABELS.get(settings.get('agent'), '?')}-{settings.get('sims', '?')}"
        if settings.get('agent') == AGENT_AZ:
            label += f" · 叶批 {settings.get('leaf_batch_size', AZ_LEAF_BATCH_SIZE)}"
        return label

    def _live_metadata(self):
        if self.game_kind == MODE_AIAI:
            red = self._agent_label(self.side_settings.get(jq.RED))
            black = self._agent_label(self.side_settings.get(jq.BLACK))
        elif self.game_kind == MODE_STUDY:
            red, black = "摆棋", "摆棋"
        else:
            opponent = self._agent_label(self.side_settings.get(self.ai_side))
            red, black = ("玩家", opponent) if self.human_side == jq.RED else (opponent, "玩家")
        return {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": self.game_kind,
                "human_side": self.human_side, "god_view": bool(self.god_view),
                "red_agent": red, "black_agent": black}

    def _replay_meta_from_live(self) -> dict:
        if not self.replay_mode:
            metadata = self._live_metadata()
            metadata['created_at'] = self.record.metadata.get('created_at', metadata['created_at'])
            self.record.metadata.update(metadata)
        return self.record.to_dict()

    @staticmethod
    def _write_replay_file(path: str, meta: dict) -> None:
        # 写入失败保留原文件；文件对话框负责向用户确认覆盖。
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".replay-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=1)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_game_file(self) -> bool:
        if self.state is None or self.record is None:
            return False
        path = filedialog.asksaveasfilename(title="保存完整棋谱（包含所有分支和起始局面）",
            initialdir=ensure_replay_dir(), initialfile=f"jieqi_{time.strftime('%Y%m%d_%H%M%S')}.json",
            defaultextension=".json", filetypes=[("揭棋对局", "*.json"), ("所有文件", "*.*")])
        if not path:
            return False
        try:
            self._write_replay_file(path, self._replay_meta_from_live())
        except Exception as exc:
            messagebox.showerror("保存失败", f"{type(exc).__name__}: {exc}")
            return False
        self._dirty = False
        forks = sum(len(n.children) > 1 for n in self.record.nodes.values())
        messagebox.showinfo("保存对局", f"已保存全部 {len(self.record.nodes) - 1} 手、{forks} 个分叉及起始局面到\n{path}")
        return True

    def _confirm_leave(self):
        if not self._dirty:
            return True
        answer = messagebox.askyesnocancel("保存棋谱", "当前棋谱有未保存的走子或分支，是否先保存？")
        if answer is None:
            return False
        return self.save_game_file() if answer else True

    def _close(self):
        if not self._confirm_leave():
            return
        self._cancel_searches()
        self._ai_runner.close()
        self._analysis_runner.close()
        self.root.destroy()

    def _load_replay_file(self, path: str) -> bool:
        try:
            with open(path, encoding="utf-8") as handle:
                meta = json.load(handle)
            game = GameRecord.from_dict(meta, migrate_rules=True)
        except Exception as exc:
            messagebox.showerror("加载失败", f"{type(exc).__name__}: {exc}")
            return False
        if not self._confirm_leave():
            return False
        self._activate_replay(game)
        self._dirty = False
        return True

    def load_game_file(self) -> None:
        path = filedialog.askopenfilename(title="加载历史对局", initialdir=ensure_replay_dir(),
            filetypes=[("揭棋对局", "*.json"), ("所有文件", "*.*")])
        if path:
            self._load_replay_file(path)

    def enter_replay_from_game(self) -> None:
        if self.record is None:
            return
        self.record.metadata.update(self._live_metadata())
        self._activate_replay(self.record)

    def _enter_replay(self, meta: dict) -> bool:
        try:
            game = GameRecord.from_dict(meta, migrate_rules=True)
        except Exception as exc:
            messagebox.showerror("无法回放", f"{type(exc).__name__}: {exc}")
            return False
        self._activate_replay(game)
        return True

    def _activate_replay(self, game):
        self._cancel_searches()
        if self.record is not game:
            self._analysis_cache.clear()
        self.record = game
        self.replay_mode = True
        self.replay_meta = game.metadata
        self.game_kind = game.metadata.get("mode", MODE_STUDY)
        self.human_side = game.metadata.get("human_side")
        self.ai_side = 1 - self.human_side if self.human_side in (jq.RED, jq.BLACK) else None
        self.god_view = bool(game.metadata.get("god_view", True))
        self.side_settings = {}
        self.paused = False
        self.setup_open = False
        self._branch_visible = False
        self.live_hint_var.set(self.hint_on)
        self.live_winrate_var.set(self.winrate_on)
        if self.main_frame.winfo_manager():
            self.main_frame.pack_forget()
            self.game_host.pack(fill="both", expand=True)
        red, black = game.metadata.get("red_agent", "?"), game.metadata.get("black_agent", "?")
        migrations = game.metadata.get("rules_migrations", [])
        suffix = ""
        if migrations:
            source = migrations[0]['source_ruleset_version']
            suffix = f" · 原规则 v{source}，按 v{jq.RULESET_VERSION} 重新校验"
            if any(item.get('initial_rule_history_reset') for item in migrations):
                suffix += "（起点将/捉历史重新计数）"
        self.header_var.set(f"棋谱研究 · 红方 {red} / 黑方 {black}{suffix}")
        self.pool_title_var.set("暗池信息（研究视角）" if self.game_kind == MODE_STUDY else "暗池信息")
        self.replay_row.pack(anchor="w", pady=(8, 0))
        self.study_tip.pack(anchor="w", pady=(6, 0))
        self._display_node(game.cursor)
        self._fit_game_geometry()

    def _display_node(self, node_id):
        self.record.cursor = node_id
        node = self.record.current
        self.state, self.belief = node.state.clone(), node.belief.clone()
        self.captured = copy.deepcopy(node.captured)
        path = self.record.path()
        self.history = [(self.record.nodes[i].record['from'], self.record.nodes[i].record['to']) for i in path]
        self.move_log = [dict(self.record.nodes[i].record) for i in path]
        self.move_count = node.depth
        self.replay_index = node.depth
        self.selected_idx = None
        self.selected_targets, self.selected_captures = set(), set()
        self.hint_move = None
        self.hint_text_var.set("")
        self.last_move = self.history[-1] if self.history else None
        if self.last_move:
            self.last_move_var.set(f"上一步：{self._format_move_text(self.state.board[self.last_move[1]], self.last_move)}")
        else:
            self.last_move_var.set("上一步：-（起始局面）")
        self.eval_var.set("当前局面评估中…" if self.winrate_on else "胜率未开启")
        self._update_eval_bar(None)
        self.update_captured_labels()
        self.update_pool_labels()
        self.update_turn_label()
        self.update_status()
        if self.replay_mode:
            side = "红方" if self.state.side == jq.RED else "黑方"
            self.ai_info_var.set(f"第 {node.depth} 手 · {side}行动 · 后续 {len(node.children)} 种走法")
        self._sync_record_tree()
        self._update_control_states()
        self._update_branch_choices()
        self.draw_board()
        self._schedule_analysis()

    def _sync_record_tree(self):
        if self.record is None:
            return
        self._tree_sync = True
        try:
            if self._tree_record is not self.record:
                self.record_tree.delete(*self.record_tree.get_children())
                self.record_tree.insert('', 'end', iid='n0', text='起始局面', open=True)
                self._tree_record = self.record
            # 连续走子显示为同级行，只有真实分叉才增加展示层级。
            # 深度优先排列，让每条变例的后续走子保持连续；展示父级不改变棋谱父节点。
            pending = [(0, '')]
            while pending:
                node_id, display_parent = pending.pop()
                node = self.record.nodes[node_id]
                iid = f'n{node.id}'
                if node.parent is not None:
                    parent = self.record.nodes[node.parent]
                    number = parent.children.index(node.id) + 1
                    piece = parent.state.board[node.record['from']]
                    text = f"{node.depth}. {self._format_move_text(piece, (node.record['from'], node.record['to']))}"
                    if len(parent.children) > 1:
                        text = f"[{number}] " + text
                    if self.record_tree.exists(iid):
                        self.record_tree.item(iid, text=text)
                        self.record_tree.move(iid, display_parent, 'end')
                    else:
                        self.record_tree.insert(display_parent, 'end', iid=iid, text=text, open=True)
                children_parent = iid if len(node.children) > 1 else display_parent
                pending.extend((child_id, children_parent) for child_id in reversed(node.children))
            iid = f'n{self.record.cursor}'
            if self.record_tree.selection() != (iid,):
                self.record_tree.selection_set(iid)
                self.record_tree.see(iid)
        finally:
            self._tree_sync = False

    def _tree_selected(self, _event):
        if self._tree_sync or self.record is None:
            return
        selection = self.record_tree.selection()
        if not selection:
            return
        node_id = int(selection[0][1:])
        if node_id == self.record.cursor:
            return
        if not self.replay_mode:
            self.enter_replay_from_game()
        self._replay_goto(node_id)

    def _replay_goto(self, node_id: int) -> None:
        if not self.replay_mode or self.record is None or node_id not in self.record.nodes:
            return
        self._cancel_analysis()
        self._branch_visible = False
        self._display_node(node_id)

    def _update_branch_choices(self):
        for widget in self.branch_row.winfo_children():
            widget.destroy()
        self.branch_row.pack_forget()
        if not self.replay_mode or not self._branch_visible:
            self.next_btn.configure(text="下一步")
            self.study_tip_var.set(STUDY_TIP)
            return
        self.next_btn.configure(text="选择分支 →")
        self.branch_row.pack(anchor="w", pady=(6, 0), before=self.study_tip)
        tk.Label(self.branch_row, text="下一步选项", bg=APP_BG, fg=ACCENT, font=("Microsoft YaHei", 10)).grid(row=0, column=0, padx=(0, 8))
        for number, node_id in enumerate(self.record.current.children, 1):
            button = self._button(self.branch_row, str(number), lambda n=number: self._choose_branch(n), width=2)
            button.grid(row=(number - 1) // 8, column=(number - 1) % 8 + 1, padx=3, pady=2)
        self.study_tip_var.set("棋盘编号与下一步选项一致；点击编号按钮或按数字键选择。")

    def _choose_branch(self, number):
        if not self.replay_mode or not self._branch_visible:
            return
        children = self.record.current.children
        if 1 <= number <= len(children):
            self._replay_goto(children[number - 1])

    def _replay_step(self, delta: int) -> None:
        if not self.replay_mode or self.record is None:
            return
        node = self.record.current
        if delta < 0:
            if node.parent is not None:
                self._replay_goto(node.parent)
        elif len(node.children) == 1:
            self._replay_goto(node.children[0])
        elif len(node.children) > 1:
            self._branch_visible = True
            self._update_branch_choices()
            self.draw_board()
            self._fit_game_geometry()

    def _replay_to_end(self) -> None:
        if not self.replay_mode or self.record is None:
            return
        node = self.record.current
        while len(node.children) == 1:
            node = self.record.nodes[node.children[0]]
        self._replay_goto(node.id)
        if len(node.children) > 1:
            self._replay_step(1)

    def open_position_editor(self):
        if self.setup_open:
            return
        was_paused = self.paused
        self._cancel_searches()
        self.setup_open = True
        self.paused = True
        self.update_status()

        def close():
            self.setup_open = False
            self.paused = was_paused
            self.update_status()
            self._maybe_ai_move()
            self._schedule_analysis()

        def apply(state):
            if not self._confirm_leave():
                return False
            game = GameRecord(state, metadata={"mode": MODE_STUDY, "god_view": True,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "red_agent": "摆棋", "black_agent": "摆棋"})
            self._activate_replay(game)
            self._dirty = True
            return True

        PositionEditor(self.root, self.state, apply, close)

    def continue_game(self):
        """从回放/摆棋节点恢复人机或 AI 对弈，原棋谱分支仍保留。"""
        if not self.replay_mode or self.record is None or self.state.game_outcome()[0] is not None:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("从当前局面对弈")
        dialog.configure(bg=CARD_BG, padx=16, pady=16)
        dialog.transient(self.root)
        mode = tk.StringVar(value="人机 · 执黑" if self.human_side == jq.BLACK else "人机 · 执红")
        tk.Label(dialog, text="保留原有棋谱，新走法成为变例", bg=CARD_BG, fg=ACCENT).pack(anchor="w")
        mode_box = self._combo(dialog, mode, ("人机 · 执红", "人机 · 执黑", "AI 对战"), width=18)
        mode_box.configure(state="readonly")
        mode_box.pack(anchor="w", pady=8)
        rows = {}
        for side in (jq.RED, jq.BLACK):
            row = tk.Frame(dialog, bg=CARD_BG)
            row.pack(fill="x", pady=4)
            tk.Label(row, text="红方 AI" if side == jq.RED else "黑方 AI", bg=CARD_BG, fg=TEXT).pack(side="left", padx=(0, 8))
            agent = tk.StringVar(value=self.human_agent_var.get())
            sims = tk.StringVar(value=self.human_sims_var.get())
            agent_box = self._combo(row, agent, tuple(AGENT_LABELS.values()), width=12)
            agent_box.configure(state="readonly")
            agent_box.pack(side="left", padx=(0, 8))
            self._combo(row, sims, AZ_SIM_CHOICES, width=8).pack(side="left")
            batch = tk.StringVar(value=self.human_leaf_batch_var.get())
            self._leaf_batch_row(row, batch, "叶批").pack(side="left", padx=(8, 0))
            rows[side] = agent, sims, batch
        tk.Label(dialog, text="仅使用非玩家一方的 AI 参数；AI 对战使用双方参数。", bg=CARD_BG, fg=MUTED).pack(anchor="w", pady=6)
        model = tk.StringVar(value=(self.analysis_settings or {}).get('model_path') or self.human_model_box.var.get())
        model_box = self._combo(dialog, model, (), width=42)
        model_box.var = model
        model_box.pack(fill="x", pady=4)
        self._button(dialog, "浏览 AlphaZero 模型", lambda: self._browse_model(model_box)).pack(anchor="w")

        def apply():
            try:
                human = None if mode.get() == "AI 对战" else (jq.RED if mode.get() == "人机 · 执红" else jq.BLACK)
                settings = {side: None if side == human else self._collect_agent_settings(
                    rows[side][0], rows[side][1], model_box, self.human_device_box,
                    rows[side][2]) for side in (jq.RED, jq.BLACK)}
            except ValueError as exc:
                messagebox.showerror("对弈参数", str(exc), parent=dialog)
                return
            self._cancel_searches()
            self.replay_mode = False
            self.game_kind = MODE_AIAI if human is None else MODE_HUMAN
            self.human_side = human
            self.ai_side = None if human is None else 1 - human
            self.side_settings = settings
            self.god_view = human is None
            self.paused = False
            self.flipped = human == jq.BLACK
            self.ai_delay_ms = SPEED_DELAYS.get(self.speed_var.get(), 450) if human is None else 0
            self.replay_row.pack_forget()
            self.branch_row.pack_forget()
            self.study_tip.pack_forget()
            self._branch_visible = False
            self.header_var.set("从研究局面继续 · AI 对战" if human is None else f"从研究局面继续 · 你执{'红' if human == jq.RED else '黑'}")
            self.pool_title_var.set("暗池信息（上帝视角）" if human is None else "暗池信息")
            self.pause_btn.configure(text="暂停")
            self._display_node(self.record.cursor)
            self._dirty = True
            dialog.destroy()
            self._maybe_ai_move()

        self._button(dialog, "开始对弈", apply, primary=True).pack(anchor="e", pady=(10, 0))


def main() -> None:
    root = tk.Tk()
    JieQiGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
