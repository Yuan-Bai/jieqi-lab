"""明/暗子摆棋编辑器：暗子假类型与真实暗池组成分开配置。"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import jieqi as jq
from game_record import validate_position

BG, FG, MUTED, GOLD = "#24211d", "#f2e9d8", "#b3a68e", "#d9a84e"
CELL = 48


class PositionEditor(tk.Toplevel):
    def __init__(self, parent, state, on_apply, on_close):
        super().__init__(parent)
        self.title("摆棋 · 明子 / 暗子")
        self.configure(bg=BG)
        self.transient(parent)
        self.on_apply, self.on_close = on_apply, on_close
        self.state = state.clone() if state is not None else jq.GameState.initial()
        self.side_var = tk.StringVar(value="红方" if self.state.side == jq.RED else "黑方")
        self.color_var = tk.StringVar(value="红方")
        self.type_var = tk.StringVar(value="车")
        self.dark_var = tk.BooleanVar(value=False)
        self.tool_var = tk.StringVar(value="放置")
        self.pool_vars = {}
        self.count_var = tk.StringVar()
        self.canvas = tk.Canvas(self, width=9 * CELL, height=10 * CELL, bg="#eedcb4", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=16, pady=16, sticky="n")
        self.canvas.bind("<Button-1>", self._place)
        self.canvas.bind("<Button-3>", lambda e: self._place(e, erase=True))
        panel = tk.Frame(self, bg=BG)
        panel.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="n")
        self._label(panel, "点击棋盘放置 · 右键移除", GOLD)
        self._label(panel, "暗子显示的兵种是行棋假类型。")
        for title, var, values in (("工具", self.tool_var, ("放置", "移除")),
                                   ("颜色", self.color_var, ("红方", "黑方")),
                                   ("兵种", self.type_var, ("帅/将", "车", "马", "相/象", "仕/士", "炮", "兵/卒"))):
            row = tk.Frame(panel, bg=BG)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=title, bg=BG, fg=FG, width=6).pack(side="left")
            ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=12, style="App.TCombobox").pack(side="left")
        tk.Checkbutton(panel, text="暗子（按所选假类型行棋）", variable=self.dark_var,
                       bg=BG, fg=FG, selectcolor=BG, activebackground=BG, activeforeground=GOLD).pack(anchor="w", pady=4)
        self._label(panel, "真实暗池（车 马 象 士 炮 兵）", GOLD)
        self._label(panel, "每方合计必须等于棋盘上的暗子数。")
        self._label(panel, "缺失棋子视为已吃；初始暗池组成公开。")
        for side in (jq.RED, jq.BLACK):
            row = tk.Frame(panel, bg=BG)
            row.pack(fill="x", pady=4)
            tk.Label(row, text="红" if side == jq.RED else "黑", bg=BG, fg=FG).pack(side="left", padx=(0, 4))
            for t in jq.POOL_TYPES:
                var = tk.StringVar(value=str(self.state.pool[side][t]))
                self.pool_vars[side, t] = var
                tk.Spinbox(row, from_=0, to=jq.POOL_INIT[t], width=2, textvariable=var,
                           bg="#413a30", fg=FG, buttonbackground="#4a4437", insertbackground=FG,
                           relief="flat", font=("Microsoft YaHei", 10)).pack(side="left", padx=2)
        tk.Label(panel, textvariable=self.count_var, bg=BG, fg=GOLD).pack(anchor="w", pady=6)
        row = tk.Frame(panel, bg=BG)
        row.pack(fill="x", pady=6)
        tk.Label(row, text="先走方", bg=BG, fg=FG).pack(side="left", padx=(0, 6))
        ttk.Combobox(row, textvariable=self.side_var, values=("红方", "黑方"), state="readonly", width=9, style="App.TCombobox").pack(side="left")
        self._label(panel, "应用后从新局面开始计将/捉/重复历史。")
        row = tk.Frame(panel, bg=BG)
        row.pack(fill="x", pady=8)
        ttk.Button(row, text="标准开局", command=self._standard, style="App.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(row, text="清空棋盘", command=self._empty, style="App.TButton").pack(side="left")
        ttk.Button(panel, text="应用为新棋谱", command=self._apply, style="App.TButton").pack(fill="x", pady=8)
        ttk.Button(panel, text="取消", command=self._close, style="App.TButton").pack(fill="x")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.grab_set()
        self._draw()

    def _label(self, panel, text, color=MUTED):
        tk.Label(panel, text=text, bg=BG, fg=color, font=("Microsoft YaHei", 9),
                 wraplength=300, justify="left").pack(anchor="w", pady=3)

    def _place(self, event, erase=False):
        row, col = int(event.y // CELL), int(event.x // CELL)
        if not jq.in_bounds(row, col):
            return
        idx = jq.index(row, col)
        if erase or self.tool_var.get() == "移除":
            self.state.set_piece(idx, 0)
        else:
            t = ("帅/将", "车", "马", "相/象", "仕/士", "炮", "兵/卒").index(self.type_var.get()) + 1
            side = jq.RED if self.color_var.get() == "红方" else jq.BLACK
            dark = int(self.dark_var.get())
            if dark and t == jq.TYPE_KING:
                messagebox.showinfo("暗子", "将帅不能是暗子。", parent=self)
                return
            # 放新将帅时移动已有将帅，避免不小心重复。
            if t == jq.TYPE_KING and self.state.king_pos[side] >= 0:
                self.state.set_piece(self.state.king_pos[side], 0)
            self.state.set_piece(idx, jq.make_piece(t, side, dark))
        self._draw()

    def _draw(self):
        self.canvas.delete("all")
        for r in range(10):
            y = (r + .5) * CELL
            self.canvas.create_line(CELL / 2, y, 8.5 * CELL, y, fill="#7a5836")
        for c in range(9):
            x = (c + .5) * CELL
            self.canvas.create_line(x, CELL / 2, x, 9.5 * CELL, fill="#7a5836")
        for r0, r1 in ((0, 2), (7, 9)):
            for c0, c1 in ((3, 5), (5, 3)):
                self.canvas.create_line((c0 + .5) * CELL, (r0 + .5) * CELL,
                                        (c1 + .5) * CELL, (r1 + .5) * CELL, fill="#7a5836")
        counts = [0, 0]
        for i, p in enumerate(self.state.board):
            if not p:
                continue
            side, dark = jq.piece_color(p), jq.is_dark(p)
            counts[side] += bool(dark)
            x, y = (i % 9 + .5) * CELL, (i // 9 + .5) * CELL
            color = "#b8352b" if side == jq.RED else "#222222"
            self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill="#57687c" if dark else "#fff7e7",
                                    outline=color, width=2)
            self.canvas.create_text(x, y - (3 if dark else 0), text=jq.PIECE_NAMES[side][jq.piece_type(p)],
                                    fill="#ffffff" if dark else color, font=("Microsoft YaHei", 16, "bold"))
            if dark:
                self.canvas.create_text(x, y + 12, text="红暗" if side == jq.RED else "黑暗", fill="#f4c95d", font=("Microsoft YaHei", 7))
        self.count_var.set(f"棋盘暗子：红 {counts[jq.RED]} · 黑 {counts[jq.BLACK]}")

    def _standard(self):
        self.state = jq.GameState.initial()
        self.side_var.set("红方")
        self._sync_pools()

    def _empty(self):
        self.state = jq.GameState.empty(with_pools=False)
        self._sync_pools()

    def _sync_pools(self):
        for (side, t), var in self.pool_vars.items():
            var.set(str(self.state.pool[side][t]))
        self._draw()

    def _apply(self):
        try:
            state = self.state.clone()
            state.dark_types = None
            state.side = jq.RED if self.side_var.get() == "红方" else jq.BLACK
            state.pool = [[0] * 8 for _ in range(2)]
            for (side, t), var in self.pool_vars.items():
                state.pool[side][t] = int(var.get())
            state.pool_total = [sum(row) for row in state.pool]
            validate_position(state)
            state.reset_history()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("局面无效", str(exc), parent=self)
            return
        if self.on_apply(state):
            self.grab_release()
            self.destroy()

    def _close(self):
        self.grab_release()
        self.destroy()
        self.on_close()
