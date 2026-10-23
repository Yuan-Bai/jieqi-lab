"""Local web UI and HTTP adapter for the existing JieQi rules and search engines.

Run ``python web_server.py``; no additional runtime dependencies are needed for RIS.
The HTTP boundary returns public/player observations, never serialized engine state.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import random
import secrets
import threading
import time
from urllib.parse import urlsplit

import ismcts as im
import jieqi as jq
from game_record import GameRecord
from gui_tasks import LatestTaskRunner, SearchBelief
from web_runtime import configure_runtime, prepare_search_thread, restore_runtime, runtime_status

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
REASONS = {"checkmate": "将死", "stalemate": "困毙", "long_check": "长将禁着",
           "long_chase": "长捉禁着", "repetition": "三次重复局面", "no_capture": "连续 80 手无吃子"}


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label}必须是 {low}–{high} 的整数")
    return value


def model_catalog():
    """Only offer known pure-weight files, not arbitrary client-supplied paths."""
    candidates = sorted((ROOT / "data").glob("*/models/*.pt"))
    candidates += sorted((ROOT / "data" / "models").glob("*.pt"))
    candidates.sort(key=lambda p: ("train_rules5_" not in str(p), p.name != "latest.pt", str(p)))
    return {p.relative_to(ROOT).as_posix(): p for p in candidates
            if p.name in ("latest.pt", "best.pt") and not any(part.startswith("legacy") for part in p.parts)
            and p.resolve().is_relative_to(ROOT / "data")}


def move_label(game, node):
    src, dst = node.record["from"], node.record["to"]
    piece = game.nodes[node.parent].state.board[src]
    name = jq.PIECE_NAMES[jq.piece_color(piece)][jq.piece_type(piece)]
    label = f"{'暗' if jq.is_dark(piece) else ''}{name} {jq.square_to_name(src)} → {jq.square_to_name(dst)}"
    if node.record["reveal"]:
        label += " · " + jq.PIECE_NAMES[jq.piece_color(piece)][node.record["reveal"]]
    return label


class WebGame:
    def __init__(self):
        self.lock = threading.RLock()
        self.runner = LatestTaskRunner()
        self.analysis_runner = LatestTaskRunner()
        self.rng = random.Random()
        self.revision = 0
        self.last_seen = time.monotonic()
        self.engine_key = None
        self.engine = None
        self.analysis_engine_key = None
        self.analysis_engine = None
        self.analysis_cache = {}
        models = model_catalog()
        self.analysis_settings = {"engine": "az" if models else "ris", "model": next(iter(models), ""),
                                  "simulations": 200, "leaf_batch_size": 64,
                                  "auto_hint": False, "auto_evaluate": False}
        self.new({})
        self.preferences_ready = False

    def cancel_play(self):
        self.runner.invalidate()
        self.pending = None

    def invalidate_analysis(self, *, clear_cache=False):
        self.analysis_runner.invalidate()
        self.analysis_pending = None
        self.analysis_result = None
        self.analysis_result_key = None
        self.analysis_error = None
        self.analysis_failed_key = None
        self.hint = None
        if clear_cache:
            self.analysis_cache.clear()

    def cancel(self):
        self.cancel_play()
        self.invalidate_analysis()

    def close(self):
        self.runner.close()
        self.analysis_runner.close()

    @staticmethod
    def search_settings(data, defaults=None):
        defaults = defaults or {"engine": "ris", "model": "", "simulations": 64, "leaf_batch_size": 64}
        settings = {key: data.get(key, defaults[key]) for key in ("engine", "model", "simulations", "leaf_batch_size")}
        integer(settings["simulations"], 1, 100000, "模拟次数")
        integer(settings["leaf_batch_size"], 1, 4096, "叶批大小")
        if settings["engine"] not in ("ris", "az"):
            raise ValueError("不支持的引擎")
        if settings["engine"] == "az" and settings["model"] not in model_catalog():
            raise ValueError("请选择本机已有的 latest.pt 或 best.pt 模型")
        return settings

    def play_settings(self):
        return {"engine": self.engine_kind, "model": self.model, "simulations": self.simulations,
                "leaf_batch_size": self.leaf_batch_size}

    def new(self, data):
        mode = data.get("mode", "human")
        if mode not in ("human", "local", "aiai"):
            raise ValueError("不支持的对弈模式")
        side = integer(data.get("human_side", jq.RED), 0, 1, "执子颜色")
        settings = self.search_settings(data)
        self.cancel()
        self.mode, self.human_side = mode, side
        self.simulations, self.leaf_batch_size = settings["simulations"], settings["leaf_batch_size"]
        self.engine_kind, self.model = settings["engine"], settings["model"]
        self.analysis_cache.clear()
        self.game = GameRecord(metadata={"mode": "study", "source": "jieqi-web"})
        self.record_id = secrets.token_hex(8)
        self.head = 0
        self.paused = mode == "aiai"
        self.reviewing = False
        self.resigned = None
        self.hint = None
        self.error = None
        self.last_search = None
        self.revision += 1

    def outcome(self):
        if self.resigned is not None and not self.reviewing:
            return (-1 if self.resigned == jq.RED else 1), "resign"
        return self.game.current.state.game_outcome()

    def ai_turn(self):
        return self.mode == "aiai" or (self.mode == "human" and self.game.current.state.side != self.human_side)

    def start_search(self, kind):
        if kind == "hint":
            return self.start_analysis(want_hint=True)
        if self.pending:
            if self.pending["kind"] == kind:
                return
            raise ValueError("请等待当前搜索完成")
        node = self.game.current
        if self.outcome()[0] is not None:
            raise ValueError("对局已结束")
        state, belief = node.state.clone(), node.belief.clone()
        # NeuralMCTS encodes the last eight (from, to) moves, not board snapshots.
        history = [(self.game.nodes[i].record["from"], self.game.nodes[i].record["to"])
                   for i in self.game.path()[-8:]]
        key = (self.engine_kind, self.model)
        sims = self.simulations
        leaf = min(self.leaf_batch_size, sims)

        def search(rng):
            prepare_search_thread()
            observed = SearchBelief(belief, rng)
            if key[0] == "ris":
                agent = im.ISMCTSAgent(im.ISMCTSConfig(iterations=sims), rng)
                move, _ = agent.select_move(state, observed)
                return {"move": move, "value": None, "device": "cpu", "finished_at": time.monotonic()}
            if self.engine_key != key:
                from search_engine import AzEngine
                self.engine = AzEngine(str(model_catalog()[key[1]]), simulations=sims, rng=rng)
                self.engine_key = key
            self.engine.config.leaf_batch_size = leaf
            advice = self.engine.analyse(state, observed, history, simulations=sims)
            return {"move": advice.best.move if advice.best else None, "value": advice.win_rate,
                    "device": str(getattr(self.engine, "device", "unknown")), "finished_at": time.monotonic()}

        token = self.runner.submit(search)
        self.pending = {"token": token, "kind": kind, "node": node.id, "started": time.monotonic(),
                        "simulations": sims, "leaf_batch_size": leaf}
        self.error = None

    def analysis_key(self):
        cfg = self.analysis_settings
        stamp = None
        if cfg["engine"] == "az":
            path = model_catalog().get(cfg["model"])
            if path is not None:
                stat = Path(path).stat()
                stamp = (stat.st_mtime_ns, stat.st_size)
        return (id(self.game), self.game.cursor, cfg["engine"], cfg["model"],
                cfg["simulations"], cfg["leaf_batch_size"], stamp)

    def start_analysis(self, *, want_hint=False, want_evaluation=False):
        if self.outcome()[0] is not None:
            raise ValueError("对局已结束")
        cfg = dict(self.analysis_settings)
        if want_evaluation and cfg["engine"] != "az":
            raise ValueError("局势评估需要 AlphaZero，请在分析设置中选择模型；RIS 仅提供走法提示")
        key = self.analysis_key()
        if self.analysis_pending and self.analysis_pending["key"] == key:
            self.analysis_pending["want_hint"] |= want_hint
            return
        if key in self.analysis_cache:
            self.analysis_result = self.analysis_cache[key]
            self.analysis_result_key = key
            self.analysis_error = None
            if want_hint:
                self.hint = self.analysis_result["move"]
            return
        self.analysis_runner.invalidate()
        self.analysis_result = None
        self.analysis_result_key = None
        self.hint = None
        self.analysis_failed_key = self.analysis_error = None
        node = self.game.current
        state, belief = node.state.clone(), node.belief.clone()
        history = [(self.game.nodes[i].record["from"], self.game.nodes[i].record["to"])
                   for i in self.game.path()[-8:]]
        engine_key = (cfg["engine"], cfg["model"], key[-1])
        sims, leaf = cfg["simulations"], min(cfg["leaf_batch_size"], cfg["simulations"])

        def search(rng):
            prepare_search_thread()
            observed = SearchBelief(belief, rng)
            if cfg["engine"] == "ris":
                agent = im.ISMCTSAgent(im.ISMCTSConfig(iterations=sims), rng)
                move, _ = agent.select_move(state, observed)
                return {"move": move, "win_rate": None, "device": "cpu"}
            if self.analysis_engine_key != engine_key:
                from search_engine import AzEngine
                self.analysis_engine = AzEngine(str(model_catalog()[cfg["model"]]), simulations=sims, rng=rng)
                self.analysis_engine_key = engine_key
            self.analysis_engine.config.leaf_batch_size = leaf
            advice = self.analysis_engine.analyse(state, observed, history, simulations=sims)
            return {"move": advice.best.move if advice.best else None, "win_rate": advice.win_rate,
                    "device": str(getattr(self.analysis_engine, "device", "unknown"))}

        token = self.analysis_runner.submit(search)
        self.analysis_pending = {"token": token, "key": key, "node": node.id, "side": state.side,
                                 "settings": cfg, "want_hint": want_hint, "started": time.monotonic()}

    def pump_analysis(self):
        key = self.analysis_key()
        if self.analysis_result is not None and self.analysis_result_key != key:
            self.invalidate_analysis()
        if self.analysis_pending:
            if self.analysis_pending["key"] != key:
                self.invalidate_analysis()
            else:
                result = self.analysis_runner.poll(self.analysis_pending["token"])
                if result:
                    task, self.analysis_pending = self.analysis_pending, None
                    if result.error:
                        self.analysis_error = "分析失败：" + result.error
                        self.analysis_failed_key = key
                    else:
                        cfg, value = task["settings"], result.value
                        red = value["win_rate"] if task["side"] == jq.RED else (1 - value["win_rate"] if value["win_rate"] is not None else None)
                        move = list(value["move"]) if value["move"] is not None else None
                        label = ""
                        if move is not None:
                            piece = self.game.current.state.board[move[0]]
                            label = f"{'暗' if jq.is_dark(piece) else ''}{jq.PIECE_NAMES[task['side']][jq.piece_type(piece)]} {jq.square_to_name(move[0])} → {jq.square_to_name(move[1])}"
                        self.analysis_result = {"node": task["node"], "side": task["side"],
                            "move": move, "label": label, "engine": cfg["engine"],
                            "simulations": cfg["simulations"], "leaf_batch_size": cfg["leaf_batch_size"],
                            "effective_leaf_batch_size": min(cfg["leaf_batch_size"], cfg["simulations"]) if cfg["engine"] == "az" else None,
                            "seconds": round(result.elapsed, 2), "win_rate": value["win_rate"],
                            "device": value["device"],
                            "red_win_rate": red, "black_win_rate": 1 - red if red is not None else None}
                        self.analysis_cache[key] = self.analysis_result
                        self.analysis_result_key = key
                        if len(self.analysis_cache) > 32:
                            self.analysis_cache.pop(next(iter(self.analysis_cache)))
                        if task["want_hint"]:
                            self.hint = move
        cfg = self.analysis_settings
        want_hint = cfg["auto_hint"] and (not self.ai_turn() or self.reviewing)
        if (not self.analysis_pending and self.analysis_failed_key != key and self.outcome()[0] is None
                and (want_hint or cfg["auto_evaluate"])):
            if self.analysis_result is None or self.analysis_result["node"] != self.game.cursor:
                self.start_analysis(want_hint=want_hint, want_evaluation=cfg["auto_evaluate"])

    def pump(self):
        if self.pending:
            result = self.runner.poll(self.pending["token"])
            if result:
                task, self.pending = self.pending, None
                if task["node"] == self.game.cursor:
                    if result.error:
                        self.error = "引擎搜索失败：" + result.error
                        self.paused = True
                    elif result.value["move"] is not None:
                        self.last_search = {"seconds": round(result.elapsed, 2), "simulations": task["simulations"],
                                            "leaf_batch_size": task["leaf_batch_size"] if self.engine_kind == "az" else None,
                                            "device": result.value["device"],
                                            "result_wait_seconds": round(max(0, time.monotonic() - result.value["finished_at"]), 3),
                                            "win_rate": result.value["value"]}
                        if task["kind"] == "hint":
                            self.hint = list(result.value["move"])
                        else:
                            self.game.add_move(result.value["move"], rng=self.rng)
                            self.head = self.game.cursor
                            self.invalidate_analysis()
                    self.revision += 1
        if (not self.pending and not self.paused and not self.reviewing and not self.error
                and self.ai_turn() and self.outcome()[0] is None):
            self.start_search("move")

    def public_state(self):
        self.pump()
        self.pump_analysis()
        node, state = self.game.current, self.game.current.state
        outcome, reason = self.outcome()
        # Local two-player and AI spectators receive only public capture information.
        observer = self.human_side if self.mode == "human" else None
        captured = {}
        for side in (jq.BLACK, jq.RED):
            captured[str(side)] = [{"side": color, "dark": dark, "node": None, "depth": None,
                                   "type": kind if not dark or outcome is not None or observer == side else None}
                                  for color, dark, kind in node.captured[side]]
        # Associate each visible capture with its actual move, only on the current branch.
        # Initial-position captures have no known move; no private reveal fields are sent.
        for node_id in self.game.path():
            item = self.game.nodes[node_id]
            parent = self.game.nodes[item.parent]
            actor = parent.state.side
            index = len(parent.captured[actor])
            if len(item.captured[actor]) > index:
                captured[str(actor)][index].update(node=item.id, depth=item.depth,
                    source=item.record["from"], target=item.record["to"])
        history = []
        for item in self.game.nodes.values():
            if item.parent is not None:
                piece = self.game.nodes[item.parent].state.board[item.record["from"]]
                history.append({"id": item.id, "parent": item.parent, "depth": item.depth,
                                "side": jq.piece_color(piece), "label": move_label(self.game, item),
                                "from": item.record["from"], "to": item.record["to"]})
        return {"revision": self.revision, "record_id": self.record_id, "board": state.board.copy(), "side": state.side,
                "preferences_ready": self.preferences_ready,
                "legal_moves": [list(m) for m in state.legal_moves()] if outcome is None else [],
                "cursor": node.id, "head": self.head, "depth": node.depth, "history": history,
                "children": node.children.copy(), "parent": node.parent,
                "last_move": [node.record["from"], node.record["to"]] if node.record else None,
                "mode": self.mode, "human_side": self.human_side, "simulations": self.simulations,
                "leaf_batch_size": self.leaf_batch_size,
                "effective_leaf_batch_size": min(self.leaf_batch_size, self.simulations) if self.engine_kind == "az" else None,
                "engine": self.engine_kind, "model": self.model, "paused": self.paused,
                "reviewing": self.reviewing, "ai_turn": self.ai_turn(), "hint": self.hint,
                "pending": {"kind": self.pending["kind"], "seconds": round(time.monotonic() - self.pending["started"], 1)} if self.pending else None,
                "error": self.error, "last_search": self.last_search, "captured": captured,
                "export_format": "jieqi-replay" if self.can_export_replay() else "jieqi-web-public",
                "analysis_settings": dict(self.analysis_settings), "analysis": self.analysis_result,
                "analysis_pending": {"node": self.analysis_pending["node"],
                    "seconds": round(time.monotonic() - self.analysis_pending["started"], 1)} if self.analysis_pending else None,
                "analysis_error": self.analysis_error,
                "outcome": outcome, "reason": REASONS.get(reason, "认输" if reason == "resign" else reason),
                "in_check": state.is_in_check(state.side), "dark_remaining": state.pool_total.copy(),
                "no_capture_plies": state.no_capture_plies, "ruleset_version": jq.RULESET_VERSION}

    def act(self, action, data):
        if data.get("revision") != self.revision:
            raise ValueError("局面已经更新，请在当前局面重试")
        if action == "new":
            self.new(data)
            self.preferences_ready = True
            return
        if action == "settings":
            settings = self.search_settings(data, self.play_settings())
            self.cancel_play()
            self.engine_kind, self.model = settings["engine"], settings["model"]
            self.simulations, self.leaf_batch_size = settings["simulations"], settings["leaf_batch_size"]
            self.last_search = None
            if self.error:
                self.paused = False
            self.error = None
        elif action == "analysis-settings":
            settings = self.search_settings(data, self.analysis_settings)
            for name in ("auto_hint", "auto_evaluate"):
                value = data.get(name, self.analysis_settings[name])
                if type(value) is not bool:
                    raise ValueError("分析开关必须是布尔值")
                settings[name] = value
            if settings["auto_evaluate"] and settings["engine"] != "az":
                raise ValueError("自动评估需要 AlphaZero 模型")
            self.invalidate_analysis()
            self.analysis_settings = settings
        elif action == "move":
            if self.pending or self.reviewing or self.paused or self.ai_turn():
                raise ValueError("当前不能手动走子，请先回到对局或等待电脑落子")
            if self.outcome()[0] is not None:
                raise ValueError("对局已结束")
            if len(self.game.nodes) >= 2000:
                raise ValueError("棋谱已达到 2000 个节点，请导出记录后开启新局")
            move = (integer(data.get("from"), 0, 89, "起点"), integer(data.get("to"), 0, 89, "终点"))
            self.game.add_move(move, rng=self.rng)
            self.head = self.game.cursor
            self.invalidate_analysis()
            self.last_search = None
        elif action == "undo":
            if self.reviewing:
                raise ValueError("复盘中请使用上一步")
            self.cancel()
            current = self.game.current
            if current.parent is None:
                raise ValueError("还没有可以悔棋的走子")
            self.game.cursor = current.parent
            if self.mode == "human":
                while self.game.current.parent is not None and self.game.current.state.side != self.human_side:
                    self.game.cursor = self.game.current.parent
            self.head = self.game.cursor
            self.hint = None
            self.resigned = None
            self.error = None
            self.paused = self.mode == "aiai"
        elif action == "navigate":
            cursor = integer(data.get("cursor"), 0, len(self.game.nodes) - 1, "棋谱节点")
            self.cancel()
            self.game.cursor = cursor
            self.reviewing = True
            self.paused = True
            self.hint = None
        elif action == "resume":
            self.cancel()
            self.game.cursor = self.head
            self.reviewing = False
            self.paused = False
            self.error = None
            self.hint = None
        elif action == "continue":
            self.cancel()
            self.head = self.game.cursor
            self.reviewing = False
            self.paused = False
            self.resigned = None
            self.error = None
            self.hint = None
        elif action == "pause":
            self.cancel()
            self.paused = not self.paused
            self.error = None
        elif action == "hint":
            if self.ai_turn() and not self.reviewing:
                raise ValueError("请等待电脑回合完成")
            self.start_analysis(want_hint=True)
        elif action == "evaluate":
            self.start_analysis(want_evaluation=True)
        elif action == "resign":
            if self.mode != "human" or self.reviewing or self.outcome()[0] is not None:
                raise ValueError("当前不能认输")
            self.cancel()
            self.resigned = self.human_side
        elif action == "import":
            record = data.get("record")
            if not isinstance(record, dict) or len(record.get("nodes", record.get("moves", []))) > 2000:
                raise ValueError("棋谱格式错误或超过 2000 个节点")
            try:
                game = GameRecord.from_dict(record, migrate_rules=True)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError, RecursionError) as exc:
                raise ValueError("棋谱格式错误：" + str(exc)) from exc
            # A saved review cursor and the live end of the match are independent.
            resignation = record.get("web_resignation")
            resigned = None
            saved_head = game.cursor
            if resignation is not None:
                if not isinstance(resignation, dict):
                    raise ValueError("棋谱认输记录无效")
                resigned = integer(resignation.get("side"), 0, 1, "认输方")
                saved_head = integer(resignation.get("node"), 0, len(game.nodes) - 1, "认输节点")
            head = integer(record.get("web_head", saved_head), 0, len(game.nodes) - 1, "对局最新节点")
            if resignation is not None and head != saved_head:
                raise ValueError("棋谱认输节点与对局最新节点不一致")
            self.cancel()
            self.game = game
            self.record_id = secrets.token_hex(8)
            self.head = head
            self.mode = "local"
            self.reviewing = True
            self.paused = True
            self.resigned = resigned
            self.error = self.hint = self.last_search = None
        else:
            raise ValueError("未知操作")
        self.revision += 1
        self.preferences_ready = True

    def can_export_replay(self):
        # Full GUI-compatible records contain private capture identities. Export them
        # only after the *live* game ends, never merely because a review node ends.
        live_outcome = self.game.nodes[self.head].state.game_outcome()[0]
        return live_outcome is not None or self.resigned is not None

    def export(self):
        if self.can_export_replay():
            record = self.game.to_dict()
            record["web_head"] = self.head
            # Imported metadata can describe a resignation that was later undone.
            record.pop("web_resignation", None)
            if self.resigned is not None:
                record["web_resignation"] = {"side": self.resigned, "node": self.head}
            return record
        view = self.public_state()
        return {"format": "jieqi-web-public", "version": 1, "ruleset_version": jq.RULESET_VERSION,
                "note": "未终局：仅导出公开着法。完整含私有揭示的 GUI 棋谱在终局后可导出。",
                "cursor": view["cursor"], "history": view["history"]}


class WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        super().__init__(address, Handler)
        self.trusted_hosts = trusted_http_hosts(address[0], self.server_address)
        self.games = {}
        self.games_lock = threading.Lock()

    def session(self, cookie):
        parsed = SimpleCookie()
        try:
            parsed.load(cookie or "")
        except Exception:
            pass
        session_id = parsed["jieqi_session"].value if "jieqi_session" in parsed else None
        with self.games_lock:
            now = time.monotonic()
            for key in list(self.games):
                if now - self.games[key].last_seen > 12 * 3600:
                    self.games.pop(key).close()
            if session_id not in self.games:
                if len(self.games) >= 32:
                    raise ValueError("本地服务会话已满，请稍后重试")
                session_id = secrets.token_urlsafe(32)
                self.games[session_id] = WebGame()
            game = self.games[session_id]
            game.last_seen = now
            return session_id, game

    def server_close(self):
        for game in self.games.values():
            game.close()
        super().server_close()


def trusted_http_hosts(configured_host, bound_address):
    """Build an authority allowlist from server configuration, never request headers."""
    host, port = bound_address[:2]
    hosts = {configured_host.lower(), host.lower()}
    if host == "0.0.0.0" or host.startswith("127."):
        hosts.update(("127.0.0.1", "localhost"))
    hosts.discard("0.0.0.0")  # A wildcard listener does not trust arbitrary Host values.
    hosts.discard("")
    authorities = {f"{name}:{port}" for name in hosts}
    if port == 80:
        authorities.update(hosts)
    return frozenset(authorities)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if args and str(args[1] if len(args) > 1 else "") != "200":
            super().log_message(format, *args)

    def send_bytes(self, status, content, content_type, cookie=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if cookie:
            self.send_header("Set-Cookie", f"jieqi_session={cookie}; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status, value, cookie=None):
        self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", cookie)

    def trusted_request(self):
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        if len(hosts) != 1 or hosts[0].lower() not in self.server.trusted_hosts:
            return False
        if len(origins) > 1:
            return False
        # Origin is checked against a validated authority, including on read endpoints.
        if origins and origins[0] != f"http://{hosts[0].lower()}":
            return False
        return self.headers.get("Sec-Fetch-Site") != "cross-site"

    def do_GET(self):
        if not self.trusted_request():
            return self.send_json(403, {"error": "仅接受可信主机的同源请求"})
        path = urlsplit(self.path).path
        try:
            if path == "/api/health":
                return self.send_json(200, {"ok": True, "ruleset_version": jq.RULESET_VERSION,
                                            "runtime": runtime_status()})
            if path == "/api/config":
                return self.send_json(200, {"models": list(model_catalog()), "engines": ["ris", "az"]})
            if path in ("/api/state", "/api/export"):
                sid, game = self.server.session(self.headers.get("Cookie"))
                with game.lock:
                    value = game.public_state() if path.endswith("state") else game.export()
                return self.send_json(200, value, sid)
            asset = WEB / ("index.html" if path == "/" else path.lstrip("/"))
            if path.startswith("/sounds/"):
                asset = ROOT / "assets" / "sounds" / Path(path).name
            allowed = asset.resolve().is_relative_to(WEB) or (path.startswith("/sounds/") and asset.suffix == ".wav")
            if not allowed or not asset.is_file():
                return self.send_json(404, {"error": "页面不存在"})
            content_type = mimetypes.guess_type(str(asset))[0] or "application/octet-stream"
            if content_type.startswith("text/") or asset.suffix == ".js":
                content_type += "; charset=utf-8"
            self.send_bytes(200, asset.read_bytes(), content_type)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})

    def do_POST(self):
        # Browser writes require JSON and same-origin fetches. No CORS is exposed.
        if not self.trusted_request():
            return self.send_json(403, {"error": "仅接受可信主机的同源请求"})
        if self.headers.get_content_type() != "application/json":
            return self.send_json(415, {"error": "需要 application/json"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2_000_000:
                return self.send_json(413, {"error": "请求大小无效，最大 2 MB"})
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("请求必须是 JSON 对象")
            path = urlsplit(self.path).path
            if not path.startswith("/api/"):
                return self.send_json(404, {"error": "接口不存在"})
            sid, game = self.server.session(self.headers.get("Cookie"))
            with game.lock:
                game.act(path[5:], data)
                value = game.public_state()
            self.send_json(200, value, sid)
        except (ValueError, KeyError, TypeError, IndexError, RecursionError) as exc:
            self.send_json(400, {"error": str(exc) or "请求或棋谱格式不正确"})


def main():
    parser = argparse.ArgumentParser(description="Jieqi Lab 本地网页版")
    parser.add_argument("--host", default="127.0.0.1", help="默认仅本机可访问")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scheduling", choices=("responsive", "system"), default="responsive",
                        help="Windows 默认禁用本进程的后台节流并使用 AboveNormal；system 保留系统调度")
    args = parser.parse_args()
    server = WebServer((args.host, args.port))
    runtime = configure_runtime(args.scheduling)
    print(f"Jieqi Lab: http://{args.host}:{server.server_port}", flush=True)
    print("按 Ctrl+C 停止。对局保存在服务进程内，刷新页面可继续。", flush=True)
    print(f"Scheduling: {runtime['mode']}; priority={runtime['priority']}; "
          f"HighQoS={runtime['power_throttling_disabled']}", flush=True)
    for note in runtime["notes"]:
        print(note, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        restore_runtime()


if __name__ == "__main__":
    main()
