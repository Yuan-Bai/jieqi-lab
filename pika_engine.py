"""External Pikafish jieqi_old opponent, using only the acting player's view.

The project adjudicates games and restricts root moves. This bridge deliberately
does not claim identical rules inside Pika's search: current-position FEN cannot
express our complete repetition/check/chase history. Never use jieqi.to_fen here:
that export derives captured counts from the environment's hidden pool.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

import ismcts as im
import jieqi as jq


BRIDGE_VERSION = 1
RULES_SCOPE = "project_referee_and_root_legal_moves; pika_internal_rules_without_project_history"
_MOVE = re.compile(r"[a-i][0-9][a-i][0-9]\Z")
# The project exports H/E, while this Pika branch requires N/B.
_LETTERS = {jq.TYPE_KING: "K", jq.TYPE_ROOK: "R", jq.TYPE_KNIGHT: "N",
            jq.TYPE_ELEPHANT: "B", jq.TYPE_ADVISOR: "A",
            jq.TYPE_CANNON: "C", jq.TYPE_PAWN: "P"}


class PikaError(RuntimeError):
    """Protocol/compatibility failure: abort the run, never score as a chess loss."""


def _file_record(path) -> dict:
    path = Path(path).resolve(strict=True)
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(path=str(path), sha256=sha, bytes=path.stat().st_size)


def prepare_config(config: dict | None) -> dict:
    config = dict(config or {})
    executable = config.get("executable")
    if not executable:
        raise ValueError("pika opponent requires --pika-exe pointing to jieqi_old")
    source = _file_record(executable)
    if config.get("source") is not None and config["source"] != source:
        raise PikaError("Pika executable changed during evaluation")
    limit = config.get("limit", "nodes")
    threads, hash_mb = config.get("threads", 1), config.get("hash_mb", 32)
    timeout = float(config.get("timeout", 30))
    if limit not in ("nodes", "movetime"):
        raise ValueError("Pika limit must be nodes or movetime (milliseconds)")
    if not isinstance(threads, int) or not 1 <= threads <= 1024:
        raise ValueError("Pika threads must be in [1, 1024]")
    if not isinstance(hash_mb, int) or not 1 <= hash_mb <= 2048:
        raise ValueError("Pika hash_mb must be in [1, 2048]")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Pika timeout must be finite and positive")
    return dict(executable=source["path"], source=source, limit=limit,
                threads=threads, hash_mb=hash_mb, timeout=timeout,
                bridge_version=BRIDGE_VERSION, rules_scope=RULES_SCOPE,
                information_scope="acting_player_count_belief; no_environment_hidden_state",
                hash_scope="fresh_process_per_game")


def information_fen(info: im.InformationState, ply: int = 0) -> str:
    """Encode Pika's remaining-pool FEN, not the project's captured-pool FEN.

    Counts include unknown captures for the owner: sum(pool_known)-onboard dark
    count is intentionally not removed using hidden truth or a guessed particle.
    """
    rows = []
    dark_counts = [0, 0]
    kings = [0, 0]
    for row in range(jq.BOARD_H):
        parts, empty = [], 0
        for col in range(jq.BOARD_W):
            square = jq.index(row, col)
            piece = info.board[square]
            if not piece:
                empty += 1
                continue
            if empty:
                parts.append(str(empty))
                empty = 0
            color, kind = jq.piece_color(piece), jq.piece_type(piece)
            if jq.is_dark(piece):
                if kind == jq.TYPE_KING or kind != jq.fake_type_for_square(square):
                    raise ValueError("Pika cannot encode this dark piece's fake type")
                dark_counts[color] += 1
                letter = "X"
            else:
                letter = _LETTERS[kind]
                kings[color] += kind == jq.TYPE_KING
            parts.append(letter if color == jq.RED else letter.lower())
        if empty:
            parts.append(str(empty))
        rows.append("".join(parts))
    if kings != [1, 1]:
        raise ValueError("Pika requires both kings on the board")
    pool = []
    for color in (jq.RED, jq.BLACK):
        counts = info.pool_known[color]
        if counts[jq.TYPE_KING] != 0:
            raise ValueError("king cannot be in dark pool")
        if dark_counts[color] != info.pool_total_known[color]:
            raise ValueError("visible dark count and observer pool total disagree")
        if sum(counts) - dark_counts[color] != info.unknown_removals[color]:
            raise ValueError("unknown capture count and observer pool disagree")
        for kind in jq.POOL_TYPES:
            count = counts[kind]
            if not isinstance(count, int) or not 0 <= count <= jq.POOL_INIT[kind]:
                raise ValueError("invalid observer pool count")
            letter = _LETTERS[kind]
            pool.append(f"{letter if color == jq.RED else letter.lower()}{count}")
    side = "w" if info.side == jq.RED else "b"
    return f"{'/'.join(rows)} {side} {''.join(pool)} {info.no_capture_plies} {ply // 2 + 1}"


class UciProcess:
    """One process with bounded protocol waits and explicit lifetime ownership."""
    def __init__(self, command: list[str], timeout: float, cwd: str | None = None):
        self.timeout = timeout
        self.lines = queue.Queue()
        self.recent = deque(maxlen=12)
        self.process = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line.strip())
        finally:
            self.lines.put(None)

    def send(self, command: str):
        if "\n" in command or "\r" in command:
            raise ValueError("UCI command contains a newline")
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as error:
            raise PikaError("Pika input pipe closed") from error

    def wait_for(self, marker: str) -> list[str]:
        deadline = time.monotonic() + self.timeout
        response = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PikaError(f"Pika timeout waiting for {marker}: {list(self.recent)}")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty as error:
                raise PikaError(f"Pika timeout waiting for {marker}: {list(self.recent)}") from error
            if line is None:
                raise PikaError(f"Pika exited before {marker}: {list(self.recent)}")
            self.recent.append(line)
            response.append(line)
            if "ERROR" in line.upper():
                raise PikaError(f"Pika reported error: {line}")
            if line == marker or line.startswith(marker + " "):
                return response

    def close(self):
        process = self.process
        if process.poll() is None:
            try:
                self.send("quit")
                process.wait(timeout=1)
            except (PikaError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
        self.reader.join(timeout=1)
        for pipe in (process.stdin, process.stdout):
            try:
                pipe.close()
            except OSError:
                pass  # Broken/exited child can make stdin's final flush fail.


class PikaOpponent:
    def __init__(self, config: dict | None, budget: int):
        self.config = prepare_config(config)
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("Pika budget must be a positive integer")
        if self.config["limit"] == "movetime" and budget / 1000 >= self.config["timeout"]:
            raise ValueError("Pika protocol timeout must exceed movetime")
        self.budget = budget
        self.search_nodes = 0
        self.searches = 0
        self.engine = UciProcess([self.config["executable"]], self.config["timeout"],
                                 cwd=str(Path(self.config["executable"]).parent))
        self.identity = []
        try:
            self.engine.send("uci")
            response = self.engine.wait_for("uciok")
            self.identity = [line for line in response if line.startswith("id ")]
            for name, value in (("Threads", self.config["threads"]),
                                ("Hash", self.config["hash_mb"]), ("MultiPV", 1)):
                if not any(line.startswith(f"option name {name} type ") for line in response):
                    raise PikaError(f"Pika missing required UCI option: {name}")
                self.engine.send(f"setoption name {name} value {value}")
            self.engine.send("ucinewgame")
            self.engine.send("isready")
            self.engine.wait_for("readyok")
            prepare_config(self.config)  # Detect replacement during launch.
        except BaseException:
            self.close()
            raise

    def select_move(self, state: jq.GameState, belief: im.BeliefModel, history):
        legal = state.legal_moves()
        if not legal:
            return None
        # This is the only observation passed to the external engine.
        info = belief.info_state(state, state.side)
        fen = information_fen(info, len(history))
        self.engine.send("position fen " + fen)
        self.engine.send(f"go {self.config['limit']} {self.budget} searchmoves "
                         + " ".join(jq.format_move_uci(move) for move in legal))
        try:
            response = self.engine.wait_for("bestmove")
        except PikaError as error:
            raise PikaError(f"{error}; observer FEN={fen}") from error
        line = response[-1]
        node_lines = [re.search(r"\bnodes (\d+)\b", text) for text in response if text.startswith("info ")]
        node_counts = [int(match[1]) for match in node_lines if match]
        self.search_nodes += node_counts[-1] if node_counts else 0
        self.searches += 1
        fields = line.split()
        if len(fields) < 2 or not _MOVE.fullmatch(fields[1]):
            raise PikaError(f"invalid/empty Pika bestmove in nonterminal position: {line}")
        move = (jq.name_to_square(fields[1][:2]), jq.name_to_square(fields[1][2:]))
        if move not in legal:
            raise PikaError(f"Pika ignored project root legality: {line}")
        return move

    def metadata(self):
        return dict(engine_identity=self.identity, reported_search_nodes=self.search_nodes,
                    searches=self.searches, rules_scope=RULES_SCOPE)

    def close(self):
        self.engine.close()
