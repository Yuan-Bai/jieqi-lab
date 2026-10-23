"""探针：量化搜索树 children 与模拟世界不一致导致的"走后自将"着法执行率。

历史缺陷（Issue 0007，2026-10-01 已修复）：off 模式下暗子走子后的节点 children 由**首次展开时**那条
模拟的世界（揭示结果）生成，之后全搜索共享；不同模拟的揭示结果不同 → 棋盘不同
→ 合法着法集合不同。跨世界复用的 children 会包含"当前模拟世界下非法"的着法，
而 `apply_move` 不做合法性校验，非法着法会在模拟中真实执行。

口径：
- 包装 `GameState.apply_move`：搜索路径/对局步调用后，检查"移动方（走后轮到的一方
  取反）是否仍被将军"（含照面），记为 self_check。搜索内 legal_moves 试走的
  apply_move 不计入（含 has_legal_moves 与终局扫描）。
- random 模式（随机合法着法）作为基线：只反映"掀棋自将"（暗子揭示副作用），不涉及
  树结构共享。
- off/on 模式走搜索：self_check 超基线部分 ≈ 树与世界不匹配造成的非法执行率。

用法：
    python tools/probe_world_mismatch.py --mode random,off,on --games 3 \
        --sims 400 --out data/eval/probe_world_mismatch.jsonl
"""

import argparse
import json
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

import azero
import ismcts as im
import jieqi as jq


STATS = {
    "mode": "idle",
    "scan_depth": 0,
    "search_applies": 0,
    "search_self_check": 0,
    "search_self_check_dark": 0,
    "search_self_check_bright": 0,
    "game_applies": 0,
    "game_self_check": 0,
}

_orig_apply = jq.GameState.apply_move
_orig_legal = jq.GameState.legal_moves
_orig_has_legal = jq.GameState.has_legal_moves
_orig_scan = jq.GameState._move_scan


def _wrapped_legal(self):
    STATS["scan_depth"] += 1
    try:
        return _orig_legal(self)
    finally:
        STATS["scan_depth"] -= 1


def _wrapped_has_legal(self):
    STATS["scan_depth"] += 1
    try:
        return _orig_has_legal(self)
    finally:
        STATS["scan_depth"] -= 1


def _wrapped_scan(self):
    STATS["scan_depth"] += 1
    try:
        return _orig_scan(self)
    finally:
        STATS["scan_depth"] -= 1


def _wrapped_apply(self, move, rng=None, reveal_choice=None, capture_reveal_choice=None):
    dark = bool(jq.is_dark(self.board[move[0]])) if self.board[move[0]] else False
    undo = _orig_apply(
        self, move, rng=rng, reveal_choice=reveal_choice, capture_reveal_choice=capture_reveal_choice
    )
    if STATS["scan_depth"] == 0 and STATS["mode"] in ("search", "game"):
        bad = self.is_in_check(1 - self.side)
        if STATS["mode"] == "search":
            STATS["search_applies"] += 1
            if bad:
                STATS["search_self_check"] += 1
                if dark:
                    STATS["search_self_check_dark"] += 1
                else:
                    STATS["search_self_check_bright"] += 1
        else:
            STATS["game_applies"] += 1
            if bad:
                STATS["game_self_check"] += 1
    return undo


jq.GameState.apply_move = _wrapped_apply
jq.GameState.legal_moves = _wrapped_legal
jq.GameState.has_legal_moves = _wrapped_has_legal
jq.GameState._move_scan = _wrapped_scan


def snapshot():
    return {k: v for k, v in STATS.items() if k not in ("mode", "scan_depth")}


def play_one(mode, sims, max_plies, game_idx, base_seed, net, device, chance_k=0):
    rng = random.Random(base_seed + game_idx)
    dark_types = jq.materialize_dark_types(random.Random(f"wm-probe:{base_seed + game_idx}"))
    state = jq.GameState.initial(rng=rng, dark_types=dark_types)
    belief = im.init_belief_for_game(state)
    before = snapshot()
    start = time.perf_counter()
    ply = 0
    chance_applied = chance_nav_miss = chance_outcomes_added = 0
    while state.game_outcome()[0] is None and ply < max_plies:
        moves = state.legal_moves()
        if not moves:
            break
        if mode == "random":
            move = rng.choice(moves)
            STATS["mode"] = "game"
            im.apply_move_and_update_belief(state, belief, move, rng=rng)
        elif mode == "ris":
            STATS["mode"] = "search"
            move, _ = im.ris_ismcts(state, belief, iterations=sims, rng=rng)
            STATS["mode"] = "game"
            if move is None:
                break
            im.apply_move_and_update_belief(state, belief, move, rng=rng)
        else:
            config = azero.AZConfig(
                mcts_simulations=sims,
                chance_nodes=(mode == "on"),
                chance_k=chance_k,
                leaf_batch_size=64,
            )
            mcts = azero.NeuralMCTS(net, config, device, rng)
            STATS["mode"] = "search"
            _, move = mcts.search(
                state, belief, [], temperature=0.0, simulations=sims, add_dirichlet=False
            )
            chance_applied += mcts.chance_applied
            chance_nav_miss += mcts.chance_nav_miss
            chance_outcomes_added += mcts.chance_outcomes_added
            STATS["mode"] = "game"
            if move is None:
                break
            im.apply_move_and_update_belief(state, belief, move, rng=rng)
        ply += 1
    STATS["mode"] = "idle"
    after = snapshot()
    row = {"mode": mode, "game": game_idx, "plies": ply, "seconds": round(time.perf_counter() - start, 1)}
    for key in before:
        row[key] = after[key] - before[key]
    outcome, reason = state.game_outcome()
    row["outcome"] = None if outcome is None else int(outcome)
    row["reason"] = reason
    row["chance_k"] = chance_k
    row["chance_applied"] = chance_applied
    row["chance_nav_miss"] = chance_nav_miss
    row["chance_outcomes_added"] = chance_outcomes_added
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="世界不匹配污染量化探针")
    parser.add_argument("--mode", default="random,off,on", help="逗号分隔: random,off,on,ris")
    parser.add_argument("--games", type=int, default=3, help="每个 mode 的局数")
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--chance-k", type=int, default=0, help="on 模式的初始预建数量，遗漏结果按需补建")
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--model", default=os.path.join("data", "models", "v3_iter560.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=os.path.join("data", "eval", "probe_world_mismatch.jsonl"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.chance_k < 0:
        raise SystemExit("--chance-k 必须不小于 0")
    modes = [part.strip() for part in args.mode.split(",") if part.strip()]
    if not modes or any(mode not in {"random", "off", "on", "ris"} for mode in modes):
        raise SystemExit("--mode 必须是 random,off,on,ris 中的一个或多个")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = azero.PolicyValueNet()
    if not azero.load_model_if_exists(net, args.model, device):
        raise SystemExit(f"模型加载失败: {args.model}")
    net.to(device).eval()

    rows = []
    print(f"device={device} sims={args.sims} games/mode={args.games} chance_k={args.chance_k}")
    for mode in modes:
        for game_idx in range(args.games):
            row = play_one(mode, args.sims, args.max_plies, game_idx, args.seed, net, device, args.chance_k)
            rows.append(row)
            print(
                f"[{mode:6s}] game={game_idx} plies={row['plies']:3d} {row['seconds']:6.1f}s "
                f"search apply={row['search_applies']:8d} self_check={row['search_self_check']:6d} "
                f"(dark={row['search_self_check_dark']:5d} bright={row['search_self_check_bright']:5d}) "
                f"rate={row['search_self_check'] / row['search_applies'] * 100 if row['search_applies'] else 0:.3f}% "
                f"| game apply={row['game_applies']:4d} self_check={row['game_self_check']}"
            )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"结果已写入: {args.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
