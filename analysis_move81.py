# -*- coding: utf-8 -*-
"""分析 jieqi_20261003_214948.json 第 81 手红方决策。

用法:
  python analysis_move81.py position            # 复盘 + 验证 d7->c8 是否绝杀
  python analysis_move81.py search [sims]       # 用 latest.pt 复现第 81 手 AlphaZero 搜索
  python analysis_move81.py search 800 --replay path/to/replay.json --model path/to/model.pt
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import game_record as gr
import ismcts as im
import jieqi as jq

ROOT = Path(__file__).resolve().parent
REPLAY = str(ROOT / "data" / "replays" / "jieqi_20261003_214948.json")
MODEL = str(ROOT / "data" / "train_fixed_20261001" / "models" / "latest.pt")

D7C8 = (21, 11)   # 士 d7 -> c8
B2D2 = (64, 66)   # 暗子 b2 -> d2


def uci_for(state, move):
    p = state.board[move[0]]
    t = jq.piece_type(p) if p else 0
    return jq.format_move_uci(move, t)


def load_game(path=REPLAY):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    game = gr.GameRecord.from_dict(data, migrate_rules=True)
    return game


def position_report(path=REPLAY):
    game = load_game(path)
    print("== 对局信息 ==")
    print("nodes:", max(game.nodes), "end:", game.metadata.get("end") or "")
    game.cursor = 80
    node80 = game.current
    state = node80.state
    print("\n== 第 80 手后（红方走第 81 手前）==")
    print("行动方:", "红" if state.side == jq.RED else "黑")
    print("黑将位置:", jq.square_to_name(state.king_pos[jq.BLACK]),
          "红帅位置:", jq.square_to_name(state.king_pos[jq.RED]))
    print()
    print(jq.board_to_text(state, show_dark=True))
    print()
    for name, idx in [("d7", 21), ("c8", 11), ("b2", 64), ("d2", 66),
                      ("d9", 3), ("d8", 12), ("e8", 13), ("e9", 4), ("g9", 6), ("g8", 15)]:
        p = state.board[idx]
        desc = "空"
        if p:
            color = "红" if jq.piece_color(p) == jq.RED else "黑"
            kind = "暗" if jq.is_dark(p) else "明"
            desc = f"{color}{kind}{jq.PIECE_NAMES[jq.piece_color(p)][jq.piece_type(p)]}(code={p})"
        print(f"  {name} (idx={idx}): {desc}")
    print("暗池剩余 红:", state.pool[jq.RED], " 黑:", state.pool[jq.BLACK],
          "pool_total:", state.pool_total)
    b80 = node80.belief
    print("信念 pool_counts[红看红]:", b80.pool_counts[jq.RED][jq.RED],
          " pool_total:", b80.pool_total[jq.RED][jq.RED],
          " unknown_removals:", b80.unknown_removals[jq.RED][jq.RED])
    print("信念 pool_counts[红看黑]:", b80.pool_counts[jq.RED][jq.BLACK])

    print("\n== 红方第 81 手的合法着法（按走子类型）==")
    moves = state.legal_moves()
    print("合法着法总数:", len(moves))
    for m in moves:
        tag = ""
        if m == D7C8:
            tag = "  <<< d7->c8"
        if m == B2D2:
            tag = "  <<< b2->d2"
        print("  ", uci_for(state, m), tag)

    print("\n== 第 80 手后棋盘全部棋子 ==")
    for i, p in enumerate(state.board):
        if p:
            color = "红" if jq.piece_color(p) == jq.RED else "黑"
            kind = "暗" if jq.is_dark(p) else "明"
            print(f"  {jq.square_to_name(i)}({i:2d}) {color}{kind}"
                  f"{jq.PIECE_NAMES[jq.piece_color(p)][jq.piece_type(p)]} code={p}")

    print("\n== 走 d7->c8 是否绝杀（引擎裁判视角）==")
    print("  d7->c8 是否在红方合法着法内:", D7C8 in state.legal_moves())
    s2 = state.clone()
    s2.apply_move(D7C8, rng=random.Random(123))
    print("  d7->c8 后红帅是否被将军(飞将照面):", s2.is_in_check(jq.RED))
    print("  d7->c8 后黑方合法着法数:", len(s2.legal_moves()))
    print("  黑将是否被将军:", s2.is_in_check(jq.BLACK))
    for name, idx in (("d8", 12), ("e8", 13), ("e9", 4)):
        print(f"    {name} 被红方攻击: {s2.is_square_attacked(jq.RED, idx)}")
    for m in s2.legal_moves():
        print("  黑方唯一应着:", uci_for(s2, m))
    outcome, reason = s2.game_outcome()
    print("  game_outcome:", outcome, reason)

    print("\n== 同一测试：第 82 手后（黑 g9g8 之后）走 d7->c8 ==")
    s3 = game.nodes[82].state.clone()
    print("  第 82 手后棋盘差异: b2 处:", s3.board[64], "d2 处:", s3.board[66],
          "g9 处:", s3.board[6], "g8 处:", s3.board[15])
    s3.apply_move(D7C8, rng=random.Random(123))
    print("  d7->c8 后黑方合法着法数:", len(s3.legal_moves()))
    for m in s3.legal_moves():
        print("  黑方应着:", uci_for(s3, m))
    outcome, reason = s3.game_outcome()
    print("  game_outcome:", outcome, reason)
    for name, idx in (("d8", 12), ("e8", 13), ("e9", 4)):
        print(f"    {name} 被红方攻击: {s3.is_square_attacked(jq.RED, idx)}")

    print("\n== 第 81 手实际走了 b2->d2，记录揭示 reveal=7（兵）==")
    node81 = game.nodes[81]
    rec = node81.record
    print("  record:", rec)
    print("  落点 d2 棋子:", node81.state.board[66],
          "->", jq.PIECE_NAMES[jq.RED][jq.piece_type(node81.state.board[66])])
    print("  若 b2 揭出炮: d2 炮沿 d 线隔 d7 士将军 d9；若揭出其他子则无威胁")

    print("\n== 红方信念：b2 暗子是炮的概率 ==")
    b = node80.belief
    own = b.pool_counts[jq.RED][jq.RED]
    total = sum(own)
    print("  红方视角自己暗池:", own, "共", total)
    print(f"  P(b2=炮) = {own[jq.TYPE_CANNON]}/{total} = {own[jq.TYPE_CANNON]/total:.3f}")
    print("  红方视角黑方暗池:", b.pool_counts[jq.RED][jq.BLACK])

    print("\n== 双方最近 10 手 ==")
    for i in range(74, 84):
        n = game.nodes[i]
        print(f"  {i:2d}. {'红' if i % 2 == 1 else '黑'} {n.record['uci']}")


def search_report(sims, path=REPLAY, model=MODEL):
    import torch
    import azero

    game = load_game(path)
    game.cursor = 80
    state = game.nodes[80].state.clone()
    belief = game.nodes[80].belief.clone()
    history = [(game.nodes[i].record["from"], game.nodes[i].record["to"])
               for i in range(1, 81)]

    net = azero.PolicyValueNet()
    ok = azero.load_model_if_exists(net, model, torch.device("cpu"))
    print("模型加载:", ok, model)
    net.eval()
    cfg = azero.AZConfig(mcts_simulations=sims, chance_nodes=True, chance_k=0)
    mcts = azero.NeuralMCTS(net, cfg, torch.device("cpu"), random.Random(20261003))
    t0 = time.time()
    pi, move, root = mcts.search(state, belief, history, temperature=1e-6,
                                 simulations=sims, add_dirichlet=False, return_root=True)
    dt = time.time() - t0
    print(f"\n== AlphaZero 根搜索 (sims={sims}, {dt:.1f}s) ==")
    print("根访问数:", root.visits, "根价值:", round(root.value_sum / root.visits, 4)
          if root.visits else None)
    print("搜索选定着法:", move, uci_for(state, move) if move else None)

    rows = []
    for m, child in root.children.items():
        rows.append((child.visits, child.q_value(), child.prior, m, child))
    rows.sort(key=lambda r: -r[0])
    print(f"\n{'着法':<10} {'访问':>7} {'Q值':>8} {'先验':>8}  备注")
    for visits, q, prior, m, child in rows[:15]:
        note = ""
        if m == D7C8:
            note = " <<< d7->c8 绝杀着"
        if m == B2D2:
            note = " <<< b2->d2 赌炮着"
        if child.is_chance:
            subs = []
            for oc in child.children.values():
                subs.append(f"{jq.PIECE_NAMES[jq.RED][oc.reveal_moved]}:{oc.visits}")
            note += f"  [揭示分支: {', '.join(subs)}]"
        print(f"{uci_for(state, m):<10} {visits:>7} {q:>8.4f} {prior:>8.5f}{note}")
    for visits, q, prior, m, child in rows:
        if m in (D7C8, B2D2) and m not in [r[3] for r in rows[:15]]:
            print(f"{uci_for(state, m):<10} {visits:>7} {q:>8.4f} {prior:>8.5f}  (不在前 15)")

    # d7->c8 若被搜索过，打印其统计
    d7 = root.children.get(D7C8)
    b2 = root.children.get(B2D2)
    print("\n目标着法统计:")
    for label, child in (("d7->c8", d7), ("b2->d2", b2)):
        if child is None:
            print(f"  {label}: 未在根子节点中（不合法或未扩展）")
            continue
        print(f"  {label}: 访问={child.visits} Q={child.q_value():.4f} 先验={child.prior:.6f}")
        if child.is_chance:
            for oc in child.children.values():
                print(f"     揭出 {jq.PIECE_NAMES[jq.RED][oc.reveal_moved]}: "
                      f"访问={oc.visits} Q={oc.q_value():.4f}")


def gamble_report(path=REPLAY):
    game = load_game(path)
    game.cursor = 80
    state = game.nodes[80].state.clone()

    print("== 分支 1：b2->d2 若揭出【炮】(红方视角 P=1/5，反事实推演) ==")
    s = state.clone()
    s.apply_move(B2D2, rng=random.Random(1), reveal_choice=jq.TYPE_PAWN)
    # 反事实：把落点 d2 的棋子直接替换成明炮，等价于"揭出的是炮"
    s.board[66] = jq.make_piece(jq.TYPE_CANNON, jq.RED, 0)
    print("  d2 棋子:", jq.PIECE_NAMES[jq.RED][jq.piece_type(s.board[66])])
    print("  黑将是否被将军:", s.is_in_check(jq.BLACK))
    print("  黑方合法着法数:", len(s.legal_moves()))
    outcome, reason = s.game_outcome()
    print("  game_outcome:", outcome, reason)

    print("\n== 分支 2：b2->d2 揭出【兵】(P=4/5，实际对局) ==")
    s = state.clone()
    s.apply_move(B2D2, rng=random.Random(1), reveal_choice=jq.TYPE_PAWN)
    print("  d2 棋子:", jq.PIECE_NAMES[jq.RED][jq.piece_type(s.board[66])])
    print("  黑将是否被将军:", s.is_in_check(jq.BLACK))
    black_replies = s.legal_moves()
    print("  黑方当前合法着法数:", len(black_replies))
    mate_always = True
    for bm in black_replies:
        t = s.clone()
        t.apply_move(bm, rng=random.Random(2))
        t.apply_move(D7C8, rng=random.Random(3))
        oc, rs = t.game_outcome()
        tag = "绝杀✓" if (oc, rs) == (jq.RESULT_RED_WIN, jq.REASON_CHECKMATE) else f"{oc},{rs}"
        if (oc, rs) != (jq.RESULT_RED_WIN, jq.REASON_CHECKMATE):
            mate_always = False
        print(f"    黑应 {uci_for(state, bm):<8} 后红 d7c8 -> {tag}")
    print("  结论: 无论黑方如何应，红方下一步 d7->c8 均为绝杀:", mate_always)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("position", "gamble", "search"), default="position")
    parser.add_argument("sims", nargs="?", type=int, default=800)
    parser.add_argument("--replay", default=REPLAY, help="棋谱路径（需包含本案例所分析的手数）")
    parser.add_argument("--model", default=MODEL, help="模型权重路径（仅 search 使用）")
    options = parser.parse_args(argv)
    if options.mode == "position":
        position_report(options.replay)
    elif options.mode == "gamble":
        gamble_report(options.replay)
    else:
        search_report(options.sims, options.replay, options.model)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cli()
