# -*- coding: utf-8 -*-
"""分析最新回放从第 59 手起红方连续将军与"重复局面判和"是否成立。

用法:
  python analysis_longcheck.py [replay路径] [起始手数]
  python analysis_longcheck.py trace [replay路径] [起始手数]
  python analysis_longcheck.py eval [手数] [搜索次数] [clear]
  python analysis_longcheck.py eval --replay path/to/replay.json --model path/to/model.pt
"""
import argparse
import json
from pathlib import Path
import sys

import game_record as gr
import jieqi as jq

ROOT = Path(__file__).resolve().parent
REPLAY = str(ROOT / "data" / "replays" / "jieqi_20261006_102638.json")
MODEL = str(ROOT / "data" / "train_fixed_20261001" / "models" / "latest.pt")


def main(path=REPLAY, start=59):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    game = gr.GameRecord.from_dict(data, migrate_rules=True)
    n = max(game.nodes)
    print("对局:", Path(path).name)
    print("mode:", game.metadata.get("mode"), "红:", game.metadata.get("red_agent"),
          "黑:", game.metadata.get("black_agent"))
    print("记录终局:", game.metadata.get("end"))

    print("\n== 逐手规则状态（红=奇数手，黑=偶数手）==")
    head = (f"{'手':>3} {'方':<2} {'着法':<9} {'将军':<4} "
            f"{'连将红':>4} {'连将黑':>4} {'红锚':>4} {'黑锚':>4} {'捉':>3} "
            f"{'局面重复':>4} {'无吃子':>4}  裁判")
    print(head)
    for i in range(1, n + 1):
        node = game.nodes[i]
        st = node.state
        side = "红" if i % 2 == 1 else "黑"
        gave_check = st.is_in_check(st.side)
        outcome, reason = st.game_outcome()
        ban = st._banned_for_side()
        rep = st.position_counts.get(st.position_key, 0)
        mark = ""
        if i >= start:
            mark = " <"
        print(f"{i:>3} {side:<2} {node.record['uci']:<9} {'将' if gave_check else '-':<4} "
              f"{st.consecutive_checks[jq.RED]:>4} {st.consecutive_checks[jq.BLACK]:>4} "
              f"{st.check_anchor[jq.RED]:>4} {st.check_anchor[jq.BLACK]:>4} "
              f"{max(st.chase_streak):>3} {rep:>4} {st.no_capture_plies:>4}  "
              f"{outcome},{reason}{mark}")

    print("\n== 终局局面（第 %d 手后）==" % n)
    last = game.nodes[n].state
    print(jq.board_to_text(last, show_dark=True))
    print("行动方:", "红" if last.side == jq.RED else "黑")
    print("黑将被将军:", last.is_in_check(jq.BLACK), " 红帅被将军:", last.is_in_check(jq.RED))
    print("连将计数 红:", last.consecutive_checks[jq.RED], " 黑:", last.consecutive_checks[jq.BLACK])
    print("长将禁着门槛 long_check_rounds:", last.rules.long_check_rounds)
    print("红方当前是否已被禁将:", last._banned_for_side())
    print("重复局面计数:", last.position_counts.get(last.position_key, 0),
          " repetition_limit:", last.rules.repetition_limit,
          " repetition_is_loss:", last.rules.repetition_is_loss)

    print("\n== 重复局面的出现次数统计（末段）==")
    seen = {}
    for i in range(start - 1, n + 1):
        st = game.nodes[i].state
        key = st.position_key
        seen.setdefault(key, []).append(i)
    for key, ids in seen.items():
        if len(ids) > 1:
            side = "红" if key[0] == jq.RED else "黑"
            print(f"  出现 {len(ids)} 次 @手 {ids}  (轮到{side}行棋)")

    print("\n== 从第 %d 手起逐个局面重判（模拟 GUI 每步调用 game_outcome）==" % start)
    for i in range(start, n + 1):
        st = game.nodes[i].state
        outcome, reason = st.game_outcome()
        if outcome is not None:
            print(f"  第 {i} 手后即已判定: outcome={outcome} reason={reason}")
            break
    else:
        st = game.nodes[n].state
        outcome, reason = st.game_outcome()
        print(f"  第 {n} 手后: outcome={outcome} reason={reason}")


def eval_report(ply=58, sims=400, clear_repeats=False, *, path=REPLAY, model=MODEL):
    """用 AlphaZero 模型评估"红方开始来回将"之前的局面价值。"""
    import random
    import torch
    import azero

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    game = gr.GameRecord.from_dict(data, migrate_rules=True)
    game.cursor = ply
    state = game.nodes[ply].state.clone()
    if clear_repeats:
        state.position_counts = {}
    belief = game.nodes[ply].belief.clone()
    history = [(game.nodes[i].record["from"], game.nodes[i].record["to"]) for i in range(1, ply + 1)]
    print(f"评估第 {ply} 手后局面（行动方 {'红' if state.side == jq.RED else '黑'}）"
          f"{'，已清空重复计数' if clear_repeats else ''}")
    print(jq.board_to_text(state, show_dark=True))
    net = azero.PolicyValueNet()
    azero.load_model_if_exists(net, model, torch.device("cpu"))
    net.eval()
    cfg = azero.AZConfig(mcts_simulations=sims, chance_nodes=True, chance_k=0)
    mcts = azero.NeuralMCTS(net, cfg, torch.device("cpu"), random.Random(7))
    pi, move, root = mcts.search(state, belief, history, temperature=1e-6,
                                 simulations=sims, add_dirichlet=False, return_root=True)
    actor = "红" if state.side == jq.RED else "黑"
    value = root.value_sum / root.visits if root.visits else 0.0
    print(f"根价值（{actor}方视角，+1 为该方必胜）: {value:.4f}")
    rows = sorted(((c.visits, c.q_value(), m) for m, c in root.children.items()), reverse=True)[:8]
    for visits, q, m in rows:
        print(f"  {jq.format_move_uci(m, jq.piece_type(state.board[m[0]])):<9} 访问{visits:>4} Q(对方视角){q:>8.4f}")


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    trace = modes.add_parser("trace", help="逐手检查棋谱的规则状态")
    trace.add_argument("path", nargs="?", default=REPLAY, help="棋谱路径")
    trace.add_argument("start", nargs="?", type=int, default=59, help="起始手数")
    evaluate = modes.add_parser("eval", help="用模型评估指定手数后的局面")
    evaluate.add_argument("ply", nargs="?", type=int, default=58)
    evaluate.add_argument("sims", nargs="?", type=int, default=400)
    evaluate.add_argument("clear", nargs="?", choices=["clear"])
    evaluate.add_argument("--replay", default=REPLAY, help="棋谱路径（默认使用项目内历史棋谱）")
    evaluate.add_argument("--model", default=MODEL, help="模型权重路径")

    arguments = list(sys.argv[1:] if argv is None else argv)
    # 保留文件头中不带模式的调用方式；显式模式与参数解析共用一个入口。
    if not arguments or arguments[0] not in ("trace", "eval", "-h", "--help"):
        arguments.insert(0, "trace")
    options = parser.parse_args(arguments)
    if options.mode == "trace":
        main(options.path, options.start)
    else:
        eval_report(options.ply, options.sims, options.clear == "clear",
                    path=options.replay, model=options.model)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cli()
