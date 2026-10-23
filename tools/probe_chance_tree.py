"""探针：揭示结果对动作空间的影响 + off/on 搜索下的树结构（Issue 0007）。

三段输出：
[1] 开局红中兵前进后，翻出各类型时黑方合法着法数（翻炮=将军，legal 骤减）。
[2] off 模式：实际采样到的公开翻子结果各自保留 children（炮 6，其余 44）。
[3] on 模式：chance 节点的揭示分支结构（各分支独立 children）。

用法：
    python tools/probe_chance_tree.py --seeds 12 --sims 400
"""

import argparse
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

import azero
import ismcts as im
import jieqi as jq

MOVE = (jq.index(6, 4), jq.index(5, 4))  # 红中兵 (row=6,col=4) -> (row=5,col=4)
TYPES = [
    (jq.TYPE_ROOK, "车"),
    (jq.TYPE_KNIGHT, "马"),
    (jq.TYPE_ELEPHANT, "象"),
    (jq.TYPE_ADVISOR, "士"),
    (jq.TYPE_CANNON, "炮"),
    (jq.TYPE_PAWN, "兵"),
]
NAME = {t: n for t, n in TYPES}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="揭示动作空间与树结构探针")
    parser.add_argument("--seeds", type=int, default=12, help="off 模式重复的搜索 RNG 数")
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1000, help="off 模式 seed 起点")
    parser.add_argument("--model", default=os.path.join("data", "models", "v3_iter560.pt"))
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("== [1] 红中兵翻出各类型后，黑方合法着法数（开局局面）==")
    for t, name in TYPES:
        rng = random.Random(7)
        state = jq.GameState.initial(rng=rng)
        assert MOVE in state.legal_moves(), "红中兵着法不在合法列表中"
        state.apply_move(MOVE, rng=rng, reveal_choice=t)
        assert state.side == jq.BLACK
        print(
            f"  翻出{name}: 黑方 legal={len(state.legal_moves()):3d}  "
            f"被将军={state.is_in_check(state.side)}"
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = azero.PolicyValueNet()
    if not azero.load_model_if_exists(net, args.model, device):
        raise SystemExit(f"模型加载失败: {args.model}")
    net.to(device).eval()

    print(f"\n== [2] off 模式：按采样公开揭示惰性建树（{args.seeds} 个 seed）==")
    for seed in range(args.seeds):
        rng = random.Random(args.seed + seed)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        cfg = azero.AZConfig(
            mcts_simulations=args.sims, chance_nodes=False, chance_k=0, leaf_batch_size=64
        )
        mcts = azero.NeuralMCTS(net, cfg, device, rng)
        _, top1, root = mcts.search(
            state, belief, [], temperature=0.0, simulations=args.sims,
            add_dirichlet=False, return_root=True,
        )
        node = root.children.get(MOVE)
        branches = {
            NAME[reveal]: {"visits": child.visits, "information_sets": len(child.decision_nodes),
                           "children": [len(decision.children) for decision in child.decision_nodes.values()]}
            for reveal, child in node.public_outcomes.items()
        } if node is not None else {}
        print(
            f"  seed={seed:2d} top1={top1} 该节点 visits={node.visits if node else 0:3d} "
            f"公开结果={branches}"
        )

    print("\n== [3] on 模式：同一节点的揭示分支结构（各分支的 children 数）==")
    rng = random.Random(42)
    state = jq.GameState.initial(rng=rng)
    belief = im.init_belief_for_game(state)
    cfg = azero.AZConfig(
        mcts_simulations=args.sims, chance_nodes=True, chance_k=0, leaf_batch_size=64
    )
    mcts = azero.NeuralMCTS(net, cfg, device, rng)
    _, _top1, root = mcts.search(
        state, belief, [], temperature=0.0, simulations=args.sims,
        add_dirichlet=False, return_root=True,
    )
    chance = root.children.get(MOVE)
    print(f"  chance 节点 children（揭示分支）数 = {len(chance.children)}")
    for key, child in chance.children.items():
        label = NAME.get(child.reveal_moved, "?") + (
            f"+被吃{NAME.get(child.reveal_captured, '?')}"
            if getattr(child, "reveal_captured", None)
            else ""
        )
        print(
            f"    分支[{key}] reveal={label} visits={child.visits} "
            f"information_sets={len(child.decision_nodes)} "
            f"children={[len(decision.children) for decision in child.decision_nodes.values()]}"
        )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
