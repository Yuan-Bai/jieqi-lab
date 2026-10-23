# -*- coding: utf-8 -*-
"""机会节点逻辑探针：分支概率、边际一致性、回退污染与修复效果（Issue 0005 证据）。

用法（项目根目录）::

    python tools/probe_chance.py

段落：
[1] 全枚举分支概率 = pool_known 归一化 + 采样频率收敛（30 万次）
[2] 确定化世界期望类型占比 = pool_known 归一化（missing>0，边际正确性）
[3] 修复动机的静态度量：若按边际先验采样，各分支在世界池不可用（回退）的概率
[4] 真实搜索回退计数：修复后全枚举应恒为 0，并报告 chance_applied/chance_nav_miss
"""
import os
import random
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import jieqi as jq
import ismcts as im
import azero

torch.set_num_threads(1)


def find_dark_move(state, info):
    for mv in state.legal_moves():
        if azero.chance_spec_for_move(state, info, mv) is not None:
            return mv
    return None


def main():
    rng = random.Random(20260917)

    # ---------- 探针 1：机会分支概率 = pool_known 归一化（全枚举） ----------
    state = jq.GameState.initial(rng=rng)
    belief = im.init_belief_for_game(state)
    info = belief.info_state(state, state.side)
    dark_move = find_dark_move(state, info)
    assert dark_move is not None, "no dark move"
    spec = azero.chance_spec_for_move(state, info, dark_move)
    counts = spec["moved_counts"]
    total0 = sum(counts)
    expected = {t: counts[t] / total0 for t in jq.POOL_TYPES if counts[t] > 0}

    node = azero.PUCTNode(1.0, state.side)
    priors = torch.full((azero.MOVE_SIZE,), 1.0 / azero.MOVE_SIZE)
    node.expand([dark_move], priors, {dark_move: spec}, 0, rng)
    chance = node.children[dark_move]
    got = {c.reveal_moved: c.prior for c in chance.children.values()}
    print("[1] 全枚举分支概率 vs pool 归一化  (move=%s)" % (dark_move,))
    for t in sorted(expected):
        print(f"    type={t}: expand={got[t]:.6f} pool={expected[t]:.6f} diff={abs(got[t] - expected[t]):.2e}")

    N = 300000
    cnt = Counter()
    for _ in range(N):
        _, child = chance.sample_outcome(rng)
        cnt[child.reveal_moved] += 1
    print("    采样频率（sample_outcome 边际先验，N=%d）：" % N)
    for t in sorted(expected):
        freq = cnt[t] / N
        print(f"    type={t}: emp={freq:.6f} exp={expected[t]:.6f} rel_err={(freq - expected[t]) / expected[t] * 100:+.2f}%")

    # ---------- 探针 2：边际一致性 E_world[count/total] == pool_known 归一化 ----------
    print("[2] 确定化世界期望类型占比 = pool_known 归一化（missing>0）")
    side = state.side
    for missing in (4, 8):
        b2 = belief.clone()
        for obs in (0, 1):
            b2.pool_total[obs][side] -= missing
            b2.unknown_removals[obs][side] += missing
        acc = Counter()
        W = 60000
        for _ in range(W):
            st = b2.info_state(state, side).determinize(rng)
            tot = sum(st.pool[side])
            for t in jq.POOL_TYPES:
                acc[t] += st.pool[side][t] / tot
        print(f"    missing={missing}: " + ", ".join(
            f"t{t}: E={acc[t] / W:.5f}(pool={expected[t]:.5f})" for t in sorted(expected)))

    # ---------- 探针 3：修复动机的静态度量（边际采样时的回退概率） ----------
    print("[3] 静态度量：若按边际先验采样，单次 reveal 命中不可用分支的概率（修复前动机）")
    for missing in (1, 2, 4, 8, 12):
        b = belief.clone()
        for obs in (0, 1):
            b.pool_total[obs][side] -= missing
            b.unknown_removals[obs][side] += missing
        unavail = Counter()
        W = 20000
        for _ in range(W):
            st = b.info_state(state, side).determinize(rng)
            for t in jq.POOL_TYPES:
                if counts[t] > 0 and st.pool[side][t] <= 0:
                    unavail[t] += 1
        pollute = sum(expected[t] * unavail[t] / W for t in expected)
        branch = ", ".join(f"{t}:{unavail[t] / W * 100:.1f}%" for t in sorted(expected))
        print(f"    missing={missing:2d}: 回退概率≈{pollute * 100:5.2f}%  分支不可用率[{branch}]")

    # ---------- 探针 4：真实搜索中的回退计数（修复后应为 0） ----------
    print("[4] 真实搜索回退计数（hook _apply_chance_move 与内置计数器）")
    net = azero.PolicyValueNet(channels=16, blocks=2).eval()
    device = torch.device("cpu")
    orig = azero.NeuralMCTS._apply_chance_move
    stats = {"total": 0, "fallback": 0}

    def hooked(self, sim_state, move, reveal_moved, reveal_captured):
        available = True
        if reveal_moved and sim_state.pool[sim_state.side][reveal_moved] <= 0:
            available = False
        if available and reveal_captured:
            target = sim_state.board[move[1]]
            if target and jq.is_dark(target):
                cs = jq.piece_color(target)
                if sim_state.pool[cs][reveal_captured] <= 0:
                    available = False
        stats["total"] += 1
        if not available:
            stats["fallback"] += 1
        return orig(self, sim_state, move, reveal_moved, reveal_captured)

    azero.NeuralMCTS._apply_chance_move = hooked
    try:
        for missing in (0, 6):
            b3 = belief.clone()
            for obs in (0, 1):
                b3.pool_total[obs][side] -= missing
                b3.unknown_removals[obs][side] += missing
            stats["total"] = 0
            stats["fallback"] = 0
            cfg = azero.AZConfig(mcts_simulations=200, chance_nodes=True, chance_k=0, leaf_batch_size=32)
            mcts = azero.NeuralMCTS(net, cfg, device, rng=random.Random(7))
            mcts.search(state, b3, [], temperature=0.0, simulations=200)
            print(f"    missing={missing}: 机会应用 {mcts.chance_applied} 次，"
                  f"导航失败 {mcts.chance_nav_miss} 次，hook 回退 {stats['fallback']}/{stats['total']} 次"
                  f"（{stats['fallback'] / max(1, stats['total']) * 100:.2f}%）")
    finally:
        azero.NeuralMCTS._apply_chance_move = orig


if __name__ == "__main__":
    main()
