# -*- coding: utf-8 -*-
"""机会节点方差副实验：固定局面重复搜索，比较 chance on/off 的根估计稳定性。

目的（Issue 0005 主实验配套）：直接检验 Rao-Blackwell 预测——机会节点把揭示
随机性分层后，根价值估计的跨次方差应不高于无机会节点（同 sims 预算）。

流程：
1) 固定模型自博弈生成若干局，每 N ply 对 (state, belief, history) 深拷贝快照；
2) 对每个局面，chance on/off 各重复 R 次搜索（不同 RNG），记录：
   根 Q、最优着法访问占比、策略熵、top-1 着法稳定性、耗时；
3) 输出逐局面 JSONL + 聚合摘要（on/off 的根 Q 标准差配对差，>0 表示 on 更稳）。

用法（项目根目录）::

    python tools/chance_variance_probe.py --gen-games 2 --snapshot-every 10 \
        --sims 400 --repeats 32 --out data/eval/chance_variance.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import azero
import ismcts as im
import jieqi as jq


def collect_positions(net, device, games, snapshot_every, max_plies, gen_sims, base_seed):
    """固定模型自博弈生成局面快照（温度 1.0 采样保证多样性）。"""
    positions = []
    for game_idx in range(games):
        rng = random.Random(base_seed + game_idx)
        dark_types = jq.materialize_dark_types(random.Random(f"var-probe-shuffle:{base_seed + game_idx}"))
        state = jq.GameState.initial(rng=rng, dark_types=dark_types)
        belief = im.init_belief_for_game(state)
        history: list[tuple[int, int]] = []
        config = azero.AZConfig(mcts_simulations=gen_sims, leaf_batch_size=32)
        mcts = azero.NeuralMCTS(net, config, device, rng)
        ply = 0
        while state.game_outcome()[0] is None and ply < max_plies:
            if ply % snapshot_every == 0:
                positions.append(
                    {
                        "game": game_idx,
                        "ply": ply,
                        "state": copy.deepcopy(state),
                        "belief": belief.clone(),
                        "history": history.copy(),
                    }
                )
            moves = state.legal_moves()
            if not moves:
                break
            _policy, move = mcts.search(
                state, belief, history, temperature=1.0, simulations=gen_sims, add_dirichlet=False
            )
            if move is None:
                break
            im.apply_move_and_update_belief(state, belief, move, rng=rng)
            history.append(move)
            ply += 1
    return positions


def run_repeats(net, device, state, belief, history, sims, repeats, chance, base_seed):
    """同一局面重复搜索 repeats 次（不同 RNG），返回根估计稳定性统计。"""
    qs: list[float] = []
    top1s: list = []
    top_ratios: list[float] = []
    entropies: list[float] = []
    times: list[float] = []
    for rep in range(repeats):
        config = azero.AZConfig(
            mcts_simulations=sims, chance_nodes=chance, chance_k=0, leaf_batch_size=64
        )
        mcts = azero.NeuralMCTS(net, config, device, random.Random(base_seed + rep))
        start = time.perf_counter()
        _policy, _move, root = mcts.search(
            state,
            belief,
            history,
            temperature=0.0,
            simulations=sims,
            add_dirichlet=False,
            return_root=True,
        )
        times.append((time.perf_counter() - start) * 1000.0)
        qs.append(root.value_sum / root.visits if root.visits else 0.0)
        visits = [child.visits for child in root.children.values()]
        total = sum(visits)
        if root.children:
            top1s.append(max(root.children.items(), key=lambda kv: kv[1].visits)[0])
        top_ratios.append(max(visits) / total if total else 0.0)
        if total:
            probs = [v / total for v in visits if v > 0]
            entropies.append(-sum(p * math.log(p) for p in probs))
        else:
            entropies.append(0.0)
    mode_ratio = Counter(top1s).most_common(1)[0][1] / len(top1s) if top1s else 0.0
    return {
        "q_mean": statistics.fmean(qs),
        "q_sd": statistics.pstdev(qs) if len(qs) > 1 else 0.0,
        "top1_mode_ratio": mode_ratio,
        "top1_unique": len(set(top1s)),
        "top_visit_ratio_mean": statistics.fmean(top_ratios),
        "entropy_mean": statistics.fmean(entropies),
        "time_ms_mean": statistics.fmean(times),
    }


def mean_ci95(values: list[float]) -> tuple[float, float]:
    """正态近似 95% CI（n 较小时仅作参考）。"""
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="机会节点方差副实验（固定局面重复搜索）")
    parser.add_argument("--model", default=os.path.join("data", "models", "v3_iter560.pt"))
    parser.add_argument("--gen-games", type=int, default=2, help="生成局面的自博弈局数")
    parser.add_argument("--snapshot-every", type=int, default=10, help="每 N ply 快照一个局面")
    parser.add_argument("--max-plies", type=int, default=120, help="每局快照最大 ply")
    parser.add_argument("--gen-sims", type=int, default=200, help="生成局面时的搜索预算")
    parser.add_argument("--sims", type=int, default=400, help="副实验搜索预算")
    parser.add_argument("--repeats", type=int, default=32, help="每局面每配置重复次数")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=os.path.join("data", "eval", "chance_variance.jsonl"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    net = azero.PolicyValueNet()
    if not azero.load_model_if_exists(net, args.model, device):
        print(f"错误: 未找到模型 {args.model}", file=sys.stderr, flush=True)
        return 2
    net.to(device)
    net.eval()
    torch.set_num_threads(1)

    print(f"生成局面：{args.gen_games} 局 × 每 {args.snapshot_every} ply（gen_sims={args.gen_sims}）", flush=True)
    positions = collect_positions(
        net, device, args.gen_games, args.snapshot_every, args.max_plies, args.gen_sims, args.seed
    )
    print(f"共 {len(positions)} 个局面；每局面 on/off 各 {args.repeats} 次搜索（sims={args.sims}）", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as handle:
        for index, item in enumerate(positions):
            state = item["state"]
            belief = item["belief"]
            dark_count = sum(1 for piece in state.board if piece and jq.is_dark(piece))
            unknown = sum(belief.unknown_removals[0]) + sum(belief.unknown_removals[1])
            base_seed = args.seed + index * 1000
            off_stats = run_repeats(
                net, device, state, belief, item["history"], args.sims, args.repeats, False, base_seed
            )
            on_stats = run_repeats(
                net, device, state, belief, item["history"], args.sims, args.repeats, True, base_seed
            )
            row = {
                "type": "position",
                "game": item["game"],
                "ply": item["ply"],
                "side": "red" if state.side == jq.RED else "black",
                "dark_pieces": dark_count,
                "unknown_removals": unknown,
                "off": off_stats,
                "on": on_stats,
                "q_sd_gain": off_stats["q_sd"] - on_stats["q_sd"],
                "time_ratio": on_stats["time_ms_mean"] / off_stats["time_ms_mean"]
                if off_stats["time_ms_mean"] > 0
                else 0.0,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(positions)}] ply={row['ply']:3d} dark={dark_count:2d} "
                f"q_sd off={off_stats['q_sd']:.4f} on={on_stats['q_sd']:.4f} "
                f"time×{row['time_ratio']:.2f}",
                flush=True,
            )

        q_sd_gains = [row["q_sd_gain"] for row in rows]
        gain_mean, gain_ci = mean_ci95(q_sd_gains)
        time_ratios = [row["time_ratio"] for row in rows]
        time_mean, time_ci = mean_ci95(time_ratios)
        open_rows = [row for row in rows if row["dark_pieces"] >= 8]
        later_rows = [row for row in rows if row["dark_pieces"] < 8]
        summary = {
            "type": "summary",
            "positions": len(rows),
            "sims": args.sims,
            "repeats": args.repeats,
            "q_sd_gain_mean": gain_mean,
            "q_sd_gain_ci95": gain_ci,
            "q_sd_gain_open_mean": statistics.fmean([r["q_sd_gain"] for r in open_rows]) if open_rows else None,
            "q_sd_gain_late_mean": statistics.fmean([r["q_sd_gain"] for r in later_rows]) if later_rows else None,
            "time_ratio_mean": time_mean,
            "time_ratio_ci95": time_ci,
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        handle.flush()

    print("\n===== 摘要 =====")
    print(f"根 Q 跨次标准差配对差（off - on，>0 = on 更稳）：{gain_mean:+.5f} ± {gain_ci:.5f}（95% CI）")
    print(f"  暗子>=8 局面：{summary['q_sd_gain_open_mean']}")
    print(f"  暗子<8  局面：{summary['q_sd_gain_late_mean']}")
    print(f"单次搜索耗时比（on/off）：{time_mean:.3f} ± {time_ci:.3f}")
    print(f"结果已写入: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
