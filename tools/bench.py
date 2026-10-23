"""自博弈吞吐基准：测 evals/s、局/小时与单 worker 效率。

用法示例::

    python tools/bench.py --games 8 --sims 300 --workers 1 --batch 1
    python tools/bench.py --games 16 --sims 300 --workers 8 --batch 64
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

import azero


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="自博弈吞吐基准")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64, help="叶评估批大小，1 为逐模拟评估")
    parser.add_argument("--model", default="", help="可选模型路径")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    net = azero.PolicyValueNet().to(device).eval()
    if args.model and azero.load_model_if_exists(net, args.model, device):
        print(f"已加载模型: {args.model}")
    config = azero.AZConfig(
        self_play_workers=args.workers,
        mcts_simulations=args.sims,
        leaf_batch_size=max(1, args.batch),
        eval_games=0,
    )
    rng = random.Random(args.seed)
    start = time.time()
    results = azero.generate_selfplay_results(net, config, device, rng, args.games)
    elapsed = time.time() - start
    moves = sum(result.move_count for result in results)
    evaluations = moves * args.sims
    avg_moves = moves / len(results) if results else 0.0
    print(
        f"workers={args.workers} batch={args.batch} games={len(results)} sims={args.sims} "
        f"avg_moves={avg_moves:.0f} wall={elapsed:.1f}s"
    )
    print(
        f"evals/s={evaluations / elapsed:.0f} games/hour={3600 * len(results) / elapsed:.0f} "
        f"evals/s/worker={evaluations / elapsed / max(1, args.workers):.0f}"
    )


if __name__ == "__main__":
    main()
