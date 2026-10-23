"""Short, read-only AZ timing probe for Web server Windows scheduling policies.

Uses the same position, weights, seeds, simulations, leaf batch and worker entry
point for every sample. No training, weight writes, or global OS settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jieqi as jq
import ismcts as im
from gui_tasks import LatestTaskRunner, SearchBelief
from web_runtime import configure_runtime, prepare_search_thread, restore_runtime, runtime_status
from web_server import model_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--leaf-batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", help="相对于项目根目录的已发现模型路径")
    args = parser.parse_args()
    models = model_catalog()
    model = args.model or next(iter(models), None)
    if model not in models:
        parser.error("找不到指定模型或本机没有可用权重")
    if not (1 <= args.simulations <= 100000 and 1 <= args.leaf_batch_size <= 4096 and 1 <= args.repeats <= 20):
        parser.error("模拟次数、叶批或重复次数超出范围")
    from search_engine import AzEngine

    engine = AzEngine(str(models[model]), simulations=args.simulations)
    engine.config.leaf_batch_size = min(args.leaf_batch_size, args.simulations)
    state = jq.GameState.initial()
    belief = im.init_belief_for_game(state)
    phases = []
    try:
        # ABBA reduces warm-up/clock drift bias. The helper itself never focuses
        # any app, so this is a policy A/B, not a terminal-focus reproduction.
        for mode in ("system", "responsive", "responsive", "system"):
            configure_runtime(mode)
            runner = LatestTaskRunner()
            samples = []
            try:
                for i in range(args.repeats + 1):
                    def search(rng):
                        prepare_search_thread()
                        rng.seed(20261016 + max(0, i - 1))
                        engine.rng = engine.mcts.rng = rng
                        if engine.device.type == "cuda":
                            engine._torch.cuda.synchronize()
                        start = time.perf_counter()
                        advice = engine.analyse(state.clone(), SearchBelief(belief.clone(), rng), [],
                                                simulations=args.simulations)
                        if engine.device.type == "cuda":
                            engine._torch.cuda.synchronize()
                        return {"seconds": time.perf_counter() - start,
                                "move": list(advice.best.move) if advice.best else None}

                    token = runner.submit(search)
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        result = runner.poll(token)
                        if result is not None:
                            break
                        time.sleep(.01)
                    else:
                        raise TimeoutError("单次搜索超过 120 秒")
                    if result.error:
                        raise RuntimeError(result.error)
                    if i:
                        samples.append(result.value)
                phases.append({"mode": mode, "runtime": runtime_status(), "samples": samples,
                               "median_seconds": statistics.median(item["seconds"] for item in samples)})
                print(json.dumps(phases[-1], ensure_ascii=True), flush=True)
            finally:
                runner.close()
                if runner._thread is not None:
                    runner._thread.join(2)
        print(json.dumps({"model": model, "device": str(engine.device),
                          "simulations": args.simulations, "leaf_batch_size": engine.config.leaf_batch_size,
                          "torch_threads": engine._torch.get_num_threads(), "phases": phases,
                          "note": "Policy A/B only; foreground/background terminal focus was not changed."},
                         ensure_ascii=True), flush=True)
    finally:
        restore_runtime()


if __name__ == "__main__":
    main()
