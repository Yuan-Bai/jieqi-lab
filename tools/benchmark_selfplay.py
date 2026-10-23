"""Read-only paired benchmark against a saved pre-change self-play source snapshot."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import pickle
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import azero as az
import run_records as rr
from tools import benchmark_encoding as bench


def leaf_check_and_timing(baseline, weights, config, fixtures, pairs):
    net = az.PolicyValueNet().to(bench.DEVICE).eval()
    net.load_state_dict(weights)
    infos = [belief.info_state(state, state.side) for state, belief, _ in fixtures] * 4
    histories = [history for _, _, history in fixtures] * 4
    entries = [az._repetition_entries(info) for info in infos]
    first = baseline.NeuralMCTS(net, config, bench.DEVICE)
    second = az.NeuralMCTS(net, config, bench.DEVICE)
    old_policy, old_values = first._evaluate_batch(infos, histories)
    policy, values = second._evaluate_batch(infos, histories, repetition_entries=entries)
    assert policy.tobytes() == np.asarray(old_policy, dtype=np.float32).tobytes()
    assert values == old_values and all(type(value) is float for value in values)
    assert not policy.flags.writeable
    trials = []
    for pair in range(pairs):
        for mode in (("baseline", "optimized") if pair % 2 == 0 else ("optimized", "baseline")):
            evaluate = first._evaluate_batch if mode == "baseline" else second._evaluate_batch
            kwargs = {} if mode == "baseline" else dict(repetition_entries=entries)
            for _ in range(5):
                evaluate(infos, histories, **kwargs)
            bench.synchronize()
            start = time.perf_counter()
            for _ in range(40):
                evaluate(infos, histories, **kwargs)
            bench.synchronize()
            trials.append(dict(pair=pair, mode=mode, milliseconds_per_batch=(time.perf_counter()-start)*1000/40))
    bench.emit("leaf_timing", bench.mean_timing(trials, "milliseconds_per_batch"))
    del net, first, second
    gc.collect()
    torch.cuda.empty_cache()
    return trials


def complete_games_check(baseline, weights, config, count):
    net = az.PolicyValueNet().to(bench.DEVICE).eval()
    net.load_state_dict(weights)
    rows = []
    for index in range(count):
        strong = index % 2 == 1
        pair = []
        for mode in (("baseline", "optimized") if index % 2 == 0 else ("optimized", "baseline")):
            module = baseline if mode == "baseline" else az
            rng = random.Random(20261010 + index)
            bench.synchronize()
            start = time.perf_counter()
            game = module.self_play_game(net, config, bench.DEVICE, rng, strong=strong)
            bench.synchronize()
            seconds = time.perf_counter() - start
            assert game.outcome is not None
            record = dict(index=index, mode=mode, strong=strong, seconds=seconds,
                          outcome=game.outcome, moves=game.move_count, samples=len(game.samples),
                          total=game.leaf_evals_total, unique=game.leaf_evals_unique,
                          sample_digest=hashlib.sha256(pickle.dumps([asdict(s) for s in game.samples])).hexdigest(),
                          rng_digest=hashlib.sha256(pickle.dumps(rng.getstate())).hexdigest())
            rows.append(record)
            pair.append(record)
            bench.emit("complete_game", record)
        assert {k: v for k, v in pair[0].items() if k not in ("seconds", "mode")} == {
            k: v for k, v in pair[1].items() if k not in ("seconds", "mode")}, f"complete game differs at {index}"
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--complete-games", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("use a new output file in an existing directory")
    if min(args.pairs, args.blocks, args.workers) < 1 or args.complete_games < 0:
        parser.error("positive pairs/blocks/workers and nonnegative complete-games required")
    torch.set_num_threads(1)
    source = args.baseline_source.read_text(encoding="utf-8-sig")
    baseline = bench.load_baseline(source)
    paths = [args.run_dir / "models/training_checkpoint.pt", args.run_dir / "models/latest.pt",
             args.run_dir / "models/train_state.json", args.run_dir / "logs/train.log", args.baseline_source,
             ROOT / "azero.py", ROOT / "jieqi.py", ROOT / "ismcts.py"]
    hashes = {str(path): rr.file_record(path)["sha256"] for path in paths}
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir / "selfplay" / checkpoint["replay_files"][-1]["name"]
    samples, _ = az.load_compact_payload(str(replay))
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    expanded = []
    for sample in random.Random(20261005).sample(samples, 1024):
        expanded.extend((sample, az.mirror_sample(sample), az.flip_vertical_sample(sample),
                         az.flip_vertical_sample(az.mirror_sample(sample))))
    rng = random.Random(20261005)
    batches = [rng.sample(expanded, 1024) for _ in range(3)]
    training = bench.training_checks_and_timing(baseline, checkpoint, batches, pairs=0)
    config = az.AZConfig(**checkpoint["config"])
    fixtures = bench.collect_fixtures(checkpoint["model"], config)
    leaves = leaf_check_and_timing(baseline, checkpoint["model"], config, fixtures, args.pairs)
    search = bench.search_checks_and_timing(source, checkpoint["model"], config, fixtures,
                                            args.workers, args.pairs, args.blocks)
    games = complete_games_check(baseline, checkpoint["model"], config, args.complete_games)
    unchanged = all(rr.file_record(path)["sha256"] == digest for path, digest in hashes.items())
    assert unchanged, "production/source file changed during probe"
    savings = []
    for pair in range(args.pairs):
        timings = {r["mode"]: r["wall_seconds"] for r in search if r["pair"] == pair}
        savings.append(1 - timings["optimized"] / timings["baseline"])
    result = dict(source_hashes=hashes, production_unchanged=unchanged,
                  checkpoint_iteration=checkpoint["train_state"]["iteration"],
                  args={k: str(v) for k, v in vars(args).items()},
                  torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  training=training, leaf_trials=leaves, leaf_mean_ms=bench.mean_timing(leaves, "milliseconds_per_batch"),
                  search_trials=search, search_mean_seconds=bench.mean_timing(search, "wall_seconds"),
                  paired_savings=savings, mean_paired_saving=statistics.mean(savings),
                  exact_searches=args.pairs * args.blocks * len(fixtures), complete_games=games,
                  limitations=["Fixed roots from two 64-simulation games, not full production self-play",
                               "Leaf microbenchmark assumes snapshots already prepared for dedup signatures",
                               "Complete games are single-worker exactness checks, not 8-worker throughput",
                               "No full-round or long-term learning-speed claim"])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    bench.emit("complete", {key: result[key] for key in ("production_unchanged", "leaf_mean_ms",
                                                        "search_mean_seconds", "exact_searches")})


if __name__ == "__main__":
    main()
