"""Compare encoding-only optimizations with a Git baseline, without saving weights."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
from dataclasses import asdict
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import azero as az
import run_records as rr

BASELINE = NET = CONFIG = None
DEVICE = torch.device("cuda")


def emit(label, value):
    print(label + " " + json.dumps(value), flush=True)


def load_baseline(source):
    module = types.ModuleType("encoding_baseline")
    module.__file__ = str(ROOT / "azero.py")
    sys.modules[module.__name__] = module
    exec(compile(source, "<baseline azero.py>", "exec"), module.__dict__)
    return module


def synchronize():
    torch.cuda.synchronize()


def original_batch(module, samples):
    triples = [module.compact_to_tensors(s) for s in samples]
    return tuple(torch.stack(items) for items in zip(*triples))


def assert_tensors_identical(first, second, label):
    assert first.shape == second.shape and first.dtype == second.dtype, label
    assert first.stride() == second.stride(), label
    if first.dtype == torch.float32:
        first, second = first.view(torch.int32), second.view(torch.int32)
    assert torch.equal(first, second), label


def assert_nested_identical(first, second, label):
    if isinstance(first, torch.Tensor):
        assert_tensors_identical(first, second, label)
    elif isinstance(first, dict):
        assert first.keys() == second.keys(), label
        for key in first:
            assert_nested_identical(first[key], second[key], f"{label}.{key}")
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second), label
        for i, (a, b) in enumerate(zip(first, second)):
            assert_nested_identical(a, b, f"{label}.{i}")
    else:
        assert first == second, label


def make_training(checkpoint):
    net = az.PolicyValueNet().to(DEVICE).train()
    net.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    return net, optimizer


def training_checks_and_timing(baseline, checkpoint, batches, pairs):
    before_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    before_samples = hashlib.sha256(pickle.dumps(batches)).hexdigest()
    first, opt_first = make_training(checkpoint)
    second, opt_second = make_training(checkpoint)
    losses = []
    # Fixed different batches, actual full network, loaded AdamW and gradient clipping.
    for i, batch in enumerate(batches):
        expected = baseline.train_step(first, opt_first, batch, DEVICE, 1.0)
        actual = az.train_step(second, opt_second, batch, DEVICE, 1.0)
        assert actual == expected, f"loss at step {i}"
        assert_nested_identical(first.state_dict(), second.state_dict(), "network")
        for j, (a, b) in enumerate(zip(first.parameters(), second.parameters())):
            if a.grad is None:
                assert b.grad is None
            else:
                assert_tensors_identical(a.grad, b.grad, f"gradient {j}")
        assert_nested_identical(opt_first.state_dict(), opt_second.state_dict(), "optimizer")
        losses.append(actual)
    assert hashlib.sha256(pickle.dumps(batches)).hexdigest() == before_samples
    emit("cuda_exact", dict(steps=len(batches), batch=len(batches[0]), losses=losses,
                            inputs_gradients_weights_buffers_optimizer_identical=True))
    del first, second, opt_first, opt_second
    gc.collect()
    torch.cuda.empty_cache()

    trials = []
    for pair in range(pairs):
        for mode in (("baseline", "optimized") if pair % 2 == 0 else ("optimized", "baseline")):
            net, optimizer = make_training(checkpoint)
            step = baseline.train_step if mode == "baseline" else az.train_step
            for i in range(3):
                step(net, optimizer, batches[i % len(batches)], DEVICE, 1.0)
            synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for i in range(12):
                step(net, optimizer, batches[i % len(batches)], DEVICE, 1.0)
            synchronize()
            trials.append(dict(pair=pair, mode=mode, milliseconds_per_step=(time.perf_counter() - start) * 1000 / 12,
                               peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            emit("training_timing", trials[-1])
            del net, optimizer
            gc.collect()
            torch.cuda.empty_cache()
    # Model initialization intentionally consumes RNG; the production encoder must not.
    torch.set_rng_state(before_rng[0])
    torch.cuda.set_rng_state(before_rng[1])
    return dict(exact_steps=len(batches), exact_losses=losses, trials=trials)


def init_worker(source, weights, config):
    global BASELINE, NET, CONFIG
    torch.set_num_threads(1)
    BASELINE = load_baseline(source)
    CONFIG = az.AZConfig(**config)
    NET = az.PolicyValueNet().to(DEVICE).eval()
    NET.load_state_dict(weights)


def search_task(task):
    mode, index, fixture, sims = task
    module = BASELINE if mode == "baseline" else az
    state, belief, history = fixture
    search = module.NeuralMCTS(NET, CONFIG, DEVICE, random.Random(20261005 + index))
    synchronize()
    start = time.perf_counter()
    policy, move = search.search(state, belief, history, 1.0, simulations=sims,
                                 add_dirichlet=True, dirichlet_epsilon=0.0 if sims == 3200 else 0.25,
                                 prune_policy=True)
    synchronize()
    return dict(index=index, seconds=time.perf_counter() - start, move=move,
                policy_digest=hashlib.sha256(np.asarray(policy, dtype=np.float64).tobytes()).hexdigest(),
                rng_digest=hashlib.sha256(pickle.dumps(search.rng.getstate())).hexdigest(),
                total=search.leaf_evals_total, unique=search.leaf_evals_unique,
                chance_applied=search.chance_applied, chance_outcomes_added=search.chance_outcomes_added,
                chance_nav_miss=search.chance_nav_miss)


def collect_fixtures(weights, config):
    net = az.PolicyValueNet().to(DEVICE).eval()
    net.load_state_dict(weights)
    original = az.NeuralMCTS.search
    captured = []

    def capture(self, state, belief, history, *args, **kwargs):
        captured.append((state.clone(), belief.clone(), history.copy()))
        return original(self, state, belief, history, *args, **kwargs)

    az.NeuralMCTS.search = capture
    selected = []
    try:
        config = copy.copy(config)
        config.mcts_simulations = 64
        config.self_play_workers = 1
        for seed in (20261005, 20261006):
            captured.clear()
            game = az.self_play_game(net, config, DEVICE, random.Random(seed))
            assert game.outcome is not None
            indices = np.linspace(0, len(captured) - 1, 8).astype(int).tolist()
            selected.extend(captured[i] for i in indices)
            emit("fixture_game", dict(seed=seed, moves=game.move_count, selected_plies=indices))
    finally:
        az.NeuralMCTS.search = original
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return selected


def search_checks_and_timing(source, weights, config, fixtures, workers, pairs, blocks):
    config = copy.copy(config)
    config.mcts_simulations = 800
    config.self_play_workers = workers
    trials = []
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=init_worker, initargs=(source, weights, asdict(config))) as pool:
        for mode in ("baseline", "optimized"):
            list(pool.map(search_task, [(mode, i, f, 800) for i, f in enumerate(fixtures)]))
        for pair in range(pairs):
            outputs = {}
            for mode in (("baseline", "optimized") if pair % 2 == 0 else ("optimized", "baseline")):
                tasks = [(mode, block * len(fixtures) + i, fixture, 3200 if i in (2, 13) else 800)
                         for block in range(blocks) for i, fixture in enumerate(fixtures)]
                start = time.perf_counter()
                rows = list(pool.map(search_task, tasks))
                trials.append(dict(pair=pair, mode=mode, wall_seconds=time.perf_counter() - start,
                                   worker_seconds=sum(r["seconds"] for r in rows), searches=len(rows)))
                emit("search_timing", trials[-1])
                outputs[mode] = rows
            for a, b in zip(outputs["baseline"], outputs["optimized"]):
                assert {k: v for k, v in a.items() if k != "seconds"} == {
                    k: v for k, v in b.items() if k != "seconds"}, f"search differs at {a['index']}"
            emit("search_exact", dict(pair=pair, searches=len(tasks), identical=True))
    return trials


def mean_timing(trials, key):
    return {mode: statistics.mean(r[key] for r in trials if r["mode"] == mode)
            for mode in ("baseline", "optimized")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD",
                        help="git revision containing the baseline azero.py (default: HEAD)")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="training run directory containing models/, selfplay/ and logs/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("use a new output file in an existing directory")
    if min(args.pairs, args.blocks, args.workers) < 1:
        parser.error("positive pairs/blocks/workers required")
    torch.set_num_threads(1)
    source = subprocess.check_output(["git", "show", f"{args.baseline_ref}:azero.py"], cwd=ROOT).decode("utf-8-sig")
    baseline = load_baseline(source)
    paths = [args.run_dir / "models/training_checkpoint.pt", args.run_dir / "models/latest.pt",
             args.run_dir / "models/train_state.json", args.run_dir / "logs/train.log",
             ROOT / "azero.py", ROOT / "jieqi.py", ROOT / "ismcts.py"]
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir / "selfplay" / checkpoint["replay_files"][-1]["name"]
    samples, _ = az.load_compact_payload(str(replay))
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    # Include all four deterministic augmentations used in production training.
    expanded = []
    for sample in random.Random(20261005).sample(samples, 1024):
        expanded.extend((sample, az.mirror_sample(sample), az.flip_vertical_sample(sample),
                         az.flip_vertical_sample(az.mirror_sample(sample))))
    rng = random.Random(20261005)
    batches = [rng.sample(expanded, 1024) for _ in range(3)]
    for batch in batches:
        for a, b in zip(original_batch(baseline, batch), az.compact_batch_to_tensors(batch)):
            assert_tensors_identical(a, b, "training input")
    cpu_trials = []
    for pair in range(args.pairs):
        for mode in (("baseline", "optimized") if pair % 2 == 0 else ("optimized", "baseline")):
            build = (lambda b: original_batch(baseline, b)) if mode == "baseline" else az.compact_batch_to_tensors
            build(batches[0])
            start = time.perf_counter()
            for i in range(12):
                build(batches[i % len(batches)])
            cpu_trials.append(dict(pair=pair, mode=mode, milliseconds_per_batch=(time.perf_counter()-start)*1000/12))
    emit("encoding_timing", mean_timing(cpu_trials, "milliseconds_per_batch"))
    training = training_checks_and_timing(baseline, checkpoint, batches, args.pairs)
    config = az.AZConfig(**checkpoint["config"])
    fixtures = collect_fixtures(checkpoint["model"], config)
    for state, belief, history in fixtures:
        info = belief.info_state(state, state.side)
        assert baseline.encode_state_numpy(info, history).tobytes() == az.encode_state_numpy(info, history).tobytes()
    search = search_checks_and_timing(source, checkpoint["model"], config, fixtures,
                                      args.workers, args.pairs, args.blocks)
    unchanged = all(rr.file_record(path)["sha256"] == digest for path, digest in hashes.items())
    assert unchanged, "production file changed during probe"
    result = dict(baseline_ref=args.baseline_ref, checkpoint_iteration=checkpoint["train_state"]["iteration"],
                  source_hashes=hashes, production_unchanged=unchanged, args={k: str(v) for k, v in vars(args).items()},
                  torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  encoding_trials=cpu_trials, encoding_mean_ms=mean_timing(cpu_trials, "milliseconds_per_batch"),
                  training=training, training_mean_ms=mean_timing(training["trials"], "milliseconds_per_step"),
                  search_trials=search, search_mean_seconds=mean_timing(search, "wall_seconds"),
                  exact_searches=args.pairs * args.blocks * len(fixtures),
                  limitations=["Fixed roots from two 64-simulation games, not full production self-play",
                               "No full-round or long-term learning-speed claim"])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    emit("complete", {k: result[k] for k in ("production_unchanged", "encoding_mean_ms", "training_mean_ms",
                                            "search_mean_seconds", "exact_searches")})


if __name__ == "__main__":
    main()
