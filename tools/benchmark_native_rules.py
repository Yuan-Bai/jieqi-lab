"""Paired Python/Cython rules benchmark, including exact fixed-seed CUDA searches."""
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
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import azero as az
import jieqi as jq
import run_records as rr
from tools import benchmark_encoding as bench

PYTHON_ATTACK = jq.GameState._is_square_attacked_python


def attack_for(mode):
    if mode == "baseline":
        return PYTHON_ATTACK
    import _jieqi_rules
    return _jieqi_rules.is_square_attacked


def init_worker(source, weights, config):
    assert jq.RULES_BACKEND == "cython", jq.RULES_BACKEND_ERROR
    bench.init_worker(source, weights, config)


def search_task(task):
    jq.GameState.is_square_attacked = attack_for(task[0])
    return bench.search_task(task)


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
                bench.emit("search_timing", trials[-1])
                outputs[mode] = rows
            for a, b in zip(outputs["baseline"], outputs["optimized"]):
                assert {k: v for k, v in a.items() if k != "seconds"} == {
                    k: v for k, v in b.items() if k != "seconds"}, f"search differs at {a['index']}"
            bench.emit("search_exact", dict(pair=pair, searches=len(tasks), identical=True))
    return trials


def rules_checks_and_timing(fixtures, pairs):
    states = [state for state, _, _ in fixtures]
    before = pickle.dumps(states)
    trials = []
    for kind in ("attack", "legal"):
        outputs = []
        for mode in ("baseline", "optimized"):
            jq.GameState.is_square_attacked = attack_for(mode)
            outputs.append([[state.is_square_attacked(side, cell) for side in (0, 1) for cell in range(90)]
                            if kind == "attack" else state.legal_moves() for state in states])
        assert outputs[0] == outputs[1]
        assert pickle.dumps(states) == before
        for pair in range(pairs):
            for mode in (("baseline", "optimized") if pair % 2 == 0 else ("optimized", "baseline")):
                jq.GameState.is_square_attacked = attack_for(mode)
                def run():
                    for state in states:
                        if kind == "attack":
                            for side in (0, 1):
                                for cell in range(90):
                                    state.is_square_attacked(side, cell)
                        else:
                            state.legal_moves()
                run()
                start = time.perf_counter()
                for _ in range(80):
                    run()
                trials.append(dict(kind=kind, pair=pair, mode=mode,
                                   milliseconds_per_group=(time.perf_counter() - start) * 1000 / 80))
        assert pickle.dumps(states) == before
        bench.emit("rules_timing", dict(kind=kind, mean_ms=bench.mean_timing(
            [r for r in trials if r["kind"] == kind], "milliseconds_per_group")))
    jq.GameState.is_square_attacked = attack_for("optimized")
    return trials


def complete_games_check(weights, config, count):
    net = az.PolicyValueNet().to(bench.DEVICE).eval()
    net.load_state_dict(weights)
    rows = []
    for index in range(count):
        pair = []
        for mode in (("baseline", "optimized") if index % 2 == 0 else ("optimized", "baseline")):
            jq.GameState.is_square_attacked = attack_for(mode)
            rng = random.Random(20261010 + index)
            bench.synchronize()
            start = time.perf_counter()
            game = az.self_play_game(net, config, bench.DEVICE, rng, strong=index % 2 == 1)
            bench.synchronize()
            assert game.outcome is not None
            record = dict(index=index, mode=mode, strong=index % 2 == 1,
                          seconds=time.perf_counter() - start, outcome=game.outcome, moves=game.move_count,
                          samples=len(game.samples), total=game.leaf_evals_total, unique=game.leaf_evals_unique,
                          sample_digest=hashlib.sha256(pickle.dumps([asdict(s) for s in game.samples])).hexdigest(),
                          rng_digest=hashlib.sha256(pickle.dumps(rng.getstate())).hexdigest())
            rows.append(record)
            pair.append(record)
            bench.emit("complete_game", record)
        assert {k: v for k, v in pair[0].items() if k not in ("seconds", "mode")} == {
            k: v for k, v in pair[1].items() if k not in ("seconds", "mode")}, f"complete game differs at {index}"
    jq.GameState.is_square_attacked = attack_for("optimized")
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    if jq.RULES_BACKEND != "cython":
        parser.error(f"build current extension first: {jq.RULES_BACKEND_ERROR}")
    import _jieqi_rules
    torch.set_num_threads(1)
    paths = [args.run_dir / "models/training_checkpoint.pt", args.run_dir / "models/latest.pt",
             args.run_dir / "models/train_state.json", args.run_dir / "logs/train.log",
             ROOT / "azero.py", ROOT / "jieqi.py", ROOT / "ismcts.py", ROOT / "run_records.py",
             ROOT / "native/jieqi_rules.pyx", Path(_jieqi_rules.__file__)]
    hashes = {str(path): rr.file_record(path)["sha256"] for path in paths}
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir / "selfplay" / checkpoint["replay_files"][-1]["name"]
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    config = az.AZConfig(**checkpoint["config"])
    jq.GameState.is_square_attacked = PYTHON_ATTACK
    fixtures = bench.collect_fixtures(checkpoint["model"], config)
    jq.GameState.is_square_attacked = attack_for("optimized")
    rules = rules_checks_and_timing(fixtures, args.pairs)
    search = search_checks_and_timing((ROOT / "azero.py").read_text(encoding="utf-8-sig"),
                                     checkpoint["model"], config, fixtures, args.workers, args.pairs, args.blocks)
    games = complete_games_check(checkpoint["model"], config, args.complete_games)
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
                  rules_trials=rules, search_trials=search,
                  search_mean_seconds=bench.mean_timing(search, "wall_seconds"),
                  paired_savings=savings, mean_paired_saving=statistics.mean(savings),
                  exact_searches=args.pairs * args.blocks * len(fixtures), complete_games=games,
                  limitations=["Only attack detection compiled; move filtering and RNG remain Python",
                               "Fixed roots from two 64-simulation games, not full production self-play",
                               "Complete games are serial exactness checks, not 8-worker throughput",
                               "No full-round or long-term learning-speed claim"])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    bench.emit("complete", {key: result[key] for key in ("production_unchanged", "search_mean_seconds",
                                                        "exact_searches")})


if __name__ == "__main__":
    main()
