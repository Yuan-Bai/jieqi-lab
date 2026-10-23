"""Read-only, balanced factorial benchmark of all three experimental methods."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import azero as az
import jieqi as jq
import run_records as rr
from tools import benchmark_native_legal as legal_probe
from tools import benchmark_native_search as node_probe
from tools import profile_selfplay as probe

# (compiled initialization, compiled expansion, compiled legal fast path).
MODES = {
    "baseline": (False, False, False),
    "init": (True, False, False),
    "expand": (False, True, False),
    "nodes": (True, True, False),
    "legal": (False, False, True),
    "legal_init": (True, False, True),
    "legal_expand": (False, True, True),
    "all": (True, True, True),
}


def set_mode(mode):
    initialize, expand, legal = MODES[mode]
    az.PUCTNode._native_search_module = az
    jq.GameState._native_legal_module = jq
    az.PUCTNode.__init__ = node_probe.native.initialize_node if initialize else az.PUCTNode._init_python
    az.PUCTNode.expand = node_probe.native.expand if expand else az.PUCTNode._expand_python
    jq.GameState._iter_legal_moves_fast = (legal_probe.native.iter_legal_moves_fast if legal
                                         else jq.GameState._iter_legal_moves_fast_python)


@contextmanager
def preserve_methods():
    owners = ((az.PUCTNode, "__init__"), (az.PUCTNode, "expand"),
              (az.PUCTNode, "_native_search_module"),
              (jq.GameState, "_iter_legal_moves_fast"), (jq.GameState, "_native_legal_module"))
    saved = [(owner, name, name in vars(owner), getattr(owner, name, None)) for owner, name in owners]
    try:
        yield
    finally:
        for owner, name, present, value in saved:
            if present:
                setattr(owner, name, value)
            elif name in vars(owner):
                delattr(owner, name)


def balanced_orders():
    # Even-size Williams square: each mode occupies each slot once, and each
    # ordered neighbor pair occurs once. This controls first-order order effects.
    modes = tuple(MODES)
    indices = [0, 1, 7, 2, 6, 3, 5, 4]
    return [tuple(modes[(i + row) % len(modes)] for i in indices) for row in range(8)]


def strong_game(index, interval):
    return interval > 0 and (index + 1) % interval == 0


def init_worker(weights, config, rules, search, legal):
    probe.init_worker(weights, config, rules, search_backend=search, legal_backend=legal)
    set_mode("baseline")


def search_task(task):
    mode, fixture, strong, seed = task
    set_mode(mode)
    return probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")


def game_task(task):
    mode, index, strong, seed = task
    set_mode(mode)
    rng = random.Random(seed)
    probe.synchronize(probe.DEVICE)
    start, cpu_start = time.perf_counter(), time.process_time()
    game = az.self_play_game(probe.NET, probe.CONFIG, probe.DEVICE, rng, strong=strong)
    probe.synchronize(probe.DEVICE)
    seconds, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu_start
    assert game.outcome is not None, "unfinished game"
    exact = dict(index=index, seed=seed, strong=strong, outcome=game.outcome,
                 moves=game.move_count, samples=len(game.samples), total=game.leaf_evals_total,
                 unique=game.leaf_evals_unique, sample_sha256=probe.digest([asdict(s) for s in game.samples]),
                 rng_sha256=probe.digest(rng.getstate()))
    return dict(mode=mode, seconds=seconds, process_cpu_seconds=cpu_seconds, exact=exact)


def gpu_snapshot():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return dict(fields="temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
                    value=result.stdout.strip(), error=result.stderr.strip(), returncode=result.returncode)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(error=str(error))


def full_graph_checks(net, config, fixtures, record):
    indices = sorted(set([0, 1, 2, len(fixtures)//3, len(fixtures)//2-1,
                          len(fixtures)//2+5, len(fixtures)-6, len(fixtures)-2]))
    rows = []
    original = az.NeuralMCTS.search
    for index in indices:
        for strong in (False, True):
            reference = None
            for mode in MODES:
                set_mode(mode)
                saved = []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original(search, *args, **kwargs)
                    saved.append(root)
                    return policy, move

                with mock.patch.object(az.NeuralMCTS, "search", capture):
                    result = probe.run_case(net, config, probe.DEVICE, fixtures[index], strong, 20261025+index, "plain")
                row = dict(index=index, strong=strong, mode=mode, exact=result["exact"], graph_sha256=probe.digest(saved[0]))
                signature = (row["exact"], row["graph_sha256"])
                if reference is None:
                    reference = signature
                assert signature == reference, f"full tree mismatch at {index}/{strong}/{mode}"
                rows.append(row)
                del saved, result
                gc.collect()
            record("graph_exact", dict(index=index, strong=strong, modes=len(MODES), identical=True))
    return rows


def search_trials(pool, fixtures, config, rounds, blocks, record):
    trials = []
    for mode in MODES:
        for strong in (False, True):
            list(pool.map(search_task, [(mode, f, strong, 20261100+i) for i, f in enumerate(fixtures)]))
        record("warm", dict(mode=mode, searches=2*len(fixtures)))
    for repeat in range(rounds):
        outputs = {}
        order = balanced_orders()[repeat % 8]
        for slot, mode in enumerate(order):
            tasks = [(mode, f, strong_game(block*len(fixtures)+i, config.strong_self_play_interval),
                      20261200+repeat*10000+block*len(fixtures)+i)
                     for block in range(blocks) for i, f in enumerate(fixtures)]
            telemetry = gpu_snapshot()
            start = time.perf_counter()
            rows = list(pool.map(search_task, tasks))
            row = dict(repeat=repeat, slot=slot, mode=mode, wall_seconds=time.perf_counter()-start,
                       worker_seconds=sum(r["wall_seconds"] for r in rows),
                       worker_cpu_seconds=sum(r["process_cpu_seconds"] for r in rows),
                       gpu_before=telemetry, searches=rows)
            trials.append(row)
            outputs[mode] = [r["exact"] for r in rows]
            record("search_group", row, compact=True)
        assert all(value == outputs["baseline"] for value in outputs.values()), "paired search mismatch"
        record("search_round_exact", dict(repeat=repeat, searches=len(tasks), modes=len(MODES)))
    return trials


def game_trials(pool, config, rounds, count, record):
    trials = []
    # Each adjacent pair reverses its order: every mode has the same average
    # position even when four rounds (rather than a full eight) are requested.
    orders = balanced_orders()
    for repeat in range(rounds):
        forward = orders[(repeat//2)*2 % 8]
        order = forward if repeat % 2 == 0 else tuple(reversed(forward))
        outputs = {}
        for slot, mode in enumerate(order):
            tasks = [(mode, repeat*count+i, strong_game(repeat*count+i, config.strong_self_play_interval),
                      20262000+repeat*count+i) for i in range(count)]
            telemetry = gpu_snapshot()
            start = time.perf_counter()
            pending = [pool.submit(game_task, task) for task in tasks]
            rows = []
            for future in futures.as_completed(pending):
                result = future.result()
                rows.append(result)
                record("game", dict(repeat=repeat, **result))
            rows.sort(key=lambda row: row["exact"]["index"])
            row = dict(repeat=repeat, slot=slot, mode=mode, wall_seconds=time.perf_counter()-start,
                       worker_seconds=sum(r["seconds"] for r in rows),
                       worker_cpu_seconds=sum(r["process_cpu_seconds"] for r in rows),
                       gpu_before=telemetry, games=rows)
            trials.append(row)
            outputs[mode] = [r["exact"] for r in rows]
            record("game_group", row, compact=True)
        assert all(value == outputs["baseline"] for value in outputs.values()), "paired game/sample mismatch"
        record("game_round_exact", dict(repeat=repeat, games=count, modes=len(MODES)))
    return trials


def bootstrap(values):
    rng = random.Random(60261006)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000)]
    return dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values),
                bootstrap_mean_interval_95=np.percentile(means, [2.5, 97.5]).tolist(), paired_values=values)


def summarize(trials):
    if not trials:
        return {}
    repeats = sorted({r["repeat"] for r in trials})
    timing = {(r["repeat"], r["mode"]): r["wall_seconds"] for r in trials}
    result = {}
    for mode in MODES:
        values = [1-timing[repeat, mode]/timing[repeat, "baseline"] for repeat in repeats]
        baseline_seconds = sum(timing[repeat, "baseline"] for repeat in repeats)
        seconds = sum(timing[repeat, mode] for repeat in repeats)
        result[mode] = dict(bootstrap(values), total_seconds=seconds, throughput_ratio=baseline_seconds/seconds)
    return result


def factorial_summary(trials):
    if not trials:
        return {}
    result = {}
    timing = {(r["repeat"], r["mode"]): r["wall_seconds"] for r in trials}
    repeats = sorted({r["repeat"] for r in trials})
    by_flags = {flags: mode for mode, flags in MODES.items()}
    for feature, name in enumerate(("initialization", "expansion", "legal_fast_path")):
        pairs = []
        for flags, off_mode in by_flags.items():
            if not flags[feature]:
                on_flags = list(flags)
                on_flags[feature] = True
                pairs.append((off_mode, by_flags[tuple(on_flags)]))
        values = [statistics.mean(1-timing[repeat, on]/timing[repeat, off] for off, on in pairs)
                  for repeat in repeats]
        result[name] = dict(bootstrap(values), contexts=pairs)
    return result


def verify_hashes(hashes):
    for path, expected in hashes.items():
        assert rr.file_record(path)["sha256"] == expected, f"protected file changed: {path}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-rounds", type=int, default=8)
    parser.add_argument("--search-blocks", type=int, default=4)
    parser.add_argument("--game-rounds", type=int, default=4)
    parser.add_argument("--games-per-round", type=int, default=8)
    parser.add_argument("--micro-pairs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or not args.output.parent.is_dir():
        parser.error("new output and journal in an existing directory required")
    if min(args.search_rounds, args.search_blocks, args.micro_pairs, args.workers, args.games_per_round) < 1 or args.game_rounds < 0:
        parser.error("invalid benchmark sizes")
    if args.search_rounds % 8 or args.game_rounds % 2:
        parser.error("search rounds must be multiples of eight; game rounds must be even")
    if (jq.RULES_BACKEND != "cython" or az.SEARCH_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython"
            or not all((jq.RULES_BACKEND_NATIVE, az.SEARCH_BACKEND_NATIVE, jq.LEGAL_BACKEND_NATIVE))
            or not torch.cuda.is_available()):
        parser.error("CUDA and validated native identities required; set all three JIEQI_*_BACKEND=cython")
    torch.set_num_threads(1)
    paths = [args.run_dir/"models/training_checkpoint.pt", args.run_dir/"models/latest.pt",
             args.run_dir/"models/train_state.json", args.run_dir/"logs/train.log",
             ROOT/"azero.py", ROOT/"jieqi.py", ROOT/"ismcts.py", ROOT/"run_records.py",
             ROOT/"native/jieqi_rules.pyx", ROOT/"native/jieqi_search.pyx", ROOT/"native/jieqi_legal.pyx",
             Path(jq.RULES_BACKEND_NATIVE["path"]), Path(az.SEARCH_BACKEND_NATIVE["path"]),
             Path(jq.LEGAL_BACKEND_NATIVE["path"]), Path(__file__), Path(probe.__file__),
             Path(node_probe.__file__), Path(legal_probe.__file__)]
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir/"selfplay"/checkpoint["replay_files"][-1]["name"]
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    config = az.AZConfig(**checkpoint["config"])
    if config.pcr_enabled:
        parser.error("fixed budgets require PCR off")
    if config.strong_self_play_interval > 0 and args.games_per_round % config.strong_self_play_interval:
        parser.error("each game cohort must contain a whole number of strong-game intervals")
    result = dict(args={k: str(v) for k, v in vars(args).items()}, source_hashes=hashes,
                  checkpoint_iteration=checkpoint["train_state"]["iteration"], config=asdict(config),
                  loaded_native_rules=jq.RULES_BACKEND_NATIVE, loaded_native_search=az.SEARCH_BACKEND_NATIVE,
                  loaded_native_legal=jq.LEGAL_BACKEND_NATIVE, active_methods_by_mode=MODES,
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
    start = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle, preserve_methods():
        def record(label, value, compact=False):
            handle.write(json.dumps(dict(event=label, elapsed_seconds=time.perf_counter()-start, value=value)) + "\n")
            handle.flush()
            display = {k: v for k, v in value.items() if k not in ("searches", "games")} if compact else value
            probe.emit(label, display)

        record("metadata", result)
        net = az.PolicyValueNet().to(probe.DEVICE).eval()
        net.load_state_dict(checkpoint["model"])
        set_mode("baseline")
        fixtures, trajectories = probe.collect_fixtures(net, config, probe.DEVICE, 3, 6, 0)
        result.update(fixtures=[f["metadata"] for f in fixtures], trajectories=trajectories)
        record("fixtures", result["fixtures"])
        result["node_micro"] = node_probe.micro_probe(fixtures, config, args.micro_pairs)
        set_mode("baseline")
        result["legal_micro"] = legal_probe.micro_probe(fixtures, args.micro_pairs)
        result["full_graph_checks"] = full_graph_checks(net, config, fixtures, record)
        del net
        gc.collect()
        torch.cuda.empty_cache()
        with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=init_worker, initargs=(checkpoint["model"], asdict(config), jq.RULES_BACKEND_NATIVE,
                                                   (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE),
                                                   (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))) as pool:
            result["search_trials"] = search_trials(pool, fixtures, config, args.search_rounds, args.search_blocks, record)
            verify_hashes(hashes)
            record("search_summary", summarize(result["search_trials"]))
            result["game_trials"] = game_trials(pool, config, args.game_rounds, args.games_per_round, record)
        verify_hashes(hashes)
        result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-start,
                      search_summary=summarize(result["search_trials"]), game_summary=summarize(result["game_trials"]),
                      search_factorial=factorial_summary(result["search_trials"]), game_factorial=factorial_summary(result["game_trials"]),
                      limitations=["Attack remains compiled in every mode; only three experimental methods differ",
                                   "Loaded native identities are not active-mode declarations; active_methods_by_mode is authoritative",
                                   "Full tree checks retain roots and are not included in throughput timings",
                                   "Fixed roots include diagnostic endgames, not a production frequency sample",
                                   "Game groups include compact sample hashing/IPC but exclude full sample IPC, saving, augmentation and training",
                                   "Bootstrap intervals resample paired rounds, not individual searches/games; few rounds and shared hardware limit inference",
                                   "No optimizer steps, replay writes, checkpoint writes or production backend changes"])
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        record("complete", dict(output=str(args.output), elapsed_seconds=result["elapsed_seconds"], production_unchanged=True))


if __name__ == "__main__":
    main()
