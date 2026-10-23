"""Full-game paired screen after exact blocking-event fixed-root screening."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from contextlib import ExitStack
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil
import torch
import azero as az
import run_records as rr
from tools import benchmark_host_wait as roots
from tools import profile_native_selfplay as native
from tools import profile_selfplay as probe
from tools.benchmark_chance_allocation import canonical_graph
from tools.measure_search_reuse import verify_hashes
from tools.measure_selfplay_attribution import validate_config
from tools.measure_selfplay_round import append_event, runtime_identity
from tools.measure_sync_baseline import gpu_snapshot

WAITER = DIRECTORY = None


def make_tasks(checkpoint):
    rng = random.Random()
    rng.setstate(checkpoint["rng"]["selfplay"])
    seeds = [rng.randrange(1 << 30) for _ in range(32)]
    if len(set(seeds)) != len(seeds):
        raise AssertionError("ambiguous game seeds")
    return [(i, seed, (i+1) % 8 == 0) for i, seed in enumerate(seeds)]


def mode_orders(session):
    return [roots.MODES if (repeat+session) % 2 == 0 else tuple(reversed(roots.MODES)) for repeat in range(2)]


def init_worker(weights, config, expected, directory):
    global WAITER, DIRECTORY
    if json.loads(json.dumps(runtime_identity())) != json.loads(json.dumps(expected)):
        raise RuntimeError("worker identity differs")
    az._init_selfplay_worker(weights, az.AZConfig(**config), "cuda")
    probe.NET, probe.CONFIG, probe.DEVICE = az._WORKER_NET, az._WORKER_CONFIG, az._WORKER_DEVICE
    WAITER = roots.BlockingPolicyWait()
    roots.WAITER = WAITER
    DIRECTORY = Path(directory)


def game_task(task):
    mode, index, seed, strong = task
    if mode not in roots.MODES:
        raise ValueError("unknown game mode")
    original, rng_states = az.self_play_game, []

    def capture(net, config, device, rng, **kwargs):
        game = original(net, config, device, rng, **kwargs)
        rng_states.append(probe.digest(rng.getstate()))
        return game

    before = WAITER.calls
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(az, "self_play_game", capture))
        if mode == "blocking":
            WAITER.install(stack)
        probe.synchronize(probe.DEVICE)
        start, cpu, thread = time.perf_counter(), time.process_time(), time.thread_time()
        game = az._selfplay_worker((seed, strong))
        probe.synchronize(probe.DEVICE)
        row = dict(wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu,
                   thread_cpu_seconds=time.thread_time()-thread)
    if game.outcome is None or len(rng_states) != 1:
        raise AssertionError("unfinished or ambiguous game")
    waits = WAITER.calls-before
    if (mode == "blocking" and waits < 1) or (mode == "plain" and waits):
        raise AssertionError("incorrect game event coverage")
    row.update(mode=mode, pid=os.getpid(), event_waits=waits, exact=dict(index=index, seed=seed, strong=strong,
               moves=game.move_count, outcome=game.outcome, samples=len(game.samples),
               leaf_total=game.leaf_evals_total, leaf_unique=game.leaf_evals_unique,
               game_sha256=probe.digest(asdict(game)), rng_sha256=rng_states[0]))
    append_event(DIRECTORY/f"games_{os.getpid()}.jsonl", "game", row)
    probe.emit("game_complete", dict(mode=mode, index=index, strong=strong, moves=game.move_count))
    # Keep the original full result in IPC; parent verifies it after the cohort clock stops.
    return dict(row=row, game=game)


def collect_games(outputs):
    rows = []
    for output in outputs:
        row, game = output["row"], output["game"]
        if probe.digest(asdict(game)) != row["exact"]["game_sha256"]:
            raise AssertionError("full sample IPC changed the returned game")
        rows.append(row)
    return rows


def graph_checks(weights, config, fixtures, record):
    net = az.PolicyValueNet().to("cuda").eval()
    az.load_state_dict_padded(net, weights)
    waiter, original, checks = roots.BlockingPolicyWait(), az.NeuralMCTS.search, []
    for fixture in fixtures:
        for strong in (False, True):
            outputs = []
            for mode in roots.MODES:
                retained = []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original(search, *args, **kwargs)
                    retained.append(root)
                    return policy, move

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(az.NeuralMCTS, "search", capture))
                    if mode == "blocking":
                        waiter.install(stack)
                    row = probe.run_case(net, config, torch.device("cuda"), fixture, strong,
                                         202610084+fixture["metadata"]["index"], "plain")
                outputs.append(dict(exact=row["exact"], graph=canonical_graph(retained[0])))
            if outputs[0] != outputs[1]:
                raise AssertionError("without network hooks, wait changed graph/policy/RNG/counters")
            check = dict(index=fixture["metadata"]["index"], strong=strong, exact=True, network_hooks=False)
            checks.append(check)
            record("no_hook_graph_exact", check)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return checks


def exact_rows(rows):
    return json.loads(json.dumps([row["exact"] for row in rows], allow_nan=False))


def summarize(groups):
    keys = {(s, r, mode) for s in range(2) for r in range(2) for mode in roots.MODES}
    if len(groups) != 8 or {(g["session"], g["repeat"], g["mode"]) for g in groups} != keys:
        raise ValueError("require two pools with two complete 16-game pairs each")
    indexed = {(g["session"], g["repeat"], g["mode"]): g for g in groups}
    pools = []
    for session in range(2):
        seconds, cpu, gains = {m: 0.0 for m in roots.MODES}, {m: 0.0 for m in roots.MODES}, []
        for repeat in range(2):
            baseline = indexed[session, repeat, "plain"]
            candidate = indexed[session, repeat, "blocking"]
            for mode, group in (("plain", baseline), ("blocking", candidate)):
                rows = group["rows"]
                if (len(rows) != 16 or [r["exact"]["index"] for r in rows] != list(range(repeat*16, (repeat+1)*16))
                        or sum(r["exact"]["strong"] for r in rows) != 2 or group["wall_seconds"] <= 0):
                    raise ValueError("require original ordered complete 7:1 cohorts and positive clocks")
                if any((r["event_waits"] < 1 if mode == "blocking" else r["event_waits"] != 0) for r in rows):
                    raise AssertionError("incorrect event coverage")
                seconds[mode] += group["wall_seconds"]
                cpu[mode] += sum(r["process_cpu_seconds"] for r in rows)
            if exact_rows(baseline["rows"]) != exact_rows(candidate["rows"]):
                raise AssertionError("wait changed full game samples/outcomes/moves/RNG/leaf counters")
            if exact_rows(baseline["rows"]) != exact_rows(indexed[1-session, repeat, "plain"]["rows"]):
                raise AssertionError("independent pools changed complete games")
            gains.append(baseline["wall_seconds"]/candidate["wall_seconds"]-1)
        pools.append(dict(session=session, cohort_seconds=seconds, paired_gains=gains,
                          throughput_gain=seconds["plain"]/seconds["blocking"]-1,
                          worker_cpu_seconds=cpu, worker_cpu_change=cpu["blocking"]/cpu["plain"]-1))
    seconds = {m: sum(p["cohort_seconds"][m] for p in pools) for m in roots.MODES}
    gain = seconds["plain"]/seconds["blocking"]-1
    gate = gain >= 0.03 and all(p["throughput_gain"] >= 0.01 and all(g > 0 for g in p["paired_gains"]) for p in pools)
    return dict(pools=pools, cohort_seconds=seconds, aggregate_throughput_gain=gain,
                paired_games_exact=True, confirmation_gate_passed=gate, production_promotion=False,
                thresholds=dict(aggregate_gain_minimum=0.03, each_pool_gain_minimum=0.01, all_four_pairs_win=True),
                limitations=["Four paired cohorts and 32 unique games, repeated in two fresh pools",
                             "Short cohorts expose strong-game drain tails; not a 1024-game training-round gain",
                             "Passing prioritizes confirmation, not production promotion"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root-result", type=Path, required=True)
    args = parser.parse_args()
    directory, reference_path = args.output_dir.resolve(), args.root_result.resolve()
    source = ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs"
    native.validate_output(directory, source.resolve())
    if (ROOT/"data/pilots").resolve() not in reference_path.parents:
        parser.error("root result must be below workspace data/pilots")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference.get("kind") != "host_wait_root_screen" or not reference.get("production_unchanged")
            or not reference.get("summary", {}).get("full_game_gate_passed")
            or not reference.get("summary", {}).get("paired_roots_exact")):
        parser.error("require a completed passing exact root screen")
    verify_hashes(reference["source_hashes"])
    if (json.loads(json.dumps(runtime_identity())) != reference["identity"]
            or str(torch.__version__) != reference["torch"] or torch.version.cuda != reference["cuda"]
            or Path(sys.executable).resolve() != Path(reference["python"]).resolve() or not torch.cuda.is_available()):
        parser.error("runtime identity differs from root screen")
    if any(p.info["pid"] != os.getpid() and (p.info["name"] or "").lower() in ("python.exe", "pythonw.exe", "nsys.exe")
           for p in psutil.process_iter(["pid", "name"])):
        parser.error("finish other Python/Nsight work first")
    wpr = subprocess.run([str(Path(os.environ["SystemRoot"])/"System32/wpr.exe"), "-status"],
                         capture_output=True, text=True, timeout=10)
    if wpr.returncode or "WPR is not recording" not in wpr.stdout+wpr.stderr:
        parser.error("cannot confirm WPR is not recording")
    torch.set_num_threads(1)
    hashes = dict(reference["source_hashes"])
    for path in (reference_path, Path(__file__).resolve(), ROOT/"test_benchmark_host_wait_games.py",
                 ROOT/"data/pilots/host_wait_games_20261009_iter28_protocol.md"):
        hashes[str(path)] = rr.file_record(path)["sha256"]
    checkpoint = torch.load(source/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
    verify_hashes(hashes)
    config = az.AZConfig(**checkpoint["config"])
    validate_config(config)
    config.self_play_games, config.self_play_workers = 16, 8
    config.data_dir, config.model_dir, config.log_dir = (str(directory/n) for n in ("samples", "models", "logs"))
    with (ROOT/"data/pilots/chance_support_20261008_iter28/measure.fixtures.pkl").open("rb") as handle:
        fixtures = pickle.load(handle)["fixtures"]
    tasks = make_tasks(checkpoint)
    directory.mkdir(parents=True)
    origin = time.perf_counter()

    def record(event, value):
        append_event(directory/"events.jsonl", event, dict(elapsed=time.perf_counter()-origin, data=value))
        probe.emit(event, value)

    result = dict(kind="host_wait_game_screen", source_hashes=hashes, identity=runtime_identity(), python=sys.executable,
                  iteration=checkpoint["train_state"]["iteration"], config=asdict(config), groups=[], snapshots=[],
                  unique_tasks=tasks, production_training=False, precision_changed=False, full_sample_ipc=True,
                  scheduling="Persistent warmed eight-worker pools; ordered map, original worker tasks/results",
                  limitations=["Not generate_selfplay_results orchestration, replay/augmentation/save/optimizer or 4M residency",
                               "No continuous CPU thermal/clock/power capture; background throttling hypothesis not tested",
                               "Short screen only; no production promotion regardless of result"])
    record("start", dict(output=str(directory), sessions=2, pairs_per_pool=2, games_per_cohort=16, total_games=128))
    result["snapshots"].append(dict(stage="before_audits", gpu=gpu_snapshot()))
    result["graph_checks"] = graph_checks(checkpoint["model"], config, fixtures, record)
    for session in range(2):
        result["snapshots"].append(dict(stage=f"before_pool_{session}", gpu=gpu_snapshot()))
        with futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"), initializer=init_worker,
                initargs=(checkpoint["model"], asdict(config), runtime_identity(), str(directory))) as pool:
            for mode in roots.MODES if session == 0 else tuple(reversed(roots.MODES)):
                warm = list(pool.map(roots.root_task, [(mode, f, strong, 202610101+f["metadata"]["index"])
                    for f in fixtures for strong in (False, True)]))
                pids = sorted({r["pid"] for r in warm})
                if len(pids) != 8:
                    raise RuntimeError("warmup did not exercise all eight workers")
                record("warm", dict(session=session, mode=mode, pids=pids))
            for repeat, order in enumerate(mode_orders(session)):
                paired = {}
                for mode in order:
                    start = time.perf_counter()
                    outputs = list(pool.map(game_task, [(mode, *task) for task in tasks[repeat*16:(repeat+1)*16]]))
                    wall = time.perf_counter()-start
                    rows = collect_games(outputs)
                    del outputs
                    group = dict(session=session, repeat=repeat, mode=mode, wall_seconds=wall, rows=rows)
                    result["groups"].append(group)
                    paired[mode] = exact_rows(rows)
                    record("cohort", {k:v for k,v in group.items() if k != "rows"})
                if paired["plain"] != paired["blocking"]:
                    raise AssertionError("wait changed full games")
        record("pool_shutdown_complete", dict(session=session))
        result["snapshots"].append(dict(stage=f"after_pool_{session}", gpu=gpu_snapshot()))
    result["summary"] = summarize(result["groups"])
    verify_hashes(hashes)
    result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-origin)
    with (directory/"result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    record("complete", dict(elapsed=result["elapsed_seconds"], summary=result["summary"]))


if __name__ == "__main__":
    main()
