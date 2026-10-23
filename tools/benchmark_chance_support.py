"""Measure immutable k=0 support reuse, then compare exact searches and games."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from dataclasses import asdict
from functools import lru_cache
import gc
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import statistics
import sys
import time
import tracemalloc
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import azero as az
import jieqi as jq
import run_records as rr
from tools import profile_selfplay as probe
from tools.benchmark_chance_allocation import NetworkTrace, canonical_graph
from tools.measure_search_reuse import verify_hashes

MODES = ("uncached", "prepared")


def make_prototype(reference):
    # Used only before integration; later measurements select the actual core helper.
    sample, pool_types = az.sample_reveal_outcomes, jq.POOL_TYPES

    @lru_cache(maxsize=4096)
    def cached(counts):
        return tuple(tuple(row) for row in reference(counts))

    def prepared(counts):
        if (az.sample_reveal_outcomes is sample and jq.POOL_TYPES is pool_types
                and type(counts) in (tuple, az._DeferredChanceCounts)
                and all(row is None or (type(row) is tuple and all(type(n) is int for n in row))
                        for row in counts)):
            return cached(counts)
        return reference(counts)

    return prepared, cached


REFERENCE = getattr(az, "_full_chance_outcomes_uncached", az._full_chance_outcomes)
if hasattr(az, "_cached_full_chance_outcomes"):
    PREPARED, CACHE = az._full_chance_outcomes, az._cached_full_chance_outcomes
else:
    PREPARED, CACHE = make_prototype(REFERENCE)


def paired_summary(trials):
    repeats = sorted({r["repeat"] for r in trials})
    timing = {(r["repeat"], r["mode"]): r["wall_seconds"] for r in trials}
    values = [1 - timing[r, "prepared"] / timing[r, "uncached"] for r in repeats]
    rng = random.Random(202610081)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000)]
    totals = {mode: sum(timing[r, mode] for r in repeats) for mode in MODES}
    return dict(mean_wall_saving=statistics.mean(values), paired_values=values,
                bootstrap_mean_interval_95=np.percentile(means, [2.5, 97.5]).tolist(),
                total_seconds=totals, aggregate_wall_saving=1-totals["prepared"]/totals["uncached"],
                throughput_ratio=totals["uncached"]/totals["prepared"])


def capture_requests(net, config, fixtures):
    keys, slots, requests, rows = [], {}, [], []

    def observed(counts):
        result = REFERENCE(counts)
        key = tuple(None if row is None else tuple(row) for row in counts)
        index = slots.get(key)
        if index is None:
            index = len(keys)
            slots[key] = index
            keys.append(key)
        # Keep the real snapshot identity/type, not an interned value-equal key.
        requests.append(counts)
        return result

    for fixture in fixtures:
        for strong in (False, True):
            seed = 20261025 + fixture["metadata"]["index"]
            plain = probe.run_case(net, config, probe.DEVICE, fixture, strong, seed, "plain")
            before = len(requests)
            with mock.patch.object(az, "_full_chance_outcomes", observed):
                captured = probe.run_case(net, config, probe.DEVICE, fixture, strong, seed, "plain")
            if plain["exact"] != captured["exact"]:
                raise AssertionError("request tracing changed search")
            row = dict(index=fixture["metadata"]["index"], strong=strong,
                       requests=len(requests)-before, exact=plain["exact"])
            rows.append(row)
            probe.emit("requests", row)
    return keys, requests, rows


def microbench(keys, requests, repeats):
    CACHE.cache_clear()
    for counts in keys:
        expected = tuple(tuple(row) for row in REFERENCE(counts))
        actual = PREPARED(counts)
        if pickle.dumps(expected) != pickle.dumps(actual):
            raise AssertionError("support values/order/types differ")
    rows = []
    for repeat in range(repeats):
        for mode in (MODES if repeat % 2 == 0 else tuple(reversed(MODES))):
            CACHE.cache_clear()
            function = REFERENCE if mode == "uncached" else PREPARED
            start = time.perf_counter()
            for counts in requests:
                function(counts)
            row = dict(repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-start,
                       cache=CACHE.cache_info()._asdict())
            rows.append(row)
            probe.emit("micro", row)
    CACHE.cache_clear()
    tracemalloc.start()
    try:
        for counts in requests:
            PREPARED(counts)
        retained, peak = tracemalloc.get_traced_memory()
        info = CACHE.cache_info()._asdict()
    finally:
        tracemalloc.stop()
        CACHE.cache_clear()
    summary = paired_summary(rows)
    summary.update(requests=len(requests), distinct_keys=len(keys),
                   distinct_snapshot_objects=len({id(counts) for counts in requests}),
                   replay_identity="original captured snapshots, not interned keys",
                   distinct_repeat_fraction=1-len(keys)/len(requests),
                   lru_hit_fraction=info["hits"]/(info["hits"]+info["misses"]),
                   cache=info, traced_cache_retained_bytes=retained, traced_cache_peak_bytes=peak,
                   traced_memory_scope="incremental cache allocations; excludes captured request storage")
    raw = []
    for repeat in range(repeats):
        CACHE.cache_clear()
        start = time.perf_counter()
        for counts in requests:
            CACHE(counts)
        raw.append(time.perf_counter()-start)
    CACHE.cache_clear()
    summary["unguarded_cache_seconds_diagnostic_only"] = raw
    return dict(trials=rows, summary=summary)


def init_worker(weights, config, rules, search, legal):
    probe.init_worker(weights, config, rules, search_backend=search, legal_backend=legal)


def search_task(task):
    mode, fixture, strong, seed = task
    CACHE.cache_clear()
    function = REFERENCE if mode == "uncached" else PREPARED
    with mock.patch.object(az, "_full_chance_outcomes", function):
        return probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")


def game_task(task):
    mode, index, strong, seed = task
    CACHE.cache_clear()
    rng = random.Random(seed)
    function = REFERENCE if mode == "uncached" else PREPARED
    with mock.patch.object(az, "_full_chance_outcomes", function):
        probe.synchronize(probe.DEVICE)
        start, cpu_start = time.perf_counter(), time.process_time()
        game = az.self_play_game(probe.NET, probe.CONFIG, probe.DEVICE, rng, strong=strong)
        probe.synchronize(probe.DEVICE)
        seconds, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu_start
    if game.outcome is None:
        raise AssertionError("unfinished game")
    exact = dict(index=index, seed=seed, strong=strong, moves=game.move_count, outcome=game.outcome,
                 samples=len(game.samples), leaf_total=game.leaf_evals_total,
                 leaf_unique=game.leaf_evals_unique,
                 sample_sha256=probe.digest([asdict(s) for s in game.samples]),
                 rng_sha256=probe.digest(rng.getstate()))
    return dict(mode=mode, seconds=seconds, process_cpu_seconds=cpu_seconds, exact=exact,
                cache=CACHE.cache_info()._asdict())


def graph_checks(net, config, fixtures, record, seed_offset=0):
    rows = []
    original = az.NeuralMCTS.search
    for fixture in fixtures:
        for strong in (False, True):
            reference = None
            for mode in MODES:
                CACHE.cache_clear()
                trace, retained = NetworkTrace(), []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original(search, *args, **kwargs)
                    retained.append(root)
                    return policy, move

                hook = net.register_forward_hook(trace.hook)
                try:
                    function = REFERENCE if mode == "uncached" else PREPARED
                    with mock.patch.object(az, "_full_chance_outcomes", function), mock.patch.object(
                            az.NeuralMCTS, "search", capture):
                        result = probe.run_case(net, config, probe.DEVICE, fixture, strong,
                            20261025+fixture["metadata"]["index"]+seed_offset, "plain")
                        graph = canonical_graph(retained[0])
                finally:
                    hook.remove()
                signature = (result["exact"], trace.rows, graph)
                if reference is None:
                    reference = signature
                if signature != reference:
                    raise AssertionError("network/logical graph mismatch")
                rows.append(dict(index=fixture["metadata"]["index"], strong=strong, mode=mode,
                                 exact=result["exact"], graph_sha256=graph,
                                 forward_trace_sha256=probe.digest(trace.rows), forward_calls=len(trace.rows)))
                del retained, trace, signature
                gc.collect()
            record("graph_exact", dict(index=fixture["metadata"]["index"], strong=strong))
    return rows


def timed_trials(pool, fixtures, config, args, record, games=False):
    trials = []
    for repeat in range(args.game_rounds if games else args.search_rounds):
        exact = {}
        for mode in (MODES if repeat % 2 == 0 else tuple(reversed(MODES))):
            if games:
                tasks = [(mode, repeat*args.games_per_round+i,
                          config.strong_self_play_interval > 0 and (i+1) % config.strong_self_play_interval == 0,
                          20265000+repeat*args.games_per_round+i+args.seed_offset)
                         for i in range(args.games_per_round)]
            else:
                tasks = [(mode, f, strong, 20261300+repeat*1000+f["metadata"]["index"]+args.seed_offset)
                         for f in fixtures for strong in (False, True)]
            start = time.perf_counter()
            results = list(pool.map(game_task if games else search_task, tasks))
            row = dict(repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-start,
                       worker_seconds=sum(r["seconds" if games else "wall_seconds"] for r in results),
                       results=results)
            trials.append(row)
            exact[mode] = [r["exact"] for r in results]
            record("game_group" if games else "search_group", row, compact=True)
        if exact["uncached"] != exact["prepared"]:
            raise AssertionError("paired sample/RNG mismatch")
        record("game_exact" if games else "search_exact", dict(repeat=repeat, cases=len(tasks)))
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("measure", "validate"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--micro-rounds", type=int, default=6)
    parser.add_argument("--search-rounds", type=int, default=8)
    parser.add_argument("--game-rounds", type=int, default=6)
    parser.add_argument("--games-per-round", type=int, default=16)
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    fixtures_path = args.output.with_suffix(".fixtures.pkl")
    if (args.output.exists() or journal.exists() or not args.output.parent.is_dir()
            or (args.phase == "measure" and fixtures_path.exists())):
        parser.error("choose new outputs in an existing directory")
    if (min(args.workers, args.micro_rounds, args.search_rounds, args.games_per_round) < 1
            or args.game_rounds < 0 or any(n % 2 for n in (args.micro_rounds, args.search_rounds, args.game_rounds))):
        parser.error("positive sizes and even rounds required; game rounds may be zero")
    if (jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython"
            or az.SEARCH_BACKEND != "python" or not torch.cuda.is_available()):
        parser.error("require CUDA, rules/legal Cython, Python nodes")
    torch.set_num_threads(1)
    paths = [args.run_dir/"models/training_checkpoint.pt", args.run_dir/"models/latest.pt",
             args.run_dir/"models/train_state.json", args.run_dir/"logs/train.log",
             ROOT/"azero.py", ROOT/"jieqi.py", ROOT/"ismcts.py", ROOT/"run_records.py",
             ROOT/"native/jieqi_rules.pyx", ROOT/"native/jieqi_legal.pyx",
             Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"]),
             Path(__file__), Path(probe.__file__), ROOT/"tools/benchmark_chance_allocation.py",
             ROOT/"tools/measure_search_reuse.py"]
    if args.fixtures is not None:
        paths.append(args.fixtures)
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir/"selfplay"/checkpoint["replay_files"][-1]["name"]
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    config = az.AZConfig(**checkpoint["config"])
    if (config.pcr_enabled or not config.chance_nodes or config.chance_k != 0
            or not config.lazy_chance_allocation or not config.leaf_dedup):
        parser.error("require fixed budgets, chance k=0/lazy and leaf dedup")
    if config.strong_self_play_interval > 0 and args.games_per_round % config.strong_self_play_interval:
        parser.error("game cohorts must include whole strong intervals")
    result = dict(args={k: str(v) for k, v in vars(args).items()}, source_hashes=hashes,
                  checkpoint_iteration=checkpoint["train_state"]["iteration"], config=asdict(config),
                  core_helper=hasattr(az, "_cached_full_chance_outcomes"),
                  compiled_guard=getattr(jq, "_immutable_integer_pools", None) is not None,
                  rules_native=jq.RULES_BACKEND_NATIVE, legal_native=jq.LEGAL_BACKEND_NATIVE,
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32, torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name())
    start = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle:
        def record(event, value, compact=False):
            handle.write(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter()-start, value=value))+"\n")
            handle.flush()
            probe.emit(event, {k:v for k,v in value.items() if k != "results"} if compact else value)

        record("metadata", result)
        net = az.PolicyValueNet().to(probe.DEVICE).eval()
        net.load_state_dict(checkpoint["model"])
        if args.fixtures is None:
            fixtures, trajectories = probe.collect_fixtures(net, config, probe.DEVICE, 2, 6, 0)
        else:
            with args.fixtures.open("rb") as handle_in:
                snapshot = pickle.load(handle_in)
            fixtures, trajectories = snapshot["fixtures"], snapshot["trajectories"]
        result.update(fixtures=[f["metadata"] for f in fixtures], trajectories=trajectories)
        if args.phase == "measure":
            keys, requests, rows = capture_requests(net, config, fixtures)
            result["request_searches"] = rows
            result["micro"] = microbench(keys, requests, args.micro_rounds)
            result["support_keys"] = keys
            record("micro_summary", result["micro"]["summary"])
            with fixtures_path.open("xb") as output:
                pickle.dump(dict(fixtures=fixtures, trajectories=trajectories), output)
            result["fixtures_path"] = str(fixtures_path)
        else:
            result["graph_checks"] = graph_checks(net, config, fixtures, record, args.seed_offset)
            del net
            gc.collect()
            torch.cuda.empty_cache()
            identities = (checkpoint["model"], asdict(config), jq.RULES_BACKEND_NATIVE,
                          (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE), (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))
            with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                             initializer=init_worker, initargs=identities) as pool:
                for mode in MODES:
                    list(pool.map(search_task, [(mode, f, strong, 20261100+args.seed_offset)
                                              for f in fixtures for strong in (False, True)]))
                result["search_trials"] = timed_trials(pool, fixtures, config, args, record)
                result["search_summary"] = paired_summary(result["search_trials"])
                record("search_summary", result["search_summary"])
                if args.game_rounds:
                    result["game_trials"] = timed_trials(pool, fixtures, config, args, record, games=True)
                    result["game_summary"] = paired_summary(result["game_trials"])
                    record("game_summary", result["game_summary"])
        verify_hashes(hashes)
        result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-start,
                      limitations=["Support cache contains only immutable derived distributions, not samples or neural outputs",
                                   "Timed tasks clear cache per root/game; conservative versus cross-game worker reuse",
                                   "Micro replay uses original captured snapshots; excludes snapshot creation and full self-play",
                                   "Fixed roots are diagnostic, budgets split equally; games retain strong 7:1 mix",
                                   "GPU tracing and complete graph materialization are outside throughput timings",
                                   "Worker game time excludes hashing; cohort time includes digest/compact IPC, not full sample IPC",
                                   "Paired bootstrap cohorts on one machine are limited evidence; no full 1024-game iteration"])
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        record("complete", dict(output=str(args.output), production_unchanged=True, elapsed_seconds=result["elapsed_seconds"]))


if __name__ == "__main__":
    main()
