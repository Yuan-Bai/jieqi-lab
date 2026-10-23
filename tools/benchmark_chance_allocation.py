"""Read-only eager/lazy chance allocation equivalence, time and memory benchmark."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from dataclasses import asdict, replace
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import statistics
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
from tools import profile_selfplay as probe
from tools.measure_search_reuse import count_graph, verify_hashes

MODES = ("eager", "lazy")


def canonical_graph(root):
    seen, pending = set(), [root]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        az.materialize_chance_outcomes(node)
        pending.extend(node.children.values())
        pending.extend(node.public_outcomes.values())
        pending.extend(node.decision_nodes.values())
    return probe.digest(root)


class NetworkTrace:
    def __init__(self):
        self.rows = []

    def hook(self, net, args, outputs):
        tensors = [args[0], *outputs]
        self.rows.append([(tuple(t.shape), str(t.dtype),
                           hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()) for t in tensors])


def init_worker(weights, config, rules, search, legal):
    probe.init_worker(weights, config, rules, search_backend=search, legal_backend=legal)


def search_task(task):
    mode, fixture, strong, seed = task
    probe.CONFIG.lazy_chance_allocation = mode == "lazy"
    return probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")


def game_task(task):
    mode, index, strong, seed = task
    probe.CONFIG.lazy_chance_allocation = mode == "lazy"
    rng = random.Random(seed)
    probe.synchronize(probe.DEVICE)
    start, cpu_start = time.perf_counter(), time.process_time()
    game = az.self_play_game(probe.NET, probe.CONFIG, probe.DEVICE, rng, strong=strong)
    probe.synchronize(probe.DEVICE)
    seconds, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu_start
    if game.outcome is None:
        raise AssertionError("unfinished game")
    exact = dict(index=index, seed=seed, strong=strong, moves=game.move_count, outcome=game.outcome,
                 samples=len(game.samples), leaf_total=game.leaf_evals_total, leaf_unique=game.leaf_evals_unique,
                 sample_sha256=probe.digest([asdict(s) for s in game.samples]), rng_sha256=probe.digest(rng.getstate()))
    return dict(mode=mode, seconds=seconds, process_cpu_seconds=cpu_seconds, exact=exact)


def windows_memory():
    if sys.platform != "win32":
        return None
    import ctypes as ct
    from ctypes import wintypes as wt

    class Counters(ct.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
            (name, ct.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

    kernel, psapi = ct.WinDLL("kernel32", use_last_error=True), ct.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wt.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (wt.HANDLE, ct.POINTER(Counters), wt.DWORD)
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    counts = Counters()
    counts.cb = ct.sizeof(counts)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ct.byref(counts), counts.cb):
        raise ct.WinError(ct.get_last_error())
    return dict(working_set=counts.WorkingSetSize, peak_working_set=counts.PeakWorkingSetSize,
                private_bytes=counts.PrivateUsage)


def memory_task(task):
    # A fresh process per case prevents earlier eager searches from contaminating the lazy peak.
    mode, fixture, strong, seed = task
    probe.CONFIG.lazy_chance_allocation = mode == "lazy"
    warm = az.NeuralMCTS(probe.NET, probe.CONFIG, probe.DEVICE, random.Random(9))
    end = probe.controlled_endgames()[0]
    warm.search(end["state"], end["belief"], [], 1.0, simulations=64)
    del warm
    gc.collect()
    before = windows_memory()
    retained = []
    original = az.NeuralMCTS.search

    def capture(search, *args, **kwargs):
        kwargs["return_root"] = True
        policy, move, root = original(search, *args, **kwargs)
        retained.append(root)
        return policy, move

    with mock.patch.object(az.NeuralMCTS, "search", capture):
        result = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    after = windows_memory()
    return dict(mode=mode, strong=strong, before=before, root_alive=after,
                graph=count_graph(retained[0]), exact=result["exact"])


def graph_checks(net, config, fixtures, record):
    original, rows = az.NeuralMCTS.search, []
    for fixture in fixtures:
        for strong in (False, True):
            reference = None
            for mode in MODES:
                trace, retained = NetworkTrace(), []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original(search, *args, **kwargs)
                    retained.append(root)
                    return policy, move

                hook = net.register_forward_hook(trace.hook)
                try:
                    with mock.patch.object(az.NeuralMCTS, "search", capture):
                        result = probe.run_case(net, replace(config, lazy_chance_allocation=mode == "lazy"),
                            probe.DEVICE, fixture, strong, 20261025+fixture["metadata"]["index"], "plain")
                finally:
                    hook.remove()
                graph = count_graph(retained[0])
                digest = canonical_graph(retained[0])
                signature = (result["exact"], trace.rows, digest)
                if reference is None:
                    reference = signature
                if signature != reference:
                    raise AssertionError(f"graph/network mismatch: {fixture['metadata']['index']}/{strong}/{mode}")
                rows.append(dict(index=fixture["metadata"]["index"], strong=strong, mode=mode,
                                 exact=result["exact"], graph=graph, graph_sha256=digest,
                                 forward_trace_sha256=probe.digest(trace.rows), forward_calls=len(trace.rows),
                                 input_rows=sum(row[0][0][0] for row in trace.rows)))
                del retained, trace, signature
                gc.collect()
            record("graph_exact", dict(index=fixture["metadata"]["index"], strong=strong, identical=True))
    return rows


def paired_summary(trials):
    repeats = sorted({r["repeat"] for r in trials})
    timing = {(r["repeat"], r["mode"]): r["wall_seconds"] for r in trials}
    values = [1-timing[repeat, "lazy"]/timing[repeat, "eager"] for repeat in repeats]
    rng = random.Random(60261006)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000)]
    totals = {mode: sum(timing[repeat, mode] for repeat in repeats) for mode in MODES}
    return dict(mean_wall_saving=statistics.mean(values), paired_values=values,
                bootstrap_mean_interval_95=np.percentile(means, [2.5, 97.5]).tolist(),
                total_seconds=totals, throughput_ratio=totals["eager"]/totals["lazy"])


def trials(pool, fixtures, config, args, record, games=False):
    rows = []
    rounds = args.game_rounds if games else args.search_rounds
    for repeat in range(rounds):
        exact = {}
        order = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            if games:
                tasks = [(mode, repeat*args.games_per_round+i,
                          config.strong_self_play_interval > 0 and (i+1) % config.strong_self_play_interval == 0,
                          20264000+repeat*args.games_per_round+i) for i in range(args.games_per_round)]
            else:
                tasks = [(mode, f, strong, 20261200+repeat*1000+f["metadata"]["index"])
                         for f in fixtures for strong in (False, True)]
            start = time.perf_counter()
            result = list(pool.map(game_task if games else search_task, tasks))
            row = dict(repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-start,
                       worker_seconds=sum(r["seconds" if games else "wall_seconds"] for r in result),
                       results=result)
            rows.append(row)
            exact[mode] = [r["exact"] for r in result]
            record("game_group" if games else "search_group", row, compact=True)
        if exact["eager"] != exact["lazy"]:
            raise AssertionError("paired game/sample mismatch" if games else "paired search mismatch")
        record("game_round_exact" if games else "search_round_exact", dict(repeat=repeat, cases=len(tasks)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--search-rounds", type=int, default=4)
    parser.add_argument("--game-rounds", type=int, default=4)
    parser.add_argument("--games-per-round", type=int, default=16)
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or not args.output.parent.is_dir():
        parser.error("new output/journal in existing directory required")
    if min(args.workers, args.search_rounds, args.game_rounds, args.games_per_round) < 1:
        parser.error("positive sample sizes required")
    if args.search_rounds % 2 or args.game_rounds % 2:
        parser.error("even rounds required for balanced order")
    if (jq.RULES_BACKEND != "cython" or az.SEARCH_BACKEND != "python"
            or jq.LEGAL_BACKEND != "cython" or not torch.cuda.is_available()):
        parser.error("require CUDA, rules/legal Cython, Python node backend")
    torch.set_num_threads(1)
    paths = [args.run_dir/"models/training_checkpoint.pt", args.run_dir/"models/latest.pt",
             args.run_dir/"models/train_state.json", args.run_dir/"logs/train.log",
             ROOT/"azero.py", ROOT/"jieqi.py", ROOT/"ismcts.py", ROOT/"run_records.py",
             ROOT/"native/jieqi_rules.pyx", ROOT/"native/jieqi_search.pyx", ROOT/"native/jieqi_legal.pyx",
             Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"]),
             ROOT/"_jieqi_search.cp311-win_amd64.pyd", Path(__file__), Path(probe.__file__),
             ROOT/"tools/measure_search_reuse.py"]
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir/"selfplay"/checkpoint["replay_files"][-1]["name"]
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    config = az.AZConfig(**checkpoint["config"])
    config.lazy_chance_allocation = False
    if config.pcr_enabled or not config.chance_nodes or config.chance_k != 0 or not config.leaf_dedup:
        parser.error("require unchanged fixed budgets, chance k=0, leaf dedup on")
    if config.strong_self_play_interval > 0 and args.games_per_round % config.strong_self_play_interval:
        parser.error("each cohort must contain whole strong-game intervals")
    identities = (checkpoint["model"], asdict(config), jq.RULES_BACKEND_NATIVE,
                  (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE), (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))
    result = dict(args={k: str(v) for k, v in vars(args).items()}, source_hashes=hashes, config=asdict(config),
                  checkpoint_iteration=checkpoint["train_state"]["iteration"],
                  runtime=dict(rules_backend=jq.RULES_BACKEND, rules_native=jq.RULES_BACKEND_NATIVE,
                               search_backend=az.SEARCH_BACKEND, search_native=az.SEARCH_BACKEND_NATIVE,
                               legal_backend=jq.LEGAL_BACKEND, legal_native=jq.LEGAL_BACKEND_NATIVE),
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
    start = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle:
        def record(event, value, compact=False):
            handle.write(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter()-start, value=value)) + "\n")
            handle.flush()
            probe.emit(event, {k: v for k, v in value.items() if k != "results"} if compact else value)

        record("metadata", result)
        net = az.PolicyValueNet().to(probe.DEVICE).eval()
        net.load_state_dict(checkpoint["model"])
        fixtures, trajectories = probe.collect_fixtures(net, config, probe.DEVICE, 3, 6, 0)
        result.update(fixtures=[f["metadata"] for f in fixtures], trajectories=trajectories)
        record("fixtures", result["fixtures"])
        result["graph_checks"] = graph_checks(net, config, fixtures, record)
        del net
        gc.collect()
        torch.cuda.empty_cache()
        ctx = mp.get_context("spawn")
        with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                         initializer=init_worker, initargs=identities) as pool:
            for mode in MODES:
                list(pool.map(search_task, [(mode, f, strong, 20261100) for f in fixtures for strong in (False, True)]))
                record("warm", dict(mode=mode, searches=2*len(fixtures)))
            result["search_trials"] = trials(pool, fixtures, config, args, record)
            record("search_summary", paired_summary(result["search_trials"]))
            result["game_trials"] = trials(pool, fixtures, config, args, record, games=True)
        record("game_summary", paired_summary(result["game_trials"]))
        result["memory_checks"] = []
        for strong in (False, True):
            reference = None
            for mode in MODES:
                with futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx,
                                                 initializer=init_worker, initargs=identities) as pool:
                    row = pool.submit(memory_task, (mode, fixtures[0], strong, 20261025)).result()
                if reference is None:
                    reference = row["exact"]
                if row["exact"] != reference:
                    raise AssertionError("memory case mismatch")
                result["memory_checks"].append(row)
                record("memory", row)
        verify_hashes(hashes)
        result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-start,
                      search_summary=paired_summary(result["search_trials"]),
                      game_summary=paired_summary(result["game_trials"]),
                      limitations=["Timed searches use production return_root=False; allocation, GC and tree release are included",
                                   "Network tracing and full logical-tree materialization are separate from all throughput timings",
                                   "Fixed roots split 800/3200 equally; game cohorts use the production 7:1 mix",
                                   "Game timing excludes sample hashing/IPC per worker but cohort time includes compact hash/IPC",
                                   "No full sample IPC, augmentation, replay-buffer residency, optimizer steps or training writes",
                                   "Process RSS/peak includes Python, PyTorch and CUDA driver; fresh process per memory case, not 8-worker peak",
                                   "Bootstrap resamples paired cohorts; four cohorts and one machine are limited evidence"])
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        record("complete", dict(output=str(args.output), elapsed_seconds=result["elapsed_seconds"], production_unchanged=True))


if __name__ == "__main__":
    main()
