"""No-profiler timing stability gate before isolated host-wait experiments."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from contextlib import ExitStack
import csv
from dataclasses import asdict
import gc
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import threading
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil
import torch
import azero as az
import jieqi as jq
import run_records as rr
from tools import profile_native_selfplay as native
from tools import profile_selfplay as probe
from tools.benchmark_chance_allocation import NetworkTrace, canonical_graph
from tools.measure_search_reuse import verify_hashes
from tools.measure_selfplay_attribution import Attribution, instrument_evaluation, validate_config
from tools.measure_selfplay_round import append_event, runtime_identity

MODES = ("plain", "nvtx", "timers")
PHASES = ("encode", "h2d", "forward", "softmax", "policy_d2h", "value_d2h")
NVTX = TIMERS = None


class HostTimers(Attribution):
    """Cache AST setup during warmup; no events, labels or added synchronization."""

    def __init__(self):
        super().__init__(cuda_events=False, labels=False, logic=False)
        self.functions = {}

    def reset(self):
        if self.stack or self.events:
            raise RuntimeError("unfinished timer scope")
        self.rows = {}

    def install(self, stack):
        if not self.functions:
            for attribute in ("_evaluate", "_evaluate_batch"):
                self.functions[attribute] = self.wrapper("evaluation", instrument_evaluation(
                    getattr(az.NeuralMCTS, attribute), self))
        for attribute, function in self.functions.items():
            stack.enter_context(mock.patch.object(az.NeuralMCTS, attribute, function))


def mode_orders(session):
    orders = list(itertools.permutations(MODES))
    return orders if session == 0 else list(reversed(orders))


def init_worker(weights, config, expected):
    global NVTX, TIMERS
    native.init_worker(weights, config, expected)
    NVTX, TIMERS = native.NvtxLabels(), HostTimers()


def root_task(task):
    mode, fixture, strong, seed = task
    if mode not in MODES:
        raise ValueError("unknown measurement mode")
    with ExitStack() as stack:
        if mode == "nvtx":
            NVTX.install(stack)
            stack.enter_context(NVTX.scope("jieqi/root/" + ("3200" if strong else "800")))
        elif mode == "timers":
            TIMERS.reset()
            TIMERS.install(stack)
            stack.enter_context(TIMERS.scope("root"))
        row = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    row.update(mode=mode, pid=os.getpid(), native_thread_id=threading.get_native_id(),
               timings=TIMERS.finish() if mode == "timers" else None)
    return row


def exact_rows(rows):
    fields = ("index", "strong", "simulations", "seed", "exact")
    return json.loads(json.dumps([{key: row[key] for key in fields} for row in rows], allow_nan=False))


def graph_checks(weights, config, fixtures):
    net = az.PolicyValueNet().to("cuda").eval()
    az.load_state_dict_padded(net, weights)
    original, results, observer = az.NeuralMCTS.search, [], HostTimers()
    for fixture in fixtures:
        for strong in (False, True):
            outputs = []
            for mode in ("plain", "timers"):
                trace, retained = NetworkTrace(), []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original(search, *args, **kwargs)
                    retained.append(root)
                    return policy, move

                hook = net.register_forward_hook(trace.hook)
                try:
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(az.NeuralMCTS, "search", capture))
                        if mode == "timers":
                            observer.reset()
                            observer.install(stack)
                        row = probe.run_case(net, config, torch.device("cuda"), fixture, strong,
                                             202610084+fixture["metadata"]["index"], "plain")
                    outputs.append(dict(exact=row["exact"], graph=canonical_graph(retained[0]), network=trace.rows))
                finally:
                    hook.remove()
            if outputs[0] != outputs[1]:
                raise AssertionError("host timers changed graph/network/strategy/RNG")
            result = dict(index=fixture["metadata"]["index"], strong=strong,
                          network_requests=len(outputs[0]["network"]), exact=True)
            results.append(result)
            probe.emit("graph_exact", result)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return results


def summarize(groups):
    if len(groups) != 36 or {(g["session"], g["repeat"], g["mode"]) for g in groups} != {
            (session, repeat, mode) for session in range(2) for repeat in range(6) for mode in MODES}:
        raise ValueError("require two pools, six balanced permutations, three modes")
    by_key = {(g["session"], g["repeat"], g["mode"]): g for g in groups}
    phase_summaries, stability = [], []
    overhead = {mode: [] for mode in MODES[1:]}
    for session in range(2):
        plain = [by_key[session, repeat, "plain"]["wall_seconds"] for repeat in range(6)]
        if min(plain) <= 0:
            raise ValueError("nonpositive cohort duration")
        stability.append(dict(session=session, plain_seconds=plain, median=statistics.median(plain),
                              cv=statistics.stdev(plain)/statistics.mean(plain),
                              late_vs_early=statistics.mean(plain[3:])/statistics.mean(plain[:3])-1))
        phases, root_wall = {}, 0.0
        for repeat in range(6):
            expected = exact_rows(by_key[session, repeat, "plain"]["rows"])
            if len(expected) != 36:
                raise ValueError("require all 36 roots")
            if exact_rows(by_key[1-session, repeat, "plain"]["rows"]) != expected:
                raise AssertionError("separate pools changed policy/RNG/counters")
            for mode in MODES[1:]:
                group = by_key[session, repeat, mode]
                if exact_rows(group["rows"]) != expected:
                    raise AssertionError("instrumentation changed policy/RNG/counters")
                overhead[mode].append(group["wall_seconds"]/plain[repeat]-1)
            for row in by_key[session, repeat, "timers"]["rows"]:
                values = row["timings"]
                count = values["evaluation"]["calls"]
                if count < 1 or any(values[phase]["calls"] != count for phase in PHASES):
                    raise AssertionError("incomplete network phase counts")
                root_wall += values["root"]["wall_inclusive"]
                for phase, value in values.items():
                    target = phases.setdefault(phase, dict(calls=0, wall=0.0, thread_cpu=0.0))
                    target["calls"] += value["calls"]
                    target["wall"] += value["wall_inclusive"]
                    target["thread_cpu"] += value["thread_cpu_inclusive"]
        if root_wall <= 0 or phases["policy_d2h"]["wall"] <= 0:
            raise ValueError("nonpositive phase duration")
        phase_summaries.append(dict(session=session, worker_root_wall=root_wall, phases=phases,
                                   evaluation_fraction=phases["evaluation"]["wall"]/root_wall,
                                   policy_d2h_fraction=phases["policy_d2h"]["wall"]/root_wall,
                                   policy_d2h_thread_cpu_to_wall=phases["policy_d2h"]["thread_cpu"]/phases["policy_d2h"]["wall"]))
    aggregate_overhead = {mode: sum(g["wall_seconds"] for g in groups if g["mode"] == mode)/
                         sum(g["wall_seconds"] for g in groups if g["mode"] == "plain")-1 for mode in MODES[1:]}
    between_pool_median_change = stability[1]["median"]/stability[0]["median"]-1
    stable = (all(s["cv"] <= 0.05 and abs(s["late_vs_early"]) <= 0.10 for s in stability)
              and abs(between_pool_median_change) <= 0.10)
    small_overhead = all(abs(value) <= 0.03 for value in aggregate_overhead.values())
    reproducible_boundary = (abs(phase_summaries[0]["policy_d2h_fraction"]-phase_summaries[1]["policy_d2h_fraction"]) <= 0.05
                             and all(s["policy_d2h_fraction"] >= 0.20 and s["policy_d2h_thread_cpu_to_wall"] >= 0.80
                                     for s in phase_summaries))
    return dict(stability=stability, between_pool_median_change=between_pool_median_change,
                paired_overhead=overhead, aggregate_overhead=aggregate_overhead, phase_summaries=phase_summaries,
                timing_stability_passed=stable, measurement_overhead_passed=small_overhead,
                active_readback_reproduced=reproducible_boundary,
                host_wait_experiment_gate_passed=stable and small_overhead and reproducible_boundary,
                paired_roots_exact=True, thresholds=dict(cv=0.05, drift=0.10, absolute_overhead=0.03,
                    phase_fraction_difference=0.05, readback_fraction_minimum=0.20, readback_cpu_ratio_minimum=0.80))


def gpu_snapshot():
    fields = ("name", "driver_version", "pstate", "temperature.gpu", "power.draw", "clocks.current.sm", "clocks.current.memory")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu="+",".join(fields), "--format=csv,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return dict(rows=list(csv.DictReader(result.stdout.splitlines(), skipinitialspace=True)))
    except (OSError, subprocess.SubprocessError) as error:
        return dict(error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs"
    fixture_path = ROOT/"data/pilots/chance_support_20261008_iter28/measure.fixtures.pkl"
    directory = args.output_dir.resolve()
    native.validate_output(directory, source.resolve())
    reference_path = args.reference_file.resolve()
    if (ROOT/"data/pilots").resolve() not in reference_path.parents:
        parser.error("reference must be below workspace data/pilots")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    native.validate_reference(reference)
    identity = runtime_identity()
    if json.loads(json.dumps(identity)) != reference["identity"] or not torch.cuda.is_available():
        parser.error("require original CUDA/backend/precision identity")
    if any(p.info["pid"] != os.getpid() and (p.info["name"] or "").lower() in ("python.exe", "pythonw.exe", "nsys.exe")
           for p in psutil.process_iter(["pid", "name"])):
        parser.error("finish other Python/Nsight work before isolated measurement")
    wpr = subprocess.run([str(Path(os.environ["SystemRoot"])/"System32/wpr.exe"), "-status"],
                         capture_output=True, text=True, timeout=10)
    if wpr.returncode or "WPR is not recording" not in wpr.stdout+wpr.stderr:
        parser.error("cannot confirm WPR is not recording")
    torch.set_num_threads(1)
    hashes = dict(reference["source_hashes"])
    for path in (reference_path, Path(__file__).resolve(), ROOT/"tests/test_measure_sync_baseline.py"):
        hashes[str(path)] = rr.file_record(path)["sha256"]
    checkpoint = torch.load(source/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
    verify_hashes(hashes)
    config = az.AZConfig(**checkpoint["config"])
    validate_config(config)
    config.data_dir, config.model_dir, config.log_dir = (str(directory/n) for n in ("samples", "models", "logs"))
    with fixture_path.open("rb") as handle:
        fixtures = pickle.load(handle)["fixtures"]
    if len(fixtures) != 18 or [f["metadata"]["index"] for f in fixtures] != list(range(18)):
        parser.error("require trusted current18 fixtures")
    directory.mkdir(parents=True)
    origin = time.perf_counter()

    def record(event, value):
        append_event(directory/"events.jsonl", event, dict(elapsed=time.perf_counter()-origin, data=value))
        probe.emit(event, value)

    result = dict(kind="sync_baseline", python=sys.executable, source_hashes=hashes, identity=identity,
                  iteration=checkpoint["train_state"]["iteration"], config=asdict(config), groups=[], snapshots=[],
                  precision_changed=False, production_training=False, cuda_events=False, profiler=False,
                  limitations=["Fixed roots1:1 ordinary/strong, not full games or production7:1 mix",
                               "Inclusive stage/thread timers are not pure GPU or PCIe cost",
                               "Snapshots are outside timed cohorts, not thermal/power histories or controls",
                               "Finite exact gates do not prove all future inputs; no wait-mode implementation"])
    result["snapshots"].append(dict(stage="before_audits", gpu=gpu_snapshot()))
    record("start", dict(output=str(directory), sessions=2, roots_per_cohort=36, modes=MODES, repeats=6))
    result["graph_checks"] = graph_checks(checkpoint["model"], config, fixtures)
    for session in range(2):
        result["snapshots"].append(dict(stage=f"before_pool_{session}", gpu=gpu_snapshot()))
        with futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                initializer=init_worker, initargs=(checkpoint["model"], asdict(config), identity)) as pool:
            for mode in MODES:
                rows = list(pool.map(root_task, [(mode, f, strong, 202610101+f["metadata"]["index"])
                    for f in fixtures for strong in (False, True)]))
                pids = sorted({row["pid"] for row in rows})
                if len(pids) != 8:
                    raise RuntimeError("warmup did not exercise all eight workers")
                record("warm", dict(session=session, mode=mode, pids=pids))
            for repeat, order in enumerate(mode_orders(session)):
                paired = {}
                for mode in order:
                    tasks = [(mode, f, strong, 202610110+repeat*1000+f["metadata"]["index"])
                             for f in fixtures for strong in (False, True)]
                    start = time.perf_counter()
                    rows = list(pool.map(root_task, tasks))
                    group = dict(session=session, repeat=repeat, mode=mode,
                                 wall_seconds=time.perf_counter()-start, rows=rows)
                    result["groups"].append(group)
                    paired[mode] = exact_rows(rows)
                    if repeat < 4:
                        base = next(g for g in reference["groups"] if g["repeat"] == repeat and g["mode"] == "nvtx")
                        if paired[mode] != exact_rows(base["rows"]):
                            raise AssertionError("roots differ from frozen v5 policy/RNG/counters")
                    record("cohort", {k:v for k,v in group.items() if k != "rows"})
                if not all(values == paired["plain"] for values in paired.values()):
                    raise AssertionError("measurement mode changed policy/RNG/counters")
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
