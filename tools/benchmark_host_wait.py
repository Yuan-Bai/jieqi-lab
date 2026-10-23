"""Isolated blocking-event host-wait screening; never patch production on disk."""
from __future__ import annotations

import __future__
import argparse
import ast
import concurrent.futures as futures
from contextlib import ExitStack
from dataclasses import asdict
import gc
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import textwrap
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil
import torch
import azero as az
import run_records as rr
from tools import profile_native_selfplay as native
from tools import profile_selfplay as probe
from tools.benchmark_chance_allocation import NetworkTrace, canonical_graph
from tools.measure_search_reuse import verify_hashes
from tools.measure_selfplay_attribution import validate_config
from tools.measure_selfplay_round import append_event, runtime_identity
from tools.measure_sync_baseline import exact_rows, gpu_snapshot

MODES = ("plain", "blocking")
WAITER = None


def instrument_policy_cpu(function, waiter):
    """Replace only the bound policy cpu() call; fail closed on source drift."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    replaced = []

    class Calls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "cpu":
                if node.args or node.keywords:
                    raise ValueError("expected the original argument-free policy cpu()")
                replaced.append(node)
                return ast.copy_location(ast.Call(
                    func=ast.Attribute(value=ast.Name(id="_host_waiter", ctx=ast.Load()),
                                       attr="copy_policy", ctx=ast.Load()),
                    args=[node.func], keywords=[]), node)
            return node

    tree = Calls().visit(tree)
    if len(replaced) != 1:
        raise ValueError("evaluation changed: require exactly one policy cpu()")
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__, _host_waiter=waiter)
    exec(compile(tree, inspect.getsourcefile(function), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[function.__name__]


class BlockingPolicyWait:
    def __init__(self):
        self.functions, self.events = {}, {}
        self.calls = 0

    def copy_policy(self, cpu):
        tensor = cpu.__self__
        if tensor.is_cuda:
            device = tensor.device
            if device not in self.events:
                with torch.cuda.device(device):
                    self.events[device] = torch.cuda.Event(enable_timing=False, blocking=True)
            event = self.events[device]
            event.record(torch.cuda.current_stream(device))
            event.synchronize()
            self.calls += 1
        return cpu()

    def install(self, stack):
        if not self.functions:
            for name in ("_evaluate", "_evaluate_batch"):
                self.functions[name] = instrument_policy_cpu(getattr(az.NeuralMCTS, name), self)
        for name, function in self.functions.items():
            stack.enter_context(mock.patch.object(az.NeuralMCTS, name, function))


def mode_orders(session):
    return [MODES if (repeat+session) % 2 == 0 else tuple(reversed(MODES)) for repeat in range(4)]


def init_worker(weights, config, expected):
    global WAITER
    native.init_worker(weights, config, expected)
    WAITER = BlockingPolicyWait()


def root_task(task):
    mode, fixture, strong, seed = task
    if mode not in MODES:
        raise ValueError("unknown host-wait mode")
    before, cpu = WAITER.calls, time.thread_time()
    with ExitStack() as stack:
        if mode == "blocking":
            WAITER.install(stack)
        row = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    row.update(mode=mode, pid=os.getpid(), event_waits=WAITER.calls-before,
               thread_cpu_seconds=time.thread_time()-cpu)
    if (mode == "blocking" and row["event_waits"] < 1) or (mode == "plain" and row["event_waits"]):
        raise AssertionError("incorrect event coverage")
    return row


def graph_checks(weights, config, fixtures, record):
    net = az.PolicyValueNet().to("cuda").eval()
    az.load_state_dict_padded(net, weights)
    original, results, waiter = az.NeuralMCTS.search, [], BlockingPolicyWait()
    for fixture in fixtures:
        for strong in (False, True):
            outputs, waits = [], []
            for mode in MODES:
                trace, retained = NetworkTrace(), []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original(search, *args, **kwargs)
                    retained.append(root)
                    return policy, move

                before = waiter.calls
                hook = net.register_forward_hook(trace.hook)
                try:
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(az.NeuralMCTS, "search", capture))
                        if mode == "blocking":
                            waiter.install(stack)
                        row = probe.run_case(net, config, torch.device("cuda"), fixture, strong,
                                             202610084+fixture["metadata"]["index"], "plain")
                    outputs.append(dict(exact=row["exact"], graph=canonical_graph(retained[0]), network=trace.rows))
                    waits.append(waiter.calls-before)
                finally:
                    hook.remove()
            if outputs[0] != outputs[1] or waits != [0, len(outputs[0]["network"])]:
                raise AssertionError("blocking wait changed network/graph/policy/RNG or missed evaluation")
            result = dict(index=fixture["metadata"]["index"], strong=strong,
                          network_requests=len(outputs[0]["network"]), event_waits=waits[1], exact=True)
            results.append(result)
            record("graph_exact", result)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return results


def summarize(groups):
    expected_keys = {(session, repeat, mode) for session in range(2) for repeat in range(4) for mode in MODES}
    if len(groups) != len(expected_keys) or {(g["session"], g["repeat"], g["mode"]) for g in groups} != expected_keys:
        raise ValueError("require two independent pools and four balanced A/B pairs each")
    by_key = {(g["session"], g["repeat"], g["mode"]): g for g in groups}
    pools = []
    fixture_keys = [(index, strong) for index in range(18) for strong in (False, True)]
    for session in range(2):
        plain, blocking, pairs, cpu = [], [], [], {mode: 0.0 for mode in MODES}
        for repeat in range(4):
            base = by_key[session, repeat, "plain"]
            candidate = by_key[session, repeat, "blocking"]
            for mode, group in (("plain", base), ("blocking", candidate)):
                rows = group["rows"]
                if [(r["index"], r["strong"]) for r in rows] != fixture_keys or group["wall_seconds"] <= 0:
                    raise ValueError("require ordered complete 36-root cohorts and positive times")
                if any((r["event_waits"] < 1 if mode == "blocking" else r["event_waits"] != 0) for r in rows):
                    raise AssertionError("incorrect timed event coverage")
                cpu[mode] += sum(r["process_cpu_seconds"] for r in rows)
            if exact_rows(candidate["rows"]) != exact_rows(base["rows"]):
                raise AssertionError("wait mode changed policy/RNG/counters")
            if exact_rows(by_key[1-session, repeat, "plain"]["rows"]) != exact_rows(base["rows"]):
                raise AssertionError("independent pools changed policy/RNG/counters")
            plain.append(base["wall_seconds"])
            blocking.append(candidate["wall_seconds"])
            pairs.append(base["wall_seconds"]/candidate["wall_seconds"]-1)
        pools.append(dict(session=session, plain_seconds=plain, blocking_seconds=blocking,
                          plain_median=statistics.median(plain), plain_cv=statistics.stdev(plain)/statistics.mean(plain),
                          plain_late_vs_early=statistics.mean(plain[2:])/statistics.mean(plain[:2])-1,
                          throughput_gain=sum(plain)/sum(blocking)-1, paired_gains=pairs,
                          winning_pairs=sum(gain > 0 for gain in pairs), worker_cpu_seconds=cpu,
                          worker_cpu_change=cpu["blocking"]/cpu["plain"]-1))
    between = pools[1]["plain_median"]/pools[0]["plain_median"]-1
    stable = all(p["plain_cv"] <= 0.05 and abs(p["plain_late_vs_early"]) <= 0.10 for p in pools) and abs(between) <= 0.10
    total_plain = sum(g["wall_seconds"] for g in groups if g["mode"] == "plain")
    total_blocking = sum(g["wall_seconds"] for g in groups if g["mode"] == "blocking")
    gain = total_plain/total_blocking-1
    gate = stable and gain >= 0.03 and all(p["throughput_gain"] >= 0.01 and p["winning_pairs"] >= 3 for p in pools)
    return dict(pools=pools, plain_cohort_seconds=total_plain, blocking_cohort_seconds=total_blocking,
                aggregate_throughput_gain=gain, between_pool_plain_median_change=between,
                stable_baseline=stable, paired_roots_exact=True, full_game_gate_passed=gate,
                thresholds=dict(plain_cv=0.05, plain_drift=0.10, between_pool_difference=0.10,
                                aggregate_gain_minimum=0.03, each_pool_gain_minimum=0.01,
                                each_pool_winning_pairs_minimum=3))


def validate_baseline(reference, reference_path):
    if (reference.get("kind") != "sync_baseline" or not reference.get("production_unchanged")
            or not reference.get("summary", {}).get("host_wait_experiment_gate_passed")
            or len(reference.get("graph_checks", [])) != 36
            or not all(r.get("exact") for r in reference["graph_checks"])
            or not reference.get("summary", {}).get("paired_roots_exact")):
        raise ValueError("require a completed exact stable no-tracer baseline")
    verify_hashes(reference["source_hashes"])
    if (json.loads(json.dumps(runtime_identity())) != reference["identity"]
            or Path(sys.executable).resolve() != Path(reference["python"]).resolve()
            or not torch.cuda.is_available()):
        raise ValueError("require identical interpreter/CUDA/backend/precision")
    hashes = dict(reference["source_hashes"])
    for path in (reference_path, Path(__file__).resolve(), ROOT/"tests/test_benchmark_host_wait.py",
                 ROOT/"data/pilots/host_wait_20261009_iter28_protocol.md"):
        hashes[str(path)] = rr.file_record(path)["sha256"]
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    args = parser.parse_args()
    directory, reference_path = args.output_dir.resolve(), args.reference_file.resolve()
    source = ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs"
    fixture_path = ROOT/"data/pilots/chance_support_20261008_iter28/measure.fixtures.pkl"
    native.validate_output(directory, source.resolve())
    if (ROOT/"data/pilots").resolve() not in reference_path.parents:
        parser.error("reference must be below workspace data/pilots")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    hashes = validate_baseline(reference, reference_path)
    if any(p.info["pid"] != os.getpid() and (p.info["name"] or "").lower() in ("python.exe", "pythonw.exe", "nsys.exe")
           for p in psutil.process_iter(["pid", "name"])):
        parser.error("finish other Python/Nsight work before isolated measurement")
    wpr = subprocess.run([str(Path(os.environ["SystemRoot"])/"System32/wpr.exe"), "-status"],
                         capture_output=True, text=True, timeout=10)
    if wpr.returncode or "WPR is not recording" not in wpr.stdout+wpr.stderr:
        parser.error("cannot confirm WPR is not recording")
    torch.set_num_threads(1)
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

    result = dict(kind="host_wait_root_screen", python=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                  source_hashes=hashes, identity=runtime_identity(), iteration=checkpoint["train_state"]["iteration"],
                  config=asdict(config), groups=[], snapshots=[], production_training=False, precision_changed=False,
                  limitations=["Fixed roots1:1 ordinary/strong, not full games or production7:1 mix",
                               "Graph audit forward hooks synchronize; audit timings are not used for speed claims",
                               "Event only waits for queued compute before original blocking cpu(); D2H may still actively wait",
                               "Thread/process CPU counters are coarse, not pure driver/kernel time",
                               "Finite exact gates do not prove all future inputs; no system scheduling/power changes",
                               "Fixed-root gate only prioritizes full games; not evidence of production speedup"])
    record("start", dict(output=str(directory), workers=8, sessions=2, pairs_per_pool=4, modes=MODES))
    result["snapshots"].append(dict(stage="before_audits", gpu=gpu_snapshot()))
    result["graph_checks"] = graph_checks(checkpoint["model"], config, fixtures, record)
    if [dict(index=r["index"], strong=r["strong"], network_requests=r["network_requests"], exact=r["exact"])
            for r in result["graph_checks"]] != reference["graph_checks"]:
        raise AssertionError("audit network request counts differ from baseline")
    for session in range(2):
        result["snapshots"].append(dict(stage=f"before_pool_{session}", gpu=gpu_snapshot()))
        with futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                initializer=init_worker, initargs=(checkpoint["model"], asdict(config), runtime_identity())) as pool:
            for mode in MODES if session == 0 else tuple(reversed(MODES)):
                rows = list(pool.map(root_task, [(mode, f, strong, 202610101+f["metadata"]["index"])
                    for f in fixtures for strong in (False, True)]))
                pids = sorted({r["pid"] for r in rows})
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
                    base = next(g for g in reference["groups"] if g["session"] == session
                                and g["repeat"] == repeat and g["mode"] == "plain")
                    if paired[mode] != exact_rows(base["rows"]):
                        raise AssertionError("roots differ from frozen no-tracer baseline")
                    record("cohort", {k:v for k,v in group.items() if k != "rows"})
                if paired["plain"] != paired["blocking"]:
                    raise AssertionError("blocking wait changed root results")
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
