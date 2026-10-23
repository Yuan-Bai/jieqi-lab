"""Prepare exact NVTX labels, or run a bounded frozen-root native profiling workload."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict
from functools import wraps
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import azero as az
import jieqi as jq
import run_records as rr
from tools import profile_selfplay as probe
from tools.benchmark_chance_allocation import NetworkTrace, canonical_graph
from tools.measure_search_reuse import verify_hashes
from tools.measure_selfplay_attribution import instrument_evaluation, paired_overhead, validate_config
from tools.measure_selfplay_round import append_event, runtime_identity

CAPTURE_RANGE = "jieqi_native_window"
OBSERVER = None


class NvtxLabels:
    """Labels only: no timers, CUDA events, extra synchronization or tensor copies."""

    def __init__(self):
        self.functions = {}

    @contextmanager
    def scope(self, name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def call(self, name, function, *args, **kwargs):
        with self.scope("jieqi/" + name):
            return function(*args, **kwargs)

    def install(self, stack):
        if not self.functions:
            for attribute in ("_evaluate", "_evaluate_batch"):
                original = getattr(az.NeuralMCTS, attribute)
                labelled = instrument_evaluation(original, self)

                def wrapper(function):
                    @wraps(function)
                    def evaluated(*args, **kwargs):
                        return self.call("evaluation", function, *args, **kwargs)
                    return evaluated

                self.functions[attribute] = wrapper(labelled)
        for attribute, function in self.functions.items():
            stack.enter_context(mock.patch.object(az.NeuralMCTS, attribute, function))


def init_worker(weights, config, expected):
    global OBSERVER
    if runtime_identity() != expected:
        raise RuntimeError("worker backend/precision identity differs")
    probe.init_worker(weights, config, jq.RULES_BACKEND_NATIVE,
        (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE), (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))
    OBSERVER = NvtxLabels()


def root_task(task):
    mode, fixture, strong, seed = task
    with ExitStack() as stack:
        if mode == "nvtx":
            OBSERVER.install(stack)
            stack.enter_context(OBSERVER.scope("jieqi/root/" + ("3200" if strong else "800")))
        elif mode != "plain":
            raise ValueError("unknown mode")
        row = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    row.update(mode=mode, pid=os.getpid(), native_thread_id=threading.get_native_id())
    return row


def graph_checks(weights, config, fixtures):
    net = az.PolicyValueNet().to("cuda").eval()
    az.load_state_dict_padded(net, weights)
    original, rows, observer = az.NeuralMCTS.search, [], NvtxLabels()
    for fixture in fixtures:
        for strong in (False, True):
            outputs = []
            for mode in ("plain", "nvtx"):
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
                        if mode == "nvtx":
                            observer.install(stack)
                        row = probe.run_case(net, config, torch.device("cuda"), fixture, strong,
                                             202610084 + fixture["metadata"]["index"], "plain")
                    outputs.append(dict(exact=row["exact"], graph=canonical_graph(retained[0]), network=trace.rows))
                finally:
                    hook.remove()
            if outputs[0] != outputs[1]:
                raise AssertionError("NVTX labels changed graph/network/strategy/RNG")
            rows.append(dict(index=fixture["metadata"]["index"], strong=strong,
                             network_requests=len(outputs[0]["network"]), exact=True))
            probe.emit("graph_exact", rows[-1])
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def validate_output(directory, source):
    pilots = (ROOT / "data/pilots").resolve()
    if directory.exists() or pilots not in directory.parents or directory == source or source in directory.parents:
        raise ValueError("require a new directory below data/pilots, outside frozen inputs")


def validate_reference(reference):
    if (reference.get("kind") != "native_profile_preparation" or not reference.get("production_unchanged")
            or len(reference.get("graph_checks", [])) != 36
            or not all(row.get("exact") for row in reference["graph_checks"])
            or not reference.get("paired_roots_exact") or reference.get("repeats") != 4):
        raise ValueError("require a completed four-cohort NVTX preparation with all exact gates")
    verify_hashes(reference["source_hashes"])


def exact_rows(rows):
    # A persisted Move tuple becomes a JSON list; compare at the same serialization boundary.
    return json.loads(json.dumps([row["exact"] for row in rows], allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, help="Capture only; consume a successful preparation")
    args = parser.parse_args()
    source = ROOT / "data/pilots/chance_support_20261008_iter28/frozen_inputs"
    fixture_path = ROOT / "data/pilots/chance_support_20261008_iter28/measure.fixtures.pkl"
    directory = args.output_dir.resolve()
    try:
        validate_output(directory, source.resolve())
    except ValueError as error:
        parser.error(str(error))
    if (jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython" or az.SEARCH_BACKEND != "python"
            or not torch.cuda.is_available()):
        parser.error("require CUDA, rules/legal Cython and Python nodes")
    torch.set_num_threads(1)
    reference = None
    if args.reference_file:
        with args.reference_file.open(encoding="utf-8") as handle:
            reference = json.load(handle)
        validate_reference(reference)
        if runtime_identity() != reference["identity"]:
            # JSON turns native identity tuples into lists.
            if json.loads(json.dumps(runtime_identity())) != reference["identity"]:
                raise RuntimeError("reference backend/precision identity differs")
    checkpoint_path = source / "models/training_checkpoint.pt"
    before = rr.file_record(checkpoint_path)["sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if rr.file_record(checkpoint_path)["sha256"] != before:
        raise RuntimeError("frozen checkpoint changed while loading")
    config = az.AZConfig(**checkpoint["config"])
    validate_config(config)
    config.data_dir, config.model_dir, config.log_dir = (str(directory / n) for n in ("samples", "models", "logs"))
    with fixture_path.open("rb") as handle:
        fixtures = pickle.load(handle)["fixtures"]
    if len(fixtures) != 18 or [f["metadata"]["index"] for f in fixtures] != list(range(18)):
        parser.error("require the trusted current 18-position fixtures")
    production = ROOT / "data/train_rules5_20261004"
    paths = [production / n for n in ("models/training_checkpoint.pt", "models/latest.pt",
                                     "models/train_state.json", "logs/train.log",
                                     "selfplay/iter_20261008_080731_0027.pkl.gz")]
    paths += [checkpoint_path, fixture_path]
    paths += [ROOT / n for n in ("azero.py", "ismcts.py", "jieqi.py", "run_records.py", "native/jieqi_rules.pyx",
        "native/jieqi_legal.pyx", "tools/profile_selfplay.py", "tools/benchmark_chance_allocation.py",
        "tools/measure_search_reuse.py", "tools/measure_selfplay_attribution.py", "tools/measure_selfplay_round.py",
        "tools/profile_native_selfplay.py", "tools/capture_native_selfplay.ps1", "tools/native_profile_process.ps1")]
    paths += [Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"]), Path(sys.executable)]
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    if hashes[str(checkpoint_path)] != before or (reference and hashes != reference["source_hashes"]):
        raise RuntimeError("protected sources differ from the prepared workload")
    directory.mkdir(parents=True)
    origin = time.perf_counter()

    def record(event, value):
        append_event(directory / "events.jsonl", event, dict(elapsed=time.perf_counter()-origin, data=value))
        probe.emit(event, value)

    result = dict(kind="native_profile_capture" if reference else "native_profile_preparation",
        pid=os.getpid(), native_thread_id=threading.get_native_id(), python=sys.executable,
        source_hashes=hashes, identity=runtime_identity(), iteration=checkpoint["train_state"]["iteration"],
        torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        repeats=4, workers=8, config=asdict(config), capture_range=CAPTURE_RANGE, groups=[],
        limitations=["Fixed roots have a 1:1 ordinary/strong count, not the production 7:1 game mix",
            "No full games/sample IPC, 4M replay residency, optimizer or production writes",
            "Capture/reference timing difference includes separate-run machine noise, not pure profiler overhead",
            "NVTX labels retain original calls; no additional phase synchronization or CUDA timing events",
            "Native symbols/child CUDA coverage and dropped events must be checked in the actual report"])
    record("start", {k:v for k,v in result.items() if k not in ("source_hashes", "config", "groups")})
    if not reference:
        result["graph_checks"] = graph_checks(checkpoint["model"], config, fixtures)
    observer = NvtxLabels()
    with futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(checkpoint["model"], asdict(config), runtime_identity())) as pool:
        for mode in ("plain", "nvtx"):
            warm = list(pool.map(root_task, [(mode, f, strong, 202610101+f["metadata"]["index"])
                                            for f in fixtures for strong in (False, True)]))
            pids = sorted({r["pid"] for r in warm})
            if len(pids) != 8:
                raise RuntimeError("warmup did not exercise all eight workers")
            record("warm", dict(mode=mode, pids=pids))
        # One parent trigger encloses all worker cohorts; never open it during audits or warmup.
        window = observer.scope(CAPTURE_RANGE) if reference else nullcontext()
        with window:
            for repeat in range(4):
                modes = ("nvtx",) if reference else (("plain", "nvtx") if repeat % 2 == 0 else ("nvtx", "plain"))
                paired = {}
                for mode in modes:
                    tasks = [(mode, f, strong, 202610110+repeat*1000+f["metadata"]["index"])
                             for f in fixtures for strong in (False, True)]
                    start = time.perf_counter()
                    rows = list(pool.map(root_task, tasks))
                    group = dict(repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-start, rows=rows)
                    result["groups"].append(group)
                    paired[mode] = exact_rows(rows)
                    if reference:
                        expected = next(g for g in reference["groups"] if g["repeat"] == repeat and g["mode"] == "nvtx")
                        if paired[mode] != exact_rows(expected["rows"]):
                            record("exact_failure", dict(repeat=repeat, actual=paired[mode], expected=exact_rows(expected["rows"])))
                            raise AssertionError("native profiler changed policy/RNG/search counters")
                    record("cohort", {k:v for k,v in group.items() if k != "rows"})
                if not reference and paired["plain"] != paired["nvtx"]:
                    raise AssertionError("NVTX labels changed policy/RNG/search counters")
        record("capture_window_closed", dict(groups=len(result["groups"])))
        record("pool_shutdown_begin", {})
    record("pool_shutdown_complete", {})
    if not reference:
        result["label_overhead"] = paired_overhead([
            dict(g, mode="stages" if g["mode"] == "nvtx" else "plain") for g in result["groups"]])
    else:
        base_seconds = sum(g["wall_seconds"] for g in reference["groups"] if g["mode"] == "nvtx")
        result["capture_vs_preparation_wall_change"] = sum(g["wall_seconds"] for g in result["groups"])/base_seconds-1
        result["reference_file"] = rr.file_record(args.reference_file.resolve())
    verify_hashes(hashes)
    result.update(production_unchanged=True, paired_roots_exact=True, elapsed_seconds=time.perf_counter()-origin)
    pending = directory / "result.partial.json"
    with pending.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    pending.rename(directory / "result.json")
    record("complete", dict(output=str(directory), exact=True, production_unchanged=True,
                            elapsed_seconds=result["elapsed_seconds"]))


if __name__ == "__main__":
    main()
