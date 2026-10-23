"""Build and screen one isolated, bit-exact Cython encoder candidate."""
from __future__ import annotations

import argparse
from collections import deque
import concurrent.futures as futures
from contextlib import ExitStack
import copy
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import json
import marshal
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import statistics
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
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

MODES = ("plain", "compiled")
ENCODER = None
ORIGINAL = az._encode_state_into


def code_fingerprint():
    functions = (ORIGINAL, az._piece_plane_index, az.jq.is_dark, az.jq.piece_color, az.jq.piece_type)
    return hashlib.sha256(marshal.dumps(tuple(f.__code__ for f in functions))).hexdigest()


def load_encoder(directory):
    build = json.loads((directory/"build.json").read_text(encoding="utf-8"))
    verify_hashes(build["source_hashes"])
    path = Path(build["extension"]).resolve()
    if directory.resolve() not in path.parents or path.suffix not in (".pyd", ".so"):
        raise ValueError("require isolated extension below its build directory")
    if rr.file_record(path)["sha256"] != build["extension_sha256"]:
        raise ValueError("experimental binary changed")
    name = "_jieqi_encoder_experiment"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    if (Path(module.__file__).resolve() != path or module.API_VERSION != 1 or
            module.SOURCE_SHA256 != build["source_hashes"][str(ROOT/"native/experimental_encoder.pyx")] or
            module.CORE_SHA256 != build["source_hashes"][str(ROOT/"azero.py")]):
        raise ImportError("experimental source/binary identity mismatch")
    module.initialize(az, code_fingerprint())
    return module


def build_extension(directory):
    source = ROOT/"native/experimental_encoder.pyx"
    sys.path.insert(0, str(ROOT/"data/build_deps"))
    from Cython.Build import cythonize
    import Cython
    from setuptools import Extension, setup
    hashes = {str(p): rr.file_record(p)["sha256"] for p in
              (source, ROOT/"azero.py", ROOT/"jieqi.py", Path(sys.executable))}
    shutil.copyfile(source, directory/"source_snapshot.pyx")
    flags = ["/fp:strict"] if os.name == "nt" else ["-fno-fast-math", "-ffp-contract=off"]
    extensions = cythonize([Extension("_jieqi_encoder_experiment", [str(source)], extra_compile_args=flags)],
        build_dir=str(directory/"cython"), compile_time_env=dict(NATIVE_SOURCE_SHA256=hashes[str(source)],
            NATIVE_CORE_SHA256=hashes[str(ROOT/"azero.py")], NATIVE_CODE_SHA256=code_fingerprint()),
        compiler_directives=dict(language_level=3, boundscheck=True, wraparound=True), force=True)
    setup(name="jieqi-encoder-experiment", ext_modules=extensions,
          script_args=["build_ext", "--force", "--build-temp", str(directory/"temp"),
                       "--build-lib", str(directory/"lib")])
    paths = list((directory/"lib").glob("_jieqi_encoder_experiment.*"))
    if len(paths) != 1:
        raise RuntimeError("require one isolated extension")
    verify_hashes(hashes)
    result = dict(kind="exact_encoder_build", extension=str(paths[0]), extension_sha256=rr.file_record(paths[0])["sha256"],
                  source_hashes=hashes, cython=Cython.__version__, compiler_flags=flags,
                  python=sys.executable, code_fingerprint=code_fingerprint(), production_extension_rebuilt=False)
    with (directory/"build.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def paired_micro(cases, encoder):
    before = probe.digest(cases)
    fast_cases = 0
    for info, history, entries in cases:
        a = np.zeros((az.STATE_CHANNELS, az.BOARD_H, az.BOARD_W), np.float32)
        b = np.zeros_like(a)
        ORIGINAL(info, history, a, repetition_entries=entries)
        encoder.encode_into(info, history, b, repetition_entries=entries)
        fast_cases += int(encoder.supports(info, b) and encoder._supported_history(history)
                          and encoder._supported_entries(entries))
        if a.tobytes() != b.tobytes():
            raise AssertionError("captured encoding bits differ")
    trials = []
    planes = np.zeros((az.STATE_CHANNELS, az.BOARD_H, az.BOARD_W), np.float32)
    for repeat in range(8):
        for mode in MODES if repeat % 2 == 0 else tuple(reversed(MODES)):
            function = ORIGINAL if mode == "plain" else encoder.encode_into
            start = time.perf_counter()
            for _ in range(8):
                for info, history, entries in cases:
                    function(info, history, planes, repetition_entries=entries)
            trials.append(dict(repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-start))
    if probe.digest(cases) != before:
        raise AssertionError("encoder mutated captured inputs")
    plain = sum(t["wall_seconds"] for t in trials if t["mode"] == "plain")
    compiled = sum(t["wall_seconds"] for t in trials if t["mode"] == "compiled")
    gain = plain/compiled-1
    return dict(cases=len(cases), fast_cases=fast_cases, calls_per_mode=len(cases)*64, trials=trials,
                throughput_gain=gain, wall_saving=1-compiled/plain, micro_gate_passed=gain >= 0.10,
                allocation_included=False, inputs_unchanged=True, bits_exact=True)


def audit_graphs(net, config, fixtures, encoder, record):
    original_search, rows, cases = az.NeuralMCTS.search, [], []
    for fixture in fixtures:
        for strong in (False, True):
            signatures, encoded_calls, fast_inputs = [], 0, 0
            first, last = [], deque(maxlen=8)
            for mode in MODES:
                trace, retained = NetworkTrace(), []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original_search(search, *args, **kwargs)
                    retained.append(root)
                    return policy, move

                def checked(info, history, planes, *, repetition_entries=az._REPETITION_ENTRIES_UNSET):
                    nonlocal encoded_calls, fast_inputs
                    expected = np.zeros_like(planes)
                    ORIGINAL(info, history, expected, repetition_entries=repetition_entries)
                    encoder.encode_into(info, history, planes, repetition_entries=repetition_entries)
                    fast_inputs += int(encoder.supports(info, planes) and encoder._supported_history(history)
                                       and encoder._supported_entries(repetition_entries))
                    if expected.tobytes() != planes.tobytes():
                        raise AssertionError("live encoded input bits differ")
                    encoded_calls += 1
                    entry = (info, history, repetition_entries)
                    if len(first) < 8:
                        first.append(entry)
                    last.append(entry)

                hook = net.register_forward_hook(trace.hook)
                try:
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(az.NeuralMCTS, "search", capture))
                        if mode == "compiled":
                            stack.enter_context(mock.patch.object(az, "_encode_state_into", checked))
                        result = probe.run_case(net, config, probe.DEVICE, fixture, strong,
                                               202610084+fixture["metadata"]["index"], "plain")
                    signatures.append(dict(exact=result["exact"], graph=canonical_graph(retained[0]), network=trace.rows))
                finally:
                    hook.remove()
            if signatures[0] != signatures[1] or encoded_calls < 1:
                raise AssertionError("compiled encoding changed network/graph/strategy/RNG or missed encodes")
            # Entries are explicit (not the identity sentinel); ownership is frozen before later micro replay.
            for info, history, entries in first+list(last):
                actual = az._repetition_entries(info) if entries is az._REPETITION_ENTRIES_UNSET else entries
                cases.append(copy.deepcopy((info, history, actual)))
            row = dict(index=fixture["metadata"]["index"], strong=strong, encoded_inputs_checked=encoded_calls,
                       fast_inputs=fast_inputs,
                       network_requests=len(signatures[0]["network"]), graph=signatures[0]["graph"],
                       network_sha256=probe.digest(signatures[0]["network"]), exact=True)
            rows.append(row)
            record("graph_exact", row)
    return rows, cases


def init_worker(weights, config, expected, directory):
    global ENCODER
    native.init_worker(weights, config, expected)
    ENCODER = load_encoder(Path(directory))


def root_task(task):
    mode, fixture, strong, seed = task
    if mode not in MODES:
        raise ValueError("unknown encoder mode")
    with ExitStack() as stack:
        if mode == "compiled":
            stack.enter_context(mock.patch.object(az, "_encode_state_into", ENCODER.encode_into))
        row = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    row.update(mode=mode, pid=os.getpid())
    return row


def summarize_roots(groups):
    keys = {(session, repeat, mode) for session in range(2) for repeat in range(4) for mode in MODES}
    by_key = {(g["session"], g["repeat"], g["mode"]): g for g in groups}
    if len(groups) != len(keys) or set(by_key) != keys:
        raise ValueError("require two pools with four balanced paired cohorts")
    pools = []
    fixture_keys = [(i, strong) for i in range(18) for strong in (False, True)]
    for session in range(2):
        times = {mode: [] for mode in MODES}
        gains = []
        for repeat in range(4):
            pair = [by_key[session, repeat, mode] for mode in MODES]
            for mode, group in zip(MODES, pair):
                if group["wall_seconds"] <= 0 or [(r["index"], r["strong"]) for r in group["rows"]] != fixture_keys:
                    raise ValueError("require ordered complete positive-duration 36-root cohorts")
                times[mode].append(group["wall_seconds"])
            if exact_rows(pair[0]["rows"]) != exact_rows(pair[1]["rows"]):
                raise AssertionError("encoding changed exact root outputs")
            gains.append(pair[0]["wall_seconds"]/pair[1]["wall_seconds"]-1)
        plain = times["plain"]
        pools.append(dict(session=session, seconds=times, plain_cv=statistics.stdev(plain)/statistics.mean(plain),
            plain_median=statistics.median(plain), plain_drift=statistics.mean(plain[2:])/statistics.mean(plain[:2])-1,
            throughput_gain=sum(plain)/sum(times["compiled"])-1, paired_gains=gains,
            winning_pairs=sum(g > 0 for g in gains)))
    between = pools[1]["plain_median"]/pools[0]["plain_median"]-1
    stable = abs(between) <= .10 and all(p["plain_cv"] <= .05 and abs(p["plain_drift"]) <= .10 for p in pools)
    totals = {mode: sum(g["wall_seconds"] for g in groups if g["mode"] == mode) for mode in MODES}
    gain = totals["plain"]/totals["compiled"]-1
    gate = stable and gain >= .03 and all(p["throughput_gain"] >= .01 and p["winning_pairs"] >= 3 for p in pools)
    return dict(pools=pools, totals=totals, aggregate_throughput_gain=gain,
                aggregate_wall_saving=1-totals["compiled"]/totals["plain"], between_pool_change=between,
                stable_baseline=stable, roots_exact=True, full_game_screen_gate_passed=gate,
                production_promotion=False, thresholds=dict(aggregate_gain=.03,each_pool_gain=.01,
                each_pool_wins=3,plain_cv=.05,plain_drift=.10,between_pool_difference=.10))


def screen(directory, build_dir, reference_path):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference.get("kind") != "python_sampling_workload" or not reference.get("production_unchanged")
            or not reference.get("exact_against_baseline")):
        raise ValueError("require completed exact current Python sampling plain reference")
    hashes = dict(reference["source_hashes"])
    verify_hashes(hashes)
    if (json.loads(json.dumps(runtime_identity())) != reference["identity"] or
            Path(sys.executable).resolve() != Path(reference["python"]).resolve() or not torch.cuda.is_available()):
        raise ValueError("require identical interpreter/CUDA/backend/precision")
    build = json.loads((build_dir/"build.json").read_text(encoding="utf-8"))
    for path in (reference_path, Path(__file__).resolve(), ROOT/"native/experimental_encoder.pyx",
                 ROOT/"tests/test_exact_encoder_experiment.py", build_dir/"build.json", Path(build["extension"]),
                 ROOT/"data/pilots/exact_encoder_20261009_iter28_protocol.md"):
        hashes[str(path)] = rr.file_record(path)["sha256"]
    encoder = load_encoder(build_dir)
    torch.set_num_threads(1)
    source = ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs"
    checkpoint = torch.load(source/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
    config = az.AZConfig(**checkpoint["config"])
    validate_config(config)
    config.data_dir, config.model_dir, config.log_dir = (str(directory/n) for n in ("samples", "models", "logs"))
    with (source.parent/"measure.fixtures.pkl").open("rb") as handle:
        fixtures = pickle.load(handle)["fixtures"]
    if len(fixtures) != 18 or [f["metadata"]["index"] for f in fixtures] != list(range(18)):
        raise ValueError("require trusted current18 fixtures")
    start = time.perf_counter()

    def record(event, data):
        append_event(directory/"events.jsonl", event, dict(elapsed=time.perf_counter()-start, data=data))
        probe.emit(event, data)

    result = dict(kind="exact_encoder_root_screen", source_hashes=hashes, identity=runtime_identity(),
                  python=sys.executable, config=asdict(config), groups=[], production_training=False,
                  snapshots=[gpu_snapshot()], compiler_build=build)
    net = az.PolicyValueNet().to("cuda").eval()
    az.load_state_dict_padded(net, checkpoint["model"])
    result["graph_checks"], cases = audit_graphs(net, config, fixtures, encoder, record)
    with (directory/"micro_cases.pkl").open("xb") as handle:
        pickle.dump(cases, handle)
    hashes[str(directory/"micro_cases.pkl")] = rr.file_record(directory/"micro_cases.pkl")["sha256"]
    del net
    gc.collect()
    torch.cuda.empty_cache()
    result["micro"] = paired_micro(cases, encoder)
    record("micro", result["micro"])
    if result["micro"]["micro_gate_passed"]:
        for session in range(2):
            result["snapshots"].append(gpu_snapshot())
            with futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                    initializer=init_worker, initargs=(checkpoint["model"], asdict(config), runtime_identity(), str(build_dir))) as pool:
                for mode in MODES if session == 0 else tuple(reversed(MODES)):
                    warm = list(pool.map(root_task, [(mode, f, strong, 202610101+f["metadata"]["index"])
                                                    for f in fixtures for strong in (False, True)]))
                    if len({r["pid"] for r in warm}) != 8:
                        raise RuntimeError("warmup missed workers")
                    record("warm", dict(session=session, mode=mode))
                for repeat in range(4):
                    for mode in MODES if (repeat+session)%2 == 0 else tuple(reversed(MODES)):
                        tasks = [(mode, f, strong, 202610110+repeat*1000+f["metadata"]["index"])
                                 for f in fixtures for strong in (False, True)]
                        begin = time.perf_counter()
                        rows = list(pool.map(root_task, tasks))
                        group = dict(session=session, repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-begin, rows=rows)
                        if exact_rows(rows) != exact_rows(reference["groups"][repeat]["rows"]):
                            raise AssertionError("candidate roots differ from frozen reference")
                        result["groups"].append(group)
                        record("cohort", {k:v for k,v in group.items() if k != "rows"})
            record("pool_shutdown", dict(session=session))
        result["summary"] = summarize_roots(result["groups"])
    else:
        result["summary"] = dict(full_game_screen_gate_passed=False, production_promotion=False,
                                 reason="local encoder throughput gain below predeclared 10% screen")
    verify_hashes(hashes)
    result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-start)
    with (directory/"result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    record("complete", result["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build", "screen"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--reference-file", type=Path)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    native.validate_output(directory, (ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs").resolve())
    if args.phase == "screen" and (not args.build_dir or not args.reference_file):
        parser.error("screen requires build directory and exact reference")
    directory.mkdir(parents=True)
    if args.phase == "build":
        print(json.dumps(build_extension(directory)), flush=True)
    else:
        screen(directory, args.build_dir.resolve(), args.reference_file.resolve())


if __name__ == "__main__":
    main()
