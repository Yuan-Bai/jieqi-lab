"""Build and screen one isolated repetition-board identity memo candidate."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from contextlib import ExitStack
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
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import azero as az
import run_records as rr
from tools import benchmark_exact_encoder as root_screen
from tools import profile_native_selfplay as native
from tools import profile_selfplay as probe
from tools.benchmark_chance_allocation import NetworkTrace, canonical_graph
from tools.measure_search_reuse import verify_hashes
from tools.measure_selfplay_attribution import validate_config
from tools.measure_selfplay_round import append_event, runtime_identity
from tools.measure_sync_baseline import exact_rows, gpu_snapshot

ORIGINAL = az._repetition_entries
MODES = ("plain", "compiled")
MEMO = None


def code_fingerprint():
    functions = (ORIGINAL, az._public_material_signature.__wrapped__, az.jq.is_dark)
    return hashlib.sha256(marshal.dumps(tuple(f.__code__ for f in functions))).hexdigest()


def load_memo(directory):
    directory = Path(directory).resolve()
    build = json.loads((directory/"build.json").read_text(encoding="utf-8"))
    verify_hashes(build["source_hashes"])
    path = Path(build["extension"]).resolve()
    if directory not in path.parents or path.suffix not in (".pyd", ".so"):
        raise ValueError("require isolated extension below build directory")
    if rr.file_record(path)["sha256"] != build["extension_sha256"]:
        raise ValueError("experimental binary changed")
    name = "_jieqi_repetition_experiment"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    if (Path(module.__file__).resolve() != path or module.API_VERSION != 1 or
            module.SOURCE_SHA256 != build["source_hashes"][str(ROOT/"native/experimental_repetition.pyx")] or
            module.CORE_SHA256 != build["source_hashes"][str(ROOT/"azero.py")]):
        raise ImportError("experimental source/binary identity mismatch")
    module.initialize(az, code_fingerprint())
    return module


def build_extension(directory):
    source = ROOT/"native/experimental_repetition.pyx"
    sys.path.insert(0, str(ROOT/"data/build_deps"))
    from Cython.Build import cythonize
    import Cython
    from setuptools import Extension, setup
    hashes = {str(p): rr.file_record(p)["sha256"] for p in (source, ROOT/"azero.py", Path(sys.executable))}
    shutil.copyfile(source, directory/"source_snapshot.pyx")
    flags = ["/fp:strict"] if os.name == "nt" else ["-fno-fast-math", "-ffp-contract=off"]
    extensions = cythonize([Extension("_jieqi_repetition_experiment", [str(source)], extra_compile_args=flags)],
        build_dir=str(directory/"cython"), compile_time_env=dict(NATIVE_SOURCE_SHA256=hashes[str(source)],
            NATIVE_CORE_SHA256=hashes[str(ROOT/"azero.py")], NATIVE_CODE_SHA256=code_fingerprint()),
        compiler_directives=dict(language_level=3, boundscheck=True, wraparound=True), force=True)
    setup(name="jieqi-repetition-experiment", ext_modules=extensions,
          script_args=["build_ext", "--force", "--build-temp", str(directory/"temp"),
                       "--build-lib", str(directory/"lib")])
    paths = list((directory/"lib").glob("_jieqi_repetition_experiment.*"))
    if len(paths) != 1:
        raise RuntimeError("require one isolated extension")
    verify_hashes(hashes)
    result = dict(kind="repetition_memo_build", extension=str(paths[0]), extension_sha256=rr.file_record(paths[0])["sha256"],
                  source_hashes=hashes, cython=Cython.__version__, compiler_flags=flags,
                  python=sys.executable, code_fingerprint=code_fingerprint(), production_extension_rebuilt=False)
    with (directory/"build.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def micro_gate(totals):
    if set(totals) != {"plain", "uncached", "compiled"} or any(t <= 0 for t in totals.values()):
        raise ValueError("require positive complete three-mode totals")
    gain = totals["plain"]/totals["compiled"]-1
    memo_gain = totals["uncached"]/totals["compiled"]-1
    return dict(throughput_gain=gain, wall_saving=1-totals["compiled"]/totals["plain"],
                memo_vs_uncached_gain=memo_gain, micro_gate_passed=gain >= .10 and memo_gain >= .10)


def paired_micro(cases, module):
    before = probe.digest(cases)
    cached, uncached = module.RepetitionMemo(), module.RepetitionMemo(enabled=False)
    audit = module.RepetitionMemo(collect_stats=True)
    for info in cases:
        expected = ORIGINAL(info)
        if audit.entries(info) != expected or uncached.entries(info) != expected:
            raise AssertionError("captured preparation entries differ")
    audit_stats = audit.stats()
    if audit_stats["fast_calls"] != len(cases) or audit_stats["fallback_calls"]:
        raise AssertionError("real captured cases missed the prototype fast path")
    modes, trials = ("plain", "uncached", "compiled"), []
    functions = dict(plain=ORIGINAL, uncached=uncached.entries, compiled=cached.entries)
    for repeat in range(8):
        for mode in modes if repeat % 2 == 0 else tuple(reversed(modes)):
            cached.clear()
            uncached.clear()
            az._public_material_signature.cache_clear()
            function = functions[mode]
            for info in cases:
                function(info)
            start = time.perf_counter()
            for _ in range(16):
                for info in cases:
                    function(info)
            trials.append(dict(repeat=repeat, mode=mode, seconds=time.perf_counter()-start))
    if probe.digest(cases) != before:
        raise AssertionError("preparation mutated captured inputs")
    totals = {mode: sum(t["seconds"] for t in trials if t["mode"] == mode) for mode in modes}
    return dict(cases=len(cases), calls_per_mode=len(cases)*128, trials=trials, totals=totals,
                entries_exact=True, inputs_unchanged=True, audit_stats=audit_stats,
                memory_estimate=cached.stats(), **micro_gate(totals))


def audit_graphs(net, config, fixtures, module, record):
    original_search, rows = az.NeuralMCTS.search, []
    for fixture in fixtures:
        for strong in (False, True):
            signatures, prepared_calls = [], 0
            memo = module.RepetitionMemo(collect_stats=True)
            for mode in MODES:
                az._public_material_signature.cache_clear()
                memo.clear()
                trace, retained = NetworkTrace(), []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original_search(search, *args, **kwargs)
                    retained.append(root)
                    return policy, move

                def checked(info):
                    nonlocal prepared_calls
                    expected = ORIGINAL(info)
                    actual = memo.entries(info)
                    if actual != expected:
                        raise AssertionError("live repetition entries changed")
                    prepared_calls += 1
                    return actual

                hook = net.register_forward_hook(trace.hook)
                try:
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(az.NeuralMCTS, "search", capture))
                        if mode == "compiled":
                            stack.enter_context(mock.patch.object(az, "_repetition_entries", checked))
                        result = probe.run_case(net, config, probe.DEVICE, fixture, strong,
                                               202610084+fixture["metadata"]["index"], "plain")
                    signatures.append(dict(exact=result["exact"], graph=canonical_graph(retained[0]), network=trace.rows))
                finally:
                    hook.remove()
            stats = memo.stats()
            if (signatures[0] != signatures[1] or prepared_calls < 1
                    or stats["fast_calls"] != prepared_calls or stats["fallback_calls"]):
                raise AssertionError("memo changed network/graph/strategy/RNG or missed real inputs")
            row = dict(index=fixture["metadata"]["index"], strong=strong, preparations_checked=prepared_calls,
                       network_requests=len(signatures[0]["network"]), graph=signatures[0]["graph"],
                       network_sha256=probe.digest(signatures[0]["network"]), stats=stats, exact=True)
            rows.append(row)
            record("graph_exact", row)
    return rows


def init_worker(weights, config, expected, directory):
    global MEMO
    native.init_worker(weights, config, expected)
    MEMO = load_memo(directory).RepetitionMemo()


def root_task(task):
    mode, fixture, strong, seed = task
    if mode not in MODES:
        raise ValueError("unknown repetition memo mode")
    # A conservative within-root test: both caches cold for every root in both modes.
    az._public_material_signature.cache_clear()
    MEMO.clear()
    with ExitStack() as stack:
        if mode == "compiled":
            stack.enter_context(mock.patch.object(az, "_repetition_entries", MEMO.entries))
        row = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    row.update(mode=mode, pid=os.getpid())
    return row


def screen(directory, build_dir, inspection_path, reference_path):
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    if (inspection.get("kind") != "repetition_preparation_inspection"
            or not inspection.get("roots_exact") or not inspection.get("production_unchanged")
            or not inspection["summary"]["prototype_worth_screening"]):
        raise ValueError("require passed immutable-board reuse inspection")
    hashes = dict(inspection["source_hashes"])
    verify_hashes(hashes)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference.get("kind") != "python_sampling_workload" or not reference.get("exact_against_baseline")
            or not reference.get("production_unchanged") or
            json.loads(json.dumps(runtime_identity())) != inspection["identity"] or
            reference["identity"] != inspection["identity"] or
            Path(sys.executable).resolve() != Path(inspection["python"]).resolve() or not torch.cuda.is_available()):
        raise ValueError("require identical current exact reference/runtime/CUDA")
    build = json.loads((build_dir/"build.json").read_text(encoding="utf-8"))
    for path in (inspection_path, reference_path, Path(__file__).resolve(), ROOT/"native/experimental_repetition.pyx",
                 ROOT/"test_repetition_memo_experiment.py", ROOT/"tools/benchmark_exact_encoder.py",
                 build_dir/"build.json", Path(build["extension"]),
                 ROOT/"data/pilots/repetition_memo_20261009_iter28_protocol.md"):
        hashes[str(path)] = rr.file_record(path)["sha256"]
    module = load_memo(build_dir)
    torch.set_num_threads(1)
    source = ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs"
    checkpoint = torch.load(source/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
    config = az.AZConfig(**checkpoint["config"])
    validate_config(config)
    config.data_dir, config.model_dir, config.log_dir = (str(directory/n) for n in ("samples", "models", "logs"))
    with (source.parent/"measure.fixtures.pkl").open("rb") as handle:
        fixtures = pickle.load(handle)["fixtures"]
    if len(fixtures) != 18 or [f["metadata"]["index"] for f in fixtures] != list(range(18)):
        raise ValueError("require current18 fixtures")
    with (inspection_path.parent/"cases.pkl").open("rb") as handle:
        cases = pickle.load(handle)
    start = time.perf_counter()

    def record(event, data):
        append_event(directory/"events.jsonl", event, dict(elapsed=time.perf_counter()-start, data=data))
        probe.emit(event, data)

    result = dict(kind="repetition_memo_root_screen", source_hashes=hashes, identity=runtime_identity(),
                  python=sys.executable, config=asdict(config), groups=[], production_training=False,
                  snapshots=[gpu_snapshot()], compiler_build=build, caches_reset_per_root=True)
    result["micro"] = paired_micro(cases, module)
    record("micro", result["micro"])
    if result["micro"]["micro_gate_passed"]:
        net = az.PolicyValueNet().to("cuda").eval()
        az.load_state_dict_padded(net, checkpoint["model"])
        result["graph_checks"] = audit_graphs(net, config, fixtures, module, record)
        del net
        gc.collect()
        torch.cuda.empty_cache()
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
        result["summary"] = root_screen.summarize_roots(result["groups"])
    else:
        result["summary"] = dict(full_game_screen_gate_passed=False, production_promotion=False,
                                 reason="predeclared plain or memo-ablation local gain below 10%")
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
    parser.add_argument("--inspection-file", type=Path)
    parser.add_argument("--reference-file", type=Path)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    native.validate_output(directory, (ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs").resolve())
    if args.phase == "screen" and (not args.build_dir or not args.inspection_file or not args.reference_file):
        parser.error("screen requires build, inspection and reference")
    directory.mkdir(parents=True)
    if args.phase == "build":
        print(json.dumps(build_extension(directory)), flush=True)
    else:
        screen(directory, args.build_dir.resolve(), args.inspection_file.resolve(), args.reference_file.resolve())


if __name__ == "__main__":
    main()
