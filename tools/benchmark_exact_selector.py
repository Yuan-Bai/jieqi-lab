"""Build and screen one isolated, exact object-arithmetic PUCT selector."""
from __future__ import annotations

import argparse
from collections import deque
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

MODES = ("plain", "compiled")
ORIGINAL = az.PUCTNode.select_child
SELECTOR = None


def code_fingerprint():
    return hashlib.sha256(marshal.dumps((ORIGINAL.__code__, az.PUCTNode.q_value.__code__))).hexdigest()


def load_selector(directory):
    directory = Path(directory).resolve()
    build = json.loads((directory/"build.json").read_text(encoding="utf-8"))
    verify_hashes(build["source_hashes"])
    path = Path(build["extension"]).resolve()
    if directory not in path.parents or path.suffix not in (".pyd", ".so"):
        raise ValueError("require isolated extension below its build directory")
    if rr.file_record(path)["sha256"] != build["extension_sha256"]:
        raise ValueError("experimental binary changed")
    name = "_jieqi_selector_experiment"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    if (Path(module.__file__).resolve() != path or module.API_VERSION != 1 or
            module.SOURCE_SHA256 != build["source_hashes"][str(ROOT/"native/experimental_selector.pyx")] or
            module.CORE_SHA256 != build["source_hashes"][str(ROOT/"azero.py")]):
        raise ImportError("experimental source/binary identity mismatch")
    module.initialize(az, code_fingerprint())
    return module


def build_extension(directory):
    source = ROOT/"native/experimental_selector.pyx"
    sys.path.insert(0, str(ROOT/"data/build_deps"))
    from Cython.Build import cythonize
    import Cython
    from setuptools import Extension, setup
    hashes = {str(p): rr.file_record(p)["sha256"] for p in (source, ROOT/"azero.py", Path(sys.executable))}
    shutil.copyfile(source, directory/"source_snapshot.pyx")
    flags = ["/fp:strict"] if os.name == "nt" else ["-fno-fast-math", "-ffp-contract=off"]
    extensions = cythonize([Extension("_jieqi_selector_experiment", [str(source)], extra_compile_args=flags)],
        build_dir=str(directory/"cython"), compile_time_env=dict(NATIVE_SOURCE_SHA256=hashes[str(source)],
            NATIVE_CORE_SHA256=hashes[str(ROOT/"azero.py")], NATIVE_CODE_SHA256=code_fingerprint()),
        compiler_directives=dict(language_level=3, boundscheck=True, wraparound=True), force=True)
    setup(name="jieqi-selector-experiment", ext_modules=extensions,
          script_args=["build_ext", "--force", "--build-temp", str(directory/"temp"),
                       "--build-lib", str(directory/"lib")])
    paths = list((directory/"lib").glob("_jieqi_selector_experiment.*"))
    if len(paths) != 1:
        raise RuntimeError("require one isolated extension")
    verify_hashes(hashes)
    result = dict(kind="exact_selector_build", extension=str(paths[0]), extension_sha256=rr.file_record(paths[0])["sha256"],
                  source_hashes=hashes, cython=Cython.__version__, compiler_flags=flags,
                  python=sys.executable, code_fingerprint=code_fingerprint(), production_extension_rebuilt=False)
    with (directory/"build.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def selection_snapshot(node, c_puct):
    return (c_puct, tuple((move, child.prior, child.player_to_move, child.visits, child.value_sum,
                         child.is_chance) for move, child in node.children.items()))


def restore_snapshot(snapshot):
    c_puct, children = snapshot
    node = az.PUCTNode(1.0, 0)
    for move, prior, player, visits, value_sum, chance in children:
        child = az.PUCTNode(prior, player, move=move, is_chance=chance)
        child.visits, child.value_sum = visits, value_sum
        node.children[move] = child
    return node, c_puct


def paired_micro(snapshots, selector):
    cases = [restore_snapshot(s) for s in snapshots]
    before = probe.digest(snapshots)
    fast_cases = 0
    for node, c_puct in cases:
        expected, actual = ORIGINAL(node, c_puct), selector.select_child(node, c_puct)
        if expected[0] != actual[0] or expected[1] is not actual[1]:
            raise AssertionError("captured selector winner differs")
        fast_cases += int(selector.supports(node))
    trials = []
    for repeat in range(8):
        for mode in MODES if repeat % 2 == 0 else tuple(reversed(MODES)):
            function = ORIGINAL if mode == "plain" else selector.select_child
            start = time.perf_counter()
            for _ in range(32):
                for node, c_puct in cases:
                    function(node, c_puct)
            trials.append(dict(repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-start))
    after = [selection_snapshot(node, c_puct) for node, c_puct in cases]
    if probe.digest(after) != before:
        raise AssertionError("selector mutated captured inputs")
    plain = sum(t["wall_seconds"] for t in trials if t["mode"] == "plain")
    compiled = sum(t["wall_seconds"] for t in trials if t["mode"] == "compiled")
    gain = plain/compiled-1
    return dict(cases=len(cases), fast_cases=fast_cases, calls_per_mode=len(cases)*256, trials=trials,
                throughput_gain=gain, wall_saving=1-compiled/plain, micro_gate_passed=gain >= .10,
                node_creation_included=False, inputs_unchanged=True, winners_exact=True)


def audit_graphs(net, config, fixtures, selector, record):
    original_search, rows, snapshots = az.NeuralMCTS.search, [], []
    for fixture in fixtures:
        for strong in (False, True):
            signatures, selected_calls, fast_inputs = [], 0, 0
            first, sampled_last = [], deque(maxlen=8)
            for mode in MODES:
                trace, retained = NetworkTrace(), []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original_search(search, *args, **kwargs)
                    retained.append(root)
                    return policy, move

                def checked(node, c_puct):
                    nonlocal selected_calls, fast_inputs
                    expected = ORIGINAL(node, c_puct)
                    actual = selector.select_child(node, c_puct)
                    if expected[0] != actual[0] or expected[1] is not actual[1]:
                        raise AssertionError("live selector winner differs")
                    selected_calls += 1
                    fast_inputs += int(selector.supports(node))
                    if len(first) < 8:
                        first.append(selection_snapshot(node, c_puct))
                    if selected_calls % 128 == 0:
                        sampled_last.append(selection_snapshot(node, c_puct))
                    return actual

                hook = net.register_forward_hook(trace.hook)
                try:
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(az.NeuralMCTS, "search", capture))
                        if mode == "compiled":
                            stack.enter_context(mock.patch.object(az.PUCTNode, "select_child", checked))
                        result = probe.run_case(net, config, probe.DEVICE, fixture, strong,
                                               202610084+fixture["metadata"]["index"], "plain")
                    signatures.append(dict(exact=result["exact"], graph=canonical_graph(retained[0]), network=trace.rows))
                finally:
                    hook.remove()
            if signatures[0] != signatures[1] or selected_calls < 1:
                raise AssertionError("compiled selection changed network/graph/strategy/RNG or missed calls")
            snapshots.extend(first+list(sampled_last))
            row = dict(index=fixture["metadata"]["index"], strong=strong,
                       selections_checked=selected_calls, fast_inputs=fast_inputs,
                       network_requests=len(signatures[0]["network"]), graph=signatures[0]["graph"],
                       network_sha256=probe.digest(signatures[0]["network"]), exact=True)
            rows.append(row)
            record("graph_exact", row)
    return rows, snapshots


def init_worker(weights, config, expected, directory):
    global SELECTOR
    native.init_worker(weights, config, expected)
    SELECTOR = load_selector(directory)


def root_task(task):
    mode, fixture, strong, seed = task
    if mode not in MODES:
        raise ValueError("unknown selector mode")
    with ExitStack() as stack:
        if mode == "compiled":
            stack.enter_context(mock.patch.object(az.PUCTNode, "select_child", SELECTOR.select_child))
        row = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    row.update(mode=mode, pid=os.getpid())
    return row


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
    for path in (reference_path, Path(__file__).resolve(), ROOT/"native/experimental_selector.pyx",
                 ROOT/"test_exact_selector_experiment.py", ROOT/"tools/benchmark_exact_encoder.py",
                 build_dir/"build.json", Path(build["extension"]),
                 ROOT/"data/pilots/exact_selector_20261009_iter28_protocol.md"):
        hashes[str(path)] = rr.file_record(path)["sha256"]
    selector = load_selector(build_dir)
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

    result = dict(kind="exact_selector_root_screen", source_hashes=hashes, identity=runtime_identity(),
                  python=sys.executable, config=asdict(config), groups=[], production_training=False,
                  snapshots=[gpu_snapshot()], compiler_build=build)
    net = az.PolicyValueNet().to("cuda").eval()
    az.load_state_dict_padded(net, checkpoint["model"])
    result["graph_checks"], cases = audit_graphs(net, config, fixtures, selector, record)
    with (directory/"micro_cases.pkl").open("xb") as handle:
        pickle.dump(cases, handle)
    hashes[str(directory/"micro_cases.pkl")] = rr.file_record(directory/"micro_cases.pkl")["sha256"]
    del net
    gc.collect()
    torch.cuda.empty_cache()
    result["micro"] = paired_micro(cases, selector)
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
        result["summary"] = root_screen.summarize_roots(result["groups"])
    else:
        result["summary"] = dict(full_game_screen_gate_passed=False, production_promotion=False,
                                 reason="local selector throughput gain below predeclared 10% screen")
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
