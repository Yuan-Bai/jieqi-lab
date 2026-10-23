"""Read-only frozen-root sampling workload and strict py-spy Speedscope analysis."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures as futures
from dataclasses import asdict
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THIS = Path(__file__).resolve()
PROFILE_NAME = re.compile(r'^Process (\d+) Thread (\d+) "[^\"]*"$')


def normalized_file(value):
    return str(value).replace("\\", "/").casefold()


def compatibility_hot_loop(seconds):
    end, value = time.perf_counter()+seconds, 1
    while time.perf_counter() < end:
        value = (value * 1664525 + 1013904223) & 0xffffffff
    return dict(pid=os.getpid(), native_thread_id=threading.get_native_id(), value=value)


def call_root(task):
    from tools import profile_selfplay as probe
    fixture, strong, seed = task
    row = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    row.update(pid=os.getpid(), native_thread_id=threading.get_native_id())
    return row


def warm_task(task):
    return call_root(task)


def measured_ordinary(task):
    return call_root(task)


def measured_strong(task):
    return call_root(task)


def measured_task(task):
    # Distinct stack markers exclude warmup without changing the search implementation.
    return (measured_strong if task[1] else measured_ordinary)(task)


def count_rows(counter, total, line=False):
    keys = ("file", "name", "line") if line else ("file", "name")
    return [dict(zip(keys, key), samples=count, fraction=count/total)
            for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def analyze_speedscope(data, workers, *, compatibility=False, minimum_worker=200, minimum_total=2000):
    if data.get("exporter") != "py-spy@0.4.2":
        raise ValueError("require pinned py-spy 0.4.2 Speedscope export")
    expected = {(row["pid"], row["native_thread_id"]) for row in workers}
    if not expected or len({pid for pid, _ in expected}) != len(expected):
        raise ValueError("require one known main thread per worker")
    frames = data["shared"]["frames"]
    own, inclusive, lines, modules = Counter(), Counter(), Counter(), Counter()
    budgets = {label: dict(own=Counter(), inclusive=Counter(), count=0)
               for label in ("800", "3200")}
    selected, excluded, evaluation = Counter(), 0, 0
    for profile in data["profiles"]:
        match = PROFILE_NAME.fullmatch(profile.get("name", ""))
        worker = tuple(map(int, match.groups())) if match else None
        if worker not in expected:
            excluded += len(profile.get("samples", []))
            continue
        if profile.get("type") != "sampled" or profile.get("unit") != "seconds":
            raise ValueError("require sampled seconds profiles")
        samples, weights = profile["samples"], profile["weights"]
        if len(samples) != len(weights) or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("invalid sampling weights")
        for sample in samples:
            if not sample or any(type(i) is not int or i < 0 or i >= len(frames) for i in sample):
                raise ValueError("invalid frame indices")
            stack = [frames[i] for i in sample]
            markers = {f["name"] for f in stack if normalized_file(f.get("file", "")) == normalized_file(THIS)}
            search = any(f["name"] == "search" and normalized_file(f.get("file", "")) ==
                         normalized_file(ROOT/"azero.py") for f in stack)
            if compatibility:
                keep = "compatibility_hot_loop" in markers
                budget = None
            else:
                budget_markers = markers & {"measured_ordinary", "measured_strong"}
                keep = "measured_task" in markers and search and len(budget_markers) == 1
                budget = "3200" if "measured_strong" in budget_markers else "800"
            if not keep:
                excluded += 1
                continue
            selected[worker] += 1
            identities = {(f.get("file", ""), f["name"]) for f in stack}
            leaf = stack[-1]
            leaf_key = (leaf.get("file", ""), leaf["name"])
            own[leaf_key] += 1
            lines[(*leaf_key, leaf.get("line", 0))] += 1
            inclusive.update(identities)
            modules[leaf_key[0]] += 1
            if any(f["name"] in ("_evaluate", "_evaluate_batch") and normalized_file(f.get("file", "")) ==
                   normalized_file(ROOT/"azero.py") for f in stack):
                evaluation += 1
            if budget:
                budgets[budget]["count"] += 1
                budgets[budget]["own"][leaf_key] += 1
                budgets[budget]["inclusive"].update(identities)
    total = sum(selected.values())
    if not total:
        raise ValueError("no eligible known-worker measured-search samples")
    coverage = [dict(pid=pid, native_thread_id=tid, samples=selected[pid, tid]) for pid, tid in sorted(expected)]
    return dict(eligible_samples=total, excluded_samples=excluded, worker_coverage=coverage,
        coverage_gate_passed=total >= minimum_total and all(row["samples"] >= minimum_worker for row in coverage),
        thresholds=dict(minimum_worker=minimum_worker, minimum_total=minimum_total),
        evaluation_stack_samples=evaluation, evaluation_stack_fraction=evaluation/total,
        exclusive=count_rows(own, total), inclusive=count_rows(inclusive, total),
        exclusive_lines=count_rows(lines, total, line=True),
        leaf_modules=[dict(file=file, samples=count, fraction=count/total) for file, count in modules.most_common()],
        budgets={label: dict(samples=b["count"], exclusive=count_rows(b["own"], b["count"]),
                            inclusive=count_rows(b["inclusive"], b["count"]))
                 for label, b in budgets.items() if b["count"]},
        interpretation="Conditional GIL-held Python-stack samples, NOT CPU seconds or production CPU/wall share; "
                       "GIL-released native work is omitted; GIL-held C/Cython work can be charged to its Python caller; "
                       "weights and compressed profile endValue are not measured CPU durations")


def validate_groups(groups, reference):
    from tools.measure_sync_baseline import exact_rows
    if len(groups) != 4 or [g["repeat"] for g in groups] != list(range(4)):
        raise ValueError("require four complete ordered cohorts")
    expected_keys = [(index, strong) for index in range(18) for strong in (False, True)]
    for group in groups:
        if group["wall_seconds"] <= 0 or [(r["index"], r["strong"]) for r in group["rows"]] != expected_keys:
            raise ValueError("require ordered 36-root positive-duration cohorts")
        base = next(g for g in reference["groups"] if g["session"] == 0 and
                    g["repeat"] == group["repeat"] and g["mode"] == "plain")
        if exact_rows(group["rows"]) != exact_rows(base["rows"]):
            raise AssertionError("search policy/move/RNG/counters differ from frozen no-tracer reference")


def workload(directory, reference_path):
    import torch
    import azero as az
    import run_records as rr
    from tools import profile_native_selfplay as native
    from tools.measure_search_reuse import verify_hashes
    from tools.measure_selfplay_attribution import validate_config
    from tools.measure_selfplay_round import append_event, runtime_identity
    from tools.measure_sync_baseline import gpu_snapshot

    source = ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs"
    native.validate_output(directory, source.resolve())
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference.get("kind") != "host_wait_game_screen" or not reference.get("production_unchanged")
            or not reference.get("summary", {}).get("paired_games_exact")):
        raise ValueError("require completed exact full-game reference")
    hashes = dict(reference["source_hashes"])
    baseline_path = ROOT/"data/pilots/sync_baseline_20261009_iter28/result.json"
    verify_hashes(hashes)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (json.loads(json.dumps(runtime_identity())) != reference["identity"] or
            Path(sys.executable).resolve() != Path(reference["python"]).resolve() or not torch.cuda.is_available()):
        raise ValueError("require identical interpreter/CUDA/backend/precision")
    for path in (reference_path, THIS, ROOT/"tests/test_profile_python_selfplay.py",
                 ROOT/"data/pilots/python_hotspots_20261009_iter28_protocol.md"):
        hashes[str(path)] = rr.file_record(path)["sha256"]
    torch.set_num_threads(1)
    checkpoint = torch.load(source/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
    config = az.AZConfig(**checkpoint["config"])
    validate_config(config)
    config.data_dir, config.model_dir, config.log_dir = (str(directory/n) for n in ("samples", "models", "logs"))
    with (source.parent/"measure.fixtures.pkl").open("rb") as handle:
        fixtures = pickle.load(handle)["fixtures"]
    if len(fixtures) != 18 or [f["metadata"]["index"] for f in fixtures] != list(range(18)):
        raise ValueError("require trusted current18 fixtures")
    directory.mkdir(parents=True)
    start = time.perf_counter()
    result = dict(kind="python_sampling_workload", source_hashes=hashes, identity=runtime_identity(),
                  python=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
                  config=asdict(config), groups=[], snapshots=[gpu_snapshot()], production_training=False)

    def record(event, data):
        append_event(directory/"events.jsonl", event, dict(elapsed=time.perf_counter()-start, data=data))
        print(json.dumps(dict(event=event, data=data)), flush=True)

    with futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
            initializer=native.init_worker, initargs=(checkpoint["model"], asdict(config), runtime_identity())) as pool:
        for warm in range(2):
            rows = list(pool.map(warm_task, [(f, strong, 202610101+f["metadata"]["index"])
                                            for f in fixtures for strong in (False, True)]))
            if len({r["pid"] for r in rows}) != 8:
                raise RuntimeError("warmup missed workers")
            record("warm", dict(pass_index=warm, pids=sorted({r["pid"] for r in rows})))
        for repeat in range(4):
            tasks = [(f, strong, 202610110+repeat*1000+f["metadata"]["index"])
                     for f in fixtures for strong in (False, True)]
            begin = time.perf_counter()
            rows = list(pool.map(measured_task, tasks))
            group = dict(repeat=repeat, wall_seconds=time.perf_counter()-begin, rows=rows)
            result["groups"].append(group)
            record("cohort", {k:v for k,v in group.items() if k != "rows"})
    validate_groups(result["groups"], baseline)
    verify_hashes(hashes)
    result.update(production_unchanged=True, exact_against_baseline=True, elapsed_seconds=time.perf_counter()-start)
    result["snapshots"].append(gpu_snapshot())
    with (directory/"result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    record("complete", dict(elapsed=result["elapsed_seconds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--workload", action="store_true")
    modes.add_argument("--compatibility-probe", action="store_true")
    modes.add_argument("--analyze", type=Path, help="Pinned py-spy Speedscope JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--workload-result", type=Path)
    parser.add_argument("--analysis-file", type=Path)
    args = parser.parse_args()
    if args.analyze:
        if not args.workload_result or not args.analysis_file or args.analysis_file.exists():
            parser.error("require workload result and a new analysis file")
        result = json.loads(args.workload_result.read_text(encoding="utf-8"))
        compatibility = result.get("kind") == "python_sampling_compatibility"
        if not compatibility and (result.get("kind") != "python_sampling_workload" or
                not result.get("production_unchanged") or not result.get("exact_against_baseline")):
            parser.error("require completed exact workload")
        workers = result["workers"] if compatibility else [r for g in result["groups"] for r in g["rows"]]
        analysis = analyze_speedscope(json.loads(args.analyze.read_text(encoding="utf-8")), workers,
                                     compatibility=compatibility, minimum_total=200 if compatibility else 2000,
                                     minimum_worker=50 if compatibility else 200)
        from tools.measure_search_reuse import verify_hashes
        import run_records as rr
        if not compatibility:
            verify_hashes(result["source_hashes"])
        analysis["input_hashes"] = {str(p.resolve()): rr.file_record(p)["sha256"]
                                    for p in (args.analyze, args.workload_result)}
        with args.analysis_file.open("x", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, allow_nan=False)
        print(json.dumps({k:analysis[k] for k in ("eligible_samples", "coverage_gate_passed", "worker_coverage")}))
    elif args.compatibility_probe:
        from tools.profile_native_selfplay import validate_output
        if not args.output_dir:
            parser.error("require a new output directory")
        directory = args.output_dir.resolve()
        validate_output(directory, (ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs").resolve())
        directory.mkdir(parents=True)
        with futures.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
            workers = list(pool.map(compatibility_hot_loop, [8, 8]))
        if len({w["pid"] for w in workers}) != 2:
            raise RuntimeError("compatibility probe missed child")
        with (directory/"result.json").open("x", encoding="utf-8") as handle:
            json.dump(dict(kind="python_sampling_compatibility", workers=workers), handle, indent=2)
    else:
        if not args.reference_file or not args.output_dir:
            parser.error("require reference and a new output directory")
        workload(args.output_dir.resolve(), args.reference_file.resolve())


if __name__ == "__main__":
    main()
