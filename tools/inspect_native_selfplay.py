"""Read-only SQLite coverage/aggregate analysis of the frozen native capture."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from tools.inspect_native_runtime import decode_pid


def interval_union(intervals):
    """Union length of intervals ordered by start, not a sum across GPU streams."""
    total, last_end = 0, None
    for start, end in intervals:
        if end < start:
            raise ValueError("negative GPU interval")
        if last_end is None:
            total += end - start
        elif end > last_end:
            total += end - max(start, last_end)
        last_end = end if last_end is None else max(last_end, end)
    return total


def label_coverage(connection, workload):
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"NVTX_EVENTS", "StringIds"}
    missing = required - tables
    if missing:
        raise ValueError(f"label analysis requires SQLite tables: {sorted(missing)}")
    nvtx = "SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) AS label FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id"
    windows = [dict(row) for row in connection.execute(f"SELECT * FROM ({nvtx}) WHERE label='jieqi_native_window'")
               if decode_pid(row["globalTid"]) == workload.get("pid")]
    if len(windows) != 1 or windows[0]["end"] is None or windows[0]["end"] <= windows[0]["start"]:
        raise ValueError("one closed parent measurement window required")
    window = windows[0]
    start, end = window["start"], window["end"]
    bounds = (start, end)
    expected = Counter()
    threads = {}
    errors = []
    for group in workload["groups"]:
        for row in group["rows"]:
            expected[row["pid"], "jieqi/root/3200" if row["strong"] else "jieqi/root/800"] += 1
            if row["pid"] in threads and threads[row["pid"]] != row["native_thread_id"]:
                errors.append(f"PID {row['pid']}: inconsistent workload main thread")
            threads[row["pid"]] = row["native_thread_id"]
    if (workload.get("kind") != "native_profile_capture" or not workload.get("paired_roots_exact")
            or not workload.get("production_unchanged") or len(threads) != 8 or sum(expected.values()) != 144
            or len(workload["groups"]) != 4 or any(len(group["rows"]) != 36 for group in workload["groups"])):
        raise ValueError("require completed exact eight-worker/four-cohort frozen capture")
    actual = Counter()
    phases = {}
    ranges = {}
    for row in connection.execute(f"SELECT * FROM ({nvtx}) WHERE start<? AND (end IS NULL OR end>?)", (start, start)):
        if decode_pid(row["globalTid"]) in threads and (row["label"] or "").startswith("jieqi/"):
            errors.append("worker NVTX range crosses the measurement start")
    for row in connection.execute(f"SELECT * FROM ({nvtx}) WHERE start>=? AND start<=?", bounds):
        pid = decode_pid(row["globalTid"])
        label = row["label"]
        if not label or not label.startswith("jieqi/"):
            continue
        if row["end"] is None or row["end"] > end or row["end"] < row["start"]:
            errors.append(f"open/crossing NVTX range: PID {pid}, {label}")
            continue
        if pid not in threads or (row["globalTid"] & 0xFFFFFF) != threads[pid]:
            errors.append(f"unexpected labelled worker/thread: PID {pid}")
        if label.startswith("jieqi/root/"):
            actual[pid, label] += 1
        phase = phases.setdefault((pid, label), dict(count=0, wall_ns=0))
        phase["count"] += 1
        phase["wall_ns"] += row["end"] - row["start"]
        ranges.setdefault((pid, label), []).append((row["start"], row["end"]))
    if actual != expected:
        errors.append("per-worker ordinary/strong root ranges differ from all 144 workload rows")
    for pid in threads:
        count = phases.get((pid, "jieqi/evaluation"), {}).get("count", 0)
        if count == 0 or any(phases.get((pid, "jieqi/" + phase), {}).get("count", 0) != count
                             for phase in ("encode", "h2d", "forward", "softmax", "policy_d2h", "value_d2h")):
            errors.append(f"PID {pid}: incomplete evaluation/phase label counts")
    return tables, window, threads, phases, ranges, errors


def scheduling_segments(events, bounds):
    """Use state transitions, including those outside the measurement boundaries."""
    segments, anomalies = [], 0
    start, end = bounds
    for current, following in zip(events, events[1:]):
        left, right = max(start, current["start"]), min(end, following["start"])
        if left >= right:
            continue
        if current["isSchedIn"] and following["isSchedIn"]:
            anomalies += 1
            continue
        segments.append((left, right, bool(current["isSchedIn"]),
                         current["threadState"], current["threadBlock"]))
    return segments, anomalies


def intersect_ns(intervals, ranges):
    """Intersection of two sorted, non-overlapping interval sets."""
    total, i, j = 0, 0, 0
    while i < len(intervals) and j < len(ranges):
        left, right = intervals[i], ranges[j]
        total += max(0, min(left[1], right[1]) - max(left[0], right[0]))
        if left[1] <= right[1]:
            i += 1
        else:
            j += 1
    return total


def inspect_cpu(connection, workload, labels):
    tables, window, threads, phases, ranges, label_errors = labels
    errors = list(label_errors)
    required = {"COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS", "SCHED_EVENTS",
                "ENUM_SAMPLING_THREAD_STATE", "ENUM_SCHEDULING_THREAD_BLOCK"}
    if required - tables:
        raise ValueError(f"CPU analysis requires SQLite tables: {sorted(required - tables)}")
    bounds = window["start"], window["end"]
    thread_ids = {}
    for row in connection.execute("SELECT DISTINCT globalTid FROM NVTX_EVENTS WHERE start>=? AND start<?", bounds):
        pid = decode_pid(row["globalTid"])
        if pid in threads and (row["globalTid"] & 0xFFFFFF) == threads[pid]:
            if pid in thread_ids and thread_ids[pid] != row["globalTid"]:
                errors.append(f"PID {pid}: ambiguous global main-thread identity")
            thread_ids[pid] = row["globalTid"]
    if set(thread_ids) != set(threads):
        raise ValueError("missing NVTX identity for a workload main thread")
    placeholders = ",".join("?" for _ in thread_ids)
    ids = tuple(thread_ids.values())
    inverse_ids = {tid: pid for pid, tid in thread_ids.items()}
    root_ranges, lookups = {}, {}
    for pid in threads:
        root_ranges[pid] = sorted(ranges.get((pid, "jieqi/root/800"), []) + ranges.get((pid, "jieqi/root/3200"), []))
        for (owner, label), values in ranges.items():
            if owner != pid:
                continue
            values.sort()
            if any(left[1] > right[0] for left, right in zip(values, values[1:])):
                errors.append(f"PID {pid}: overlapping {label} ranges")
            lookups[pid, label] = ([row[0] for row in values], values)
        if any(left[1] > right[0] for left, right in zip(root_ranges[pid], root_ranges[pid][1:])):
            errors.append(f"PID {pid}: overlapping root ranges")

    def contains(values, timestamp):
        starts, intervals = values
        index = bisect_right(starts, timestamp) - 1
        return index >= 0 and timestamp < intervals[index][1]

    leaf_phases = ("encode", "h2d", "forward", "softmax", "policy_d2h", "value_d2h")
    roots_lookup = {pid: ([row[0] for row in values], values) for pid, values in root_ranges.items()}
    for pid in threads:
        combined = sorted(interval for phase in leaf_phases for interval in ranges.get((pid, "jieqi/" + phase), []))
        if any(left[1] > right[0] for left, right in zip(combined, combined[1:])):
            errors.append(f"PID {pid}: overlapping network subphase ranges")
        for label in ("evaluation",) + leaf_phases:
            parent = roots_lookup[pid] if label == "evaluation" else lookups.get((pid, "jieqi/evaluation"), ([], []))
            for left, right in ranges.get((pid, "jieqi/" + label), []):
                index = bisect_right(parent[0], left) - 1
                if index < 0 or right > parent[1][index][1]:
                    errors.append(f"PID {pid}: {label} outside its enclosing range")
    samples, module_counts, phase_modules = Counter(), Counter(), {}
    missing_leaf, unresolved, unknown_mode, outside_roots = Counter(), Counter(), Counter(), Counter()
    phase_samples = Counter()
    # CpuCycles=1 is the periodic-sample marker in this export, not a cycle count.
    for row in connection.execute(f"""
        SELECT e.start,e.globalTid,f.module,f.unresolved,f.kernelMode,f.originalIP,s.value AS module_name
        FROM COMPOSITE_EVENTS e LEFT JOIN SAMPLING_CALLCHAINS f ON e.id=f.id AND f.stackDepth=0
        LEFT JOIN StringIds s ON f.module=s.id
        WHERE e.globalTid IN ({placeholders}) AND e.start>=? AND e.start<? AND e.cpuCycles=1
    """, ids + bounds):
        pid = inverse_ids[row["globalTid"]]
        samples[pid] += 1
        if not row["module_name"] or row["originalIP"] is None:
            missing_leaf[pid] += 1
        unresolved[pid] += bool(row["unresolved"])
        unknown_mode[pid] += row["kernelMode"] is None
        if not contains(roots_lookup[pid], row["start"]):
            outside_roots[pid] += 1
            continue
        active = [phase for phase in leaf_phases
                  if contains(lookups.get((pid, "jieqi/" + phase), ([], [])), row["start"])]
        if len(active) > 1:
            errors.append(f"PID {pid}: overlapping network subphases at a periodic sample")
            continue
        phase = active[0] if active else ("evaluation_other" if contains(
            lookups.get((pid, "jieqi/evaluation"), ([], [])), row["start"]) else "search_other")
        phase_samples[phase] += 1
        module = row["module_name"] or "<unmapped>"
        module_counts[module] += 1
        phase_modules.setdefault(phase, Counter())[module] += 1

    state_names = {row["id"]: row["name"] for row in connection.execute("SELECT * FROM ENUM_SAMPLING_THREAD_STATE")}
    block_names = {row["id"]: row["name"] for row in connection.execute("SELECT * FROM ENUM_SCHEDULING_THREAD_BLOCK")}
    sched_events = {pid: [] for pid in threads}
    for row in connection.execute(f"SELECT * FROM SCHED_EVENTS WHERE globalTid IN ({placeholders}) ORDER BY globalTid,start", ids):
        sched_events[inverse_ids[row["globalTid"]]].append(dict(row))
    phase_times, per_pid, reasons = {}, {}, Counter()
    for pid in threads:
        segments, anomalies = scheduling_segments(sched_events[pid], bounds)
        on = [(left, right) for left, right, active, _, _ in segments if active]
        off = [(left, right) for left, right, active, _, _ in segments if not active]
        selected = {label.removeprefix("jieqi/"): values for (owner, label), values in ranges.items()
                    if owner == pid and not label.startswith("jieqi/root/")}
        for label in ("evaluation",) + leaf_phases:
            selected.setdefault(label, [])
        selected["root"] = root_ranges[pid]
        times = {}
        for label, values in selected.items():
            wall = sum(end-start for start, end in values)
            on_ns, off_ns = intersect_ns(on, values), intersect_ns(off, values)
            times[label] = dict(wall_ns=wall, scheduled_on_ns=on_ns, scheduled_off_ns=off_ns,
                                unaccounted_ns=wall-on_ns-off_ns)
        for name, parent, children in (("search_other", "root", ("evaluation",)),
                                       ("evaluation_other", "evaluation", leaf_phases)):
            times[name] = {key: value - sum(times[child][key] for child in children)
                           for key, value in times[parent].items()}
            if any(value < 0 for value in times[name].values()):
                errors.append(f"PID {pid}: network phases do not partition their enclosing range")
        for name, values in times.items():
            total = phase_times.setdefault(name, Counter())
            total.update(values)
        for left, right, active, state, block in segments:
            if not active:
                reasons[state_names.get(state, str(state)), block_names.get(block, str(block))] += intersect_ns(
                    [(left, right)], root_ranges[pid])
        if not samples[pid] or missing_leaf[pid] or not segments or anomalies or times["root"]["unaccounted_ns"]:
            errors.append(f"PID {pid}: incomplete periodic leaf or scheduling coverage")
        per_pid[pid] = dict(native_thread_id=threads[pid], global_tid=thread_ids[pid],
                           periodic_window_samples=samples[pid], periodic_outside_root_samples=outside_roots[pid],
                           missing_periodic_leaf=missing_leaf[pid], unresolved_periodic_leaf=unresolved[pid],
                           unknown_kernel_mode_leaf=unknown_mode[pid], sched_events=len(sched_events[pid]),
                           consecutive_sched_in_anomalies=anomalies, root_scheduling=times["root"])
    diagnostics = [dict(row) for row in connection.execute("SELECT * FROM DIAGNOSTIC_EVENT")] if "DIAGNOSTIC_EVENT" in tables else []
    diagnostic_errors = [row for row in diagnostics if row["severity"] == 3]
    if diagnostic_errors:
        errors.append("CPU capture contains collector Error diagnostics")
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_RUNTIME"):
        if table in tables and connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]:
            errors.append("CPU-only capture unexpectedly contains CUDA activity")
    return dict(kind="native_selfplay_coverage", trace_mode="cpu",
                known_root_label_coverage_passed=not label_errors, known_cpu_coverage_passed=not errors,
                lossless_trace_verified=False, errors=sorted(set(errors)), window=window,
                window_seconds=(bounds[1]-bounds[0])/1e9, per_pid=per_pid,
                phases=[dict(pid=pid, label=label, **values) for (pid, label), values in sorted(phases.items())],
                phase_scheduling={name: dict(values) for name, values in phase_times.items()},
                periodic_root_samples=sum(module_counts.values()),
                periodic_leaf_modules=[dict(module=module, samples=count) for module, count in module_counts.most_common()],
                periodic_phases={phase: dict(samples=count, leaf_modules=[dict(module=module, samples=n)
                    for module, n in phase_modules.get(phase, {}).most_common()]) for phase, count in phase_samples.items()},
                root_off_cpu_reasons=[dict(thread_state=state, block_reason=block, duration_ns=duration)
                    for (state, block), duration in reasons.most_common() if duration],
                diagnostics=diagnostics, warnings=[row for row in diagnostics if row["severity"] == 2], diagnostic_errors=diagnostic_errors,
                limitations=["Only CpuCycles=1 periodic leaf samples are used; context-switch stack/frame counts are not hotspot weights",
                             "Unresolved leaf-module sampling is not Python function attribution or proof of spin-wait",
                             "Scheduled-in intervals are reconstructed from transitions, not inferred from threadState alone",
                             "Across-worker sums are not parent wall decomposition, production utilization or recoverable time",
                             "Separate CPU/CUDA traces do not provide simultaneous CPU/GPU attribution; losslessness is unproved"])


def inspect(connection, workload, trace_mode="cuda"):
    labels = label_coverage(connection, workload)
    if trace_mode == "cpu":
        return inspect_cpu(connection, workload, labels)
    if trace_mode != "cuda":
        raise ValueError("only isolated CUDA/CPU captures are supported")
    tables, window, threads, phases, _, errors = labels
    required = {"CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_RUNTIME"}
    if required - tables:
        raise ValueError(f"CUDA-only analysis requires SQLite tables: {sorted(required - tables)}")
    start, end = window["start"], window["end"]
    bounds = start, end
    per_pid = {}
    for table, key in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernels"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "copies")):
        crossing = connection.execute(f"SELECT count(*) FROM {table} WHERE start<? AND end>? AND (start<? OR end>?)", (end, start, start, end)).fetchone()[0]
        if crossing:
            errors.append(f"{key}: {crossing} GPU events cross measurement boundaries")
        for row in connection.execute(f"SELECT globalPid,count(*) AS count,sum(end-start) AS duration_ns FROM {table} WHERE start>=? AND end<=? GROUP BY globalPid", bounds):
            pid = decode_pid(row["globalPid"])
            if pid not in threads:
                errors.append(f"GPU activity from unexpected PID {pid}")
            per_pid.setdefault(pid, {})[key] = dict(count=row["count"], duration_ns=row["duration_ns"])
    if any(per_pid.get(pid, {}).get("kernels", {}).get("count", 0) == 0 for pid in threads):
        errors.append("missing GPU kernels from an expected worker")
    correlations = {}
    for table, key, prefix in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernels", "cudaLaunchKernel%"),
                               ("CUPTI_ACTIVITY_KIND_MEMCPY", "copies", "cudaMemcpyAsync%")):
        # Match process identity as well as correlationId; IDs are not globally unique.
        unmatched = connection.execute(f"""
            SELECT count(*) FROM {table} g LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME a
              ON a.correlationId=g.correlationId AND (a.globalTid & -16777216)=g.globalPid
              AND a.start>=? AND a.end<=?
            WHERE g.start>=? AND g.end<=? AND a.correlationId IS NULL
        """, bounds + bounds).fetchone()[0]
        launches = connection.execute("""
            SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON a.nameId=s.id
            WHERE a.start>=? AND a.end<=? AND s.value LIKE ?
        """, bounds + (prefix,)).fetchone()[0]
        activities = sum(row.get(key, {}).get("count", 0) for row in per_pid.values())
        correlations[key] = dict(recorded_api_calls=launches, gpu_events=activities, unmatched_events=unmatched)
        if unmatched or launches != activities:
            errors.append(f"{key}: recorded CUDA API/GPU count or PID/correlation mismatch")
    runtime = [dict(row) for row in connection.execute("""
        SELECT s.value AS name,count(*) AS count,sum(a.end-a.start) AS duration_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME a LEFT JOIN StringIds s ON a.nameId=s.id
        WHERE a.start>=? AND a.end<=? GROUP BY a.nameId ORDER BY duration_ns DESC
    """, bounds)]
    copies = [dict(row) for row in connection.execute("""
        SELECT copyKind,count(*) AS count,sum(bytes) AS bytes,sum(end-start) AS duration_ns
        FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? GROUP BY copyKind
    """, bounds)]
    kernels = [dict(row) for row in connection.execute("""
        SELECT s.value AS name,count(*) AS count,sum(k.end-k.start) AS duration_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k.demangledName=s.id
        WHERE k.start>=? AND k.end<=? GROUP BY k.demangledName ORDER BY duration_ns DESC LIMIT 20
    """, bounds)]
    gpu_intervals = connection.execute("""
        SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?
        UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? ORDER BY start
    """, bounds + bounds)
    gpu_union = interval_union(gpu_intervals)
    diagnostics = [dict(row) for row in connection.execute("SELECT * FROM DIAGNOSTIC_EVENT")] if "DIAGNOSTIC_EVENT" in tables else []
    warnings = [row for row in diagnostics if row["severity"] == 2]
    diagnostic_errors = [row for row in diagnostics if row["severity"] == 3]
    return dict(kind="native_selfplay_coverage", known_root_label_coverage_passed=not errors,
                gpu_event_completeness_verified=False, lossless_trace_verified=False, errors=errors,
                recorded_cuda_correlations=correlations,
                window=window, window_seconds=(end-start)/1e9, per_pid=per_pid,
                phases=[dict(pid=pid, label=label, **values) for (pid, label), values in sorted(phases.items())],
                runtime=runtime, copies=copies, top_kernels=kernels,
                gpu_kernel_copy_union_seconds=gpu_union/1e9, gpu_kernel_copy_union_fraction=gpu_union/(end-start),
                diagnostics=diagnostics, warnings=warnings, diagnostic_errors=diagnostic_errors,
                limitations=["Observed GPU union excludes unrecorded activity and is not utilization or theoretical speedup",
                             "Phase/API sums overlap across workers; do not divide them by parent wall as a wall-time decomposition",
                             "No native CPU stack evidence; long synchronous CUDA calls do not alone prove spin-wait",
                             "Root/phase label counts are checked; full kernel-event completeness is not independently proved"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="coverage.json", help="New JSON filename only; never overwrite evidence")
    args = parser.parse_args()
    directory = args.capture_dir.resolve()
    pilots = Path(__file__).resolve().parents[1] / "data/pilots"
    if pilots.resolve() not in directory.parents:
        parser.error("require a capture under this workspace data/pilots")
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".json"):
        parser.error("output-name must be a JSON filename without directories")
    output = directory / args.output_name
    if output.exists():
        parser.error("refusing to overwrite coverage.json")
    metadata = json.loads((directory / "capture.json").read_text(encoding="utf-8-sig"))
    if metadata.get("capture_scope") != "lifecycle" or metadata.get("trace_mode") not in ("cuda", "cpu"):
        parser.error("this analyzer requires lifecycle isolated CUDA/CPU capture")
    if (metadata.get("exit_code") != 0 or metadata.get("process_failed") or metadata.get("process_timed_out")
            or not metadata.get("pipes_complete") or metadata.get("sqlite_export", {}).get("failed") is not False
            or metadata.get("startup_probe", {}).get("python_marker_seen") is not True):
        parser.error("capture process did not complete")
    workload = json.loads((directory / "workload/result.json").read_text(encoding="utf-8"))
    database = directory / "native.sqlite"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        result = inspect(connection, workload, metadata["trace_mode"])
    with database.open("rb") as handle:
        result["sqlite_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    result["workload_sha256"] = hashlib.sha256((directory / "workload/result.json").read_bytes()).hexdigest()
    result["capture_sha256"] = hashlib.sha256((directory / "capture.json").read_bytes()).hexdigest()
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(output=str(output), known_root_label_coverage_passed=result["known_root_label_coverage_passed"],
                          known_cpu_coverage_passed=result.get("known_cpu_coverage_passed"),
                          window_seconds=result["window_seconds"], gpu_union_fraction=result.get("gpu_kernel_copy_union_fraction"),
                          warnings=len(result["warnings"]), diagnostic_errors=len(result["diagnostic_errors"]), errors=result["errors"])))
    if not result["known_root_label_coverage_passed"] or not result.get("known_cpu_coverage_passed", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
