"""Read-only coverage gate for the pinned synthetic Nsight compatibility probe."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from tools.probe_native_runtime import CAPTURE_RANGE


def decode_pid(global_id):
    # Nsight GlobalId: low 24 bits TID, next 24 PID, upper bits machine ID.
    return (int(global_id) >> 24) & 0xFFFFFF


def cpu_coverage(connection, events, window, results, tables):
    """Compatibility only: stack-event counts are not CPU time or hotspot shares."""
    required = {"COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS", "SCHED_EVENTS", "StringIds"}
    missing = sorted(required - tables)
    if missing or window is None:
        return dict(errors=[f"CPU tables/window unavailable: {missing}"], per_pid=[])
    bounds = (window["start"], window["end"])
    errors, per_pid = [], []
    for result in results:
        pid, tid = result["pid"], result["thread_id"]
        global_tids = {event["globalTid"] for event in events if event["pid"] == pid
                       and (event["globalTid"] & 0xFFFFFF) == tid}
        if len(global_tids) != 1:
            errors.append(f"PID {pid}: require the workload main thread's NVTX identity")
            continue
        global_tid = next(iter(global_tids))
        samples = [dict(row) for row in connection.execute("""
            SELECT id,start,threadState,cpuCycles FROM COMPOSITE_EVENTS
            WHERE globalTid=? AND start>=? AND start<=?
        """, (global_tid,) + bounds)]
        frames = [dict(row) for row in connection.execute("""
            SELECT f.id,f.stackDepth,f.originalIP,f.unresolved,f.kernelMode,m.value AS module
            FROM SAMPLING_CALLCHAINS f JOIN COMPOSITE_EVENTS e ON f.id=e.id
            LEFT JOIN StringIds m ON f.module=m.id
            WHERE e.globalTid=? AND e.start>=? AND e.start<=?
        """, (global_tid,) + bounds)]
        stack_ids = {frame["id"] for frame in frames}
        mapped_stack_ids = {frame["id"] for frame in frames if frame["module"] and frame["originalIP"] is not None}
        sched_count = connection.execute("SELECT count(*) FROM SCHED_EVENTS WHERE globalTid=? AND start>=? AND start<=?",
                                         (global_tid,) + bounds).fetchone()[0]
        lifecycle_count = connection.execute("SELECT count(*) FROM COMPOSITE_EVENTS WHERE globalTid=?", (global_tid,)).fetchone()[0]
        if not samples or not mapped_stack_ids or sched_count == 0:
            errors.append(f"PID {pid}: missing window main-thread stack/module or scheduling coverage")
        per_pid.append(dict(pid=pid, thread_id=tid, global_tid=global_tid,
                            lifecycle_main_thread_stack_events=lifecycle_count, window_stack_events=len(samples),
                            window_events_with_frames=len(stack_ids), window_events_with_modules=len(mapped_stack_ids),
                            window_frame_rows=len(frames), unresolved_frame_rows=sum(bool(frame["unresolved"]) for frame in frames),
                            unknown_kernel_mode_rows=sum(frame["kernelMode"] is None for frame in frames),
                            window_sched_events=sched_count, positive_cpu_cycle_events=sum(row["cpuCycles"] > 0 for row in samples),
                            modules=sorted({frame["module"] for frame in frames if frame["module"]})))
    return dict(errors=errors, per_pid=per_pid,
                limitations=["Tiny window proves capture compatibility, not hotspot estimates or losslessness",
                             "Composite events may contain context-switch backtraces; event counts are not CPU time",
                             "Null kernelMode is unknown, not proof of user/kernel execution; unresolved IPs are not resolved functions",
                             "SQLite can contain other processes; only workload PID/main-thread/window are used here"])


def inspect(connection, workload, capture_scope="triggered", trace_mode="cuda"):
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def rows(table):
        return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')] if table in tables else []

    strings = {row["id"]: row["value"] for row in rows("StringIds")}
    events = rows("NVTX_EVENTS")
    for event in events:
        event["label"] = event.get("text") or strings.get(event.get("textId"))
        event["pid"] = decode_pid(event["globalTid"])
    kernels = rows("CUPTI_ACTIVITY_KIND_KERNEL")
    copies = rows("CUPTI_ACTIVITY_KIND_MEMCPY")
    runtime = rows("CUPTI_ACTIVITY_KIND_RUNTIME")
    diagnostics = rows("DIAGNOSTIC_EVENT")
    errors = []
    if trace_mode not in ("cuda", "cpu"):
        errors.append("only isolated CUDA/CPU compatibility analysis is supported")
    mode = workload.get("mode")
    expected_rows = {"single": 1, "spawn": 2}.get(mode)
    results = workload.get("rows", [])
    if workload.get("kind") != "native_runtime_probe" or expected_rows is None or len(results) != expected_rows:
        errors.append("require a completed single/spawn synthetic workload")
    if workload.get("models_loaded") is not False or workload.get("training_started") is not False:
        errors.append("probe must not load models or train")
    if any(row.get("exact") is not True or row.get("dtype") != "torch.float32"
           or row.get("tf32_matmul") is not False or row.get("tf32_cudnn") is not True for row in results):
        errors.append("synthetic numerical/precision gate failed")
    pids = [row["pid"] for row in results]
    if len(set(pids)) != len(pids) or (mode == "single" and pids != [workload.get("pid")]):
        errors.append("workload PID identity is inconsistent")
    if mode == "spawn" and workload.get("pid") in pids:
        errors.append("spawn workload must come from two distinct children")
    windows = [event for event in events if event["label"] == CAPTURE_RANGE
               and event["pid"] == workload.get("pid")]
    window = windows[0] if len(windows) == 1 else None
    if window is None or window.get("end") is None or window["end"] <= window["start"]:
        errors.append("one closed parent capture window is required")
        window = None

    def inside(event):
        return window is not None and event.get("end") is not None and (
            window["start"] <= event["start"] <= event["end"] <= window["end"])

    total_counts = dict(kernels=len(kernels), copies=len(copies), runtime=len(runtime), nvtx=len(events))
    if capture_scope not in ("triggered", "lifecycle"):
        errors.append("unknown capture scope")
    if capture_scope == "lifecycle" and window is not None:
        # Keep complete lifecycle in SQLite; analyze only the closed measurement window.
        for row in kernels + copies + runtime:
            if row.get("end") is not None and row["start"] < window["end"] and row["end"] > window["start"] and not inside(row):
                errors.append("CUDA event crosses a measurement window boundary")
        kernels = [row for row in kernels if inside(row)]
        copies = [row for row in copies if inside(row)]
        runtime = [row for row in runtime if inside(row)]
        events = [row for row in events if inside(row)]
    if any(decode_pid(row["globalPid"]) not in pids for row in kernels + copies):
        errors.append("unexpected GPU process in capture")
    if any(not inside(row) for row in kernels + copies + runtime):
        errors.append("CUDA events missing an end or outside the parent capture window")
    per_pid = []
    for pid in pids:
        forwards = [event for event in events if event["pid"] == pid and event["label"] == "jieqi/compat/forward"]
        readbacks = [event for event in events if event["pid"] == pid and event["label"] == "jieqi/compat/readback"]
        pid_kernels = [row for row in kernels if decode_pid(row["globalPid"]) == pid]
        pid_copies = [row for row in copies if decode_pid(row["globalPid"]) == pid]
        pid_runtime = [row for row in runtime if decode_pid(row["globalTid"]) == pid]
        api_by_correlation = {row["correlationId"]: row for row in pid_runtime}
        if len(forwards) != 8 or len(readbacks) != 1 or any(not inside(event) for event in forwards + readbacks):
            errors.append(f"PID {pid}: require all eight forwards and one closed readback")
        # Exact counts are for this pinned torch/cuDNN/4060 workload, not a general CUDA invariant.
        if trace_mode == "cuda":
            if len(pid_kernels) != 16 or len(pid_copies) != 2:
                errors.append(f"PID {pid}: expected sixteen kernels and two copies on this pinned stack")
            if sorted((row["copyKind"], row["bytes"]) for row in pid_copies) != [(2, 14400), (2, 65536)]:
                errors.append(f"PID {pid}: expected two full device-to-host result buffers")
            if any(row["correlationId"] not in api_by_correlation for row in pid_kernels + pid_copies):
                errors.append(f"PID {pid}: GPU event without matching CUDA runtime API")
        if trace_mode == "cuda" and len(forwards) == 8:
            for forward in forwards:
                correlated = [row for row in pid_kernels if row["correlationId"] in api_by_correlation
                              and forward["start"] <= api_by_correlation[row["correlationId"]]["start"]
                              <= api_by_correlation[row["correlationId"]]["end"] <= forward["end"]]
                if len(correlated) != 2:
                    errors.append(f"PID {pid}: forward interval lacks its two correlated kernel launches")
        per_pid.append(dict(pid=pid, kernels=len(pid_kernels), copies=len(pid_copies), runtime=len(pid_runtime),
                            forward_ranges=len(forwards), readback_ranges=len(readbacks)))
    cpu = None
    diagnostic_errors = [row for row in diagnostics if row["severity"] == 3]
    if trace_mode == "cpu":
        if kernels or copies or runtime:
            errors.append("CPU-only probe unexpectedly includes CUDA activity")
        cpu = cpu_coverage(connection, events, window, results, tables)
        errors.extend(cpu["errors"])
        if diagnostic_errors:
            errors.append("CPU-only capture has collector error diagnostics")
    warnings = [row for row in diagnostics if row["severity"] == 2]
    return dict(kind="native_runtime_coverage", mode=mode, trace_mode=trace_mode, capture_scope=capture_scope, known_workload_coverage_passed=not errors,
                lossless_trace_verified=False, errors=errors, warnings=warnings, diagnostic_errors=diagnostic_errors, diagnostics=diagnostics,
                cpu=cpu,
                capture_window=window, per_pid=per_pid,
                counts=dict(kernels=len(kernels), copies=len(copies), runtime=len(runtime), nvtx=len(events)),
                total_counts=total_counts,
                note="Synthetic compatibility only. Preserve collection warnings; no full-workload losslessness or speed claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.probe_dir.resolve()
    pilots = Path(__file__).resolve().parents[1] / "data/pilots"
    if pilots.resolve() not in directory.parents:
        parser.error("require a probe under this workspace data/pilots")
    output = directory / "coverage.json"
    if output.exists():
        parser.error("refusing to overwrite existing coverage.json")
    database = directory / "native.sqlite"
    metadata = json.loads((directory / "probe.json").read_text(encoding="utf-8-sig"))
    workload = json.loads((directory / "workload/result.json").read_text(encoding="utf-8"))
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        result = inspect(connection, workload, metadata.get("capture_scope", "triggered"), metadata.get("trace_mode", "cuda"))
    with database.open("rb") as handle:
        result["sqlite_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    result["workload_sha256"] = hashlib.sha256((directory / "workload/result.json").read_bytes()).hexdigest()
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(output=str(output), known_workload_coverage_passed=result["known_workload_coverage_passed"],
                          counts=result["counts"], warnings=len(result["warnings"]), diagnostic_errors=len(result["diagnostic_errors"]), errors=result["errors"])))
    if not result["known_workload_coverage_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
