"""Inspect immutable repetition-board reuse without changing preparation outputs."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import concurrent.futures as futures
import copy
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import azero as az
import run_records as rr
from tools import profile_native_selfplay as native
from tools import profile_selfplay as probe
from tools.measure_search_reuse import verify_hashes
from tools.measure_selfplay_attribution import validate_config
from tools.measure_selfplay_round import append_event, runtime_identity
from tools.measure_sync_baseline import exact_rows, gpu_snapshot

ORIGINAL = az._repetition_entries
CAPACITY = 4096


def standard_board(board):
    return (type(board) is tuple and len(board) == 90
            and all(type(piece) is int and 0 <= piece <= 31 for piece in board))


class IdentityTracker:
    """Bounded per-root shadow table; no result reuse and no production cache."""
    def __init__(self, capacity=CAPACITY):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("require a positive integer capacity")
        self.capacity, self.boards = capacity, {}
        self.counts, self.history_lengths, self.entry_lengths = Counter(), Counter(), Counter()

    def observe(self, info, entries):
        self.counts["calls"] += 1
        if entries is None:
            self.counts["incomplete_calls"] += 1
            return
        self.counts["complete_calls"] += 1
        self.history_lengths[len(info.position_counts)] += 1
        self.entry_lengths[len(entries)] += 1
        current = tuple(info.board)
        occupied = sum(piece != 0 for piece in current)
        dark = sum(az.jq.is_dark(piece) for piece in current if piece)
        self.counts["missing_current_calls"] += int((info.side, current) not in info.position_counts)
        for (_, board), _count in info.position_counts.items():
            self.counts["board_queries"] += 1
            key = id(board)
            cached = self.boards.get(key)
            hit = cached is not None and cached[0] is board
            if hit:
                self.counts["identity_hits"] += 1
            else:
                self.counts["identity_misses"] += 1
                if not standard_board(board):
                    self.counts["unsupported_queries"] += 1
                    continue
                old_occupied = sum(piece != 0 for piece in board)
                old_dark = sum(az.jq.is_dark(piece) for piece in board if piece)
                cached = [board, old_occupied, old_dark, False]
                if len(self.boards) == self.capacity:
                    self.boards.pop(next(iter(self.boards)))
                    self.counts["evictions"] += 1
                self.boards[key] = cached
            if cached[1] > occupied or cached[2] > dark:
                self.counts["filtered_queries"] += 1
                continue
            self.counts["converted_queries"] += 1
            self.counts["repeated_conversions"] += int(cached[3])
            cached[3] = True
        self.counts["peak_boards"] = max(self.counts["peak_boards"], len(self.boards))

    def result(self):
        return dict(capacity=self.capacity, counts=dict(self.counts),
                    history_lengths=dict(self.history_lengths), entry_lengths=dict(self.entry_lengths))


def inspect_task(task):
    fixture, strong, seed = task
    tracker = IdentityTracker()
    first, last, timings = [], deque(maxlen=8), []
    calls = 0

    def observed(info):
        nonlocal calls
        sampled = calls % 64 == 0
        begin = time.perf_counter_ns() if sampled else None
        entries = ORIGINAL(info)
        if sampled:
            timings.append(time.perf_counter_ns()-begin)
        calls += 1
        tracker.observe(info, entries)
        if len(first) < 8:
            first.append(info)
        last.append(info)
        return entries

    with mock.patch.object(az, "_repetition_entries", observed):
        row = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    row.update(pid=os.getpid(), inspection=tracker.result(), sampled_preparation_ns=timings)
    return row, copy.deepcopy(first+list(last))


def cached_lookup(board, table):
    entry = table.get(id(board))
    if entry is None or entry[0] is not board:
        raise AssertionError("component benchmark requires hot identity hits")
    return entry


def component_micro(cases):
    boards = [board for info in cases if info.position_counts is not None
              for (_, board) in info.position_counts if standard_board(board)]
    if not boards:
        raise ValueError("require real standard repetition boards")
    before = probe.digest(boards)
    table = {id(board): (board, az._public_material_signature(board), bytes(board)) for board in boards}
    # Prewarm the existing value-keyed cache before each signature cohort too.
    trials = []
    modes = ("signature", "bytes", "identity_lookup")
    for repeat in range(8):
        for mode in modes if repeat % 2 == 0 else tuple(reversed(modes)):
            for board in boards:
                az._public_material_signature(board)
            cache_before = az._public_material_signature.cache_info()
            begin = time.perf_counter()
            for _ in range(16):
                if mode == "signature":
                    for board in boards:
                        az._public_material_signature(board)
                elif mode == "bytes":
                    for board in boards:
                        bytes(board)
                else:
                    for board in boards:
                        cached_lookup(board, table)
            seconds = time.perf_counter()-begin
            cache_after = az._public_material_signature.cache_info()
            trials.append(dict(repeat=repeat, mode=mode, seconds=seconds,
                               signature_hits=cache_after.hits-cache_before.hits,
                               signature_misses=cache_after.misses-cache_before.misses))
    if probe.digest(boards) != before:
        raise AssertionError("component measurement mutated board inputs")
    totals = {mode: sum(t["seconds"] for t in trials if t["mode"] == mode) for mode in modes}
    return dict(board_queries=len(boards), unique_objects=len(table), calls_per_mode=len(boards)*128,
                trials=trials, totals=totals,
                identity_vs_signature_gain=totals["signature"]/totals["identity_lookup"]-1,
                all_identity_hits=True, allocation_on_miss_included=False, full_preparation_measured=False,
                inputs_unchanged=True)


def summarize(rows, components):
    counts = Counter()
    for row in rows:
        for key, value in row["inspection"]["counts"].items():
            if key != "peak_boards":
                counts[key] += value
    queries = counts["board_queries"]
    hits = counts["identity_hits"]/queries if queries else 0
    conversion_reuse = counts["repeated_conversions"]/counts["converted_queries"] if counts["converted_queries"] else 0
    gate = (counts["unsupported_queries"] == 0 and hits >= .80
            and components["identity_vs_signature_gain"] >= .20)
    return dict(counts=dict(counts), identity_hit_rate=hits, conversion_reuse_rate=conversion_reuse,
                prototype_worth_screening=gate, production_promotion=False,
                thresholds=dict(identity_hit_rate=.80, identity_vs_signature_gain=.20,
                                unsupported_queries=0), scope="per-root bounded FIFO shadow, not whole-game reuse")


def run(directory, reference_path):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference.get("kind") != "python_sampling_workload" or not reference.get("production_unchanged")
            or not reference.get("exact_against_baseline")):
        raise ValueError("require exact current Python sampling plain reference")
    hashes = dict(reference["source_hashes"])
    verify_hashes(hashes)
    if (json.loads(json.dumps(runtime_identity())) != reference["identity"] or
            Path(sys.executable).resolve() != Path(reference["python"]).resolve() or not torch.cuda.is_available()):
        raise ValueError("require identical interpreter/CUDA/backend/precision")
    for path in (reference_path, Path(__file__).resolve(), ROOT/"test_measure_repetition_preparation.py",
                 ROOT/"data/pilots/repetition_preparation_20261009_iter28_protocol.md"):
        hashes[str(path)] = rr.file_record(path)["sha256"]
    source = ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs"
    checkpoint = torch.load(source/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
    config = az.AZConfig(**checkpoint["config"])
    validate_config(config)
    config.data_dir, config.model_dir, config.log_dir = (str(directory/n) for n in ("samples", "models", "logs"))
    with (source.parent/"measure.fixtures.pkl").open("rb") as handle:
        fixtures = pickle.load(handle)["fixtures"]
    if len(fixtures) != 18 or [f["metadata"]["index"] for f in fixtures] != list(range(18)):
        raise ValueError("require trusted current18 fixtures")
    torch.set_num_threads(1)
    start = time.perf_counter()
    result = dict(kind="repetition_preparation_inspection", source_hashes=hashes, identity=runtime_identity(),
                  config=asdict(config), python=sys.executable, production_training=False, gpu_before=gpu_snapshot())
    with futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
            initializer=native.init_worker, initargs=(checkpoint["model"], asdict(config), runtime_identity())) as pool:
        warm = list(pool.map(native.root_task, [("plain", f, strong, 202610101+f["metadata"]["index"])
                                              for f in fixtures for strong in (False, True)]))
        if len({r["pid"] for r in warm}) != 8:
            raise RuntimeError("warmup missed workers")
        tasks = [(f, strong, 202610110+f["metadata"]["index"]) for f in fixtures for strong in (False, True)]
        outputs = list(pool.map(inspect_task, tasks))
    result["rows"] = [r for r, _cases in outputs]
    if exact_rows(result["rows"]) != exact_rows(reference["groups"][0]["rows"]):
        raise AssertionError("shadow inspection changed root outputs")
    cases = [info for _r, snapshots in outputs for info in snapshots]
    with (directory/"cases.pkl").open("xb") as handle:
        pickle.dump(cases, handle)
    hashes[str(directory/"cases.pkl")] = rr.file_record(directory/"cases.pkl")["sha256"]
    gc.collect()
    result["components"] = component_micro(cases)
    result["summary"] = summarize(result["rows"], result["components"])
    verify_hashes(hashes)
    result.update(roots_exact=True, production_unchanged=True, elapsed_seconds=time.perf_counter()-start)
    with (directory/"result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    append_event(directory/"events.jsonl", "complete", result["summary"])
    probe.emit("inspection_complete", result["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    native.validate_output(directory, (ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs").resolve())
    directory.mkdir(parents=True)
    run(directory, args.reference_file.resolve())


if __name__ == "__main__":
    main()
