"""Task-only measurement preserves delegation, result order and timing units."""
import concurrent.futures as futures
from dataclasses import asdict
from pathlib import Path
import random
import sys
import tempfile
import threading
from unittest import mock
import uuid

import pytest

import azero as az
import jieqi as jq
from tools import measure_selfplay_round as measure


def game(seed, start, end, *, pid=1, strong=False):
    return dict(seed=seed, start=start, end=end, pid=pid, strong=strong,
        wall_seconds=end-start, process_cpu_seconds=(end-start)/2,
        moves=10, samples=10, leaf_total=100, leaf_unique=80,
        cache_hits=90, cache_misses=10, cache_size=20)


def test_active_intervals_and_drained_tail():
    games = [game(11, 1, 5), game(12, 2, 8, pid=2, strong=True)]
    receipts = [dict(seed=11, received=5.1), dict(seed=12, received=8.2)]
    summary = measure.summarize_games(games, receipts, 10, 2)
    assert summary["active_game_seconds"] == {0: 3, 1: 4, 2: 3}
    assert summary["tail_active_game_seconds"] == {0: 0, 1: 3, 2: 3}
    assert summary["drained_queue_tail_seconds"] == 6
    assert summary["worker_game_occupancy"] == 0.5
    assert summary["worker_cpu_core_equivalents"] == 0.5
    assert summary["strong_games"] == 1


def test_summary_rejects_failed_or_ambiguous_receipts():
    import pytest
    games = [game(11, 1, 5)]
    for receipts in ([], [dict(seed=11, received=5, error="failed")]):
        with pytest.raises(AssertionError):
            measure.summarize_games(games, receipts, 10, 2)
    with pytest.raises(AssertionError, match="ambiguous"):
        measure.summarize_games(games*2, [], 10, 2)


def test_measured_worker_calls_original_once_and_returns_same_object():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"worker.jsonl"
        output = az.SelfPlayResult([], jq.RESULT_DRAW, 17, 100, 80)
        task = (83, True)
        with mock.patch.object(measure, "WORKER_LOG", path), mock.patch.object(measure, "WORKER_ORIGIN", 1.0), mock.patch.object(
                az, "_selfplay_worker", return_value=output) as original, mock.patch.object(
                measure.time, "perf_counter", side_effect=[2.0, 5.0]), mock.patch.object(
                measure.time, "process_time", side_effect=[10.0, 11.0]):
            assert measure.measured_worker(task) is output
        original.assert_called_once_with(task)
        row = measure.read_events(path, "game")[0]
        assert row["start"] == 1 and row["end"] == 4
        assert row["wall_seconds"] == 3 and row["process_cpu_seconds"] == 1
        assert row["seed"] == 83 and row["strong"]


def test_executor_receipt_keeps_submission_index_not_completion_order():
    with tempfile.TemporaryDirectory() as directory:
        executor = measure.RoundExecutor.__new__(measure.RoundExecutor)
        executor.task_index = 0
        executor.receipts = []
        executor.receipt_lock = threading.Lock()
        executor.directory, executor.origin = directory, 0
        first, second = futures.Future(), futures.Future()
        with mock.patch.object(measure.EXECUTOR, "submit", side_effect=[first, second]) as submit:
            assert executor.submit(az._selfplay_worker, (11, False)) is first
            assert executor.submit(az._selfplay_worker, (12, True)) is second
        second.set_result("second")
        first.set_result("first")
        assert [r["index"] for r in executor.receipts] == [1, 0]
        assert [r["seed"] for r in executor.receipts] == [12, 11]
        assert submit.call_args_list[0].args == (measure.measured_worker, (11, False))
        assert [r["index"] for r in measure.read_events(Path(directory)/"receipts.jsonl", "received")] == [1, 0]


def test_resource_summary_excludes_post_round_samples():
    rows = [dict(seconds=s, gpu=[{k:s for k in measure.GPU_FIELDS}],
                 processes=[dict(rss_bytes=100)], system=dict(available_bytes=1000, cpu_percent=30))
            for s in (1, 3, 20)]
    summary = measure.resource_summary(rows, 5)
    assert summary["samples"] == summary["gpu_samples"] == 2
    assert summary["gpu"]["utilization.gpu"]["mean"] == 2


def test_hotspot_instrumentation_restores_methods():
    from contextlib import ExitStack
    original = az.PUCTNode.expand
    with ExitStack() as stack:
        measure.RoundBoundaries().install(stack)
        assert az.PUCTNode.expand is not original
    assert az.PUCTNode.expand is original


def test_serial_worker_option_rejected_before_loading_checkpoint(capsys):
    directory = measure.ROOT/"data/pilots"/uuid.uuid4().hex
    argv = ["measure_selfplay_round", "--output-dir", str(directory),
            "--fixtures", "unused.pkl", "--workers", "1"]
    with mock.patch.object(sys, "argv", argv), mock.patch.object(measure.rr, "file_record") as record:
        with pytest.raises(SystemExit) as error:
            measure.main()
    assert error.value.code == 2
    assert "at least two workers" in capsys.readouterr().err
    record.assert_not_called()
    assert not directory.exists()


def test_cpu_config_rejected_before_output_creation(capsys):
    directory = measure.ROOT/"data/pilots"/uuid.uuid4().hex
    argv = ["measure_selfplay_round", "--output-dir", str(directory), "--fixtures", "unused.pkl"]
    config = az.AZConfig(self_play_device="cpu")
    with mock.patch.object(sys, "argv", argv), mock.patch.object(jq, "RULES_BACKEND", "cython"), mock.patch.object(
            jq, "LEGAL_BACKEND", "cython"), mock.patch.object(az, "SEARCH_BACKEND", "python"), mock.patch.object(
            measure.torch.cuda, "is_available", return_value=True), mock.patch.object(
            measure.rr, "file_record", return_value=dict(sha256="abc")), mock.patch.object(
            measure.torch, "load", return_value=dict(config=asdict(config))):
        with pytest.raises(SystemExit) as error:
            measure.main()
    assert error.value.code == 2
    assert "CUDA device 0" in capsys.readouterr().err
    assert not directory.exists()


def test_hotspot_wrappers_keep_native_fast_paths():
    if jq.LEGAL_BACKEND != "cython" or jq.RULES_BACKEND != "cython":
        pytest.skip("native rules/legal required")
    from contextlib import ExitStack
    state = jq.GameState.initial(rng=random.Random(83))
    expected = state.legal_moves()
    with ExitStack() as stack:
        measure.RoundBoundaries().install(stack)
        for name in ("_iter_legal_moves_fast_python", "_is_square_attacked_python"):
            stack.enter_context(mock.patch.object(jq.GameState, name, side_effect=AssertionError("unexpected fallback")))
        assert state.legal_moves() == expected
