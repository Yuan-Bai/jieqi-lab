"""Real spawn failures with sleeping UCI children; no training or model search."""
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from unittest import mock

import psutil
import pytest
import torch

import evaluation_runtime
import jieqi as jq
import pika_engine
from tools import eval as evaluator, paired_stats


def _record_process(path, pid):
    path.write_text(json.dumps(dict(pid=pid, created=psutil.Process(pid).create_time())), encoding="utf-8")


def _test_worker_init(*args):
    # Exercise the production ownership wrapper, replacing only model loading.
    with mock.patch.object(evaluator, "_init_eval_worker"):
        evaluator._start_eval_worker(*args)
    root = Path(os.environ["JIEQI_EVAL_CLEANUP_TEST"])
    _record_process(root / f"worker-{os.getpid()}.json", os.getpid())


def _wait_for(path):
    deadline = time.monotonic() + 15
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"test worker did not start: {path}")
        time.sleep(.01)


def _synthetic_game(task):
    spec, pair, seed, side, _ = task
    root = Path(os.environ["JIEQI_EVAL_CLEANUP_TEST"])
    scenario = os.environ["JIEQI_EVAL_CLEANUP_SCENARIO"]
    index = pair * 2 + (side == jq.BLACK)
    if scenario == "success":
        time.sleep(.1 if index % 2 == 0 else .01)
        score = [1, -1, 0, 1][index % 4]
        outcome = (jq.RESULT_DRAW if not score else
                   jq.RESULT_RED_WIN if (score > 0) == (side == jq.RED) else jq.RESULT_BLACK_WIN)
        return dict(type="game", opponent=spec, pair=pair, seed=seed, score=score, outcome=outcome,
                    neural_side="red" if side == jq.RED else "black")
    failure_index = 0 if scenario in ("first", "parent_interrupt") else 1
    if index == failure_index:
        _wait_for(root / f"engine-{1 - index}.json")
        (root / "failure-time.txt").write_text(str(time.monotonic()))
        if scenario == "crash":
            os._exit(23)
        if scenario == "parent_interrupt":
            return dict(type="game", opponent=spec, pair=pair, seed=seed, score=0,
                        outcome=jq.RESULT_DRAW, neural_side="red")
        raise pika_engine.PikaError("injected protocol failure")
    engine = pika_engine.UciProcess([sys.executable, "-u", "-c", "import time; time.sleep(60)"], 60)
    try:
        _record_process(root / f"engine-{index}.json", engine.process.pid)
        engine.wait_for("readyok")
    finally:
        engine.close()


def _run_evaluation(root, scenario):
    root = Path(root)
    os.environ["JIEQI_EVAL_CLEANUP_TEST"] = str(root)
    os.environ["JIEQI_EVAL_CLEANUP_SCENARIO"] = scenario
    if scenario == "nested":
        # Training on Windows already owns a job: evaluation must work in a nested job.
        evaluation_runtime.isolate_worker()
    real_executor = concurrent.futures.ProcessPoolExecutor
    unrelated = mp.get_context("spawn").Process(target=time.sleep, args=(60,))
    unrelated.start()
    _record_process(root / "unrelated.json", unrelated.pid)

    def pool_factory(**kwargs):
        kwargs["initializer"] = _test_worker_init
        return real_executor(**kwargs)

    result = {}
    try:
        with mock.patch.object(evaluator.futures, "ProcessPoolExecutor", side_effect=pool_factory), \
             mock.patch.object(evaluator, "_eval_game_task", _synthetic_game):
            interrupt = (mock.patch.object(evaluator, "tally_game", side_effect=KeyboardInterrupt("injected interrupt"))
                         if scenario == "parent_interrupt" else nullcontext())
            with interrupt:
                try:
                    result["summaries"] = evaluator.evaluate(
                        torch.nn.Linear(1, 1), torch.device("cpu"), ["random"], 8, 1, 17, 1,
                        str(root / "eval.jsonl"), quiet=True, workers=2, bootstrap=100)
                except BaseException as error:
                    result.update(error_type=type(error).__name__, error=str(error),
                                  cleanup_notes=getattr(error, "__notes__", []))
                    result["error_delay"] = time.monotonic() - float((root / "failure-time.txt").read_text())
        result["unrelated_survived"] = unrelated.is_alive()
    finally:
        unrelated.kill()
        unrelated.join(5)
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")


def _live_recorded_process(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    try:
        process = psutil.Process(record["pid"])
        if process.create_time() == record["created"] and process.status() != psutil.STATUS_ZOMBIE:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def _run_probe(tmp_path, scenario):
    process = mp.get_context("spawn").Process(target=_run_evaluation, args=(str(tmp_path), scenario))
    process.start()
    try:
        process.join(30)
        assert not process.is_alive(), "evaluation did not stop within the test deadline"
        assert process.exitcode == 0
        result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
        assert result["unrelated_survived"], result
        owned = [*tmp_path.glob("worker-*.json"), *tmp_path.glob("engine-*.json")]
        assert len(list(tmp_path.glob("worker-*.json"))) == 2
        deadline = time.monotonic() + 3
        while any(_live_recorded_process(path) for path in owned) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not any(_live_recorded_process(path) for path in owned), "owned process survived cleanup"
        return result
    finally:
        # A regression must not leave this test's synthetic processes behind.
        if process.is_alive():
            children = psutil.Process(process.pid).children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
        process.join(5)
        for path in [*tmp_path.glob("worker-*.json"), *tmp_path.glob("engine-*.json"), tmp_path / "unrelated.json"]:
            if path.exists():
                child = _live_recorded_process(path)
                if child is not None:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass


@pytest.mark.parametrize("scenario, error_type", [
    ("first", pika_engine.PikaError.__name__),
    ("later", pika_engine.PikaError.__name__),
    ("nested", pika_engine.PikaError.__name__),
    ("crash", BrokenProcessPool.__name__),
    ("parent_interrupt", KeyboardInterrupt.__name__),
])
def test_failure_stops_owned_workers_and_engines(tmp_path, scenario, error_type):
    result = _run_probe(tmp_path, scenario)
    assert result["error_type"] == error_type, result
    assert result["error_delay"] < 6, result
    assert not result["cleanup_notes"], result
    assert list(tmp_path.glob("engine-*.json")), "no engine child was exercised"
    assert not list(tmp_path.glob("runs/*/complete.json")), "failed evaluation marked complete"


def test_success_keeps_pair_order_statistics_and_completion(tmp_path):
    result = _run_probe(tmp_path, "success")
    records = [json.loads(line) for line in (tmp_path / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
    games = [row for row in records if row["type"] == "game"]
    assert [(row["pair"], row["neural_side"]) for row in games] == [
        (pair, side) for pair in range(4) for side in ("red", "black")]
    stats = paired_stats.analyze(str(tmp_path / "eval.jsonl"), 100, 100, 17)
    for field in ("score", "pair_mean", "bootstrap_ci95", "pentanomial"):
        assert result["summaries"][0][field] == stats[field]
    assert len(list(tmp_path.glob("runs/*/complete.json"))) == 1
