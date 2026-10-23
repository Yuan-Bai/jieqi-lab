"""冻结对手来源、训练随机流隔离和评估提交边界。"""
import json
import logging
import random
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

import azero as az
import jieqi as jq
import run_records as rr
import test_training_checkpoint as checkpoint_tests
from tools import eval as evaluator


def make_pool(folder, model_bytes=b"frozen test model"):
    model = folder / "anchor.pt"
    model.write_bytes(model_bytes)
    path = folder / "pool.json"
    value = dict(version=1, id="test-pool", ruleset_version=jq.RULESET_VERSION,
                 models=[dict(path=model.name, sims=1, sha256=rr.file_record(model)["sha256"])])
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_pool_resolves_relative_paths_and_checks_hashes(tmp_path, monkeypatch):
    path, value = make_pool(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    pool = evaluator.load_opponent_pool(str(path))
    assert pool["opponents"] == [f"model:{tmp_path / 'anchor.pt'}:1"]
    assert pool["source"] == rr.file_record(path)
    (tmp_path / "anchor.pt").write_bytes(b"different model")
    with pytest.raises(ValueError, match="SHA256"):
        evaluator.load_opponent_pool(str(path))


@pytest.mark.parametrize("change", ["rules", "version", "empty", "duplicate", "budget", "hash", "not_object"])
def test_pool_rejects_invalid_baselines(tmp_path, change):
    path, value = make_pool(tmp_path)
    if change == "rules":
        value["ruleset_version"] -= 1
    elif change == "version":
        value["version"] += 1
    elif change == "empty":
        value["models"] = []
    elif change == "duplicate":
        value["models"] *= 2
    elif change == "budget":
        value["models"][0]["sims"] = 0
    elif change == "hash":
        value["models"][0]["sha256"] = "invalid"
    else:
        value = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        evaluator.load_opponent_pool(str(path))


def test_evaluator_rechecks_frozen_sources_before_starting(tmp_path):
    path, _ = make_pool(tmp_path)
    pool = evaluator.load_opponent_pool(str(path))
    (tmp_path / "anchor.pt").write_bytes(b"changed after preflight")
    output = tmp_path / "eval.jsonl"
    with pytest.raises(RuntimeError, match="冻结清单"):
        evaluator.evaluate(None, None, pool["opponents"], 2, 1, 7, 4, str(output),
                           expected_opponent_sources=pool["sources"])
    assert not output.exists()


def test_cli_selection_policy_and_conflicts():
    config = az.parse_args(["--eval-pool", "anchors.json", "--eval-games", "100", "--eval-seed", "123"])
    assert not config.legacy_ris_best
    assert (config.eval_mcts_simulations, config.eval_leaf_batch_size) == (1600, 16)
    assert config.eval_pool == "anchors.json" and config.eval_seed == 123
    assert az.parse_args(["--legacy-ris-best"]).legacy_ris_best
    for args in (["--eval-pool", "a", "--legacy-ris-best"], ["--eval-games", "-1"],
                 ["--eval-max-plies", "0"], ["--eval-seed", "-1"], ["--eval-leaf-batch-size", "0"]):
        with pytest.raises(SystemExit):
            az.parse_args(args)
    assert evaluator.parse_args(["--pool", "a"]).pool == "a"
    for args in (["--pool", "a", "--opponents", "random"], ["--pool", "a", "--quick"], ["--seed", "-1"]):
        with pytest.raises(SystemExit):
            evaluator.parse_args(args)


def test_declared_pool_search_rejects_mismatched_budgets_before_training(tmp_path):
    path, value = make_pool(tmp_path)
    value["search"] = dict(mcts_simulations=1600, leaf_batch_size=16)
    value["models"][0]["sims"] = 1600
    path.write_text(json.dumps(value))
    pool = evaluator.load_opponent_pool(str(path))
    evaluator.validate_pool_search(pool, 1600, 16)
    az.prepare_training_eval(az.AZConfig(eval_pool=str(path)))
    for sims, leaf in ((800, 16), (1600, 64)):
        with pytest.raises(ValueError, match="冻结池不一致"):
            az.prepare_training_eval(az.AZConfig(eval_pool=str(path), eval_mcts_simulations=sims,
                                                 eval_leaf_batch_size=leaf))
        with pytest.raises(ValueError, match="冻结池不一致"):
            evaluator.main(["--pool", str(path), "--model", "missing.pt", "--sims", str(sims),
                            "--leaf-batch-size", str(leaf)])
    value["models"][0]["sims"] = 800
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="相同模拟预算"):
        evaluator.load_opponent_pool(str(path))
    value["search"]["leaf_batch_size"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="正整数"):
        evaluator.load_opponent_pool(str(path))


def test_pool_failure_restores_rng_network_and_threads(tmp_path):
    path, _ = make_pool(tmp_path)
    pool = evaluator.load_opponent_pool(str(path))
    rng, net = random.Random(21), torch.nn.Linear(1, 1).train()
    config = az.AZConfig(eval_pool=str(path), data_dir=str(tmp_path), eval_device="cpu")
    saved = az.capture_training_rng(rng)
    weight, threads = rr.model_hash(net), torch.get_num_threads()

    def fail(clone, *args, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        rng.random()
        clone.weight.data.zero_()
        raise RuntimeError("engine failed")

    with mock.patch.object(evaluator, "evaluate", side_effect=fail), pytest.raises(RuntimeError, match="engine failed"):
        az.evaluate_training_pool(net, config, torch.device("cpu"), rng, pool, 0, {}, logging.getLogger("test"))
    checkpoint_tests.TestTrainingCheckpoint().equal_tree(saved, az.capture_training_rng(rng))
    assert rr.model_hash(net) == weight and net.training
    assert torch.get_num_threads() == threads
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="清单改变"):
        az.evaluate_training_pool(net, config, torch.device("cpu"), rng, pool, 0, {}, logging.getLogger("test"))


@pytest.fixture
def harness():
    instance = checkpoint_tests.TestTrainingCheckpoint()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    instance.setUp()
    try:
        yield instance
    finally:
        instance.tearDown()
        torch.set_num_threads(threads)


def install_tiny_pool(harness):
    path, _ = make_pool(harness.root)
    model = harness.root / "anchor.pt"
    az.save_model(checkpoint_tests.TinyNet(), str(model))
    value = json.loads(path.read_text())
    value["models"][0]["sha256"] = rr.file_record(model)["sha256"]
    path.write_text(json.dumps(value))
    return str(path)


def fake_game(neural_factory, opponent_factory, side, seed, max_plies):
    # 有意消耗全局 RNG，验证评估仍不会污染优化或下一轮自博弈。
    random.random()
    np.random.rand()
    torch.rand(1)
    return dict(seed=seed, neural_side="red" if side == jq.RED else "black",
                outcome=jq.RESULT_RED_WIN, score=1 if side == jq.RED else -1, plies=10, reason="checkmate")


def test_pool_training_matches_no_eval_rng_and_weights(harness):
    pool_path = install_tiny_pool(harness)
    plain = harness.loop_config(harness.root / "plain")
    config = replace(harness.loop_config(harness.root / "pool"), eval_pool=pool_path,
                     eval_games=4, eval_interval=1, eval_workers=1, eval_device="cpu", eval_mcts_simulations=1,
                     eval_leaf_batch_size=7)
    harness.seed()
    harness.run_loop(plain)
    harness.seed()
    with mock.patch.object(evaluator, "play_one_game", side_effect=fake_game):
        harness.run_loop(config)
    saved = torch.load(Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME, weights_only=True)
    reference = torch.load(Path(plain.model_dir) / az.TRAINING_CHECKPOINT_NAME, weights_only=True)
    for key in ("model", "optimizer", "scheduler", "train_state", "rng", "buffer_size"):
        harness.equal_tree(saved[key], reference[key])
    assert not saved["best_updated"] and not (Path(config.model_dir) / az.MODEL_BEST_NAME).exists()
    record = saved["provenance"]["evaluation"]
    assert record["selection_policy"] == "report_only" and record["opponents"][0]["pairs_used"] == 2
    assert record["opponents"][0]["bootstrap_ci95"] == [.5, .5]
    assert record["opponents"][0]["bootstrap_degenerate"]
    assert rr.attempt_status(config.data_dir, record["provenance"]) == "committed"
    run = json.loads(Path(record["output"]["path"]).read_text().splitlines()[0])
    assert run["manifest"]["parent"]["context"]["training_attempt"] == record["provenance"]
    assert run["manifest"]["parent"]["opponent_files"] == evaluator.load_opponent_pool(pool_path)["sources"]
    assert record["leaf_batch_size"] == run["leaf_batch_size"] == run["manifest"]["config"]["leaf_batch_size"] == 7


def test_pool_failure_before_training_and_failed_commit_remain_auditable(harness):
    config = replace(harness.loop_config(harness.root / "retry", iterations=1),
                     eval_pool=str(harness.root / "missing.json"), eval_games=2,
                     eval_interval=1, eval_workers=1, eval_device="cpu", eval_mcts_simulations=1)
    with pytest.raises(FileNotFoundError):
        harness.run_loop(config)
    assert not Path(config.model_dir).exists()
    config.eval_pool = install_tiny_pool(harness)
    real_save = az.save_training_checkpoint

    def fail_round(path, net, opt, scheduler, state, *args, **kwargs):
        if state.iteration == 1:
            raise OSError("checkpoint failure")
        real_save(path, net, opt, scheduler, state, *args, **kwargs)

    with mock.patch.object(evaluator, "play_one_game", side_effect=fake_game):
        with pytest.raises(OSError, match="checkpoint failure"):
            harness.run_loop(config, save_training_checkpoint=dict(side_effect=fail_round))
        harness.run_loop(config)
    rows = [json.loads(line) for line in (Path(config.data_dir) / "eval/train_eval.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["output"]["path"] != rows[1]["output"]["path"]
    assert rr.attempt_status(config.data_dir, rows[0]["provenance"]) == "unconfirmed"
    assert rr.attempt_status(config.data_dir, rows[1]["provenance"]) == "committed"
    assert all(Path(row["output"]["path"]).exists() for row in rows)
