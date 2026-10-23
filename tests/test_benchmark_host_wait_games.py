from dataclasses import asdict
import random
from unittest import mock

import pytest
import torch
import azero as az
from tools import benchmark_host_wait_games as bench
from tools import profile_selfplay as probe


def test_tasks_restore_original_seed_stream_and_seven_to_one_flags():
    rng = random.Random(82)
    checkpoint = dict(rng=dict(selfplay=rng.getstate()))
    tasks = bench.make_tasks(checkpoint)
    assert [t[1] for t in tasks] == [rng.randrange(1 << 30) for _ in range(32)]
    assert [t[0] for t in tasks] == list(range(32))
    assert [t[0] for t in tasks if t[2]] == [7, 15, 23, 31]
    assert checkpoint["rng"]["selfplay"] == random.Random(82).getstate()


def test_original_worker_full_result_and_final_rng_are_captured(tmp_path):
    game = az.SelfPlayResult(samples=[], outcome=1, move_count=0)

    def fake_game(net, config, device, rng, **kwargs):
        rng.random()
        return game

    with mock.patch.object(az, "self_play_game", fake_game), mock.patch.object(az, "_WORKER_NET", object()), mock.patch.object(
            az, "_WORKER_CONFIG", az.AZConfig()), mock.patch.object(az, "_WORKER_DEVICE", torch.device("cpu")), mock.patch.object(
            bench, "WAITER", mock.Mock(calls=0)), mock.patch.object(bench, "DIRECTORY", tmp_path), mock.patch.object(
            bench.probe, "DEVICE", torch.device("cpu")):
        output = bench.game_task(("plain", 0, 71, False))
    rng = random.Random(71)
    rng.random()
    assert output["game"] is game
    assert output["row"]["exact"]["rng_sha256"] == probe.digest(rng.getstate())
    assert bench.collect_games([output]) == [output["row"]]
    output["game"].outcome = -1
    with pytest.raises(AssertionError):
        bench.collect_games([output])


def groups(gain=0.06):
    result = []
    for session in range(2):
        for repeat in range(2):
            for mode in bench.roots.MODES:
                rows = [dict(event_waits=10 if mode == "blocking" else 0, process_cpu_seconds=1.0,
                             exact=dict(index=i, seed=i, strong=(i+1) % 8 == 0, moves=20, outcome=1,
                                        samples=20, game_sha256="same", rng_sha256="same"))
                        for i in range(repeat*16, (repeat+1)*16)]
                result.append(dict(session=session, repeat=repeat, mode=mode, rows=rows,
                                   wall_seconds=10/(1+gain) if mode == "blocking" else 10))
    return result


def test_summary_exact_positive_remains_only_confirmation_gate():
    result = bench.summarize(groups())
    assert result["confirmation_gate_passed"] and result["paired_games_exact"]
    assert result["aggregate_throughput_gain"] == pytest.approx(0.06)
    assert not result["production_promotion"]
    assert bench.mode_orders(1) == list(reversed(bench.mode_orders(0)))


def test_summary_rejects_sample_rng_seed_and_cross_pool_changes():
    for field in ("game_sha256", "rng_sha256", "seed"):
        data = groups()
        data[1]["rows"][0]["exact"][field] = "changed"
        with pytest.raises(AssertionError):
            bench.summarize(data)
    data = groups()
    for g in data:
        if g["session"] == 1:
            g["rows"][0]["exact"]["game_sha256"] = "changed"
    with pytest.raises(AssertionError):
        bench.summarize(data)


def test_summary_rejects_missing_cohort_flags_or_event_coverage():
    data = groups()
    with pytest.raises(ValueError):
        bench.summarize(data[:-1])
    data = groups()
    data[0]["rows"][0]["exact"]["strong"] = True
    with pytest.raises(ValueError):
        bench.summarize(data)
    data = groups()
    data[1]["rows"][0]["event_waits"] = 0
    with pytest.raises(AssertionError):
        bench.summarize(data)


def test_summary_noisy_small_gain_or_one_regression_does_not_pass():
    assert not bench.summarize(groups(0.01))["confirmation_gate_passed"]
    data = groups(0.2)
    data[1]["wall_seconds"] = 10.1
    result = bench.summarize(data)
    assert result["aggregate_throughput_gain"] > 0.03 and not result["confirmation_gate_passed"]
