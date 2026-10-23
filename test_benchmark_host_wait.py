from contextlib import ExitStack, nullcontext
import random
from unittest import mock

import numpy as np
import pytest
import torch
import azero as az
import ismcts as im
import jieqi as jq
from tools import benchmark_host_wait as bench


def test_cpu_outputs_and_original_methods_are_preserved_without_cuda_calls():
    torch.set_num_threads(1)
    state = jq.GameState.initial(rng=random.Random(89))
    info = im.init_belief_for_game(state).info_state(state, state.side)
    search = az.NeuralMCTS(az.PolicyValueNet(channels=8, blocks=1).eval(), az.AZConfig(), torch.device("cpu"))
    expected = (search._evaluate(info, []), search._evaluate_batch([info, info], [[], []]))
    original = (az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch, jq.GameState.apply_move)
    waiter = bench.BlockingPolicyWait()
    with mock.patch.object(torch.cuda, "Event", side_effect=AssertionError("unexpected event")), mock.patch.object(
            torch.cuda, "synchronize", side_effect=AssertionError("unexpected global sync")):
        for _ in range(2):
            with ExitStack() as stack:
                waiter.install(stack)
                assert jq.GameState.apply_move is original[2]
                actual = (search._evaluate(info, []), search._evaluate_batch([info, info], [[], []]))
            assert torch.equal(expected[0][0], actual[0][0]) and expected[0][1] == actual[0][1]
            assert np.array_equal(expected[1][0], actual[1][0]) and expected[1][1] == actual[1][1]
            assert not actual[1][0].flags.writeable and all(type(v) is float for v in actual[1][1])
    assert (az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch) == original[:2]
    assert waiter.calls == 0 and not waiter.events


def test_cuda_event_order_reuse_device_and_original_copy():
    log = []

    class Tensor:
        is_cuda = True
        device = torch.device("cuda:0")

        def cpu(self):
            log.append("cpu")
            return self

    event = mock.Mock()
    event.record.side_effect = lambda stream: log.append(("record", stream))
    event.synchronize.side_effect = lambda: log.append("host_sync")
    waiter, tensor = bench.BlockingPolicyWait(), Tensor()
    with mock.patch.object(torch.cuda, "device", return_value=nullcontext()) as device, mock.patch.object(
            torch.cuda, "Event", return_value=event) as create, mock.patch.object(
            torch.cuda, "current_stream", return_value="current") as stream, mock.patch.object(
            torch.cuda, "synchronize", side_effect=AssertionError("global sync")):
        for _ in range(2):
            assert waiter.copy_policy(tensor.cpu) is tensor
        create.assert_called_once_with(enable_timing=False, blocking=True)
        device.assert_called_once_with(tensor.device)
        assert stream.call_args_list == [mock.call(tensor.device)]*2
    assert log == [("record", "current"), "host_sync", "cpu"]*2
    assert waiter.calls == 2
    event.wait.assert_not_called()
    event.query.assert_not_called()


def no_cpu():
    return 1


def two_cpu(x):
    return x.cpu(), x.cpu()


def cpu_argument(x):
    return x.cpu(memory_format=torch.contiguous_format)


@pytest.mark.parametrize("function", [no_cpu, two_cpu, cpu_argument])
def test_source_drift_fails_closed(function):
    with pytest.raises(ValueError):
        bench.instrument_policy_cpu(function, bench.BlockingPolicyWait())


def groups(gain=0.08):
    rows = []
    for session in range(2):
        for repeat in range(4):
            for mode in bench.MODES:
                roots = [dict(index=index, strong=strong, simulations=3200 if strong else 800,
                              seed=202610110+repeat*1000+index, exact=dict(policy="same"),
                              event_waits=3 if mode == "blocking" else 0,
                              process_cpu_seconds=0.7 if mode == "blocking" else 1.0)
                         for index in range(18) for strong in (False, True)]
                rows.append(dict(session=session, repeat=repeat, mode=mode, rows=roots,
                                 wall_seconds=10/(1+gain) if mode == "blocking" else 10))
    return rows


def test_balanced_order_and_screen_positive():
    assert bench.mode_orders(0) == [bench.MODES, tuple(reversed(bench.MODES))]*2
    assert bench.mode_orders(1) == list(reversed(bench.mode_orders(0)))
    result = bench.summarize(groups())
    assert result["full_game_gate_passed"] and result["stable_baseline"]
    assert result["aggregate_throughput_gain"] == pytest.approx(0.08)
    assert result["pools"][0]["worker_cpu_change"] == pytest.approx(-0.3)


def test_cpu_reduction_without_speed_does_not_pass_screen():
    result = bench.summarize(groups(0.01))
    assert result["stable_baseline"] and not result["full_game_gate_passed"]
    assert result["pools"][0]["worker_cpu_change"] < 0


def test_screen_rejects_missing_duplicated_and_misordered_roots():
    data = groups()
    with pytest.raises(ValueError):
        bench.summarize(data[:-1])
    data[0]["rows"][0], data[0]["rows"][1] = data[0]["rows"][1], data[0]["rows"][0]
    with pytest.raises(ValueError):
        bench.summarize(data)


def test_screen_rejects_policy_seed_and_event_coverage():
    for field, value in (("exact", dict(policy="changed")), ("seed", 999), ("event_waits", 0)):
        data = groups()
        data[1]["rows"][0][field] = value
        with pytest.raises(AssertionError):
            bench.summarize(data)


def test_unstable_or_one_pool_regression_cannot_pass_screen():
    data = groups()
    for g in data:
        if g["session"] == 1:
            g["wall_seconds"] *= 1.3
    assert not bench.summarize(data)["full_game_gate_passed"]
    data = groups(0.2)
    for g in data:
        if g["session"] == 1 and g["mode"] == "blocking":
            g["wall_seconds"] = 10.1
    result = bench.summarize(data)
    assert result["aggregate_throughput_gain"] > 0.03 and not result["full_game_gate_passed"]


def test_gate_requires_wins_in_both_orders_not_just_aggregate_gain():
    data = groups()
    for g in data:
        if g["mode"] == "blocking":
            g["wall_seconds"] = 6 if g["repeat"] < 2 else 10.1
    result = bench.summarize(data)
    assert result["aggregate_throughput_gain"] > 0.03 and not result["full_game_gate_passed"]


def test_invalid_reference_is_rejected_before_cuda():
    with pytest.raises(ValueError):
        bench.validate_baseline(dict(kind="sync_baseline"), None)
