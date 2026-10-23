from contextlib import ExitStack
import random
from unittest import mock

import numpy as np
import pytest
import torch
import azero as az
import ismcts as im
import jieqi as jq
from tools import measure_sync_baseline as measure


def test_cached_host_timers_preserve_outputs_without_events_sync_or_logic_wrappers():
    torch.set_num_threads(1)
    state = jq.GameState.initial(rng=random.Random(89))
    info = im.init_belief_for_game(state).info_state(state, state.side)
    search = az.NeuralMCTS(az.PolicyValueNet(channels=8, blocks=1).eval(), az.AZConfig(), torch.device("cpu"))
    before = search._evaluate_batch([info,info], [[],[]])
    observer = measure.HostTimers()
    original = (az.NeuralMCTS._evaluate_batch, jq.GameState.apply_move, az.PUCTNode.select_child)
    with mock.patch.object(torch.cuda, "Event", side_effect=AssertionError("event")), mock.patch.object(
            torch.cuda, "synchronize", side_effect=AssertionError("sync")):
        for _ in range(2):
            observer.reset()
            with ExitStack() as stack:
                observer.install(stack)
                assert original[1:] == (jq.GameState.apply_move, az.PUCTNode.select_child)
                with observer.scope("root"):
                    after = search._evaluate_batch([info,info], [[],[]])
            assert np.array_equal(before[0], after[0]) and before[1] == after[1]
            assert not after[0].flags.writeable
            assert all(type(value) is float for value in after[1])
            assert observer.finish()["evaluation"]["calls"] == 1
    assert az.NeuralMCTS._evaluate_batch is original[0]


def test_timer_reset_rejects_leaked_scope():
    observer = measure.HostTimers()
    with observer.scope("root"):
        with pytest.raises(RuntimeError):
            observer.reset()
    observer.reset()
    assert observer.finish() == {}


def test_mode_orders_balance_every_position_in_each_pool():
    for session in range(2):
        orders = measure.mode_orders(session)
        assert len(set(orders)) == 6
        for position in range(3):
            for mode in measure.MODES:
                assert sum(order[position] == mode for order in orders) == 2


def groups():
    result = []
    for session in range(2):
        for repeat in range(6):
            for mode in measure.MODES:
                rows = []
                for index in range(18):
                    for strong in (False,True):
                        timings = {p: dict(calls=1, wall_inclusive=0.01, thread_cpu_inclusive=0.009)
                                   for p in measure.PHASES}
                        timings.update(root=dict(calls=1, wall_inclusive=1.0, thread_cpu_inclusive=0.95),
                                       evaluation=dict(calls=1, wall_inclusive=0.5, thread_cpu_inclusive=0.45),
                                       policy_d2h=dict(calls=1, wall_inclusive=0.3, thread_cpu_inclusive=0.29))
                        rows.append(dict(index=index, strong=strong, simulations=3200 if strong else 800,
                            seed=repeat*1000+index, exact=dict(policy="same", move=[1,2]), timings=timings))
                result.append(dict(session=session, repeat=repeat, mode=mode, wall_seconds=10.0, rows=rows))
    return result


def test_summary_passes_balanced_exact_stable_baseline():
    result = measure.summarize(groups())
    assert result["host_wait_experiment_gate_passed"]
    assert result["phase_summaries"][0]["policy_d2h_fraction"] == pytest.approx(0.3)


def test_summary_rejects_missing_cohort_or_roots():
    data = groups()
    with pytest.raises(ValueError):
        measure.summarize(data[:-1])
    data[0]["rows"].pop()
    with pytest.raises((ValueError,AssertionError)):
        measure.summarize(data)


def test_summary_rejects_policy_seed_and_cross_pool_differences():
    for field, value in (("exact",dict(policy="different")), ("seed",9999)):
        data = groups()
        data[1]["rows"][0][field] = value
        with pytest.raises(AssertionError):
            measure.summarize(data)
    data = groups()
    for group in data:
        if group["session"] == 1:
            group["rows"][0]["exact"] = dict(policy="different")
    with pytest.raises(AssertionError):
        measure.summarize(data)


def test_unstable_drift_and_large_measurement_overhead_stop_next_gate():
    data = groups()
    for group in data:
        if group["session"] == 1:
            group["wall_seconds"] *= 1.5
    result = measure.summarize(data)
    assert not result["timing_stability_passed"] and not result["host_wait_experiment_gate_passed"]
    data = groups()
    for group in data:
        if group["mode"] == "timers":
            group["wall_seconds"] *= 1.05
    assert not measure.summarize(data)["measurement_overhead_passed"]


def test_low_readback_or_different_phase_share_stops_candidate_gate():
    data = groups()
    for group in data:
        if group["session"] == 1:
            for row in group["rows"]:
                row["timings"]["policy_d2h"]["wall_inclusive"] = 0.1
    result = measure.summarize(data)
    assert result["timing_stability_passed"] and not result["active_readback_reproduced"]


def test_incomplete_phase_counts_rejected():
    data = groups()
    next(g for g in data if g["mode"] == "timers")["rows"][0]["timings"]["h2d"]["calls"] = 0
    with pytest.raises(AssertionError):
        measure.summarize(data)
