from contextlib import ExitStack
from dataclasses import replace
import random
import sys
from unittest import mock
import uuid

import numpy as np
import pytest
import torch
import azero as az
import ismcts as im
import jieqi as jq
from tools import measure_selfplay_attribution as measure


def test_nested_scopes_have_exclusive_partition_and_restore_after_failure():
    observer = measure.Attribution(cuda_events=False)
    with mock.patch.object(measure.time, "perf_counter", side_effect=[0,1,3,5]), mock.patch.object(
            measure.time, "thread_time", side_effect=[0,0.5,1.5,2]):
        with observer.scope("outer"):
            with pytest.raises(ValueError), observer.scope("inner"):
                raise ValueError("intentional")
    assert observer.stack == []
    assert observer.rows["outer"]["wall_inclusive"] == 5
    assert observer.rows["outer"]["wall_exclusive"] == 3
    assert observer.rows["outer"]["thread_cpu_exclusive"] == 1
    assert observer.rows["inner"]["wall_exclusive"] == 2


def test_instrumented_evaluation_preserves_cpu_inputs_outputs_and_value_types():
    torch.set_num_threads(1)
    state = jq.GameState.initial(rng=random.Random(891))
    info = im.init_belief_for_game(state).info_state(state, state.side)
    net = az.PolicyValueNet(channels=8, blocks=1).eval()
    search = az.NeuralMCTS(net, az.AZConfig(), torch.device("cpu"))
    observer = measure.Attribution(cuda_events=False)
    for name, arguments in (("_evaluate", (info, [])), ("_evaluate_batch", ([info,info], [[],[]]))):
        original = getattr(az.NeuralMCTS, name)
        before = original(search, *arguments)
        after = measure.instrument_evaluation(original, observer)(search, *arguments)
        if name == "_evaluate":
            assert torch.equal(before[0], after[0]) and before[1] == after[1]
            assert type(after[1]) is float
        else:
            assert np.array_equal(before[0], after[0]) and before[1] == after[1]
            assert not after[0].flags.writeable
            assert all(type(v) is float for v in after[1])
    assert {k: v["calls"] for k,v in observer.finish().items()} == {
        "encode":2, "h2d":2, "forward":2, "softmax":2, "policy_d2h":2, "value_d2h":2}


def test_changed_evaluation_is_rejected():
    with mock.patch.object(measure.inspect, "getsource", return_value="def changed(self):\n    return 1\n"):
        with pytest.raises(ValueError, match="evaluation changed"):
            measure.instrument_evaluation(az.NeuralMCTS._evaluate, measure.Attribution(False))


def test_wrappers_restore_methods_and_do_not_disable_native_paths():
    if jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython":
        pytest.skip("native rules/legal required")
    state = jq.GameState.initial(rng=random.Random(891))
    original = az.NeuralMCTS._evaluate_batch
    expected = state.legal_moves()
    with ExitStack() as stack:
        measure.Attribution(False).install(stack)
        for name in ("_iter_legal_moves_fast_python", "_is_square_attacked_python"):
            stack.enter_context(mock.patch.object(jq.GameState, name, side_effect=AssertionError("native fallback")))
        assert state.legal_moves() == expected
        state.apply_move(expected[0], rng=random.Random(19))
        assert state.legal_moves()
    assert az.NeuralMCTS._evaluate_batch is original


def test_evaluation_only_does_not_wrap_rule_tree_methods():
    original = (jq.GameState.legal_moves, jq.GameState.apply_move, az.PUCTNode.expand, az.PUCTNode.select_child)
    evaluator = az.NeuralMCTS._evaluate_batch
    with ExitStack() as stack:
        measure.Attribution(False, logic=False).install(stack)
        assert (jq.GameState.legal_moves, jq.GameState.apply_move, az.PUCTNode.expand, az.PUCTNode.select_child) == original
        assert az.NeuralMCTS._evaluate_batch is not evaluator
    assert az.NeuralMCTS._evaluate_batch is evaluator


def test_event_finish_has_no_added_synchronization():
    observer = measure.Attribution()
    start, end = mock.Mock(), mock.Mock()
    start.elapsed_time.return_value = 10
    observer.rows["forward"] = dict(calls=1)
    observer.events = [("forward", (start,end))]
    with mock.patch.object(torch.cuda, "synchronize") as sync:
        assert observer.finish()["forward"]["cuda_event_seconds"] == 0.01
        sync.assert_not_called()
    assert observer.events == []


def test_config_guard_rejects_cpu_precision_protocol_and_budget_changes():
    valid = az.AZConfig(mcts_simulations=800, strong_mcts_simulations=3200, strong_self_play_interval=8,
                       leaf_batch_size=64, leaf_dedup=True, virtual_loss=3, chance_nodes=True, chance_k=0)
    measure.validate_config(valid)
    for field,value in (("self_play_device","cpu"), ("leaf_batch_size",32), ("virtual_loss",1),
                        ("pcr_enabled",True), ("chance_k",1), ("lazy_chance_allocation",False),
                        ("leaf_dedup",False), ("mcts_simulations",100)):
        with pytest.raises(ValueError):
            measure.validate_config(replace(valid, **{field:value}))


def test_bad_worker_count_rejected_before_any_data_read():
    directory = measure.ROOT/"data/pilots"/uuid.uuid4().hex
    with mock.patch.object(sys, "argv", ["measure", "--output-dir", str(directory), "--workers", "1"]), mock.patch.object(
            measure.rr, "file_record") as record:
        with pytest.raises(SystemExit):
            measure.main()
        record.assert_not_called()
    assert not directory.exists()


def test_paired_overhead_matches_balanced_cohort_totals():
    groups = [dict(repeat=i, mode=m, wall_seconds=t) for i,m,t in (
        (0,"plain",10), (0,"stages",11), (1,"stages",24), (1,"plain",20))]
    summary = measure.paired_overhead(groups)
    assert summary["pairs"] == 2
    assert summary["mean_overhead"] == pytest.approx(0.15)
    assert summary["aggregate_overhead"] == pytest.approx(35/30-1)
