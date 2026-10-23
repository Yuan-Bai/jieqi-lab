from contextlib import ExitStack
import random
from unittest import mock

import numpy as np
import pytest
import torch
import azero as az
import ismcts as im
import jieqi as jq
from tools import profile_native_selfplay as native


def test_nvtx_scopes_balance_after_exception_without_cuda_events_or_sync():
    observer = native.NvtxLabels()
    with mock.patch.object(torch.cuda.nvtx, "range_push") as push, mock.patch.object(
            torch.cuda.nvtx, "range_pop") as pop, mock.patch.object(torch.cuda, "Event") as event, mock.patch.object(
            torch.cuda, "synchronize") as sync:
        with pytest.raises(ValueError), observer.scope("outer"):
            with observer.scope("inner"):
                raise ValueError("intentional")
    assert push.call_args_list == [mock.call("outer"), mock.call("inner")]
    assert pop.call_count == 2
    event.assert_not_called()
    sync.assert_not_called()


def test_nvtx_evaluation_preserves_inputs_outputs_types_and_original_methods():
    torch.set_num_threads(1)
    state = jq.GameState.initial(rng=random.Random(891))
    info = im.init_belief_for_game(state).info_state(state, state.side)
    search = az.NeuralMCTS(az.PolicyValueNet(channels=8, blocks=1).eval(), az.AZConfig(), torch.device("cpu"))
    methods = (az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch)
    plain = [methods[0](search, info, []), methods[1](search, [info,info], [[],[]])]
    rules = (jq.GameState.legal_moves, jq.GameState.apply_move, az.PUCTNode.expand)
    with mock.patch.object(torch.cuda.nvtx, "range_push") as push, mock.patch.object(torch.cuda.nvtx, "range_pop") as pop:
        with ExitStack() as stack:
            native.NvtxLabels().install(stack)
            labelled = [search._evaluate(info, []), search._evaluate_batch([info,info], [[],[]])]
            assert rules == (jq.GameState.legal_moves, jq.GameState.apply_move, az.PUCTNode.expand)
    assert torch.equal(plain[0][0], labelled[0][0]) and plain[0][1] == labelled[0][1]
    assert type(labelled[0][1]) is float
    assert np.array_equal(plain[1][0], labelled[1][0]) and plain[1][1] == labelled[1][1]
    assert not labelled[1][0].flags.writeable and all(type(v) is float for v in labelled[1][1])
    assert push.call_count == pop.call_count == 14
    assert methods == (az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch)


def test_output_guard_rejects_existing_source_and_outside_directories(tmp_path):
    source = native.ROOT / "data/pilots/chance_support_20261008_iter28/frozen_inputs"
    for directory in (tmp_path, source, source / "new"):
        with pytest.raises(ValueError):
            native.validate_output(directory.resolve(), source.resolve())


def test_reference_guard_requires_completed_exact_preparation():
    valid = dict(kind="native_profile_preparation", production_unchanged=True, paired_roots_exact=True,
                 repeats=4, graph_checks=[dict(exact=True)]*36, source_hashes={})
    native.validate_reference(valid)
    for field, value in (("kind","other"), ("production_unchanged",False), ("paired_roots_exact",False),
                         ("repeats",3), ("graph_checks",[dict(exact=False)]*36)):
        with pytest.raises(ValueError):
            native.validate_reference(dict(valid, **{field:value}))


def test_reference_guard_detects_changed_protected_files():
    valid = dict(kind="native_profile_preparation", production_unchanged=True, paired_roots_exact=True,
                 repeats=4, graph_checks=[dict(exact=True)]*36, source_hashes={"file":"hash"})
    with mock.patch.object(native, "verify_hashes", side_effect=AssertionError("changed")):
        with pytest.raises(AssertionError, match="changed"):
            native.validate_reference(valid)


def test_cross_process_exact_comparison_normalizes_only_json_container_types():
    actual = [dict(exact=dict(move=(70,65), policy_sha256="policy", rng_sha256="rng", leaf_total=800))]
    saved = [dict(exact=dict(move=[70,65], policy_sha256="policy", rng_sha256="rng", leaf_total=800))]
    assert native.exact_rows(actual) == native.exact_rows(saved)
    for field, value in (("move",[70,64]), ("policy_sha256","changed"), ("rng_sha256","changed"), ("leaf_total",799)):
        changed = [dict(exact=dict(saved[0]["exact"], **{field:value}))]
        assert native.exact_rows(actual) != native.exact_rows(changed)


def test_cross_process_exact_comparison_rejects_nonfinite_numbers():
    with pytest.raises(ValueError):
        native.exact_rows([dict(exact=dict(value=float("nan")))])
