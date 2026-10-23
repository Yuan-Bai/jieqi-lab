"""A shadow cache must count prior batches and preserve original evaluation."""
import copy
from contextlib import ExitStack
import random
import unittest
from unittest import mock

import numpy as np
import torch
import azero as az
import ismcts as im
import jieqi as jq
from tools import measure_network_cache as measure
from tools import profile_selfplay as probe


class TestCacheObserver(unittest.TestCase):
    def setUp(self):
        state = jq.GameState.initial(rng=random.Random(31))
        self.info = im.init_belief_for_game(state).info_state(state, state.side)

    def observe(self, tracker, infos, role="leaf", policy=None, values=None):
        histories = [[] for _ in infos]
        entries = [az._repetition_entries(info) for info in infos]
        batch = az._encode_batch_numpy(infos, histories, repetition_entries=entries)
        before = batch.copy()
        context = tracker.prepare(batch, infos, histories, entries, role)
        policies = np.zeros((len(infos), az.MOVE_SIZE), dtype=np.float32)
        if policy is not None:
            policies[:] = policy
        values = values if values is not None else [0.0]*len(infos)
        tracker.finish(context, policies, values, 0.1)
        np.testing.assert_array_equal(batch, before)
        return context

    def test_within_batch_is_not_counted_as_cross_batch_lru_hit(self):
        tracker = measure.CacheObserver(capacities=(2,))
        first = self.observe(tracker, [self.info, self.info])
        self.assertFalse(first["identical_batch"])
        self.assertEqual(first["hits"][2], 0)
        self.assertEqual(tracker.stats["first_game_inputs"], 1)
        self.assertEqual(tracker.stats["within_batch_repeats"], 1)
        second = self.observe(tracker, [self.info, self.info])
        self.assertTrue(second["identical_batch"])
        self.assertEqual(second["hits"][2], 2)
        self.assertEqual(tracker.stats["lru_2_leaf_all_hit_calls"], 1)
        self.assertEqual(tracker.stats["input_only_output_comparisons"], 2)

    def test_cross_search_root_leaf_accounting_and_exact_batch_identity(self):
        tracker = measure.CacheObserver(capacities=(2,))
        other = copy.deepcopy(self.info)
        other.no_capture_plies += 1
        self.observe(tracker, [self.info, other])
        tracker.start_search()
        self.observe(tracker, [self.info], "root")
        self.assertEqual(tracker.stats["root_game_additional_hits"], 1)
        self.assertEqual(tracker.stats["lru_2_root_hits"], 1)
        self.assertEqual(tracker.stats["input_only_leaf_to_root_comparisons"], 1)
        self.assertNotIn("batch_size_output_comparisons", tracker.stats)
        current = self.observe(tracker, [other, self.info])
        self.assertFalse(current["identical_batch"])
        self.assertEqual(tracker.stats["leaf_search_prior_batch_hits"], 1)
        self.assertEqual(tracker.stats["leaf_game_additional_hits"], 1)
        fresh = measure.CacheObserver(capacities=(2,))
        self.assertEqual(self.observe(fresh, [self.info])["hits"][2], 0)

    def test_output_disagreement_detects_value_policy_and_shape_context(self):
        tracker = measure.CacheObserver(capacities=(2,))
        self.observe(tracker, [self.info], "root", policy=1.0, values=[0.25])
        self.observe(tracker, [self.info, self.info], policy=1.001, values=[0.3, 0.3])
        self.assertEqual(tracker.stats["input_only_output_mismatches"], 2)
        self.assertEqual(tracker.stats["input_only_policy_mismatches"], 2)
        self.assertEqual(tracker.stats["input_only_value_mismatches"], 2)
        self.assertEqual(tracker.stats["input_only_same_size_mismatches"], 0)
        self.assertNotIn("batch_size_output_comparisons", tracker.stats)
        self.observe(tracker, [self.info, self.info], policy=1.002, values=[0.4, 0.4])
        self.assertEqual(tracker.stats["batch_size_output_mismatches"], 2)
        self.assertGreater(tracker.stats["batch_size_max_value_abs_diff"], 0)
        self.assertEqual(tracker.examples[0]["prior_batch_size"], 1)

    def test_bitwise_output_equality_does_not_hide_signed_zero(self):
        tracker = measure.CacheObserver(capacities=(2,))
        self.observe(tracker, [self.info], values=[0.0])
        self.observe(tracker, [self.info], values=[-0.0])
        self.assertEqual(tracker.stats["input_only_value_mismatches"], 1)

    def test_eviction_and_first_output_reference_are_not_overwritten(self):
        tracker = measure.CacheObserver(capacities=(1,), output_capacity=1)
        self.observe(tracker, [self.info], values=[0.1])
        self.observe(tracker, [self.info], values=[0.2])
        self.observe(tracker, [self.info], values=[0.1])
        self.assertEqual(tracker.stats["input_only_output_mismatches"], 1)
        other = copy.deepcopy(self.info)
        other.no_capture_plies += 1
        self.observe(tracker, [other])
        result = self.observe(tracker, [self.info])
        self.assertEqual(result["hits"][1], 0)
        self.assertEqual(len(tracker.outputs["input_only"]), 1)

    def test_candidate_collision_is_rejected_before_any_output_reuse(self):
        tracker = measure.CacheObserver(capacities=(2,))
        other = copy.deepcopy(self.info)
        other.no_capture_plies += 1
        with mock.patch.object(az, "leaf_input_signature", return_value=("collision",)):
            with self.assertRaisesRegex(AssertionError, "collision"):
                self.observe(tracker, [self.info, other])

    def test_summary_uses_maximum_differences_and_sums_counts_histograms(self):
        rows = [dict(stats={"rows": 8, "input_only_max_policy_abs_diff": 0.2}, histograms={2: {"leaf:8:2": 1}}),
                dict(stats={"rows": 4, "input_only_max_policy_abs_diff": 0.1}, histograms={2: {"leaf:8:2": 2}})]
        result = measure.summary(rows)
        self.assertEqual(result["stats"]["rows"], 12)
        self.assertEqual(result["stats"]["input_only_max_policy_abs_diff"], 0.2)
        self.assertEqual(result["histograms"]["2"]["leaf:8:2"], 3)


class TestEvaluationAudit(unittest.TestCase):
    def test_patch_lifetime_is_restored_on_failure(self):
        original = (az.NeuralMCTS.search, az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch,
                    az.encode_state_numpy, az._encode_batch_numpy)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with measure.EvaluationAudit().install():
                raise RuntimeError("intentional")
        self.assertEqual(original, (az.NeuralMCTS.search, az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch,
                                   az.encode_state_numpy, az._encode_batch_numpy))

    def test_actual_search_graph_policy_rng_and_network_are_unchanged(self):
        threads, rng = torch.get_num_threads(), torch.get_rng_state()
        try:
            torch.set_num_threads(1)
            torch.manual_seed(20261006)
            net = az.PolicyValueNet(channels=8, blocks=1).eval()
            state = jq.GameState.initial(rng=random.Random(79))
            belief = im.init_belief_for_game(state)
            config = az.AZConfig(mcts_simulations=64, leaf_batch_size=8, lazy_chance_allocation=True)
            before = probe.digest((state, belief, config, net.state_dict()))
            rows = []
            for measured in (False, True):
                search = az.NeuralMCTS(net, config, torch.device("cpu"), random.Random(23))
                audit = measure.EvaluationAudit()
                with audit.install() if measured else ExitStack():
                    policy, move, root = search.search(state, belief, [], 1.0, add_dirichlet=True,
                                                      prune_policy=True, return_root=True)
                rows.append((policy, move, probe.digest(root), search.rng.getstate(),
                             search.leaf_evals_total, search.leaf_evals_unique))
                if measured:
                    stats = measure.summary(audit.rows)["stats"]
                    self.assertEqual(stats["root_rows"], 1)
                    self.assertEqual(stats["leaf_rows"], search.leaf_evals_unique)
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(probe.digest((state, belief, config, net.state_dict())), before)
        finally:
            torch.set_num_threads(threads)
            torch.set_rng_state(rng)


if __name__ == "__main__":
    unittest.main()
