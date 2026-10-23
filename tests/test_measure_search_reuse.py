"""Exact input accounting and instrumentation safety checks."""
import copy
import itertools
import random
import unittest
from unittest import mock

import numpy as np
import torch
import azero as az
import ismcts as im
import jieqi as jq
from tools import measure_search_reuse as measure
from tools import profile_selfplay as probe


class TestInputReuse(unittest.TestCase):
    def setUp(self):
        state = jq.GameState.initial(rng=random.Random(93))
        self.info = im.init_belief_for_game(state).info_state(state, state.side)

    def observe(self, tracker, infos, histories=None):
        histories = histories or [[] for _ in infos]
        entries = [az._repetition_entries(info) for info in infos]
        encoded = az._encode_batch_numpy(infos, histories, repetition_entries=entries)
        before = encoded.copy()
        tracker.observe(encoded, infos, histories, entries, "leaf")
        np.testing.assert_array_equal(encoded, before)

    def test_distinguishes_within_batch_cross_batch_and_cross_search(self):
        tracker = measure.InputReuse(capacities=(2,))
        self.observe(tracker, [self.info, self.info])
        self.assertEqual(tracker.stats["first_game_inputs"], 1)
        self.assertEqual(tracker.stats["within_batch_repeats"], 1)
        self.assertEqual(tracker.stats["search_prior_batch_hits"], 0)
        self.observe(tracker, [self.info])
        self.assertEqual(tracker.stats["search_prior_batch_hits"], 1)
        self.assertEqual(tracker.stats["candidate_prior_batch_hits"], 1)
        tracker.start_search()
        self.observe(tracker, [self.info])
        self.assertEqual(tracker.stats["game_additional_hits"], 1)
        self.assertEqual(tracker.stats["search_prior_batch_hits"], 0)
        self.assertEqual(tracker.stats["lru_2_hits"], 1)
        self.assertEqual(len(tracker.search_seen), 1)

    def test_candidate_uses_only_encoder_recent_history(self):
        tracker = measure.InputReuse()
        recent = [(i, i+1) for i in range(8)]
        self.observe(tracker, [self.info], histories=[[(81, 82)] + recent])
        self.observe(tracker, [self.info], histories=[[(83, 84)] + recent])
        self.assertEqual(tracker.stats["search_prior_batch_hits"], 1)
        self.assertEqual(tracker.stats["candidate_prior_batch_hits"], 1)
        self.assertEqual(tracker.stats["candidate_false_hits"], 0)

    def test_exact_equality_retains_private_rule_and_repetition_inputs(self):
        tracker = measure.InputReuse()
        variants = [self.info]
        private = copy.deepcopy(self.info)
        private.pool_known[jq.BLACK][jq.TYPE_ROOK] -= 1
        variants.append(private)
        chase = copy.deepcopy(self.info)
        chase.chase_streak[jq.RED] += 1
        variants.append(chase)
        repetition = copy.deepcopy(self.info)
        current = (self.info.side, tuple(self.info.board))
        repetition.position_counts[current] += 1
        variants.append(repetition)
        for info in variants:
            self.observe(tracker, [info])
        self.assertEqual(tracker.stats["first_game_inputs"], 4)
        self.assertEqual(tracker.stats["search_prior_batch_hits"], 0)

    def test_false_candidate_collision_is_rejected_even_within_batch(self):
        tracker = measure.InputReuse()
        other = copy.deepcopy(self.info)
        other.no_capture_plies += 1
        with mock.patch.object(az, "leaf_input_signature", return_value=("forced collision",)):
            with self.assertRaisesRegex(AssertionError, "different actual inputs"):
                self.observe(tracker, [self.info, other])
        self.assertEqual(tracker.stats["candidate_false_hits"], 1)

    def test_lru_eviction_and_game_isolation(self):
        tracker = measure.InputReuse(capacities=(1,))
        other = copy.deepcopy(self.info)
        other.no_capture_plies += 1
        self.observe(tracker, [self.info])
        self.observe(tracker, [other])
        self.observe(tracker, [self.info])
        self.assertEqual(tracker.stats["lru_1_hits"], 0)
        self.assertEqual(tracker.stats["lru_1_evictions"], 2)
        self.assertEqual(tracker.stats["search_prior_batch_hits"], 1)
        new_game = measure.InputReuse()
        self.observe(new_game, [self.info])
        self.assertEqual(new_game.stats["game_additional_hits"], 0)


class TestSearchReuseAudit(unittest.TestCase):
    def test_graph_counts_shared_decisions_once_and_classifies_chance_children(self):
        root = az.PUCTNode(1.0, jq.RED)
        entry = az.PUCTNode(1.0, jq.BLACK, move=(0, 1), is_chance=True)
        layer = az.PUCTNode(1.0, jq.BLACK, is_chance=True)
        used, unused = az.PUCTNode(1.0, jq.BLACK), az.PUCTNode(1.0, jq.BLACK)
        decision = az.PUCTNode(1.0, jq.BLACK)
        root.children[(0, 1)] = entry
        entry.children[0] = layer
        layer.children.update({0: used, 1: unused})
        used.decision_nodes[()] = decision
        unused.decision_nodes[()] = decision
        root.visits = entry.visits = layer.visits = used.visits = decision.visits = 1
        result = measure.count_graph(root)
        self.assertEqual(result["all_nodes"]["total"], 6)
        self.assertEqual(result["all_nodes"]["unvisited"], 1)
        self.assertEqual(result["chance_children"]["total"], 3)
        self.assertEqual(result["chance_children"]["unvisited"], 1)
        self.assertEqual(result["chance_results"]["total"], 2)
        self.assertEqual(result["chance_entries"]["total"], 1)
        self.assertEqual(result["chance_layers"]["total"], 1)

    def test_instrumentation_restores_functions_on_exception(self):
        original = (az.NeuralMCTS.search, az._encode_batch_numpy, az.encode_state_numpy)
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with measure.SearchAudit(True).install():
                raise RuntimeError("intentional failure")
        self.assertEqual((az.NeuralMCTS.search, az._encode_batch_numpy, az.encode_state_numpy), original)

    def test_search_preserves_entire_graph_rng_policy_inputs_and_config(self):
        rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            torch.manual_seed(20261006)
            net = az.PolicyValueNet(channels=8, blocks=1).eval()
            config = az.AZConfig(mcts_simulations=24, strong_mcts_simulations=48,
                                 leaf_batch_size=8, chance_nodes=True, chance_k=0)
            state = jq.GameState.initial(rng=random.Random(79))
            fixture = probe.make_fixture(state, im.init_belief_for_game(state), [])
            for strong, lazy in itertools.product((False, True), (False, True)):
                config.lazy_chance_allocation = lazy
                before = probe.digest((fixture, config))
                rows = []
                for measured in (False, True):
                    audit = measure.SearchAudit(measured, graph_hash=True)
                    with audit.install():
                        row = probe.run_case(net, config, torch.device("cpu"), fixture, strong, 91, "plain")
                    rows.append((row, audit.rows[0]))
                self.assertEqual(rows[0][0]["exact"], rows[1][0]["exact"])
                self.assertEqual(rows[0][1]["graph_sha256"], rows[1][1]["graph_sha256"])
                inputs = rows[1][1]["inputs"]
                self.assertEqual(inputs["leaf_rows"], rows[1][0]["exact"]["leaf_unique"])
                self.assertEqual(inputs["root_rows"], 1)
                self.assertEqual(inputs["rows"], sum(inputs[name] for name in
                    ("first_game_inputs", "within_batch_repeats", "search_prior_batch_hits", "game_additional_hits")))
                unvisited = rows[1][1]["graph"]["chance_children"]["unvisited"]
                if lazy:
                    self.assertEqual(unvisited, 0)
                else:
                    self.assertGreater(unvisited, 0)
                self.assertEqual(probe.digest((fixture, config)), before)
        finally:
            torch.set_rng_state(rng)
            torch.set_num_threads(threads)

    def test_summary_uses_counts_not_average_of_search_percentages(self):
        rows = [dict(search_seconds=1.0, leaf_total=8, leaf_unique=7,
                     graph={"chance_children": dict(total=10, unvisited=9, shallow_bytes=100,
                                                    unvisited_shallow_bytes=90)},
                     inputs=dict(rows=8, search_prior_batch_hits=2, game_additional_hits=1)),
                dict(search_seconds=2.0, leaf_total=4, leaf_unique=3,
                     graph={"chance_children": dict(total=2, unvisited=0, shallow_bytes=20,
                                                    unvisited_shallow_bytes=0)},
                     inputs=dict(rows=4, search_prior_batch_hits=0, game_additional_hits=0))]
        result = measure.summary(rows)
        self.assertEqual(result["graph"]["chance_children"]["unvisited_fraction"], 9/12)
        self.assertEqual(result["inputs"]["search_cross_batch_fraction"], 2/12)
        self.assertEqual(result["search_seconds"], 3.0)


if __name__ == "__main__":
    unittest.main()
