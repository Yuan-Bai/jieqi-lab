"""Conservative transposition keys and read-only search instrumentation."""
import copy
from dataclasses import replace
import random
import unittest
from unittest import mock

import numpy as np
import torch

import azero as az
import ismcts as im
import jieqi as jq
from tools import measure_graph_transpositions as measure
from tools import profile_selfplay as probe


class TestGraphCandidates(unittest.TestCase):
    def setUp(self):
        self.fixture = probe.controlled_endgames()[0]
        self.info = self.fixture["belief"].info_state(self.fixture["state"], self.fixture["state"].side)
        # Synthetic equal-state histories test the key boundary, not natural frequency.
        suffix = [((i, i+1), 0) for i in range(8)]
        self.paths = [[((80, 81), 0)]+suffix, [((82, 83), 0)]+suffix]

    def keys(self, info=None, path=None, private=()):
        path = self.paths[0] if path is None else path
        return measure.decision_keys(info or self.info, [item[0] for item in path], path, private)

    def test_clean_public_candidate_drops_only_old_observations(self):
        first, clean = self.keys(path=self.paths[0])
        second, other_clean = self.keys(path=self.paths[1])
        self.assertTrue(clean and other_clean)
        self.assertEqual(first["input_state"], second["input_state"])
        self.assertEqual(first["conservative"], second["conservative"])
        self.assertNotEqual(first["recall"], second["recall"])
        np.testing.assert_array_equal(measure.REFERENCE_ENCODER(self.info, [p[0] for p in self.paths[0]]),
                                      measure.REFERENCE_ENCODER(self.info, [p[0] for p in self.paths[1]]))

    def test_recent_history_and_depth_are_not_discarded(self):
        first, _ = self.keys()
        recent = self.paths[0].copy()
        recent[-1] = ((20, 21), 0)
        self.assertNotEqual(first["conservative"], self.keys(path=recent)[0]["conservative"])
        self.assertNotEqual(first["conservative"], self.keys(path=self.paths[0][1:])[0]["conservative"])

    def test_rule_counters_anchors_and_repetition_are_retained(self):
        first, _ = self.keys()
        for attr in ("consecutive_checks", "check_anchor", "chase_streak", "chase_target",
                     "chase_target_code", "chase_anchor"):
            other = copy.deepcopy(self.info)
            getattr(other, attr)[jq.RED] += 1
            with self.subTest(attr=attr):
                self.assertNotEqual(first["conservative"], self.keys(other)[0]["conservative"])
        variants = []
        other = copy.deepcopy(self.info)
        other.no_capture_plies += 1
        variants.append(other)
        other = copy.deepcopy(self.info)
        other.rules = replace(other.rules, repetition_limit=other.rules.repetition_limit+1)
        variants.append(other)
        other = copy.deepcopy(self.info)
        other.position_counts[(other.side, tuple(other.board))] += 1
        variants.append(other)
        for other in variants:
            self.assertNotEqual(first["conservative"], self.keys(other)[0]["conservative"])

    def test_hidden_information_keeps_full_recall(self):
        state = jq.GameState.initial(rng=random.Random(19))
        info = im.init_belief_for_game(state).info_state(state, state.side)
        first, clean = self.keys(info, self.paths[0])
        second, _ = self.keys(info, self.paths[1])
        self.assertFalse(clean)
        self.assertEqual(first["input_state"], second["input_state"])
        self.assertNotEqual(first["conservative"], second["conservative"])

    def test_all_bright_board_is_not_sufficient(self):
        for attr in ("pool_known", "pool_total_known", "unknown_removals", "public_pool_known",
                     "position_counts", "repetition_history_complete"):
            other = copy.deepcopy(self.info)
            if attr in ("pool_known", "public_pool_known"):
                getattr(other, attr)[jq.BLACK][jq.TYPE_ROOK] = 1
            elif attr in ("pool_total_known", "unknown_removals"):
                getattr(other, attr)[jq.BLACK] = 1
            elif attr == "position_counts":
                other.position_counts = None
            else:
                other.repetition_history_complete = False
            with self.subTest(attr=attr):
                self.assertFalse(self.keys(other)[1])
        other = copy.deepcopy(self.info)
        other.public_pool_known = None
        self.assertFalse(self.keys(other)[1])
        self.assertFalse(self.keys(private=((0, jq.TYPE_ROOK),))[1])

    def test_no_world_rng_is_used_by_observable_validation(self):
        before = probe.digest(self.info)
        self.assertEqual(measure.observable_continuation(self.info), measure.observable_continuation(self.info))
        self.assertEqual(before, probe.digest(self.info))

    def make_audit_graph(self):
        audit = measure.TranspositionAudit([])
        root_key = im.information_set_key(self.info, (), ())
        audit.observe(self.info, (), (), root_key)
        root = az.PUCTNode(1.0, self.info.side)
        for index, path in enumerate(self.paths):
            key = im.information_set_key(self.info, path, ())
            audit.observe(self.info, path, (), key)
            decision = az.PUCTNode(1.0, self.info.side)
            decision.children[(0, 1)] = az.PUCTNode(1.0, 1-self.info.side)
            root.decision_nodes[key] = decision
            if index == 0:
                alias = az.PUCTNode(1.0, self.info.side)
                alias.decision_nodes[key] = decision
                root.public_outcomes[(0, 1)] = alias
                audit.observe(self.info, path, (), key)
        return audit, root

    def test_existing_shared_nodes_and_revisits_are_not_new_candidates(self):
        audit, root = self.make_audit_graph()
        result = audit.finish(root)
        self.assertEqual(result["distinct_decisions"], 3)
        self.assertEqual(result["expanded_decisions"], 2)
        self.assertEqual(result["same_existing_decision_revisits"], 1)
        self.assertEqual(result["existing_shared_extra_links"], 1)
        self.assertEqual(result["layers"]["conservative"]["extra_decisions"], 1)
        self.assertEqual(result["layers"]["recall"]["extra_decisions"], 0)
        self.assertEqual(result["validated_input_state_pairs"], 1)

    def test_candidate_actual_input_collision_is_rejected(self):
        audit, root = self.make_audit_graph()
        with mock.patch.object(measure, "REFERENCE_ENCODER", side_effect=[np.zeros(2), np.ones(2)]):
            with self.assertRaisesRegex(AssertionError, "actual-input/legal/terminal"):
                audit.finish(root)

    def test_candidate_legal_terminal_collision_is_rejected(self):
        audit, root = self.make_audit_graph()
        with mock.patch.object(measure, "observable_continuation", side_effect=[((), None), ((), jq.RED)]):
            with self.assertRaisesRegex(AssertionError, "actual-input/legal/terminal"):
                audit.finish(root)

    def test_hooks_restore_on_exception(self):
        original = im.information_set_key
        original_direct = im.BeliefModel.information_set_key
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with measure.TranspositionAudit([]).install():
                raise RuntimeError("intentional")
        self.assertIs(im.information_set_key, original)
        self.assertIs(im.BeliefModel.information_set_key, original_direct)

    def test_summary_weights_counts_and_separates_fixture_types(self):
        rows = []
        for total, extra, kind in ((10, 1, "neural_trajectory"), (2, 1, "controlled_endgame")):
            audit = dict(expanded_decisions=total, clean_public_expanded=total,
                         expanded_depth_counts={9: total}, existing_shared_extra_links=0,
                         validated_input_state_pairs=extra, key_build_seconds=0.01, existing_lookup_seconds=0.001,
                         layers={name: dict(extra_decisions=extra) for name in measure.LAYERS})
            rows.append(dict(simulations=800, block=0, metadata=dict(kind=kind), audit=audit,
                             plain_seconds=1.0, measured_seconds=1.1))
        summary = measure.summarize(rows)["800"]
        self.assertEqual(summary["layers"]["conservative"]["fraction"], 2/12)
        self.assertEqual(summary["by_fixture_kind"]["neural_trajectory"]["layers"]["conservative"]["fraction"], 1/10)


class TestReadOnlyGraphAudit(unittest.TestCase):
    def test_plain_and_audit_preserve_graph_policy_rng_and_network_stream(self):
        rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            torch.manual_seed(20261007)
            net = az.PolicyValueNet(channels=8, blocks=1).eval()
            config = az.AZConfig(mcts_simulations=128, strong_mcts_simulations=256,
                                 leaf_batch_size=8, chance_nodes=True, chance_k=0)
            state = jq.GameState.initial(rng=random.Random(79))
            fixtures = [probe.make_fixture(state, im.init_belief_for_game(state), []),
                        *probe.controlled_endgames()]
            for fixture in fixtures:
                with self.subTest(kind=fixture["metadata"].get("kind", "opening")):
                    plain = measure.run_search(net, config, torch.device("cpu"), fixture, 128, 91, False)
                    audited = measure.run_search(net, config, torch.device("cpu"), fixture, 128, 91, True)
                    self.assertEqual(plain["exact"], audited["exact"])
                    self.assertGreater(audited["audit"]["expanded_decisions"], 0)
                    self.assertGreater(audited["audit"]["distinct_decisions"], 1)
        finally:
            torch.set_rng_state(rng)
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
