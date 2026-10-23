"""Checks for read-only compiled-search profiling and timer boundaries."""
from contextlib import ExitStack
import cProfile
import random
import unittest
from unittest import mock

import torch

import azero as az
import ismcts as im
import jieqi as jq
from tools import profile_selfplay as probe
from tools.profile_search_expansion import ExpansionBoundaries, normalise_exact


class TestProfileSelfplay(unittest.TestCase):
    def test_worker_rejects_changed_legal_backend_before_loading_model(self):
        expected_nodes = (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE)
        with mock.patch.object(jq, "RULES_BACKEND", "cython"), mock.patch.object(
                probe.az, "PolicyValueNet") as factory:
            with self.assertRaises(AssertionError):
                probe.init_worker({}, {}, jq.RULES_BACKEND_NATIVE, expected_nodes, ("different", None))
            factory.assert_not_called()

    def test_controlled_endgames_are_valid_and_distinct(self):
        fixtures = probe.controlled_endgames()
        self.assertEqual(len(fixtures), 2)
        for fixture in fixtures:
            self.assertEqual(fixture["metadata"]["pieces"], 8)
            self.assertEqual(fixture["metadata"]["dark_pieces"], 0)
            self.assertEqual(fixture["metadata"]["material_stage"], "endgame")
            self.assertGreater(fixture["metadata"]["legal_moves"], 0)
            self.assertIsNone(fixture["state"].game_outcome()[0])
        self.assertNotEqual(fixtures[0]["state"].board, fixtures[1]["state"].board)

    def test_fixture_owns_state_belief_history_and_rng(self):
        state = jq.GameState.initial(rng=random.Random(71))
        belief = im.init_belief_for_game(state)
        history = []
        before = probe.digest((state, belief, history))
        fixture = probe.make_fixture(state, belief, history)
        self.assertEqual(probe.digest((state, belief, history)), before)
        self.assertIsNot(fixture["state"].rng, state.rng)
        self.assertEqual(fixture["state"].rng.getstate(), state.rng.getstate())
        self.assertIsNot(fixture["belief"].unknown_removals[0], belief.unknown_removals[0])
        fixture["state"].board[0] = 0
        fixture["state"].rng.random()
        fixture["belief"].unknown_removals[0][0] += 1
        fixture["history"].append((0, 1))
        self.assertEqual(probe.digest((state, belief, history)), before)

    def test_rule_timer_does_not_double_count_nested_queries(self):
        boundaries = probe.Boundaries()
        native = boundaries.wrapper("native_attack", lambda *, value: value)
        inner = boundaries.wrapper("inner", lambda: native(value=19), rule=True)
        outer = boundaries.wrapper("outer", inner, rule=True)
        with mock.patch.object(probe.time, "perf_counter", side_effect=[0.0, 1.0, 2.0, 3.0]):
            self.assertEqual(outer(), 19)
        self.assertNotIn("inner", boundaries.rows)
        self.assertEqual(boundaries.rows["rules_total"], dict(calls=1, seconds=3.0))
        self.assertEqual(boundaries.rows["native_inside_rules"], dict(calls=1, seconds=1.0))
        self.assertEqual(boundaries.rule_depth, 0)

    def test_wrappers_restore_on_exception(self):
        original = jq.GameState.is_square_attacked
        boundaries = probe.Boundaries()
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with ExitStack() as stack:
                boundaries.install(stack)
                self.assertIsNot(jq.GameState.is_square_attacked, original)
                fail = boundaries.wrapper("failure", mock.Mock(side_effect=RuntimeError("probe failure")), rule=True)
                fail()
        self.assertIs(jq.GameState.is_square_attacked, original)
        self.assertEqual(boundaries.rule_depth, 0)
        self.assertEqual(boundaries.rows["rules_total"]["calls"], 1)

    def test_instrumented_search_preserves_outputs_inputs_and_config(self):
        torch_rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        original = jq.GameState.is_square_attacked
        try:
            torch.set_num_threads(1)
            torch.manual_seed(101)
            net = az.PolicyValueNet(channels=8, blocks=1).eval()
            config = az.AZConfig(mcts_simulations=24, strong_mcts_simulations=48,
                                 leaf_batch_size=8, dirichlet_alpha=0.17,
                                 chance_nodes=True, chance_k=0)
            state = jq.GameState.initial(rng=random.Random(83))
            fixture = probe.make_fixture(state, im.init_belief_for_game(state), [])
            before = probe.digest((fixture, config))
            for strong in (False, True):
                rows = [probe.run_case(net, config, torch.device("cpu"), fixture, strong, 89, mode)
                        for mode in ("plain", "cprofile", "boundaries")]
                self.assertEqual(rows[0]["exact"], rows[1]["exact"])
                self.assertEqual(rows[0]["exact"], rows[2]["exact"])
                with mock.patch.object(probe, "Boundaries", ExpansionBoundaries):
                    confirmation = probe.run_case(net, config, torch.device("cpu"), fixture, strong, 89, "boundaries")
                self.assertEqual(rows[0]["exact"], confirmation["exact"])
                self.assertGreater(confirmation["boundaries"]["expand"]["calls"], 0)
                self.assertEqual(rows[0]["simulations"], 48 if strong else 24)
                self.assertGreater(rows[1]["profile"]["total_seconds"], 0)
                self.assertGreater(rows[2]["boundaries"]["native_attack"]["calls"], 0)
                self.assertIs(jq.GameState.is_square_attacked, original)
                self.assertEqual(probe.digest((fixture, config)), before)
            with self.assertRaisesRegex(ValueError, "unknown profiling mode"):
                probe.run_case(net, config, torch.device("cpu"), fixture, False, 89, "invalid")
        finally:
            torch.set_rng_state(torch_rng)
            torch.set_num_threads(threads)

    def test_profile_aggregation_sums_own_and_cumulative_separately(self):
        profile = cProfile.Profile()
        profile.runcall(lambda: sum(range(50)))
        row = dict(profile=probe.profile_rows(profile))
        aggregate = probe.aggregate_profiles([row, row])
        self.assertEqual(aggregate["total_seconds"], 2 * row["profile"]["total_seconds"])
        self.assertEqual(len(aggregate["functions"]), len(row["profile"]["functions"]))
        for source, combined in zip(row["profile"]["functions"], aggregate["functions"]):
            for field in ("calls", "primitive_calls", "own_seconds", "cumulative_seconds"):
                self.assertEqual(combined[field], 2 * source[field])

    def test_expansion_timers_do_not_wrap_constructors_or_native_attack(self):
        constructor, attack = az.PUCTNode.__init__, jq.GameState.is_square_attacked
        expand, select = az.PUCTNode.expand, az.PUCTNode.select_child
        boundaries = ExpansionBoundaries()
        with ExitStack() as stack:
            boundaries.install(stack)
            self.assertIs(az.PUCTNode.__init__, constructor)
            self.assertIs(jq.GameState.is_square_attacked, attack)
            root = az.PUCTNode(1.0, jq.RED)
            root.expand([(0, 1)], [1.0] * az.MOVE_SIZE)
            self.assertEqual(root.select_child(1.5)[0], (0, 1))
        self.assertIs(az.PUCTNode.expand, expand)
        self.assertIs(az.PUCTNode.select_child, select)
        self.assertEqual(set(boundaries.rows), {"expand", "select_child"})
        self.assertEqual(normalise_exact(dict(move=[0, 1])), dict(move=(0, 1)))
        self.assertEqual(normalise_exact(dict(move=None)), dict(move=None))


if __name__ == "__main__":
    unittest.main()
