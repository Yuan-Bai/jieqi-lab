"""Direct keys must be identical, isolated and independent of pending leaf snapshots."""
import copy
import itertools
import random
from types import ModuleType
import unittest
from unittest import mock

import torch
import azero as az
import ismcts as im
import jieqi as jq
from test_search_information_sets import capture_position
from tools import benchmark_information_snapshots as bench


def reference_key(info, public, private):
    return (tuple(public), tuple(private), info.side, bytes(info.board),
            tuple(info.pool_known[jq.BLACK]), tuple(info.pool_known[jq.RED]),
            tuple(info.pool_total_known), tuple(info.unknown_removals),
            info.no_capture_plies, tuple(info.consecutive_checks), tuple(info.check_anchor),
            tuple(info.chase_streak), tuple(info.chase_target), tuple(info.chase_target_code),
            tuple(info.chase_anchor))


class TestDirectInformationKeys(unittest.TestCase):
    def test_matches_old_schema_for_both_actors_after_private_captures(self):
        for dark, captured, seed in itertools.product((False, True), (jq.TYPE_ROOK, jq.TYPE_KNIGHT), range(4)):
            state, move, _ = capture_position(dark)
            belief = im.init_belief_for_game(state)
            root = belief.info_state(state, state.side)
            world = root.determinize(random.Random(seed))
            view = im.BeliefModel.for_simulation(root, state.side, world)
            undo = world.apply_move(move, reveal_choice=jq.TYPE_PAWN if dark else None,
                                    capture_reveal_choice=captured)
            view.observe_move(undo)
            public, private = [], [[], []]
            im.record_observation(move, undo, public, private)
            for actor in (jq.BLACK, jq.RED):
                info = view.info_state(world, actor)
                expected = reference_key(info, public, private[actor])
                self.assertEqual(im.information_set_key(info, public, private[actor]), expected)
                self.assertEqual(view.information_set_key(world, actor, public, private[actor]), expected)

    def test_all_key_fields_are_sensitive_and_frozen(self):
        state, _, _ = capture_position()
        view = im.init_belief_for_game(state)
        public, private = [((1, 2), 0)], [(0, jq.TYPE_ROOK)]
        original = view.information_set_key(state, state.side, public, private)
        self.assertEqual(hash(original), hash(reference_key(view.info_state(state, state.side), public, private)))
        mutations = [lambda s, v, p, q: p.append(((3, 4), 1)), lambda s, v, p, q: q.append((1, 2)),
                     lambda s, v, p, q: setattr(s, "side", 1-s.side),
                     lambda s, v, p, q: s.board.__setitem__(0, 1),
                     lambda s, v, p, q: v.pool_counts[s.side][0].__setitem__(1, 99),
                     lambda s, v, p, q: v.pool_counts[s.side][1].__setitem__(1, 99),
                     lambda s, v, p, q: v.pool_total[s.side].__setitem__(0, 99),
                     lambda s, v, p, q: v.unknown_removals[s.side].__setitem__(0, 99),
                     lambda s, v, p, q: setattr(s, "no_capture_plies", 99)]
        for field in ("consecutive_checks", "check_anchor", "chase_streak", "chase_target",
                      "chase_target_code", "chase_anchor"):
            mutations.append(lambda s, v, p, q, field=field: getattr(s, field).__setitem__(0, 99))
        for index, mutation in enumerate(mutations):
            s, v, p, q = copy.deepcopy((state, view, public, private))
            frozen = v.information_set_key(s, s.side, p, q)
            actor = s.side
            mutation(s, v, p, q)
            changed = v.information_set_key(s, actor, p, q)
            self.assertNotEqual(changed, original, index)
            self.assertEqual(frozen, original)
            self.assertEqual(changed, reference_key(v.info_state(s, actor), p, q))

    def test_does_not_access_world_pools_other_observer_or_snapshot_only_fields(self):
        state, _, _ = capture_position()
        view = im.init_belief_for_game(state)
        actor = state.side
        expected = reference_key(view.info_state(state, actor), (), ())

        class Poison:
            def __getitem__(self, key):
                raise AssertionError("private or snapshot-only data accessed")

            def __iter__(self):
                raise AssertionError("private or snapshot-only data accessed")

            def copy(self):
                raise AssertionError("full snapshot allocated")

        state.pool = state.pool_total = state.position_counts = Poison()
        view.pool_counts[1-actor] = view.pool_total[1-actor] = view.unknown_removals[1-actor] = Poison()
        view.public_pool_counts = Poison()
        with mock.patch.object(im.BeliefModel, "info_state", side_effect=AssertionError("snapshot")):
            self.assertEqual(view.information_set_key(state, actor, (), ()), expected)

    def test_direct_projection_matches_snapshot_two_color_vectors(self):
        state, _, _ = capture_position()
        view = im.init_belief_for_game(state)
        view.pool_total[state.side].append(91)
        view.unknown_removals[state.side].append(92)
        self.assertEqual(view.information_set_key(state, state.side, (), ()),
                         reference_key(view.info_state(state, state.side), (), ()))


class TestLeafSnapshots(unittest.TestCase):
    def test_only_root_and_pending_leaves_allocate_independent_snapshots(self):
        state, _, _ = capture_position()
        belief = im.init_belief_for_game(state)
        config = az.AZConfig(mcts_simulations=128, leaf_batch_size=8, leaf_dedup=False)
        search = az.NeuralMCTS(None, config, torch.device("cpu"), random.Random(12))
        snapshots, frozen, batches, keys = [], [], [], []
        original_info, original_key = im.BeliefModel.info_state, im.BeliefModel.information_set_key

        def snapshot(view, world, actor):
            info = original_info(view, world, actor)
            snapshots.append(info)
            frozen.append(copy.deepcopy(info))
            return info

        def key(view, *args):
            keys.append(1)
            return original_key(view, *args)

        def evaluate(infos, histories, **kwargs):
            for info in infos:
                index = next(i for i, known in enumerate(snapshots) if known is info)
                self.assertEqual(info, frozen[index])
            batches.extend(infos)
            return [[1.0]*az.MOVE_SIZE for _ in infos], [0.0]*len(infos)

        with mock.patch.object(im.BeliefModel, "info_state", snapshot), mock.patch.object(
                im.BeliefModel, "information_set_key", key), mock.patch.object(
                search, "_evaluate", return_value=([1.0]*az.MOVE_SIZE, 0.0)), mock.patch.object(
                search, "_evaluate_batch", side_effect=evaluate):
            search.search(state, belief, [], 1.0)
        self.assertEqual(len(snapshots), 1+search.leaf_evals_total)
        self.assertEqual(len(batches), search.leaf_evals_total)
        self.assertGreater(len(keys), len(batches))
        self.assertEqual(len({id(info) for info in snapshots}), len(snapshots))
        self.assertEqual(snapshots, frozen)

    def test_root_snapshot_is_reused_when_no_simulated_move_occurs(self):
        state, _, _ = capture_position()
        state.side = jq.BLACK
        belief = im.init_belief_for_game(state)
        config = az.AZConfig(mcts_simulations=4, leaf_batch_size=4)
        search = az.NeuralMCTS(None, config, torch.device("cpu"), random.Random(3))
        root_info = belief.info_state(state, state.side)
        # A custom root expansion leaves the root unexpanded, exercising the zero-depth leaf.
        root_info.position_counts = {}
        received = []

        def evaluate(infos, histories, **kwargs):
            received.extend(infos)
            return [[1.0]*az.MOVE_SIZE for _ in infos], [0.0]*len(infos)

        with mock.patch.object(im.BeliefModel, "info_state", return_value=root_info) as snapshot, mock.patch.object(
                search, "_expand_node"), mock.patch.object(search, "_evaluate", return_value=([1.0]*az.MOVE_SIZE, 0.0)), \
                mock.patch.object(search, "_evaluate_batch", side_effect=evaluate):
            search.search(state, belief, [], 0.0)
        self.assertEqual(snapshot.call_count, 1)
        self.assertTrue(received)
        self.assertTrue(all(info is root_info for info in received))
        self.assertEqual(root_info.position_counts, {})

    def test_terminal_or_no_legal_leaf_does_not_allocate_a_snapshot(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                state, _, _ = capture_position()
                belief = im.init_belief_for_game(state)
                board = tuple(state.board)
                config = az.AZConfig(mcts_simulations=16, leaf_batch_size=4)
                search = az.NeuralMCTS(None, config, torch.device("cpu"), random.Random(17))
                original_info, original_legal = im.BeliefModel.info_state, jq.GameState.legal_moves
                count = [0]

                def snapshot(view, *args):
                    count[0] += 1
                    return original_info(view, *args)

                def legal(world):
                    return original_legal(world) if tuple(world.board) == board else []

                def outcome(world):
                    return (jq.RESULT_DRAW if terminal and tuple(world.board) != board else None, None)

                with mock.patch.object(im.BeliefModel, "info_state", snapshot), mock.patch.object(
                        jq.GameState, "legal_moves", legal), mock.patch.object(
                        jq.GameState, "outcome_checked", outcome), mock.patch.object(
                        search, "_evaluate", return_value=([1.0]*az.MOVE_SIZE, 0.0)), mock.patch.object(
                        search, "_evaluate_batch", side_effect=AssertionError("terminal leaf evaluation")):
                    policy, move, root = search.search(state, belief, [], 0.0, return_root=True)
                self.assertEqual(count[0], 1)
                self.assertEqual(search.leaf_evals_total, 0)
                self.assertEqual(root.visits, 16)
                self.assertIsNotNone(move)
                self.assertAlmostEqual(sum(policy), 1.0)


class TestSnapshotBenchmark(unittest.TestCase):
    def test_paired_summary_uses_matching_cohorts(self):
        trials = [dict(repeat=r, mode=m, wall_seconds=10 if m == "snapshots" else 9)
                  for r in range(4) for m in bench.MODES]
        result = bench.paired_summary(trials)
        self.assertAlmostEqual(result["mean_wall_saving"], 0.1)
        self.assertEqual(result["total_seconds"], dict(snapshots=40, direct=36))

    def test_extraction_does_not_execute_module_body(self):
        source = "raise RuntimeError('must not run')\nclass Model:\n    def search(self, x: Missing):\n        return x+2\n"
        search, _ = bench.extract_function(source, "search", {}, "reference", "Model")
        self.assertEqual(search(None, 3), 5)

    def test_reference_keeps_module_attribute_lookup_and_original_leaf_assignment(self):
        sources = {
            "ismcts.py": "def information_set_key(info, p, q):\n    return (p, q, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)\n",
            "azero.py": "class NeuralMCTS:\n    def search(self, root_state, belief):\n"
                        "        root_info = belief.info_state(root_state, 0)\n"
                        "        sim_view = belief\n"
                        "        actor_info = sim_view.info_state(root_state, 0)\n"
                        "        leaf_info = actor_info\n"
                        "        return im.information_set_key(leaf_info, (), ())\n",
        }
        before = mock.MagicMock()
        before.__truediv__.side_effect = lambda name: mock.Mock(read_text=lambda **kwargs: sources[name])
        reference = bench.load_reference(before)
        self.assertIsInstance(reference.__globals__["im"], ModuleType)
        belief = mock.Mock()
        self.assertEqual(reference(None, None, belief), ((), (), *range(2, 15)))
        self.assertEqual(belief.info_state.call_count, 2)


if __name__ == "__main__":
    unittest.main()
