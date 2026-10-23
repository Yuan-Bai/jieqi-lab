"""搜索信息边界：采样的对手知识、不可区分结果与决策统计共享。"""

import itertools
import random
import unittest
from unittest import mock

import torch

import azero as az
import ismcts as im
import jieqi as jq


def capture_position(dark_mover=True):
    state = jq.GameState.empty(rng=random.Random(7), with_pools=False)
    state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
    state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
    state.place_piece(5, 4, jq.TYPE_PAWN, jq.RED, 0)
    state.place_piece(5, 0, jq.TYPE_ROOK, jq.RED, int(dark_mover))
    state.place_piece(3, 0, jq.TYPE_PAWN, jq.BLACK, 1)
    state.place_piece(3, 8, jq.TYPE_PAWN, jq.BLACK, 1)
    state.place_piece(6, 0, jq.TYPE_ROOK, jq.BLACK, 0)
    state.place_piece(6, 8, jq.TYPE_PAWN, jq.RED, 1)
    state.place_piece(7, 8, jq.TYPE_PAWN, jq.RED, 1)
    for side in (jq.RED, jq.BLACK):
        state.pool[side][jq.TYPE_ROOK] = 1
        state.pool[side][jq.TYPE_KNIGHT] = 1
        state.pool_total[side] = 2
    if dark_mover:
        state.pool[jq.RED][jq.TYPE_PAWN] = 1
        state.pool_total[jq.RED] += 1
    state.reset_history()
    return state, (jq.index(5, 0), jq.index(3, 0)), (jq.index(6, 0), jq.index(6, 8))


def decisions_after(edge):
    """只遍历机会/公开结果边，取紧接着的决策节点，避免进入后续动作。"""
    if edge.decision_nodes:
        return list(edge.decision_nodes.values())
    branches = edge.children.values() if edge.is_chance else edge.public_outcomes.values()
    return [node for branch in branches for node in decisions_after(branch)]


class TestSimulatedKnowledge(unittest.TestCase):
    def test_sampled_opponent_knows_its_capture_root_does_not(self):
        state, _, capture = capture_position(False)
        belief = im.init_belief_for_game(state)
        state.side = jq.BLACK
        undo = state.apply_move(capture, capture_reveal_choice=jq.TYPE_ROOK)
        belief.observe_move(undo)
        info = belief.info_state(state, jq.RED)
        private_rows = set()
        root_rows = set()
        for seed in range(24):
            world = info.determinize(random.Random(seed))
            sampled = im.BeliefModel.for_simulation(info, jq.RED, world)
            root = sampled.info_state(world, jq.RED)
            opponent = sampled.info_state(world, jq.BLACK)
            self.assertEqual(root.pool_known, info.pool_known)
            self.assertEqual(root.unknown_removals[jq.RED], 1)
            self.assertEqual(opponent.pool_known[jq.RED], world.pool[jq.RED])
            self.assertEqual(opponent.unknown_removals[jq.RED], 0)
            self.assertEqual(opponent.pool_known[jq.BLACK], info.public_pool_known[jq.BLACK])
            root_rows.add(tuple(root.pool_known[jq.RED]))
            private_rows.add(tuple(opponent.pool_known[jq.RED]))
        self.assertEqual(len(root_rows), 1)
        self.assertEqual(len(private_rows), 2)

    def test_capture_type_is_hidden_from_victim_but_known_to_capturer(self):
        state, move, _ = capture_position(True)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, jq.RED)
        actor_keys, root_keys = [], []
        for captured in (jq.TYPE_ROOK, jq.TYPE_KNIGHT):
            world = info.determinize(random.Random(1))
            sampled = im.BeliefModel.for_simulation(info, jq.RED, world)
            undo = world.apply_move(move, reveal_choice=jq.TYPE_PAWN, capture_reveal_choice=captured)
            sampled.observe_move(undo)
            public = ((move, undo.moved_reveal),)
            actor_keys.append(im.information_set_key(sampled.info_state(world, jq.BLACK), public, ()))
            root_keys.append(im.information_set_key(sampled.info_state(world, jq.RED), public, ((0, captured),)))
        self.assertEqual(actor_keys[0], actor_keys[1])
        self.assertNotEqual(root_keys[0], root_keys[1])

    def test_simulated_private_observations_keep_recall(self):
        state, _, _ = capture_position(False)
        info = im.init_belief_for_game(state).info_state(state, jq.RED)
        public = (((0, 1), 0), ((2, 3), 0), ((4, 5), 0))
        first = im.information_set_key(info, public, ((0, jq.TYPE_ROOK), (2, jq.TYPE_KNIGHT)))
        second = im.information_set_key(info, public, ((0, jq.TYPE_KNIGHT), (2, jq.TYPE_ROOK)))
        self.assertNotEqual(first, second)


class TestNeuralInformationSets(unittest.TestCase):
    def test_opponent_policy_uses_sampled_secret_and_ignores_real_secret(self):
        searches = []
        for captured in (jq.TYPE_ROOK, jq.TYPE_KNIGHT):
            state, _, capture = capture_position(False)
            belief = im.init_belief_for_game(state)
            state.side = jq.BLACK
            belief.observe_move(state.apply_move(capture, capture_reveal_choice=captured))
            root_move = (jq.index(9, 4), jq.index(9, 3))
            replies = ((jq.index(0, 4), jq.index(1, 4)), (jq.index(6, 8), jq.index(6, 7)))
            self.assertIn(root_move, state.legal_moves())
            cfg = az.AZConfig(mcts_simulations=64, leaf_batch_size=4, c_puct=1000.0)
            mcts = az.NeuralMCTS(None, cfg, torch.device("cpu"), random.Random(8))
            root_prior = torch.zeros(az.MOVE_SIZE)
            root_prior[az.move_to_index(root_move)] = 1.0
            traces = []

            def evaluate(infos, histories, **kwargs):
                rows = []
                for info, history in zip(infos, histories):
                    row = [1.0] * az.MOVE_SIZE
                    if len(history) == 1:
                        traces.append(tuple(info.pool_known[jq.RED]))
                        row = [0.0] * az.MOVE_SIZE
                        row[az.move_to_index(replies[int(info.pool_known[jq.RED][jq.TYPE_ROOK] > 0)])] = 1.0
                    rows.append(row)
                return rows, [0.0] * len(infos)

            with mock.patch.object(mcts, "_evaluate", return_value=(root_prior, 0.0)), \
                    mock.patch.object(mcts, "_evaluate_batch", side_effect=evaluate):
                policy, move, root = mcts.search(state, belief, [], 0.0, return_root=True)
            black_nodes = list(root.children[root_move].decision_nodes.values())
            self.assertEqual(len(black_nodes), 2)
            self.assertEqual(len(set(traces)), 2)
            self.assertEqual({max(node.children, key=lambda m: node.children[m].prior) for node in black_nodes}, set(replies))
            self.assertEqual(sum(node.visits for node in black_nodes), 64)
            searches.append((policy, move, root.value_sum, traces))
        self.assertEqual(searches[0], searches[1])

    def test_private_results_share_victim_decisions_in_on_and_off_search(self):
        for (chance_nodes, lazy), dark_mover in itertools.product(
                ((True, None), (True, False), (False, None)), (True, False)):
            with self.subTest(chance_nodes=chance_nodes, lazy=lazy, dark_mover=dark_mover):
                state, red_capture, black_capture = capture_position(dark_mover)
                belief = im.init_belief_for_game(state)
                cfg = az.AZConfig(chance_nodes=chance_nodes, mcts_simulations=96,
                                  leaf_batch_size=4, c_puct=1000.0, lazy_chance_allocation=lazy)
                mcts = az.NeuralMCTS(None, cfg, torch.device("cpu"), random.Random(5))
                root_prior = torch.zeros(az.MOVE_SIZE)
                root_prior[az.move_to_index(red_capture)] = 1.0
                root_captures = itertools.cycle((jq.TYPE_ROOK, jq.TYPE_KNIGHT))
                reply_captures = itertools.cycle((jq.TYPE_ROOK, jq.TYPE_ROOK, jq.TYPE_KNIGHT, jq.TYPE_KNIGHT))
                real_apply = jq.GameState.apply_move
                initial_board = tuple(state.board)
                real_nav = mcts._navigate_chance
                black_inputs = []

                def apply(world, move, rng=None, reveal_choice=None, capture_reveal_choice=None):
                    if move == red_capture and tuple(world.board) == initial_board:
                        reveal_choice = jq.TYPE_PAWN if dark_mover else None
                        if capture_reveal_choice is None:
                            capture_reveal_choice = next(root_captures)
                    elif move == black_capture and jq.is_dark(world.board[move[1]]):
                        if capture_reveal_choice is None:
                            capture_reveal_choice = next(reply_captures)
                    return real_apply(world, move, rng, reveal_choice, capture_reveal_choice)

                def navigate(edge, world, rng):
                    if edge.move not in (red_capture, black_capture):
                        return real_nav(edge, world, rng)
                    captured = next(root_captures if edge.move == red_capture else reply_captures)
                    moved = jq.TYPE_PAWN if edge.move == red_capture and dark_mover else 0
                    # The stub inspects full logical support, not just allocated objects.
                    az.materialize_chance_outcomes(edge)
                    layer = next(node for node in edge.children.values() if node.reveal_moved == moved)
                    outcome = next(node for node in layer.children.values() if node.reveal_captured == captured)
                    return outcome, [layer, outcome]

                def evaluate(infos, histories, **kwargs):
                    rows = []
                    for info, history in zip(infos, histories):
                        row = [1.0] * az.MOVE_SIZE
                        if info.side == jq.BLACK and len(history) == 1:
                            black_inputs.append((tuple(info.pool_known[jq.BLACK]), info.unknown_removals[jq.BLACK]))
                            row = [0.0] * az.MOVE_SIZE
                            row[az.move_to_index(black_capture)] = 1.0
                        rows.append(row)
                    return rows, [0.0] * len(infos)

                with (
                    mock.patch.object(jq.GameState, "apply_move", apply),
                    mock.patch.object(mcts, "_evaluate", return_value=(root_prior, 0.0)),
                    mock.patch.object(mcts, "_evaluate_batch", side_effect=evaluate),
                    mock.patch.object(mcts, "_navigate_chance", side_effect=navigate),
                ):
                    policy, move, root = mcts.search(state, belief, [], 0.0, return_root=True)
                self.assertEqual(move, red_capture)
                self.assertAlmostEqual(sum(policy), 1.0)
                black_nodes = {id(node): node for node in decisions_after(root.children[red_capture])}
                self.assertEqual(len(black_nodes), 1, "红方私知捕获类型不能拆分黑方策略")
                black = next(iter(black_nodes.values()))
                self.assertEqual(black.visits, 96)
                self.assertEqual(len(set(black_inputs)), 1)
                self.assertEqual(black_inputs[0], (tuple(belief.public_pool_counts[jq.BLACK]), 1))
                red_nodes = {id(node): node for node in decisions_after(black.children[black_capture])}
                self.assertEqual(len(red_nodes), 2, "红方应区分自己捕获的类型，不能区分黑方捕获的类型")
                for node in (root, black, *red_nodes.values()):
                    self.assertLessEqual(abs(node.q_value()), 1.0, "虚拟损失未完整回收")

    def test_bright_capture_has_world_conditioned_chance_outcomes(self):
        state, move, _ = capture_position(False)
        info = im.init_belief_for_game(state).info_state(state, jq.RED)
        spec = az.chance_spec_for_move(state, info, move)
        self.assertIsNotNone(spec)
        self.assertIsNone(spec["moved_counts"])
        root = az.PUCTNode(1.0, jq.RED)
        root.expand([move], torch.ones(az.MOVE_SIZE), {move: spec}, 0, random.Random(1))
        world = info.determinize(random.Random(1))
        world.pool[jq.BLACK] = [0] * 8
        world.pool[jq.BLACK][jq.TYPE_KNIGHT] = 1
        world.pool_total[jq.BLACK] = 1
        mcts = az.NeuralMCTS(None, az.AZConfig(), torch.device("cpu"), random.Random(2))
        outcome, _ = mcts._navigate_chance(root.children[move], world, random.Random(3))
        self.assertEqual(outcome.reveal_moved, 0)
        self.assertEqual(outcome.reveal_captured, jq.TYPE_KNIGHT)


class TestIsmctsInformationSets(unittest.TestCase):
    def test_private_results_share_decisions_and_backup_uses_actual_path(self):
        for dark_mover in (False, True):
            with self.subTest(dark_mover=dark_mover):
                state, red_capture, black_capture = capture_position(dark_mover)
                belief = im.init_belief_for_game(state)
                initial_board = tuple(state.board)
                root_captures = itertools.cycle((jq.TYPE_ROOK, jq.TYPE_KNIGHT))
                reply_captures = itertools.cycle((jq.TYPE_ROOK, jq.TYPE_ROOK, jq.TYPE_KNIGHT, jq.TYPE_KNIGHT))
                real_apply = jq.GameState.apply_move
                real_legal = jq.GameState.legal_moves

                def apply(world, move, rng=None, reveal_choice=None, capture_reveal_choice=None):
                    if move == red_capture and tuple(world.board) == initial_board:
                        reveal_choice = jq.TYPE_PAWN if dark_mover else None
                        capture_reveal_choice = next(root_captures)
                    elif move == black_capture and jq.is_dark(world.board[move[1]]):
                        capture_reveal_choice = next(reply_captures)
                    return real_apply(world, move, rng, reveal_choice, capture_reveal_choice)

                def legal(world):
                    if world is state:
                        return [red_capture]
                    if world.side == jq.BLACK and world.board[black_capture[0]] and jq.is_dark(world.board[black_capture[1]]):
                        self.assertIn(black_capture, real_legal(world))
                        return [black_capture]
                    return real_legal(world)

                with mock.patch.object(jq.GameState, "apply_move", apply), \
                        mock.patch.object(jq.GameState, "legal_moves", legal):
                    move, root = im.ris_ismcts(state, belief, iterations=96,
                                             rollout_max_plies=1, rng=random.Random(5))
                self.assertEqual(move, red_capture)
                edge = root.children[0]
                if dark_mover:
                    edge = edge.public_outcomes[jq.TYPE_PAWN]
                self.assertEqual(len(edge.decision_nodes), 1)
                black = next(iter(edge.decision_nodes.values()))
                self.assertEqual(black.visits, 96)
                self.assertEqual(len(black.children), 1)
                red_nodes = list(black.children[0].decision_nodes.values())
                self.assertEqual(len(red_nodes), 2)
                self.assertEqual(sum(node.visits for node in red_nodes), black.children[0].visits)
                for node in (root, black, *red_nodes):
                    self.assertLessEqual(abs(node.value / node.visits), 1.0)

    def test_real_private_truth_cannot_change_root_search(self):
        searches = []
        for captured in (jq.TYPE_ROOK, jq.TYPE_KNIGHT):
            state, _, move = capture_position(False)
            belief = im.init_belief_for_game(state)
            state.side = jq.BLACK
            belief.observe_move(state.apply_move(move, capture_reveal_choice=captured))
            best, root = im.ris_ismcts(state, belief, iterations=32,
                                     rollout_max_plies=1, rng=random.Random(8))
            searches.append((best, [(child.move, child.visits, child.value) for child in root.children], root.value))
        self.assertEqual(searches[0], searches[1])


if __name__ == "__main__":
    unittest.main()
