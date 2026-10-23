"""回归：公开翻子结果改变合法动作，搜索不得复用另一个结果的子树。"""

import itertools
import random
import unittest
from unittest import mock

import torch

import azero as az
import ismcts as im
import jieqi as jq


OPENING_MOVE = (jq.index(6, 4), jq.index(5, 4))
REVEALS = (jq.TYPE_CANNON, jq.TYPE_PAWN)


class TestPublicRevealSearch(unittest.TestCase):
    def _oracle(self, root_state, first_reveal):
        """每个真实搜索动作先用当前世界的规则验证；规则内部试走不递归验证。"""
        real_apply = jq.GameState.apply_move
        real_legal = jq.GameState.legal_moves
        opening_board = tuple(root_state.board)
        reveal_cycle = itertools.cycle((first_reveal, next(t for t in REVEALS if t != first_reveal)))
        checked_moves = []
        scan_depth = 0

        def legal(state):
            nonlocal scan_depth
            scan_depth += 1
            try:
                return real_legal(state)
            finally:
                scan_depth -= 1

        def apply(state, move, rng=None, reveal_choice=None, capture_reveal_choice=None):
            if scan_depth == 0:
                self.assertIn(move, legal(state), "搜索执行了当前模拟世界中的非法着法")
                checked_moves.append(move)
            if move == OPENING_MOVE and state.side == jq.RED and tuple(state.board) == opening_board:
                if reveal_choice is None:
                    reveal_choice = next(reveal_cycle)
            return real_apply(state, move, rng, reveal_choice, capture_reveal_choice)

        return legal, apply, checked_moves

    def _expected_moves(self, reveal):
        state = jq.GameState.initial(rng=random.Random(1))
        state.apply_move(OPENING_MOVE, reveal_choice=reveal)
        moves = set(state.legal_moves())
        self.assertEqual(len(moves), 6 if reveal == jq.TYPE_CANNON else 44)
        return moves

    def test_neural_search_separates_reveals_in_both_expansion_orders(self):
        for (chance_nodes, chance_k), first_reveal in itertools.product(
            ((False, 0), (True, 0), (True, 1), (True, 2)), REVEALS
        ):
            with self.subTest(chance_nodes=chance_nodes, chance_k=chance_k, first_reveal=first_reveal):
                state = jq.GameState.initial(rng=random.Random(1))
                belief = im.init_belief_for_game(state)
                cfg = az.AZConfig(chance_nodes=chance_nodes, mcts_simulations=96,
                                  chance_k=chance_k, leaf_batch_size=8, c_puct=1000.0, virtual_loss=3.0)
                mcts = az.NeuralMCTS(None, cfg, torch.device("cpu"), random.Random(5))
                priors = torch.zeros(az.MOVE_SIZE)
                priors[az.move_to_index(OPENING_MOVE)] = 1.0
                legal, apply, checked = self._oracle(state, first_reveal)
                real_navigate = mcts._navigate_chance
                reveal_cycle = itertools.cycle((first_reveal, next(t for t in REVEALS if t != first_reveal)))

                def navigate(node, world, rng):
                    if node.move == OPENING_MOVE and tuple(world.board) == tuple(state.board):
                        reveal = next(reveal_cycle)
                        pool = world.pool[jq.RED]
                        self.assertGreater(pool[reveal], 0)
                        # 强制完整池里的抽签位置，仍经过真实导航与遗漏分支补建。
                        seat = sum(pool[t] for t in jq.POOL_TYPES[:jq.POOL_TYPES.index(reveal)])
                        forced_rng = mock.Mock(randrange=lambda total: seat)
                        return real_navigate(node, world, forced_rng)
                    return real_navigate(node, world, rng)

                def evaluate_batch(infos, histories, **kwargs):
                    return [[1.0] * az.MOVE_SIZE for _ in infos], [0.0] * len(infos)

                with (
                    mock.patch.object(jq.GameState, "legal_moves", legal),
                    mock.patch.object(jq.GameState, "apply_move", apply),
                    mock.patch.object(mcts, "_evaluate", return_value=(priors, 0.0)),
                    mock.patch.object(mcts, "_evaluate_batch", side_effect=evaluate_batch),
                    mock.patch.object(mcts, "_navigate_chance", side_effect=navigate),
                ):
                    policy, move, root = mcts.search(state, belief, [], 0.0, return_root=True)

                self.assertEqual(move, OPENING_MOVE)
                self.assertAlmostEqual(sum(policy), 1.0)
                self.assertGreater(len(checked), cfg.mcts_simulations)
                action = root.children[OPENING_MOVE]
                self.assertEqual(action.visits, cfg.mcts_simulations)
                outcomes = ({child.reveal_moved: child for child in action.children.values()}
                            if chance_nodes else action.public_outcomes)
                for reveal in REVEALS:
                    outcome = outcomes[reveal]
                    self.assertEqual(len(outcome.decision_nodes), 1)
                    decision = next(iter(outcome.decision_nodes.values()))
                    self.assertEqual(set(decision.children), self._expected_moves(reveal))
                    self.assertGreater(outcome.visits, 0)
                    self.assertLessEqual(abs(outcome.q_value()), 1.0)
                self.assertEqual(sum(child.visits for child in outcomes.values()), action.visits)
                self.assertEqual(mcts.chance_nav_miss, 0)
                if chance_nodes and chance_k == 1:
                    self.assertGreater(mcts.chance_outcomes_added, 0)

    def test_ismcts_separates_reveals_in_both_expansion_orders(self):
        for first_reveal in REVEALS:
            with self.subTest(first_reveal=first_reveal):
                state = jq.GameState.initial(rng=random.Random(1))
                belief = im.init_belief_for_game(state)
                legal, apply, checked = self._oracle(state, first_reveal)

                def root_only_legal(world):
                    return [OPENING_MOVE] if world is state else legal(world)

                with (
                    mock.patch.object(jq.GameState, "legal_moves", root_only_legal),
                    mock.patch.object(jq.GameState, "apply_move", apply),
                ):
                    move, root = im.ris_ismcts(state, belief, iterations=128,
                                             rollout_max_plies=1, rng=random.Random(5))

                self.assertEqual(move, OPENING_MOVE)
                self.assertGreater(len(checked), 128)
                action = root.children[0]
                self.assertEqual(action.visits, 128)
                for reveal in REVEALS:
                    outcome = action.public_outcomes[reveal]
                    self.assertEqual(len(outcome.decision_nodes), 1)
                    decision = next(iter(outcome.decision_nodes.values()))
                    expanded = {child.move for child in decision.children}
                    self.assertEqual(expanded | set(decision.untried_moves), self._expected_moves(reveal))
                    self.assertGreater(outcome.visits, 0)
                self.assertEqual(sum(child.visits for child in action.public_outcomes.values()), action.visits)

    def test_neural_backup_public_outcome_keeps_perspective_and_refunds_loss(self):
        root = az.PUCTNode(1.0, jq.RED)
        action = az.PUCTNode(1.0, jq.BLACK)
        outcome = action.public_outcome(jq.TYPE_CANNON)
        for node in (action, outcome):
            node.visits = 1
            node.value_sum = 3.0
        az.NeuralMCTS._backup([root, action, outcome], 0.75, 3.0)
        self.assertEqual(root.value_sum, -0.75)
        self.assertEqual(action.value_sum, 0.75)
        self.assertEqual(outcome.value_sum, 0.75)

    def test_ismcts_backup_public_outcome_does_not_flip_twice(self):
        state = jq.GameState.initial(rng=random.Random(1))
        belief = im.init_belief_for_game(state)
        legal, apply, _ = self._oracle(state, jq.TYPE_CANNON)

        def root_only_legal(world):
            return [OPENING_MOVE] if world is state else legal(world)

        with (
            mock.patch.object(jq.GameState, "legal_moves", root_only_legal),
            mock.patch.object(jq.GameState, "apply_move", apply),
            mock.patch.object(im, "evaluate_for_player", return_value=0.6),
        ):
            _, root = im.ris_ismcts(state, belief, iterations=1,
                                   rollout_max_plies=1, rng=random.Random(5))
        action = root.children[0]
        outcome = action.public_outcomes[jq.TYPE_CANNON]
        self.assertAlmostEqual(root.value, -0.6)
        self.assertAlmostEqual(action.value, 0.6)
        self.assertAlmostEqual(outcome.value, 0.6)
        self.assertEqual((root.visits, action.visits, outcome.visits), (1, 1, 1))

    def test_default_search_settings_match_training_gui_and_evaluation(self):
        import gui
        from tools import eval as evaluation

        training = az.parse_args([])
        eval_args = evaluation.parse_args([])
        config = evaluation.build_search_config(400)
        for enabled in (training.chance_nodes, eval_args.chance_nodes,
                        config.chance_nodes, gui.AZ_CHANCE_NODES):
            self.assertTrue(enabled)
        self.assertEqual((training.chance_k, eval_args.chance_k, config.chance_k, gui.AZ_CHANCE_K), (0, 0, 0, 0))
        gui_config = az.AZConfig(chance_nodes=gui.AZ_CHANCE_NODES, chance_k=gui.AZ_CHANCE_K)
        for search_config in (training, config, gui_config):
            self.assertIs(search_config.lazy_chance_allocation, True)
        for search_config in (evaluation.build_search_config(400, False),
                              evaluation.build_search_config(400, True, 1)):
            self.assertIs(search_config.lazy_chance_allocation, False)


if __name__ == "__main__":
    unittest.main()
