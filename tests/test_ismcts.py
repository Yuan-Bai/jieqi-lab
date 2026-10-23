import math
import random
import unittest
from unittest import mock

import ismcts as im
import jieqi as jq


class TestIsmctsBackprop(unittest.TestCase):
    def test_backprop_perspective_with_odd_rollout_depth(self) -> None:
        """回归：终局发生在奇数个 rollout ply 后时，节点值必须按该节点执棋方记账。"""
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(5, 2, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_ROOK, jq.BLACK, 0)
        state.place_piece(8, 0, jq.TYPE_ROOK, jq.BLACK, 0)
        state.place_piece(1, 3, jq.TYPE_ROOK, jq.BLACK, 0)
        state.place_piece(1, 5, jq.TYPE_ROOK, jq.BLACK, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.side = jq.RED
        state.reset_history()

        block = (jq.index(5, 2), jq.index(5, 4))
        mate = (jq.index(0, 4), jq.index(5, 4))
        self.assertEqual(state.legal_moves(), [block])

        belief = im.init_belief_for_game(state)
        with mock.patch("ismcts.select_rollout_move", return_value=mate):
            move, root = im.ris_ismcts(state, belief, iterations=1, rng=random.Random(0))

        self.assertEqual(move, block)
        child = root.children[0]
        self.assertEqual(child.player_to_move, jq.BLACK)
        self.assertAlmostEqual(child.value, 1.0, places=6)
        self.assertAlmostEqual(root.value, -1.0, places=6)


class TestIsmctsRollout(unittest.TestCase):
    def test_ris_ismcts_determinizes_only_root_observer(self) -> None:
        """R3 回归：ris_ismcts 每条迭代只在根观察者处确定化一次。

        真实 belief 只读取根 observer；其他视图必须来自合成观测。
        确定化只用同一个根信息态，每条迭代一次。
        """
        rng = random.Random(3)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        root_player = state.side
        observers: list[int] = []
        determinized = []
        real_info_state = im.BeliefModel.info_state
        real_determinize = im.InformationState.determinize

        def spy_info_state(self, st, observer):
            if self is belief:
                observers.append(observer)
            return real_info_state(self, st, observer)

        def spy_determinize(info, rng):
            determinized.append(info)
            return real_determinize(info, rng)

        with mock.patch.object(im.BeliefModel, "info_state", spy_info_state), \
                mock.patch.object(im.InformationState, "determinize", spy_determinize):
            move, root = im.ris_ismcts(state, belief, iterations=7, rng=random.Random(5))

        self.assertEqual(sorted(set(observers)), [root_player])
        self.assertEqual(observers, [root_player])
        self.assertEqual(len(determinized), 7)
        self.assertEqual(len({id(info) for info in determinized}), 1)

    def test_uct_select_child_uses_parent_perspective(self) -> None:
        parent = im.Node(parent=None, move=None, player_to_move=jq.RED)
        child_bad_for_parent = im.Node(parent=parent, move=(0, 1), player_to_move=jq.BLACK)
        child_good_for_parent = im.Node(parent=parent, move=(0, 2), player_to_move=jq.BLACK)
        child_bad_for_parent.visits = 10
        child_bad_for_parent.value = 8.0
        child_good_for_parent.visits = 10
        child_good_for_parent.value = -8.0
        parent.children = [child_bad_for_parent, child_good_for_parent]
        parent.visits = 20
        selected = parent.uct_select_child(0.0)
        self.assertEqual(selected.move, (0, 2))

    def test_outcome_value_for_player_sign(self) -> None:
        self.assertEqual(im.outcome_value_for_player(jq.RESULT_RED_WIN, jq.RED), 1.0)
        self.assertEqual(im.outcome_value_for_player(jq.RESULT_RED_WIN, jq.BLACK), -1.0)
        self.assertEqual(im.outcome_value_for_player(jq.RESULT_BLACK_WIN, jq.BLACK), 1.0)
        self.assertEqual(im.outcome_value_for_player(jq.RESULT_BLACK_WIN, jq.RED), -1.0)
        self.assertEqual(im.outcome_value_for_player(jq.RESULT_DRAW, jq.RED), 0.0)

    def test_select_rollout_move_prefers_capture(self) -> None:
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(4, 4, jq.TYPE_ROOK, jq.RED, 0)
        target_idx = state.place_piece(4, 6, jq.TYPE_PAWN, jq.BLACK, 0)
        state.side = jq.RED
        state.reset_history()

        moves = state.legal_moves()
        move = im.select_rollout_move(state, moves, random.Random(0), "heuristic")
        self.assertEqual(move[1], target_idx)

    def test_select_rollout_move_scores_dark_target_by_pool_expectation(self) -> None:
        """回归：暗子吃子分用剩余暗池期望，而非假类型面值。

        假兵暗子（暗池只剩车 → 期望 5）应优先于明炮（4）；旧实现按假兵
        计 1 分会选明炮。
        """
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(4, 4, jq.TYPE_ROOK, jq.RED, 0)
        dark_idx = state.place_piece(4, 6, jq.TYPE_PAWN, jq.BLACK, 1)
        state.place_piece(4, 8, jq.TYPE_CANNON, jq.BLACK, 0)
        state.pool[jq.BLACK][jq.TYPE_ROOK] = 1
        state.pool_total[jq.BLACK] = 1
        state.side = jq.RED
        state.reset_history()

        move = im.select_rollout_move(state, state.legal_moves(), random.Random(0), "heuristic")
        self.assertEqual(move[1], dark_idx)


class TestLeafEvaluation(unittest.TestCase):
    def _base_state(self):
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 5, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(5, 0, jq.TYPE_PAWN, jq.BLACK, 1)
        state.side = jq.RED
        state.reset_history()
        return state

    def test_material_eval_counts_dark_pieces_at_pool_expectation(self) -> None:
        state = self._base_state()
        state.pool[jq.BLACK][jq.TYPE_ROOK] = 1  # 暗池只剩一车
        state.pool_total[jq.BLACK] = 1
        # 明车(5) 对 暗车期望(5)：均势
        self.assertAlmostEqual(im.evaluate_for_player(state, jq.RED, 10.0), 0.0, places=6)
        # 暗池只剩卒(1)：红方 +4 子力
        state.pool[jq.BLACK][jq.TYPE_ROOK] = 0
        state.pool[jq.BLACK][jq.TYPE_PAWN] = 1
        self.assertAlmostEqual(
            im.evaluate_for_player(state, jq.RED, 10.0), math.tanh(0.4), places=6
        )
        self.assertAlmostEqual(
            im.evaluate_for_player(state, jq.BLACK, 10.0), -math.tanh(0.4), places=6
        )

    def test_rollout_truncation_uses_leaf_eval(self) -> None:
        """回归：rollout 超深截断时，回传值来自子力评估而非终局胜负。"""
        state = self._base_state()
        belief = im.init_belief_for_game(state)

        def first_legal(st, moves, rng, policy):
            return moves[0]

        with (
            mock.patch("ismcts.select_rollout_move", side_effect=first_legal),
            mock.patch("ismcts.evaluate_for_player", return_value=0.5) as eval_mock,
        ):
            _, root = im.ris_ismcts(
                state, belief, iterations=1, rng=random.Random(0), rollout_max_plies=1
            )

        self.assertTrue(eval_mock.called)
        self.assertEqual(len(root.children), 1)
        self.assertAlmostEqual(root.children[0].value, 0.5, places=6)
        self.assertAlmostEqual(root.value, -0.5, places=6)


if __name__ == "__main__":
    unittest.main()
