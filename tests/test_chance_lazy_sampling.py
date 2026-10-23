"""候选缓存不能改变世界揭示分布，遗漏分支也必须进入正常回传。"""
import collections
import itertools
import math
import random
import unittest
from unittest import mock

import torch

import azero as az
import ismcts as im
import jieqi as jq
from test_search_information_sets import capture_position, decisions_after


MOVE = (jq.index(6, 4), jq.index(5, 4))


class LastWarmup(random.Random):
    def random(self):
        return 0.999  # 预建时故意只留下列表末尾的类型。


class Seats:
    """枚举无放回池中每个等概率位置，避免频率测试的随机波动。"""
    def __init__(self, *seats):
        self.seats = iter(seats)

    def randrange(self, total):
        seat = next(self.seats)
        assert 0 <= seat < total, (seat, total)
        return seat


def set_pool(state, side, weights):
    state.pool[side] = [weights.get(t, 0) for t in range(8)]
    state.pool_total[side] = sum(weights.values())


def make_edge(k=1, capture=False, bright=False):
    state = jq.GameState.initial(rng=random.Random(3))
    if bright:
        state.place_piece(6, 4, jq.TYPE_ROOK, jq.RED, 0)
    if capture:
        state.place_piece(5, 4, jq.TYPE_PAWN, jq.BLACK, 1)
    set_pool(state, jq.RED, {jq.TYPE_ROOK: 5, jq.TYPE_CANNON: 3, jq.TYPE_PAWN: 2})
    set_pool(state, jq.BLACK, {jq.TYPE_KNIGHT: 1, jq.TYPE_CANNON: 3})
    state.reset_history()
    info = im.init_belief_for_game(state).info_state(state, jq.RED)
    root = az.PUCTNode(1.0, jq.RED)
    spec = az.chance_spec_for_move(state, info, MOVE)
    root.expand([MOVE], torch.ones(az.MOVE_SIZE), {MOVE: spec}, k, LastWarmup())
    mcts = az.NeuralMCTS(None, az.AZConfig(chance_k=k), torch.device("cpu"), random.Random(4))
    return state, root, root.children[MOVE], mcts


class TestChanceLazySampling(unittest.TestCase):
    def test_unknown_removals_match_exact_hypergeometric_distribution(self):
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(6, 0, jq.TYPE_PAWN, jq.RED, 1)
        state.place_piece(6, 2, jq.TYPE_PAWN, jq.RED, 1)
        state.reset_history()
        info = im.init_belief_for_game(state).info_state(state, jq.RED)
        types = (jq.TYPE_ROOK, jq.TYPE_KNIGHT, jq.TYPE_CANNON)
        original = (2, 1, 1)
        for kind, count in zip(types, original):
            info.pool_known[jq.RED][kind] = count
        info.pool_total_known[jq.RED] = 2
        info.unknown_removals[jq.RED] = 2
        before = [row.copy() for row in info.pool_known]
        hits = collections.Counter()
        for seats in itertools.product(range(4), range(3)):
            world = info.determinize(Seats(*seats))
            hits[tuple(world.pool[jq.RED][kind] for kind in types)] += 1
            self.assertEqual(world.pool_total[jq.RED], 2)
            self.assertEqual(world.board, info.board)
        expected = {}
        for remaining in itertools.product(range(3), range(2), range(2)):
            if sum(remaining) == 2:
                ways = math.prod(math.comb(count, left) for count, left in zip(original, remaining))
                expected[remaining] = ways * 2  # 12 ordered draws / C(4,2).
        self.assertEqual(hits, expected)
        self.assertEqual(info.pool_known, before)

    def test_double_reveal_backup_signs_visits_and_virtual_loss_conserve(self):
        for k in (0, 1):
            state, root, edge, mcts = make_edge(k=k, capture=True)
            values = []
            for moved_seat, captured_seat in itertools.product(range(10), range(4)):
                outcome, visited = mcts._navigate_chance(edge, state, Seats(moved_seat, captured_seat))
                value = 1.0 if outcome.reveal_moved == jq.TYPE_ROOK else -1.0
                if outcome.reveal_captured == jq.TYPE_CANNON:
                    value *= 0.5
                values.append(value)
                path = [root, edge, *visited]
                for node in path[1:]:
                    node.visits += 1
                    node.value_sum += 3.0
                mcts._backup(path, value, 3.0)
            expected = sum(values) / 40
            self.assertEqual(root.visits, 40)
            self.assertEqual(edge.visits, 40)
            self.assertAlmostEqual(root.q_value(), -expected)
            self.assertAlmostEqual(edge.q_value(), expected)
            self.assertEqual(sum(layer.visits for layer in edge.children.values()), 40)
            self.assertEqual(sum(leaf.visits for layer in edge.children.values()
                                 for leaf in layer.children.values()), 40)
            self.assertAlmostEqual(edge.value_sum, sum(layer.value_sum for layer in edge.children.values()))

    def test_all_world_seats_preserve_probability_and_value_for_k0_k1_k2(self):
        weights = {jq.TYPE_ROOK: 5, jq.TYPE_CANNON: 3, jq.TYPE_PAWN: 2}
        values = {jq.TYPE_ROOK: 1.0, jq.TYPE_CANNON: 0.0, jq.TYPE_PAWN: -1.0}
        for k in (0, 1, 2):
            with self.subTest(k=k):
                state, root, edge, mcts = make_edge(k)
                self.assertEqual(len(edge.children), k or 3)
                hits = collections.Counter()
                for seat in range(10):
                    outcome, visited = mcts._navigate_chance(edge, state, Seats(seat))
                    self.assertIsNotNone(outcome)
                    hits[outcome.reveal_moved] += 1
                    path = [root, edge, *visited]
                    for node in path[1:]:
                        node.visits += 1
                        node.value_sum += 3.0
                    mcts._backup(path, values[outcome.reveal_moved], 3.0)
                self.assertEqual(hits, weights)
                self.assertAlmostEqual(edge.q_value(), 0.3)
                self.assertAlmostEqual(root.q_value(), -0.3)
                self.assertEqual(sum(n.visits for n in edge.children.values()), edge.visits)
                self.assertEqual(len(edge.children), 3)
                self.assertAlmostEqual(sum(n.prior for n in edge.children.values()), 1.0)

    def test_disjoint_initial_cache_adds_available_world_result(self):
        state, _, edge, mcts = make_edge()
        self.assertEqual({n.reveal_moved for n in edge.children.values()}, {jq.TYPE_PAWN})
        set_pool(state, jq.RED, {jq.TYPE_ROOK: 1})
        outcome, visited = mcts._navigate_chance(edge, state, Seats(0))
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.reveal_moved, jq.TYPE_ROOK)
        self.assertEqual(visited, [outcome])
        self.assertIs(mcts._navigate_chance(edge, state, Seats(0))[0], outcome)
        self.assertEqual(mcts.chance_outcomes_added, 1, "同一结果必须复用已补建节点")
        undo = mcts._apply_chance_move(state, MOVE, outcome.reveal_moved, outcome.reveal_captured)
        self.assertEqual(undo.moved_reveal, jq.TYPE_ROOK)
        self.assertEqual(state.pool[jq.RED][jq.TYPE_ROOK], 0)
        state.undo_move(undo)
        self.assertEqual(state.pool[jq.RED][jq.TYPE_ROOK], 1)

    def test_double_reveal_enumerates_exact_joint_world_distribution(self):
        state, _, edge, mcts = make_edge(capture=True)
        hits = collections.Counter()
        for moved_seat, captured_seat in itertools.product(range(10), range(4)):
            outcome, visited = mcts._navigate_chance(edge, state, Seats(moved_seat, captured_seat))
            self.assertEqual(len(visited), 2)
            self.assertIs(visited[-1], outcome)
            hits[outcome.reveal_moved, outcome.reveal_captured] += 1
            undo = mcts._apply_chance_move(state, MOVE, outcome.reveal_moved, outcome.reveal_captured)
            self.assertEqual((undo.moved_reveal, undo.captured_reveal),
                             (outcome.reveal_moved, outcome.reveal_captured))
            state.undo_move(undo)
        expected = {(m, c): wm * wc
                    for m, wm in ((jq.TYPE_ROOK, 5), (jq.TYPE_CANNON, 3), (jq.TYPE_PAWN, 2))
                    for c, wc in ((jq.TYPE_KNIGHT, 1), (jq.TYPE_CANNON, 3))}
        self.assertEqual(hits, expected)
        self.assertEqual(len(edge.children), 3)
        self.assertTrue(all(len(layer.children) == 2 for layer in edge.children.values()))

    def test_bright_capture_samples_only_victim_full_pool(self):
        state, _, edge, mcts = make_edge(capture=True, bright=True)
        set_pool(state, jq.RED, {})  # 明子无需移动方暗池，不能多抽一次。
        hits = collections.Counter()
        for seat in range(4):
            outcome, visited = mcts._navigate_chance(edge, state, Seats(seat))
            self.assertEqual(outcome.reveal_moved, 0)
            self.assertEqual(len(visited), 2)
            hits[outcome.reveal_captured] += 1
        self.assertEqual(hits, {jq.TYPE_KNIGHT: 1, jq.TYPE_CANNON: 3})

    def test_prefill_priors_keep_original_probability_mass(self):
        state, _, edge, mcts = make_edge()
        self.assertEqual(len(edge.children), 1)
        only = next(iter(edge.children.values()))
        self.assertEqual(only.reveal_moved, jq.TYPE_PAWN)
        self.assertAlmostEqual(only.prior, 0.2)
        set_pool(state, jq.RED, {jq.TYPE_ROOK: 9, jq.TYPE_PAWN: 1})
        outcome, _ = mcts._navigate_chance(edge, state, Seats(0))
        # 新节点显示原信念的概率 0.5；实际导航必须使用新世界的 0.9。
        self.assertAlmostEqual(outcome.prior, 0.5)
        hits = collections.Counter(mcts._navigate_chance(edge, state, Seats(i))[0].reveal_moved for i in range(10))
        self.assertEqual(hits, {jq.TYPE_ROOK: 9, jq.TYPE_PAWN: 1})

    def test_lazy_private_outcomes_share_victim_decision_and_refund_virtual_loss(self):
        state, move, _ = capture_position(False)
        belief = im.init_belief_for_game(state)
        initial_board = tuple(state.board)
        priors = torch.zeros(az.MOVE_SIZE)
        priors[az.move_to_index(move)] = 1.0
        cfg = az.AZConfig(chance_k=1, mcts_simulations=64, leaf_batch_size=8, c_puct=1000.0)
        mcts = az.NeuralMCTS(None, cfg, torch.device("cpu"), random.Random(6))
        real_legal = jq.GameState.legal_moves

        def legal(world):
            return [move] if tuple(world.board) == initial_board else real_legal(world)

        def evaluate(infos, histories, **kwargs):
            return [[1.0] * az.MOVE_SIZE for _ in infos], [0.0] * len(infos)

        with (mock.patch.object(jq.GameState, "legal_moves", side_effect=legal, autospec=True),
              mock.patch.object(mcts, "_evaluate", return_value=(priors, 0.0)),
              mock.patch.object(mcts, "_evaluate_batch", side_effect=evaluate)):
            policy, chosen, root = mcts.search(state, belief, [], 0.0, return_root=True)
        self.assertEqual(chosen, move)
        self.assertAlmostEqual(sum(policy), 1.0)
        edge = root.children[move]
        layer = next(iter(edge.children.values()))
        self.assertEqual({n.reveal_captured for n in layer.children.values()}, {jq.TYPE_ROOK, jq.TYPE_KNIGHT})
        decisions = {id(n): n for n in decisions_after(edge)}
        self.assertEqual(len(decisions), 1, "被吃方不能按吃方私知类型拆策略")
        decision = next(iter(decisions.values()))
        self.assertEqual(decision.visits, cfg.mcts_simulations)
        self.assertEqual(root.visits, cfg.mcts_simulations)
        self.assertEqual(edge.visits, cfg.mcts_simulations)
        self.assertEqual(sum(n.visits for n in layer.children.values()), layer.visits)
        self.assertEqual(mcts.chance_nav_miss, 0)
        for node in (root, edge, layer, decision, *layer.children.values()):
            self.assertLessEqual(abs(node.q_value()), 1.0)


if __name__ == "__main__":
    unittest.main()
