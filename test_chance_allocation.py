"""Exact deferred allocation: full support, ordered priors, RNG and shared graph."""
import itertools
import pickle
import random
import unittest
from dataclasses import asdict

import numpy as np
import torch

import azero as az
import ismcts as im
import jieqi as jq
from test_chance_lazy_sampling import MOVE, Seats, make_edge
from test_search_information_sets import capture_position


def materialize_graph(root):
    seen, pending = set(), [root]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        az.materialize_chance_outcomes(node)
        pending.extend(node.children.values())
        pending.extend(node.public_outcomes.values())
        pending.extend(node.decision_nodes.values())


def deferred_edge(state, move=MOVE):
    info = im.init_belief_for_game(state).info_state(state, state.side)
    mcts = az.NeuralMCTS(None, az.AZConfig(lazy_chance_allocation=True), torch.device("cpu"), random.Random(4))
    root = az.PUCTNode(1.0, state.side)
    spec = az.chance_spec_for_move(state, info, move)
    mcts._expand_node(root, [move], torch.ones(az.MOVE_SIZE), {move: spec})
    return root, root.children[move], mcts


class TinyNet(torch.nn.Module):
    def forward(self, x):
        scale = x[:, :8].sum(dim=(1, 2, 3)) / 100
        logits = torch.arange(az.MOVE_SIZE, dtype=x.dtype, device=x.device)[None] % 17
        return logits * scale[:, None], torch.tanh(scale - 0.5)


class TestChanceAllocation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_flag_default_and_invalid_config(self):
        self.assertTrue(az.AZConfig().lazy_chance_allocation)
        self.assertIs(asdict(az.AZConfig())["lazy_chance_allocation"], True)
        self.assertTrue(az.parse_args([]).lazy_chance_allocation)
        self.assertTrue(az.parse_args(["--lazy-chance-allocation"]).lazy_chance_allocation)
        self.assertFalse(az.parse_args(["--no-lazy-chance-allocation"]).lazy_chance_allocation)
        for args in (["--chance-k", "1"], ["--no-chance-nodes"]):
            self.assertFalse(az.parse_args(args).lazy_chance_allocation)
            self.assertFalse(az.parse_args(["--no-lazy-chance-allocation", *args]).lazy_chance_allocation)
            with self.assertRaises(SystemExit):
                az.parse_args(["--lazy-chance-allocation", *args])
        for kwargs in (dict(chance_k=1), dict(chance_nodes=False)):
            config = az.AZConfig(**kwargs)
            self.assertFalse(config.lazy_chance_allocation)
            az.NeuralMCTS(None, config, torch.device("cpu"))
            with self.assertRaises(ValueError):
                az.NeuralMCTS(None, az.AZConfig(lazy_chance_allocation=True, **kwargs), torch.device("cpu"))

    def test_default_expansion_defers_objects_and_disable_restores_eager(self):
        state, _, reference, _ = make_edge(0, capture=True)
        info = im.init_belief_for_game(state).info_state(state, state.side)
        spec = az.chance_spec_for_move(state, info, MOVE)
        roots = []
        for args in ([], ["--no-lazy-chance-allocation"]):
            search = az.NeuralMCTS(None, az.parse_args(args), torch.device("cpu"), random.Random(4))
            root = az.PUCTNode(1.0, state.side)
            search._expand_node(root, [MOVE], torch.ones(az.MOVE_SIZE), {MOVE: spec})
            roots.append(root)
        self.assertEqual(roots[0].children[MOVE].children, {})
        self.assertEqual(list(roots[1].children[MOVE].children), list(reference.children))
        materialize_graph(roots[0])
        self.assertEqual(pickle.dumps(roots[0]), pickle.dumps(roots[1]))

    def test_full_world_support_priors_indices_backup_and_rng(self):
        for capture, bright in ((False, False), (True, False), (True, True)):
            with self.subTest(capture=capture, bright=bright):
                state, eager_root, eager, reference = make_edge(0, capture=capture, bright=bright)
                root, edge, mcts = deferred_edge(state)
                self.assertEqual(edge.children, {})
                self.assertEqual(edge.chance_counts, eager.chance_counts)
                seats = list(itertools.product(range(10), range(4))) if capture else [(s,) for s in range(10)]
                if bright:
                    seats = [(s,) for s in range(4)]
                # Visit the last types first: physical insertion order must not determine keys.
                for draw in reversed(seats):
                    actual, path = mcts._navigate_chance(edge, state, Seats(*draw))
                    expected, expected_path = reference._navigate_chance(eager, state, Seats(*draw))
                    self.assertEqual([(n.prior, n.reveal_moved, n.reveal_captured, n.is_chance) for n in path],
                                     [(n.prior, n.reveal_moved, n.reveal_captured, n.is_chance) for n in expected_path])
                    self.assertEqual(next(k for k, v in edge.children.items() if v is path[0]),
                                     next(k for k, v in eager.children.items() if v is expected_path[0]))
                    for tree, action, nodes, search in ((root, edge, path, mcts),
                                                       (eager_root, eager, expected_path, reference)):
                        for node in [action, *nodes]:
                            node.visits += 1
                            node.value_sum += 3.0
                        search._backup([tree, action, *nodes], actual.reveal_moved/7 - actual.reveal_captured/8, 3.0)
                self.assertEqual(mcts.chance_outcomes_added, 0)
                before = mcts.rng.getstate()
                materialize_graph(root)
                self.assertEqual(mcts.rng.getstate(), before)
                self.assertEqual(pickle.dumps(root), pickle.dumps(eager_root))

    def test_snapshot_and_auxiliary_sampling_keep_full_distribution(self):
        state, eager_root, eager, _ = make_edge(0, capture=True)
        root, edge, _ = deferred_edge(state)
        for seed in range(8):
            a, b = random.Random(seed), random.Random(seed)
            key, layer = edge.sample_outcome(a)
            expected_key, expected_layer = eager.sample_outcome(b)
            self.assertEqual(key, expected_key)
            self.assertEqual(layer.sample_outcome(a)[0], expected_layer.sample_outcome(b)[0])
            self.assertEqual(a.getstate(), b.getstate())
        self.assertEqual(pickle.dumps(root), pickle.dumps(eager_root))

    def test_partial_materialization_preserves_visited_objects_and_aliases(self):
        state, _, eager, _ = make_edge(0, capture=True)
        root, edge, mcts = deferred_edge(state)
        outcome, path = mcts._navigate_chance(edge, state, Seats(9, 3))
        shared = az.PUCTNode(1.0, jq.RED)
        outcome.decision_nodes[("a",)] = shared
        root.decision_nodes[("b",)] = shared
        outcome.visits, outcome.value_sum = 9, 1.25
        keys = (next(k for k, v in edge.children.items() if v is path[0]),
                next(k for k, v in path[0].children.items() if v is outcome))
        self.assertEqual(keys, (2, 1))
        self.assertEqual(len(edge.children), 1)
        materialize_graph(root)
        self.assertIs(edge.children[2].children[1], outcome)
        self.assertIs(outcome.decision_nodes[("a",)], root.decision_nodes[("b",)])
        self.assertEqual((outcome.visits, outcome.value_sum), (9, 1.25))
        self.assertEqual(list(edge.children), list(eager.children))
        for layer in edge.children.values():
            self.assertEqual(list(layer.children), [0, 1])

    def test_counts_snapshot_does_not_alias_mutable_specs(self):
        counts = [0, 0, 0, 0, 1, 0, 2, 0]
        specs = {MOVE: dict(moved_counts=counts, capture_counts=None)}
        root = az.PUCTNode(1.0, jq.RED)
        mcts = az.NeuralMCTS(None, az.AZConfig(lazy_chance_allocation=True), torch.device("cpu"))
        mcts._expand_node(root, [MOVE], np.ones(az.MOVE_SIZE), specs)
        saved = tuple(counts)
        counts[4] = 100
        self.assertEqual(root.children[MOVE].chance_counts[0], saved)

    def test_inconsistent_snapshot_preserves_eager_fallback(self):
        state, _, _, _ = make_edge(0, capture=True)
        for counts in ((None, None), ([0]*8, None), (None, [0]*8)):
            roots, searches = [], []
            for lazy in (False, True):
                root = az.PUCTNode(1.0, jq.RED)
                mcts = az.NeuralMCTS(None, az.AZConfig(lazy_chance_allocation=lazy), torch.device("cpu"))
                mcts._expand_node(root, [MOVE], np.ones(az.MOVE_SIZE),
                                  {MOVE: dict(moved_counts=counts[0], capture_counts=counts[1])})
                roots.append(root)
                searches.append(mcts)
            errors = []
            for root, mcts in zip(roots, searches):
                try:
                    mcts._navigate_chance(root.children[MOVE], state, Seats(9, 3))
                except Exception as error:
                    errors.append(type(error))
                else:
                    errors.append(None)
            self.assertEqual(errors[0], errors[1])
            materialize_graph(roots[1])
            self.assertEqual(pickle.dumps(roots[0]), pickle.dumps(roots[1]))
            self.assertEqual(searches[0].chance_outcomes_added, searches[1].chance_outcomes_added)

    def test_search_exact_inputs_outputs_rng_graph_noise_and_pruning(self):
        states = [jq.GameState.initial(rng=random.Random(3)), capture_position(True)[0], capture_position(False)[0]]
        for index, state in enumerate(states):
            belief = im.init_belief_for_game(state)
            for batch, dedup, loss in ((1, True, 0.0), (8, True, 3.0), (8, False, 3.0)):
                rows = []
                for lazy in (False, True, None):
                    inputs = []
                    net = TinyNet()
                    net.register_forward_pre_hook(lambda owner, args: inputs.append(args[0].numpy().copy()))
                    config = az.AZConfig(lazy_chance_allocation=lazy, leaf_batch_size=batch,
                                         leaf_dedup=dedup, virtual_loss=loss)
                    search = az.NeuralMCTS(net, config, torch.device("cpu"), random.Random(71))
                    before = pickle.dumps((state, belief))
                    policy, move, root = search.search(state, belief, [], 0.7, simulations=128,
                                                      add_dirichlet=True, prune_policy=True, return_root=True)
                    self.assertEqual(pickle.dumps((state, belief)), before)
                    materialize_graph(root)
                    rows.append((policy, move, search.rng.getstate(), search.leaf_evals_total,
                                 search.leaf_evals_unique, search.chance_applied, search.chance_nav_miss,
                                 search.chance_outcomes_added, pickle.dumps(root), inputs))
                for row in rows[1:]:
                    self.assertEqual(rows[0][:-1], row[:-1], (index, batch, dedup, loss))
                    self.assertEqual(len(rows[0][-1]), len(row[-1]))
                    for a, b in zip(rows[0][-1], row[-1]):
                        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
