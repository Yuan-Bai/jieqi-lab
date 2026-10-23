"""Exactness and lifetime checks for leaf output and repetition snapshot reuse."""
import copy
import random
import unittest
from unittest import mock

import numpy as np
import torch

import azero as az
import ismcts as im
import jieqi as jq
from test_encoding_acceleration import fixtures, reference_encode


class ReferenceMCTS(az.NeuralMCTS):
    def _evaluate_batch(self, infos, histories, **kwargs):
        self.net.eval()
        with torch.no_grad():
            batch = az._encode_batch_numpy(infos, histories)
            logits, values = self.net(torch.from_numpy(batch).to(self.device))
            return torch.softmax(logits, dim=1).tolist(), values.tolist()


def tree_records(root):
    indices = {id(root): 0}
    nodes = [root]
    records = []

    def edges(mapping):
        result = []
        for key, node in mapping.items():
            if id(node) not in indices:
                indices[id(node)] = len(nodes)
                nodes.append(node)
            result.append((key, indices[id(node)]))
        return result

    for node in nodes:
        records.append((node.prior.hex(), node.value_sum.hex(), node.visits, node.player_to_move,
                        node.move, node.is_chance, node.reveal_moved, node.reveal_captured, node.chance_counts,
                        edges(node.children), edges(node.public_outcomes), edges(node.decision_nodes)))
    return records


class TestSelfplayAcceleration(unittest.TestCase):
    def setUp(self):
        self.torch_rng = torch.get_rng_state().clone()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_rng_state(self.torch_rng)
        torch.set_num_threads(self.threads)

    def test_prepared_history_matches_input_and_signature_without_recomputing(self):
        pairs = fixtures()
        entries = [az._repetition_entries(info) for info, _ in pairs]
        signatures = [az.leaf_input_signature(*pair) for pair in pairs]
        expected = np.stack([reference_encode(*pair) for pair in pairs])
        with mock.patch.object(az, "_repetition_entries", side_effect=AssertionError("recomputed history")):
            for pair, snapshot, expected_signature in zip(pairs, entries, signatures):
                self.assertEqual(az.leaf_input_signature(*pair, repetition_entries=snapshot), expected_signature)
            actual = az._encode_batch_numpy(*zip(*pairs), repetition_entries=entries)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(actual.strides, expected.strides)

    def test_known_incomplete_history_is_not_confused_with_missing_snapshot(self):
        info, history = fixtures()[0]
        self.assertIsNone(az._repetition_entries(info))
        expected = az.encode_state_numpy(info, history)
        with mock.patch.object(az, "_repetition_entries", side_effect=AssertionError("unexpected history query")):
            actual = az._encode_batch_numpy([info], [history], repetition_entries=[None])[0]
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(float(actual[75, 0, 0]), 0.0)

    def test_no_snapshot_survives_info_mutation_or_next_batch(self):
        info, history = fixtures()[5]
        snapshot = az._repetition_entries(info)
        saved = copy.deepcopy(snapshot)
        before_signature = az.leaf_input_signature(info, history)
        before = az._encode_batch_numpy([info], [history], repetition_entries=[snapshot])
        info.position_counts[(info.side, tuple(info.board))] += 1
        self.assertEqual(snapshot, saved)
        self.assertNotEqual(az.leaf_input_signature(info, history), before_signature)
        after = az._encode_batch_numpy([info], [history])
        self.assertEqual(after[0].tobytes(), reference_encode(info, history).tobytes())
        self.assertNotEqual(before.tobytes(), after.tobytes())

    def test_snapshot_batch_length_is_checked(self):
        info, history = fixtures()[1]
        for entries in ([], [None, None]):
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                az._encode_batch_numpy([info], [history], repetition_entries=entries)

    def test_policy_array_and_python_values_match_list_output_bits(self):
        torch.manual_seed(17)
        net = az.PolicyValueNet(channels=8, blocks=1)
        pairs = fixtures()[:9]
        infos, histories = map(list, zip(*pairs))
        entries = [az._repetition_entries(info) for info in infos]
        ref = ReferenceMCTS(net, az.AZConfig(), torch.device("cpu"))
        mcts = az.NeuralMCTS(net, az.AZConfig(), torch.device("cpu"))
        expected_policy, expected_values = ref._evaluate_batch(infos, histories)
        policies, values = mcts._evaluate_batch(infos, histories, repetition_entries=entries)
        self.assertEqual(policies.dtype, np.float32)
        self.assertEqual(policies.shape, (len(infos), az.MOVE_SIZE))
        self.assertTrue(policies.flags.c_contiguous)
        self.assertFalse(policies.flags.writeable)
        self.assertEqual(policies.tobytes(), np.asarray(expected_policy, dtype=np.float32).tobytes())
        self.assertEqual(values, expected_values)
        self.assertTrue(all(type(value) is float for value in values))
        before = policies.copy()
        state = jq.GameState.initial(rng=random.Random(22))
        moves = state.legal_moves()
        a, b = az.PUCTNode(1.0, state.side), az.PUCTNode(1.0, state.side)
        a.expand(moves, expected_policy[0])
        b.expand(moves, policies[0])
        self.assertEqual(tree_records(a), tree_records(b))
        self.assertEqual(policies.tobytes(), before.tobytes())

    def test_complete_search_and_rng_match_original_output_path(self):
        torch.manual_seed(31)
        net = az.PolicyValueNet(channels=8, blocks=1)
        state = jq.GameState.initial(rng=random.Random(19))
        belief = im.init_belief_for_game(state)
        for chance, chance_k in ((False, 0), (True, 0), (True, 1)):
            for dedup in (False, True):
                with self.subTest(chance=chance, chance_k=chance_k, dedup=dedup):
                    config = az.AZConfig(mcts_simulations=64, leaf_batch_size=16, leaf_dedup=dedup,
                                         chance_nodes=chance, chance_k=chance_k)
                    first = ReferenceMCTS(net, config, torch.device("cpu"), random.Random(27))
                    second = az.NeuralMCTS(net, config, torch.device("cpu"), random.Random(27))
                    kwargs = dict(temperature=1.0, add_dirichlet=True, prune_policy=True, return_root=True)
                    policy_a, move_a, root_a = first.search(state, belief, [], **kwargs)
                    policy_b, move_b, root_b = second.search(state, belief, [], **kwargs)
                    self.assertEqual(policy_a, policy_b)
                    self.assertEqual(move_a, move_b)
                    self.assertEqual(tree_records(root_a), tree_records(root_b))
                    self.assertEqual(first.rng.getstate(), second.rng.getstate())
                    for attr in ("leaf_evals_total", "leaf_evals_unique", "chance_applied",
                                 "chance_outcomes_added", "chance_nav_miss"):
                        self.assertEqual(getattr(first, attr), getattr(second, attr), attr)

    def test_search_prepares_history_once_per_collected_leaf(self):
        torch.manual_seed(37)
        net = az.PolicyValueNet(channels=8, blocks=1)
        state = jq.GameState.initial(rng=random.Random(23))
        belief = im.init_belief_for_game(state)
        for dedup in (True, False):
            config = az.AZConfig(mcts_simulations=64, leaf_batch_size=16, leaf_dedup=dedup)
            mcts = az.NeuralMCTS(net, config, torch.device("cpu"), random.Random(41))
            with mock.patch.object(az, "_repetition_entries", wraps=az._repetition_entries) as prepare:
                mcts.search(state, belief, [], 1.0)
            self.assertGreater(mcts.leaf_evals_total, 0)
            self.assertEqual(prepare.call_count, 1 + mcts.leaf_evals_total)


if __name__ == "__main__":
    unittest.main()
