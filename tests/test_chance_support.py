"""Immutable chance support must preserve probabilities, graph, inputs and RNG."""
import itertools
import pickle
import random
import unittest
from dataclasses import asdict
from unittest import mock

import numpy as np
import torch

import azero as az
import ismcts as im
import jieqi as jq
from test_chance_allocation import TinyNet, materialize_graph
from test_search_information_sets import capture_position
from tools import benchmark_chance_support as bench


def pool(rooks=1, knights=2, cannons=3):
    return (0, 0, rooks, knights, 0, 0, cannons, 0)


class TestChanceSupport(unittest.TestCase):
    def setUp(self):
        az._cached_full_chance_outcomes.cache_clear()
        # Exercise the cache on Python-only CI; production falls back without native support.
        if jq._immutable_integer_pools is None:
            guard = mock.patch.object(jq, "_immutable_integer_pools", lambda counts: all(
                row is None or (type(row) is tuple and all(type(n) is int for n in row))
                for row in counts))
            guard.start()
            self.addCleanup(guard.stop)

    def tearDown(self):
        az._cached_full_chance_outcomes.cache_clear()

    def test_small_pools_have_identical_order_probability_bits_and_types(self):
        pools = [tuple([0]*8), pool(), pool(0, 0, 1), pool(-1, 1, 2), pool(2**80, 2**79, 3)]
        for rooks, knights, cannons in itertools.product(range(3), repeat=3):
            pools.append(pool(rooks, knights, cannons))
        for moved, captured in itertools.product([None, *pools], repeat=2):
            counts = az._DeferredChanceCounts((moved, captured))
            expected = tuple(tuple(row) for row in az._full_chance_outcomes_uncached(counts))
            first = az._full_chance_outcomes(counts)
            second = az._full_chance_outcomes(tuple(counts))
            self.assertEqual(pickle.dumps(first), pickle.dumps(expected))
            self.assertIs(first, second)

    def test_cached_result_is_immutable_and_not_affected_by_later_specs(self):
        source = list(pool())
        counts = az._DeferredChanceCounts((tuple(source), None))
        result = az._full_chance_outcomes(counts)
        with self.assertRaises(TypeError):
            result[0][0] = (7, 1.0)
        with self.assertRaises(TypeError):
            result[0][0][1] = 0.0
        source[2] = 100
        self.assertIs(az._full_chance_outcomes(counts), result)
        self.assertNotEqual(az._full_chance_outcomes((tuple(source), None)), result)

    def test_mutable_and_numeric_duck_types_do_not_hit_integer_cache(self):
        base = pool()
        az._full_chance_outcomes((base, None))
        inputs = [(list(base), None), [base, None], (tuple(float(n) for n in base), None),
                  (tuple(np.int64(n) for n in base), None),
                  (tuple(np.float32(n) for n in base), None),
                  ((False, 0, True, 2, 0, 0, 3, 0), None)]
        info = az._cached_full_chance_outcomes.cache_info()
        for counts in inputs:
            expected = az._full_chance_outcomes_uncached(counts)
            actual = az._full_chance_outcomes(counts)
            self.assertEqual(pickle.dumps(actual), pickle.dumps(expected))
            self.assertIs(type(actual[0]), list)
        self.assertEqual(az._cached_full_chance_outcomes.cache_info(), info)

    def test_large_float_counts_are_not_confused_with_equal_integer_keys(self):
        integer = pool(2**55, 4, 3)
        floating = tuple(float(n) for n in integer)
        az._full_chance_outcomes((integer, None))
        expected = az._full_chance_outcomes_uncached((floating, None))
        actual = az._full_chance_outcomes((floating, None))
        self.assertEqual(pickle.dumps(actual), pickle.dumps(expected))
        self.assertNotEqual(tuple(expected[0]), az._full_chance_outcomes((integer, None))[0])

    def test_enumerator_and_pool_order_overrides_bypass_warm_cache(self):
        counts = (pool(), pool())
        az._full_chance_outcomes(counts)
        info = az._cached_full_chance_outcomes.cache_info()
        with mock.patch.object(az, "sample_reveal_outcomes", return_value=[(7, 0.123)]) as sample:
            self.assertEqual(az._full_chance_outcomes(counts), ([(7, 0.123)], [(7, 0.123)]))
            self.assertEqual(sample.call_count, 2)
        with mock.patch.object(jq, "POOL_TYPES", tuple(reversed(jq.POOL_TYPES))):
            expected = az._full_chance_outcomes_uncached(counts)
            self.assertEqual(az._full_chance_outcomes(counts), expected)
        self.assertEqual(az._cached_full_chance_outcomes.cache_info(), info)

    def test_invalid_inputs_preserve_original_exception_types(self):
        for counts in (None, (), (None,), ((), None), ([1], None), ((0,)*8, ())):
            errors = []
            for function in (az._full_chance_outcomes_uncached, az._full_chance_outcomes):
                try:
                    function(counts)
                except Exception as error:
                    errors.append(type(error))
                else:
                    errors.append(None)
            self.assertEqual(errors[0], errors[1], counts)

    def test_missing_compiled_guard_preserves_original_path(self):
        counts = (pool(), pool())
        az._full_chance_outcomes(counts)
        before = az._cached_full_chance_outcomes.cache_info()
        with mock.patch.object(jq, "_immutable_integer_pools", None):
            expected = az._full_chance_outcomes_uncached(counts)
            actual = az._full_chance_outcomes(counts)
            self.assertEqual(pickle.dumps(actual), pickle.dumps(expected))
            self.assertIs(type(actual[0]), list)
        self.assertEqual(az._cached_full_chance_outcomes.cache_info(), before)

    def test_cache_is_bounded_and_eviction_preserves_outputs(self):
        first = (pool(), None)
        expected = az._full_chance_outcomes(first)
        for count in range(4200):
            az._full_chance_outcomes((pool(count+100), None))
        info = az._cached_full_chance_outcomes.cache_info()
        self.assertEqual((info.maxsize, info.currsize), (4096, 4096))
        self.assertEqual(az._full_chance_outcomes(first), expected)

    def test_search_preserves_network_stream_graph_noise_pruning_and_rng(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            states = [jq.GameState.initial(rng=random.Random(3)),
                      capture_position(True)[0], capture_position(False)[0]]
            prepared = az._full_chance_outcomes
            for state in states:
                belief = im.init_belief_for_game(state)
                for batch, dedup in ((1, True), (8, True), (8, False)):
                    rows = []
                    for function in (az._full_chance_outcomes_uncached, prepared):
                        inputs, outputs = [], []
                        net = TinyNet()

                        def observe(owner, args, result):
                            inputs.append(args[0].numpy().tobytes())
                            outputs.append(tuple(t.numpy().tobytes() for t in result))

                        net.register_forward_hook(observe)
                        config = az.AZConfig(leaf_batch_size=batch, leaf_dedup=dedup)
                        search = az.NeuralMCTS(net, config, torch.device("cpu"), random.Random(71))
                        before = pickle.dumps((state, belief))
                        with mock.patch.object(az, "_full_chance_outcomes", function):
                            policy, move, root = search.search(state, belief, [], 0.7, simulations=128,
                                add_dirichlet=True, prune_policy=True, return_root=True)
                            materialize_graph(root)
                        self.assertEqual(pickle.dumps((state, belief)), before)
                        rows.append((policy, move, search.rng.getstate(), search.leaf_evals_total,
                                     search.leaf_evals_unique, search.chance_applied,
                                     search.chance_outcomes_added, search.chance_nav_miss,
                                     pickle.dumps(root), inputs, outputs))
                    self.assertEqual(rows[0], rows[1])
        finally:
            torch.set_num_threads(threads)

    def test_full_games_preserve_compact_samples_and_rng(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            prepared = az._full_chance_outcomes
            for strong in (False, True):
                rows = []
                for function in (az._full_chance_outcomes_uncached, prepared):
                    rng = random.Random(20265001)
                    config = az.AZConfig(mcts_simulations=16, strong_mcts_simulations=32, leaf_batch_size=8)
                    with mock.patch.object(az, "_full_chance_outcomes", function):
                        game = az.self_play_game(TinyNet(), config, torch.device("cpu"), rng, strong=strong)
                    self.assertIsNotNone(game.outcome)
                    rows.append(pickle.dumps((asdict(game), rng.getstate())))
                self.assertEqual(rows[0], rows[1])
        finally:
            torch.set_num_threads(threads)

    def test_probe_selects_actual_core_helper_and_balances_summary(self):
        self.assertIs(bench.PREPARED, az._full_chance_outcomes)
        self.assertIs(bench.REFERENCE, az._full_chance_outcomes_uncached)
        self.assertIs(bench.CACHE, az._cached_full_chance_outcomes)
        trials = [dict(repeat=repeat, mode=mode, wall_seconds=10 if mode == "uncached" else 9)
                  for repeat in range(4) for mode in bench.MODES]
        summary = bench.paired_summary(trials)
        self.assertAlmostEqual(summary["mean_wall_saving"], 0.1)
        self.assertEqual(summary["total_seconds"], dict(uncached=40, prepared=36))

    def test_probe_replays_original_snapshot_identity_and_type(self):
        generated = []

        def run_case(*args):
            for _ in range(3):
                counts = az._DeferredChanceCounts((tuple(list(pool())), None))
                generated.append(counts)
                az._full_chance_outcomes(counts)
            return {"exact": "same"}

        with mock.patch.object(bench.probe, "run_case", side_effect=run_case), mock.patch.object(
                bench.probe, "emit"):
            keys, requests, rows = bench.capture_requests(None, None, [{"metadata": {"index": 0}}])
        expected = generated[3:6]+generated[9:12]
        self.assertEqual(len(keys), 1)
        self.assertEqual(len({id(counts) for counts in requests}), 6)
        self.assertEqual([row["requests"] for row in rows], [3, 3])
        self.assertTrue(all(actual is original for actual, original in zip(requests, expected)))


@unittest.skipUnless(jq._immutable_integer_pools is not None, "build native rules extension")
class TestNativeSnapshotGuard(unittest.TestCase):
    def test_matches_exact_python_predicate_without_integer_conversion(self):
        guard = jq._immutable_integer_pools

        class Integer(int):
            pass

        class Tuple(tuple):
            pass

        pools = [None, (), pool(), pool(2**1000, -(2**1000), 3), list(pool()),
                 tuple(float(n) for n in pool()), tuple(np.int64(n) for n in pool()),
                 tuple(np.float32(n) for n in pool()), (False, True),
                 (object(),), ("1",), ((1,),), (Integer(3),), Tuple(pool())]
        for moved, captured in itertools.product(pools, repeat=2):
            counts = az._DeferredChanceCounts((moved, captured))
            expected = all(row is None or (type(row) is tuple and all(type(n) is int for n in row))
                           for row in counts)
            self.assertIs(guard(counts), expected)


if __name__ == "__main__":
    unittest.main()
