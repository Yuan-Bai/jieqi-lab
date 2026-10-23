"""Ordering, isolation, and exact-output contracts for the long benchmark."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from tools import benchmark_compiled_paths as bench


class TestCompiledBenchmark(unittest.TestCase):
    def test_williams_orders_balance_slots_and_predecessors(self):
        orders = bench.balanced_orders()
        modes = set(bench.MODES)
        for order in orders:
            self.assertEqual(set(order), modes)
        for slot in range(8):
            self.assertEqual({order[slot] for order in orders}, modes)
        neighbors = [(left, right) for order in orders for left, right in zip(order, order[1:])]
        self.assertEqual(len(neighbors), len(set(neighbors)))
        self.assertEqual(len(neighbors), 56)

    def test_modes_switch_three_methods_and_restore(self):
        owners = ((bench.az.PUCTNode, "__init__"), (bench.az.PUCTNode, "expand"),
                  (bench.jq.GameState, "_iter_legal_moves_fast"))
        before = [getattr(owner, name) for owner, name in owners]
        fake_nodes = mock.Mock(initialize_node=lambda *a: None, expand=lambda *a: None)
        fake_legal = mock.Mock(iter_legal_moves_fast=lambda *a: None)
        with mock.patch.object(bench.node_probe, "native", fake_nodes), \
                mock.patch.object(bench.legal_probe, "native", fake_legal), bench.preserve_methods():
            for mode, (initialize, expand, legal) in bench.MODES.items():
                bench.set_mode(mode)
                expected = [fake_nodes.initialize_node if initialize else bench.az.PUCTNode._init_python,
                            fake_nodes.expand if expand else bench.az.PUCTNode._expand_python,
                            fake_legal.iter_legal_moves_fast if legal else bench.jq.GameState._iter_legal_moves_fast_python]
                self.assertEqual([getattr(owner, name) for owner, name in owners], expected)
        self.assertEqual([getattr(owner, name) for owner, name in owners], before)

    def test_strong_schedule_matches_production(self):
        self.assertEqual([i for i in range(32) if bench.strong_game(i, 8)], [7, 15, 23, 31])
        self.assertFalse(bench.strong_game(7, 0))

    def test_worker_checks_all_loaded_identities_before_switching(self):
        rules, nodes, legal = {}, ("cython", {"node": 1}), ("cython", {"legal": 1})
        with mock.patch.object(bench.probe, "init_worker") as initialize, mock.patch.object(bench, "set_mode") as switch:
            bench.init_worker({}, {}, rules, nodes, legal)
        initialize.assert_called_once_with({}, {}, rules, search_backend=nodes, legal_backend=legal)
        switch.assert_called_once_with("baseline")
        with mock.patch.object(bench.probe, "init_worker", side_effect=AssertionError), mock.patch.object(bench, "set_mode") as switch:
            with self.assertRaises(AssertionError):
                bench.init_worker({}, {}, rules, nodes, legal)
            switch.assert_not_called()

    def test_bad_preflight_never_loads_checkpoint_or_writes_journal(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(sys, "argv", ["benchmark", "--output", str(Path(tmp)/"result.json")]), \
                mock.patch.object(bench.jq, "LEGAL_BACKEND", "python"), \
                mock.patch.object(bench.torch, "load") as load, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                bench.main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_summary_uses_paired_rounds_and_factorial_contexts(self):
        trials = []
        for repeat in range(2):
            for mode, flags in bench.MODES.items():
                trials.append(dict(repeat=repeat, mode=mode, wall_seconds=10 * (0.9 if flags[0] else 1)))
        summary = bench.summarize(trials)
        self.assertAlmostEqual(summary["init"]["mean"], 0.1)
        self.assertEqual(len(summary["init"]["paired_values"]), 2)
        self.assertAlmostEqual(bench.factorial_summary(trials)["initialization"]["mean"], 0.1)
        self.assertAlmostEqual(bench.factorial_summary(trials)["expansion"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()
