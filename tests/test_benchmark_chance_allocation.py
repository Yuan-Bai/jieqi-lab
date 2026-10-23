"""Benchmark audit helpers must not merge visits or contaminate timing modes."""
import random
import unittest

import torch

import azero as az
from test_chance_allocation import deferred_edge
from test_chance_lazy_sampling import make_edge, Seats
from tools import benchmark_chance_allocation as bench
from tools import profile_selfplay as probe


class TestChanceAllocationBenchmark(unittest.TestCase):
    def test_canonical_hash_includes_unvisited_branches_and_shared_decisions(self):
        state, eager, _, _ = make_edge(0, capture=True)
        root, edge, search = deferred_edge(state)
        self.assertEqual(bench.canonical_graph(root), probe.digest(eager))
        actual, _ = search._navigate_chance(edge, state, Seats(9, 3))
        actual.visits = 9
        shared = az.PUCTNode(1.0, state.side)
        actual.decision_nodes[("a",)] = shared
        root.decision_nodes[("b",)] = shared
        before = search.rng.getstate()
        self.assertNotEqual(bench.canonical_graph(root), probe.digest(eager))
        self.assertEqual(search.rng.getstate(), before)
        self.assertIs(actual.decision_nodes[("a",)], root.decision_nodes[("b",)])
        self.assertEqual(actual.visits, 9)

    def test_trace_keeps_batch_boundaries_inputs_and_raw_outputs(self):
        trace = bench.NetworkTrace()
        net = torch.nn.Identity()
        x = torch.ones(3, 8)
        trace.hook(net, (x,), (x, torch.zeros(3)))
        trace.hook(net, (x[:1],), (x[:1], torch.zeros(1)))
        self.assertEqual([row[0][0] for row in trace.rows], [(3, 8), (1, 8)])
        other = bench.NetworkTrace()
        other.hook(net, (x,), (x*2, torch.zeros(3)))
        self.assertEqual(trace.rows[0][0], other.rows[0][0])
        self.assertNotEqual(trace.rows[0][1], other.rows[0][1])

    def test_summary_uses_cohort_wall_not_summed_worker_times(self):
        rows = [dict(repeat=i, mode=mode, wall_seconds=seconds, worker_seconds=random.random()*100)
                for i in range(4) for mode, seconds in (("eager", 10), ("lazy", 8))]
        summary = bench.paired_summary(rows)
        self.assertAlmostEqual(summary["mean_wall_saving"], 0.2)
        self.assertAlmostEqual(summary["throughput_ratio"], 1.25)
        self.assertEqual(summary["total_seconds"], dict(eager=40, lazy=32))
        for value in summary["bootstrap_mean_interval_95"]:
            self.assertAlmostEqual(value, 0.2)

    def test_memory_snapshot_reports_process_values_not_tree_bytes(self):
        result = bench.windows_memory()
        if result is not None:
            self.assertGreater(result["working_set"], 0)
            self.assertGreaterEqual(result["peak_working_set"], result["working_set"])
            self.assertGreater(result["private_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
