"""The precision experiment must restore flags and preserve the baseline path."""
import random
import unittest
from unittest import mock

import numpy as np
import torch

import azero as az
import ismcts as im
import jieqi as jq
from tools import benchmark_inference_precision as bench
from tools import central_inference as central


class TestPrecisionProbe(unittest.TestCase):
    def setUp(self):
        self.rng_state = torch.get_rng_state()

    def tearDown(self):
        torch.set_rng_state(self.rng_state)

    def test_scope_restores_flags_after_exception(self):
        before = central.precision_settings()
        with mock.patch.object(torch, "autocast", return_value=mock.MagicMock()):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with bench.precision_scope("tf32", before):
                    self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                    self.assertTrue(torch.backends.cudnn.allow_tf32)
                    raise RuntimeError("intentional")
        self.assertEqual(before, central.precision_settings())

    def test_unknown_mode_does_not_change_flags(self):
        before = central.precision_settings()
        with self.assertRaises(ValueError):
            with bench.precision_scope("unknown", before):
                pass
        self.assertEqual(before, central.precision_settings())

    def test_wrapper_promotes_outputs_without_mutating_weights(self):
        base = torch.nn.Linear(2, 2)
        before = base.weight.detach().clone()
        class HalfOutputs(torch.nn.Module):
            def forward(self, inputs):
                return base(inputs).half(), torch.ones(len(inputs), dtype=torch.bfloat16)
        net = bench.FP32Outputs(HalfOutputs()).eval()
        p, v = net(torch.ones(3, 2))
        self.assertEqual(p.dtype, torch.float32)
        self.assertEqual(v.dtype, torch.float32)
        self.assertTrue(torch.equal(before, base.weight))

    def test_output_metrics_capture_signed_zero_and_reject_nonfinite(self):
        p = np.array([[0.1, 0.9], [0.3, 0.7]], dtype=np.float32)
        v = np.array([0.0, 0.2], dtype=np.float32)
        same = bench.compare_outputs((p, v), (p.copy(), v.copy()))
        self.assertTrue(same["bitwise_equal"])
        other = v.copy()
        other[0] = -0.0
        self.assertFalse(bench.compare_outputs((p, v), (p, other))["bitwise_equal"])
        other[0] = np.inf
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            bench.compare_outputs((p, v), (p, other))

    def test_paired_summary_uses_wall_totals_not_worker_seconds(self):
        rows = [dict(repeat=r, mode=m, wall_seconds=s, worker_seconds=1000)
                for r in range(2) for m, s in (("baseline", 10), ("tf32", 8))]
        result = bench.paired_summary(rows, ("baseline", "tf32"))["tf32"]
        self.assertAlmostEqual(result["mean_paired_time_saving"], 0.2)
        self.assertAlmostEqual(result["throughput_ratio"], 1.25)
        self.assertEqual(result["total_seconds"], 16)

    def test_adjacent_balanced_orders_reverse_each_other(self):
        modes = bench.MODES[:4]
        for repeat in (0, 2, 4):
            self.assertEqual(bench.balanced_order(modes, repeat),
                             tuple(reversed(bench.balanced_order(modes, repeat+1))))

    def test_throughput_gate_is_a_predeclared_pilot_not_cherry_picked_rounds(self):
        def trials(savings):
            return [dict(repeat=r, mode=m, wall_seconds=s)
                    for r, saving in enumerate(savings)
                    for m, s in (("baseline", 10), ("amp_fp16", 10*(1-saving)))]
        self.assertFalse(bench.throughput_gate([], 0.05)["passed"])
        self.assertFalse(bench.throughput_gate(trials([0.2]*6), 0.05)["passed"])
        self.assertFalse(bench.throughput_gate(trials([0.02]*8), 0.05)["passed"])
        self.assertFalse(bench.throughput_gate(trials([0.5, -0.3]*4), 0.05)["passed"])
        result = bench.throughput_gate(trials([0.2]*8), 0.05)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["aggregate_time_saving"], 0.2)

    def test_disabled_gpu_telemetry_does_not_launch_subprocess(self):
        with mock.patch.object(bench, "gpu_snapshot") as snapshot:
            with bench.GpuTelemetry(0) as telemetry:
                pass
        snapshot.assert_not_called()
        self.assertEqual(telemetry.samples, [])

    def test_gpu_telemetry_is_joined_even_on_failure(self):
        with mock.patch.object(bench, "gpu_snapshot", return_value={"value": "test"}):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with bench.GpuTelemetry(60) as telemetry:
                    raise RuntimeError("intentional")
        self.assertFalse(telemetry.thread.is_alive())
        self.assertEqual(len(telemetry.samples), 2)
        self.assertTrue(all("elapsed_seconds" in row for row in telemetry.samples))

    def test_search_comparison_requires_paired_nonempty_inputs(self):
        for old, new in (([], []), ([{}], []), ([], [{}])):
            with self.assertRaises(ValueError):
                bench.compare_searches(old, new)

    def test_baseline_wrapper_preserves_search_policy_rng_and_weights(self):
        threads, rng = torch.get_num_threads(), torch.get_rng_state()
        try:
            torch.set_num_threads(1)
            torch.manual_seed(23)
            net = az.PolicyValueNet(channels=8, blocks=1).eval()
            before = {k: t.clone() for k, t in net.state_dict().items()}
            state = jq.GameState.initial(rng=random.Random(31))
            belief = im.init_belief_for_game(state)
            rows = []
            for candidate in (net, bench.FP32Outputs(net).eval()):
                search = az.NeuralMCTS(candidate, az.AZConfig(mcts_simulations=32, leaf_batch_size=8),
                                       torch.device("cpu"), random.Random(41))
                with torch.no_grad():
                    p, move = search.search(state, belief, [], 1.0, add_dirichlet=True, prune_policy=True)
                rows.append((p, move, search.rng.getstate(), search.leaf_evals_unique))
            self.assertEqual(rows[0], rows[1])
            self.assertTrue(all(torch.equal(t, net.state_dict()[k]) for k, t in before.items()))
        finally:
            torch.set_num_threads(threads)
            torch.set_rng_state(rng)


if __name__ == "__main__":
    unittest.main()
