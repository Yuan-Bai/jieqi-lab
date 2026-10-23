"""Precision adapters reuse paired evaluation without lowering the baseline opponent."""
from contextlib import contextmanager
import unittest
from unittest import mock

import torch

import jieqi as jq
from tools import eval_inference_precision as experiment


class TestPrecisionMatches(unittest.TestCase):
    def test_match_output_guard_preserves_finite_outputs_and_rejects_nan(self):
        class Outputs(torch.nn.Module):
            def forward(self, inputs):
                return inputs, inputs
        net = experiment.CheckedOutputs(Outputs())
        inputs = torch.tensor([[0.0, -0.0, 1.0]])
        logits, values = net(inputs)
        self.assertIs(logits, inputs)
        self.assertIs(values, inputs)
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            net(torch.tensor([[float("nan")]]))

    def test_tasks_share_seed_and_swap_candidate_color(self):
        tasks = experiment.paired_tasks(3, 111, 600)
        self.assertEqual(len(tasks), 6)
        for pair in range(3):
            red, black = tasks[2*pair:2*pair+2]
            self.assertEqual(red, ("amp_fp16", pair, 111+pair, jq.RED, 600))
            self.assertEqual(black, ("amp_fp16", pair, 111+pair, jq.BLACK, 600))

    def test_player_scopes_end_before_other_player_acts(self):
        modes = []
        active = []

        @contextmanager
        def scope(mode, settings):
            self.assertFalse(active)
            active.append(mode)
            modes.append(mode)
            try:
                yield
            finally:
                active.pop()

        with mock.patch.object(experiment.evaluation.azero, "NeuralMCTS") as mcts, \
                mock.patch.object(experiment.bench, "precision_scope", side_effect=scope):
            mcts.return_value.search.side_effect = lambda *args, **kwargs: ([], (1, 2))
            candidate = experiment.PrecisionOpponent(None, None, None, None, "amp_fp16", {})
            baseline = experiment.PrecisionOpponent(None, None, None, None, "baseline", {})
            self.assertEqual(candidate.select_move(None, None, []), (1, 2))
            self.assertFalse(active)
            self.assertEqual(baseline.select_move(None, None, []), (1, 2))
        self.assertFalse(active)
        self.assertEqual(modes, ["amp_fp16", "baseline"])

    def test_adapter_restores_scope_when_search_raises(self):
        active = []

        @contextmanager
        def scope(mode, settings):
            active.append(mode)
            try:
                yield
            finally:
                active.pop()

        with mock.patch.object(experiment.evaluation.azero, "NeuralMCTS") as mcts, \
                mock.patch.object(experiment.bench, "precision_scope", side_effect=scope):
            mcts.return_value.search.side_effect = RuntimeError("intentional")
            agent = experiment.PrecisionOpponent(None, None, None, None, "amp_fp16", {})
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                agent.select_move(None, None, [])
        self.assertFalse(active)
        self.assertEqual(agent.moves, [])

    def test_strength_entry_rejects_unverified_or_failed_throughput(self):
        for evidence in ({}, {"args": {"pilot_min_time_saving": 0.01}, "production_unchanged": True}):
            with self.assertRaisesRegex(ValueError, "predeclared"):
                experiment.validate_throughput(evidence)
        rows = [dict(repeat=r, mode=m, wall_seconds=t)
                for r in range(8) for m, t in (("baseline", 10), ("amp_fp16", 11))]
        evidence = dict(args={"pilot_min_time_saving": 0.05}, production_unchanged=True,
                        game_trials=rows, throughput_gate={"passed": True})
        with self.assertRaisesRegex(ValueError, "did not pass"):
            experiment.validate_throughput(evidence)


if __name__ == "__main__":
    unittest.main()
