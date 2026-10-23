"""Whole-game exclusion, real metric math, RNG/mode preservation and committed resume."""
import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import torch

import azero as az
import run_records as rr
import training_diagnostics as td
import test_training_checkpoint as checkpoint_tests
from tools.diagnose_training import inspect_diagnostics


class TestTrainingDiagnostics(TestCase):
    def setUp(self):
        self.harness = checkpoint_tests.TestTrainingCheckpoint()
        self.harness.setUp()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        self.harness.tearDown()
        torch.set_num_threads(self.threads)

    def config(self, name="run", **kwargs):
        return self.harness.loop_config(self.harness.root / name, diagnostic_games=1, **kwargs)

    @staticmethod
    def checkpoint(config):
        return Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME

    def saved(self, config):
        return torch.load(self.checkpoint(config), weights_only=True)

    def test_split_is_whole_game_disjoint_stratified_and_does_not_advance_rng(self):
        before = az.capture_training_rng(__import__("random").Random(9))
        flags = [(i + 1) % 8 == 0 for i in range(1024)]
        heldout, reference = td.select_games(flags, 32)
        self.assertEqual(len(heldout), 32)
        self.assertEqual(len(reference), 32)
        self.assertFalse(set(heldout) & set(reference))
        self.assertEqual(sum(flags[i] for i in heldout), 4)
        self.assertEqual(sum(flags[i] for i in reference), 4)
        self.assertEqual((heldout, reference), td.select_games(flags, 32))
        self.harness.equal_tree(before, az.capture_training_rng(__import__("random").Random(9)))
        with self.assertRaises(ValueError):
            td.select_games(flags[:3], 2)

    def test_metric_math_weights_entropy_value_and_calibration(self):
        class KnownNet(torch.nn.Module):
            def forward(self, states):
                return states[:, :2], states[:, 2]

        samples = [SimpleNamespace(policy_weight=1.0), SimpleNamespace(policy_weight=0.0),
                   SimpleNamespace(policy_weight=2.0)]
        triples = [(torch.tensor([math.log(.75), math.log(.25), 1.0]), torch.tensor([.5, .5]), torch.tensor(1.0)),
                   (torch.tensor([0., 0., -1.]), torch.tensor([1., 0.]), torch.tensor(1.0)),
                   (torch.tensor([math.log(.25), math.log(.75), 0.]), torch.tensor([1., 0.]), torch.tensor(-1.0))]
        lookup = {id(s): t for s, t in zip(samples, triples)}
        net = KnownNet()
        metrics = td.evaluate_targets(net, samples, torch.device("cpu"), lambda s: lookup[id(s)], batch_size=2)
        expected_ce = (-.5 * math.log(.75) - .5 * math.log(.25) - 2 * math.log(.25)) / 3
        self.assertAlmostEqual(metrics["policy_ce"], expected_ce, places=6)
        self.assertAlmostEqual(metrics["target_entropy"], math.log(2) / 3, places=6)
        self.assertAlmostEqual(metrics["policy_kl"], expected_ce - math.log(2) / 3, places=6)
        self.assertEqual(metrics["policy_positions"], 2)
        self.assertEqual(metrics["positions"], 3)
        self.assertAlmostEqual(metrics["value_mse"], 5 / 3)
        self.assertAlmostEqual(metrics["value_constant_baseline_mse"], 8 / 9)
        self.assertAlmostEqual(metrics["calibration_mse"], 5 / 3)
        self.assertEqual([b["count"] for b in metrics["calibration"]], [1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
        self.assertTrue(net.training)

    def test_all_fast_has_no_policy_metric_and_keeps_value(self):
        samples = copy.deepcopy(self.harness.samples)
        for sample in samples:
            sample.policy_weight = 0.0
        metrics = td.evaluate_targets(checkpoint_tests.TinyNet(), samples, torch.device("cpu"), az.compact_to_tensors, 2)
        self.assertIsNone(metrics["policy_kl"])
        self.assertIsNone(metrics["target_entropy"])
        self.assertEqual(metrics["policy_positions"], 0)
        self.assertGreaterEqual(metrics["value_mse"], 0)

    def test_eval_preserves_rng_bn_parameters_and_every_module_mode(self):
        net = checkpoint_tests.TinyNet()
        net.train()
        net.residual.eval()  # Mixed modes must survive diagnostics, too.
        modes = [m.training for m in net.modules()]
        tensors = copy.deepcopy(net.state_dict())
        rng = __import__("random").Random(97)
        before = az.capture_training_rng(rng)
        first = td.evaluate_targets(net, self.harness.samples, torch.device("cpu"), az.compact_to_tensors, 2)
        second = td.evaluate_targets(net, self.harness.samples, torch.device("cpu"), az.compact_to_tensors, 2)
        self.assertEqual(first, second)
        self.assertEqual(modes, [m.training for m in net.modules()])
        self.harness.equal_tree(tensors, net.state_dict())
        self.harness.equal_tree(before, az.capture_training_rng(rng))
        with self.assertRaisesRegex(RuntimeError, "encode failure"):
            td.evaluate_targets(net, self.harness.samples, torch.device("cpu"),
                                lambda _: (_ for _ in ()).throw(RuntimeError("encode failure")), 2)
        self.assertEqual(modes, [m.training for m in net.modules()])

    def test_leave_entire_game_out_before_all_augmentation_and_save_only_training(self):
        config = self.config(run_iterations=1)
        self.harness.run_loop(config)
        saved = self.saved(config)
        state = saved["diagnostics"]
        samples = td.load_fixed_sets(state, config.data_dir, az.load_compact_payload, saved["ruleset_version"])
        record = saved["replay_files"][0]
        raw, meta = az.load_compact_payload(str(Path(config.data_dir) / "selfplay" / record["name"]))
        excluded = state["dataset"]["sets"]["heldout"]["game_indices"]
        self.assertEqual(meta["excluded_holdout_game_indices"], excluded)
        self.assertEqual(meta["generated_games"], 2)
        self.assertEqual(meta["games"], 1)
        self.assertEqual(meta["game_lengths"], [2])
        self.assertEqual(meta["moves"], 2)
        self.assertEqual(meta["generated_moves"], 4)
        self.assertEqual(meta["outcomes"]["red"], 1)
        self.assertEqual(raw, samples["training_reference"])
        self.assertEqual(record["augmented_count"], 8)
        self.assertEqual(saved["buffer_size"], 8)
        self.assertEqual(saved["train_state"]["total_games"], 2)
        self.assertEqual(len(samples["heldout"]), 2)
        self.assertFalse(set(excluded) & set(meta["game_indices"]))
        self.assertEqual(saved["diagnostics"]["latest"]["training"]["expected_draws_per_buffer_sample"], 1.0)
        # The normal legacy loader also cannot accidentally load heldout files.
        buffer = az.ReplayBuffer(100)
        az.load_selfplay_data(str(Path(config.data_dir) / "selfplay"), buffer, augment=True, augment_flip=True)
        self.assertEqual(buffer.data, az.augment_samples(samples["training_reference"], mirror=True, flip=True))

    def test_old_checkpoint_enables_future_data_only_and_reuses_set_without_flag(self):
        config = replace(self.config(iterations=4, run_iterations=1), diagnostic_games=0)
        real_save = az.atomic_torch_save
        def legacy_save(payload, path):
            if payload.get("format") == "jieqi-training":
                payload = dict(payload)
                payload.pop("diagnostics")  # Older schema-v1 files do not have this field.
            real_save(payload, path)
        self.harness.run_loop(config, atomic_torch_save=dict(side_effect=legacy_save))
        self.assertNotIn("diagnostics", self.saved(config))
        self.harness.run_loop(replace(config, diagnostic_games=1))
        fixed = self.saved(config)["diagnostics"]["dataset"]
        self.assertEqual(fixed["created_iteration"], 1)
        self.assertEqual(self.saved(config)["buffer_size"], 23)
        self.harness.run_loop(config)  # No flag: automatically preserve/evaluate the committed set.
        third = self.saved(config)
        self.assertEqual(third["diagnostics"]["dataset"], fixed)
        self.assertEqual(third["train_state"]["iteration"], 3)
        raw, meta = az.load_compact_payload(str(Path(config.data_dir) / "selfplay" / third["replay_files"][-1]["name"]))
        self.assertEqual(len(raw), 4)
        self.assertEqual(meta["excluded_holdout_game_indices"], [])

    def test_actual_old_config_without_new_fields_restores_optimizer_and_rng(self):
        checkpoint, original, expected_state = self.harness.prepare()
        saved = torch.load(checkpoint, weights_only=True)
        saved.pop("diagnostics")
        for key in ("diagnostic_games", "diagnostic_batch_size", "run_iterations"):
            saved["config"].pop(key)
        az.atomic_torch_save(saved, str(checkpoint))
        config = replace(self.harness.config, diagnostic_games=1, run_iterations=1)
        resumed, state, _ = self.harness.restore(checkpoint, config)
        self.assertEqual(state, expected_state)
        self.harness.equal_tree(original[1].state_dict(), resumed[1].state_dict())
        self.harness.equal_tree(saved["rng"], az.capture_training_rng(resumed[3]))

    def test_continuous_and_additional_iteration_resume_match_real_optimization_and_rng(self):
        continuous = self.config("continuous", iterations=3)
        resumed = self.config("resumed", iterations=3, run_iterations=1)
        self.harness.seed()
        self.harness.run_loop(continuous)
        self.harness.seed()
        for _ in range(3):
            self.harness.run_loop(resumed)
        original, restored = self.saved(continuous), self.saved(resumed)
        for key in ("model", "optimizer", "scheduler", "train_state", "rng", "buffer_size"):
            self.harness.equal_tree(original[key], restored[key])
        for key in ("before", "after", "training", "heldout_minus_reference"):
            self.assertEqual(original["diagnostics"]["latest"][key], restored["diagnostics"]["latest"][key])
        self.assertEqual(restored["scheduler"]["T_max"], 3)
        self.assertEqual(restored["scheduler"]["last_epoch"], 3)

    def test_buffer_growth_preserves_fixed_sets_and_diagnostics_across_bootstrap_and_restart(self):
        config = self.config("growth", iterations=4, run_iterations=2)
        self.harness.run_loop(config)
        previous = self.saved(config)
        self.assertEqual(previous["buffer_size"], 23)
        fixed_files = {path: path.read_bytes() for path in
                       (Path(config.data_dir) / "diagnostics" / "sets").glob("*/*.pkl.gz")}
        grown = replace(config, replay_buffer_size=50, allow_buffer_growth=True, run_iterations=1)
        with mock.patch.object(td, "create_fixed_sets", side_effect=AssertionError("must keep fixed sets")):
            with self.assertRaisesRegex(InterruptedError, "stop after migration"):
                self.harness.run_loop(grown, generate_selfplay_results=dict(
                    side_effect=InterruptedError("stop after migration")))
            migrated = self.saved(config)
            self.harness.equal_tree(migrated["diagnostics"], previous["diagnostics"])
            for key in ("model", "optimizer", "scheduler", "train_state", "rng"):
                self.harness.equal_tree(migrated[key], previous[key])
            self.assertEqual(migrated["replay_files"][0]["skip_augmented"], 1)
            self.harness.run_loop(replace(grown, allow_buffer_growth=False))
        continued = self.saved(config)
        self.assertEqual(continued["diagnostics"]["dataset"], previous["diagnostics"]["dataset"])
        self.assertEqual(continued["diagnostics"]["latest"]["completed_iterations"], 3)
        self.assertEqual(continued["buffer_size"], 39)
        self.assertEqual(continued["provenance"]["manifest"]["parent"]["resume_mode"], "full")
        for path, contents in fixed_files.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_failed_first_round_does_not_adopt_orphan_set_on_retry(self):
        config = self.config(run_iterations=1)
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.harness.run_loop(config, train_step=dict(side_effect=RuntimeError("training failed")))
        self.assertIsNone(self.saved(config)["diagnostics"])
        folders = list((Path(config.data_dir) / "diagnostics" / "sets").iterdir())
        old_files = {p: p.read_bytes() for p in folders[0].iterdir()}
        self.harness.run_loop(config)
        self.assertNotEqual(self.saved(config)["diagnostics"]["dataset"]["source"]["attempt_id"], folders[0].name)
        for path, payload in old_files.items():
            self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(self.saved(config)["buffer_size"], 8)

    def test_uncommitted_report_is_visible_but_cannot_be_mistaken_for_completed_iteration(self):
        config = self.config(run_iterations=1)
        real_save = az.save_training_checkpoint
        def fail_commit(*args, **kwargs):
            if args[4].iteration:
                raise OSError("commit failed")
            return real_save(*args, **kwargs)
        with self.assertRaisesRegex(OSError, "commit failed"):
            self.harness.run_loop(config, save_training_checkpoint=dict(side_effect=fail_commit))
        status = inspect_diagnostics(config.data_dir, self.checkpoint(config))
        self.assertIsNone(status["dataset"])
        self.assertEqual([r["status"] for r in status["reports"]], ["unconfirmed"])
        self.harness.run_loop(config)
        status = inspect_diagnostics(config.data_dir, self.checkpoint(config))
        self.assertEqual(sorted(r["status"] for r in status["reports"]), ["committed", "unconfirmed"])

    def test_missing_or_changed_dataset_and_changed_count_refuse_resume(self):
        config = self.config(run_iterations=1)
        self.harness.run_loop(config)
        record = self.saved(config)["diagnostics"]["dataset"]["sets"]["heldout"]
        path = td.dataset_path(config.data_dir, record["path"])
        contents = path.read_bytes()
        path.write_bytes(contents + b"changed")
        with self.assertRaisesRegex(RuntimeError, "诊断数据内容改变"):
            self.harness.run_loop(config)
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "拒绝静默降级"):
            self.harness.run_loop(config)
        path.write_bytes(contents)
        with self.assertRaisesRegex(RuntimeError, "已有固定留出局数不可改变"):
            self.harness.run_loop(replace(config, diagnostic_games=2))

    def test_report_can_be_recovered_from_committed_checkpoint_after_marker_failure(self):
        config = self.config(run_iterations=1)
        real_commit = rr.commit_record
        def fail_marker(data_dir, lineage, checkpoint):
            if lineage["attempt"] is not None:
                raise OSError("marker failed")
            return real_commit(data_dir, lineage, checkpoint)
        with mock.patch.object(rr, "commit_record", side_effect=fail_marker):
            with self.assertRaisesRegex(OSError, "marker failed"):
                self.harness.run_loop(config)
        self.assertEqual(inspect_diagnostics(config.data_dir, self.checkpoint(config))["reports"][0]["status"], "committed")
        report = self.saved(config)["diagnostics"]["latest"]
        path = Path(config.data_dir) / "diagnostics" / "reports" / (report["provenance"]["attempt_id"] + ".json")
        path.unlink()
        # Reopen preserving the same cosine plan, then stop before generating new data.
        with self.assertRaisesRegex(InterruptedError, "no new game"):
            self.harness.run_loop(config, generate_selfplay_results=dict(side_effect=InterruptedError("no new game")))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
        self.assertEqual(inspect_diagnostics(config.data_dir)["reports"][0]["status"], "committed")

    def test_invalid_path_and_nonfinite_metrics_are_rejected(self):
        for path in ("../models/latest.pt", "diagnostics/sets/../../heldout.pkl.gz", "C:/tmp/heldout.pkl.gz"):
            with self.assertRaises(ValueError):
                td.dataset_path(self.harness.root, path)
        class BadNet(torch.nn.Module):
            def forward(self, states):
                return torch.zeros(len(states), 2), torch.full((len(states),), float("nan"))
        with self.assertRaisesRegex(ValueError, "非有限值"):
            td.evaluate_targets(BadNet(), [SimpleNamespace(policy_weight=1.0)], torch.device("cpu"),
                                lambda _: (torch.zeros(1), torch.tensor([1., 0.]), torch.tensor(1.)))

    def test_changed_committed_report_is_not_overwritten_on_resume(self):
        config = self.config(run_iterations=1)
        self.harness.run_loop(config)
        report = self.saved(config)["diagnostics"]["latest"]
        path = Path(config.data_dir) / "diagnostics" / "reports" / (report["provenance"]["attempt_id"] + ".json")
        changed = dict(report, evaluated_model_sha256="changed")
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "immutable run record differs"):
            self.harness.run_loop(config)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), changed)
        with self.assertRaisesRegex(ValueError, "differs from committed checkpoint"):
            inspect_diagnostics(config.data_dir, self.checkpoint(config))

    def test_cli_validates_diagnostics_and_stop_limit(self):
        config = az.parse_args(["--iterations", "2000", "--run-iterations", "1", "--diagnostic-games", "32"])
        self.assertEqual(config.iterations, 2000)
        self.assertEqual(config.run_iterations, 1)
        self.assertEqual(config.diagnostic_games, 32)
        for arg, value in (("--run-iterations", "-1"), ("--diagnostic-games", "-1"), ("--diagnostic-batch-size", "0")):
            with self.assertRaises(SystemExit):
                az.parse_args([arg, value])
