"""Provenance must describe actual execution, including retries and interrupted commits."""
import json
import hashlib
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

import azero as az
import run_records as rr
import test_training_checkpoint as checkpoint_tests
from tools import eval as eval_tool, run_status


class TestRunRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.harness = checkpoint_tests.TestTrainingCheckpoint("runTest")
        self.harness.setUp()
        self.root = self.harness.root

    def tearDown(self):
        self.harness.tearDown()

    def config(self, **kwargs):
        return self.harness.loop_config(self.root / "training", **kwargs)

    def saved(self, config):
        return torch.load(Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME, weights_only=True)

    def manifest(self, **kwargs):
        return rr.create_run(self.root, "training", kwargs.get("config", {}), None, {}, {}, "cpu",
                             replay_files=kwargs.get("replay_files"))

    def test_records_do_not_consume_any_training_rng(self):
        net = checkpoint_tests.TinyNet()
        rng = random.Random(43)
        before = az.capture_training_rng(rng)
        manifest = self.manifest()
        rr.start_attempt(self.root, manifest, 0, rr.model_hash(net), {})
        self.harness.equal_tree(before, az.capture_training_rng(rng))

    def test_manifest_copies_mutable_inputs(self):
        config, replay = {"nested": {"sims": 8}}, [{"name": "old", "augmented_count": 4}]
        manifest = self.manifest(config=config, replay_files=replay)
        digest = rr.record_hash(manifest)
        replay.append({"name": "new"})
        config["nested"]["sims"] = 100
        self.assertEqual(digest, rr.record_hash(manifest))
        self.assertEqual(len(manifest["initial_replay_files"]), 1)
        self.assertEqual(manifest["config"]["nested"]["sims"], 8)

    def test_model_hash_includes_weights_and_bn_buffers(self):
        net = checkpoint_tests.TinyNet()
        first = rr.model_hash(net)
        self.assertEqual(first, rr.model_hash(net))
        net.residual[0].bn1.num_batches_tracked.add_(1)
        self.assertNotEqual(first, rr.model_hash(net))
        second = rr.model_hash(net)
        with torch.no_grad():
            net.conv.weight[0, 0, 0, 0].add_(0.1)
        self.assertNotEqual(second, rr.model_hash(net))

    def test_code_fingerprint_covers_untracked_runtime_source_and_git_unavailable(self):
        root = self.root / "code"
        (root / "tools").mkdir(parents=True)
        (root / "azero.py").write_text("x = 1\n", encoding="utf-8")
        extra = root / "tools" / "new.py"
        extra.write_text("y = 1\n", encoding="utf-8")
        with mock.patch.object(rr.subprocess, "run", side_effect=FileNotFoundError):
            before = rr.code_record(root)
            extra.write_text("y = 2\n", encoding="utf-8")
            after = rr.code_record(root)
        self.assertNotEqual(before["sha256"], after["sha256"])
        self.assertIn("tools/new.py", after["files"])
        self.assertIsNone(after["git"]["dirty"])
        self.assertFalse(after["git"]["available"])

    def test_code_fingerprint_covers_native_source_and_binary(self):
        root = self.root / "native_code"
        (root / "native").mkdir(parents=True)
        source = root / "native/jieqi_rules.pyx"
        binary = root / "_jieqi_rules.cp311-win_amd64.pyd"
        source.write_text("x = 1\n", encoding="utf-8")
        binary.write_bytes(b"test binary")
        with mock.patch.object(rr.subprocess, "run", side_effect=FileNotFoundError):
            first = rr.code_record(root)
            binary.write_bytes(b"different binary")
            second = rr.code_record(root)
            source.write_text("x = 2\n", encoding="utf-8")
            third = rr.code_record(root)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(second["sha256"], third["sha256"])
        self.assertIn("native/jieqi_rules.pyx", third["files"])
        self.assertIn(binary.name, third["files"])
        self.assertEqual(self.manifest()["runtime"]["rules_backend"], az.jq.RULES_BACKEND)

    def test_loaded_native_identity_matches_manifest_and_code_fingerprint(self):
        manifest = self.manifest()
        loaded = az.jq.RULES_BACKEND_NATIVE
        self.assertEqual(manifest["runtime"]["rules_backend_native"], loaded)
        if loaded is None:
            self.assertEqual(az.jq.RULES_BACKEND, "python")
            return
        self.assertEqual({key: loaded[key] for key in ("path", "sha256", "bytes")},
                         rr.file_record(loaded["path"]))
        self.assertEqual(loaded["api_version"], 1)
        self.assertEqual(loaded["ruleset_version"], az.jq.RULESET_VERSION)
        self.assertEqual(loaded["source_sha256"], rr.file_record(rr.PROJECT_ROOT / "native/jieqi_rules.pyx")["sha256"])
        binary = Path(loaded["path"])
        try:
            key = binary.relative_to(rr.PROJECT_ROOT).as_posix()
        except ValueError:
            key = binary.as_posix()
        self.assertEqual(manifest["code"]["files"][key], loaded["sha256"])

    def test_changed_loaded_file_refuses_inaccurate_provenance(self):
        binary = self.root / "external.so"
        binary.write_bytes(b"changed after load")
        loaded = dict(path=str(binary), sha256=hashlib.sha256(b"loaded content").hexdigest())
        with mock.patch.object(az.jq, "RULES_BACKEND_NATIVE", loaded):
            with self.assertRaisesRegex(RuntimeError, "loaded native rules file changed"):
                rr.code_record()

    def test_external_native_extension_and_preloaded_module_are_recorded(self):
        try:
            import _jieqi_rules as native
        except ImportError:
            self.skipTest("build native extension first")
        source = rr.PROJECT_ROOT / "native/jieqi_rules.pyx"
        if native.SOURCE_SHA256 != rr.file_record(source)["sha256"]:
            self.skipTest("rebuild current native extension first")
        external = self.root / "external_extension"
        external.mkdir()
        binary = external / Path(native.__file__).name
        content = Path(native.__file__).read_bytes()
        # Windows PE overlays change the digest without modifying executable sections.
        if sys.platform == "win32":
            content += b"native-provenance-regression-overlay"
        binary.write_bytes(content)
        expected = rr.file_record(binary)
        code = """
import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
if sys.argv[3] == "preload":
    import _jieqi_rules
import jieqi as jq, run_records as rr
manifest = rr.create_run(sys.argv[2], "training", {}, None, {}, {}, "cpu")
assert jq.RULES_BACKEND == "cython", jq.RULES_BACKEND_ERROR
assert jq.RULES_BACKEND_NATIVE["path"] == str(Path(sys.argv[1], sys.argv[4]).resolve())
assert manifest["code"]["files"][Path(jq.RULES_BACKEND_NATIVE["path"]).as_posix()] == jq.RULES_BACKEND_NATIVE["sha256"]
print(json.dumps(manifest))
"""
        for mode in ("import", "preload"):
            with self.subTest(mode=mode):
                target = self.root / mode
                result = subprocess.run([sys.executable, "-c", code, str(external), str(target), mode, binary.name],
                                        cwd=rr.PROJECT_ROOT, env=dict(os.environ, JIEQI_RULES_BACKEND="cython"),
                                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                manifest = json.loads(result.stdout)
                loaded = manifest["runtime"]["rules_backend_native"]
                self.assertEqual({key: loaded[key] for key in expected}, expected)
                self.assertEqual(loaded["source_sha256"], native.SOURCE_SHA256)
                self.assertEqual(loaded["api_version"], native.API_VERSION)
                self.assertEqual(loaded["ruleset_version"], native.RULESET_VERSION)
                rr.validate_lineage(dict(manifest=manifest, manifest_sha256=rr.record_hash(manifest)))
                if sys.platform == "win32":
                    self.assertNotEqual(loaded["sha256"], rr.file_record(native.__file__)["sha256"])

    def test_immutable_records_are_atomic_and_never_overwritten(self):
        path = self.root / "record.json"
        rr.write_record(path, {"one": 1})
        first = path.read_bytes()
        rr.write_record(path, {"one": 1})
        with self.assertRaisesRegex(ValueError, "immutable"):
            rr.write_record(path, {"one": 2})
        self.assertEqual(first, path.read_bytes())
        with mock.patch.object(rr.os, "link", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                rr.write_record(self.root / "absent.json", {"one": 1})
        self.assertFalse((self.root / "absent.json").exists())
        self.assertEqual(list(self.root.glob(".record-*")), [])

    def test_two_rounds_trace_generating_and_trained_models(self):
        config = self.config()
        self.harness.run_loop(config)
        saved = self.saved(config)
        lineage = saved["provenance"]
        manifest = lineage["manifest"]
        self.assertEqual(manifest["config"], saved["config"])
        self.assertEqual(manifest["network"], saved["network"])
        self.assertEqual(manifest["versions"], dict(ruleset=az.jq.RULESET_VERSION, data=az.DATA_VERSION, search=az.SEARCH_VERSION))
        self.assertEqual(manifest["parent"]["resume_mode"], "fresh")
        self.assertEqual(manifest["code"]["files"]["run_records.py"], rr.file_record(rr.PROJECT_ROOT / "run_records.py")["sha256"])
        sources = []
        for record in saved["replay_files"]:
            _, meta = az.load_compact_payload(str(Path(config.data_dir) / "selfplay" / record["name"]))
            sources.append(meta["provenance"])
        self.assertEqual(sources[0]["generating_model_sha256"], manifest["parent"]["model_sha256"])
        folder = rr.run_dir(config.data_dir, manifest["run_id"])
        first_commit = json.loads((folder / "commits" / (sources[0]["attempt_id"] + ".json")).read_text())
        self.assertEqual(sources[1]["generating_model_sha256"], first_commit["model_sha256"])
        self.assertEqual(sources[1]["parent_checkpoint"]["sha256"], first_commit["checkpoint_sha256"])
        self.assertNotEqual(sources[1]["generating_model_sha256"], lineage["model_sha256"])
        self.assertEqual(rr.attempt_status(config.data_dir, sources[1]), "committed")
        report = run_status.inspect_runs(config.data_dir)
        self.assertEqual([a["status"] for a in report["runs"][0]["attempts"]], ["committed", "committed"])

    def test_restart_has_new_run_and_exact_parent_checkpoint(self):
        config = self.config()
        with self.assertRaises(InterruptedError):
            self.harness.run_loop(config, stop_after=1)
        previous = self.saved(config)["provenance"]
        checkpoint = Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME
        previous_hash = rr.file_record(checkpoint)["sha256"]
        self.harness.run_loop(config)
        current = self.saved(config)["provenance"]["manifest"]
        self.assertNotEqual(previous["manifest"]["run_id"], current["run_id"])
        self.assertEqual(current["parent"]["run_id"], previous["manifest"]["run_id"])
        self.assertEqual(current["parent"]["checkpoint"]["sha256"], previous_hash)
        self.assertEqual(current["parent"]["model_sha256"], previous["model_sha256"])
        report = run_status.inspect_runs(config.data_dir, checkpoint)
        statuses = [a["status"] for run in report["runs"] for a in run["attempts"]]
        self.assertEqual(statuses.count("committed"), 2)
        self.assertEqual(statuses.count("unconfirmed"), 1)

    def test_evaluation_before_failed_checkpoint_stays_unconfirmed_after_retry(self):
        config = self.config(iterations=1, eval_games=2, eval_interval=1, eval_workers=1)
        real_save = az.save_training_checkpoint
        def fail_round(path, net, opt, scheduler, state, *args, **kwargs):
            if state.iteration == 1:
                raise OSError("injected checkpoint failure")
            real_save(path, net, opt, scheduler, state, *args, **kwargs)
        with self.assertRaises(OSError):
            self.harness.run_loop(config, save_training_checkpoint=dict(side_effect=fail_round),
                                  _play_eval_game=dict(return_value=0.0))
        eval_path = Path(config.data_dir) / "eval" / "train_eval.jsonl"
        failed = json.loads(eval_path.read_text(encoding="utf-8").splitlines()[0])
        source = failed["provenance"]
        self.assertEqual(source["iteration"], 0)
        self.assertEqual(failed["pairs"], 1)
        self.assertIn("base_seed", failed)
        self.assertNotEqual(source["evaluated_model_sha256"], source["generating_model_sha256"])
        self.assertEqual(rr.attempt_status(config.data_dir, source, self.saved(config)["provenance"]), "unconfirmed")
        self.harness.run_loop(config, _play_eval_game=dict(return_value=0.0))
        successful = json.loads(eval_path.read_text(encoding="utf-8").splitlines()[1])
        self.assertEqual(rr.attempt_status(config.data_dir, successful["provenance"]), "committed")
        self.assertEqual(rr.attempt_status(config.data_dir, source), "unconfirmed")
        self.assertEqual(source["evaluated_model_sha256"], successful["provenance"]["evaluated_model_sha256"])
        self.assertNotEqual(source["attempt_id"], successful["provenance"]["attempt_id"])

    def test_marker_failure_keeps_checkpoint_authoritative_and_resume_repairs(self):
        config = self.config(iterations=1)
        real_commit = rr.commit_record
        def fail_marker(data_dir, lineage, checkpoint):
            if lineage.get("attempt") is not None:
                raise OSError("injected marker failure")
            real_commit(data_dir, lineage, checkpoint)
        with mock.patch.object(rr, "commit_record", side_effect=fail_marker), self.assertRaises(OSError):
            self.harness.run_loop(config)
        saved = self.saved(config)
        source = saved["provenance"]["attempt"]
        self.assertEqual(saved["train_state"]["iteration"], 1)
        self.assertEqual(rr.attempt_status(config.data_dir, source), "unconfirmed")
        self.assertEqual(rr.attempt_status(config.data_dir, source, saved["provenance"]), "committed")
        report = run_status.inspect_runs(config.data_dir, Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME)
        self.assertEqual(report["runs"][0]["attempts"][0]["status"], "committed")
        self.harness.run_loop(config)  # No further round, still repairs the existing marker.
        self.assertEqual(rr.attempt_status(config.data_dir, source), "committed")

    def test_previous_full_checkpoint_without_provenance_remains_compatible(self):
        config = self.config()
        checkpoint, _, _ = self.harness.prepare(config)
        saved = torch.load(checkpoint, weights_only=True)
        saved.pop("provenance")
        Path(config.model_dir).mkdir(parents=True)
        target = Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME
        az.atomic_torch_save(saved, str(target))
        target_selfplay = Path(config.data_dir) / "selfplay"
        target_selfplay.mkdir()
        (target_selfplay / "committed.pkl.gz").write_bytes((self.harness.selfplay / "committed.pkl.gz").read_bytes())
        self.harness.run_loop(config)
        parent = self.saved(config)["provenance"]["manifest"]["parent"]
        self.assertEqual(parent["resume_mode"], "full")
        self.assertEqual(parent["history"], "unrecorded")
        self.assertIsNone(parent["run_id"])

    def test_invalid_lineage_refuses_resume_instead_of_inventing_source(self):
        config = self.config(iterations=1)
        self.harness.run_loop(config)
        saved = self.saved(config)
        saved["provenance"]["manifest"]["config"]["chance_nodes"] = False
        az.atomic_torch_save(saved, str(Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME))
        with self.assertRaisesRegex(RuntimeError, "manifest hash mismatch"):
            self.harness.run_loop(config)

    def test_legacy_parent_records_file_and_loaded_model_without_invented_run_id(self):
        config = self.config(iterations=1)
        Path(config.model_dir).mkdir(parents=True)
        net = checkpoint_tests.TinyNet()
        model_path = Path(config.model_dir) / az.MODEL_LATEST_NAME
        az.save_model(net, str(model_path))
        original = rr.file_record(model_path)["sha256"]
        loaded_hash = rr.model_hash(net)
        self.harness.run_loop(config)
        parent = self.saved(config)["provenance"]["manifest"]["parent"]
        self.assertEqual(parent["resume_mode"], "legacy_weights")
        self.assertEqual(parent["model_file"]["sha256"], original)
        self.assertEqual(parent["model_sha256"], loaded_hash)
        self.assertEqual(parent["history"], "unrecorded")
        self.assertIsNone(parent["run_id"])

    def test_optimizer_reset_is_identified_separately_in_run_lineage(self):
        from dataclasses import replace
        config = self.config()
        with self.assertRaises(InterruptedError):
            self.harness.run_loop(config, stop_after=1)
        previous = self.saved(config)["provenance"]["manifest"]["run_id"]
        config = replace(config, reset_optimizer=True, learning_rate=config.learning_rate / 2)
        self.harness.run_loop(config)
        manifest = self.saved(config)["provenance"]["manifest"]
        self.assertEqual(manifest["parent"]["resume_mode"], "reset_optimizer")
        self.assertEqual(manifest["parent"]["run_id"], previous)
        self.assertTrue(manifest["config"]["reset_optimizer"])

    def test_standalone_evaluation_records_model_code_and_completion(self):
        net = checkpoint_tests.TinyNet()
        path = self.root / "eval.jsonl"
        fake = dict(outcome=az.jq.RESULT_DRAW, neural_side="red")
        with mock.patch.object(eval_tool, "play_one_game", return_value=fake):
            eval_tool.evaluate(net, torch.device("cpu"), ["random"], 2, 8, 17, 2, str(path), quiet=True)
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        source = records[0]["provenance"]
        self.assertEqual(source["model_sha256"], rr.model_hash(net))
        self.assertEqual(records[0]["manifest"]["config"]["max_plies"], 2)
        self.assertTrue(all(r["provenance"] == source for r in records))
        complete = rr.run_dir(self.root, source["run_id"]) / "complete.json"
        marker = json.loads(complete.read_text(encoding="utf-8"))
        self.assertEqual(marker["output"]["sha256"], rr.file_record(path)["sha256"])
        self.assertEqual(run_status.inspect_runs(self.root)["runs"][0]["status"], "complete")
        path.write_text("edited results\n", encoding="utf-8")
        self.assertEqual(run_status.inspect_runs(self.root)["runs"][0]["status"], "output_changed")

    def test_interrupted_standalone_evaluation_has_no_completion_marker(self):
        path = self.root / "eval.jsonl"
        with mock.patch.object(eval_tool, "play_one_game", side_effect=InterruptedError), self.assertRaises(InterruptedError):
            eval_tool.evaluate(checkpoint_tests.TinyNet(), torch.device("cpu"), ["random"], 2, 8, 17, 2, str(path), quiet=True)
        self.assertEqual(run_status.inspect_runs(self.root)["runs"][0]["status"], "unconfirmed")

    def test_evaluation_never_overwrites_an_existing_output(self):
        path = self.root / "eval.jsonl"
        path.write_bytes(b"existing user results")
        with self.assertRaises(FileExistsError):
            eval_tool.evaluate(None, None, ["random"], 2, 8, 17, 2, str(path))
        self.assertEqual(path.read_bytes(), b"existing user results")
        self.assertFalse((self.root / "runs").exists())

    def test_opponent_source_change_is_rejected_for_serial_and_worker_loading(self):
        path = self.root / "opponent.pt"
        path.write_bytes(b"original")
        expected = rr.file_record(path)
        path.write_bytes(b"changed")
        spec = "model:" + str(path) + ":8"
        with mock.patch.object(eval_tool.azero, "PolicyValueNet", checkpoint_tests.TinyNet):
            with self.assertRaisesRegex(RuntimeError, "来源"):
                eval_tool.build_opponent_factory(spec, torch.device("cpu"), checkpoint_tests.TinyNet(), expected_source=expected)
            with mock.patch.dict(eval_tool._WORKER_STATE, {"opponents": {}, "opponent_sources": {spec: expected}}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "来源"):
                    eval_tool._build_opponent_factory_worker(spec, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
