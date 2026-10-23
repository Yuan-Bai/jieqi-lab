"""完整续训：真实优化等价、提交失败、缓冲顺序与旧纯权重兼容。"""
import copy
import concurrent.futures
import gc
import json
import logging
import os
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

import azero as az
import ismcts as im
import jieqi as jq


class TinyNet(torch.nn.Module):
    """保留 BN 与随机 dropout，用小网络测试优化/RNG，而非模拟更新结果。"""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(az.STATE_CHANNELS, 4, 1, bias=False)
        self.residual = torch.nn.Sequential(az.ResidualBlock(4))
        self.policy_fc = torch.nn.Linear(4, az.MOVE_SIZE)
        self.value_fc = torch.nn.Linear(4, 1)

    def forward(self, inputs):
        features = self.residual(F.relu(self.conv(inputs))).mean(dim=(2, 3))
        features = F.dropout(features, p=0.2, training=self.training)
        return self.policy_fc(features), self.value_fc(features).tanh().squeeze(1)


def make_samples():
    state = jq.GameState.initial(rng=random.Random(19))
    belief = im.init_belief_for_game(state)
    history, raw = [], []
    for _ in range(4):
        move = state.legal_moves()[0]
        policy = [0.0] * az.MOVE_SIZE
        policy[az.move_to_index(move)] = 1.0
        raw.append(az.raw_sample_from_info(belief.info_state(state, state.side), history, policy))
        im.apply_move_and_update_belief(state, belief, move, rng=random.Random(23))
        history.append(move)
    return az.finalize_samples(raw, jq.RESULT_RED_WIN)


class TestTrainingCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.selfplay = self.root / "selfplay"
        self.selfplay.mkdir()
        self.config = az.AZConfig(iterations=4, batch_size=4, train_steps=2,
                                  replay_buffer_size=23, self_play_games=2,
                                  self_play_workers=1, eval_games=0)
        self.samples = make_samples()
        self.logger = logging.getLogger("checkpoint-test")
        self.seed()

    def tearDown(self):
        for handler in list(logging.getLogger("azero").handlers):
            handler.close()
            logging.getLogger("azero").removeHandler(handler)
        self.tmp.cleanup()
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)

    @staticmethod
    def seed():
        random.seed(31)
        np.random.seed(37)
        torch.manual_seed(41)

    def objects(self, config=None, device="cpu"):
        config = config or self.config
        net = TinyNet().to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.iterations, eta_min=config.lr_min)
                     if config.lr_schedule == "cosine" else None)
        return net, optimizer, scheduler, random.Random(43), az.ReplayBuffer(config.replay_buffer_size)

    def prepare(self, config=None, device="cpu"):
        config = config or self.config
        objects = self.objects(config, device)
        net, optimizer, scheduler, rng, buffer = objects
        meta = dict(game_lengths=[2, 2], ruleset_version=jq.RULESET_VERSION)
        path = self.selfplay / "committed.pkl.gz"
        az.save_compact_samples(self.samples, str(path), meta)
        count = az.add_saved_samples(buffer, self.samples, meta, True, True)
        records = [az.replay_file_record(str(path), count)]
        self.optimize(objects, device)
        state = az.TrainingState(iteration=1, total_games=2)
        checkpoint = self.root / az.TRAINING_CHECKPOINT_NAME
        az.save_training_checkpoint(str(checkpoint), net, optimizer, scheduler, state, rng, config, records, len(buffer))
        return checkpoint, objects, state

    @staticmethod
    def optimize(objects, device="cpu"):
        net, optimizer, scheduler, rng, buffer = objects
        net.train()
        for _ in range(2):
            rng.random()
            np.random.rand()
            torch.rand(1, device=device)
            az.train_step(net, optimizer, buffer.sample(4), torch.device(device), 1.0)
        if scheduler is not None:
            scheduler.step()

    def restore(self, checkpoint, config=None, device="cpu"):
        config = config or self.config
        objects = self.objects(config, device)
        net, optimizer, scheduler, rng, buffer = objects
        state, records, best = az.load_training_checkpoint(str(checkpoint), net, optimizer, scheduler, rng,
                                                         config, buffer, str(self.selfplay), self.logger)
        return objects, state, records

    def equal_tree(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.equal_tree(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual))
            for a, b in zip(expected, actual):
                self.equal_tree(a, b)
        else:
            self.assertEqual(expected, actual)

    def test_real_adamw_updates_match_continuation_with_all_rng_and_bn(self):
        checkpoint, uninterrupted, state = self.prepare()
        expected_buffer = copy.deepcopy(uninterrupted[-1].data)
        self.optimize(uninterrupted)
        expected_rng = az.capture_training_rng(uninterrupted[3])
        resumed, restored_state, records = self.restore(checkpoint)
        self.assertEqual(resumed[-1].data, expected_buffer)
        self.assertEqual(restored_state, state)
        self.optimize(resumed)
        self.equal_tree(uninterrupted[0].state_dict(), resumed[0].state_dict())
        self.equal_tree(uninterrupted[1].state_dict(), resumed[1].state_dict())
        self.equal_tree(uninterrupted[2].state_dict(), resumed[2].state_dict())
        self.equal_tree(expected_rng, az.capture_training_rng(resumed[3]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_optimizer_and_rng_roundtrip(self):
        checkpoint, uninterrupted, _ = self.prepare(device="cuda")
        self.optimize(uninterrupted, "cuda")
        expected_rng = az.capture_training_rng(uninterrupted[3])
        resumed, _, _ = self.restore(checkpoint, device="cuda")
        self.optimize(resumed, "cuda")
        self.equal_tree(uninterrupted[0].state_dict(), resumed[0].state_dict())
        self.equal_tree(uninterrupted[1].state_dict(), resumed[1].state_dict())
        self.equal_tree(expected_rng, az.capture_training_rng(resumed[3]))

    def test_none_scheduler_restores_and_allows_extending_iteration_limit(self):
        config = replace(self.config, lr_schedule="none")
        checkpoint, original, _ = self.prepare(config)
        resumed, _, _ = self.restore(checkpoint, replace(config, iterations=20))
        self.assertIsNone(resumed[2])
        self.equal_tree(original[1].state_dict(), resumed[1].state_dict())

    def test_changed_optimizer_config_requires_explicit_reset(self):
        checkpoint, _, _ = self.prepare()
        for key, value in (("learning_rate", 2e-4), ("iterations", 8), ("lr_schedule", "none")):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "reset-optimizer"):
                self.restore(checkpoint, replace(self.config, **{key: value}))
        config = replace(self.config, learning_rate=2e-4, reset_optimizer=True)
        resumed, state, _ = self.restore(checkpoint, config)
        self.assertEqual(len(resumed[1].state), 0)
        az.fast_forward_scheduler(resumed[2], state.iteration)
        expected = config.lr_min + (config.learning_rate - config.lr_min) * (1 + np.cos(np.pi / config.iterations)) / 2
        self.assertAlmostEqual(resumed[1].param_groups[0]["lr"], expected)

    def test_old_checkpoint_without_allocation_flag_resumes_both_modes(self):
        checkpoint, original, expected_state = self.prepare()
        saved = torch.load(checkpoint, weights_only=True)
        saved["config"].pop("lazy_chance_allocation")
        manifest = az.rr.create_run(self.root, "training", saved["config"], saved["network"],
                                    dict(ruleset=jq.RULESET_VERSION, data=az.DATA_VERSION, search=az.SEARCH_VERSION),
                                    dict(mode="test"), torch.device("cpu"))
        saved["provenance"] = dict(manifest=manifest, manifest_sha256=az.rr.record_hash(manifest),
                                   attempt=None, completed_iterations=expected_state.iteration,
                                   model_sha256=az.rr.model_hash(original[0]))
        az.atomic_torch_save(saved, str(checkpoint))
        before = checkpoint.read_bytes()
        for lazy in (False, True):
            restored, state, records = self.restore(checkpoint, replace(self.config, lazy_chance_allocation=lazy))
            self.assertEqual(state, expected_state)
            self.equal_tree(restored[0].state_dict(), saved["model"])
            self.equal_tree(restored[1].state_dict(), saved["optimizer"])
            self.equal_tree(restored[2].state_dict(), saved["scheduler"])
            self.equal_tree(az.capture_training_rng(restored[3]), saved["rng"])
            self.assertEqual(restored[-1].data, original[-1].data)
            self.assertEqual(records, saved["replay_files"])
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_eager_checkpoint_resumes_with_new_default_without_reset(self):
        checkpoint, original, expected_state = self.prepare(replace(self.config, lazy_chance_allocation=False))
        saved = torch.load(checkpoint, weights_only=True)
        self.assertIs(saved["config"]["lazy_chance_allocation"], False)
        self.assertIs(self.config.lazy_chance_allocation, True)
        before = checkpoint.read_bytes()
        restored, state, records = self.restore(checkpoint)
        self.assertEqual(state, expected_state)
        self.equal_tree(restored[0].state_dict(), saved["model"])
        self.equal_tree(restored[1].state_dict(), saved["optimizer"])
        self.equal_tree(restored[2].state_dict(), saved["scheduler"])
        self.equal_tree(az.capture_training_rng(restored[3]), saved["rng"])
        self.assertEqual(restored[-1].data, original[-1].data)
        self.assertEqual(records, saved["replay_files"])
        self.assertIs(self.config.lazy_chance_allocation, True)
        self.assertEqual(checkpoint.read_bytes(), before)

    def test_incompatible_versions_architecture_and_replay_settings_refused(self):
        checkpoint, _, _ = self.prepare()
        valid = torch.load(checkpoint, weights_only=True)
        for key in ("version", "ruleset_version", "data_version", "search_version", "network"):
            invalid = dict(valid, **{key: -1})
            az.atomic_torch_save(invalid, str(checkpoint))
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "不兼容"):
                self.restore(checkpoint)
        az.atomic_torch_save(valid, str(checkpoint))
        with self.assertRaisesRegex(RuntimeError, "replay_buffer_size"):
            self.restore(checkpoint, replace(self.config, replay_buffer_size=24))

    def test_explicit_growth_preserves_fifo_optimizer_rng_and_underfilled_roundtrip(self):
        config = replace(self.config, replay_buffer_size=9)
        checkpoint, original, expected_state = self.prepare(config)
        saved = torch.load(checkpoint, weights_only=True)
        saved["config"].pop("allow_buffer_growth")  # Existing checkpoints have no opt-in field.
        az.atomic_torch_save(saved, str(checkpoint))
        previous_bytes = checkpoint.read_bytes()
        az.save_compact_samples([replace(self.samples[0], value=99.0)],
                               str(self.selfplay / "orphan.pkl.gz"))
        grown_config = replace(config, replay_buffer_size=29, allow_buffer_growth=True)
        with mock.patch.object(az, "load_selfplay_data", side_effect=AssertionError("must not scan")):
            resumed, state, records = self.restore(checkpoint, grown_config)
        self.assertEqual(checkpoint.read_bytes(), previous_bytes)
        self.assertEqual(state, expected_state)
        self.assertEqual(resumed[-1].capacity, 29)
        self.assertEqual(resumed[-1].data, original[-1].data)
        self.assertEqual(records[0]["skip_augmented"], 7)
        self.assertEqual(records[0]["augmented_count"], 16)
        self.equal_tree(resumed[0].state_dict(), saved["model"])
        self.equal_tree(resumed[1].state_dict(), saved["optimizer"])
        self.equal_tree(resumed[2].state_dict(), saved["scheduler"])
        self.equal_tree(az.capture_training_rng(resumed[3]), saved["rng"])

        def commit(objects, current_records, current_config):
            net, optimizer, scheduler, rng, buffer = objects
            az.save_training_checkpoint(str(checkpoint), net, optimizer, scheduler, state, rng,
                                        current_config, current_records, len(buffer))

        commit(resumed, records, grown_config)
        ordinary_config = replace(grown_config, allow_buffer_growth=False)
        restored, _, records = self.restore(checkpoint, ordinary_config)
        self.assertEqual(restored[-1].data, original[-1].data)
        self.equal_tree(az.capture_training_rng(restored[3]), saved["rng"])
        same_size, _, same_records = self.restore(checkpoint, grown_config)
        self.assertEqual(same_size[-1].data, original[-1].data)
        self.assertEqual(same_records, records)
        # Further growth preserves an existing offset.
        again_config = replace(grown_config, replay_buffer_size=45)
        again, _, again_records = self.restore(checkpoint, again_config)
        self.assertEqual(again[-1].data, original[-1].data)
        self.assertEqual(again_records, records)

        meta = dict(game_lengths=[2, 2], ruleset_version=jq.RULESET_VERSION)
        fresh = [replace(sample, value=0.25) for sample in self.samples]
        for index, expected_size in enumerate((25, 29)):
            path = self.selfplay / f"fresh-{index}.pkl.gz"
            az.save_compact_samples(fresh, str(path), meta)
            count = az.add_saved_samples(restored[-1], fresh, meta, True, True)
            records.append(az.replay_file_record(str(path), count))
            expected_data = copy.deepcopy(restored[-1].data)
            commit(restored, records, ordinary_config)
            restored, _, records = self.restore(checkpoint, ordinary_config)
            self.assertEqual(len(restored[-1]), expected_size)
            self.assertEqual(restored[-1].data, expected_data)
        self.assertNotIn("committed.pkl.gz", [record["name"] for record in records])
        # Once full, a later growth must also persist the new partial-file FIFO boundary.
        final, _, final_records = self.restore(checkpoint, replace(ordinary_config, replay_buffer_size=50,
                                                                   allow_buffer_growth=True))
        self.assertEqual(final[-1].data, expected_data)
        self.assertEqual(final_records[0]["skip_augmented"], 3)

    def test_growth_flag_does_not_allow_shrink_augmentation_or_optimizer_changes(self):
        checkpoint, _, _ = self.prepare()
        for size in (22, 0, -1):
            with self.subTest(size=size), self.assertRaisesRegex(RuntimeError, "只支持从正容量扩容"):
                self.restore(checkpoint, replace(self.config, replay_buffer_size=size, allow_buffer_growth=True))
        for key in ("augment_mirror", "augment_flip"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                self.restore(checkpoint, replace(self.config, replay_buffer_size=24, allow_buffer_growth=True,
                                                 **{key: False}))
        with self.assertRaisesRegex(RuntimeError, "reset-optimizer"):
            self.restore(checkpoint, replace(self.config, replay_buffer_size=24, allow_buffer_growth=True,
                                             learning_rate=2e-4))
        source = self.selfplay / "committed.pkl.gz"
        source.write_bytes(source.read_bytes() + b"tampered")
        with self.assertRaisesRegex(RuntimeError, "重放文件内容改变"):
            self.restore(checkpoint, replace(self.config, replay_buffer_size=24, allow_buffer_growth=True))

    def test_invalid_replay_prefix_is_rejected(self):
        checkpoint, _, _ = self.prepare()
        saved = torch.load(checkpoint, weights_only=True)
        for skipped in (-1, 17, True, "1"):
            invalid = copy.deepcopy(saved)
            invalid["replay_files"][0]["skip_augmented"] = skipped
            az.atomic_torch_save(invalid, str(checkpoint))
            with self.subTest(skipped=skipped), self.assertRaisesRegex(RuntimeError, "无效重放"):
                self.restore(checkpoint)

    def test_corrupt_checkpoint_and_changed_or_missing_replay_do_not_fall_back(self):
        checkpoint, _, _ = self.prepare()
        source = self.selfplay / "committed.pkl.gz"
        contents = source.read_bytes()
        source.write_bytes(contents + b"changed")
        with self.assertRaisesRegex(RuntimeError, "内容改变"):
            self.restore(checkpoint)
        source.unlink()
        with self.assertRaisesRegex(RuntimeError, "拒绝静默降级"):
            self.restore(checkpoint)
        source.write_bytes(contents)
        checkpoint.write_bytes(b"partial write")
        with self.assertRaisesRegex(RuntimeError, "拒绝静默降级"):
            self.restore(checkpoint)

    def test_atomic_write_or_commit_failure_preserves_prior_checkpoint(self):
        checkpoint, _, _ = self.prepare()
        previous = checkpoint.read_bytes()
        for target in ("torch.save", "os.replace"):
            with self.subTest(target=target), mock.patch("azero." + target, side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    az.atomic_torch_save(dict(changed=True), str(checkpoint))
            self.assertEqual(checkpoint.read_bytes(), previous)
            self.assertFalse(list(self.root.glob(".checkpoint-*.tmp")))

    def test_inconsistent_counter_or_buffer_manifest_cannot_replace_checkpoint(self):
        checkpoint, objects, state = self.prepare()
        net, optimizer, scheduler, rng, buffer = objects
        previous = checkpoint.read_bytes()
        record = az.replay_file_record(str(self.selfplay / "committed.pkl.gz"), 16)
        for invalid_state, invalid_size in ((replace(state, iteration=2), len(buffer)),
                                            (state, len(buffer) + 1)):
            with self.assertRaisesRegex(ValueError, "拒绝提交"):
                az.save_training_checkpoint(str(checkpoint), net, optimizer, scheduler, invalid_state,
                                             rng, self.config, [record], invalid_size)
            self.assertEqual(checkpoint.read_bytes(), previous)

    def test_file_manifest_ignores_newer_files_and_mtime_changes(self):
        checkpoint, original, _ = self.prepare()
        az.save_compact_samples(self.samples, str(self.selfplay / "uncommitted.pkl.gz"),
                               dict(ruleset_version=jq.RULESET_VERSION))
        os.utime(self.selfplay / "committed.pkl.gz", (1, 1))
        resumed, _, records = self.restore(checkpoint)
        self.assertEqual(resumed[-1].data, original[-1].data)
        self.assertEqual([r["name"] for r in records], ["committed.pkl.gz"])

    def test_game_boundaries_keep_fifo_order_on_legacy_loader(self):
        expected = az.ReplayBuffer(11)
        for game in (self.samples[:2], self.samples[2:]):
            expected.add_game(az.augment_samples(game, mirror=True, flip=True))
        path = self.selfplay / "groups.pkl.gz"
        az.save_compact_samples(self.samples, str(path), dict(game_lengths=[2, 2], ruleset_version=jq.RULESET_VERSION))
        actual = az.ReplayBuffer(11)
        records = []
        az.load_selfplay_data(str(self.selfplay), actual, augment=True, augment_flip=True, loaded_files=records)
        self.assertEqual(expected.data, actual.data)
        self.assertEqual(records[0]["augmented_count"], 16)
        with self.assertRaisesRegex(ValueError, "game_lengths"):
            az.add_saved_samples(actual, self.samples, dict(game_lengths=[3]), True, True)

    def test_batched_replay_preserves_per_game_fifo_and_sampling_for_all_symmetries(self):
        games = [[replace(sample, value=index + offset / 10, policy_weight=0.5)
                  for offset, sample in enumerate(self.samples[:1 + index % 4])]
                 for index in range(12)]
        flags = [(False, False), (True, False), (False, True), (True, True)]
        for stored_mirror, stored_flip in flags:
            stored = [az.augment_samples(game, stored_mirror, stored_flip) for game in games]
            samples = [sample for game in stored for sample in game]
            meta = dict(game_lengths=[len(game) for game in stored],
                        augment=stored_mirror, augment_flip=stored_flip)
            for mirror, flip in flags:
                augmented = [az.augment_samples(game, mirror or stored_mirror, flip or stored_flip)
                             for game in games]
                for capacity in (1, 11, 37, 0, -2):
                    with self.subTest(stored=(stored_mirror, stored_flip), requested=(mirror, flip),
                                      capacity=capacity):
                        # 独立保留旧的逐局 FIFO 操作，覆盖非空缓冲与部分整局裁剪。
                        expected = self.samples[:2].copy()
                        for game in augmented:
                            expected.extend(game)
                            if len(expected) > capacity:
                                expected = expected[-capacity:]
                        actual = az.ReplayBuffer(capacity)
                        actual.data = self.samples[:2].copy()
                        self.assertEqual(az.add_saved_samples(actual, samples, meta, mirror, flip),
                                         sum(len(game) for game in augmented))
                        self.assertEqual(actual.data, expected)
                        seed_state = random.Random(20261011).getstate()
                        random.setstate(seed_state)
                        expected_draw = random.sample(expected, min(7, len(expected)))
                        random.setstate(seed_state)
                        self.assertEqual(actual.sample(min(7, len(expected))), expected_draw)

    def test_large_replay_batch_keeps_fifo_across_flush_boundary(self):
        games = [[replace(self.samples[index % 4], value=float(index))] * 1003
                 for index in range(70)]
        samples = [sample for game in games for sample in game]
        expected = self.samples[:2].copy()
        for game in games:
            expected.extend(game)
            expected = expected[-65537:]
        actual = az.ReplayBuffer(65537)
        actual.data = self.samples[:2].copy()
        self.assertEqual(az.add_saved_samples(actual, samples,
                                             dict(game_lengths=[len(game) for game in games]), False, False),
                         len(samples))
        self.assertEqual(actual.data, expected)

    def test_replay_prefix_can_cut_inside_each_augmentation_block(self):
        meta = dict(game_lengths=[2, 2])
        for mirror, flip in ((False, False), (True, False), (False, True), (True, True)):
            expected = [sample for game in (self.samples[:2], self.samples[2:])
                        for sample in az.augment_samples(game, mirror, flip)]
            for skipped in range(len(expected) + 1):
                with self.subTest(mirror=mirror, flip=flip, skipped=skipped):
                    actual = az.ReplayBuffer(100)
                    count = az.add_saved_samples(actual, self.samples, meta, mirror, flip, skipped)
                    self.assertEqual(count, len(expected))
                    self.assertEqual(actual.data, expected[skipped:])
        with self.assertRaisesRegex(ValueError, "超过文件样本数"):
            az.add_saved_samples(az.ReplayBuffer(100), self.samples, meta, False, False, 5)

    def test_restore_preserves_gc_setting_on_success_and_replay_failure(self):
        checkpoint, original, _ = self.prepare()
        originally_enabled = gc.isenabled()
        add_saved = az.add_saved_samples
        try:
            for enabled in (True, False):
                for fail in (False, True):
                    with self.subTest(enabled=enabled, fail=fail):
                        gc.enable() if enabled else gc.disable()
                        observed = []
                        def rebuild(*args, **kwargs):
                            observed.append(gc.isenabled())
                            if fail:
                                raise ValueError("injected replay failure")
                            return add_saved(*args, **kwargs)
                        with mock.patch.object(az, "add_saved_samples", side_effect=rebuild):
                            if fail:
                                with self.assertRaisesRegex(RuntimeError, "injected replay failure"):
                                    self.restore(checkpoint)
                            else:
                                resumed, _, _ = self.restore(checkpoint)
                                self.assertEqual(resumed[-1].data, original[-1].data)
                        self.assertEqual(observed, [False])
                        self.assertEqual(gc.isenabled(), enabled)
        finally:
            gc.enable() if originally_enabled else gc.disable()

    def test_restore_logs_start_file_progress_and_elapsed_time(self):
        checkpoint, _, _ = self.prepare()
        with self.assertLogs(self.logger, level="INFO") as capture:
            self.restore(checkpoint)
        messages = [record.getMessage() for record in capture.records]
        self.assertTrue(messages[0].startswith("开始恢复完整检查点:"))
        started = next(index for index, msg in enumerate(messages) if "恢复缓冲 [1/1]:" in msg)
        completed = next(index for index, msg in enumerate(messages) if "恢复缓冲 [1/1] 完成:" in msg)
        final = next(index for index, msg in enumerate(messages) if "完整检查点已恢复:" in msg)
        self.assertLess(started, completed)
        self.assertLess(completed, final)
        self.assertIn("buffer=16/23", messages[completed])
        self.assertIn("用时=", messages[final])

    def test_retry_filename_preserves_prior_selfplay_file(self):
        with mock.patch.object(az.time, "strftime", return_value="20261001_010101"):
            first = az.save_compact_iteration(self.samples, str(self.selfplay), 0)
            previous = Path(first).read_bytes()
            second = az.save_compact_iteration(self.samples[:1], str(self.selfplay), 0)
        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).read_bytes(), previous)

    def test_merge_preserves_games_and_does_not_augment_twice(self):
        az.save_compact_samples(self.samples, str(self.selfplay / "raw.pkl.gz"),
                               dict(game_lengths=[2, 2], ruleset_version=jq.RULESET_VERSION))
        merged_path = str(self.root / "merged.pkl.gz")
        self.assertEqual(az.merge_selfplay_files(str(self.selfplay), merged_path, True, True), 16)
        samples, meta = az.load_compact_payload(merged_path)
        self.assertEqual(meta["game_lengths"], [8, 8])
        expected = az.ReplayBuffer(11)
        for game in (self.samples[:2], self.samples[2:]):
            expected.add_game(az.augment_samples(game, mirror=True, flip=True))
        actual = az.ReplayBuffer(11)
        self.assertEqual(az.add_saved_samples(actual, samples, meta, True, True), 16)
        self.assertEqual(actual.data, expected.data)

    def test_preaugmented_file_can_add_missing_transform_canonically(self):
        mirrored = az.augment_samples(self.samples, mirror=True, flip=False)
        meta = dict(augment=True, augment_flip=False, game_lengths=[8])
        actual = az.ReplayBuffer(100)
        self.assertEqual(az.add_saved_samples(actual, mirrored, meta, True, True), 16)
        self.assertEqual(actual.data, az.augment_samples(self.samples, mirror=True, flip=True))

    def test_merge_cannot_relabel_old_rules(self):
        az.save_compact_samples(self.samples, str(self.selfplay / "old.pkl.gz"),
                               dict(ruleset_version=jq.RULESET_VERSION - 1))
        output = self.root / "merged.pkl.gz"
        with self.assertRaisesRegex(ValueError, "ruleset_version"):
            az.merge_selfplay_files(str(self.selfplay), str(output), ruleset_version=jq.RULESET_VERSION)
        self.assertFalse(output.exists())

    def test_merge_does_not_overwrite_existing_output(self):
        output = self.root / "existing.pkl.gz"
        output.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            az.merge_selfplay_files(str(self.selfplay), str(output))
        self.assertEqual(output.read_bytes(), b"preserve")

    def loop_config(self, folder, iterations=2, **kwargs):
        return replace(self.config, iterations=iterations, model_dir=str(folder / "models"),
                       data_dir=str(folder), log_dir=str(folder / "logs"), **kwargs)

    def run_loop(self, config, stop_after=None, **patches):
        random_class = random.Random

        def generate(net, config, device, rng, games, *args, iteration=None, **kwargs):
            if stop_after is not None and iteration >= stop_after:
                raise InterruptedError("simulated interruption")
            rng.random()
            random.random()
            np.random.rand()
            torch.rand(1)
            return [az.SelfPlayResult(self.samples[:2], jq.RESULT_RED_WIN, 2),
                    az.SelfPlayResult(self.samples[2:], jq.RESULT_RED_WIN, 2)]

        with mock.patch.object(az, "PolicyValueNet", TinyNet), \
                mock.patch.object(az.torch.cuda, "is_available", return_value=False), \
                mock.patch.object(az.random, "Random", side_effect=lambda: random_class(43)), \
                mock.patch.object(az, "generate_selfplay_results", side_effect=generate):
            from contextlib import ExitStack
            with ExitStack() as stack:
                for name, replacement in patches.items():
                    stack.enter_context(mock.patch.object(az, name, **replacement))
                az.training_loop(config)

    def test_training_loop_continuous_and_restart_are_bitwise_equivalent(self):
        continuous = self.loop_config(self.root / "continuous")
        resumed = self.loop_config(self.root / "resumed")
        self.seed()
        self.run_loop(continuous)
        self.seed()
        with self.assertRaises(InterruptedError):
            self.run_loop(resumed, stop_after=1)
        checkpoint = Path(resumed.model_dir) / az.TRAINING_CHECKPOINT_NAME
        interrupted = torch.load(checkpoint, weights_only=True)
        self.assertEqual(interrupted["train_state"]["iteration"], 1)
        # 旧兼容导出缺失/损坏也不能替代已提交的完整状态。
        (Path(resumed.model_dir) / az.MODEL_LATEST_NAME).write_bytes(b"stale export")
        (Path(resumed.model_dir) / az.STATE_FILE_NAME).write_text("{", encoding="utf-8")
        self.run_loop(resumed)
        original = torch.load(Path(continuous.model_dir) / az.TRAINING_CHECKPOINT_NAME, weights_only=True)
        restored = torch.load(checkpoint, weights_only=True)
        for key in ("model", "optimizer", "scheduler", "train_state", "rng", "buffer_size"):
            self.equal_tree(original[key], restored[key])
        self.assertEqual(restored["buffer_size"], 23)  # 跨文件 FIFO 部分截断。
        self.assertEqual(len(restored["replay_files"]), 2)

    def test_growth_commits_before_failed_round_and_restarts_without_opt_in(self):
        config = self.loop_config(self.root / "growth", iterations=5, run_iterations=2)
        self.run_loop(config)
        checkpoint = Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME
        previous = torch.load(checkpoint, weights_only=True)
        original_files = {p.name: p.read_bytes() for p in (Path(config.data_dir) / "selfplay").glob("*.pkl.gz")}
        grown_config = replace(config, replay_buffer_size=60, allow_buffer_growth=True, run_iterations=1)
        with self.assertRaisesRegex(RuntimeError, "growth round failure"):
            self.run_loop(grown_config, train_step=dict(side_effect=RuntimeError("growth round failure")))
        migrated = torch.load(checkpoint, weights_only=True)
        for key in ("model", "optimizer", "scheduler", "train_state", "rng", "buffer_size", "diagnostics"):
            self.equal_tree(migrated[key], previous[key])
        self.assertEqual(migrated["buffer_size"], 23)
        self.assertEqual(migrated["replay_files"][0]["skip_augmented"], 9)
        manifest = migrated["provenance"]["manifest"]
        self.assertEqual(manifest["parent"]["resume_mode"], "buffer_growth")
        self.assertEqual(manifest["parent"]["buffer_growth"],
                         dict(old_capacity=23, new_capacity=60, retained_samples=23))
        self.assertIsNone(migrated["provenance"]["attempt"])
        for name, contents in original_files.items():
            self.assertEqual((Path(config.data_dir) / "selfplay" / name).read_bytes(), contents)
        orphan_files = set((Path(config.data_dir) / "selfplay").glob("*.pkl.gz")) - {
            Path(config.data_dir) / "selfplay" / name for name in original_files}
        self.assertEqual(len(orphan_files), 1)
        self.run_loop(replace(grown_config, allow_buffer_growth=False))
        continued = torch.load(checkpoint, weights_only=True)
        self.assertEqual(continued["train_state"]["iteration"], 3)
        self.assertEqual(continued["buffer_size"], 39)
        self.assertEqual(continued["replay_files"][0]["skip_augmented"], 9)
        self.assertTrue(all(p.exists() for p in orphan_files))
        self.assertTrue(all(p.name not in {r["name"] for r in continued["replay_files"]} for p in orphan_files))

    def test_growth_checkpoint_write_failure_preserves_previous_commit(self):
        config = self.loop_config(self.root / "growth-write", iterations=3, run_iterations=1)
        self.run_loop(config)
        checkpoint = Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME
        previous = checkpoint.read_bytes()
        with self.assertRaisesRegex(OSError, "growth commit failure"):
            self.run_loop(replace(config, replay_buffer_size=40, allow_buffer_growth=True),
                          atomic_torch_save=dict(side_effect=OSError("growth commit failure")))
        self.assertEqual(checkpoint.read_bytes(), previous)

    def test_growth_requires_full_checkpoint_instead_of_legacy_or_fresh_start(self):
        config = self.loop_config(self.root / "no-checkpoint", allow_buffer_growth=True)
        with self.assertRaisesRegex(ValueError, "仅用于恢复已有完整检查点"):
            self.run_loop(config)
        self.assertFalse((Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME).exists())

    def test_first_round_interruption_uses_bootstrap_commit_and_ignores_orphan_data(self):
        config = self.loop_config(self.root / "first", iterations=1)
        with self.assertRaisesRegex(RuntimeError, "simulated train failure"):
            self.run_loop(config, train_step=dict(side_effect=RuntimeError("simulated train failure")))
        checkpoint = Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME
        initial = torch.load(checkpoint, weights_only=True)
        self.assertEqual(initial["train_state"]["iteration"], 0)
        self.assertEqual(initial["replay_files"], [])
        old_files = set((Path(config.data_dir) / "selfplay").glob("*.pkl.gz"))
        self.assertEqual(len(old_files), 1)
        self.run_loop(config)
        completed = torch.load(checkpoint, weights_only=True)
        self.assertEqual(completed["train_state"]["total_games"], 2)
        self.assertEqual(completed["buffer_size"], 16)
        self.assertEqual(len(completed["replay_files"]), 1)
        self.assertTrue(all(path.exists() for path in old_files))
        self.assertNotIn(next(iter(old_files)).name, [r["name"] for r in completed["replay_files"]])

    def test_missing_or_unfinished_selfplay_results_do_not_commit_round(self):
        for index, results in enumerate(([az.SelfPlayResult(self.samples, jq.RESULT_RED_WIN, 4)],
                                         [az.SelfPlayResult(self.samples[:2], None, 2),
                                          az.SelfPlayResult(self.samples[2:], jq.RESULT_RED_WIN, 2)])):
            config = self.loop_config(self.root / f"incomplete-{index}", iterations=1)
            with mock.patch.object(az, "PolicyValueNet", TinyNet), \
                    mock.patch.object(az.torch.cuda, "is_available", return_value=False), \
                    mock.patch.object(az, "generate_selfplay_results", return_value=results):
                with self.assertRaisesRegex(RuntimeError, "结果缺失或未结束"):
                    az.training_loop(config)
            checkpoint = torch.load(Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME, weights_only=True)
            self.assertEqual(checkpoint["train_state"]["iteration"], 0)
            self.assertEqual(checkpoint["train_state"]["total_games"], 0)
            self.assertEqual(checkpoint["replay_files"], [])

    def test_committed_checkpoint_repairs_latest_state_and_new_best_exports(self):
        config = self.loop_config(self.root / "exports", iterations=1, eval_games=2, eval_interval=1,
                                  legacy_ris_best=True)
        stats = dict(wins=2, draws=0, losses=0, score=1.0)
        with self.assertRaisesRegex(OSError, "export failure"):
            self.run_loop(config, evaluate_vs_ris=dict(return_value=stats),
                          save_training_state=dict(side_effect=OSError("export failure")))
        folder = Path(config.model_dir)
        for name in (az.MODEL_LATEST_NAME, az.MODEL_BEST_NAME, az.STATE_FILE_NAME):
            (folder / name).write_bytes(b"stale export")
        self.run_loop(config)
        committed = torch.load(folder / az.TRAINING_CHECKPOINT_NAME, weights_only=True)
        self.assertTrue(committed["best_updated"])
        self.assertEqual(json.loads((folder / az.STATE_FILE_NAME).read_text())["iteration"], 1)
        for name in (az.MODEL_LATEST_NAME, az.MODEL_BEST_NAME):
            self.equal_tree(committed["model"], torch.load(folder / name, weights_only=True))

    def test_ris_report_only_preserves_existing_best_after_high_score(self):
        config = self.loop_config(self.root / "report-only", iterations=2, run_iterations=1,
                                  eval_games=2, eval_interval=1, legacy_ris_best=True)
        self.run_loop(config, evaluate_vs_ris=dict(return_value=dict(wins=1, draws=0, losses=1, score=.5)))
        best = Path(config.model_dir) / az.MODEL_BEST_NAME
        original = best.read_bytes()
        self.run_loop(replace(config, legacy_ris_best=False),
                      evaluate_vs_ris=dict(return_value=dict(wins=2, draws=0, losses=0, score=1.)))
        saved = torch.load(Path(config.model_dir) / az.TRAINING_CHECKPOINT_NAME, weights_only=True)
        self.assertEqual(saved["train_state"]["iteration"], 2)
        self.assertEqual(saved["train_state"]["best_score"], .5)
        self.assertFalse(saved["best_updated"])
        self.assertEqual(saved["provenance"]["evaluation"]["selection_policy"], "report_only")
        self.assertEqual(best.read_bytes(), original)

    def test_old_pure_weights_migrate_with_warning_and_gui_format_stays_plain(self):
        config = self.loop_config(self.root / "legacy", iterations=1)
        folder = Path(config.model_dir)
        folder.mkdir(parents=True)
        net = TinyNet()
        az.save_model(net, str(folder / az.MODEL_LATEST_NAME))
        az.save_training_state(str(folder / az.STATE_FILE_NAME), az.TrainingState())
        with self.assertLogs("azero", level="WARNING") as captured:
            self.run_loop(config)
        self.assertTrue(any("旧纯权重兼容续训" in line for line in captured.output))
        weights = torch.load(folder / az.MODEL_LATEST_NAME, weights_only=True)
        self.assertIn("conv.weight", weights)
        self.assertNotIn("optimizer", weights)
        loaded = TinyNet()
        self.assertTrue(az.load_model_if_exists(loaded, str(folder / az.MODEL_LATEST_NAME), torch.device("cpu")))
        self.equal_tree(weights, loaded.state_dict())
        self.assertTrue((folder / az.TRAINING_CHECKPOINT_NAME).exists())

    def test_previous_rule_checkpoint_is_rejected_without_rewriting_it(self):
        checkpoint, _, _ = self.prepare()
        payload = torch.load(checkpoint, weights_only=True)
        payload["ruleset_version"] = jq.RULESET_VERSION - 1
        payload["train_state"]["ruleset_version"] = jq.RULESET_VERSION - 1
        torch.save(payload, checkpoint)
        previous = checkpoint.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "ruleset_version 不兼容"):
            self.restore(checkpoint)
        self.assertEqual(checkpoint.read_bytes(), previous)

    def test_previous_rule_legacy_directory_does_not_restart_or_overwrite_weights(self):
        config = self.loop_config(self.root / "previous-rules", iterations=20)
        folder = Path(config.model_dir)
        folder.mkdir(parents=True)
        model = folder / az.MODEL_LATEST_NAME
        state = folder / az.STATE_FILE_NAME
        az.save_model(TinyNet(), str(model))
        az.save_training_state(str(state), az.TrainingState(
            iteration=13, total_games=13_312, ruleset_version=jq.RULESET_VERSION - 1))
        before = (model.read_bytes(), state.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "拒绝在旧目录从头训练"):
            self.run_loop(config)
        self.assertEqual((model.read_bytes(), state.read_bytes()), before)
        self.assertFalse((folder / az.TRAINING_CHECKPOINT_NAME).exists())

    def test_previous_rule_selfplay_is_not_loaded_into_current_buffer(self):
        path = self.selfplay / "previous-rules.pkl.gz"
        az.save_compact_samples(self.samples, str(path),
                               dict(ruleset_version=jq.RULESET_VERSION - 1))
        before = path.read_bytes()
        buffer = az.ReplayBuffer(100)
        loaded = az.load_selfplay_data(str(self.selfplay), buffer, augment=True,
                                      augment_flip=True, ruleset_version=jq.RULESET_VERSION)
        self.assertEqual(loaded, 0)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(path.read_bytes(), before)

    def test_parse_explicit_reset_optimizer(self):
        self.assertFalse(az.parse_args([]).reset_optimizer)
        self.assertTrue(az.parse_args(["--reset-optimizer"]).reset_optimizer)

    def test_parse_explicit_buffer_growth(self):
        self.assertFalse(az.parse_args([]).allow_buffer_growth)
        config = az.parse_args(["--allow-buffer-growth", "--replay-buffer-size", "5000000"])
        self.assertTrue(config.allow_buffer_growth)
        self.assertEqual(config.replay_buffer_size, 5_000_000)
        for args in (["--allow-buffer-growth", "--no-resume"], ["--replay-buffer-size", "0"],
                     ["--replay-buffer-size", "-1"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                az.parse_args(args)

    def test_worker_completion_order_does_not_change_replay_order(self):
        results = [az.SelfPlayResult(self.samples, jq.RESULT_RED_WIN, i) for i in range(3)]
        futures = [concurrent.futures.Future() for _ in results]
        for future, result in zip(futures, results):
            future.set_result(result)
        with mock.patch.object(az.torch.cuda, "is_available", return_value=True), \
                mock.patch.object(az.futures, "ProcessPoolExecutor") as executor, \
                mock.patch.object(az.futures, "wait", side_effect=[({futures[2]}, {futures[0], futures[1]}),
                                                                 ({futures[0]}, {futures[1]}),
                                                                 ({futures[1]}, set())]):
            executor.return_value.submit.side_effect = futures
            config = replace(self.config, self_play_workers=3)
            actual = az.generate_selfplay_results(TinyNet(), config, torch.device("cuda"), random.Random(43), 3)
        self.assertEqual(actual, results)


if __name__ == "__main__":
    unittest.main()
