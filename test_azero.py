import gzip
import json
import logging
import math
import os
import pickle
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import azero
import ismcts as im
import jieqi as jq


class TestAzeroUtils(unittest.TestCase):
    def test_value_target_current_player_perspective(self) -> None:
        rng = random.Random(7)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info_red = belief.info_state(state, jq.RED)
        info_black = im.InformationState(
            board=info_red.board.copy(),
            side=jq.BLACK,
            pool_known=[info_red.pool_known[0].copy(), info_red.pool_known[1].copy()],
            pool_total_known=[info_red.pool_total_known[0], info_red.pool_total_known[1]],
            unknown_removals=[info_red.unknown_removals[0], info_red.unknown_removals[1]],
            no_capture_plies=info_red.no_capture_plies,
            consecutive_checks=info_red.consecutive_checks.copy(),
        )
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw_red = azero.raw_sample_from_info(info_red, [], policy)
        raw_black = azero.raw_sample_from_info(info_black, [], policy)
        red_sample, black_sample = azero.finalize_samples(
            [raw_red, raw_black],
            jq.RESULT_RED_WIN,
        )
        self.assertEqual(red_sample.side, jq.RED)
        self.assertEqual(black_sample.side, jq.BLACK)
        self.assertAlmostEqual(red_sample.value, 1.0, places=6)
        self.assertAlmostEqual(black_sample.value, -1.0, places=6)
        _, _, red_value = azero.compact_to_tensors(red_sample)
        _, _, black_value = azero.compact_to_tensors(black_sample)
        self.assertAlmostEqual(float(red_value.item()), 1.0, places=6)
        self.assertAlmostEqual(float(black_value.item()), -1.0, places=6)

    def test_puct_select_child_uses_parent_perspective(self) -> None:
        parent = azero.PUCTNode(1.0, jq.RED)
        child_bad_for_parent = azero.PUCTNode(0.5, jq.BLACK)
        child_good_for_parent = azero.PUCTNode(0.5, jq.BLACK)
        child_bad_for_parent.visits = 10
        child_bad_for_parent.value_sum = 8.0
        child_good_for_parent.visits = 10
        child_good_for_parent.value_sum = -8.0
        parent.children = {
            (0, 1): child_bad_for_parent,
            (0, 2): child_good_for_parent,
        }
        move, _ = parent.select_child(0.0)
        self.assertEqual(move, (0, 2))

    def test_mcts_sampling_uses_instance_rng(self) -> None:
        config = azero.AZConfig(mcts_simulations=2, self_play_workers=1)
        net = azero.PolicyValueNet()
        mcts = azero.NeuralMCTS(net, config, torch.device("cpu"), random.Random(0))
        state = jq.GameState.initial(rng=random.Random(1))
        belief = im.init_belief_for_game(state)
        with (
            mock.patch("azero.math.exp", return_value=0.0),
            mock.patch("azero.random.choice", side_effect=RuntimeError("global random used")),
        ):
            _, move = mcts.search(
                state,
                belief,
                history=[],
                temperature=1.0,
                simulations=2,
                add_dirichlet=False,
            )
        self.assertIsNotNone(move)

    def test_resolve_eval_device_fallback(self) -> None:
        with mock.patch("azero.torch.cuda.is_available", return_value=False):
            self.assertEqual(azero._resolve_eval_device("cuda", torch.device("cpu")), "cpu")
            self.assertEqual(azero._resolve_eval_device("cuda:0", torch.device("cpu")), "cpu")
            self.assertEqual(azero._resolve_eval_device("", torch.device("cpu")), "cpu")
            self.assertEqual(azero._resolve_eval_device(None, torch.device("cpu")), "cpu")
            self.assertEqual(azero._resolve_eval_device("cpu", torch.device("cpu")), "cpu")

    def test_mirror_index_roundtrip(self) -> None:
        for idx in range(jq.BOARD_SIZE):
            self.assertEqual(azero.mirror_index(azero.mirror_index(idx)), idx)

    def test_mirror_move_roundtrip(self) -> None:
        for from_idx in range(0, jq.BOARD_SIZE, 7):
            for to_idx in range(0, jq.BOARD_SIZE, 11):
                move = (from_idx, to_idx)
                self.assertEqual(azero.mirror_move(azero.mirror_move(move)), move)

    def test_sparse_policy_roundtrip(self) -> None:
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 0.6
        policy[10] = 0.4
        indices, probs = azero.sparse_policy_from_dense(policy)
        dense = azero.dense_policy_from_sparse(indices, probs)
        self.assertAlmostEqual(float(dense[0]), 0.6, places=6)
        self.assertAlmostEqual(float(dense[10]), 0.4, places=6)
        self.assertAlmostEqual(float(dense.sum()), 1.0, places=6)

    def test_outcome_label(self) -> None:
        self.assertEqual(azero.outcome_label(jq.RESULT_RED_WIN), "\u7ea2\u80dc")
        self.assertEqual(azero.outcome_label(jq.RESULT_BLACK_WIN), "\u9ed1\u80dc")
        self.assertEqual(azero.outcome_label(jq.RESULT_DRAW), "\u548c\u68cb")
        self.assertEqual(azero.outcome_label(None), "\u672a\u7ed3\u675f")

    def test_legal_move_table_size(self) -> None:
        table = azero.build_legal_move_table()
        self.assertEqual(len(table), 2550)
        self.assertEqual(len(azero.LEGAL_MOVE_TABLE), 2550)

    def test_temperature_for_move(self) -> None:
        self.assertEqual(azero.temperature_for_move(0, 1.0, 0.1, 0), 0.1)
        self.assertAlmostEqual(azero.temperature_for_move(0, 1.0, 0.1, 10), 1.0, places=6)
        self.assertAlmostEqual(azero.temperature_for_move(10, 1.0, 0.1, 10), 0.1, places=6)
        self.assertAlmostEqual(azero.temperature_for_move(20, 1.0, 0.1, 10), 0.1, places=6)

    def test_dirichlet_noise_noop(self) -> None:
        priors = torch.zeros(azero.MOVE_SIZE, dtype=torch.float32)
        legal = [(0, 1), (2, 3)]
        out = azero.apply_dirichlet_noise(priors, legal, alpha=0.0, epsilon=0.25)
        self.assertTrue(torch.equal(out, priors))
        out = azero.apply_dirichlet_noise(priors, legal, alpha=0.3, epsilon=0.0)
        self.assertTrue(torch.equal(out, priors))

    def test_dirichlet_noise_changes_legal(self) -> None:
        torch.manual_seed(0)
        priors = torch.zeros(azero.MOVE_SIZE, dtype=torch.float32)
        legal = [(0, 1), (2, 3)]
        for move in legal:
            priors[azero.move_to_index(move)] = 0.5
        out = azero.apply_dirichlet_noise(priors, legal, alpha=0.3, epsilon=0.25)
        idx0 = azero.move_to_index(legal[0])
        idx1 = azero.move_to_index(legal[1])
        self.assertFalse(torch.equal(out, priors))
        self.assertEqual(out.sum().item(), priors.sum().item())
        self.assertNotEqual(float(out[idx0]), float(priors[idx0]))
        self.assertNotEqual(float(out[idx1]), float(priors[idx1]))

    def test_dirichlet_noise_uses_instance_rng(self) -> None:
        priors = torch.zeros(azero.MOVE_SIZE, dtype=torch.float32)
        legal = [(0, 1), (2, 3), (4, 5)]
        for move in legal:
            priors[azero.move_to_index(move)] = 1.0 / 3.0
        out_a = azero.apply_dirichlet_noise(
            priors, legal, alpha=0.3, epsilon=0.25, rng=random.Random(42)
        )
        out_b = azero.apply_dirichlet_noise(
            priors, legal, alpha=0.3, epsilon=0.25, rng=random.Random(42)
        )
        out_c = azero.apply_dirichlet_noise(
            priors, legal, alpha=0.3, epsilon=0.25, rng=random.Random(43)
        )
        self.assertTrue(torch.equal(out_a, out_b))
        self.assertFalse(torch.equal(out_a, out_c))

    def test_root_noise_uses_configured_fraction_after_legal_normalization(self) -> None:
        legal = [(0, 1), (2, 3)]
        indices = [azero.move_to_index(move) for move in legal]
        baseline = torch.zeros(azero.MOVE_SIZE)
        baseline[indices] = torch.tensor([0.8, 0.2])
        expected_noise = azero.apply_dirichlet_noise(
            baseline, legal, alpha=0.3, epsilon=1.0, rng=random.Random(7)
        )[indices]
        for legal_mass in (1.0, 0.5, 44 / 2550):
            for epsilon in (0.0, 0.25, 1.0):
                with self.subTest(legal_mass=legal_mass, epsilon=epsilon):
                    prior = baseline * legal_mass
                    prior[10] = 1 - legal_mass
                    before = prior.clone()
                    mixed = azero.apply_dirichlet_noise(
                        prior, legal, alpha=0.3, epsilon=epsilon, rng=random.Random(7)
                    )
                    root = azero.PUCTNode(1.0, jq.RED)
                    root.expand(legal, mixed)
                    actual = torch.tensor([root.children[move].prior for move in legal])
                    expected = (1 - epsilon) * baseline[indices] + epsilon * expected_noise
                    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
                    self.assertTrue(torch.equal(prior, before))

    def test_root_noise_zero_legal_mass_uses_uniform_fallback(self) -> None:
        legal = [(0, 1), (2, 3)]
        indices = [azero.move_to_index(move) for move in legal]
        prior = torch.zeros(azero.MOVE_SIZE)
        prior[10] = 1.0
        normalized = torch.zeros_like(prior)
        normalized[indices] = 0.5
        noise = azero.apply_dirichlet_noise(normalized, legal, .3, 1., random.Random(7))[indices]
        mixed = azero.apply_dirichlet_noise(prior, legal, .3, .25, random.Random(7))
        root = azero.PUCTNode(1.0, jq.RED)
        root.expand(legal, mixed)
        actual = torch.tensor([root.children[move].prior for move in legal])
        torch.testing.assert_close(actual, .75 * normalized[indices] + .25 * noise, rtol=0, atol=1e-7)

    def test_strong_flags_mismatch(self) -> None:
        config = azero.AZConfig(self_play_games=2, self_play_workers=1)
        net = azero.PolicyValueNet()
        device = torch.device("cpu")
        rng = random.Random(0)
        with self.assertRaises(ValueError):
            azero.generate_selfplay_results(
                net,
                config,
                device,
                rng,
                games=2,
                strong_flags=[True],
            )

    def test_stop_child_processes_terminates_workers(self) -> None:
        """中断清理：spawn 子进程被 terminate，避免 Ctrl+C 后残留 worker。"""
        import multiprocessing as mp
        import time as _time

        procs = [mp.Process(target=_time.sleep, args=(30,)) for _ in range(2)]
        for proc in procs:
            proc.start()
        self.addCleanup(lambda: [p.terminate() for p in procs if p.is_alive()])
        for proc in procs:
            proc.join(timeout=10)
            self.assertTrue(proc.is_alive())
        killed = azero.stop_child_processes()
        self.assertGreaterEqual(killed, 2)
        for proc in procs:
            proc.join(timeout=10)
            self.assertFalse(proc.is_alive())

    def test_sigint_handler_terminates_children_and_hard_exits(self) -> None:
        """Ctrl+C 处理器：先终止子进程再 os._exit(130)，不进 executor 关闭路径。"""
        with (
            mock.patch.object(azero, "stop_child_processes", return_value=3) as mock_stop,
            mock.patch.object(azero.os, "_exit") as mock_exit,
            mock.patch.object(azero.sys, "stderr") as mock_stderr,
        ):
            azero._sigint_handler(2, None)
        mock_stop.assert_called_once()
        mock_exit.assert_called_once_with(130)
        self.assertIn("已终止 3 个残留子进程", mock_stderr.write.call_args[0][0])

    def test_load_selfplay_with_augment(self) -> None:
        rng = random.Random(2)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        samples = azero.finalize_samples([raw], jq.RESULT_DRAW)
        with tempfile.TemporaryDirectory() as tmp:
            azero.save_compact_game(samples, tmp)
            buffer = azero.ReplayBuffer(100)
            loaded = azero.load_selfplay_data(tmp, buffer, max_files=0, augment=True)
            self.assertEqual(loaded, 2)
            self.assertEqual(len(buffer), 2)

    def test_load_selfplay_with_augment_flip(self) -> None:
        """augment_flip 加载：每样本变为原样 + 翻面（行翻面并红黑互换）两份。"""
        rng = random.Random(6)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        samples = azero.finalize_samples([raw], jq.RESULT_DRAW)
        with tempfile.TemporaryDirectory() as tmp:
            azero.save_compact_game(samples, tmp)
            buffer = azero.ReplayBuffer(100)
            loaded = azero.load_selfplay_data(tmp, buffer, max_files=0, augment_flip=True)
            self.assertEqual(loaded, 2)
            self.assertEqual(len(buffer), 2)
            self.assertEqual(buffer.data[0].board, samples[0].board)
            self.assertEqual(buffer.data[1].board, azero.flip_vertical_board(samples[0].board))
            self.assertEqual(buffer.data[1].side, 1 - samples[0].side)
            buffer4 = azero.ReplayBuffer(100)
            loaded4 = azero.load_selfplay_data(
                tmp, buffer4, max_files=0, augment=True, augment_flip=True
            )
            self.assertEqual(loaded4, 4)
            self.assertEqual(len(buffer4), 4)

    def test_save_compact_iteration(self) -> None:
        rng = random.Random(4)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        samples = azero.finalize_samples([raw], jq.RESULT_DRAW)
        with tempfile.TemporaryDirectory() as tmp:
            path = azero.save_compact_iteration(samples, tmp, iteration=3, meta={"games": 1})
            self.assertTrue(os.path.exists(path))
            loaded = azero.load_compact_game(path)
            self.assertEqual(len(loaded), 1)

    def test_merge_selfplay_files(self) -> None:
        rng = random.Random(5)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[1] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        samples = azero.finalize_samples([raw], jq.RESULT_DRAW)
        with tempfile.TemporaryDirectory() as tmp:
            azero.save_compact_game(samples, tmp)
            azero.save_compact_game(samples, tmp)
            output_path = os.path.join(tmp, "merged.pkl.gz")
            count = azero.merge_selfplay_files(tmp, output_path, augment=False)
            self.assertEqual(count, 2)
            merged = azero.load_compact_game(output_path)
            self.assertEqual(len(merged), 2)

    def test_merge_selfplay_files_carries_ruleset_version(self) -> None:
        rng = random.Random(8)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[2] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        samples = azero.finalize_samples([raw], jq.RESULT_DRAW)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.pkl.gz", "b.pkl.gz"):
                azero.save_compact_samples(
                    samples,
                    os.path.join(tmp, name),
                    meta={"ruleset_version": jq.RULESET_VERSION},
                )
            output_path = os.path.join(tmp, "merged.pkl.gz")
            count = azero.merge_selfplay_files(tmp, output_path, augment=False)
            self.assertEqual(count, 2)
            _, meta = azero.load_compact_payload(output_path)
            self.assertEqual(meta.get("ruleset_version"), jq.RULESET_VERSION)

            buffer = azero.ReplayBuffer(100)
            loaded = azero.load_selfplay_data(tmp, buffer, ruleset_version=jq.RULESET_VERSION)
            self.assertEqual(loaded, 4)

    def test_compact_roundtrip_and_save(self) -> None:
        rng = random.Random(0)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        samples = azero.finalize_samples([raw], jq.RESULT_DRAW)
        state_tensor, policy_tensor, value_tensor = azero.compact_to_tensors(samples[0])
        self.assertEqual(state_tensor.shape, (azero.STATE_CHANNELS, jq.BOARD_H, jq.BOARD_W))
        self.assertEqual(policy_tensor.shape, (azero.MOVE_SIZE,))
        self.assertAlmostEqual(float(value_tensor.item()), 0.0, places=6)
        with tempfile.TemporaryDirectory() as tmp:
            path = azero.save_compact_game(samples, tmp)
            loaded = azero.load_compact_game(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].policy_idx, samples[0].policy_idx)
            self.assertEqual(loaded[0].history, samples[0].history)

    def test_load_compact_payload_from_main_module(self) -> None:
        rng = random.Random(0)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        samples = azero.finalize_samples([raw], jq.RESULT_DRAW)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.pkl.gz")
            main_module = sys.modules["__main__"]
            original_module = azero.CompactSample.__module__
            had_attr = hasattr(main_module, "CompactSample")
            try:
                azero.CompactSample.__module__ = "__main__"
                setattr(main_module, "CompactSample", azero.CompactSample)
                azero.save_compact_samples(samples, path, meta={"ruleset_version": jq.RULESET_VERSION})
            finally:
                azero.CompactSample.__module__ = original_module
                if not had_attr:
                    delattr(main_module, "CompactSample")
            loaded, meta = azero.load_compact_payload(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].policy_idx, samples[0].policy_idx)
            self.assertEqual(meta["ruleset_version"], jq.RULESET_VERSION)

    def test_mirror_sample_roundtrip(self) -> None:
        rng = random.Random(1)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[5] = 1.0
        raw = azero.raw_sample_from_info(info, [(0, 1), (2, 3)], policy)
        sample = azero.finalize_samples([raw], jq.RESULT_RED_WIN)[0]
        mirrored = azero.mirror_sample(sample)
        roundtrip = azero.mirror_sample(mirrored)
        self.assertEqual(roundtrip.board, sample.board)
        self.assertEqual(roundtrip.history, sample.history)
        self.assertEqual(roundtrip.policy_idx, sample.policy_idx)

    def test_encode_state_rule_channels(self) -> None:
        """保持旧 R2 前缀：64~70 逐通道对应公开规则状态。"""
        rng = random.Random(11)
        state = jq.GameState.initial(rng=rng)
        info = im.init_belief_for_game(state).info_state(state, state.side)
        target = 9 * 9 + 4
        anchor = 9 * 2 + 3
        info.chase_streak = [0, 0]
        info.chase_streak[jq.RED] = 5
        info.chase_target = [-1, -1]
        info.chase_target[jq.RED] = target
        info.chase_target_code = [0, 0]
        info.chase_target_code[jq.RED] = jq.TYPE_ROOK
        info.chase_anchor = [-1, -1]
        info.chase_anchor[jq.RED] = anchor
        info.position_counts = {(info.side, tuple(info.board)): 2}
        planes = azero.encode_state_numpy(info, [])
        self.assertEqual(planes.shape, (azero.STATE_CHANNELS, jq.BOARD_H, jq.BOARD_W))
        self.assertEqual(azero.STATE_CHANNELS, 238)
        self.assertAlmostEqual(float(planes[64].mean()), 2 / jq.REPETITION_LIMIT, places=6)
        self.assertAlmostEqual(float(planes[65].mean()), 5 / jq.LONG_CHASE_ROUNDS, places=6)
        self.assertAlmostEqual(float(planes[66].mean()), 0.0, places=6)
        self.assertEqual(float(planes[67].sum()), 1.0)
        self.assertEqual(float(planes[67, jq.ROWS[target], jq.COLS[target]]), 1.0)
        self.assertEqual(float(planes[68].sum()), 0.0)
        self.assertEqual(float(planes[69].sum()), 1.0)
        self.assertEqual(float(planes[69, jq.ROWS[anchor], jq.COLS[anchor]]), 1.0)
        self.assertEqual(float(planes[70].sum()), 0.0)

    def test_encode_state_distinguishes_chase_and_repetition(self) -> None:
        """R2：仅 chase_streak/rep_count 不同也必须编码不同（修复非 Markov 输入）。"""
        rng = random.Random(12)
        state = jq.GameState.initial(rng=rng)
        info = im.init_belief_for_game(state).info_state(state, state.side)
        base = azero.encode_state_numpy(info, [])
        chasing = im.InformationState(
            board=info.board.copy(),
            side=info.side,
            pool_known=[info.pool_known[0].copy(), info.pool_known[1].copy()],
            pool_total_known=list(info.pool_total_known),
            unknown_removals=list(info.unknown_removals),
            no_capture_plies=info.no_capture_plies,
            consecutive_checks=info.consecutive_checks.copy(),
            chase_streak=[5, 0],
        )
        repeated = im.InformationState(
            board=info.board.copy(),
            side=info.side,
            pool_known=[info.pool_known[0].copy(), info.pool_known[1].copy()],
            pool_total_known=list(info.pool_total_known),
            unknown_removals=list(info.unknown_removals),
            no_capture_plies=info.no_capture_plies,
            consecutive_checks=info.consecutive_checks.copy(),
            position_counts={(info.side, tuple(info.board)): 2},
        )
        self.assertFalse(np.array_equal(base, azero.encode_state_numpy(chasing, [])))
        self.assertFalse(np.array_equal(base, azero.encode_state_numpy(repeated, [])))

    def test_compact_to_tensors_restores_rule_state(self) -> None:
        """样本规则字段经 compact_to_tensors 无损还原到当前编码。"""
        rng = random.Random(13)
        state = jq.GameState.initial(rng=rng)
        info = im.init_belief_for_game(state).info_state(state, state.side)
        info.chase_streak = [3, 1]
        info.chase_target = [40, -1]
        info.chase_target_code = [jq.TYPE_CANNON, 0]
        info.chase_anchor = [30, -1]
        info.position_counts = {(info.side, tuple(info.board)): 2}
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        sample = azero.finalize_samples([raw], jq.RESULT_DRAW)[0]
        tensor, policy_tensor, value_tensor = azero.compact_to_tensors(sample)
        np.testing.assert_array_equal(tensor.numpy(), azero.encode_state_numpy(info, []))
        self.assertEqual(policy_tensor.shape, (azero.MOVE_SIZE,))
        self.assertAlmostEqual(float(value_tensor.item()), 0.0, places=6)

    def test_mirror_sample_mirrors_rule_state(self) -> None:
        """R2：镜像翻转空间规则字段，非空间规则字段原样复制。"""
        rng = random.Random(14)
        state = jq.GameState.initial(rng=rng)
        info = im.init_belief_for_game(state).info_state(state, state.side)
        info.check_anchor = [-1, -1]
        info.check_anchor[jq.RED] = 40
        info.chase_streak = [0, 0]
        info.chase_streak[jq.BLACK] = 2
        info.chase_streak[jq.RED] = 4
        info.chase_target = [-1, -1]
        info.chase_target[jq.RED] = 50
        info.chase_target[jq.BLACK] = 51
        info.chase_target_code = [0, 0]
        info.chase_target_code[jq.RED] = jq.TYPE_ROOK
        info.chase_target_code[jq.BLACK] = jq.TYPE_KNIGHT
        info.chase_anchor = [-1, -1]
        info.chase_anchor[jq.RED] = 60
        info.position_counts = {(info.side, tuple(info.board)): 2}
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        sample = azero.finalize_samples([azero.raw_sample_from_info(info, [], policy)], jq.RESULT_DRAW)[0]
        mirrored = azero.mirror_sample(sample)
        self.assertEqual(mirrored.check_anchor, [azero.mirror_index(40) if side == jq.RED else -1 for side in (jq.BLACK, jq.RED)])
        self.assertEqual(mirrored.chase_streak[jq.RED], 4)
        self.assertEqual(mirrored.chase_streak[jq.BLACK], 2)
        self.assertEqual(mirrored.chase_target[jq.RED], azero.mirror_index(50))
        self.assertEqual(mirrored.chase_target[jq.BLACK], azero.mirror_index(51))
        self.assertEqual(mirrored.chase_target_code[jq.RED], jq.TYPE_ROOK)
        self.assertEqual(mirrored.chase_target_code[jq.BLACK], jq.TYPE_KNIGHT)
        self.assertEqual(mirrored.chase_anchor[jq.RED], azero.mirror_index(60))
        self.assertEqual(mirrored.chase_anchor[jq.BLACK], -1)
        self.assertEqual(mirrored.rep_count, 2)
        back = azero.mirror_sample(mirrored)
        self.assertEqual(back.check_anchor, sample.check_anchor)
        self.assertEqual(back.chase_target, sample.chase_target)
        self.assertEqual(back.chase_anchor, sample.chase_anchor)

    def test_flip_vertical_sample_transforms_all_fields(self) -> None:
        """翻面增强：行翻面+红黑互换；颜色槽交换、锚格翻面、棋子码颜色位翻转。"""
        rng = random.Random(16)
        state = jq.GameState.initial(rng=rng)
        info = im.init_belief_for_game(state).info_state(state, state.side)
        info.check_anchor = [-1, -1]
        info.check_anchor[jq.RED] = 40
        info.chase_streak = [0, 0]
        info.chase_streak[jq.BLACK] = 2
        info.chase_streak[jq.RED] = 4
        info.chase_target = [-1, -1]
        info.chase_target[jq.RED] = 50
        info.chase_target[jq.BLACK] = 51
        info.chase_target_code = [0, 0]
        info.chase_target_code[jq.RED] = jq.make_piece(jq.TYPE_ROOK, jq.BLACK, 1)
        info.chase_target_code[jq.BLACK] = jq.make_piece(jq.TYPE_KNIGHT, jq.RED, 0)
        info.chase_anchor = [-1, -1]
        info.chase_anchor[jq.RED] = 60
        info.position_counts = {(info.side, tuple(info.board)): 2}
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        sample = azero.finalize_samples(
            [azero.raw_sample_from_info(info, [], policy)], jq.RESULT_DRAW
        )[0]
        board_before = sample.board.copy()
        code_before = list(sample.chase_target_code)
        flipped = azero.flip_vertical_sample(sample)
        # 原样本不被就地修改
        self.assertEqual(sample.board, board_before)
        self.assertEqual(sample.chase_target_code, code_before)
        self.assertEqual(flipped.side, 1 - sample.side)
        self.assertEqual(flipped.board, azero.flip_vertical_board(sample.board))
        self.assertEqual(flipped.pool_known, [sample.pool_known[1], sample.pool_known[0]])
        self.assertEqual(
            flipped.pool_total_known, [sample.pool_total_known[1], sample.pool_total_known[0]]
        )
        self.assertEqual(
            flipped.unknown_removals, [sample.unknown_removals[1], sample.unknown_removals[0]]
        )
        # 每色槽位互换后锚格再套行翻面
        self.assertEqual(flipped.check_anchor[jq.BLACK], azero.flip_vertical_square(40))
        self.assertEqual(flipped.check_anchor[jq.RED], -1)
        self.assertEqual(flipped.chase_streak, [4, 2])
        self.assertEqual(flipped.chase_target[jq.BLACK], azero.flip_vertical_square(50))
        self.assertEqual(flipped.chase_target[jq.RED], azero.flip_vertical_square(51))
        # chase_target_code 是含颜色位的完整棋子码：交换槽位后颜色位翻转
        self.assertEqual(
            flipped.chase_target_code[jq.BLACK], jq.make_piece(jq.TYPE_ROOK, jq.RED, 1)
        )
        self.assertEqual(
            flipped.chase_target_code[jq.RED], jq.make_piece(jq.TYPE_KNIGHT, jq.BLACK, 0)
        )
        self.assertEqual(flipped.chase_anchor[jq.BLACK], azero.flip_vertical_square(60))
        self.assertEqual(flipped.chase_anchor[jq.RED], -1)
        self.assertEqual(flipped.rep_count, 2)
        self.assertEqual(flipped.value, sample.value)
        # 翻面两次回到原样本（对合）
        back = azero.flip_vertical_sample(flipped)
        self.assertEqual(back.board, sample.board)
        self.assertEqual(back.side, sample.side)
        self.assertEqual(back.chase_target_code, sample.chase_target_code)
        self.assertEqual(back.policy_idx, sample.policy_idx)
        # 初始阵型在“翻面+换色”下自对应
        initial_board, _ = jq.initial_board()
        self.assertEqual(azero.flip_vertical_board(initial_board), initial_board)

    def test_augment_samples_group_sizes(self) -> None:
        """两个增强开关组合：同开时恰好产出完整 4 元棋盘+颜色对称群。"""
        rng = random.Random(17)
        state = jq.GameState.initial(rng=rng)
        info = im.init_belief_for_game(state).info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        sample = azero.finalize_samples(
            [azero.raw_sample_from_info(info, [], policy)], jq.RESULT_DRAW
        )[0]
        self.assertEqual(len(azero.augment_samples([sample], mirror=True, flip=True)), 4)
        self.assertEqual(len(azero.augment_samples([sample], mirror=True, flip=False)), 2)
        self.assertEqual(len(azero.augment_samples([sample], mirror=False, flip=True)), 2)
        self.assertEqual(len(azero.augment_samples([sample], mirror=False, flip=False)), 1)
        full = azero.augment_samples([sample], mirror=True, flip=True)
        self.assertEqual(full[0], sample)
        self.assertEqual(full[1].board, azero.mirror_board(sample.board))
        self.assertEqual(full[2].board, azero.flip_vertical_board(sample.board))
        self.assertEqual(
            full[3].board,
            azero.flip_vertical_board(azero.mirror_board(sample.board)),
        )

    def test_flip_vertical_lockstep_with_engine(self) -> None:
        """固定种子下翻转样本与引擎翻转局记录逐字段一致（端到端等变性）。

        校验棋盘/走子合法性/观察结构/规则记账/样本字段：本测试种子在前 60 步
        内不触发引擎内部的索引序 tie-break 稀有近似（该近似不影响合法性）。
        """
        seed = 1000
        play_rng = random.Random(seed * 7 + 3)
        state = jq.GameState.initial(rng=random.Random(seed * 2 + 1))
        assignment = jq.materialize_dark_types(random.Random(seed * 2 + 2))
        state.dark_types = assignment
        board, _ = jq.initial_board()
        flipped_state = jq.GameState(
            board,
            jq.BLACK,
            [jq.POOL_INIT.copy(), jq.POOL_INIT.copy()],
            [jq.POOL_TOTAL, jq.POOL_TOTAL],
            im.infer_king_pos(board),
            random.Random(0),
            state.rules,
        )
        flipped_state.dark_types = {
            azero.flip_vertical_index(idx): ptype for idx, ptype in assignment.items()
        }
        belief = im.init_belief_for_game(state)
        flipped_belief = im.init_belief_for_game(flipped_state)
        history: list[tuple[int, int]] = []
        flipped_history: list[tuple[int, int]] = []
        played = 0
        fields = (
            "board",
            "side",
            "pool_known",
            "pool_total_known",
            "unknown_removals",
            "no_capture_plies",
            "consecutive_checks",
            "check_anchor",
            "chase_streak",
            "chase_target",
            "chase_target_code",
            "chase_anchor",
            "rep_count",
            "history",
            "policy_idx",
            "policy_prob",
            "value",
            "policy_weight",
        )
        for ply in range(60):
            moves = state.legal_moves()
            mapped = sorted(
                (azero.flip_vertical_index(a), azero.flip_vertical_index(b)) for a, b in moves
            )
            self.assertEqual(mapped, sorted(flipped_state.legal_moves()), f"ply={ply}")
            outcome, reason = state.game_outcome()
            flipped_outcome, flipped_reason = flipped_state.game_outcome()
            expected_outcome = None if outcome is None else -outcome
            self.assertEqual(flipped_outcome, expected_outcome, f"ply={ply}")
            self.assertEqual(flipped_reason, reason, f"ply={ply}")
            if not moves:
                break
            move = play_rng.choice(moves)
            undo = state.apply_move(move, rng=random.Random(0))
            flipped_move = azero.flip_vertical_move(move)
            flipped_undo = flipped_state.apply_move(flipped_move, rng=random.Random(0))
            belief.observe_move(undo)
            flipped_belief.observe_move(flipped_undo)
            history.append(move)
            flipped_history.append(flipped_move)
            policy = [0.0] * azero.MOVE_SIZE
            policy[azero.move_to_index(move)] = 1.0
            flipped_policy = [0.0] * azero.MOVE_SIZE
            flipped_policy[azero.move_to_index(flipped_move)] = 1.0
            info = belief.info_state(state, state.side)
            flipped_info = flipped_belief.info_state(flipped_state, flipped_state.side)
            sample = azero.finalize_samples(
                [azero.raw_sample_from_info(info, history, policy)], jq.RESULT_DRAW
            )[0]
            flipped_sample = azero.finalize_samples(
                [azero.raw_sample_from_info(flipped_info, flipped_history, flipped_policy)],
                jq.RESULT_DRAW,
            )[0]
            transformed = azero.flip_vertical_sample(sample)
            for field in fields:
                self.assertEqual(
                    getattr(transformed, field),
                    getattr(flipped_sample, field),
                    f"ply={ply} field={field}",
                )
            played += 1
        self.assertGreaterEqual(played, 10)

    def test_pad_load_old_64_channel_model(self) -> None:
        """R2：旧 64 通道检查点 pad-load 后前向不崩，旧通道权重逐元素相等。"""
        torch.manual_seed(0)
        new_net = azero.PolicyValueNet(channels=16, blocks=1)
        old_state = {key: value.clone() for key, value in new_net.state_dict().items()}
        old_state["conv.weight"] = old_state["conv.weight"][:, :64].clone()
        fresh = azero.PolicyValueNet(channels=16, blocks=1)
        padded = azero.load_state_dict_padded(fresh, old_state)
        self.assertTrue(padded)
        torch.testing.assert_close(fresh.conv.weight[:, :64], old_state["conv.weight"])
        self.assertEqual(float(fresh.conv.weight[:, 64:].abs().sum().item()), 0.0)
        fresh_state = fresh.state_dict()
        for key, value in old_state.items():
            if key == "conv.weight":
                continue
            self.assertTrue(torch.equal(fresh_state[key], value), key)
        fresh.eval()
        with torch.no_grad():
            policy, value = fresh(torch.zeros(2, azero.STATE_CHANNELS, jq.BOARD_H, jq.BOARD_W))
        self.assertEqual(policy.shape, (2, azero.MOVE_SIZE))
        self.assertEqual(value.shape, (2,))

    def test_pad_load_rejects_incompatible_shape(self) -> None:
        """R2：非首层输入通道的权重形状不兼容必须报错，不得静默加载。"""
        net = azero.PolicyValueNet(channels=16, blocks=1)
        state = {key: value.clone() for key, value in net.state_dict().items()}
        state["conv.weight"] = torch.zeros(16, azero.STATE_CHANNELS + 1, 3, 3)
        with self.assertRaises(RuntimeError):
            azero.load_state_dict_padded(azero.PolicyValueNet(channels=16, blocks=1), state)
        state = {key: value.clone() for key, value in net.state_dict().items()}
        state["policy_fc.weight"] = torch.zeros(azero.MOVE_SIZE, 10)
        with self.assertRaises(RuntimeError):
            azero.load_state_dict_padded(azero.PolicyValueNet(channels=16, blocks=1), state)

    def test_load_compact_payload_rejects_v1_data(self) -> None:
        """R2：DATA_VERSION=1 的旧自博弈文件必须拒绝（缺规则字段，不可静默降级）。"""
        rng = random.Random(15)
        state = jq.GameState.initial(rng=rng)
        info = im.init_belief_for_game(state).info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        sample = azero.finalize_samples([azero.raw_sample_from_info(info, [], policy)], jq.RESULT_DRAW)[0]
        self.assertEqual(azero.DATA_VERSION, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v1.pkl.gz")
            with gzip.open(path, "wb") as handle:
                pickle.dump({"version": 1, "samples": [sample], "meta": {}}, handle)
            with self.assertRaises(ValueError):
                azero.load_compact_payload(path)
            buffer = azero.ReplayBuffer(10)
            with self.assertLogs("v1-test", level="WARNING"):
                loaded = azero.load_selfplay_data(tmp, buffer, logging.getLogger("v1-test"))
            self.assertEqual(loaded, 0)

    def test_infer_king_pos_matches_state(self) -> None:
        rng = random.Random(3)
        state = jq.GameState.initial(rng=rng)
        inferred = im.infer_king_pos(state.board)
        self.assertEqual(inferred[jq.RED], state.king_pos[jq.RED])
        self.assertEqual(inferred[jq.BLACK], state.king_pos[jq.BLACK])

    def test_pruned_policy_target_reduces_noise_visits(self) -> None:
        root = azero.PUCTNode(1.0, jq.RED)
        noisy_move = azero.LEGAL_MOVE_TABLE[0]
        good_move = azero.LEGAL_MOVE_TABLE[1]
        noisy = azero.PUCTNode(0.9, jq.BLACK)
        noisy.visits = 100
        noisy.value_sum = 0.0
        good = azero.PUCTNode(0.1, jq.BLACK)
        good.visits = 900
        good.value_sum = -900.0
        root.children = {noisy_move: noisy, good_move: good}
        policy = azero.pruned_policy_from_root(root, c_puct=1.5)
        noisy_share = float(policy[azero.move_to_index(noisy_move)])
        good_share = float(policy[azero.move_to_index(good_move)])
        self.assertAlmostEqual(noisy_share + good_share, 1.0, places=6)
        # At most floor(sqrt(2 * .9 * 1000)) = 42 visits may be removed.
        self.assertAlmostEqual(noisy_share, 58 / 958, places=6)
        self.assertAlmostEqual(good_share, 900 / 958, places=6)

    def test_pruning_preserves_most_visited_move_and_ignores_unvisited_threshold(self):
        root = azero.PUCTNode(1.0, jq.RED)
        moves = azero.LEGAL_MOVE_TABLE[:3]
        best = azero.PUCTNode(0.001, jq.BLACK)
        best.visits, best.value_sum = 100, -50.0
        other = azero.PUCTNode(0.999, jq.BLACK)
        other.visits, other.value_sum = 1, -0.9
        unvisited = azero.PUCTNode(0.0, jq.BLACK)
        root.children = dict(zip(moves, (best, other, unvisited)))
        before = [(node.visits, node.value_sum, node.prior) for node in root.children.values()]
        policy = azero.pruned_policy_from_root(root, 1.5)
        self.assertGreaterEqual(float(policy[azero.move_to_index(moves[0])]), 100 / 101 - 1e-6)
        self.assertEqual(float(policy[azero.move_to_index(moves[2])]), 0.0)
        self.assertEqual(before, [(node.visits, node.value_sum, node.prior) for node in root.children.values()])

    def test_unfinished_selfplay_cannot_be_labelled_draw(self):
        with self.assertRaisesRegex(ValueError, "terminal"):
            azero.finalize_samples([], None)

    def test_sample_reveal_outcomes_enumerates_all(self) -> None:
        counts = [0] * 8
        counts[jq.TYPE_ROOK] = 1
        counts[jq.TYPE_CANNON] = 3
        outcomes = azero.sample_reveal_outcomes(counts, 0, random.Random(0))
        self.assertEqual(len(outcomes), 2)
        self.assertAlmostEqual(sum(prob for _, prob in outcomes), 1.0, places=9)
        probs = {p_type: prob for p_type, prob in outcomes}
        self.assertAlmostEqual(probs[jq.TYPE_ROOK], 0.25, places=9)
        self.assertAlmostEqual(probs[jq.TYPE_CANNON], 0.75, places=9)

    def test_sample_reveal_outcomes_prefills_with_original_probability(self) -> None:
        counts = [0] * 8
        counts[jq.TYPE_ROOK] = 1
        counts[jq.TYPE_PAWN] = 3
        outcomes = azero.sample_reveal_outcomes(counts, 1, random.Random(1))
        self.assertEqual(len(outcomes), 1)
        self.assertIn(outcomes[0][0], (jq.TYPE_ROOK, jq.TYPE_PAWN))
        expected = counts[outcomes[0][0]] / sum(counts)
        self.assertAlmostEqual(outcomes[0][1], expected, places=9)
        self.assertEqual(azero.sample_reveal_outcomes([0] * 8, 0, random.Random(1)), [])

    def test_chance_outcome_selection_follows_probability(self) -> None:
        """回归：机会节点必须按概率采样揭示，而非按价值挑“对行动方有利”的结果。"""
        chance = azero.PUCTNode(1.0, jq.BLACK, is_chance=True)
        high_prob = azero.PUCTNode(0.9, jq.BLACK)
        high_prob.visits = 100
        high_prob.value_sum = 100.0
        low_prob = azero.PUCTNode(0.1, jq.BLACK)
        low_prob.visits = 100
        low_prob.value_sum = -100.0
        chance.children = {0: high_prob, 1: low_prob}
        rng = random.Random(0)
        counts = {0: 0, 1: 0}
        for _ in range(4000):
            key, _ = chance.sample_outcome(rng)
            counts[key] += 1
        self.assertGreater(counts[0] / 4000, 0.85)
        self.assertLess(counts[0] / 4000, 0.95)

    def test_chance_spec_only_for_dark_moves(self) -> None:
        rng = random.Random(2)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        dark_move = None
        bright_move = None
        for move in state.legal_moves():
            if jq.is_dark(state.board[move[0]]):
                dark_move = dark_move or move
            else:
                bright_move = bright_move or move
        self.assertIsNotNone(dark_move)
        self.assertIsNotNone(bright_move)
        spec = azero.chance_spec_for_move(state, info, dark_move)
        self.assertIsNotNone(spec)
        self.assertGreater(sum(spec["moved_counts"]), 0)
        self.assertIsNone(azero.chance_spec_for_move(state, info, bright_move))

    def test_chance_spec_detects_capture_reveal(self) -> None:
        rng = random.Random(11)
        state = jq.GameState.empty(rng=rng)
        state.place_piece(9, 3, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 5, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 5, jq.TYPE_PAWN, jq.RED, 1)
        state.place_piece(4, 5, jq.TYPE_PAWN, jq.BLACK, 1)
        state.side = jq.RED
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, jq.RED)
        move = (jq.index(5, 5), jq.index(4, 5))
        self.assertIn(move, state.legal_moves())
        spec = azero.chance_spec_for_move(state, info, move)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec["capture_counts"])
        self.assertGreater(sum(spec["moved_counts"]), 0)
        self.assertGreater(sum(spec["capture_counts"]), 0)

    def test_puct_expand_creates_chance_children(self) -> None:
        rng = random.Random(3)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        legal = state.legal_moves()
        specs = {}
        for move in legal:
            spec = azero.chance_spec_for_move(state, info, move)
            if spec is not None:
                specs[move] = spec
        self.assertTrue(specs)
        node = azero.PUCTNode(1.0, state.side)
        node.expand(legal, torch.ones(azero.MOVE_SIZE), specs, 0, rng)
        chance_children = [child for child in node.children.values() if child.is_chance]
        self.assertEqual(len(chance_children), len(specs))
        leaves = []
        for child in chance_children:
            self.assertTrue(child.children)
            stack = list(child.children.values())
            while stack:
                current = stack.pop()
                if current.is_chance:
                    self.assertAlmostEqual(
                        sum(outcome.prior for outcome in current.children.values()), 1.0, places=6
                    )
                    stack.extend(current.children.values())
                else:
                    leaves.append(current)
        self.assertTrue(leaves)
        for outcome in leaves:
            self.assertGreater(outcome.reveal_moved, 0)

    def test_expand_double_reveal_creates_two_layers(self) -> None:
        """双揭示（暗子吃暗子）必须拆成“翻移动子→翻被吃子”两层机会节点。"""
        rng = random.Random(12)
        state = jq.GameState.empty(rng=rng)
        state.place_piece(9, 3, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 5, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 5, jq.TYPE_PAWN, jq.RED, 1)
        state.place_piece(4, 5, jq.TYPE_PAWN, jq.BLACK, 1)
        state.side = jq.RED
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, jq.RED)
        move = (jq.index(5, 5), jq.index(4, 5))
        spec = azero.chance_spec_for_move(state, info, move)
        self.assertIsNotNone(spec)
        node = azero.PUCTNode(1.0, jq.RED)
        node.expand([move], torch.ones(azero.MOVE_SIZE), {move: spec}, 0, rng)
        chance = node.children[move]
        self.assertTrue(chance.is_chance)
        self.assertTrue(chance.children)
        for layer1 in chance.children.values():
            self.assertTrue(layer1.is_chance)
            self.assertGreater(layer1.reveal_moved, 0)
            self.assertTrue(layer1.children)
            for outcome in layer1.children.values():
                self.assertFalse(outcome.is_chance)
                self.assertEqual(outcome.reveal_moved, layer1.reveal_moved)
                self.assertGreater(outcome.reveal_captured, 0)

    def test_backup_chance_edge_keeps_perspective(self) -> None:
        root = azero.PUCTNode(1.0, jq.RED)
        chance = azero.PUCTNode(1.0, jq.BLACK, is_chance=True)
        outcome = azero.PUCTNode(0.5, jq.BLACK)
        root.children = {(0, 1): chance}
        chance.children = {0: outcome}
        azero.NeuralMCTS._backup([root, chance, outcome], 0.7, 0.0)
        self.assertAlmostEqual(outcome.value_sum, 0.7, places=9)
        self.assertAlmostEqual(chance.value_sum, 0.7, places=9)
        self.assertAlmostEqual(root.value_sum, -0.7, places=9)

    def test_mcts_with_chance_nodes_runs(self) -> None:
        config = azero.AZConfig(mcts_simulations=8, self_play_workers=1, chance_nodes=True, chance_k=0)
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), config, torch.device("cpu"), random.Random(4))
        state = jq.GameState.initial(rng=random.Random(5))
        belief = im.init_belief_for_game(state)
        policy, move = mcts.search(
            state,
            belief,
            history=[],
            temperature=0.0,
            simulations=8,
            add_dirichlet=False,
        )
        self.assertIsNotNone(move)

    def test_search_chance_nav_miss_does_not_crash(self) -> None:
        """回归（主实验崩溃路径）：机会节点导航 miss 时不得把揭示子节点键当着法。

        机会节点由先前的叶展开按当时信念建立，不同模拟的世界池演化不同，
        树分支与世界池支持集可能不相交（missing 扣除/揭示消耗）。修复前 miss
        分支继续 while 导航，对机会节点 select_child 取出 int 键（揭示类型），
        下一轮被当作普通着法传给 apply_move，抛 TypeError。此处直接让
        _navigate_chance 恒返回 (None, [])，确定性覆盖该路径。
        """
        config = azero.AZConfig(mcts_simulations=16, self_play_workers=1, chance_nodes=True, chance_k=0)
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), config, torch.device("cpu"), random.Random(6))
        state = jq.GameState.initial(rng=random.Random(5))
        belief = im.init_belief_for_game(state)
        with mock.patch.object(azero.NeuralMCTS, "_navigate_chance", return_value=(None, [])):
            policy, move = mcts.search(
                state,
                belief,
                history=[],
                temperature=0.0,
                simulations=16,
                add_dirichlet=False,
            )
        self.assertIsNotNone(move)
        self.assertGreater(mcts.chance_nav_miss, 0)
        self.assertEqual(mcts.chance_applied, 0)

    def test_chance_node_virtual_loss_refunded_in_search(self) -> None:
        """回归（Issue 0004-A）：机会节点必须进入回传路径，q 不得恒等于虚拟损失。"""
        config = azero.AZConfig(
            mcts_simulations=32,
            leaf_batch_size=8,
            chance_nodes=True,
            chance_k=0,
            self_play_workers=1,
        )
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), config, torch.device("cpu"), random.Random(7))
        state = jq.GameState.initial(rng=random.Random(8))
        belief = im.init_belief_for_game(state)
        _policy, _move, root = mcts.search(
            state,
            belief,
            history=[],
            temperature=1.0,
            simulations=32,
            add_dirichlet=False,
            return_root=True,
        )
        self.assertEqual(root.visits, 32)
        visited_chance = []
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                if child.is_chance and child.visits > 0:
                    visited_chance.append(child)
                stack.append(child)
        self.assertTrue(visited_chance)
        for node in visited_chance:
            q = node.value_sum / node.visits
            self.assertLessEqual(abs(q), 1.0 + 1e-9, "机会节点 q 超出真实价值范围（VL 未冲销）")

    def _make_chance_state(self, rng):
        """构造受控小局面：红暗卒在(5,5)、黑暗卒在(4,5)，池由调用方覆盖。"""
        state = jq.GameState.empty(rng=rng)
        state.place_piece(9, 3, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 5, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 5, jq.TYPE_PAWN, jq.RED, 1)
        state.place_piece(4, 5, jq.TYPE_PAWN, jq.BLACK, 1)
        state.side = jq.RED
        return state

    @staticmethod
    def _set_pool(state, side, counts: dict) -> None:
        for t in jq.POOL_TYPES:
            state.pool[side][t] = 0
        for t, n in counts.items():
            state.pool[side][t] = n
        state.pool_total[side] = sum(counts.values())

    def _make_single_reveal_chance(self, move):
        """单揭示机会节点：两个分支（车/卒），reval 类型写在结果节点上。"""
        chance = azero.PUCTNode(1.0, jq.RED, move=move, is_chance=True)
        for index, p_type in enumerate((jq.TYPE_ROOK, jq.TYPE_PAWN)):
            outcome = azero.PUCTNode(0.5, jq.BLACK)
            outcome.reveal_moved = p_type
            chance.children[index] = outcome
        return chance

    def test_navigate_chance_follows_world_pool(self) -> None:
        """回归（Issue 0005）：机会导航必须服从当前模拟世界的池，而非边际先验。"""
        rng = random.Random(31)
        state = self._make_chance_state(rng)
        self._set_pool(state, jq.RED, {jq.TYPE_ROOK: 2})
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), azero.AZConfig(), torch.device("cpu"), rng)
        move = (jq.index(5, 5), jq.index(4, 4))
        chance = self._make_single_reveal_chance(move)
        for _ in range(20):
            outcome, visited = mcts._navigate_chance(chance, state, rng)
            self.assertIsNotNone(outcome)
            self.assertEqual(outcome.reveal_moved, jq.TYPE_ROOK)
            self.assertEqual(visited, [outcome])

    def test_navigate_chance_weighted_by_world_counts(self) -> None:
        """世界池计数决定分支频率（车:卒=3:1 → 采样约 75%）。"""
        rng = random.Random(32)
        state = self._make_chance_state(rng)
        self._set_pool(state, jq.RED, {jq.TYPE_ROOK: 3, jq.TYPE_PAWN: 1})
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), azero.AZConfig(), torch.device("cpu"), rng)
        move = (jq.index(5, 5), jq.index(4, 4))
        chance = self._make_single_reveal_chance(move)
        rook_hits = 0
        trials = 4000
        for _ in range(trials):
            outcome, _ = mcts._navigate_chance(chance, state, rng)
            if outcome.reveal_moved == jq.TYPE_ROOK:
                rook_hits += 1
        ratio = rook_hits / trials
        self.assertGreater(ratio, 0.70)
        self.assertLess(ratio, 0.80)

    def test_navigate_chance_double_reveal_two_layers(self) -> None:
        """双揭示：第一层用移动方世界池、第二层用被吃方世界池，visited 含中间层。"""
        rng = random.Random(33)
        state = self._make_chance_state(rng)
        self._set_pool(state, jq.RED, {jq.TYPE_ROOK: 2})
        self._set_pool(state, jq.BLACK, {jq.TYPE_PAWN: 1})
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), azero.AZConfig(), torch.device("cpu"), rng)
        move = (jq.index(5, 5), jq.index(4, 5))
        chance = azero.PUCTNode(1.0, jq.RED, move=move, is_chance=True)
        for index, moved_type in enumerate((jq.TYPE_ROOK, jq.TYPE_PAWN)):
            layer2 = azero.PUCTNode(0.5, jq.BLACK, is_chance=True)
            layer2.reveal_moved = moved_type
            outcome = azero.PUCTNode(0.5, jq.BLACK)
            outcome.reveal_moved = moved_type
            outcome.reveal_captured = jq.TYPE_PAWN
            layer2.children[0] = outcome
            chance.children[index] = layer2
        outcome, visited = mcts._navigate_chance(chance, state, rng)
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.reveal_moved, jq.TYPE_ROOK)
        self.assertEqual(outcome.reveal_captured, jq.TYPE_PAWN)
        self.assertEqual(len(visited), 2)
        self.assertIs(visited[1], outcome)

    def test_search_chance_never_falls_back_with_full_enum(self) -> None:
        """回归（Issue 0005）：未知被吃>0 且全枚举时，搜索中的揭示类型必须始终可用。"""
        config = azero.AZConfig(
            mcts_simulations=64,
            leaf_batch_size=8,
            chance_nodes=True,
            chance_k=0,
            self_play_workers=1,
        )
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), config, torch.device("cpu"), random.Random(41))
        state = jq.GameState.initial(rng=random.Random(42))
        belief = im.init_belief_for_game(state)
        # 人为制造未知被吃：观察者只知道暗池少了 6 个，但不知道类型。
        for observer in (jq.BLACK, jq.RED):
            belief.pool_total[observer][state.side] -= 6
            belief.unknown_removals[observer][state.side] += 6
        fallbacks = {"count": 0}
        original = azero.NeuralMCTS._apply_chance_move

        def hooked(self, sim_state, move, reveal_moved, reveal_captured):
            if reveal_moved and sim_state.pool[sim_state.side][reveal_moved] <= 0:
                fallbacks["count"] += 1
            if reveal_captured:
                target = sim_state.board[move[1]]
                if target and jq.is_dark(target):
                    if sim_state.pool[jq.piece_color(target)][reveal_captured] <= 0:
                        fallbacks["count"] += 1
            return original(self, sim_state, move, reveal_moved, reveal_captured)

        azero.NeuralMCTS._apply_chance_move = hooked
        try:
            mcts.search(state, belief, history=[], temperature=1.0, simulations=64, add_dirichlet=False)
        finally:
            azero.NeuralMCTS._apply_chance_move = original
        self.assertGreater(mcts.chance_applied, 0)
        self.assertEqual(mcts.chance_nav_miss, 0)
        self.assertEqual(fallbacks["count"], 0)

    def test_mcts_search_return_root(self) -> None:
        config = azero.AZConfig(mcts_simulations=4, self_play_workers=1)
        net = azero.PolicyValueNet()
        mcts = azero.NeuralMCTS(net, config, torch.device("cpu"), random.Random(3))
        state = jq.GameState.initial(rng=random.Random(5))
        belief = im.init_belief_for_game(state)
        policy, move, root = mcts.search(
            state,
            belief,
            history=[],
            temperature=1.0,
            simulations=4,
            add_dirichlet=False,
            return_root=True,
        )
        self.assertEqual(len(policy), 2550)
        self.assertIsNotNone(move)
        self.assertGreater(root.visits, 0)
        self.assertLessEqual(abs(root.value_sum / root.visits), 1.0)
        self.assertAlmostEqual(sum(policy), 1.0, places=6)

    def test_apply_chance_move_falls_back_when_unavailable(self) -> None:
        rng = random.Random(6)
        state = jq.GameState.initial(rng=rng)
        dark_move = next(move for move in state.legal_moves() if jq.is_dark(state.board[move[0]]))
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), azero.AZConfig(), torch.device("cpu"), rng)
        undo = mcts._apply_chance_move(state, dark_move, jq.TYPE_KING, 0)
        self.assertNotEqual(undo.moved_reveal, jq.TYPE_KING)
        self.assertGreater(undo.moved_reveal, 0)

    def test_parse_args_chance_flags(self) -> None:
        default = azero.parse_args([])
        self.assertTrue(default.chance_nodes)
        self.assertEqual(default.chance_k, 0)
        disabled = azero.parse_args(["--no-chance-nodes"])
        self.assertFalse(disabled.chance_nodes)
        enabled = azero.parse_args(["--chance-nodes", "--chance-k", "6"])
        self.assertTrue(enabled.chance_nodes)
        self.assertEqual(enabled.chance_k, 6)
        buffer_cfg = azero.parse_args(["--replay-buffer-size", "250000"])
        self.assertEqual(buffer_cfg.replay_buffer_size, 250000)
        with self.assertRaises(SystemExit):
            azero.parse_args(["--chance-k=-1"])

    def test_compute_train_steps(self) -> None:
        self.assertEqual(azero.compute_train_steps(1000, 512, 100000), 1000)
        self.assertEqual(azero.compute_train_steps(500, 512, 1_000_000), 500)
        self.assertEqual(azero.compute_train_steps(1000, 512, 100), 0)
        self.assertEqual(azero.compute_train_steps(0, 512, 100000), 0)
        self.assertEqual(azero.compute_train_steps(1000, 0, 100000), 0)

    def test_fast_forward_scheduler(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0)
        azero.fast_forward_scheduler(scheduler, 0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1, places=12)
        azero.fast_forward_scheduler(scheduler, 3)
        expected = 0.1 * (1.0 + math.cos(math.pi * 3 / 10)) / 2.0
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected, places=12)

    def test_parse_args_lr_overrides(self) -> None:
        default = azero.parse_args([])
        self.assertAlmostEqual(default.learning_rate, 5e-4)
        self.assertEqual(default.lr_schedule, "cosine")
        overridden = azero.parse_args(["--lr", "5e-5", "--lr-schedule", "none"])
        self.assertAlmostEqual(overridden.learning_rate, 5e-5)
        self.assertEqual(overridden.lr_schedule, "none")
        with self.assertRaises(SystemExit):
            azero.parse_args(["--lr", "1e-6"])

    def test_resume_lr_soft_continuation(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=672, eta_min=1e-5)
        azero.fast_forward_scheduler(scheduler, 352)
        expected = 1e-5 + (5e-5 - 1e-5) * (1.0 + math.cos(math.pi * 352 / 672)) / 2.0
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected, places=6)

    def test_training_state_ruleset_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train_state.json")
            state = azero.TrainingState(iteration=5, best_score=0.5, total_games=64, ruleset_version=2)
            azero.save_training_state(path, state)
            loaded = azero.load_training_state(path)
            self.assertEqual(loaded.iteration, 5)
            self.assertEqual(loaded.ruleset_version, 2)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"iteration": 1, "best_score": 0.25, "total_games": 32}, handle)
            legacy = azero.load_training_state(path)
            self.assertEqual(legacy.ruleset_version, -1)

    def test_parse_args_rejects_non_positive_limits(self) -> None:
        """P3-9：--eval-interval/--mcts-simulations 必须为正（避免除零/空搜索）。"""
        with self.assertRaises(SystemExit):
            azero.parse_args(["--eval-interval", "0"])
        with self.assertRaises(SystemExit):
            azero.parse_args(["--mcts-simulations", "0"])

    def test_search_terminal_root_returns_empty_policy(self) -> None:
        """P3-9：根已终局（无访问回填）时返回空策略/None，不得除零崩溃。"""
        cfg = azero.AZConfig(mcts_simulations=8, leaf_batch_size=4, self_play_workers=1)
        state = jq.GameState.initial(rng=random.Random(1))
        state.no_capture_plies = jq.NO_CAPTURE_LIMIT_TIANTIAN_PLIES
        belief = im.init_belief_for_game(state)
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), cfg, torch.device("cpu"), random.Random(1))
        policy, move = mcts.search(
            state, belief, history=[], temperature=1.0, simulations=8, add_dirichlet=False
        )
        self.assertEqual(sum(policy), 0.0)
        self.assertIsNone(move)
        policy, move = mcts.search(
            state, belief, history=[], temperature=1.0, simulations=8,
            add_dirichlet=False, prune_policy=True,
        )
        self.assertEqual(sum(policy), 0.0)
        self.assertIsNone(move)

    def test_leaf_input_signature_covers_rule_fields(self) -> None:
        """P1-4：去重签名必须覆盖编码字段（规则状态 + history）。"""
        rng = random.Random(21)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        base = azero.leaf_input_signature(info, [])
        same = azero.leaf_input_signature(belief.info_state(state, state.side), [])
        self.assertEqual(base, same)
        altered = belief.info_state(state, state.side)
        altered.chase_streak[altered.side] = 1
        self.assertNotEqual(base, azero.leaf_input_signature(altered, []))
        altered = belief.info_state(state, state.side)
        altered.check_anchor[altered.side] = 3
        self.assertNotEqual(base, azero.leaf_input_signature(altered, []))
        altered = belief.info_state(state, state.side)
        key = (altered.side, tuple(altered.board))
        altered.position_counts = {key: 2}
        self.assertNotEqual(base, azero.leaf_input_signature(altered, []))
        self.assertNotEqual(base, azero.leaf_input_signature(info, [(0, 1)]))

    def test_parse_args_leaf_dedup_flag(self) -> None:
        self.assertTrue(azero.parse_args([]).leaf_dedup)
        self.assertFalse(azero.parse_args(["--no-leaf-dedup"]).leaf_dedup)

    def test_leaf_dedup_reduces_evals_without_changing_result(self) -> None:
        """P1-4：同批同输入只做一次前向；chance 关闭时搜索结果与去重前完全一致。"""
        shared_net = azero.PolicyValueNet()

        def run(dedup: bool):
            config = azero.AZConfig(
                mcts_simulations=64,
                leaf_batch_size=64,
                self_play_workers=1,
                leaf_dedup=dedup,
            )
            mcts = azero.NeuralMCTS(
                shared_net, config, torch.device("cpu"), random.Random(11)
            )
            evaluated: list[int] = []
            original = mcts._evaluate_batch

            def counting(infos, histories, **kwargs):
                evaluated.append(len(infos))
                return original(infos, histories, **kwargs)

            mcts._evaluate_batch = counting
            state = jq.GameState.empty(rng=random.Random(12), with_pools=False)
            state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
            state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
            state.place_piece(9, 8, jq.TYPE_ROOK, jq.RED, 0)
            state.place_piece(0, 8, jq.TYPE_ROOK, jq.BLACK, 0)
            belief = im.init_belief_for_game(state)
            policy, move, root = mcts.search(
                state,
                belief,
                history=[],
                temperature=1.0,
                simulations=64,
                add_dirichlet=False,
                return_root=True,
            )
            return mcts, policy, move, root, evaluated

        mcts_on, policy_on, move_on, root_on, evals_on = run(True)
        mcts_off, policy_off, move_off, root_off, evals_off = run(False)
        self.assertEqual(sum(evals_on), mcts_on.leaf_evals_unique)
        self.assertEqual(sum(evals_off), mcts_off.leaf_evals_unique)
        self.assertEqual(mcts_on.leaf_evals_total, mcts_off.leaf_evals_total)
        self.assertLess(mcts_on.leaf_evals_unique, mcts_off.leaf_evals_unique)
        self.assertEqual(policy_on, policy_off)
        self.assertEqual(move_on, move_off)
        self.assertEqual(root_on.visits, root_off.visits)
        # 批量前向的归约顺序不同会带来 ~1e-8 级浮点噪声，语义须一致。
        self.assertAlmostEqual(root_on.value_sum, root_off.value_sum, places=5)

    def _make_weighted_batch(self, fast_weight: float):
        rng = random.Random(31)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        full_policy = [0.0] * azero.MOVE_SIZE
        full_policy[0] = 1.0
        fast_policy = [0.0] * azero.MOVE_SIZE
        fast_policy[1] = 1.0
        raw_full = azero.raw_sample_from_info(info, [], full_policy, policy_weight=1.0)
        raw_fast = azero.raw_sample_from_info(info, [], fast_policy, policy_weight=fast_weight)
        return azero.finalize_samples([raw_full, raw_fast], jq.RESULT_DRAW)

    def test_sample_playout_cap_distribution(self) -> None:
        """P1-5：PCR 档位比例与权重（固定随机源）。"""
        config = azero.AZConfig(mcts_simulations=400, pcr_fast_sims=100, pcr_full_prob=0.25)
        rng = random.Random(7)
        draws = 4000
        full = 0
        for _ in range(draws):
            sims, weight = azero.sample_playout_cap(config, rng)
            if weight > 0.0:
                full += 1
                self.assertEqual(sims, 400)
                self.assertAlmostEqual(weight, 1.0, places=9)
            else:
                self.assertEqual(sims, 100)
                self.assertAlmostEqual(weight, 0.0, places=9)
        self.assertGreater(full / draws, 0.22)
        self.assertLess(full / draws, 0.28)

    def test_pcr_self_play_records_policy_weight(self) -> None:
        """P1-5：fast 回合样本策略目标权重为 0，full 回合为 1；两者都保留 value 目标。"""
        config = azero.AZConfig(
            mcts_simulations=4,
            pcr_enabled=True,
            pcr_fast_sims=2,
            pcr_full_prob=0.5,
            self_play_workers=1,
        )
        result = azero.self_play_game(
            azero.PolicyValueNet(), config, torch.device("cpu"), random.Random(5)
        )
        weights = [float(getattr(sample, "policy_weight", 1.0)) for sample in result.samples]
        self.assertTrue(weights)
        self.assertIn(1.0, weights)
        self.assertIn(0.0, weights)
        ratio = sum(weights) / len(weights)
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 0.7)

    def test_train_step_pcr_weights_policy_target(self) -> None:
        """P1-5：policy_weight=0 的 fast 样本不影响策略损失；value 损失仍使用全部样本。"""
        net = azero.PolicyValueNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
        device = torch.device("cpu")

        batch = self._make_weighted_batch(0.0)
        _, policy_a, value_a = azero.train_step(net, optimizer, batch, device, 0.0)

        batch_fast_changed = self._make_weighted_batch(0.0)
        batch_fast_changed[1].policy_idx = [77]
        batch_fast_changed[1].policy_prob = [1.0]
        _, policy_b, value_b = azero.train_step(net, optimizer, batch_fast_changed, device, 0.0)
        self.assertAlmostEqual(policy_a, policy_b, places=6)
        self.assertAlmostEqual(value_a, value_b, places=6)

        batch_fast_changed[1].value = 1.0
        _, policy_c, value_c = azero.train_step(net, optimizer, batch_fast_changed, device, 0.0)
        self.assertAlmostEqual(policy_c, policy_a, places=6)
        self.assertNotAlmostEqual(value_c, value_a, places=4)

        batch_full_changed = self._make_weighted_batch(1.0)
        batch_full_changed[1].policy_idx = [77]
        batch_full_changed[1].policy_prob = [1.0]
        _, policy_d, _ = azero.train_step(net, optimizer, batch_full_changed, device, 0.0)
        self.assertNotAlmostEqual(policy_d, policy_a, places=4)

    def test_train_step_handles_legacy_sample_without_policy_weight(self) -> None:
        """旧格式样本（无 policy_weight 槽）按 1.0 处理，不崩溃。"""
        net = azero.PolicyValueNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
        batch = self._make_weighted_batch(1.0)
        del batch[0].policy_weight
        loss, policy_loss, value_loss = azero.train_step(
            net, optimizer, batch, torch.device("cpu"), 0.0
        )
        self.assertTrue(all(math.isfinite(v) for v in (loss, policy_loss, value_loss)))

    def test_mirror_sample_preserves_policy_weight(self) -> None:
        batch = self._make_weighted_batch(0.0)
        sample = batch[1]
        self.assertEqual(sample.policy_weight, 0.0)
        self.assertEqual(azero.mirror_sample(sample).policy_weight, 0.0)

    def test_policy_weight_roundtrip_save_load(self) -> None:
        batch = self._make_weighted_batch(0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = azero.save_compact_game(batch, tmp)
            loaded = azero.load_compact_game(path)
            weights = [sample.policy_weight for sample in loaded]
            self.assertEqual(weights, [1.0, 0.0])

    def test_legacy_pickled_sample_without_policy_weight_loads(self) -> None:
        """模拟旧格式文件：pickle 中 CompactSample 缺少 policy_weight 槽仍可加载。"""
        sample = self._make_weighted_batch(1.0)[0]
        del sample.policy_weight
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy_weight.pkl.gz")
            payload = {"version": azero.DATA_VERSION, "samples": [sample], "meta": {}}
            with gzip.open(path, "wb") as handle:
                pickle.dump(payload, handle)
            loaded = azero.load_compact_game(path)
            self.assertEqual(float(getattr(loaded[0], "policy_weight", 1.0)), 1.0)

    def test_parse_args_pcr_flags(self) -> None:
        default = azero.parse_args([])
        self.assertFalse(default.pcr_enabled)
        self.assertEqual(default.pcr_fast_sims, 100)
        self.assertAlmostEqual(default.pcr_full_prob, 0.25)
        enabled = azero.parse_args(["--pcr", "--pcr-fast-sims", "120", "--pcr-full-prob", "0.4"])
        self.assertTrue(enabled.pcr_enabled)
        self.assertEqual(enabled.pcr_fast_sims, 120)
        self.assertAlmostEqual(enabled.pcr_full_prob, 0.4)
        with self.assertRaises(SystemExit):
            azero.parse_args(["--pcr", "--pcr-fast-sims", "800"])
        with self.assertRaises(SystemExit):
            azero.parse_args(["--pcr", "--pcr-fast-sims", "0"])
        with self.assertRaises(SystemExit):
            azero.parse_args(["--pcr", "--pcr-full-prob", "0"])

    def test_load_selfplay_skips_truncated_file(self) -> None:
        """P0-3：截断/空 .pkl.gz 应跳过并告警，不得让 resume 崩溃。"""
        rng = random.Random(2)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        samples = azero.finalize_samples([raw] * 3, jq.RESULT_DRAW)
        with tempfile.TemporaryDirectory() as tmp:
            good_path = os.path.join(tmp, "good.pkl.gz")
            azero.save_compact_samples(samples, good_path, meta={"ruleset_version": jq.RULESET_VERSION})
            with open(good_path, "rb") as handle:
                raw_bytes = handle.read()
            for name, data in (("trunc.pkl.gz", raw_bytes[: len(raw_bytes) // 2]), ("empty.pkl.gz", b"")):
                with open(os.path.join(tmp, name), "wb") as handle:
                    handle.write(data)
            buffer = azero.ReplayBuffer(100)
            with self.assertLogs("trunc-test", level="WARNING"):
                loaded = azero.load_selfplay_data(
                    tmp, buffer, logging.getLogger("trunc-test"), ruleset_version=jq.RULESET_VERSION
                )
            self.assertEqual(loaded, 3)

    def test_load_compact_payload_rejects_wrong_data_version(self) -> None:
        """P3-10：DATA_VERSION 不匹配的样本文件必须拒绝加载。"""
        rng = random.Random(3)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        sample = azero.finalize_samples([raw], jq.RESULT_DRAW)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bogus.pkl.gz")
            payload = {"version": azero.DATA_VERSION + 998, "samples": [sample], "meta": {}}
            with gzip.open(path, "wb") as handle:
                pickle.dump(payload, handle)
            with self.assertRaises(ValueError):
                azero.load_compact_payload(path)
            buffer = azero.ReplayBuffer(10)
            with self.assertLogs("version-test", level="WARNING"):
                loaded = azero.load_selfplay_data(tmp, buffer, logging.getLogger("version-test"))
            self.assertEqual(loaded, 0)

    def test_load_training_state_warns_on_corrupt(self) -> None:
        """P0-3：损坏的 train_state.json 必须显式告警，不得静默重置。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train_state.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"iteration": 560, "best_sco')
            with self.assertLogs("state-test", level="WARNING"):
                state = azero.load_training_state(path, logging.getLogger("state-test"))
            self.assertEqual(state.iteration, 0)
            self.assertEqual(state.ruleset_version, jq.RULESET_VERSION)

    def test_resume_refuses_missing_latest_model(self) -> None:
        """P0-3：state.iteration>0 但 latest.pt 缺失时，拒绝以随机权重续训。"""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "models")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(model_dir)
            os.makedirs(os.path.join(data_dir, "selfplay"))
            azero.save_training_state(
                os.path.join(model_dir, azero.STATE_FILE_NAME),
                azero.TrainingState(
                    iteration=5, best_score=0.5, total_games=320, ruleset_version=jq.RULESET_VERSION
                ),
            )
            config = azero.AZConfig(
                iterations=6,
                resume=True,
                data_dir=data_dir,
                model_dir=model_dir,
                log_dir=os.path.join(tmp, "logs"),
            )
            try:
                with mock.patch.object(torch.cuda, "is_available", return_value=False):
                    with self.assertRaises(RuntimeError):
                        azero.training_loop(config)
            finally:
                # Windows：关闭日志文件句柄，否则 TemporaryDirectory 清理失败
                logger = logging.getLogger("azero")
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)

    def test_load_selfplay_skips_old_ruleset(self) -> None:
        rng = random.Random(6)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        info = belief.info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        raw = azero.raw_sample_from_info(info, [], policy)
        samples = azero.finalize_samples([raw], jq.RESULT_DRAW)
        with tempfile.TemporaryDirectory() as tmp:
            azero.save_compact_samples(samples, os.path.join(tmp, "old.pkl.gz"), meta={"ruleset_version": 1})
            azero.save_compact_samples(
                samples,
                os.path.join(tmp, "new.pkl.gz"),
                meta={"ruleset_version": jq.RULESET_VERSION},
            )
            buffer = azero.ReplayBuffer(100)
            loaded = azero.load_selfplay_data(tmp, buffer, ruleset_version=jq.RULESET_VERSION)
            self.assertEqual(loaded, 1)
            buffer_all = azero.ReplayBuffer(100)
            loaded_all = azero.load_selfplay_data(tmp, buffer_all)
            self.assertEqual(loaded_all, 2)

    def _samples_with_value(self, value: float, count: int = 1):
        """构造 value 可区分的样本，用于验证加载顺序与保留集合。"""
        rng = random.Random(0)
        state = jq.GameState.initial(rng=rng)
        info = im.init_belief_for_game(state).info_state(state, state.side)
        policy = [0.0] * azero.MOVE_SIZE
        policy[0] = 1.0
        samples = azero.finalize_samples(
            [azero.raw_sample_from_info(info, [], policy) for _ in range(count)],
            jq.RESULT_DRAW,
        )
        for sample in samples:
            sample.value = value
        return samples

    def _save_value_file(self, tmp: str, idx: int, value: float, count: int, ruleset: int | None = None):
        path = os.path.join(tmp, f"iter_{idx}.pkl.gz")
        meta = {"ruleset_version": jq.RULESET_VERSION if ruleset is None else ruleset}
        azero.save_compact_samples(self._samples_with_value(value, count), path, meta=meta)
        stamp = 1_000_000 + idx * 100
        os.utime(path, (stamp, stamp))
        return path

    def test_load_selfplay_capacity_adaptive_keeps_newest(self) -> None:
        """默认 max_files=0：按容量从新到旧回填，FIFO 后保留最新样本。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._save_value_file(tmp, 0, 1.0, 2)
            self._save_value_file(tmp, 1, 2.0, 2)
            self._save_value_file(tmp, 2, 3.0, 2)
            # 版本不符的文件不占预算（即使它是最新文件）
            self._save_value_file(tmp, 3, 9.0, 2, ruleset=1)
            buffer = azero.ReplayBuffer(5)
            loaded = azero.load_selfplay_data(
                tmp, buffer, ruleset_version=jq.RULESET_VERSION
            )
            self.assertEqual(loaded, 5)
            self.assertEqual(len(buffer), 5)
            values = [sample.value for sample in buffer.data]
            self.assertEqual(values.count(3.0), 2)
            self.assertEqual(values.count(2.0), 2)
            self.assertEqual(values.count(1.0), 1)
            self.assertNotIn(9.0, values)

    def test_load_selfplay_capacity_adaptive_with_augment(self) -> None:
        """augment 时预算按 ×2 计；最新文件的原始+镜像都应保留。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._save_value_file(tmp, 0, 1.0, 1)
            self._save_value_file(tmp, 1, 2.0, 1)
            buffer = azero.ReplayBuffer(3)
            loaded = azero.load_selfplay_data(
                tmp, buffer, augment=True, ruleset_version=jq.RULESET_VERSION
            )
            self.assertEqual(loaded, 3)
            self.assertEqual(len(buffer), 3)
            values = [sample.value for sample in buffer.data]
            self.assertEqual(values.count(2.0), 2)
            self.assertEqual(values.count(1.0), 1)

    def test_load_selfplay_capacity_adaptive_with_both_augments(self) -> None:
        """两个增强同开时预算按 ×4 计；最新文件的 4 个副本都应保留。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._save_value_file(tmp, 0, 1.0, 1)
            self._save_value_file(tmp, 1, 2.0, 1)
            buffer = azero.ReplayBuffer(5)
            loaded = azero.load_selfplay_data(
                tmp,
                buffer,
                augment=True,
                augment_flip=True,
                ruleset_version=jq.RULESET_VERSION,
            )
            self.assertEqual(loaded, 5)
            self.assertEqual(len(buffer), 5)
            values = [sample.value for sample in buffer.data]
            self.assertEqual(values.count(2.0), 4)
            self.assertEqual(values.count(1.0), 1)

    def test_load_selfplay_max_files_legacy_semantics(self) -> None:
        """显式 max_files>0 保持旧语义：只取最近 N 个文件，且不做容量裁剪。"""
        with tempfile.TemporaryDirectory() as tmp:
            self._save_value_file(tmp, 0, 1.0, 2)
            self._save_value_file(tmp, 1, 2.0, 2)
            self._save_value_file(tmp, 2, 3.0, 2)
            buffer = azero.ReplayBuffer(100)
            loaded = azero.load_selfplay_data(tmp, buffer, max_files=1)
            self.assertEqual(loaded, 2)
            self.assertEqual({sample.value for sample in buffer.data}, {3.0})
            # 旧语义下加载量可超过容量：返回全部 add 数，buffer 自行 FIFO 截断
            buffer_small = azero.ReplayBuffer(1)
            loaded_small = azero.load_selfplay_data(tmp, buffer_small, max_files=2)
            self.assertEqual(loaded_small, 4)
            self.assertEqual(len(buffer_small), 1)


class TestSearchInformationSafety(unittest.TestCase):
    """搜索/训练信息安全性回归：评估入参不得含隐藏信息，增强不得破坏语义。"""

    def _random_game_samples(self, seed: int, plies: int = 40):
        """随机对局若干步，逐拍生成 (sample, 当时的 InformationState)。"""
        rng = random.Random(seed)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        history: list[tuple[int, int]] = []
        produced = []
        for _ in range(plies):
            if state.game_outcome()[0] is not None:
                break
            info = belief.info_state(state, state.side)
            legal = state.legal_moves()
            raw = azero.raw_sample_from_info(info, list(history), [0.0] * azero.MOVE_SIZE)
            raw.policy_idx = [azero.move_to_index(move) for move in legal]
            raw.policy_prob = [1.0 / len(legal)] * len(legal)
            sample = azero.finalize_samples([raw], jq.RESULT_DRAW)[0]
            produced.append((sample, info, legal))
            move = rng.choice(legal)
            im.apply_move_and_update_belief(state, belief, move, rng=rng)
            history.append(move)
        return produced

    def test_belief_is_observer_upper_bound_of_true_pool(self) -> None:
        """belief 只为观察者保留可见信息：真值 <= 上限，且未知被吃 = 上限-总数。"""
        rng = random.Random(21)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        for _ in range(300):
            if state.game_outcome()[0] is not None:
                break
            move = rng.choice(state.legal_moves())
            im.apply_move_and_update_belief(state, belief, move, rng=rng)
            for observer in (jq.RED, jq.BLACK):
                for side in (jq.RED, jq.BLACK):
                    known = belief.pool_counts[observer][side]
                    total = belief.pool_total[observer][side]
                    unknown = belief.unknown_removals[observer][side]
                    for p_type in jq.POOL_TYPES:
                        self.assertGreaterEqual(
                            known[p_type],
                            state.pool[side][p_type],
                            f"观察者 {observer} 对 {side} 的 {p_type} 计数低于真值",
                        )
                    self.assertEqual(total, state.pool_total[side])
                    self.assertEqual(sum(known) - total, unknown)

    def test_info_state_never_exposes_true_pool(self) -> None:
        """搜索评估用的 InformationState 只能携带该观察者已知的信息。"""
        rng = random.Random(22)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        for _ in range(200):
            if state.game_outcome()[0] is not None:
                break
            for observer in (jq.RED, jq.BLACK):
                info = belief.info_state(state, observer)
                for side in (jq.RED, jq.BLACK):
                    for p_type in jq.POOL_TYPES:
                        self.assertGreaterEqual(info.pool_known[side][p_type], state.pool[side][p_type])
                    self.assertEqual(info.pool_total_known[side], state.pool_total[side])
            move = rng.choice(state.legal_moves())
            im.apply_move_and_update_belief(state, belief, move, rng=rng)

    def test_mirror_sample_preserves_encoding_and_legality(self) -> None:
        """镜像增强：编码左右镜像、动作仍在镜像局面合法集内、双重镜像还原。"""
        for seed in (31, 32, 33):
            for sample, info, legal in self._random_game_samples(seed):
                mirrored = azero.mirror_sample(sample)
                # 编码等价
                encoded = azero.encode_state_numpy(info, sample.history)
                mirrored_info = im.InformationState(
                    board=mirrored.board,
                    side=mirrored.side,
                    pool_known=[mirrored.pool_known[0].copy(), mirrored.pool_known[1].copy()],
                    pool_total_known=[
                        mirrored.pool_total_known[0],
                        mirrored.pool_total_known[1],
                    ],
                    unknown_removals=[
                        mirrored.unknown_removals[0],
                        mirrored.unknown_removals[1],
                    ],
                    no_capture_plies=mirrored.no_capture_plies,
                    consecutive_checks=mirrored.consecutive_checks.copy(),
                    check_anchor=mirrored.check_anchor.copy(),
                    chase_streak=mirrored.chase_streak.copy(),
                    chase_target=mirrored.chase_target.copy(),
                    chase_target_code=mirrored.chase_target_code.copy(),
                    chase_anchor=mirrored.chase_anchor.copy(),
                    position_counts=azero.unpack_repetition_history(mirrored.repetition_history),
                )
                mirrored_encoded = azero.encode_state_numpy(mirrored_info, mirrored.history)
                self.assertTrue(np.array_equal(mirrored_encoded, encoded[:, :, ::-1]))
                # policy 概率守恒
                self.assertAlmostEqual(sum(mirrored.policy_prob), 1.0, places=6)
                # 镜像后的动作索引与镜像着法一一对应
                expected_idx = sorted(azero.move_to_index(azero.mirror_move(move)) for move in legal)
                self.assertEqual(sorted(mirrored.policy_idx), expected_idx)
                # 双重镜像还原
                back = azero.mirror_sample(mirrored)
                self.assertEqual(back.board, sample.board)
                self.assertEqual(back.history, sample.history)
                self.assertEqual(back.policy_idx, sample.policy_idx)
                self.assertEqual(back.pool_known, sample.pool_known)

    def test_augment_mirror_does_not_mutate_original(self) -> None:
        produced = self._random_game_samples(34, plies=8)
        sample = produced[3][0]
        board = sample.board.copy()
        policy_idx = list(sample.policy_idx)
        history = list(sample.history)
        augmented = azero.augment_mirror([sample])
        self.assertEqual(sample.board, board)
        self.assertEqual(sample.policy_idx, policy_idx)
        self.assertEqual(sample.history, history)
        self.assertEqual(augmented[0].board, azero.mirror_board(board))

    def _snapshot_black_captured_red_dark(self, seed: int, max_plies: int = 200):
        """随机对局到“黑吃掉红暗子”的局面快照（黑因此获得私有知识）。

        返回 (state, belief, captured_type)；找不到返回 None。
        """
        rng = random.Random(seed)
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        for _ in range(max_plies):
            if state.game_outcome()[0] is not None:
                return None
            legal = state.legal_moves()
            if not legal:
                return None
            mover = state.side
            undo = state.apply_move(rng.choice(legal), rng=rng)
            belief.observe_move(undo, strict=True)
            if (
                mover == jq.BLACK
                and undo.captured_before
                and jq.is_dark(undo.captured_before)
                and jq.piece_color(undo.captured_before) == jq.RED
            ):
                return state, belief, undo.captured_reveal
        return None

    def _build_private_lead_belief(self, belief, captured_type: int, degraded: bool):
        """A（degraded=False）= 黑保留真实私有知识；B = 黑按公开信息退化。

        只改黑方对红池的车/被吃计数，红方观察者行保持不变。
        """
        out = belief.clone()
        if degraded:
            out.pool_counts[jq.BLACK][jq.RED][captured_type] += 1
            out.unknown_removals[jq.BLACK][jq.RED] += 1
            out.pool_total[jq.BLACK][jq.RED] -= 1
        return out

    def _run_search_with_trace(self, state, belief, sims: int = 40):
        """跑一次搜索并记录每个叶评估携带的信息（side + 双方池计数）。

        桩网络确定性且对池平面敏感：若叶评估读到对手私有行，trace 与结果会变化。
        """
        cfg = azero.AZConfig(mcts_simulations=sims, self_play_workers=1, leaf_batch_size=8)
        net = azero.PolicyValueNet()

        def stub_forward(x):
            batch = x.size(0)
            logits = torch.zeros(batch, azero.MOVE_SIZE)
            pool_offset = 29
            red_pool_elephant = x[
                :,
                azero._pool_plane_index(jq.RED, pool_offset) + jq.POOL_TYPES.index(jq.TYPE_ELEPHANT),
                0,
                0,
            ]
            red_to_move = x[:, 28, 0, 0] > 0.5
            value = torch.where(red_to_move, -red_pool_elephant, red_pool_elephant)
            return logits, value

        net.forward = stub_forward
        mcts = azero.NeuralMCTS(net, cfg, torch.device("cpu"), random.Random(5))
        traces: list[tuple] = []
        original_batch = mcts._evaluate_batch

        def spy_batch(infos, histories, **kwargs):
            policies, values = original_batch(infos, histories, **kwargs)
            for info in infos:
                traces.append((
                    info.side,
                    tuple(info.pool_known[jq.RED]),
                    tuple(info.pool_known[jq.BLACK]),
                ))
            return policies, values

        mcts._evaluate_batch = spy_batch
        policy, move, root = mcts.search(
            state, belief, history=[], temperature=1e-6, simulations=sims,
            add_dirichlet=False, return_root=True,
        )
        return policy, move, root, traces

    def test_opponent_private_belief_does_not_leak_into_search(self) -> None:
        """R3 回归：搜索不得读取真实对手私有信念；模拟知识仅从根采样生成。

        黑真实吃过红暗子后，只有黑知道该子类型。退化黑方那条私有知识
        （红方观察者逐字段不变）后，搜索的全部叶评估输入与根结果必须一致。
        修复前每个节点按行动方重抽，叶 trace 会带上只由黑私信决定的池行。
        """
        snapshot = self._snapshot_black_captured_red_dark(seed=1)
        self.assertIsNotNone(snapshot, "未能录到黑吃红暗子的局面")
        state, belief, captured_type = snapshot
        belief_a = self._build_private_lead_belief(belief, captured_type, degraded=False)
        belief_b = self._build_private_lead_belief(belief, captured_type, degraded=True)

        # 根信息态（红视角）在 A/B 必须完全一致
        info_a = belief_a.info_state(state, jq.RED)
        info_b = belief_b.info_state(state, jq.RED)
        self.assertEqual(info_a.pool_known, info_b.pool_known)
        self.assertEqual(info_a.pool_total_known, info_b.pool_total_known)
        self.assertEqual(info_a.unknown_removals, info_b.unknown_removals)

        policy_a, move_a, root_a, trace_a = self._run_search_with_trace(state, belief_a)
        policy_b, move_b, root_b, trace_b = self._run_search_with_trace(state, belief_b)

        # 1) 根方叶观测不超出根计数；合成对手对自己被吃类型只知道公共上界。
        self.assertTrue(trace_a)
        for side, red_row, black_row in trace_a + trace_b:
            for got, bound in zip(red_row, info_a.pool_known[jq.RED]):
                self.assertLessEqual(got, bound)
            black_bound = (info_a.public_pool_known[jq.BLACK] if side == jq.BLACK else
                           info_a.pool_known[jq.BLACK])
            for got, bound in zip(black_row, black_bound):
                self.assertLessEqual(got, bound)
        # 2) 仅改对手私信：叶 trace 与根结果必须完全一致
        self.assertEqual(trace_a, trace_b)
        self.assertEqual(move_a, move_b)
        self.assertEqual(policy_a, policy_b)
        self.assertAlmostEqual(
            root_a.value_sum / root_a.visits, root_b.value_sum / root_b.visits, places=9
        )

    def test_leaf_evaluation_never_reads_real_opponent_observer(self) -> None:
        """真实 belief 仅读取根槽；允许从根一致粒子生成合成行动者观测。

        修复前的叶评估在轮到对手时读取对手私有观测槽（sim_state.side）；
        本测试区分真实 belief 与模拟 belief，避免把模拟对手的知识也抹掉。
        """
        snapshot = self._snapshot_black_captured_red_dark(seed=1)
        self.assertIsNotNone(snapshot, "未能录到黑吃红暗子的局面")
        state, belief, _captured_type = snapshot
        root_player = state.side
        observers: list[int] = []
        real_info_state = im.BeliefModel.info_state

        def spy_info_state(model, game_state, observer):
            if model is belief:
                observers.append(observer)
            return real_info_state(model, game_state, observer)

        cfg = azero.AZConfig(mcts_simulations=40, self_play_workers=1, leaf_batch_size=8)
        mcts = azero.NeuralMCTS(azero.PolicyValueNet(), cfg, torch.device("cpu"), random.Random(5))
        with mock.patch.object(im.BeliefModel, "info_state", spy_info_state):
            mcts.search(
                state, belief, history=[], temperature=1e-6, simulations=40, add_dirichlet=False
            )
        self.assertTrue(observers)
        self.assertEqual(set(observers), {root_player})


if __name__ == "__main__":
    unittest.main()
