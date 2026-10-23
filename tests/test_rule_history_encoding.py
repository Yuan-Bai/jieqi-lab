"""规则状态必须能区分禁着与下一步重复和棋，并在数据管线中保留。"""
import gzip
import pickle
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import azero as az
import ismcts as im
import jieqi as jq


def quiet_state():
    state = jq.GameState.empty(rng=random.Random(3), with_pools=False)
    state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
    state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
    state.place_piece(5, 4, jq.TYPE_PAWN, jq.RED, 0)
    state.place_piece(7, 0, jq.TYPE_ROOK, jq.RED, 0)
    state.reset_history()
    return state


def sample_from_info(info):
    policy = [0.0] * az.MOVE_SIZE
    policy[az.move_to_index((jq.index(7, 0), jq.index(7, 1)))] = 1.0
    return az.finalize_samples([az.raw_sample_from_info(info, [], policy)], jq.RESULT_DRAW)[0]


class TestRuleHistoryEncoding(unittest.TestCase):
    def test_chase_victim_code_changes_legality_and_network_input(self):
        state = quiet_state()
        state.place_piece(5, 1, jq.TYPE_KNIGHT, jq.BLACK, 0)
        state.reset_history()
        belief = im.init_belief_for_game(state)
        first, second = [belief.info_state(state, jq.RED) for _ in range(2)]
        for info in (first, second):
            info.chase_streak[jq.RED] = jq.LONG_CHASE_ROUNDS
            info.chase_target[jq.RED] = jq.index(5, 1)  # 目标格随受害子的公开移动更新。
            info.chase_anchor[jq.RED] = jq.index(7, 0)
        first.chase_target_code[jq.RED] = jq.make_piece(jq.TYPE_KNIGHT, jq.BLACK, 0)
        second.chase_target_code[jq.RED] = jq.make_piece(jq.TYPE_CANNON, jq.BLACK, 0)
        move = (jq.index(7, 0), jq.index(7, 1))
        self.assertEqual([move in x.determinize(random.Random(3)).legal_moves()
                          for x in (first, second)], [False, True])
        self.assertFalse(np.array_equal(az.encode_state_numpy(first, []), az.encode_state_numpy(second, [])))

    def test_check_anchor_changes_legality_and_network_input(self):
        state = quiet_state()
        state.board[jq.index(7, 0)] = 0
        state.place_piece(2, 3, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(2, 5, jq.TYPE_ROOK, jq.RED, 0)
        state.reset_history()
        belief = im.init_belief_for_game(state)
        first, second = [belief.info_state(state, jq.RED) for _ in range(2)]
        first.consecutive_checks[jq.RED] = second.consecutive_checks[jq.RED] = 6
        first.check_anchor[jq.RED], second.check_anchor[jq.RED] = jq.index(2, 3), jq.index(2, 5)
        move = (jq.index(2, 3), jq.index(2, 4))
        self.assertEqual([move in x.determinize(random.Random(3)).legal_moves()
                          for x in (first, second)], [False, True])
        self.assertFalse(np.array_equal(az.encode_state_numpy(first, []), az.encode_state_numpy(second, [])))

    def test_other_position_count_changes_next_outcome_input_and_dedup(self):
        state = quiet_state()
        move = (jq.index(7, 0), jq.index(7, 1))
        undo = state.apply_move(move)
        future_key = state.position_key
        state.undo_move(undo)
        belief = im.init_belief_for_game(state)
        first, second = [belief.info_state(state, jq.RED) for _ in range(2)]
        first.position_counts[future_key], second.position_counts[future_key] = 1, 2
        outcomes = []
        for info in (first, second):
            world = info.determinize(random.Random(3))
            world.apply_move(move)
            outcomes.append(world.game_outcome()[0])
        self.assertEqual(outcomes, [None, jq.RESULT_DRAW])
        a, b = [az.encode_state_numpy(x, []) for x in (first, second)]
        np.testing.assert_array_equal(a[:71], b[:71])
        self.assertFalse(np.array_equal(a, b))
        self.assertNotEqual(az.leaf_input_signature(first, []), az.leaf_input_signature(second, []))

    def test_full_history_roundtrip_augmentation_and_old_entry(self):
        state = quiet_state()
        info = im.init_belief_for_game(state).info_state(state, jq.RED)
        # 超过最近八步的不同公共局面；固定行车格，保持物料不变。
        for idx in range(12):
            board = info.board.copy()
            board[jq.index(7, 0)] = 0
            board[idx + 18] = jq.make_piece(jq.TYPE_ROOK, jq.RED, 0)
            info.position_counts[(idx % 2, tuple(board))] = 1 + idx % 2
        info.check_anchor = [jq.index(0, 4), jq.index(7, 0)]
        info.chase_target_code = [jq.make_piece(jq.TYPE_ROOK, jq.RED, 0), jq.make_piece(jq.TYPE_KNIGHT, jq.BLACK, 0)]
        sample = sample_from_info(info)
        self.assertEqual(az.unpack_repetition_history(sample.repetition_history), info.position_counts)
        expected = az.encode_state(info, [], info.side)
        self.assertTrue(torch.equal(az.compact_to_tensors(sample)[0], expected))
        first_key = list(info.position_counts)[1]
        info.position_counts[first_key] += 1
        self.assertFalse(torch.equal(az.encode_state(info, [], info.side), expected))
        for transform, board_transform, side_transform in (
            (az.mirror_sample, az.mirror_board, lambda s: s),
            (az.flip_vertical_sample, az.flip_vertical_board, lambda s: 1 - s),
        ):
            transformed = transform(sample)
            reference = {(side_transform(side), tuple(board_transform(list(board)))): count
                         for (side, board), count in az.unpack_repetition_history(sample.repetition_history).items()}
            self.assertEqual(az.unpack_repetition_history(transformed.repetition_history), reference)
            self.assertEqual(transform(transformed), sample)
            # 变换后的字节记录真的进入张量，而不是只保存在样本属性里。
            encoded = az.compact_to_tensors(transformed)[0]
            self.assertTrue(torch.equal(az.compact_to_tensors(transform(transformed))[0], expected))
            self.assertEqual(int(torch.count_nonzero(encoded[az.REPETITION_HISTORY_OFFSET + 1::2, 0, 0])), len(reference))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "history.pkl.gz")
            az.save_compact_samples([sample], path)
            loaded = az.load_compact_game(path)[0]
            self.assertEqual(loaded, sample)
            self.assertTrue(torch.equal(az.compact_to_tensors(loaded)[0], expected))

    def test_irreversible_history_pruned_but_future_lower_material_retained(self):
        state = quiet_state()
        info = im.init_belief_for_game(state).info_state(state, jq.RED)
        extra = info.board.copy()
        extra[jq.index(1, 0)] = jq.make_piece(jq.TYPE_ROOK, jq.BLACK, 0)
        dark = info.board.copy()
        dark[jq.index(7, 0)] = jq.make_piece(jq.TYPE_ROOK, jq.RED, 1)
        lower = info.board.copy()
        lower[jq.index(7, 0)] = 0
        info.position_counts.update({(jq.BLACK, tuple(extra)): 2, (jq.BLACK, tuple(dark)): 2,
                                     (jq.BLACK, tuple(lower)): 2})
        packed = az.unpack_repetition_history(az.pack_repetition_history(info))
        self.assertEqual(set(packed), {(info.side, tuple(info.board)), (jq.BLACK, tuple(lower))})

    def test_history_overflow_is_explicit_not_silently_truncated(self):
        state = quiet_state()
        info = im.init_belief_for_game(state).info_state(state, jq.RED)
        # 不合法的超长历史也不能被当作完整历史静默截断。
        for idx in range(az.REPETITION_HISTORY_SIZE + 1):
            board = [0] * jq.BOARD_SIZE
            board[idx] = jq.make_piece(jq.TYPE_ROOK, jq.RED, 0)
            info.position_counts[(jq.RED, tuple(board))] = 1
        with self.assertRaisesRegex(ValueError, "repetition history"):
            az.encode_state_numpy(info, [])
        with self.assertRaisesRegex(ValueError, "repetition history"):
            az.pack_repetition_history(info)

    def test_legacy_v2_missing_slot_keeps_current_count_and_marks_history_unknown(self):
        state = quiet_state()
        info = im.init_belief_for_game(state).info_state(state, jq.RED)
        info.position_counts[(info.side, tuple(info.board))] = 2
        sample = sample_from_info(info)
        del sample.repetition_history  # 模拟真正的 v2 slots pickle，字段根本不存在。
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "legacy.pkl.gz")
            with gzip.open(path, "wb") as file:
                pickle.dump({"version": 2, "samples": [sample], "meta": {"ruleset_version": 3}}, file)
            loaded = az.load_compact_game(path)[0]
            self.assertIsNone(loaded.repetition_history)
            for augmented in (loaded, az.mirror_sample(loaded), az.flip_vertical_sample(loaded)):
                tensor = az.compact_to_tensors(augmented)[0]
                self.assertAlmostEqual(float(tensor[64, 0, 0]), 2 / 3)
                self.assertEqual(float(tensor[az.REPETITION_HISTORY_COMPLETE_CHANNEL, 0, 0]), 0)
                self.assertEqual(int(torch.count_nonzero(tensor[az.REPETITION_HISTORY_OFFSET:])), 0)
            self.assertEqual(az.mirror_sample(az.mirror_sample(loaded)), loaded)
        incomplete = im.init_belief_for_game(state).info_state(state, jq.RED)
        incomplete.position_counts = dict(info.position_counts)
        incomplete.repetition_history_complete = False
        self.assertNotEqual(az.leaf_input_signature(info, []), az.leaf_input_signature(incomplete, []))

    def test_old_71_channel_model_preserves_forward_after_padding(self):
        torch.manual_seed(42)
        old = az.PolicyValueNet(channels=16, blocks=1)
        old.conv = torch.nn.Conv2d(71, 16, kernel_size=3, padding=1, bias=False)
        fresh = az.PolicyValueNet(channels=16, blocks=1)
        self.assertTrue(az.load_state_dict_padded(fresh, old.state_dict()))
        self.assertTrue(torch.equal(fresh.conv.weight[:, :71], old.conv.weight))
        self.assertEqual(int(torch.count_nonzero(fresh.conv.weight[:, 71:])), 0)
        state = quiet_state()
        tensor = az.compact_to_tensors(sample_from_info(im.init_belief_for_game(state).info_state(state, jq.RED)))[0][None]
        old.eval()
        fresh.eval()
        with torch.no_grad():
            for actual, expected in zip(fresh(tensor), old(tensor[:, :71])):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
