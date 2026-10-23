import copy
import os
from pathlib import Path
import pickle
import unittest
from unittest import mock

import numpy as np
import azero as az
import jieqi as jq
from test_encoding_acceleration import fixtures
from tools import benchmark_repetition_memo as experiment


class RepetitionMicroGateTests(unittest.TestCase):
    def test_both_original_and_uncached_gains_required(self):
        self.assertTrue(experiment.micro_gate(dict(plain=10, uncached=9, compiled=8))["micro_gate_passed"])
        self.assertFalse(experiment.micro_gate(dict(plain=10, uncached=8.5, compiled=8))["micro_gate_passed"])
        self.assertFalse(experiment.micro_gate(dict(plain=8.5, uncached=10, compiled=8))["micro_gate_passed"])
        with self.assertRaises(ValueError):
            experiment.micro_gate(dict(plain=10, compiled=8))


class RepetitionMemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get("JIEQI_EXPERIMENT_REPETITION_DIR")
        if not directory:
            raise unittest.SkipTest("set JIEQI_EXPERIMENT_REPETITION_DIR to the isolated build")
        cls.module = experiment.load_memo(Path(directory))

    def setUp(self):
        self.memo = self.module.RepetitionMemo(collect_stats=True)

    def compare(self, info, memo=None):
        memo = self.memo if memo is None else memo
        before = pickle.dumps(info, protocol=5)
        results = []
        for function in (experiment.ORIGINAL, memo.entries):
            try:
                results.append((function(info), None))
            except Exception as error:
                results.append((type(error), error.args))
            self.assertEqual(pickle.dumps(info, protocol=5), before)
        self.assertEqual(results[0], results[1])
        return results[0]

    def test_fixtures_are_exact_fast_and_uncached_ablation(self):
        for info, _ in fixtures():
            self.compare(info)
            self.compare(info, self.module.RepetitionMemo(enabled=False))
        self.assertGreater(self.memo.stats()["fast_calls"], 0)

    def test_count_side_and_dictionary_order_are_never_cached(self):
        info, _ = fixtures()[1]
        self.compare(info)
        self.compare(info)
        key = next(iter(info.position_counts))
        info.position_counts[key] = 2**70+1
        self.compare(info)
        info.position_counts = dict(reversed(list(info.position_counts.items())))
        self.compare(info)
        self.assertGreater(self.memo.stats()["hits"], 0)
        self.assertEqual(experiment.ORIGINAL(info), self.memo.entries(info))

    def test_current_missing_incomplete_and_current_board_changes(self):
        info, _ = fixtures()[1]
        self.compare(info)
        info.position_counts.pop((info.side, tuple(info.board)), None)
        self.compare(info)
        info.board = [0]*90
        self.compare(info)
        info.repetition_history_complete = False
        self.compare(info)
        info.repetition_history_complete = True
        info.position_counts = None
        self.compare(info)

    def test_filtered_invalid_bytes_only_raise_when_original_does(self):
        info, _ = fixtures()[1]
        for piece in (32, 256, -1, np.int64(31), True):
            info.position_counts = {(info.side, tuple([piece]*90)): 1}
            self.compare(info)
        self.assertGreater(self.memo.stats()["fallback_calls"], 0)

    def test_bad_key_count_board_and_non_standard_current_fallback(self):
        info, _ = fixtures()[1]
        for counts in ({(0,): 1}, {(0, (0,)*89): 1}, {(0, (0,)*91): 1},
                       {(0, (0,)*90): np.int64(2)}, {(False, (0,)*90): 1}):
            info.position_counts = counts
            self.compare(info)
        info.position_counts = {}
        for board in (tuple([0]*90), [np.int64(0)]*90, [0]*89, [0]*91):
            info.board = board
            self.compare(info)

    def test_fifo_capacity_and_equal_distinct_objects(self):
        info, _ = fixtures()[1]
        memo = self.module.RepetitionMemo(capacity=1, collect_stats=True)
        first, second = tuple(info.board), tuple(list(info.board))
        self.assertIsNot(first, second)
        for board in (first, first, second, first):
            info.position_counts = {(info.side, board): 1}
            self.compare(info, memo)
        stats = memo.stats()
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 3)
        self.assertEqual(stats["evictions"], 2)
        memo.clear()
        self.assertEqual(memo.stats()["size"], 0)

    def test_full_history_capacity_overflow_and_dynamic_limit(self):
        info, _ = fixtures()[1]
        info.board = [0]*90
        info.board[89] = jq.make_piece(jq.TYPE_ROOK, jq.RED, 0)
        info.position_counts = {}
        for index in range(81):
            board = [0]*90
            board[index] = info.board[89]
            info.position_counts[index % 2, tuple(board)] = index+1
        self.assertEqual(self.compare(info)[0], ValueError)
        with mock.patch.object(az, "REPETITION_HISTORY_SIZE", 90):
            self.compare(info)

    def test_dependency_overrides_clear_stale_memo(self):
        info, _ = fixtures()[1]
        self.compare(info)
        for module, name, value in ((az, "_public_material_signature", lambda board: (0, 0)),
                                    (jq, "is_dark", lambda piece: 0), (jq, "DARK_SHIFT", 2)):
            with mock.patch.object(module, name, value):
                self.assertFalse(self.memo.supports(info))
                self.compare(info)
                self.assertEqual(self.memo.stats()["size"], 0)
            self.compare(info)
        with mock.patch.object(az, "bytes", lambda board: b"x"*90, create=True):
            self.compare(info)
        self.compare(info)

    def test_encoding_packing_signature_and_batches_are_exact(self):
        pairs = fixtures()
        expected_arrays = az._encode_batch_numpy(*zip(*pairs))
        expected_packed = [az.pack_repetition_history(info) for info, _ in pairs]
        expected_signatures = [az.leaf_input_signature(info, history) for info, history in pairs]
        with mock.patch.object(az, "_repetition_entries", self.memo.entries):
            actual_arrays = az._encode_batch_numpy(*zip(*pairs))
            actual_packed = [az.pack_repetition_history(info) for info, _ in pairs]
            actual_signatures = [az.leaf_input_signature(info, history) for info, history in pairs]
        self.assertEqual(actual_arrays.tobytes(), expected_arrays.tobytes())
        self.assertEqual(actual_packed, expected_packed)
        self.assertEqual(actual_signatures, expected_signatures)

    def test_invalid_capacity_and_stats_disabled(self):
        for capacity in (0, -1, 4097, True, 1.5):
            with self.assertRaises(ValueError):
                self.module.RepetitionMemo(capacity=capacity)
        memo = self.module.RepetitionMemo()
        self.compare(fixtures()[1][0], memo)
        self.assertEqual(memo.stats()["calls"], 0)


if __name__ == "__main__":
    unittest.main()
