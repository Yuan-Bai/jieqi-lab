"""Run compiled tests only with an explicit isolated experimental build directory."""
import copy
import os
from pathlib import Path
import pickle
import unittest
from unittest import mock

import numpy as np
import azero as az
import jieqi as jq
from test_encoding_acceleration import fixtures, sample_fixtures
from tools import benchmark_exact_encoder as experiment


class EncoderScreenTests(unittest.TestCase):
    def groups(self, candidate_seconds=9.5):
        groups = []
        for session in range(2):
            for repeat in range(4):
                rows = [dict(index=index, strong=strong, simulations=3200 if strong else 800,
                             seed=1000*repeat+index, exact=dict(policy_sha256="same"))
                        for index in range(18) for strong in (False, True)]
                for mode in experiment.MODES:
                    groups.append(dict(session=session, repeat=repeat, mode=mode,
                        wall_seconds=10 if mode == "plain" else candidate_seconds, rows=copy.deepcopy(rows)))
        return groups

    def test_gate_pass_requires_real_speed_and_both_pools(self):
        self.assertTrue(experiment.summarize_roots(self.groups())["full_game_screen_gate_passed"])
        self.assertFalse(experiment.summarize_roots(self.groups(9.8))["full_game_screen_gate_passed"])
        groups = self.groups()
        for g in groups:
            if g["session"] == 1 and g["mode"] == "compiled":
                g["wall_seconds"] = 10.1
        self.assertFalse(experiment.summarize_roots(groups)["full_game_screen_gate_passed"])

    def test_missing_cohort_and_partial_roots_rejected(self):
        with self.assertRaises(ValueError):
            experiment.summarize_roots(self.groups()[:-1])
        groups = self.groups()
        groups[0]["rows"].pop()
        with self.assertRaises(ValueError):
            experiment.summarize_roots(groups)

    def test_inexact_outputs_rejected(self):
        groups = self.groups()
        groups[1]["rows"][0]["exact"]["policy_sha256"] = "different"
        with self.assertRaises(AssertionError):
            experiment.summarize_roots(groups)

    def test_unstable_baseline_cannot_pass(self):
        groups = self.groups()
        groups[0]["wall_seconds"] = 20
        self.assertFalse(experiment.summarize_roots(groups)["full_game_screen_gate_passed"])


class CompiledEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get("JIEQI_EXPERIMENT_ENCODER_DIR")
        if not directory:
            raise unittest.SkipTest("set JIEQI_EXPERIMENT_ENCODER_DIR to the isolated build")
        cls.encoder = experiment.load_encoder(Path(directory).resolve())

    def compare(self, info, history, entries=az._REPETITION_ENTRIES_UNSET, *, dtype=np.float32,
                strided=False, fill=0, readonly=False):
        before = pickle.dumps((info, history), protocol=5)
        arrays, outcomes = [], []
        for function in (experiment.ORIGINAL, self.encoder.encode_into):
            array = np.full((az.STATE_CHANNELS, 10, 18 if strided else 9), fill, dtype=dtype)
            if strided:
                array = array[:, :, ::2]
            if readonly:
                array.setflags(write=False)
            try:
                value = function(info, history, array, repetition_entries=entries)
                outcomes.append((value, None))
            except Exception as error:
                outcomes.append((type(error), error.args))
            arrays.append(array)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(arrays[0].tobytes(), arrays[1].tobytes())
        self.assertEqual(arrays[0].strides, arrays[1].strides)
        self.assertEqual(before, pickle.dumps((info, history), protocol=5))
        return outcomes[0]

    def test_all_channels_synthetic_histories_and_explicit_entries(self):
        for info, history in fixtures():
            self.compare(info, history)
            self.compare(info, history, None)
            entries = az._repetition_entries(info)
            self.compare(info, history, entries)

    def test_arrays_bits_layout_nonzero_signed_zero_and_readonly(self):
        info, history = fixtures()[1]
        for fill in (0, -0.0, 17.125):
            self.compare(info, history, strided=True, fill=fill)
        self.compare(info, history, readonly=True)
        self.compare(info, history, dtype=np.float64)

    def test_pool_rounding_large_python_integers_and_rule_fields(self):
        info, history = fixtures()[1]
        info.pool_known = [[2**70 + index for index in range(8)] for _ in range(2)]
        info.pool_total_known = [2**65, 2**65+1]
        info.no_capture_plies = 2**65+3
        info.consecutive_checks = [2**65+7, 2**65+9]
        info.chase_streak = [2**65+11, 2**65+13]
        self.compare(info, history)

    def test_full_history_capacity_and_overflow_exception(self):
        info, history = fixtures()[1]
        info.board = [0]*90
        info.board[89] = jq.make_piece(jq.TYPE_ROOK, jq.RED, 0)
        info.position_counts = {}
        for index in range(az.REPETITION_HISTORY_SIZE-1):
            board = [0]*90
            board[index] = info.board[89]
            info.position_counts[index%2, tuple(board)] = 1+index
        self.compare(info, history)
        extra = [0]*90
        extra[82] = info.board[89]
        info.position_counts[0, tuple(extra)] = 82
        outcome = self.compare(info, history)
        self.assertEqual(outcome[0], ValueError)

    def test_fallback_for_non_native_board_and_piece_types(self):
        for conversion in (tuple, lambda b: [np.int64(p) for p in b],
                           lambda b: [bool(p) for p in b]):
            info, history = fixtures()[1]
            info.board = conversion(info.board)
            self.assertFalse(self.encoder.supports(info, np.zeros((238, 10, 9), np.float32)))
            self.compare(info, history)
        for piece in (-1, 32, 2**80):
            info, history = fixtures()[1]
            info.board[0] = piece
            self.assertFalse(self.encoder.supports(info, np.zeros((238, 10, 9), np.float32)))
            self.compare(info, history, None)

    def test_all_piece_codes_and_helpers_changes(self):
        info, history = fixtures()[1]
        info.position_counts = None
        info.board = [index%32 for index in range(90)]
        self.compare(info, history)
        for module, name, value in ((az, "_piece_plane_index", lambda dark, side, kind: 0),
                                    (jq, "is_dark", lambda piece: 0),
                                    (jq, "COLOR_SHIFT", 2), (jq, "TYPE_MASK", 3),
                                    (az, "HISTORY_PLIES", 1)):
            with mock.patch.object(module, name, value):
                self.assertFalse(self.encoder.supports(info, np.zeros((238, 10, 9), np.float32)))
                self.compare(info, history)

    def test_live_coordinate_table_changes_and_bad_history(self):
        info, history = fixtures()[1]
        rows = list(jq.ROWS)
        rows[1] = rows[2]
        with mock.patch.object(jq, "ROWS", rows):
            self.compare(info, history)
        for moves in ([(0, -1)], [(0, 91)], [(0,)], []):
            self.compare(info, moves)

    def test_malformed_explicit_entries_preserve_exceptions_and_partial_output(self):
        info, history = fixtures()[1]
        for entries in (((0, b"short", 1),), ((0,),), [(0, bytes(info.board), 1)],
                        ((0, bytes(info.board), 2**80),)):
            self.compare(info, history, entries)

    def test_batch_and_augmented_compact_tensors(self):
        pairs = fixtures()
        expected = az._encode_batch_numpy(*zip(*pairs))
        with mock.patch.object(az, "_encode_state_into", self.encoder.encode_into):
            actual = az._encode_batch_numpy(*zip(*pairs))
        self.assertEqual(actual.tobytes(), expected.tobytes())
        samples = sample_fixtures()
        expected = az.compact_batch_to_tensors(samples)
        with mock.patch.object(az, "_encode_state_into", self.encoder.encode_into):
            actual = az.compact_batch_to_tensors(samples)
        for a, b in zip(actual, expected):
            self.assertEqual(a.numpy().tobytes(), b.numpy().tobytes())

    def test_guarded_current_inputs_use_fast_path(self):
        for info, _ in fixtures():
            self.assertTrue(self.encoder.supports(info, np.zeros((238, 10, 9), np.float32)))


if __name__ == "__main__":
    unittest.main()
