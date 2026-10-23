import copy
import unittest

import azero as az
from test_encoding_acceleration import fixtures
from tools import measure_repetition_preparation as measure


class RepetitionInspectionTests(unittest.TestCase):
    def test_shadow_does_not_change_info_or_outputs(self):
        info, _ = fixtures()[1]
        original = copy.deepcopy(info)
        tracker = measure.IdentityTracker()
        for _ in range(3):
            entries = az._repetition_entries(info)
            tracker.observe(info, entries)
            self.assertEqual(entries, az._repetition_entries(original))
        self.assertEqual(info, original)
        counts = tracker.result()["counts"]
        self.assertGreater(counts["identity_hits"], 0)
        self.assertGreater(counts["repeated_conversions"], 0)

    def test_capacity_and_equal_distinct_tuple_identity(self):
        info, _ = fixtures()[1]
        info.position_counts = {}
        tracker = measure.IdentityTracker(capacity=1)
        first = tuple(info.board)
        second = tuple(list(first))
        self.assertIsNot(first, second)
        for board in (first, first, second, first):
            info.position_counts = {(info.side, board): 1}
            tracker.observe(info, az._repetition_entries(info))
        counts = tracker.result()["counts"]
        self.assertEqual(counts["identity_hits"], 1)
        self.assertEqual(counts["identity_misses"], 3)
        self.assertEqual(counts["evictions"], 2)
        self.assertEqual(counts["peak_boards"], 1)

    def test_incomplete_and_filtered_boards(self):
        info, _ = fixtures()[1]
        tracker = measure.IdentityTracker()
        info.position_counts = None
        tracker.observe(info, az._repetition_entries(info))
        info.position_counts = {(info.side, tuple([31]*90)): 1}
        tracker.observe(info, az._repetition_entries(info))
        self.assertEqual(tracker.counts["incomplete_calls"], 1)
        self.assertEqual(tracker.counts["filtered_queries"], 1)
        self.assertEqual(tracker.counts["converted_queries"], 0)

    def test_reject_unsupported_and_invalid_capacity(self):
        self.assertFalse(measure.standard_board(tuple([True]*90)))
        self.assertFalse(measure.standard_board(tuple([32]*90)))
        self.assertFalse(measure.standard_board(tuple([0]*89)))
        for capacity in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                measure.IdentityTracker(capacity)

    def test_prototype_gate_requires_both_rate_and_lookup_cost(self):
        rows = [dict(inspection=dict(counts=dict(board_queries=100, identity_hits=90,
                      converted_queries=50, repeated_conversions=40)))]
        component = dict(identity_vs_signature_gain=.3)
        self.assertTrue(measure.summarize(rows, component)["prototype_worth_screening"])
        self.assertFalse(measure.summarize(rows, dict(identity_vs_signature_gain=.1))["prototype_worth_screening"])
        rows[0]["inspection"]["counts"]["unsupported_queries"] = 1
        self.assertFalse(measure.summarize(rows, component)["prototype_worth_screening"])


if __name__ == "__main__":
    unittest.main()
