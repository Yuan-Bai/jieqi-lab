"""The verifier must detect incorrect probabilities, timestamps, and rule history."""
import contextlib
import copy
import io
import random
import unittest
from unittest import mock

import azero as az
from tools import audit_augmentation, verify_flip_symmetry as verifier


class TestSymmetryVerifier(unittest.TestCase):
    def test_policy_probability_pairing_is_compared(self):
        reference = audit_augmentation.tensor_fixture()
        wrong = copy.deepcopy(reference)
        wrong.policy_prob[0], wrong.policy_prob[1] = wrong.policy_prob[1], wrong.policy_prob[0]
        self.assertFalse(verifier.sample_fields_equal(wrong, reference))

    def test_all_policy_targets_belong_to_current_state(self):
        original = az.raw_sample_from_info
        checked = []

        def check(info, history, policy, *args, **kwargs):
            legal = info.determinize(random.Random(0)).legal_moves()
            self.assertEqual({az.move_to_index(move) for move in legal},
                             {i for i, prob in enumerate(policy) if prob > 0})
            checked.append(len(history))
            return original(info, history, policy, *args, **kwargs)

        with mock.patch.object(az, "raw_sample_from_info", side_effect=check), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verifier.main(["--games", "0"]), 0)
        self.assertIn(0, checked)
        self.assertIn(14, checked)

    def test_repetition_field_mismatch_alone_fails(self):
        original = verifier.state_fields
        calls = 0

        def corrupt(state):
            nonlocal calls
            calls += 1
            fields = original(state)
            return fields[:-1] + ((),) if calls % 2 else fields

        output = io.StringIO()
        with mock.patch.object(verifier, "state_fields", side_effect=corrupt), \
                contextlib.redirect_stdout(output):
            self.assertEqual(verifier.main(["--games", "0"]), 1)
        self.assertIn("position_counts", output.getvalue())


if __name__ == "__main__":
    unittest.main()
