"""Native/Python rules must agree without changing state or search trajectories."""
import copy
import hashlib
import os
from pathlib import Path
import random
import sys
import types
import unittest
from unittest import mock

import jieqi as jq

try:
    import _jieqi_rules as native
except ImportError:
    native = None

PYTHON_ATTACK = jq.GameState._is_square_attacked_python
NATIVE_CURRENT = native is not None and getattr(native, "SOURCE_SHA256", None) == hashlib.sha256(
    (Path(__file__).parent / "native/jieqi_rules.pyx").read_bytes()).hexdigest()


def snapshot(state):
    return {name: copy.deepcopy(getattr(state, name)) for name in state.__slots__ if name != "rng"} | {
        "rng": state.rng.getstate()}


class TestRulesBackendLoading(unittest.TestCase):
    def test_python_mode_never_imports_extension(self):
        with mock.patch.dict(os.environ, JIEQI_RULES_BACKEND="python"), mock.patch.dict(
                sys.modules, {"_jieqi_rules": None}):
            self.assertEqual(jq._load_rules_backend(), ("python", None, None))

    def test_missing_extension_falls_back_or_fails_when_required(self):
        with mock.patch.dict(sys.modules, {"_jieqi_rules": None}):
            with mock.patch.dict(os.environ, JIEQI_RULES_BACKEND="auto"):
                backend, error, record = jq._load_rules_backend()
                self.assertEqual(backend, "python")
                self.assertIsNotNone(error)
                self.assertIsNone(record)
            with mock.patch.dict(os.environ, JIEQI_RULES_BACKEND="cython"):
                with self.assertRaisesRegex(ImportError, "unavailable"):
                    jq._load_rules_backend()

    def test_stale_and_incompatible_extensions_are_rejected(self):
        for version, digest in ((0, "unused"), (1, "stale")):
            fake = types.SimpleNamespace(API_VERSION=version, RULESET_VERSION=jq.RULESET_VERSION,
                                         SOURCE_SHA256=digest)
            with mock.patch.dict(sys.modules, {"_jieqi_rules": fake}), mock.patch.dict(
                    os.environ, JIEQI_RULES_BACKEND="auto"):
                backend, error, record = jq._load_rules_backend()
                self.assertEqual(backend, "python")
                self.assertIn("rebuild", error)
                self.assertIsNone(record)

    def test_missing_snapshot_guard_is_rejected_before_installation(self):
        digest = hashlib.sha256((Path(__file__).parent / "native/jieqi_rules.pyx").read_bytes()).hexdigest()
        fake = types.SimpleNamespace(API_VERSION=1, RULESET_VERSION=jq.RULESET_VERSION,
                                     SOURCE_SHA256=digest)
        attack, guard = jq.GameState.is_square_attacked, jq._immutable_integer_pools
        with mock.patch.dict(sys.modules, {"_jieqi_rules": fake}), mock.patch.dict(
                os.environ, JIEQI_RULES_BACKEND="auto"):
            backend, error, record = jq._load_rules_backend()
            self.assertEqual(backend, "python")
            self.assertIn("snapshot guard", error)
            self.assertIsNone(record)
        self.assertIs(jq.GameState.is_square_attacked, attack)
        self.assertIs(jq._immutable_integer_pools, guard)

    def test_unknown_mode_is_not_silently_ignored(self):
        with mock.patch.dict(os.environ, JIEQI_RULES_BACKEND="typo"):
            with self.assertRaises(ValueError):
                jq._load_rules_backend()


@unittest.skipUnless(NATIVE_CURRENT, "build current extension with python tools/build_rules.py")
class TestNativeRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native.initialize(jq)

    def test_all_squares_random_boards_and_unbounded_integer_codes(self):
        rng = random.Random(20261005)
        state = jq.GameState.empty(rng=random.Random(71))
        codes = [jq.make_piece(t, side, dark) for t in range(1, 8) for side in (0, 1) for dark in (0, 1)]
        codes += [32, 1 << 100, (1 << 100) + 15, -(1 << 100) + 7, -1]
        for _ in range(300):
            state.board = [rng.choice(codes) if rng.random() < 0.35 else 0 for _ in range(90)]
            before = snapshot(state)
            for side in (0, 1):
                for cell in range(90):
                    self.assertEqual(native.is_square_attacked(state, side, cell),
                                     PYTHON_ATTACK(state, side, cell), (side, cell, state.board))
            self.assertEqual(snapshot(state), before)

    def test_each_piece_source_target_and_blocker(self):
        state = jq.GameState.empty()
        for side in (0, 1):
            for dark in (0, 1):
                for ptype in range(1, 8):
                    for source in range(90):
                        state.board = [0] * 90
                        state.board[source] = jq.make_piece(ptype, side, dark)
                        for blocker in (None, (source + 1) % 90):
                            if blocker is not None:
                                state.board[blocker] = jq.make_piece(7, 1 - side, 0)
                            for cell in range(90):
                                self.assertEqual(native.is_square_attacked(state, side, cell),
                                                 PYTHON_ATTACK(state, side, cell))

    def test_python_indexing_and_errors_are_preserved(self):
        state = jq.GameState.initial()
        for board in (state.board, tuple(state.board), state.board[:5]):
            state.board = board
            for side, idx in ((0, -1), (1, -90), (0, 90), (0, -91), (7, 40), (0, 4.0),
                              (True, True), (0, 1 << 100), (1 << 100, 4)):
                try:
                    expected = PYTHON_ATTACK(state, side, idx)
                except Exception as error:
                    with self.assertRaises(type(error)):
                        native.is_square_attacked(state, side, idx)
                else:
                    self.assertEqual(native.is_square_attacked(state, side, idx), expected)

    def test_real_games_order_outcome_bans_rng_and_apply_undo(self):
        for seed in range(12):
            states = [jq.GameState.initial(rng=random.Random(seed)) for _ in range(2)]
            rngs = [random.Random(seed + 1000) for _ in range(2)]
            for _ in range(120):
                results = []
                for state, attack in zip(states, (PYTHON_ATTACK, native.is_square_attacked)):
                    before = snapshot(state)
                    with mock.patch.object(jq.GameState, "is_square_attacked", attack):
                        results.append((state.legal_moves(), state.has_legal_moves(), state.game_outcome(),
                                        state.is_in_check(0), state.is_in_check(1)))
                    self.assertEqual(snapshot(state), before)
                self.assertEqual(results[0], results[1])
                self.assertEqual(snapshot(states[0]), snapshot(states[1]))
                # Exercise the apply/undo-based long-check/chase filtering too.
                for attack in (PYTHON_ATTACK, native.is_square_attacked):
                    probe = states[0].clone()
                    probe.consecutive_checks[probe.side] = probe.rules.long_check_rounds
                    probe.chase_streak[probe.side] = probe.rules.long_chase_rounds
                    probe.chase_target[probe.side] = (probe.king_pos[1 - probe.side] + 1) % 90
                    before = snapshot(probe)
                    with mock.patch.object(jq.GameState, "is_square_attacked", attack):
                        banned = (probe.legal_moves(), probe.game_outcome())
                    self.assertEqual(snapshot(probe), before)
                    if attack is PYTHON_ATTACK:
                        expected_banned = banned
                    else:
                        self.assertEqual(banned, expected_banned)
                if results[0][2][0] is not None:
                    break
                for state, rng, attack in zip(states, rngs, (PYTHON_ATTACK, native.is_square_attacked)):
                    move = rng.choice(results[0][0])
                    rng_state = state.rng.getstate()
                    before = snapshot(state)
                    with mock.patch.object(jq.GameState, "is_square_attacked", attack):
                        undo = state.apply_move(move)
                        state.undo_move(undo)
                        state.rng.setstate(rng_state)
                        self.assertEqual(snapshot(state), before)
                        state.apply_move(move)

    def test_full_search_graph_and_rng_are_identical(self):
        import numpy as np
        import torch
        import azero as az
        import ismcts as im
        from test_selfplay_acceleration import tree_records

        before_rng, threads = torch.get_rng_state(), torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(59)
            net = az.PolicyValueNet(channels=8, blocks=1)
            state = jq.GameState.initial(rng=random.Random(43))
            belief = im.init_belief_for_game(state)
            before = snapshot(state)
            for chance, k in ((False, 0), (True, 0), (True, 1)):
                for dedup in (False, True):
                    records = []
                    config = az.AZConfig(mcts_simulations=64, leaf_batch_size=16, chance_nodes=chance,
                                         chance_k=k, leaf_dedup=dedup)
                    for attack in (PYTHON_ATTACK, native.is_square_attacked):
                        mcts = az.NeuralMCTS(net, config, torch.device("cpu"), random.Random(47))
                        with mock.patch.object(jq.GameState, "is_square_attacked", attack):
                            policy, move, root = mcts.search(state, belief, [], 1.0, return_root=True,
                                                             add_dirichlet=True, prune_policy=True)
                        records.append((np.asarray(policy).tobytes(), move, tree_records(root),
                                        mcts.rng.getstate(), mcts.leaf_evals_total, mcts.leaf_evals_unique,
                                        mcts.chance_applied, mcts.chance_outcomes_added, mcts.chance_nav_miss))
                        self.assertEqual(snapshot(state), before)
                    self.assertEqual(records[0], records[1])
        finally:
            torch.set_rng_state(before_rng)
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
