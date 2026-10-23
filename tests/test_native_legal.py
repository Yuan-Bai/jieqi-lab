"""Exactness and provenance for the optional no-ban legal-move compilation."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import jieqi as jq
from test_native_rules import snapshot

ROOT = Path(__file__).resolve().parents[1]
try:
    import _jieqi_legal as native
except ImportError:
    native = None
CURRENT = (native is not None and getattr(native, "API_VERSION", None) == 1
           and getattr(native, "RULESET_VERSION", None) == jq.RULESET_VERSION
           and getattr(native, "SOURCE_SHA256", None) == hashlib.sha256(
               (ROOT / "native/jieqi_legal.pyx").read_bytes()).hexdigest()
           and getattr(native, "METHODS_SHA256", None) == jq._legal_methods_sha256())


@contextmanager
def method(mode):
    old = jq.GameState._iter_legal_moves_fast
    old_context = getattr(jq.GameState, "_native_legal_module", jq)
    jq.GameState._native_legal_module = jq
    jq.GameState._iter_legal_moves_fast = (jq.GameState._iter_legal_moves_fast_python
                                         if mode == "python" else native.iter_legal_moves_fast)
    try:
        yield
    finally:
        jq.GameState._iter_legal_moves_fast = old
        jq.GameState._native_legal_module = old_context


class TestLegalLoading(unittest.TestCase):
    def setUp(self):
        self.original = jq.GameState._iter_legal_moves_fast

    def tearDown(self):
        jq.GameState._iter_legal_moves_fast = self.original

    def test_python_does_not_import(self):
        with mock.patch.dict(os.environ, JIEQI_LEGAL_BACKEND="python"), mock.patch.dict(
                sys.modules, {"_jieqi_legal": None}):
            self.assertEqual(jq._load_legal_backend(), ("python", None, None))
            self.assertIs(jq.GameState._iter_legal_moves_fast, jq.GameState._iter_legal_moves_fast_python)

    def test_missing_binary_fallback_and_explicit_failure(self):
        with mock.patch.dict(sys.modules, {"_jieqi_legal": None}):
            with mock.patch.dict(os.environ, JIEQI_LEGAL_BACKEND="auto"):
                backend, error, record = jq._load_legal_backend()
                self.assertEqual(backend, "python")
                self.assertTrue(error)
                self.assertIsNone(record)
            with mock.patch.dict(os.environ, JIEQI_LEGAL_BACKEND="cython"):
                with self.assertRaisesRegex(ImportError, "unavailable"):
                    jq._load_legal_backend()

    def test_version_source_and_method_guards(self):
        source = hashlib.sha256((ROOT / "native/jieqi_legal.pyx").read_bytes()).hexdigest()
        for api, sha, methods in ((0, source, jq._legal_methods_sha256()),
                                  (1, "stale", jq._legal_methods_sha256()), (1, source, "stale")):
            fake = types.SimpleNamespace(API_VERSION=api, RULESET_VERSION=jq.RULESET_VERSION,
                                         SOURCE_SHA256=sha, METHODS_SHA256=methods)
            with mock.patch.dict(sys.modules, {"_jieqi_legal": fake}), mock.patch.dict(
                    os.environ, JIEQI_LEGAL_BACKEND="auto"):
                backend, error, record = jq._load_legal_backend()
                self.assertEqual(backend, "python")
                self.assertIn("rebuild", error)
                self.assertIsNone(record)

    def test_bad_mode(self):
        with mock.patch.dict(os.environ, JIEQI_LEGAL_BACKEND="bad"):
            with self.assertRaises(ValueError):
                jq._load_legal_backend()


@unittest.skipUnless(CURRENT, "build current extension with python tools/build_legal.py")
class TestNativeLegal(unittest.TestCase):
    def compare_fast(self, state, side):
        rows = []
        for mode in ("python", "cython"):
            clone = copy.deepcopy(state)
            before = snapshot(clone)
            with method(mode):
                try:
                    moves = list(clone._iter_legal_moves_fast(side))
                except Exception as error:
                    rows.append((type(error), str(error), snapshot(clone)))
                else:
                    self.assertEqual(snapshot(clone), before)
                    rows.append((moves, snapshot(clone)))
        self.assertEqual(rows[0], rows[1])

    def test_piece_sources_blockers_and_random_integer_boards(self):
        rng = random.Random(20261006)
        state = jq.GameState.empty(rng=random.Random(51))
        for side in (0, 1):
            for dark in (0, 1):
                for kind in range(1, 8):
                    for source in range(90):
                        state.board = [0] * 90
                        state.board[source] = jq.make_piece(kind, side, dark)
                        for blocker in (None, (source + 1) % 90):
                            if blocker is not None:
                                state.board[blocker] = jq.make_piece(7, 1-side, 0)
                            before = snapshot(state)
                            self.assertEqual(list(native._pseudo(state, side, jq)),
                                             list(state.iter_pseudo_moves(side)))
                            self.assertEqual(snapshot(state), before)
        codes = [jq.make_piece(k, s, d) for k in range(1, 8) for s in (0, 1) for d in (0, 1)]
        codes += [32, 1 << 100, (1 << 100) + 15, -(1 << 100) + 7, -1]
        for _ in range(150):
            state.board = [rng.choice(codes) if rng.random() < 0.3 else 0 for _ in range(90)]
            state.king_pos = [4, 85]
            for side in (0, 1):
                self.assertEqual(list(native._pseudo(state, side, jq)), list(state.iter_pseudo_moves(side)))
                self.compare_fast(state, side)

    def test_real_games_bans_queries_and_apply_undo(self):
        for seed in range(8):
            states = [jq.GameState.initial(rng=random.Random(seed)) for _ in range(2)]
            picker = random.Random(seed + 1000)
            for ply in range(100):
                rows = []
                for state, mode in zip(states, ("python", "cython")):
                    before = snapshot(state)
                    with method(mode):
                        rows.append((state.legal_moves(), state.has_legal_moves(), state._move_scan(),
                                     state.game_outcome()))
                    self.assertEqual(snapshot(state), before)
                self.assertEqual(rows[0], rows[1])
                if ply % 10 == 0:
                    for check, chase in ((True, False), (False, True), (True, True)):
                        outputs = []
                        for state, mode in zip(states, ("python", "cython")):
                            clone = state.clone()
                            side = clone.side
                            clone.consecutive_checks[side] = clone.rules.long_check_rounds if check else 0
                            clone.chase_streak[side] = clone.rules.long_chase_rounds if chase else 0
                            clone.chase_target[side] = (clone.king_pos[1-side] + 1) % 90
                            before = snapshot(clone)
                            with method(mode):
                                outputs.append((clone.legal_moves(), clone._move_scan(), clone.game_outcome()))
                            self.assertEqual(snapshot(clone), before)
                        self.assertEqual(outputs[0], outputs[1])
                if rows[0][-1][0] is not None:
                    break
                move = picker.choice(rows[0][0])
                for state, mode in zip(states, ("python", "cython")):
                    before = snapshot(state)
                    with method(mode):
                        undo = state.apply_move(move, rng=random.Random(103))
                        state.undo_move(undo)
                        self.assertEqual(snapshot(state), before)
                        state.apply_move(move)
                self.assertEqual(snapshot(states[0]), snapshot(states[1]))

    def test_partial_iteration_close_and_error_state(self):
        state = jq.GameState.initial(rng=random.Random(7))
        for mode in ("python", "cython"):
            clone = state.clone()
            before = snapshot(clone)
            with method(mode):
                iterator = clone._iter_legal_moves_fast(clone.side)
                next(iterator)
                self.assertEqual(snapshot(clone), before)
                iterator.close()
                self.assertEqual(snapshot(clone), before)
                iterator = clone._iter_legal_moves_fast(clone.side)
                next(iterator)
                with self.assertRaisesRegex(RuntimeError, "stop iteration"):
                    iterator.throw(RuntimeError("stop iteration"))
                self.assertEqual(snapshot(clone), before)
        results = []
        for mode in ("python", "cython"):
            clone = state.clone()
            def fail(*args):
                raise ValueError("attack callback")
            with method(mode), mock.patch.object(jq.GameState, "is_square_attacked", fail):
                with self.assertRaisesRegex(ValueError, "attack callback"):
                    list(clone._iter_legal_moves_fast(clone.side))
            results.append(snapshot(clone))
        self.assertEqual(results[0], results[1])

    def test_invalid_types_missing_kings_and_subclasses_fall_back(self):
        state = jq.GameState.initial(rng=random.Random(8))
        class CustomBoard(list):
            pass
        for board in (state.board, tuple(state.board), CustomBoard(state.board), state.board[:5]):
            clone = state.clone()
            clone.board = board
            for side in (0, 1, True, None, -1, 2, 0.0, 1 << 100):
                self.compare_fast(clone, side)
        for kings in ((-1, 85), (4, -1), (90, 85), (4, 1 << 100)):
            clone = state.clone()
            clone.king_pos = list(kings)
            for side in (0, 1):
                self.compare_fast(clone, side)
        clone = state.clone()
        clone.king_pos = tuple(clone.king_pos)
        self.compare_fast(clone, clone.side)
        class CustomState(jq.GameState):
            pass
        clone = CustomState.initial(rng=random.Random(9))
        self.compare_fast(clone, clone.side)

    def test_python_callbacks_helpers_and_constants_are_respected(self):
        state = jq.GameState.initial(rng=random.Random(10))
        pseudo = jq.GameState._iter_pseudo_moves_python
        calls = []
        def changed(self, side=None):
            calls.append(side)
            yield from reversed(list(pseudo(self, side)))
        with mock.patch.object(jq.GameState, "iter_pseudo_moves", changed):
            self.compare_fast(state, state.side)
        self.assertEqual(calls, [state.side, state.side])
        helper = jq.piece_color
        with mock.patch.object(jq, "piece_color", lambda p: helper(p)):
            self.compare_fast(state, state.side)
        with mock.patch.object(jq, "TYPE_PAWN", 99):
            self.compare_fast(state, state.side)
        import numpy as np
        clone = state.clone()
        clone.board = [np.int64(p) for p in clone.board]
        self.compare_fast(clone, clone.side)
        class CustomCode(int):
            def __rshift__(self, count):
                return super().__rshift__(count) ^ (1 if count == 3 else 0)
        clone = state.clone()
        clone.board = [CustomCode(p) if p else 0 for p in clone.board]
        self.compare_fast(clone, clone.side)

    def test_module_isolation_and_method_fingerprint(self):
        name = "jieqi_legal_clone"
        spec = importlib.util.spec_from_file_location(name, ROOT / "jieqi.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {name: module}), mock.patch.dict(
                os.environ, JIEQI_LEGAL_BACKEND="cython", JIEQI_RULES_BACKEND="python"):
            spec.loader.exec_module(module)
            self.assertEqual(module._legal_methods_sha256(), native.METHODS_SHA256)
            self.assertIs(module.GameState._native_legal_module, module)
            clone = module.GameState.initial(rng=random.Random(17))
            self.assertEqual(list(clone._iter_legal_moves_fast(clone.side)),
                             list(module.GameState._iter_legal_moves_fast_python(clone, clone.side)))
            with method("cython"):
                self.assertIs(jq.GameState._native_legal_module, jq)
                self.compare_fast(jq.GameState.initial(), jq.RED)

    def test_full_cpu_search_tree_policy_rng_and_inputs(self):
        import numpy as np
        import torch
        import azero as az
        import ismcts as im
        from test_selfplay_acceleration import tree_records
        saved_rng, threads = torch.get_rng_state(), torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(63)
            net = az.PolicyValueNet(channels=8, blocks=1)
            state = jq.GameState.initial(rng=random.Random(45))
            belief = im.init_belief_for_game(state)
            before = snapshot(state)
            for chance, k in ((False, 0), (True, 0), (True, 1)):
                for dedup in (False, True):
                    outputs = []
                    config = az.AZConfig(mcts_simulations=64, leaf_batch_size=16, chance_nodes=chance,
                                         chance_k=k, leaf_dedup=dedup)
                    for mode in ("python", "cython"):
                        search = az.NeuralMCTS(net, config, torch.device("cpu"), random.Random(49))
                        with method(mode):
                            policy, move, root = search.search(state, belief, [], 1.0, return_root=True,
                                                               add_dirichlet=True, prune_policy=True)
                        outputs.append((np.asarray(policy).tobytes(), move, pickle.dumps(tree_records(root)),
                                        search.rng.getstate(), search.leaf_evals_total, search.leaf_evals_unique,
                                        search.chance_applied, search.chance_outcomes_added, search.chance_nav_miss))
                        self.assertEqual(snapshot(state), before)
                    self.assertEqual(outputs[0], outputs[1])
        finally:
            torch.set_rng_state(saved_rng)
            torch.set_num_threads(threads)

    def test_loaded_binary_provenance_and_changed_file_rejected(self):
        import run_records as rr
        with method("cython"), mock.patch.dict(os.environ, JIEQI_LEGAL_BACKEND="cython"):
            backend, error, record = jq._load_legal_backend()
            with mock.patch.multiple(jq, LEGAL_BACKEND=backend, LEGAL_BACKEND_ERROR=error,
                                     LEGAL_BACKEND_NATIVE=record), tempfile.TemporaryDirectory() as tmp:
                manifest = rr.create_run(tmp, "probe", {}, None, {}, {}, "cpu")
                self.assertEqual(manifest["runtime"]["legal_backend_native"], record)
                key = Path(record["path"]).relative_to(ROOT).as_posix()
                self.assertEqual(manifest["code"]["files"][key], record["sha256"])
                bad = dict(record, path=str(ROOT / "jieqi.py"))
                with mock.patch.object(jq, "LEGAL_BACKEND_NATIVE", bad):
                    with self.assertRaisesRegex(RuntimeError, "loaded native legal file changed"):
                        rr.create_run(Path(tmp)/"not_created", "probe", {}, None, {}, {}, "cpu")
                self.assertFalse((Path(tmp)/"not_created").exists())

    def test_external_preloaded_binary_and_cli(self):
        # A PE overlay gives the external copy a distinct, still-loadable identity.
        with tempfile.TemporaryDirectory() as tmp:
            external = Path(tmp) / Path(native.__file__).name
            external.write_bytes(Path(native.__file__).read_bytes() + b"LEGAL_PROVENANCE_TEST")
            code = ("import importlib.util,sys,pathlib; p=pathlib.Path(sys.argv[1]); "
                    "s=importlib.util.spec_from_file_location('_jieqi_legal',p); "
                    "m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); "
                    "import jieqi as jq,run_records as rr; "
                    "assert jq.LEGAL_BACKEND=='cython'; r=jq.LEGAL_BACKEND_NATIVE; "
                    "assert pathlib.Path(r['path'])==p; "
                    "assert rr.code_record()['files'][p.as_posix()]==r['sha256']; "
                    "assert jq.GameState.initial().legal_moves()")
            env = dict(os.environ, JIEQI_LEGAL_BACKEND="cython")
            subprocess.run([sys.executable, "-c", code, str(external)], cwd=ROOT, env=env,
                           check=True, capture_output=True, text=True, timeout=30)
            subprocess.run([sys.executable, "azero.py", "--help"], cwd=ROOT, env=env,
                           check=True, capture_output=True, text=True, timeout=30)

    def test_micro_probe_preserves_fixtures_and_labels_both_modes(self):
        from tools import benchmark_native_legal as benchmark
        from tools import profile_selfplay as probe
        fixtures = probe.controlled_endgames()
        before = probe.digest(fixtures)
        with method("python"), mock.patch.object(probe, "emit"):
            rows = benchmark.micro_probe(fixtures, pairs=2)
        self.assertEqual(probe.digest(fixtures), before)
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["mode"] for row in rows}, {"python", "cython"})
        self.assertEqual({row["kind"] for row in rows}, {"legal_moves", "has_legal_moves", "game_outcome"})
        self.assertTrue(all(row["operations"] == 256 and row["seconds"] >= 0 for row in rows))
        with self.assertRaises(ValueError):
            benchmark.set_mode("invalid")


if __name__ == "__main__":
    unittest.main()
