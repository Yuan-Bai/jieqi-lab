"""Exact node methods, loader fallback, context isolation and native provenance."""
from contextlib import contextmanager
import copy
import hashlib
import json
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

import numpy as np
import torch

import azero as az
import ismcts as im
import run_records as rr
from test_selfplay_acceleration import tree_records

try:
    import _jieqi_search as native
except ImportError:
    native = None
CURRENT = bool(native and native.API_VERSION == 1 and native.SEARCH_VERSION == az.SEARCH_VERSION
               and native.SOURCE_SHA256 == hashlib.sha256(
                   (Path(az.__file__).parent / "native/jieqi_search.pyx").read_bytes()).hexdigest()
               and native.METHODS_SHA256 == az._node_methods_sha256())


@contextmanager
def methods(mode):
    init = native.initialize_node if mode in ("init", "both") else az.PUCTNode._init_python
    expand = native.expand if mode in ("expand", "both") else az.PUCTNode._expand_python
    with mock.patch.object(az.PUCTNode, "__init__", init), mock.patch.object(az.PUCTNode, "expand", expand), \
            mock.patch.object(az.PUCTNode, "_native_search_module", az, create=True):
        yield


class TestSearchLoader(unittest.TestCase):
    def setUp(self):
        self.init, self.expand = az.PUCTNode.__init__, az.PUCTNode.expand

    def tearDown(self):
        az.PUCTNode.__init__, az.PUCTNode.expand = self.init, self.expand

    def test_explicit_python_restores_methods_and_invalid_mode_fails(self):
        with mock.patch.dict(os.environ, JIEQI_SEARCH_BACKEND="python"):
            self.assertEqual(az._load_search_backend(), ("python", None, None))
            self.assertIs(az.PUCTNode.__init__, az.PUCTNode._init_python)
            self.assertIs(az.PUCTNode.expand, az.PUCTNode._expand_python)
        with mock.patch.dict(os.environ, JIEQI_SEARCH_BACKEND="typo"):
            with self.assertRaises(ValueError):
                az._load_search_backend()

    def test_missing_extension_falls_back_or_fails_explicitly(self):
        with mock.patch.dict(sys.modules, {"_jieqi_search": None}):
            with mock.patch.dict(os.environ, JIEQI_SEARCH_BACKEND="auto"):
                backend, error, record = az._load_search_backend()
                self.assertEqual(backend, "python")
                self.assertIsNotNone(error)
                self.assertIsNone(record)
                self.assertIs(az.PUCTNode.expand, az.PUCTNode._expand_python)
            with mock.patch.dict(os.environ, JIEQI_SEARCH_BACKEND="cython"):
                with self.assertRaisesRegex(ImportError, "unavailable"):
                    az._load_search_backend()

    def test_version_source_and_actual_python_methods_are_checked(self):
        fields = dict(API_VERSION=1, SEARCH_VERSION=az.SEARCH_VERSION,
                      SOURCE_SHA256=hashlib.sha256((Path(az.__file__).parent / "native/jieqi_search.pyx").read_bytes()).hexdigest(),
                      METHODS_SHA256=az._node_methods_sha256())
        for field in fields:
            fake = types.SimpleNamespace(**dict(fields, **{field: "stale"}))
            with mock.patch.dict(sys.modules, {"_jieqi_search": fake}), mock.patch.dict(
                    os.environ, JIEQI_SEARCH_BACKEND="auto"):
                backend, error, record = az._load_search_backend()
                self.assertEqual(backend, "python")
                self.assertIn("rebuild", error)
                self.assertIsNone(record)
        before = az._node_methods_sha256()
        with mock.patch.object(az.PUCTNode, "_expand_python", lambda self, *args, **kwargs: None):
            self.assertNotEqual(az._node_methods_sha256(), before)


@unittest.skipUnless(CURRENT, "build current extension with python tools/build_search.py")
class TestNativeSearch(unittest.TestCase):
    def compare_expansion(self, moves, priors, specs=None, k=0, side=1, rng_enabled=True):
        before = pickle.dumps((moves, priors, specs))
        outputs = []
        for mode in ("python", "init", "expand", "both"):
            rng = random.Random(71)
            with methods(mode):
                root = az.PUCTNode(0.125, side)
                root.children[(-1, -1)] = az.PUCTNode(0.5, side, move=(-1, -1))
                root.children[(-1, -1)].visits = 19
                root.public_outcomes[1] = root.children[(-1, -1)]
                root.decision_nodes[("existing",)] = root.children[(-1, -1)]
                error = None
                try:
                    root.expand(moves, priors, specs, k, rng if rng_enabled else None)
                except Exception as exc:
                    error = type(exc)
                outputs.append((pickle.dumps(root), rng.getstate(), error))
            self.assertEqual(pickle.dumps((moves, priors, specs)), before)
        self.assertTrue(all(row == outputs[0] for row in outputs))
        return outputs[0][2]

    def test_initializer_retains_objects_slot_order_and_fresh_dictionaries(self):
        prior, player, move, chance = object(), object(), object(), object()
        rows = []
        for mode in ("python", "init", "expand", "both"):
            with methods(mode):
                first = az.PUCTNode(prior, player, move, chance)
                second = az.PUCTNode(prior, player)
            self.assertIs(first.prior, prior)
            self.assertIs(first.player_to_move, player)
            self.assertIs(first.move, move)
            self.assertIs(first.is_chance, chance)
            for name in ("children", "public_outcomes", "decision_nodes"):
                self.assertIsNot(getattr(first, name), getattr(second, name))
                self.assertIs(type(getattr(first, name)), dict)
            rows.append([(name, getattr(first, name)) for name in az.PUCTNode.__slots__])
        self.assertTrue(all(row == rows[0] for row in rows))

        class Observed(az.PUCTNode):
            def __setattr__(self, name, value):
                assignments.append(name)
                super().__setattr__(name, value)
        recorded = []
        for fn in (az.PUCTNode._init_python, native.initialize_node):
            assignments = []
            fn(Observed.__new__(Observed), 0.25, 1)
            recorded.append(assignments)
        self.assertEqual(recorded[0], recorded[1])

    def test_priors_types_rounding_nonfinite_unbounded_side_and_zero_fallback(self):
        moves = az.jq.GameState.initial(rng=random.Random(7)).legal_moves()[:9]
        rng = np.random.default_rng(79)
        values = rng.random(az.MOVE_SIZE)
        variants = [values.tolist(), values.astype(np.float32), values, values.astype(np.float16),
                    torch.from_numpy(values.astype(np.float32)), np.zeros(az.MOVE_SIZE),
                    -values, np.full(az.MOVE_SIZE, 5e-324), np.full(az.MOVE_SIZE, float("inf")),
                    np.full(az.MOVE_SIZE, float("nan")), np.full(az.MOVE_SIZE, -0.0)]
        for priors in variants:
            for side in (0, 1, 1 << 100):
                with self.subTest(dtype=getattr(priors, "dtype", "list"), side=side):
                    self.assertIsNone(self.compare_expansion(moves, priors, side=side))

    def test_chance_cartesian_support_order_rng_and_count_snapshots(self):
        moves = az.jq.GameState.initial(rng=random.Random(11)).legal_moves()[:4]
        counts = [0, 0, 2, 2, 2, 2, 2, 5]
        huge = [0, 0, 1 << 100, 1, 0, 0, 0, 1]
        specs = {moves[0]: dict(moved_counts=counts, capture_counts=counts),
                 moves[1]: dict(moved_counts=None, capture_counts=huge),
                 moves[2]: dict(moved_counts=huge, capture_counts=None),
                 moves[3]: dict(moved_counts=[0]*8, capture_counts=counts)}
        for k in (-1, 0, 1, 2, 6, 99):
            for rng_enabled in (False, True):
                self.assertIsNone(self.compare_expansion(moves, np.ones(az.MOVE_SIZE), specs, k,
                                                        rng_enabled=rng_enabled))
        # Mutating original counts after expansion must not alter saved snapshots.
        with methods("both"):
            root = az.PUCTNode(1.0, 1)
            root.expand(moves, np.ones(az.MOVE_SIZE), specs, 0, random.Random(13))
        before = pickle.dumps(root)
        counts[2] += 1
        self.assertEqual(pickle.dumps(root), before)

    def test_empty_moves_errors_and_partial_updates_match(self):
        self.assertIsNone(self.compare_expansion([], None))
        for move in ((0, 0), (1 << 100, 0), (0,), (0.5, 1)):
            self.assertIsNotNone(self.compare_expansion([move], [1.0]*az.MOVE_SIZE))
        moves = az.jq.GameState.initial().legal_moves()[:2]
        specs = {moves[1]: dict(moved_counts=[0, 0, 1, 0, 0, 0, 0, 0])}
        self.assertIs(self.compare_expansion(moves, [1.0]*az.MOVE_SIZE, specs), KeyError)

    def test_python_callbacks_and_factory_are_not_bypassed(self):
        node_type = az.PUCTNode
        moves = az.jq.GameState.initial().legal_moves()[:2]
        specs = {m: dict(moved_counts=[0, 0, 1, 1, 0, 0, 0, 0], capture_counts=None) for m in moves}
        results = []
        original_sample = az.sample_reveal_outcomes
        for mode in ("python", "both"):
            events = []
            class Priors:
                def __getitem__(self, index):
                    events.append(("prior", index))
                    return 0.25
            def factory(*args, **kwargs):
                events.append(("node", args, kwargs))
                return node_type(*args, **kwargs)
            def sample(*args):
                events.append(("sample", copy.deepcopy(args[:-1])))
                return original_sample(*args)
            with methods(mode):
                root = node_type(1.0, 1)
                with mock.patch.object(az, "PUCTNode", factory), mock.patch.object(az, "sample_reveal_outcomes", sample):
                    root.expand(moves, Priors(), specs, 1, random.Random(17))
            results.append((events, tree_records(root)))
        self.assertEqual(results[0], results[1])

    def test_separate_module_context_and_pickle_are_preserved(self):
        from tools.benchmark_encoding import load_baseline
        old = sys.modules.get("encoding_baseline")
        try:
            with mock.patch.dict(os.environ, JIEQI_SEARCH_BACKEND="cython"):
                clone = load_baseline(Path(az.__file__).read_text(encoding="utf-8-sig"))
            self.assertEqual(clone.SEARCH_BACKEND, "cython", clone.SEARCH_BACKEND_ERROR)
            moves = az.jq.GameState.initial().legal_moves()[:2]
            for module in (az, clone, az):
                root = module.PUCTNode(1.0, 1)
                root.expand(moves, np.ones(az.MOVE_SIZE))
                self.assertTrue(all(type(node) is module.PUCTNode for node in root.children.values()))
                loaded = pickle.loads(pickle.dumps(root))
                self.assertEqual(tree_records(root), tree_records(loaded))
        finally:
            if old is None:
                sys.modules.pop("encoding_baseline", None)
            else:
                sys.modules["encoding_baseline"] = old

    def test_full_search_graph_policy_rng_and_input_state_match(self):
        rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            torch.manual_seed(89)
            net = az.PolicyValueNet(channels=8, blocks=1).eval()
            state = az.jq.GameState.initial(rng=random.Random(19))
            belief, history = im.init_belief_for_game(state), []
            for ply in (0, 12):
                if ply:
                    env = random.Random(97)
                    for _ in range(ply):
                        legal = state.legal_moves()
                        captures = [m for m in legal if state.board[m[1]]]
                        move = env.choice(captures or legal)
                        im.apply_move_and_update_belief(state, belief, move, rng=env)
                        history.append(move)
                before = pickle.dumps((state, belief, history))
                for chance, k in ((False, 0), (True, 0), (True, 1)):
                    for dedup in (False, True):
                        config = az.AZConfig(mcts_simulations=48, leaf_batch_size=8,
                                             chance_nodes=chance, chance_k=k, leaf_dedup=dedup)
                        results = []
                        for mode in ("python", "init", "expand", "both"):
                            search = az.NeuralMCTS(net, config, torch.device("cpu"), random.Random(101))
                            with methods(mode):
                                policy, move, root = search.search(state, belief, history, 1.0,
                                    add_dirichlet=True, prune_policy=True, return_root=True)
                            results.append((np.asarray(policy, dtype=np.float64).tobytes(), move,
                                            tree_records(root), search.rng.getstate(), search.leaf_evals_total,
                                            search.leaf_evals_unique, search.chance_applied,
                                            search.chance_outcomes_added, search.chance_nav_miss))
                            self.assertEqual(pickle.dumps((state, belief, history)), before)
                        self.assertTrue(all(row == results[0] for row in results))
        finally:
            torch.set_rng_state(rng)
            torch.set_num_threads(threads)

    def test_cli_import_and_actual_native_provenance(self):
        result = subprocess.run([sys.executable, "azero.py", "--help"], cwd=rr.PROJECT_ROOT,
                                env=dict(os.environ, JIEQI_SEARCH_BACKEND="cython"), capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        with tempfile.TemporaryDirectory() as directory:
            record = rr.create_run(directory, "training", {}, None, {}, {}, "cpu")
            loaded = az.SEARCH_BACKEND_NATIVE
            self.assertEqual(record["runtime"]["search_backend_native"], loaded)
            if loaded:
                binary = Path(loaded["path"])
                self.assertEqual(record["code"]["files"][binary.relative_to(rr.PROJECT_ROOT).as_posix()], loaded["sha256"])
                self.assertEqual(loaded["methods"], ["expand"])
                self.assertIs(az.PUCTNode.__init__, az.PUCTNode._init_python)

    def test_changed_native_file_is_rejected_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "loaded.pyd"
            binary.write_bytes(b"loaded snapshot")
            loaded = rr.file_record(binary)
            binary.write_bytes(b"changed file")
            with mock.patch.object(az, "SEARCH_BACKEND_NATIVE", loaded):
                with self.assertRaisesRegex(RuntimeError, "native search file changed"):
                    rr.create_run(root, "training", {}, None, {}, {}, "cpu")
            self.assertFalse((root / "runs").exists())

    def test_external_or_preloaded_native_binary_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external"
            external.mkdir()
            binary = external / Path(native.__file__).name
            content = Path(native.__file__).read_bytes()
            if sys.platform == "win32":
                content += b"search-provenance-regression-overlay"
            binary.write_bytes(content)
            code = """
import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
if sys.argv[3] == 'preload':
    import _jieqi_search
import azero as az, run_records as rr
assert az.SEARCH_BACKEND == 'cython', az.SEARCH_BACKEND_ERROR
manifest = rr.create_run(sys.argv[2], 'training', {}, None, {}, {}, 'cpu')
loaded = manifest['runtime']['search_backend_native']
assert Path(loaded['path']).parent == Path(sys.argv[1])
assert manifest['code']['files'][Path(loaded['path']).as_posix()] == loaded['sha256']
print(json.dumps(manifest))
"""
            for mode in ("import", "preload"):
                result = subprocess.run([sys.executable, "-c", code, str(external), str(root / mode), mode],
                    cwd=rr.PROJECT_ROOT, env=dict(os.environ, JIEQI_SEARCH_BACKEND="cython"),
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                loaded = json.loads(result.stdout)["runtime"]["search_backend_native"]
                self.assertEqual({key: loaded[key] for key in rr.file_record(binary)}, rr.file_record(binary))


if __name__ == "__main__":
    unittest.main()
