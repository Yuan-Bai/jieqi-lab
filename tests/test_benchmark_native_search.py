"""Node benchmarks must isolate and record the legal backend before any work."""
from contextlib import ExitStack, redirect_stderr
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from tools import benchmark_native_search as benchmark


class TestNodeBenchmarkIsolation(unittest.TestCase):
    def test_compiled_legal_backend_rejected_before_checkpoint_or_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", ["benchmark", "--output", str(output)]), \
                    mock.patch.object(benchmark.jq, "LEGAL_BACKEND", "cython"), \
                    mock.patch.object(benchmark.torch, "load") as load, \
                    mock.patch.object(benchmark.probe, "collect_fixtures") as collect, \
                    mock.patch.object(benchmark.rr, "file_record") as record, redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as error:
                    benchmark.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("JIEQI_LEGAL_BACKEND=python", stderr.getvalue())
            load.assert_not_called()
            collect.assert_not_called()
            record.assert_not_called()
            self.assertFalse(output.exists())

    def test_python_run_records_legal_identity_and_protects_legal_source(self):
        config = benchmark.az.AZConfig()
        checkpoint = dict(config=asdict(config), model={}, train_state=dict(iteration=15),
                          replay_files=[dict(name="replay.pkl.gz")])
        fixtures = [dict(metadata=dict(index=0))]
        searches = [dict(pair=0, mode="python", wall_seconds=2.0),
                    dict(pair=0, mode="expand", wall_seconds=1.0)]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            output = Path(tmp) / "result.json"
            stack.enter_context(mock.patch.object(sys, "argv", ["benchmark", "--output", str(output),
                                                               "--pairs", "1", "--complete-games", "0"]))
            stack.enter_context(mock.patch.multiple(benchmark.jq, LEGAL_BACKEND="python",
                LEGAL_BACKEND_ERROR=None, LEGAL_BACKEND_NATIVE=None,
                RULES_BACKEND_NATIVE=dict(path=str(Path(tmp)/"rules.pyd"))))
            stack.enter_context(mock.patch.object(benchmark.az, "SEARCH_BACKEND_NATIVE",
                                                  dict(path=str(Path(tmp)/"search.pyd"))))
            records = stack.enter_context(mock.patch.object(benchmark.rr, "file_record",
                                                             return_value=dict(sha256="fixed")))
            stack.enter_context(mock.patch.object(benchmark.torch.cuda, "is_available", return_value=True))
            stack.enter_context(mock.patch.object(benchmark.torch.cuda, "get_device_name", return_value="fake"))
            stack.enter_context(mock.patch.object(benchmark.torch.cuda, "empty_cache"))
            stack.enter_context(mock.patch.object(benchmark.torch, "set_num_threads"))
            stack.enter_context(mock.patch.object(benchmark.torch, "load", return_value=checkpoint))
            stack.enter_context(mock.patch.object(benchmark.az, "PolicyValueNet"))
            stack.enter_context(mock.patch.object(benchmark, "set_mode"))
            stack.enter_context(mock.patch.object(benchmark.probe, "collect_fixtures",
                                                  return_value=(fixtures, [])))
            for name, value in (("micro_probe", []), ("full_graph_checks", []),
                                ("parallel_probe", searches), ("complete_games", [])):
                stack.enter_context(mock.patch.object(benchmark, name, return_value=value))
            stack.enter_context(mock.patch.object(benchmark.probe, "emit"))
            benchmark.main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["legal_backend"], "python")
            self.assertIsNone(result["legal_backend_error"])
            self.assertIsNone(result["native_legal"])
            source = str(benchmark.ROOT/"native/jieqi_legal.pyx")
            self.assertIn(source, result["source_hashes"])
            self.assertEqual(sum(str(call.args[0]) == source for call in records.call_args_list), 2)

    def test_initializer_forwards_parent_legal_identity(self):
        legal = ("python", None)
        weights, config, rules = {}, {}, dict(path="rules")
        with mock.patch.object(benchmark.probe, "init_worker") as initialize:
            benchmark.init_worker(weights, config, rules, benchmark.az.SEARCH_BACKEND_NATIVE, legal)
        initialize.assert_called_once_with(weights, config, rules, legal_backend=legal)

    def test_initializer_rejects_worker_backend_or_identity_mismatch_before_model(self):
        legal = ("python", None)
        rules = benchmark.jq.RULES_BACKEND_NATIVE
        for mode, record in (("cython", dict(path="other")), ("python", dict(path="other"))):
            with self.subTest(mode=mode), \
                    mock.patch.multiple(benchmark.jq, RULES_BACKEND="cython", LEGAL_BACKEND=mode,
                                        LEGAL_BACKEND_NATIVE=record), \
                    mock.patch.object(benchmark.az, "PolicyValueNet") as factory:
                with self.assertRaises(AssertionError):
                    benchmark.init_worker({}, {}, rules, benchmark.az.SEARCH_BACKEND_NATIVE, legal)
                factory.assert_not_called()

    def test_pool_passes_parent_legal_identity_to_initializer(self):
        weights, config = {}, benchmark.az.AZConfig()
        fixtures = [dict(metadata=dict(index=0))]
        with mock.patch.multiple(benchmark.jq, LEGAL_BACKEND="python", LEGAL_BACKEND_NATIVE=None), \
                mock.patch.object(benchmark.futures, "ProcessPoolExecutor") as executor, \
                mock.patch.object(benchmark.probe, "emit"):
            pool = executor.return_value.__enter__.return_value
            pool.map.side_effect = lambda function, tasks: [dict(wall_seconds=0.1, exact={}) for _ in tasks]
            benchmark.parallel_probe(weights, config, fixtures, 1, 1, 1, "expand")
            executor.assert_called_once_with(max_workers=1, mp_context=mock.ANY,
                initializer=benchmark.init_worker,
                initargs=(weights, asdict(config), benchmark.jq.RULES_BACKEND_NATIVE,
                          benchmark.az.SEARCH_BACKEND_NATIVE, ("python", None)))


if __name__ == "__main__":
    unittest.main()
