"""Transport must preserve batches, output ownership and failure boundaries."""
import hashlib
import multiprocessing as mp
import queue
import random
import time
import unittest

import numpy as np
import torch
import azero as az
import ismcts as im
import jieqi as jq
from tools import central_inference as central
from tools import benchmark_central_inference as bench
from tools import profile_selfplay as probe
from tools import probe_merged_inference as merged


def output_digest(policies, values):
    digest = hashlib.sha256(policies.tobytes())
    digest.update(np.asarray(values, dtype=np.float64).tobytes())
    return digest.hexdigest()


def concurrent_client(args, seed, result):
    try:
        client = central.InferenceClient(*args)
        batch = np.random.default_rng(seed).standard_normal((3, az.STATE_CHANNELS, 10, 9)).astype(np.float32)
        hashes = [output_digest(*client.evaluate(batch, "leaf")),
                  output_digest(*client.evaluate(batch[:1], "root"))]
        result.send((seed, hashes, client.metrics["calls"], client.metrics["rows"]))
    finally:
        result.close()


class TestCentralInference(unittest.TestCase):
    def setUp(self):
        self.threads, self.rng = torch.get_num_threads(), torch.get_rng_state()
        torch.set_num_threads(1)
        torch.manual_seed(75)
        self.net = az.PolicyValueNet(channels=8, blocks=1).eval()
        self.service = None

    def tearDown(self):
        if self.service is not None:
            self.service.close(abort=not self.service.closed)
        torch.set_rng_state(self.rng)
        torch.set_num_threads(self.threads)

    def start(self, workers=1):
        self.service = central.InferenceService(self.net.state_dict(), dict(channels=8, blocks=1),
            device="cpu", workers=workers, max_batch=8, timeout=30.0, generation="model75").start()
        return central.InferenceClient(*self.service.client_args())

    def reference(self, batch):
        with torch.no_grad():
            logits, values = self.net(torch.from_numpy(batch))
            return torch.softmax(logits, dim=1).numpy(), values.tolist()

    def test_roundtrip_preserves_shape_bits_python_values_and_output_ownership(self):
        client = self.start()
        batch = np.random.default_rng(41).standard_normal((5, az.STATE_CHANNELS, 10, 9)).astype(np.float32)
        before = batch.copy()
        policies, values = client.evaluate(batch, "leaf")
        self.assertEqual(output_digest(policies, values), output_digest(*self.reference(batch)))
        np.testing.assert_array_equal(batch, before)
        self.assertFalse(policies.flags.writeable)
        self.assertTrue(all(type(v) is float for v in values))
        old = policies.copy()
        client.evaluate(batch[:1], "root")
        np.testing.assert_array_equal(policies, old)
        self.service.close()
        self.assertEqual(self.service.report["calls"], 2)
        self.assertEqual(self.service.report["rows"], 6)
        self.assertEqual(self.service.process.exitcode, 0)

    def test_search_inputs_outputs_graph_rng_and_model_are_exact(self):
        client = self.start()
        state = jq.GameState.initial(rng=random.Random(27))
        belief = im.init_belief_for_game(state)
        config = az.AZConfig(mcts_simulations=64, leaf_batch_size=8, lazy_chance_allocation=True)
        before = probe.digest((state, belief, config, self.net.state_dict()))
        rows = []
        for remote in (False, True):
            mcts = az.NeuralMCTS(None if remote else self.net, config, torch.device("cpu"), random.Random(31))
            trace = bench.EvaluationTrace(True)
            with client.install() if remote else central.ExitStack():
                with trace.install():
                    policy, move, root = mcts.search(state, belief, [], 1.0, return_root=True,
                        add_dirichlet=True, prune_policy=True)
            rows.append((policy, move, probe.digest(root), mcts.rng.getstate(),
                         trace.inputs.hexdigest(), trace.outputs.hexdigest(),
                         mcts.leaf_evals_total, mcts.leaf_evals_unique))
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(probe.digest((state, belief, config, self.net.state_dict())), before)

    def test_independent_concurrent_slots_do_not_mix_results_or_merge_batches(self):
        self.start(workers=4)
        # The parent owns one slot; three spawned clients own the remaining slots.
        context = mp.get_context("spawn")
        processes, pipes = [], []
        try:
            for seed in range(3):
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(target=concurrent_client,
                    args=(self.service.client_args(), seed, writer))
                process.start()
                writer.close()
                processes.append(process)
                pipes.append(reader)
            for seed, reader in enumerate(pipes):
                self.assertTrue(reader.poll(30))
                result = reader.recv()
                batch = np.random.default_rng(seed).standard_normal((3, az.STATE_CHANNELS, 10, 9)).astype(np.float32)
                expected = [output_digest(*self.reference(batch)), output_digest(*self.reference(batch[:1]))]
                self.assertEqual(result, (seed, expected, 2, 4))
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            self.service.close()
            self.assertEqual(self.service.report["calls"], 6)
            self.assertEqual(self.service.report["rows"], 12)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(10)
            for reader in pipes:
                reader.close()

    def test_patch_restored_after_exception(self):
        client = self.start()
        original = (az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with client.install():
                raise RuntimeError("intentional")
        self.assertEqual(original, (az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch))

    def test_service_model_snapshot_unchanged_after_multiple_requests(self):
        client = self.start()
        before = probe.digest(self.net.state_dict())
        batch = np.zeros((2, az.STATE_CHANNELS, 10, 9), dtype=np.float32)
        for _ in range(3):
            self.assertEqual(output_digest(*client.evaluate(batch, "leaf")), output_digest(*self.reference(batch)))
        self.assertEqual(probe.digest(self.net.state_dict()), before)

    def test_startup_error_is_propagated_and_child_is_joined(self):
        self.service = central.InferenceService({}, dict(channels=8, blocks=1), device="cpu", workers=1,
            max_batch=8, timeout=30)
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            self.service.start()
        self.assertFalse(self.service.process.is_alive())

    def test_dead_service_fails_with_bounded_wait_and_poisons_client(self):
        client = self.start()
        client.timeout = 0.05
        self.service.process.terminate()
        self.service.process.join(10)
        batch = np.zeros((1, az.STATE_CHANNELS, 10, 9), dtype=np.float32)
        start = time.perf_counter()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            client.evaluate(batch, "root")
        self.assertLess(time.perf_counter()-start, 2.0)
        self.assertTrue(client.failed)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            client.evaluate(batch+1, "root")


class TestInferenceProtocol(unittest.TestCase):
    def test_rejects_wrong_model_sequence_shape_role_and_slot(self):
        good = (0, 1, 4, "leaf", "model", 1.0)
        self.assertEqual(central.validate_request(good, "model", 2, 8, [0, 0]), good)
        bad = [(0, 0, 4, "leaf", "model", 1.0), (0, 1, 4, "leaf", "old", 1.0),
               (2, 1, 4, "leaf", "model", 1.0), (0, 1, 9, "leaf", "model", 1.0),
               (0, 1, 4, "root", "model", 1.0), (0, 1, 4, "unknown", "model", 1.0)]
        for request in bad:
            with self.subTest(request=request), self.assertRaises(ValueError):
                central.validate_request(request, "model", 2, 8, [0, 0])

    def fake_client(self):
        context = mp.get_context("spawn")
        slots = [(context.RawArray("f", 2*az.STATE_CHANNELS*90),
                  context.RawArray("f", 2*az.MOVE_SIZE), context.RawArray("f", 2))]
        available, requests, reply = queue.Queue(), queue.Queue(), queue.Queue()
        available.put(0)
        return central.InferenceClient(slots, 2, "model", requests, [reply], available,
                                       context.Event(), 0.01)

    def test_stale_response_aborts_instead_of_consuming_other_results(self):
        client = self.fake_client()
        client.reply.put((2, 1, "root", "model", 0.0, 0.0))
        with self.assertRaisesRegex(RuntimeError, "mismatched"):
            client.evaluate(np.zeros((1, az.STATE_CHANNELS, 10, 9), dtype=np.float32), "root")
        self.assertTrue(client.stopped.is_set())
        self.assertEqual(client.metrics["calls"], 0)

    def test_invalid_batch_not_submitted_and_oversize_not_truncated(self):
        client = self.fake_client()
        for batch, role in [(np.zeros((3, az.STATE_CHANNELS, 10, 9), dtype=np.float32), "leaf"),
                            (np.zeros((1, az.STATE_CHANNELS, 10, 9), dtype=np.float64), "root"),
                            (np.zeros((2, az.STATE_CHANNELS, 10, 9), dtype=np.float32), "root")]:
            with self.subTest(shape=batch.shape, role=role), self.assertRaises(ValueError):
                client.evaluate(batch, role)
        self.assertTrue(client.requests.empty())
        self.assertEqual(client.sequence, 0)


class TestOfflineMergeProbe(unittest.TestCase):
    def test_splits_original_requests_in_order_without_mutating_them(self):
        class Net(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((len(x), 2)), x[:, 0, 0, 0]

        batches = [np.full((n, 1, 1, 1), i, dtype=np.float32) for i, n in enumerate((1, 2, 3))]
        before = [batch.copy() for batch in batches]
        rows = merged.compare_group(Net(), batches, torch.device("cpu"))
        self.assertEqual([row["offset"] for row in rows], [0, 1, 3])
        self.assertTrue(all(row["merged_batch"] == 6 for row in rows))
        self.assertTrue(all(row["policy_equal"] and row["value_equal"] for row in rows))
        for batch, old in zip(batches, before):
            np.testing.assert_array_equal(batch, old)

    def test_batch_sensitive_outputs_are_reported_not_used_for_search(self):
        class Net(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((len(x), 2)), x[:, 0, 0, 0] + len(x)/1000

        batches = [np.zeros((n, 1, 1, 1), dtype=np.float32) for n in (1, 2)]
        rows = merged.compare_group(Net(), batches, torch.device("cpu"))
        self.assertTrue(all(row["policy_equal"] for row in rows))
        self.assertTrue(all(not row["value_equal"] and row["value_max_abs_diff"] > 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
