"""Experimental synchronous inference transport; never combines requests."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import multiprocessing as mp
import queue
import time
from unittest import mock

import numpy as np
import torch

import azero as az


def precision_settings():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic)


def configure_precision(settings):
    torch.backends.cuda.matmul.allow_tf32 = settings["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn_tf32"]
    torch.backends.cudnn.benchmark = settings["benchmark"]
    torch.backends.cudnn.deterministic = settings["deterministic"]


def views(slot, max_batch):
    inputs, policies, values = slot
    return (np.frombuffer(inputs, dtype=np.float32).reshape(max_batch, az.STATE_CHANNELS, 10, 9),
            np.frombuffer(policies, dtype=np.float32).reshape(max_batch, az.MOVE_SIZE),
            np.frombuffer(values, dtype=np.float32))


def validate_request(request, generation, slots, max_batch, sequences):
    slot, sequence, n, role, version, submitted = request
    if (type(slot) is not int or not 0 <= slot < slots or type(sequence) is not int
            or sequence != sequences[slot] + 1 or type(n) is not int or not 1 <= n <= max_batch
            or role not in ("root", "leaf") or (role == "root" and n != 1)
            or version != generation or not isinstance(submitted, (int, float))):
        raise ValueError("invalid or stale inference request")
    return slot, sequence, n, role, version, submitted


def serve(weights, architecture, device_str, settings, generation, slots, max_batch,
          requests, replies, status, stopped):
    try:
        torch.set_num_threads(1)
        configure_precision(settings)
        device = torch.device(device_str)
        net = az.PolicyValueNet(**architecture).to(device).eval()
        net.load_state_dict(weights)
        buffers = [views(slot, max_batch) for slot in slots]
        with torch.no_grad():
            for n in sorted({1, max_batch}):
                logits, values = net(torch.zeros((n, az.STATE_CHANNELS, 10, 9), device=device))
                torch.softmax(logits, dim=1).cpu()
                values.tolist()
        status.put(("ready", None))
        sequences = [0] * len(slots)
        calls = rows = 0
        evaluation_seconds = 0.0
        while not stopped.is_set():
            try:
                request = requests.get(timeout=0.2)
            except queue.Empty:
                continue
            if request is None:
                status.put(("complete", dict(calls=calls, rows=rows, evaluation_seconds=evaluation_seconds)))
                return
            slot, sequence, n, role, version, submitted = validate_request(
                request, generation, len(slots), max_batch, sequences)
            inputs, policy_out, value_out = buffers[slot]
            queue_seconds = time.perf_counter() - submitted
            start = time.perf_counter()
            # Keep each original batch intact, including softmax before CPU transfer.
            with torch.no_grad():
                logits, values = net(torch.from_numpy(inputs[:n]).to(device))
                policy_out[:n] = torch.softmax(logits, dim=1).cpu().numpy()
                value_out[:n] = values.cpu().numpy()
            seconds = time.perf_counter() - start
            sequences[slot] = sequence
            calls += 1
            rows += n
            evaluation_seconds += seconds
            replies[slot].put((sequence, n, role, version, queue_seconds, seconds), timeout=2.0)
    except BaseException as exc:
        status.put(("error", repr(exc)))
        raise
    finally:
        stopped.set()


class InferenceService:
    def __init__(self, weights, architecture, device="cuda", workers=8, max_batch=64,
                 generation="frozen", timeout=60.0, settings=None):
        if workers < 1 or max_batch < 1 or timeout <= 0:
            raise ValueError("positive worker count, batch capacity and timeout required")
        self.context = mp.get_context("spawn")
        self.timeout = timeout
        self.workers, self.max_batch, self.generation = workers, max_batch, generation
        self.requests = self.context.Queue(maxsize=workers)
        self.replies = [self.context.Queue(maxsize=1) for _ in range(workers)]
        self.status = self.context.Queue()
        self.available = self.context.Queue(maxsize=workers)
        self.stopped = self.context.Event()
        self.slots = [(self.context.RawArray("f", max_batch*az.STATE_CHANNELS*90),
                       self.context.RawArray("f", max_batch*az.MOVE_SIZE),
                       self.context.RawArray("f", max_batch)) for _ in range(workers)]
        for i in range(workers):
            self.available.put(i)
        self.process = self.context.Process(target=serve,
            args=(weights, architecture, device, settings or precision_settings(), generation, self.slots,
                  max_batch, self.requests, self.replies, self.status, self.stopped), daemon=True)
        self.closed = False
        self.report = None

    def start(self):
        try:
            self.process.start()
            kind, detail = self.status.get(timeout=self.timeout)
            if kind != "ready":
                raise RuntimeError(f"inference startup failed: {detail}")
            return self
        except BaseException:
            self.close(abort=True)
            raise

    def client_args(self):
        return (self.slots, self.max_batch, self.generation, self.requests, self.replies,
                self.available, self.stopped, self.timeout)

    def close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.pid is not None and self.process.is_alive():
                if abort:
                    self.stopped.set()
                else:
                    self.requests.put(None, timeout=2.0)
                self.process.join(timeout=10.0)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=10.0)
                    if not abort:
                        raise RuntimeError("inference service did not shut down cleanly")
            if not abort and self.process.pid is not None:
                try:
                    kind, detail = self.status.get(timeout=2.0)
                except queue.Empty as exc:
                    raise RuntimeError("inference service exited without completion") from exc
                if kind != "complete" or self.process.exitcode != 0:
                    raise RuntimeError(f"inference service failed: {detail}")
                self.report = detail
        finally:
            for q in [self.requests, self.status, self.available, *self.replies]:
                q.cancel_join_thread()
                q.close()


class InferenceClient:
    def __init__(self, slots, max_batch, generation, requests, replies, available, stopped, timeout):
        self.slot = available.get(timeout=timeout)
        self.buffers = views(slots[self.slot], max_batch)
        self.max_batch, self.generation = max_batch, generation
        self.requests, self.reply, self.stopped = requests, replies[self.slot], stopped
        self.timeout, self.sequence, self.failed = timeout, 0, False
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = dict(calls=0, rows=0, input_copy_seconds=0.0,
                            result_copy_seconds=0.0, queue_seconds=0.0, evaluation_seconds=0.0,
                            latencies=[])

    def evaluate(self, batch, role):
        if self.failed or self.stopped.is_set():
            raise RuntimeError("inference service unavailable")
        if (batch.dtype != np.float32 or batch.shape[1:] != (az.STATE_CHANNELS, 10, 9)
                or not 1 <= len(batch) <= self.max_batch or role not in ("root", "leaf")
                or (role == "root" and len(batch) != 1)):
            raise ValueError("invalid inference batch")
        start = time.perf_counter()
        n = len(batch)
        inputs, policy_out, value_out = self.buffers
        inputs[:n] = batch
        self.metrics["input_copy_seconds"] += time.perf_counter() - start
        self.sequence += 1
        request = (self.slot, self.sequence, n, role, self.generation, time.perf_counter())
        try:
            self.requests.put(request, timeout=self.timeout)
            deadline = time.perf_counter() + self.timeout
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0 or self.stopped.is_set():
                    raise RuntimeError("inference service stopped or request timed out")
                try:
                    response = self.reply.get(timeout=min(0.2, remaining))
                    break
                except queue.Empty:
                    continue
            if tuple(response[:4]) != request[1:5]:
                raise RuntimeError("stale or mismatched inference response")
            copy_start = time.perf_counter()
            policies, values = policy_out[:n].copy(), value_out[:n].tolist()
            policies.setflags(write=False)
            self.metrics["result_copy_seconds"] += time.perf_counter() - copy_start
            self.metrics["calls"] += 1
            self.metrics["rows"] += n
            self.metrics["queue_seconds"] += response[4]
            self.metrics["evaluation_seconds"] += response[5]
            self.metrics["latencies"].append(time.perf_counter() - start)
            return policies, values
        except BaseException:
            # A timed-out slot cannot be reused while the server might still read it.
            self.failed = True
            self.stopped.set()
            raise

    @contextmanager
    def install(self):
        def root(mcts, info, history):
            batch = az.encode_state_numpy(info, history)[None]
            policies, values = self.evaluate(batch, "root")
            return torch.from_numpy(policies[0].copy()), values[0]

        def leaf(mcts, infos, histories, *, repetition_entries=None):
            batch = az._encode_batch_numpy(infos, histories, repetition_entries=repetition_entries)
            return self.evaluate(batch, "leaf")

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(az.NeuralMCTS, "_evaluate", root))
            stack.enter_context(mock.patch.object(az.NeuralMCTS, "_evaluate_batch", leaf))
            yield self
