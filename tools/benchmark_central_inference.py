"""Read-only paired full-game test of centralized, unmerged GPU inference."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import azero as az
import jieqi as jq
import run_records as rr
from tools import central_inference as central
from tools import profile_selfplay as probe
from tools.measure_search_reuse import verify_hashes

NET = CONFIG = CLIENT = PATCH = None
DEVICE = torch.device("cuda")


class EvaluationTrace:
    def __init__(self, enabled):
        self.enabled = enabled
        self.inputs, self.outputs = hashlib.sha256(), hashlib.sha256()
        self.metrics = dict(root_calls=0, leaf_calls=0, rows=0, evaluation_seconds=0.0, latencies=[])

    @contextmanager
    def install(self):
        root, leaf = az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch
        encode_root, encode_leaf = az.encode_state_numpy, az._encode_batch_numpy

        def record_input(batch, role):
            self.inputs.update(role.encode("ascii") + struct.pack("q", len(batch)))
            self.inputs.update(memoryview(batch).cast("B"))

        def root_encode(*args, **kwargs):
            result = encode_root(*args, **kwargs)
            record_input(result[None], "root")
            return result

        def leaf_encode(*args, **kwargs):
            result = encode_leaf(*args, **kwargs)
            record_input(result, "leaf")
            return result

        def evaluate(function, role, *args, **kwargs):
            start = time.perf_counter()
            result = function(*args, **kwargs)
            seconds = time.perf_counter() - start
            policies, values = result
            if role == "root":
                policies, values = policies.numpy()[None], [values]
            self.metrics[role+"_calls"] += 1
            self.metrics["rows"] += len(values)
            self.metrics["evaluation_seconds"] += seconds
            self.metrics["latencies"].append(seconds)
            if self.enabled:
                self.outputs.update(role.encode("ascii") + struct.pack("q", len(values)))
                self.outputs.update(memoryview(policies).cast("B"))
                self.outputs.update(np.asarray(values, dtype=np.float64).tobytes())
            return result

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(az.NeuralMCTS, "_evaluate",
                lambda *a, **kw: evaluate(root, "root", *a, **kw)))
            stack.enter_context(mock.patch.object(az.NeuralMCTS, "_evaluate_batch",
                lambda *a, **kw: evaluate(leaf, "leaf", *a, **kw)))
            if self.enabled:
                stack.enter_context(mock.patch.object(az, "encode_state_numpy", root_encode))
                stack.enter_context(mock.patch.object(az, "_encode_batch_numpy", leaf_encode))
            yield self


def init_worker(weights, config, architecture, mode, client_args, runtime, settings, ready):
    global NET, CONFIG, CLIENT, PATCH
    if (jq.RULES_BACKEND_NATIVE != runtime["rules"] or jq.RULES_BACKEND != "cython"
            or (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE) != ("python", None)
            or jq.LEGAL_BACKEND_NATIVE != runtime["legal"] or jq.LEGAL_BACKEND != "cython"):
        raise RuntimeError("worker backend mismatch")
    torch.set_num_threads(1)
    central.configure_precision(settings)
    CONFIG = az.AZConfig(**config)
    if mode == "local":
        NET = az.PolicyValueNet(**architecture).to(DEVICE).eval()
        NET.load_state_dict(weights)
        with torch.no_grad():
            for n in sorted({1, CONFIG.leaf_batch_size}):
                logits, values = NET(torch.zeros((n, az.STATE_CHANNELS, 10, 9), device=DEVICE))
                torch.softmax(logits, dim=1).cpu()
                values.tolist()
    elif mode == "central":
        CLIENT = central.InferenceClient(*client_args)
        PATCH = CLIENT.install()
        PATCH.__enter__()
    else:
        raise ValueError("unknown inference mode")
    ready.put(dict(pid=os.getpid(), cuda_initialized=torch.cuda.is_initialized()))


def ready_task():
    return os.getpid()


def game_task(task):
    index, seed, trace_enabled = task
    strong = CONFIG.strong_self_play_interval > 0 and (index+1) % CONFIG.strong_self_play_interval == 0
    rng = random.Random(seed)
    trace = EvaluationTrace(trace_enabled)
    if CLIENT is not None:
        CLIENT.reset_metrics()
    start = time.perf_counter()
    with trace.install():
        game = az.self_play_game(NET, CONFIG, DEVICE, rng, strong=strong)
    seconds = time.perf_counter() - start
    if game.outcome is None:
        raise AssertionError("unfinished game")
    exact = dict(index=index, seed=seed, strong=strong, moves=game.move_count, outcome=game.outcome,
                 samples=len(game.samples), leaf_total=game.leaf_evals_total, leaf_unique=game.leaf_evals_unique,
                 sample_sha256=probe.digest([asdict(s) for s in game.samples]), rng_sha256=probe.digest(rng.getstate()),
                 root_calls=trace.metrics["root_calls"], leaf_calls=trace.metrics["leaf_calls"], rows=trace.metrics["rows"])
    if trace_enabled:
        exact.update(input_sha256=trace.inputs.hexdigest(), output_sha256=trace.outputs.hexdigest())
    return dict(seconds=seconds, exact=exact, evaluation=trace.metrics,
                transport=CLIENT.metrics if CLIENT is not None else None)


def run_group(saved, mode, tasks, workers, timeout, trace, record):
    config = saved["config"]
    architecture = {k: saved["network"][k] for k in ("channels", "blocks")}
    runtime = dict(rules=jq.RULES_BACKEND_NATIVE, legal=jq.LEGAL_BACKEND_NATIVE)
    settings = central.precision_settings()
    context = central.mp.get_context("spawn")
    ready = context.Queue()
    service = None
    executor = None
    startup = time.perf_counter()
    completed = False
    try:
        if mode == "central":
            service = central.InferenceService(saved["model"], architecture, workers=workers,
                max_batch=config["leaf_batch_size"], generation=saved["_inference_generation"],
                timeout=timeout, settings=settings).start()
        executor = futures.ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=init_worker, initargs=(saved["model"] if mode == "local" else None, config,
                architecture, mode, service.client_args() if service else None, runtime, settings, ready))
        warm = [executor.submit(ready_task) for _ in range(workers)]
        deadline = time.perf_counter() + timeout
        identities = [ready.get(timeout=max(0.001, deadline-time.perf_counter())) for _ in range(workers)]
        for future in warm:
            future.result(timeout=timeout)
        if len({row["pid"] for row in identities}) != workers:
            raise AssertionError("missing worker initialization")
        if mode == "central" and any(row["cuda_initialized"] for row in identities):
            raise AssertionError("central search worker initialized CUDA")
        startup_seconds = time.perf_counter()-startup
        start = time.perf_counter()
        pending = {executor.submit(game_task, (index, seed, trace)) for index, seed in tasks}
        rows = []
        deadline = start + timeout*len(tasks)
        while pending:
            if time.perf_counter() > deadline or (service is not None and not service.process.is_alive()):
                raise RuntimeError("group timed out or inference server died")
            done, pending = futures.wait(pending, timeout=1.0, return_when=futures.FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                record("game", dict(mode=mode, **row["exact"], seconds=row["seconds"]))
        seconds = time.perf_counter()-start
        executor.shutdown(wait=True)
        executor = None
        if service is not None:
            service.close()
            if (service.report["calls"] != sum(r["transport"]["calls"] for r in rows)
                    or service.report["rows"] != sum(r["transport"]["rows"] for r in rows)):
                raise AssertionError("server accounting mismatch")
        completed = True
        return dict(mode=mode, seconds=seconds, startup_seconds=startup_seconds, workers=identities,
                    server=service.report if service else None,
                    games=sorted(rows, key=lambda r: r["exact"]["index"]))
    finally:
        if executor is not None:
            # Only stop this experiment's pool; do not touch unrelated processes.
            for process in list((executor._processes or {}).values()):
                if process.is_alive():
                    process.terminate()
            executor.shutdown(wait=True, cancel_futures=True)
        if service is not None:
            service.close(abort=not completed)
        ready.cancel_join_thread()
        ready.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=16, help="games per round")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--trace", action="store_true", help="hash every actual input/output; not a speed measurement")
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or not args.output.parent.is_dir():
        parser.error("new output and journal in an existing directory required")
    if (args.games < 1 or args.rounds < 1 or args.workers < 1 or args.workers > args.games
            or args.timeout <= 0):
        parser.error("positive games/rounds/timeout and workers <= games required")
    if (jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython" or az.SEARCH_BACKEND != "python"
            or not torch.cuda.is_available()):
        parser.error("require CUDA, rules/legal Cython and Python nodes")
    torch.set_num_threads(1)
    checkpoint = args.run_dir/"models/training_checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = az.AZConfig(**saved["config"])
    config.lazy_chance_allocation = True
    if config.pcr_enabled or not config.chance_nodes or config.chance_k != 0 or not config.leaf_dedup:
        parser.error("require fixed budgets, full chance support and leaf dedup")
    if config.strong_self_play_interval > 0 and args.games % config.strong_self_play_interval:
        parser.error("whole strong-game intervals required")
    saved["config"] = asdict(config)
    paths = [checkpoint, args.run_dir/"models/latest.pt", args.run_dir/"models/train_state.json",
             args.run_dir/"logs/train.log", args.run_dir/"selfplay"/saved["replay_files"][-1]["name"],
             ROOT/"azero.py", ROOT/"jieqi.py", ROOT/"ismcts.py", ROOT/"run_records.py", Path(__file__),
             Path(central.__file__), Path(probe.__file__), ROOT/"tools/measure_search_reuse.py",
             ROOT/"native/jieqi_rules.pyx", ROOT/"native/jieqi_legal.pyx",
             Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"])]
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    saved["_inference_generation"] = hashes[str(checkpoint)]
    result = dict(args={k: str(v) for k, v in vars(args).items()}, source_hashes=hashes,
                  checkpoint_iteration=saved["train_state"]["iteration"], config=asdict(config),
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), precision=central.precision_settings(),
                  rounds=[], implementation="one synchronous request per worker; original batches never merged")
    start = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle:
        def record(event, value):
            handle.write(json.dumps(dict(event=event, value=value, elapsed_seconds=time.perf_counter()-start))+"\n")
            handle.flush()
            probe.emit(event, value)

        record("metadata", result)
        for round_index in range(args.rounds):
            tasks = [(i, 20265000+i) for i in range(round_index*args.games, (round_index+1)*args.games)]
            groups = {}
            for mode in (("local", "central") if round_index % 2 == 0 else ("central", "local")):
                groups[mode] = run_group(saved, mode, tasks, args.workers, args.timeout, args.trace, record)
                record("group", dict(round=round_index, mode=mode, seconds=groups[mode]["seconds"],
                    startup_seconds=groups[mode]["startup_seconds"], server=groups[mode]["server"]))
            for local, remote in zip(groups["local"]["games"], groups["central"]["games"]):
                if local["exact"] != remote["exact"]:
                    record("mismatch", dict(local=local["exact"], central=remote["exact"]))
                    raise AssertionError("central inference changed evaluation or self-play result")
            result["rounds"].append(groups)
            record("paired_exact", dict(round=round_index, games=args.games, identical=True))
        verify_hashes(hashes)
        result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-start)
        with args.output.open("x", encoding="utf-8") as handle_out:
            json.dump(result, handle_out, indent=2)
        record("complete", dict(output=str(args.output), elapsed_seconds=result["elapsed_seconds"], production_unchanged=True))


if __name__ == "__main__":
    main()
