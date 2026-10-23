"""Frozen-model full self-play round with task-level and resource measurements."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from dataclasses import asdict
import gc
import json
import logging
import os
from pathlib import Path
import pickle
import random
import statistics
import subprocess
import sys
import threading
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch
import azero as az
import jieqi as jq
import run_records as rr
from tools import profile_selfplay as probe
from tools.measure_search_reuse import verify_hashes

EXECUTOR = futures.ProcessPoolExecutor
WORKER_LOG = None
WORKER_ORIGIN = 0.0
GPU_FIELDS = ("utilization.gpu", "utilization.memory", "memory.used", "temperature.gpu",
              "power.draw", "clocks.sm", "clocks.mem")


def append_event(path, event, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event=event, value=value)) + "\n")


def runtime_identity():
    return dict(rules=(jq.RULES_BACKEND, jq.RULES_BACKEND_NATIVE),
                legal=(jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE),
                nodes=(az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32)


def measured_init(weights, config, device, directory, origin, expected):
    global WORKER_LOG, WORKER_ORIGIN
    WORKER_LOG = Path(directory) / f"worker_{os.getpid()}.jsonl"
    WORKER_ORIGIN = origin
    start = time.perf_counter()
    if runtime_identity() != expected:
        raise RuntimeError("worker backend/precision identity differs")
    az._init_selfplay_worker(weights, config, device)
    append_event(WORKER_LOG, "ready", dict(pid=os.getpid(), start=start-origin,
                 end=time.perf_counter()-origin, identity=expected))


def measured_worker(task):
    start, cpu_start = time.perf_counter(), time.process_time()
    cache_start = az._cached_full_chance_outcomes.cache_info()
    game = az._selfplay_worker(task)
    end, cpu_end = time.perf_counter(), time.process_time()
    cache_end = az._cached_full_chance_outcomes.cache_info()
    append_event(WORKER_LOG, "game", dict(pid=os.getpid(), seed=task[0], strong=task[1],
        start=start-WORKER_ORIGIN, end=end-WORKER_ORIGIN, wall_seconds=end-start,
        process_cpu_seconds=cpu_end-cpu_start, outcome=game.outcome, moves=game.move_count,
        samples=len(game.samples), leaf_total=game.leaf_evals_total, leaf_unique=game.leaf_evals_unique,
        cache_hits=cache_end.hits-cache_start.hits, cache_misses=cache_end.misses-cache_start.misses,
        cache_size=cache_end.currsize))
    return game


class RoundExecutor(EXECUTOR):
    directory = origin = expected = None

    def __init__(self, *args, **kwargs):
        if kwargs.get("initializer") is not az._init_selfplay_worker:
            raise RuntimeError("unexpected initializer")
        kwargs["initializer"] = measured_init
        kwargs["initargs"] = (*kwargs["initargs"], self.directory, self.origin, self.expected)
        self.task_index = 0
        self.receipts = []
        self.receipt_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def submit(self, function, task):
        if function is not az._selfplay_worker:
            raise RuntimeError("unexpected worker task")
        index = self.task_index
        self.task_index += 1
        submitted = time.perf_counter()-self.origin
        future = super().submit(measured_worker, task)

        def received(done):
            # This runs after the executor has unpickled the original full result.
            row = dict(index=index, seed=task[0], strong=task[1], submitted=submitted,
                       received=time.perf_counter()-self.origin)
            if done.cancelled():
                row["error"] = "cancelled"
            elif done.exception() is not None:
                row["error"] = str(done.exception())
            with self.receipt_lock:
                self.receipts.append(row)
            append_event(Path(self.directory)/"receipts.jsonl", "received", row)

        future.add_done_callback(received)
        return future


class ResourceMonitor:
    def __init__(self, path, origin, interval):
        self.path, self.origin, self.interval = path, origin, interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        process = psutil.Process()
        while not self.stop.is_set():
            row = dict(seconds=time.perf_counter()-self.origin, processes=[])
            for child in [process, *process.children(recursive=True)]:
                try:
                    if child.pid != process.pid and "python" not in child.name().lower():
                        continue
                    cpu = child.cpu_times()
                    row["processes"].append(dict(pid=child.pid, created=child.create_time(),
                        cpu_seconds=cpu.user+cpu.system, rss_bytes=child.memory_info().rss))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            row["system"] = dict(cpu_percent=psutil.cpu_percent(),
                available_bytes=psutil.virtual_memory().available)
            try:
                output = subprocess.run(["nvidia-smi", "--query-gpu="+",".join(GPU_FIELDS),
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True).stdout
                row["gpu"] = [dict(zip(GPU_FIELDS, [float(v.strip()) for v in line.split(",")]))
                              for line in output.strip().splitlines()]
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row["gpu_error"] = str(exc)
            append_event(self.path, "resource", row)
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


def distribution(values):
    if not values:
        return None
    return dict(count=len(values), mean=statistics.mean(values), total=sum(values),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), minimum=min(values), maximum=max(values))


def active_intervals(games, start, end):
    events = [(max(start, row["start"]), 1) for row in games if row["end"] > start and row["start"] < end]
    events += [(min(end, row["end"]), -1) for row in games if row["end"] > start and row["start"] < end]
    active, previous, seconds = 0, start, {}
    for when, delta in sorted(events):
        seconds[active] = seconds.get(active, 0.0) + when-previous
        active += delta
        previous = when
    seconds[active] = seconds.get(active, 0.0) + end-previous
    return seconds


def summarize_games(games, receipts, wall_seconds, workers):
    if len({r["seed"] for r in games}) != len(games):
        raise AssertionError("ambiguous seed matching")
    by_seed = {r["seed"]: r for r in games}
    if len(receipts) != len(games) or any("error" in row for row in receipts):
        raise AssertionError("missing/failed worker results")
    gaps = [row["received"]-by_seed[row["seed"]]["end"] for row in receipts]
    last_start = max(r["start"] for r in games)
    last_end = max(r["end"] for r in games)
    tail = [r for r in games if r["end"] > last_start]
    worker_rows = {}
    for pid in sorted({r["pid"] for r in games}):
        selected = [r for r in games if r["pid"] == pid]
        worker_rows[pid] = dict(games=len(selected), strong=sum(r["strong"] for r in selected),
            game_wall_seconds=sum(r["wall_seconds"] for r in selected),
            process_cpu_seconds=sum(r["process_cpu_seconds"] for r in selected),
            last_game_end=max(r["end"] for r in selected))
    cpu = sum(r["process_cpu_seconds"] for r in games)
    game_wall = sum(r["wall_seconds"] for r in games)
    return dict(games=len(games), strong_games=sum(r["strong"] for r in games),
        moves=sum(r["moves"] for r in games), samples=sum(r["samples"] for r in games),
        leaf_total=sum(r["leaf_total"] for r in games), leaf_unique=sum(r["leaf_unique"] for r in games),
        games_per_second=len(games)/wall_seconds,
        moves_per_second=sum(r["moves"] for r in games)/wall_seconds,
        worker_game_wall_seconds=game_wall, worker_process_cpu_seconds=cpu,
        worker_cpu_core_equivalents=cpu/wall_seconds,
        worker_cpu_to_game_wall_ratio=cpu/game_wall,
        worker_game_occupancy=game_wall/(workers*wall_seconds),
        normal_game_seconds=distribution([r["wall_seconds"] for r in games if not r["strong"]]),
        strong_game_seconds=distribution([r["wall_seconds"] for r in games if r["strong"]]),
        ipc_receipt_gap_seconds=distribution(gaps),
        last_task_start=last_start, last_task_end=last_end,
        drained_queue_tail_seconds=last_end-last_start, tail_games=tail,
        active_game_seconds=active_intervals(games, 0.0, wall_seconds),
        tail_active_game_seconds=active_intervals(games, last_start, last_end),
        per_worker=worker_rows, cache_hits=sum(r["cache_hits"] for r in games),
        cache_misses=sum(r["cache_misses"] for r in games),
        maximum_cache_size=max(r["cache_size"] for r in games))


def read_events(path, event):
    return [row["value"] for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
            if row["event"] == event]


def resource_summary(rows, selfplay_end):
    rows = [r for r in rows if r["seconds"] <= selfplay_end]
    gpu = [r["gpu"][0] for r in rows if r.get("gpu")]
    return dict(samples=len(rows), gpu_samples=len(gpu),
        gpu={key: distribution([r[key] for r in gpu]) for key in GPU_FIELDS},
        summed_process_rss_bytes=distribution([sum(p["rss_bytes"] for p in r["processes"]) for r in rows]),
        system_available_bytes=distribution([r["system"]["available_bytes"] for r in rows]),
        system_cpu_percent=distribution([r["system"]["cpu_percent"] for r in rows]),
        gpu_errors=[r["gpu_error"] for r in rows if "gpu_error" in r])


class RoundBoundaries(probe.Boundaries):
    def install(self, stack):
        super().install(stack)
        for name in ("expand", "select_child"):
            stack.enter_context(mock.patch.object(az.PUCTNode, name,
                self.wrapper(name, getattr(az.PUCTNode, name))))


def hotspot_init(*args):
    probe.init_worker(*args)
    probe.Boundaries = RoundBoundaries


def hotspot_probe(weights, config, fixtures_path, workers, record):
    with fixtures_path.open("rb") as handle:
        fixtures = pickle.load(handle)["fixtures"]
    trials = []
    identities = (weights, asdict(config), jq.RULES_BACKEND_NATIVE,
                  (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE),
                  (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))
    with EXECUTOR(max_workers=workers, mp_context=az.mp.get_context("spawn"),
                  initializer=hotspot_init, initargs=identities) as pool:
        tasks = [(f, strong, 202610081+i, "plain") for i, f in enumerate(fixtures) for strong in (False, True)]
        list(pool.map(probe.worker_task, tasks))
        for repeat in range(2):
            exact = {}
            for mode in (("plain", "boundaries") if repeat == 0 else ("boundaries", "plain")):
                start = time.perf_counter()
                results = list(pool.map(probe.worker_task, [(*t[:3], mode) for t in tasks]))
                row = dict(repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-start, results=results)
                exact[mode] = [r["exact"] for r in results]
                trials.append(row)
                record("hotspot_group", {k:v for k,v in row.items() if k != "results"})
            if exact["plain"] != exact["boundaries"]:
                raise AssertionError("hotspot timers changed policy/RNG/counters")
    return trials


def game_signature(game):
    return probe.digest(asdict(game))


def worker_exact_check(net, config, directory):
    global WORKER_LOG, WORKER_ORIGIN
    WORKER_LOG, WORKER_ORIGIN = directory/"exact_worker.jsonl", time.perf_counter()
    device = torch.device(config.self_play_device)
    rows = []
    with mock.patch.object(az, "_WORKER_NET", net), mock.patch.object(az, "_WORKER_CONFIG", config), mock.patch.object(
            az, "_WORKER_DEVICE", device):
        for strong in (False, True):
            task = (202610080+int(strong), strong)
            baseline = az._selfplay_worker(task)
            measured = measured_worker(task)
            if game_signature(baseline) != game_signature(measured):
                raise AssertionError("task timers changed complete game samples")
            rows.append(dict(strong=strong, samples=len(measured.samples), moves=measured.move_count,
                             signature=game_signature(measured)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resource-interval", type=float, default=5.0)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    production = args.run_dir.resolve()
    if (directory.exists() or production == directory or production in directory.parents
            or ROOT/"data/pilots" not in directory.parents):
        parser.error("use a new output directory below data/pilots, outside production")
    if args.games < 1 or args.workers < 2 or args.resource_interval < 1:
        parser.error("positive games, at least two workers and resource interval >=1 required")
    if (jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython"
            or az.SEARCH_BACKEND != "python" or not torch.cuda.is_available()):
        parser.error("require CUDA, rules/legal Cython, Python nodes")
    torch.set_num_threads(1)
    checkpoint_path = production/"models/training_checkpoint.pt"
    checkpoint_hash = rr.file_record(checkpoint_path)["sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if rr.file_record(checkpoint_path)["sha256"] != checkpoint_hash:
        raise RuntimeError("checkpoint changed while loading")
    config = az.AZConfig(**checkpoint["config"])
    if (config.self_play_device not in ("cuda", "cuda:0")
            or config.leaf_batch_size != 64 or config.virtual_loss != 3.0):
        parser.error("require CUDA device 0, leaf batch 64 and virtual loss 3")
    if (config.pcr_enabled or not config.chance_nodes or config.chance_k != 0
            or not config.lazy_chance_allocation or not config.leaf_dedup
            or config.mcts_simulations != 800 or config.strong_mcts_simulations != 3200
            or config.strong_self_play_interval != 8 or args.games % 8 or args.workers > args.games):
        parser.error("requires current 800/3200 fixed-budget protocol, 7:1 full cohorts")
    config.self_play_games, config.self_play_workers = args.games, args.workers
    config.data_dir, config.model_dir, config.log_dir = (str(directory/n) for n in ("samples", "models", "logs"))
    paths = [production/n for n in ("models/training_checkpoint.pt", "models/latest.pt", "models/train_state.json", "logs/train.log")]
    paths += [production/"selfplay"/checkpoint["replay_files"][-1]["name"]]
    paths += [ROOT/n for n in ("azero.py", "ismcts.py", "jieqi.py", "run_records.py", "training_runtime.py",
        "native/jieqi_rules.pyx", "native/jieqi_legal.pyx", "tools/profile_selfplay.py", "tools/measure_search_reuse.py")]
    paths += [Path(__file__).resolve(), args.fixtures.resolve(), Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"])]
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    if hashes[str(checkpoint_path)] != checkpoint_hash:
        raise RuntimeError("checkpoint changed before freezing")
    directory.mkdir(parents=True)
    frozen = directory/"frozen_training_checkpoint.pt"
    with checkpoint_path.open("rb") as source, frozen.open("xb") as target:
        import shutil
        shutil.copyfileobj(source, target)
    if rr.file_record(frozen)["sha256"] != hashes[str(checkpoint_path)]:
        raise AssertionError("freeze failed")
    start = time.perf_counter()
    def record(event, value):
        append_event(directory/"events.jsonl", event, dict(elapsed=time.perf_counter()-start, data=value))
        probe.emit(event, value)
    identity = runtime_identity()
    result = dict(args={k:str(v) for k,v in vars(args).items()}, source_hashes=hashes,
        checkpoint_iteration=checkpoint["train_state"]["iteration"], config=asdict(config),
        identity=identity, torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), psutil=psutil.__version__,
        protocol=["No optimizer, evaluation, diagnostic creation, production writes or full replay reload",
                  "Unchanged production scheduling/full sample IPC; only per-task timers, receipts and 5s telemetry",
                  "Chance cache stays warm across games, like production; no per-game clearing",
                  "Single current baseline, not a paired speedup comparison",
                  "GPU duty is sampled, not CUDA queue/kernel/wait attribution",
                  "Augmentation uses an empty isolated buffer, not the saturated 4M eviction cost"])
    record("metadata", result)
    net = az.PolicyValueNet().to(config.self_play_device).eval()
    az.load_state_dict_padded(net, checkpoint["model"])
    result["worker_exact"] = worker_exact_check(net, config, directory)
    record("worker_exact", result["worker_exact"])
    logger = logging.getLogger("round_measurement")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    flags = [(i+1) % 8 == 0 for i in range(args.games)]
    rng = random.Random()
    rng.setstate(checkpoint["rng"]["selfplay"])
    seeds_rng = random.Random()
    seeds_rng.setstate(rng.getstate())
    seeds = [seeds_rng.randrange(1 << 30) for _ in range(args.games)]
    if len(set(seeds)) != len(seeds):
        raise AssertionError("seed collision; keep task index telemetry instead")
    RoundExecutor.directory, RoundExecutor.expected = str(directory), identity
    origin = time.perf_counter()
    RoundExecutor.origin = origin
    with ResourceMonitor(directory/"resources.jsonl", origin, args.resource_interval):
        with mock.patch.object(az.futures, "ProcessPoolExecutor", RoundExecutor):
            results = az.generate_selfplay_results(net, config, torch.device(config.self_play_device), rng,
                args.games, logger, strong_flags=flags, iteration=checkpoint["train_state"]["iteration"])
        selfplay_end = time.perf_counter()-origin
        if len(results) != args.games or any(r.outcome is None for r in results) or rng.getstate() != seeds_rng.getstate():
            raise AssertionError("round incomplete or scheduling RNG changed")
        games, ready = [], []
        for path in directory.glob("worker_*.jsonl"):
            games.extend(read_events(path, "game"))
            ready.extend(read_events(path, "ready"))
        receipts = read_events(directory/"receipts.jsonl", "received")
        by_seed = {r["seed"]: r for r in games}
        for seed, flag, game in zip(seeds, flags, results):
            row = by_seed[seed]
            if (flag, game.outcome, game.move_count, len(game.samples), game.leaf_evals_total, game.leaf_evals_unique) != (
                    row["strong"], row["outcome"], row["moves"], row["samples"], row["leaf_total"], row["leaf_unique"]):
                raise AssertionError("production task/result order changed")
        result["selfplay_seconds"] = selfplay_end
        result["round"] = summarize_games(games, receipts, selfplay_end, args.workers)
        result["worker_ready"] = ready
        record("round_summary", result["round"])
        phase = time.perf_counter()
        samples, buffer = [], az.ReplayBuffer(config.replay_buffer_size)
        for game in results:
            samples.extend(game.samples)
            buffer.add_game(az.augment_samples(game.samples, mirror=config.augment_mirror, flip=config.augment_flip))
        result["augmentation_seconds"] = time.perf_counter()-phase
        result["augmented_samples"] = len(buffer)
        phase = time.perf_counter()
        payload = directory/"samples"/"round.pkl.gz"
        az.save_compact_samples(samples, str(payload), dict(measurement_only=True, game_lengths=[len(r.samples) for r in results],
            iteration=checkpoint["train_state"]["iteration"], source_checkpoint_sha256=hashes[str(checkpoint_path)],
            generated_games=len(results), strong_games=sum(flags), ruleset_version=jq.RULESET_VERSION))
        result["save_seconds"] = time.perf_counter()-phase
        result["selfplay_augment_save_seconds"] = time.perf_counter()-origin
        phase = time.perf_counter()
        restored, _ = az.load_compact_payload(str(payload))
        if probe.digest(restored) != probe.digest(samples):
            raise AssertionError("saved sample roundtrip differs")
        result["roundtrip_check_seconds"] = time.perf_counter()-phase
        result["saved_payload"] = rr.file_record(payload)
        del samples, restored, buffer, results, net
        gc.collect()
        torch.cuda.empty_cache()
    result["resources"] = resource_summary(read_events(directory/"resources.jsonl", "resource"), selfplay_end)
    record("pipeline", {k:result[k] for k in ("selfplay_seconds", "augmentation_seconds", "save_seconds",
        "selfplay_augment_save_seconds", "augmented_samples", "roundtrip_check_seconds")})
    result["hotspots"] = hotspot_probe(checkpoint["model"], config, args.fixtures, args.workers, record)
    verify_hashes(hashes)
    if rr.file_record(frozen)["sha256"] != hashes[str(checkpoint_path)]:
        raise AssertionError("frozen checkpoint changed")
    result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-start)
    with (directory/"result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    record("complete", dict(directory=str(directory), production_unchanged=True))


if __name__ == "__main__":
    main()
