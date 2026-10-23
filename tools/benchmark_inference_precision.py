"""Isolated CUDA inference precision probe; never updates production weights or data."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
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
import torch

import azero as az
import jieqi as jq
import run_records as rr
from tools import central_inference as central
from tools import profile_selfplay as probe
from tools.measure_search_reuse import verify_hashes

MODES = ("baseline", "tf32", "amp_fp16", "amp_bf16", "ieee_fp32")
AMP_DTYPES = {"amp_fp16": torch.float16, "amp_bf16": torch.bfloat16}
NET = CONFIG = SETTINGS = None
DEVICE = torch.device("cuda")


@contextmanager
def precision_scope(mode, baseline):
    if mode not in MODES:
        raise ValueError("unknown precision mode")
    previous = central.precision_settings()
    settings = dict(baseline)
    if mode == "tf32":
        settings.update(matmul_tf32=True, cudnn_tf32=True)
    elif mode == "ieee_fp32":
        settings.update(matmul_tf32=False, cudnn_tf32=False)
    try:
        central.configure_precision(settings)
        # Keep the autocast weight cache alive over a frozen search/game, not just one call.
        with torch.no_grad(), torch.autocast("cuda", dtype=AMP_DTYPES.get(mode, torch.float16),
                                             enabled=mode in AMP_DTYPES):
            yield
    finally:
        central.configure_precision(previous)


class FP32Outputs(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        logits, values = self.net(inputs)
        return logits.float(), values.float()


def forward(net, batch):
    logits, values = net(torch.from_numpy(batch).to(DEVICE))
    return torch.softmax(logits, dim=1).cpu().numpy(), np.asarray(values.tolist(), dtype=np.float32)


def compare_outputs(reference, candidate):
    p, v = reference
    q, w = candidate
    if p.shape != q.shape or v.shape != w.shape or not len(p):
        raise ValueError("nonempty matching outputs required")
    finite = bool(np.isfinite(q).all() and np.isfinite(w).all())
    if not finite:
        raise AssertionError("nonfinite inference output")
    p64, q64 = p.astype(np.float64), q.astype(np.float64)
    tiny = np.finfo(np.float64).tiny
    return dict(rows=len(p), finite=finite,
                bitwise_equal=p.tobytes() == q.tobytes() and v.tobytes() == w.tobytes(),
                policy_max_abs=float(np.max(np.abs(p64-q64))),
                value_max_abs=float(np.max(np.abs(v.astype(np.float64)-w.astype(np.float64)))),
                value_mean_abs=float(np.mean(np.abs(v.astype(np.float64)-w.astype(np.float64)))),
                policy_mean_l1=float(np.abs(p64-q64).sum(axis=1).mean()),
                policy_max_l1=float(np.abs(p64-q64).sum(axis=1).max()),
                policy_mean_kl=float((p64*(np.log(np.maximum(p64, tiny))-
                                               np.log(np.maximum(q64, tiny)))).sum(axis=1).mean()),
                full_policy_argmax_equal=int(np.sum(p.argmax(axis=1) == q.argmax(axis=1))))


def balanced_order(modes, repeat):
    order = tuple(modes)
    shift = (repeat//2) % len(order)
    order = order[shift:]+order[:shift]
    return order if repeat % 2 == 0 else tuple(reversed(order))


def paired_summary(trials, modes):
    if not trials:
        return {}
    repeats = sorted({row["repeat"] for row in trials})
    times = {(r["repeat"], r["mode"]): r["wall_seconds"] for r in trials}
    result = {}
    for mode in modes:
        saving = [1-times[r, mode]/times[r, "baseline"] for r in repeats]
        rng = random.Random(20261008)
        means = [statistics.mean(rng.choices(saving, k=len(saving))) for _ in range(3000)]
        total = sum(times[r, mode] for r in repeats)
        baseline_total = sum(times[r, "baseline"] for r in repeats)
        result[mode] = dict(total_seconds=total, mean_paired_time_saving=statistics.mean(saving),
                            paired_savings=saving, throughput_ratio=baseline_total/total,
                            descriptive_bootstrap_interval=np.percentile(means, [2.5, 97.5]).tolist())
    return result


def throughput_gate(trials, minimum_saving, minimum_rounds=8):
    summary = paired_summary(trials, ("baseline", "amp_fp16"))
    if not summary:
        return dict(passed=False, reason="no complete game cohorts")
    candidate = summary["amp_fp16"]
    rounds = len(candidate["paired_savings"])
    saving = 1-1/candidate["throughput_ratio"]
    lower = candidate["descriptive_bootstrap_interval"][0]
    return dict(passed=rounds >= minimum_rounds and saving >= minimum_saving and lower > 0,
                rounds=rounds, minimum_rounds=minimum_rounds,
                aggregate_time_saving=saving, required_time_saving=minimum_saving,
                descriptive_interval_lower=lower,
                scope="pilot screening only; not strength, convergence or production acceptance")


def gpu_snapshot():
    fields = "temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used"
    row = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), fields=fields)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu="+fields,
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10)
        row.update(value=result.stdout.strip(), error=result.stderr.strip(), returncode=result.returncode)
    except (OSError, subprocess.TimeoutExpired) as error:
        row["error"] = str(error)
    return row


class GpuTelemetry:
    def __init__(self, interval):
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        row = gpu_snapshot()
        row["elapsed_seconds"] = time.perf_counter()-self.started
        self.samples.append(row)

    def collect(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.started = time.perf_counter()
        if self.interval > 0:
            self.sample()
            self.thread = threading.Thread(target=self.collect, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.thread is not None:
            self.stop.set()
            self.thread.join()
            self.sample()


def capture_requests(net, config, fixtures):
    batches = []
    original = az._encode_batch_numpy
    for fixture in fixtures:
        info = fixture["belief"].info_state(fixture["state"], fixture["state"].side)
        batches.append(az.encode_state_numpy(info, fixture["history"])[None])
        captured = []

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(captured) < 4:
                captured.append(result.copy())
            return result

        with mock.patch.object(az, "_encode_batch_numpy", capture):
            probe.run_case(net, config, DEVICE, fixture, False,
                           20268000+fixture["metadata"]["index"], "plain")
        batches.extend(captured)
    return batches


def numerical_and_micro(net, requests, modes, settings, rounds, repeats):
    with precision_scope("baseline", settings):
        reference = [forward(net, batch) for batch in requests]
    numerical = {}
    for mode in modes:
        with precision_scope(mode, settings):
            comparisons = [compare_outputs(old, forward(net, batch))
                           for batch, old in zip(requests, reference)]
        rows = sum(r["rows"] for r in comparisons)
        numerical[mode] = dict(requests=len(comparisons), rows=rows,
            mismatching_requests=sum(not r["bitwise_equal"] for r in comparisons),
            policy_max_abs=max(r["policy_max_abs"] for r in comparisons),
            value_max_abs=max(r["value_max_abs"] for r in comparisons),
            policy_max_l1=max(r["policy_max_l1"] for r in comparisons),
            policy_mean_l1=sum(r["policy_mean_l1"]*r["rows"] for r in comparisons)/rows,
            policy_mean_kl=sum(r["policy_mean_kl"]*r["rows"] for r in comparisons)/rows,
            value_mean_abs=sum(r["value_mean_abs"]*r["rows"] for r in comparisons)/rows,
            full_policy_argmax_equal=sum(r["full_policy_argmax_equal"] for r in comparisons),
            comparisons=comparisons)
        probe.emit("numerical", dict(mode=mode, **{k: v for k, v in numerical[mode].items()
                                                   if k != "comparisons"}))
    selected = sorted(requests, key=len)
    selected = [selected[i] for i in sorted(set(np.linspace(0, len(selected)-1, 8).astype(int)))]
    trials = []
    for repeat in range(rounds):
        for mode in balanced_order(modes, repeat):
            with precision_scope(mode, settings):
                for batch in selected:
                    forward(net, batch)
                probe.synchronize(DEVICE)
                start = time.perf_counter()
                for _ in range(repeats):
                    for batch in selected:
                        forward(net, batch)
                probe.synchronize(DEVICE)
                seconds = time.perf_counter()-start
            row = dict(repeat=repeat, mode=mode, wall_seconds=seconds,
                       calls=repeats*len(selected), rows=repeats*sum(map(len, selected)),
                       milliseconds_per_call=seconds*1000/(repeats*len(selected)))
            trials.append(row)
            probe.emit("micro", row)
    return numerical, trials, list(map(len, selected))


def init_worker(weights, architecture, config, settings, modes, backends):
    global NET, CONFIG, SETTINGS
    if (jq.RULES_BACKEND_NATIVE, jq.LEGAL_BACKEND_NATIVE, az.SEARCH_BACKEND_NATIVE) != backends:
        raise AssertionError("worker backend identity changed")
    torch.set_num_threads(1)
    CONFIG, SETTINGS = az.AZConfig(**config), settings
    base = az.PolicyValueNet(**architecture).to(DEVICE).eval()
    base.load_state_dict(weights)
    NET = FP32Outputs(base).eval()
    for mode in modes:
        with precision_scope(mode, settings):
            for n in (1, CONFIG.leaf_batch_size):
                forward(NET, np.zeros((n, az.STATE_CHANNELS, 10, 9), dtype=np.float32))


def search_task(task):
    mode, fixture, strong, seed = task
    search = az.NeuralMCTS(NET, CONFIG, DEVICE, random.Random(seed))
    history = fixture["history"]
    temperature = az.temperature_for_move(len(history), CONFIG.temperature_initial,
        CONFIG.strong_temperature_final if strong else CONFIG.temperature_final,
        CONFIG.strong_temperature_decay_moves if strong else CONFIG.temperature_decay_moves)
    probe.synchronize(DEVICE)
    start = time.perf_counter()
    with precision_scope(mode, SETTINGS):
        policy, move = search.search(fixture["state"], fixture["belief"], history, temperature,
            simulations=CONFIG.strong_mcts_simulations if strong else CONFIG.mcts_simulations,
            add_dirichlet=True, dirichlet_alpha=CONFIG.dirichlet_alpha,
            dirichlet_epsilon=CONFIG.strong_dirichlet_epsilon if strong else CONFIG.dirichlet_epsilon,
            prune_policy=True)
    probe.synchronize(DEVICE)
    return dict(index=fixture["metadata"]["index"], strong=strong, seed=seed,
                seconds=time.perf_counter()-start, move=move,
                policy={str(i): float(p) for i, p in enumerate(policy) if p},
                rng_sha256=probe.digest(search.rng.getstate()),
                leaf_total=search.leaf_evals_total, leaf_unique=search.leaf_evals_unique)


def game_task(task):
    mode, index, strong, seed = task
    rng = random.Random(seed)
    probe.synchronize(DEVICE)
    start = time.perf_counter()
    cpu_start = time.process_time()
    with precision_scope(mode, SETTINGS):
        game = az.self_play_game(NET, CONFIG, DEVICE, rng, strong=strong)
    probe.synchronize(DEVICE)
    if game.outcome is None:
        raise AssertionError("unfinished game")
    seconds, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu_start
    return dict(index=index, strong=strong, seed=seed, seconds=seconds, process_cpu_seconds=cpu_seconds,
                moves=game.move_count, outcome=game.outcome, samples=len(game.samples),
                sample_sha256=probe.digest([asdict(s) for s in game.samples]),
                rng_sha256=probe.digest(rng.getstate()),
                leaf_total=game.leaf_evals_total, leaf_unique=game.leaf_evals_unique)


def compare_searches(reference, rows):
    if not rows or len(reference) != len(rows):
        raise ValueError("nonempty paired searches required")
    l1 = []
    for old, new in zip(reference, rows):
        keys = old["policy"].keys() | new["policy"].keys()
        l1.append(sum(abs(old["policy"].get(k, 0)-new["policy"].get(k, 0)) for k in keys))
    return dict(cases=len(rows), selected_move_equal=sum(a["move"] == b["move"] for a, b in zip(reference, rows)),
                policy_equal=sum(a["policy"] == b["policy"] for a, b in zip(reference, rows)),
                policy_mean_l1=statistics.mean(l1), policy_max_l1=max(l1),
                rng_equal=sum(a["rng_sha256"] == b["rng_sha256"] for a, b in zip(reference, rows)))


def group_trials(pool, fixtures, config, modes, rounds, count, record, games=False,
                 game_seed=20269000, telemetry_interval=0):
    trials = []
    for repeat in range(rounds):
        groups = {}
        for mode in balanced_order(modes, repeat):
            if games:
                tasks = [(mode, repeat*count+i,
                    config.strong_self_play_interval > 0 and (i+1) % config.strong_self_play_interval == 0,
                    game_seed+repeat*count+i) for i in range(count)]
            else:
                tasks = [(mode, fixture,
                    config.strong_self_play_interval > 0 and (block*len(fixtures)+i+1) % config.strong_self_play_interval == 0,
                    20270000+repeat*10000+block*len(fixtures)+i)
                    for block in range(count) for i, fixture in enumerate(fixtures)]
            record("cohort_start", dict(repeat=repeat, mode=mode, games=games, tasks=len(tasks),
                                        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            with GpuTelemetry(telemetry_interval) as telemetry:
                start = time.perf_counter()
                if games:
                    pending = [pool.submit(game_task, task) for task in tasks]
                    rows = []
                    for future in futures.as_completed(pending):
                        item = future.result()
                        rows.append(item)
                        record("game", dict(repeat=repeat, mode=mode, **item))
                    rows.sort(key=lambda item: item["index"])
                else:
                    rows = list(pool.map(search_task, tasks))
                seconds = time.perf_counter()-start
            groups[mode] = rows
            row = dict(repeat=repeat, mode=mode, wall_seconds=seconds, gpu_samples=telemetry.samples,
                       worker_seconds=sum(r["seconds"] for r in rows), results=rows,
                       leaf_unique=sum(r["leaf_unique"] for r in rows))
            if games:
                row.update(moves=sum(r["moves"] for r in rows), samples=sum(r["samples"] for r in rows))
                row["worker_cpu_seconds"] = sum(r["process_cpu_seconds"] for r in rows)
                row["plies_per_second"] = row["moves"]/row["wall_seconds"]
            trials.append(row)
            record("games" if games else "searches", row)
        for mode in modes:
            row = next(r for r in trials if r["repeat"] == repeat and r["mode"] == mode)
            if games:
                row["sample_equal_games"] = sum(a["sample_sha256"] == b["sample_sha256"]
                                                for a, b in zip(groups["baseline"], groups[mode]))
            else:
                row["comparison"] = compare_searches(groups["baseline"], groups[mode])
            record("comparison", dict(repeat=repeat, mode=mode,
                **({"sample_equal_games": row["sample_equal_games"]} if games else row["comparison"])))
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protect-run", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES[:-1]))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--micro-rounds", type=int, default=6)
    parser.add_argument("--micro-repeats", type=int, default=10)
    parser.add_argument("--search-rounds", type=int, default=4)
    parser.add_argument("--search-blocks", type=int, default=4)
    parser.add_argument("--game-rounds", type=int, default=2)
    parser.add_argument("--games-per-round", type=int, default=16)
    parser.add_argument("--game-seed", type=int, default=20269000)
    parser.add_argument("--telemetry-interval", type=float, default=0,
                        help="seconds between GPU samples; zero disables telemetry")
    parser.add_argument("--pilot-min-time-saving", type=float, default=None,
                        help="predeclared FP16 pilot gate; requires >=8 complete paired rounds")
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or not args.output.parent.is_dir():
        parser.error("new output and journal in existing directory required")
    if "baseline" not in args.modes or len(set(args.modes)) != len(args.modes):
        parser.error("unique modes including baseline required")
    if min(args.workers, args.micro_rounds, args.micro_repeats, args.search_blocks, args.games_per_round) < 1 or min(args.search_rounds, args.game_rounds) < 0:
        parser.error("positive sizes and nonnegative search/game rounds required")
    if args.telemetry_interval < 0 or not np.isfinite(args.telemetry_interval):
        parser.error("finite nonnegative telemetry interval required")
    if args.pilot_min_time_saving is not None and (not 0 < args.pilot_min_time_saving < 1
            or set(args.modes) != {"baseline", "amp_fp16"} or args.game_rounds < 8 or args.game_rounds % 2):
        parser.error("pilot gate requires baseline/amp_fp16, >=8 even game rounds and saving in (0,1)")
    if not torch.cuda.is_available() or jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython" or az.SEARCH_BACKEND != "python":
        parser.error("require CUDA, compiled rules/legal, Python nodes")
    if "amp_bf16" in args.modes and not torch.cuda.is_bf16_supported():
        parser.error("BF16 is not supported by this CUDA device")
    torch.set_num_threads(1)
    settings = central.precision_settings()
    if settings != dict(matmul_tf32=False, cudnn_tf32=True, benchmark=False, deterministic=False):
        parser.error("baseline flags differ from the recorded production-default settings")
    checkpoint = args.run_dir/"models/training_checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = az.AZConfig(**saved["config"])
    config.lazy_chance_allocation = True
    if config.pcr_enabled or (args.game_rounds and config.strong_self_play_interval > 0
            and args.games_per_round % config.strong_self_play_interval):
        parser.error("fixed budgets require PCR off and whole strong-game intervals per cohort")
    architecture = {k: saved["network"][k] for k in ("channels", "blocks")}
    paths = [checkpoint, ROOT/"azero.py", ROOT/"jieqi.py", ROOT/"ismcts.py", Path(__file__),
             ROOT/"tools/profile_selfplay.py", ROOT/"tools/central_inference.py",
             ROOT/"tools/measure_search_reuse.py", ROOT/"native/jieqi_rules.pyx", ROOT/"native/jieqi_legal.pyx",
             Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"])]
    for run in (args.run_dir, args.protect_run):
        paths.extend(run/relative for relative in ("models/training_checkpoint.pt", "models/latest.pt",
                                                  "models/train_state.json", "logs/train.log"))
        state = torch.load(run/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
        paths.append(run/"selfplay"/state["replay_files"][-1]["name"])
        del state
    if (ROOT/"analysis_longcheck.py").exists():
        paths.append(ROOT/"analysis_longcheck.py")
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    base = az.PolicyValueNet(**architecture).to(DEVICE).eval()
    base.load_state_dict(saved["model"])
    net = FP32Outputs(base).eval()
    started = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle:
        def record(label, row):
            handle.write(json.dumps(dict(event=label, **row))+"\n")
            handle.flush()
            probe.emit(label, {k: v for k, v in row.items() if k not in ("results", "gpu_samples")})

        record("protocol", dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                source_hashes=hashes, baseline_precision=settings,
                                checkpoint_iteration=saved["train_state"]["iteration"],
                                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        with precision_scope("baseline", settings):
            fixtures, trajectories = probe.collect_fixtures(net, config, DEVICE, 2, 4, 64)
            requests = capture_requests(net, config, fixtures)
        numerical, micro, batch_sizes = numerical_and_micro(net, requests, args.modes, settings,
                                                           args.micro_rounds, args.micro_repeats)
        result = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            source_hashes=hashes, checkpoint_iteration=saved["train_state"]["iteration"],
            config=asdict(config), architecture=architecture, baseline_precision=settings,
            environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                bf16_supported=torch.cuda.is_bf16_supported(),
                fp16_reduced_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            backends=dict(rules=jq.RULES_BACKEND_NATIVE, legal=jq.LEGAL_BACKEND_NATIVE, search=az.SEARCH_BACKEND),
            trajectories=trajectories, fixtures=[f["metadata"] for f in fixtures], numerical=numerical,
            micro_trials=micro, micro_summary=paired_summary(micro, args.modes), micro_batch_sizes=batch_sizes)
        del net, base, requests
        gc.collect()
        torch.cuda.empty_cache()
        search, games, control = [], [], None
        if args.search_rounds or args.game_rounds:
            with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=init_worker, initargs=(saved["model"], architecture, asdict(config), settings,
                    args.modes, (jq.RULES_BACKEND_NATIVE, jq.LEGAL_BACKEND_NATIVE, az.SEARCH_BACKEND_NATIVE))) as pool:
                # Exclude process startup and exercise real roots before steady-state timing.
                list(pool.map(search_task, [(mode, f, False, 20271000+i)
                    for mode in args.modes for i, f in enumerate(fixtures)]))
                control_tasks = [("baseline", f, False, 20271100+i) for i, f in enumerate(fixtures)]
                first = list(pool.map(search_task, control_tasks))
                second = list(pool.map(search_task, control_tasks))
                control = compare_searches(first, second)
                control["leaf_counts_equal"] = sum((a["leaf_total"], a["leaf_unique"]) ==
                    (b["leaf_total"], b["leaf_unique"]) for a, b in zip(first, second))
                if any(control[key] != len(fixtures) for key in
                       ("selected_move_equal", "policy_equal", "rng_equal", "leaf_counts_equal")):
                    raise AssertionError("baseline repeatability control failed")
                record("baseline_repeatability", control)
                search = group_trials(pool, fixtures, config, args.modes, args.search_rounds,
                                      args.search_blocks, record)
                games = group_trials(pool, fixtures, config, args.modes, args.game_rounds,
                                     args.games_per_round, record, games=True, game_seed=args.game_seed,
                                     telemetry_interval=args.telemetry_interval)
        verify_hashes(hashes)
        result.update(baseline_repeatability=control,
                      search_trials=search, search_summary=paired_summary(search, args.modes),
                      game_trials=games, game_summary=paired_summary(games, args.modes),
                      elapsed_seconds=time.perf_counter()-started, production_unchanged=True,
                      limitations=["No training updates, strength evaluation or convergence test",
                        "AMP uses an outer no_grad/autocast scope with a frozen weight cache; outputs/softmax stay FP32",
                        "Micro includes transfer/forward/softmax/CPU outputs but excludes encoding/search/worker contention",
                        "Changed trajectories and game lengths confound game-time ratios; inspect plies/input counts too",
                        "Few balanced batches yield descriptive intervals, not guaranteed production speedup",
                        "GPU telemetry/progress logging have unmeasured overhead; enabled identically for both modes",
                        "GPU telemetry does not prove the absence of external CPU/GPU load or measure CPU thermals",
                        "Fixtures use two 64-simulation trajectories plus capture-biased and controlled roots",
                        "All modes use legal Cython, unlike the latest production manifest with legal Python"])
        if args.pilot_min_time_saving is not None:
            result["throughput_gate"] = throughput_gate(games, args.pilot_min_time_saving)
            record("throughput_gate", result["throughput_gate"])
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        probe.emit("complete", dict(output=str(args.output), elapsed_seconds=result["elapsed_seconds"],
            production_unchanged=True, micro=result["micro_summary"], search=result["search_summary"],
            games=result["game_summary"]))


if __name__ == "__main__":
    main()
