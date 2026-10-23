"""Read-only multi-position profiling of the currently compiled self-play search."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import concurrent.futures as futures
import copy
import cProfile
from dataclasses import asdict
from functools import wraps
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import pstats
import random
import statistics
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import azero as az
import ismcts as im
import jieqi as jq
import run_records as rr

NET = CONFIG = None
DEVICE = torch.device("cuda")


def emit(label, value):
    print(label + " " + json.dumps(value), flush=True)


def digest(value):
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Boundaries:
    """Inclusive timers; encoding is nested in evaluation, never additive to it."""

    def __init__(self):
        self.rows = {}
        self.rule_depth = 0

    def add(self, name, seconds):
        row = self.rows.setdefault(name, dict(calls=0, seconds=0.0))
        row["calls"] += 1
        row["seconds"] += seconds

    def total(self, name):
        return self.rows.get(name, {}).get("seconds", 0.0)

    def wrapper(self, name, function, rule=False):
        @wraps(function)
        def measured(*args, **kwargs):
            if rule and self.rule_depth:
                return function(*args, **kwargs)
            before_native = self.total("native_attack") if rule else 0.0
            if rule:
                self.rule_depth += 1
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self.add(name, elapsed)
                if rule:
                    self.rule_depth -= 1
                    self.add("rules_total", elapsed)
                    native = self.total("native_attack") - before_native
                    self.add("native_inside_rules", native)
        return measured

    def install(self, stack):
        targets = [
            (jq.GameState, "is_square_attacked", "native_attack", False),
            (jq.GameState, "legal_moves", "legal_moves", True),
            (jq.GameState, "has_legal_moves", "has_legal_moves", True),
            (jq.GameState, "_move_scan", "move_scan", True),
            (az.NeuralMCTS, "_evaluate_batch", "evaluate_batch", False),
            (az.NeuralMCTS, "_evaluate", "evaluate_root", False),
            (az, "_encode_batch_numpy", "encode_batch", False),
            (az, "encode_state_numpy", "encode_root", False),
            (az, "leaf_input_signature", "leaf_signature", False),
        ]
        for owner, attribute, name, rule in targets:
            function = getattr(owner, attribute)
            stack.enter_context(mock.patch.object(owner, attribute, self.wrapper(name, function, rule)))


def profile_rows(profile):
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        path = Path(filename)
        try:
            filename = path.relative_to(ROOT).as_posix()
        except ValueError:
            pass
        rows.append(dict(file=filename, line=line, function=name, primitive_calls=primitive,
                         calls=calls, own_seconds=own, cumulative_seconds=cumulative))
    return dict(total_seconds=stats.total_tt, functions=sorted(rows, key=lambda r: -r["own_seconds"]))


def make_fixture(state, belief, history, **metadata):
    pieces = sum(bool(p) for p in state.board)
    return dict(state=copy.deepcopy(state), belief=belief.clone(), history=history.copy(),
                metadata=dict(**metadata, ply=len(history), pieces=pieces,
                              dark_pieces=sum(bool(p) and jq.is_dark(p) for p in state.board),
                              private_captures=sum(sum(row) for row in belief.unknown_removals),
                              legal_moves=len(state.legal_moves()),
                              position_count=len(state.position_counts),
                              material_stage="endgame" if pieces <= 12 else
                              ("opening" if len(history) < 10 and pieces >= 26 else "middlegame")))


def controlled_endgames():
    fixtures = []
    for name, major in (("rook_knight", jq.TYPE_ROOK), ("cannon_knight", jq.TYPE_CANNON)):
        state = jq.GameState.empty(with_pools=False, rng=random.Random(91))
        for row, col, kind, side in [
            (0, 4, jq.TYPE_KING, jq.BLACK), (9, 4, jq.TYPE_KING, jq.RED),
            (5, 4, jq.TYPE_PAWN, jq.RED), (4, 5, jq.TYPE_PAWN, jq.BLACK),
            (7, 0, major, jq.RED), (2, 8, major, jq.BLACK),
            (8, 6, jq.TYPE_KNIGHT, jq.RED), (1, 2, jq.TYPE_KNIGHT, jq.BLACK),
        ]:
            state.place_piece(row, col, kind, side, 0)
        state.reset_history()
        if state.game_outcome()[0] is not None:
            raise AssertionError("controlled endgame is terminal")
        fixtures.append(make_fixture(state, im.init_belief_for_game(state), [],
                                     kind="controlled_endgame", name=name))
    return fixtures


def collect_fixtures(net, config, device, games, roots_per_game, trajectory_sims):
    fixtures, trajectories = [], []
    original = az.NeuralMCTS.search
    config = copy.copy(config)
    if trajectory_sims:
        config.mcts_simulations = trajectory_sims
    for index in range(games):
        captured = []
        seed = (20261005, 20261006, 20261010)[index] if index < 3 else 20261030 + index

        def capture(search, state, belief, history, *args, **kwargs):
            captured.append(make_fixture(state, belief, history, kind="neural_trajectory", seed=seed))
            return original(search, state, belief, history, *args, **kwargs)

        with mock.patch.object(az.NeuralMCTS, "search", capture):
            start = time.perf_counter()
            game = az.self_play_game(net, config, device, random.Random(seed))
        assert game.outcome is not None and captured
        indices = sorted(set(np.linspace(0, len(captured) - 1, min(roots_per_game, len(captured))).astype(int)))
        for ply in indices:
            fixture = captured[ply]
            fraction = ply / max(1, len(captured) - 1)
            fixture["metadata"]["trajectory_phase"] = "early" if fraction < 1/3 else (
                "middle" if fraction < 2/3 else "late")
            fixtures.append(fixture)
        row = dict(seed=seed, simulations=config.mcts_simulations, moves=game.move_count,
                   outcome=game.outcome, seconds=time.perf_counter() - start,
                   selected_plies=[int(i) for i in indices],
                   max_private_captures=max(f["metadata"]["private_captures"] for f in captured))
        trajectories.append(row)
        emit("fixture_game", row)
    rng = random.Random(20261005)
    state = jq.GameState.initial(rng=rng)
    belief = im.init_belief_for_game(state)
    history = []
    for ply in range(25):
        if state.game_outcome()[0] is not None:
            break
        if ply in (6, 12, 18, 24):
            fixtures.append(make_fixture(state, belief, history, kind="capture_biased_trajectory", seed=20261005))
        moves = state.legal_moves()
        captures = [move for move in moves if state.board[move[1]]]
        move = rng.choice(captures or moves)
        im.apply_move_and_update_belief(state, belief, move, rng=rng)
        history.append(move)
    fixtures.extend(controlled_endgames())
    for index, fixture in enumerate(fixtures):
        fixture["metadata"]["index"] = index
    return fixtures, trajectories


def run_case(net, config, device, fixture, strong, seed, mode):
    if mode not in ("plain", "cprofile", "boundaries"):
        raise ValueError("unknown profiling mode")
    state, belief, history = (fixture[name] for name in ("state", "belief", "history"))
    before = digest((state, belief, history))
    search = az.NeuralMCTS(net, config, device, random.Random(seed))
    sims = config.strong_mcts_simulations if strong else config.mcts_simulations
    epsilon = config.strong_dirichlet_epsilon if strong else config.dirichlet_epsilon
    temperature = az.temperature_for_move(len(history), config.temperature_initial,
        config.strong_temperature_final if strong else config.temperature_final,
        config.strong_temperature_decay_moves if strong else config.temperature_decay_moves)
    profiler = cProfile.Profile() if mode == "cprofile" else None
    boundaries = Boundaries() if mode == "boundaries" else None
    with ExitStack() as stack:
        if boundaries is not None:
            boundaries.install(stack)
        synchronize(device)
        start, cpu_start = time.perf_counter(), time.process_time()
        if profiler is not None:
            profiler.enable()
        try:
            policy, move = search.search(state, belief, history, temperature, simulations=sims,
                                        add_dirichlet=True, dirichlet_alpha=config.dirichlet_alpha,
                                        dirichlet_epsilon=epsilon, prune_policy=True)
        finally:
            if profiler is not None:
                profiler.disable()
        synchronize(device)
        cpu_seconds = time.process_time() - cpu_start
        seconds = time.perf_counter() - start
    assert digest((state, belief, history)) == before, "profiling changed input state/belief/history"
    exact = dict(move=move, policy_sha256=hashlib.sha256(np.asarray(policy, dtype=np.float64).tobytes()).hexdigest(),
                 rng_sha256=digest(search.rng.getstate()), leaf_total=search.leaf_evals_total,
                 leaf_unique=search.leaf_evals_unique, chance_applied=search.chance_applied,
                 chance_outcomes_added=search.chance_outcomes_added, chance_nav_miss=search.chance_nav_miss)
    return dict(index=fixture["metadata"].get("index", 0), strong=strong, simulations=sims, seed=seed,
                mode=mode, wall_seconds=seconds, process_cpu_seconds=cpu_seconds, exact=exact,
                boundaries=boundaries.rows if boundaries else None,
                profile=profile_rows(profiler) if profiler else None)


def init_worker(weights, config, native_record, search_backend=None, legal_backend=None):
    global NET, CONFIG
    assert jq.RULES_BACKEND == "cython" and jq.RULES_BACKEND_NATIVE == native_record
    if search_backend is not None:
        assert (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE) == search_backend
    if legal_backend is not None:
        assert (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE) == legal_backend
    torch.set_num_threads(1)
    CONFIG = az.AZConfig(**config)
    NET = az.PolicyValueNet().to(DEVICE).eval()
    NET.load_state_dict(weights)


def worker_task(task):
    fixture, strong, seed, mode = task
    return run_case(NET, CONFIG, DEVICE, fixture, strong, seed, mode)


def parallel_probe(weights, config, fixtures, workers, pairs):
    trials = []
    if pairs == 0:
        return trials
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=init_worker,
                                     initargs=(weights, asdict(config), jq.RULES_BACKEND_NATIVE,
                                               (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE),
                                               (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))) as pool:
        for strong in (False, True):
            list(pool.map(worker_task, [(f, strong, 20261025+i, "plain") for i, f in enumerate(fixtures)]))
        for pair in range(pairs):
            for strong in (False, True):
                outputs = {}
                for mode in (("plain", "boundaries") if pair % 2 == 0 else ("boundaries", "plain")):
                    tasks = [(f, strong, 20261025+i, mode) for i, f in enumerate(fixtures)]
                    start = time.perf_counter()
                    rows = list(pool.map(worker_task, tasks))
                    trial = dict(pair=pair, strong=strong, mode=mode, wall_seconds=time.perf_counter()-start,
                                 worker_seconds=sum(r["wall_seconds"] for r in rows), searches=rows)
                    trials.append(trial)
                    outputs[mode] = rows
                    emit("parallel", {k: v for k, v in trial.items() if k != "searches"})
                assert [r["exact"] for r in outputs["plain"]] == [r["exact"] for r in outputs["boundaries"]]
    return trials


def aggregate_profiles(rows):
    functions = {}
    for row in rows:
        for function in row["profile"]["functions"]:
            key = (function["file"], function["line"], function["function"])
            if key not in functions:
                functions[key] = dict(function, primitive_calls=0, calls=0, own_seconds=0.0, cumulative_seconds=0.0)
            for field in ("primitive_calls", "calls", "own_seconds", "cumulative_seconds"):
                functions[key][field] += function[field]
    return dict(total_seconds=sum(r["profile"]["total_seconds"] for r in rows),
                functions=sorted(functions.values(), key=lambda r: -r["own_seconds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--roots-per-game", type=int, default=6)
    parser.add_argument("--trajectory-sims", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--parallel-pairs", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("choose a new output file in an existing directory")
    if min(args.roots_per_game, args.repeats, args.workers) < 1 or min(args.games, args.parallel_pairs, args.trajectory_sims) < 0:
        parser.error("invalid negative/zero probe size")
    if jq.RULES_BACKEND != "cython" or not torch.cuda.is_available():
        parser.error("this probe requires current Cython backend and CUDA")
    torch.set_num_threads(1)
    paths = [args.run_dir / "models/training_checkpoint.pt", args.run_dir / "models/latest.pt",
             args.run_dir / "models/train_state.json", args.run_dir / "logs/train.log",
             ROOT / "azero.py", ROOT / "jieqi.py", ROOT / "ismcts.py", ROOT / "run_records.py",
             ROOT / "native/jieqi_rules.pyx", Path(jq.RULES_BACKEND_NATIVE["path"]), Path(__file__)]
    if az.SEARCH_BACKEND_NATIVE is not None:
        paths.extend([ROOT / "native/jieqi_search.pyx", Path(az.SEARCH_BACKEND_NATIVE["path"])])
    if jq.LEGAL_BACKEND_NATIVE is not None:
        paths.extend([ROOT / "native/jieqi_legal.pyx", Path(jq.LEGAL_BACKEND_NATIVE["path"])])
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir / "selfplay" / checkpoint["replay_files"][-1]["name"]
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    config = az.AZConfig(**checkpoint["config"])
    if config.pcr_enabled:
        parser.error("probe assumes current fixed-budget self-play, not PCR")
    net = az.PolicyValueNet().to(DEVICE).eval()
    net.load_state_dict(checkpoint["model"])
    fixtures, trajectories = collect_fixtures(net, config, DEVICE, args.games, args.roots_per_game, args.trajectory_sims)
    emit("fixtures", [f["metadata"] for f in fixtures])
    serial = []
    for index, fixture in enumerate(fixtures):
        for strong in (False, True):
            seed = 20261025 + index
            warm = run_case(net, config, DEVICE, fixture, strong, seed, "plain")
            plain = [run_case(net, config, DEVICE, fixture, strong, seed, "plain")]
            instrumented = [run_case(net, config, DEVICE, fixture, strong, seed, mode)
                            for mode in ("cprofile", "boundaries")]
            plain.extend(run_case(net, config, DEVICE, fixture, strong, seed, "plain")
                         for _ in range(args.repeats - 1))
            assert all(r["exact"] == warm["exact"] for r in plain + instrumented), "instrumentation changed search"
            serial.extend(plain + instrumented)
            emit("serial", dict(index=index, strong=strong, plain_seconds=statistics.mean(r["wall_seconds"] for r in plain),
                                cprofile_seconds=instrumented[0]["wall_seconds"], boundary_seconds=instrumented[1]["wall_seconds"]))
    del net
    gc.collect()
    torch.cuda.empty_cache()
    parallel = parallel_probe(checkpoint["model"], config, fixtures, args.workers, args.parallel_pairs)
    assert all(rr.file_record(path)["sha256"] == expected for path, expected in hashes.items()), "source/production changed"
    result = dict(source_hashes=hashes, production_unchanged=True, checkpoint_iteration=checkpoint["train_state"]["iteration"],
                  args={k: str(v) for k, v in vars(args).items()}, config=asdict(config),
                  native=jq.RULES_BACKEND_NATIVE, native_search=az.SEARCH_BACKEND_NATIVE,
                  search_backend=az.SEARCH_BACKEND, search_backend_error=az.SEARCH_BACKEND_ERROR,
                  legal_backend=jq.LEGAL_BACKEND, legal_backend_error=jq.LEGAL_BACKEND_ERROR,
                  native_legal=jq.LEGAL_BACKEND_NATIVE,
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  trajectories=trajectories, fixtures=[f["metadata"] for f in fixtures], serial=serial, parallel=parallel,
                  aggregate_profiles={str(strong): aggregate_profiles([r for r in serial if r["mode"] == "cprofile" and r["strong"] == strong])
                                      for strong in (False, True)},
                  limitations=["cProfile wall attribution is not a CPU/GPU production percentage",
                               "Native build has no internal profiling hooks; native time may be attributed to its Python caller",
                               "Boundary timers add overhead; rules_total minus native_inside_rules still includes timer overhead",
                               "Evaluation includes encoding, transfer, GPU execution and waiting; nested encoding times are not additive",
                               "Capture-biased and controlled endgames are diagnostic fixtures, not production frequency estimates",
                               "Ordinary and strong are measured separately, not a production-weighted full round",
                               "No sample/model saving, no optimization implemented, no full 1024-game round"])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    emit("complete", dict(output=str(args.output), roots=len(fixtures), exact_serial_cases=2*len(fixtures), production_unchanged=True))


if __name__ == "__main__":
    main()
