"""Exact pre-change search versus direct actor keys and leaf-only snapshots."""
from __future__ import annotations

import __future__
import argparse
import ast
import concurrent.futures as futures
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import statistics
import sys
import time
from types import ModuleType
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import azero as az
import ismcts as im
import jieqi as jq
import run_records as rr
from tools import profile_selfplay as probe
from tools.benchmark_chance_allocation import NetworkTrace, canonical_graph
from tools.measure_search_reuse import verify_hashes

MODES = ("snapshots", "direct")
REFERENCE = None
DIRECT = az.NeuralMCTS.search


def extract_function(source, name, namespace, filename, class_name=None):
    nodes = ast.parse(source, filename=filename).body
    if class_name is not None:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    function = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    if function.decorator_list:
        raise ValueError("reference must be an undecorated function")
    module = ast.Module(body=[function], type_ignores=[])
    scope = dict(namespace)
    exec(compile(module, filename, "exec", flags=__future__.annotations.compiler_flag), scope)
    return scope[name], function


def load_reference(before):
    # Execute only two trusted local function definitions, not an old module/import graph.
    key, key_ast = extract_function((before/"ismcts.py").read_text(encoding="utf-8-sig"),
                                   "information_set_key", vars(im), str(before/"ismcts.py"))
    if not any(isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
               and len(n.value.elts) == 15 for n in ast.walk(key_ast)):
        raise ValueError("reference key must contain the original 15 fields")
    namespace = dict(vars(az))
    legacy_im = ModuleType(im.__name__)
    legacy_im.__dict__.update(vars(im))
    legacy_im.information_set_key = key
    namespace["im"] = legacy_im
    search, search_ast = extract_function((before/"azero.py").read_text(encoding="utf-8-sig"),
                                         "search", namespace, str(before/"azero.py"), "NeuralMCTS")
    calls = [n for n in ast.walk(search_ast) if isinstance(n, ast.Call)]
    if (sum(isinstance(n.func, ast.Attribute) and n.func.attr == "info_state" for n in calls) != 2
            or any(isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                   and n.func.value.id == "sim_view" and n.func.attr == "information_set_key" for n in calls)):
        raise ValueError("reference must use full snapshots at root and intermediate layers")
    if not any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Name)
               and n.value.id == "actor_info" and any(isinstance(t, ast.Name) and t.id == "leaf_info"
                                                     for t in n.targets) for n in ast.walk(search_ast)):
        raise ValueError("reference must reuse the last intermediate snapshot at the leaf")
    return search


def paired_summary(trials):
    repeats = sorted({r["repeat"] for r in trials})
    timing = {(r["repeat"], r["mode"]): r["wall_seconds"] for r in trials}
    values = [1-timing[r, "direct"]/timing[r, "snapshots"] for r in repeats]
    rng = random.Random(202610082)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000)]
    totals = {m: sum(timing[r, m] for r in repeats) for m in MODES}
    return dict(mean_wall_saving=statistics.mean(values), paired_values=values,
                bootstrap_mean_interval_95=np.percentile(means, [2.5, 97.5]).tolist(),
                total_seconds=totals, aggregate_wall_saving=1-totals["direct"]/totals["snapshots"],
                throughput_ratio=totals["snapshots"]/totals["direct"])


def init_worker(weights, config, rules, search, legal, before):
    global REFERENCE
    probe.init_worker(weights, config, rules, search_backend=search, legal_backend=legal)
    REFERENCE = load_reference(before)


def search_task(task):
    mode, fixture, strong, seed = task
    az._cached_full_chance_outcomes.cache_clear()
    with mock.patch.object(az.NeuralMCTS, "search", REFERENCE if mode == "snapshots" else DIRECT):
        return probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")


def game_task(task):
    mode, index, strong, seed = task
    az._cached_full_chance_outcomes.cache_clear()
    rng = random.Random(seed)
    with mock.patch.object(az.NeuralMCTS, "search", REFERENCE if mode == "snapshots" else DIRECT):
        probe.synchronize(probe.DEVICE)
        start, cpu_start = time.perf_counter(), time.process_time()
        game = az.self_play_game(probe.NET, probe.CONFIG, probe.DEVICE, rng, strong=strong)
        probe.synchronize(probe.DEVICE)
        seconds, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu_start
    if game.outcome is None:
        raise AssertionError("unfinished game")
    exact = dict(index=index, seed=seed, strong=strong, moves=game.move_count, outcome=game.outcome,
                 samples=len(game.samples), leaf_total=game.leaf_evals_total, leaf_unique=game.leaf_evals_unique,
                 sample_sha256=probe.digest([asdict(s) for s in game.samples]), rng_sha256=probe.digest(rng.getstate()))
    return dict(mode=mode, seconds=seconds, process_cpu_seconds=cpu_seconds, exact=exact)


def graph_checks(net, config, fixtures, record, seed_offset=0):
    rows = []
    original_info = im.BeliefModel.info_state
    for fixture in fixtures:
        for strong in (False, True):
            reference = None
            for mode in MODES:
                az._cached_full_chance_outcomes.cache_clear()
                trace, roots, counts = NetworkTrace(), [], [0]
                function = REFERENCE if mode == "snapshots" else DIRECT

                def snapshot(view, *args):
                    counts[0] += 1
                    return original_info(view, *args)

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = function(search, *args, **kwargs)
                    roots.append(root)
                    return policy, move

                hook = net.register_forward_hook(trace.hook)
                try:
                    with mock.patch.object(az.NeuralMCTS, "search", capture), mock.patch.object(
                            im.BeliefModel, "info_state", snapshot):
                        result = probe.run_case(net, config, probe.DEVICE, fixture, strong,
                            20285025+fixture["metadata"]["index"]+seed_offset, "plain")
                    graph = canonical_graph(roots[0])
                finally:
                    hook.remove()
                signature = result["exact"], trace.rows, graph
                if reference is None:
                    reference = signature
                if signature != reference:
                    raise AssertionError("network/logical graph/RNG mismatch")
                row = dict(index=fixture["metadata"]["index"], strong=strong, mode=mode,
                           snapshots=counts[0], exact=result["exact"], graph_sha256=graph,
                           forward_trace_sha256=probe.digest(trace.rows), forward_calls=len(trace.rows))
                rows.append(row)
                del roots, trace, signature
                gc.collect()
            record("graph_exact", dict(index=fixture["metadata"]["index"], strong=strong,
                                       snapshots={r["mode"]: r["snapshots"] for r in rows[-2:]}))
    return rows


def timed_trials(pool, fixtures, config, args, record, games=False):
    trials = []
    for repeat in range(args.game_rounds if games else args.search_rounds):
        exact = {}
        for mode in (MODES if repeat % 2 == 0 else tuple(reversed(MODES))):
            if games:
                tasks = [(mode, repeat*args.games_per_round+i,
                          config.strong_self_play_interval > 0 and (i+1) % config.strong_self_play_interval == 0,
                          20285000+repeat*args.games_per_round+i+args.seed_offset) for i in range(args.games_per_round)]
            else:
                tasks = [(mode, f, strong, 20285300+repeat*1000+f["metadata"]["index"]+args.seed_offset)
                         for f in fixtures for strong in (False, True)]
            start = time.perf_counter()
            results = list(pool.map(game_task if games else search_task, tasks))
            row = dict(repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-start,
                       worker_seconds=sum(r["seconds" if games else "wall_seconds"] for r in results), results=results)
            trials.append(row)
            exact[mode] = [r["exact"] for r in results]
            record("game_group" if games else "search_group", row, compact=True)
        if exact["snapshots"] != exact["direct"]:
            raise AssertionError("paired sample/RNG mismatch")
        record("game_exact" if games else "search_exact", dict(repeat=repeat, cases=len(tasks)))
    return trials


def main():
    global REFERENCE
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "before", "fixtures", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--production-dir", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--search-rounds", type=int, default=8)
    parser.add_argument("--game-rounds", type=int, default=6)
    parser.add_argument("--games-per-round", type=int, default=16)
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or not args.output.parent.is_dir():
        parser.error("choose new outputs in an existing directory")
    if (min(args.workers, args.search_rounds, args.games_per_round) < 1 or args.game_rounds < 0
            or args.search_rounds % 2 or args.game_rounds % 2):
        parser.error("positive sizes and even rounds required; game rounds may be zero")
    if (jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython"
            or az.SEARCH_BACKEND != "python" or not torch.cuda.is_available()):
        parser.error("require CUDA, rules/legal Cython, Python nodes")
    torch.set_num_threads(1)
    paths = [ROOT/n for n in ("azero.py", "ismcts.py", "jieqi.py", "run_records.py",
                             "native/jieqi_rules.pyx", "native/jieqi_legal.pyx",
                             "tools/profile_selfplay.py", "tools/benchmark_chance_allocation.py",
                             "tools/measure_search_reuse.py")]
    paths += [Path(__file__), args.fixtures, args.before/"azero.py", args.before/"ismcts.py",
              Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"])]
    paths += [p for p in (args.before.parent/"protocol.md", args.before.parent/"protocol_module_reference.md")
              if p.exists()]
    for directory in dict.fromkeys((args.run_dir, args.production_dir)):
        paths += [directory/n for n in ("models/training_checkpoint.pt", "models/latest.pt",
                                        "models/train_state.json", "logs/train.log")]
        saved = torch.load(directory/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
        paths.append(directory/"selfplay"/saved["replay_files"][-1]["name"])
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    checkpoint = torch.load(args.run_dir/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
    config = az.AZConfig(**checkpoint["config"])
    if (config.pcr_enabled or not config.chance_nodes or config.chance_k != 0
            or not config.lazy_chance_allocation or not config.leaf_dedup):
        parser.error("require fixed budgets, chance k=0/lazy and leaf dedup")
    if config.strong_self_play_interval > 0 and args.games_per_round % config.strong_self_play_interval:
        parser.error("game cohorts must include whole strong intervals")
    REFERENCE = load_reference(args.before)
    result = dict(args={k: str(v) for k, v in vars(args).items()}, source_hashes=hashes,
                  checkpoint_iteration=checkpoint["train_state"]["iteration"], config=asdict(config),
                  rules_native=jq.RULES_BACKEND_NATIVE, legal_native=jq.LEGAL_BACKEND_NATIVE,
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  chance_support_cache_in_both_modes=True)
    start = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle:
        def record(event, value, compact=False):
            handle.write(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter()-start, value=value))+"\n")
            handle.flush()
            probe.emit(event, {k:v for k,v in value.items() if k != "results"} if compact else value)

        record("metadata", result)
        with args.fixtures.open("rb") as source:
            fixtures = pickle.load(source)["fixtures"]
        result["fixtures"] = [f["metadata"] for f in fixtures]
        net = az.PolicyValueNet().to(probe.DEVICE).eval()
        net.load_state_dict(checkpoint["model"])
        result["graph_checks"] = graph_checks(net, config, fixtures, record, args.seed_offset)
        del net
        gc.collect()
        torch.cuda.empty_cache()
        identities = (checkpoint["model"], asdict(config), jq.RULES_BACKEND_NATIVE,
                      (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE), (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE), args.before)
        with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                         initializer=init_worker, initargs=identities) as pool:
            for mode in MODES:
                list(pool.map(search_task, [(mode, f, strong, 20285100+args.seed_offset) for f in fixtures for strong in (False, True)]))
            for games in (False, True):
                if games and not args.game_rounds:
                    continue
                prefix = "game" if games else "search"
                result[prefix+"_trials"] = timed_trials(pool, fixtures, config, args, record, games=games)
                result[prefix+"_summary"] = paired_summary(result[prefix+"_trials"])
                record(prefix+"_summary", result[prefix+"_summary"])
        verify_hashes(hashes)
        result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-start,
                      limitations=["Frozen fixed-model self-play, not optimizer updates or a 1024-game production iteration",
                                   "Network trace and snapshot counting are outside throughput timings",
                                   "Cohort timing includes digest/compact IPC, not full sample save/augmentation",
                                   "Bootstrap over a small number of paired cohorts is descriptive",
                                   "Both modes keep the already-retained chance support cache; cleared per root/game"])
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
        record("complete", dict(output=str(args.output), production_unchanged=True))


if __name__ == "__main__":
    main()
