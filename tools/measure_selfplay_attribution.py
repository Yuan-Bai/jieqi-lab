"""Independent short CPU/thread-time and CUDA-event attribution, not an optimizer."""
from __future__ import annotations

import __future__
import argparse
import ast
import concurrent.futures as futures
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict
from functools import wraps
import gc
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import statistics
import sys
import textwrap
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import azero as az
import ismcts as im
import jieqi as jq
import run_records as rr
from tools import profile_selfplay as probe
from tools.benchmark_chance_allocation import NetworkTrace, canonical_graph
from tools.measure_search_reuse import verify_hashes
from tools.measure_selfplay_round import append_event, runtime_identity

DIRECTORY = None
MEASURE_LOGIC = True
GPU_PHASES = {"h2d", "forward", "softmax", "policy_d2h", "value_d2h"}


class Attribution:
    """Main-thread scopes; GPU event spans are NOT pure kernel execution times."""

    def __init__(self, cuda_events=True, labels=False, logic=True):
        self.cuda_events, self.labels, self.logic = cuda_events, labels, logic
        self.rows, self.stack, self.events = {}, [], []

    @contextmanager
    def scope(self, name, gpu=False):
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if (
            gpu and self.cuda_events) else None
        label = torch.profiler.record_function("jieqi/"+name) if self.labels else nullcontext()
        frame = dict(wall_child=0.0, cpu_child=0.0)
        start, cpu_start = time.perf_counter(), time.thread_time()
        self.stack.append(frame)
        try:
            with label:
                if events:
                    events[0].record()
                try:
                    yield
                finally:
                    if events:
                        events[1].record()
        finally:
            wall, cpu = time.perf_counter()-start, time.thread_time()-cpu_start
            self.stack.pop()
            row = self.rows.setdefault(name, dict(calls=0, wall_inclusive=0.0, thread_cpu_inclusive=0.0,
                wall_exclusive=0.0, thread_cpu_exclusive=0.0))
            row["calls"] += 1
            row["wall_inclusive"] += wall
            row["thread_cpu_inclusive"] += cpu
            row["wall_exclusive"] += wall-frame["wall_child"]
            row["thread_cpu_exclusive"] += cpu-frame["cpu_child"]
            if self.stack:
                self.stack[-1]["wall_child"] += wall
                self.stack[-1]["cpu_child"] += cpu
            if events:
                self.events.append((name, events))

    def call(self, name, function, *args, **kwargs):
        with self.scope(name, name in GPU_PHASES):
            return function(*args, **kwargs)

    def wrapper(self, name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            return self.call(name, function, *args, **kwargs)
        return measured

    def finish(self):
        # The caller has already synchronized once at the original search/game boundary.
        for name, events in self.events:
            row = self.rows[name]
            row["cuda_event_seconds"] = row.get("cuda_event_seconds", 0.0) + events[0].elapsed_time(events[1])/1000
        self.events.clear()
        return self.rows

    def install(self, stack):
        for attribute in ("_evaluate", "_evaluate_batch"):
            function = instrument_evaluation(getattr(az.NeuralMCTS, attribute), self)
            stack.enter_context(mock.patch.object(az.NeuralMCTS, attribute, self.wrapper("evaluation", function)))
        targets = [
            (jq.GameState, "legal_moves", "legal"),
            (jq.GameState, "has_legal_moves", "legal"),
            (jq.GameState, "_move_scan", "outcome_scan"),
            (jq.GameState, "apply_move", "apply_move"),
            (im.InformationState, "determinize", "determinize"),
            (im.BeliefModel, "info_state", "info_snapshot"),
            (im.BeliefModel, "information_set_key", "info_key"),
            (im.BeliefModel, "observe_move", "observe_move"),
            (az.PUCTNode, "expand", "expand"),
            (az.PUCTNode, "select_child", "select"),
            (az.NeuralMCTS, "_chance_specs", "chance_specs"),
            (az, "leaf_input_signature", "leaf_signature"),
        ] if self.logic else []
        for owner, attribute, name in targets:
            stack.enter_context(mock.patch.object(owner, attribute, self.wrapper(name, getattr(owner, attribute))))


def instrument_evaluation(function, attribution):
    """Wrap calls in the current AST; retain expressions, argument and operation order."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    transformed = []

    class Calls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            target, phase = node.func, None
            if isinstance(target, ast.Name) and target.id in ("_encode_batch_numpy", "encode_state_numpy"):
                phase = "encode"
            elif isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id == "self" and target.attr == "net":
                    phase = "forward"
                elif isinstance(target.value, ast.Name) and target.value.id == "torch" and target.attr == "softmax":
                    phase = "softmax"
                elif target.attr == "to":
                    phase = "h2d"
                elif target.attr == "cpu":
                    phase = "policy_d2h"
                elif target.attr in ("tolist", "item"):
                    phase = "value_d2h"
            if phase:
                transformed.append(phase)
                return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id="_attribution", ctx=ast.Load()),
                    attr="call", ctx=ast.Load()), args=[ast.Constant(phase), target, *node.args],
                    keywords=node.keywords), node)
            return node

    tree = Calls().visit(tree)
    if sorted(transformed) != sorted(["encode", "h2d", "forward", "softmax", "policy_d2h", "value_d2h"]):
        raise ValueError("evaluation changed: update and revalidate the attribution instrumenter")
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__, _attribution=attribution)
    exec(compile(tree, inspect.getsourcefile(function), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[function.__name__]


def init_worker(weights, config, expected, directory, logic=True):
    global DIRECTORY, MEASURE_LOGIC
    if runtime_identity() != expected:
        raise RuntimeError("worker backend/precision identity differs")
    DIRECTORY = Path(directory)
    MEASURE_LOGIC = logic
    probe.init_worker(weights, config, jq.RULES_BACKEND_NATIVE,
                      (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE), (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))


def timed_task(task):
    kind, mode, payload, strong, seed = task
    observer = Attribution(labels=mode == "torchtrace", logic=MEASURE_LOGIC) if mode != "plain" else None
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False,
                                      with_stack=False) if mode == "torchtrace" else None
    with ExitStack() as stack:
        if observer:
            observer.install(stack)
        if profiler:
            stack.enter_context(profiler)
        if kind == "root":
            search_function = az.NeuralMCTS.search
            thread_seconds = [0.0]

            def thread_search(*args, **kwargs):
                start = time.thread_time()
                try:
                    return search_function(*args, **kwargs)
                finally:
                    thread_seconds[0] += time.thread_time()-start

            stack.enter_context(mock.patch.object(az.NeuralMCTS, "search", thread_search))
            row = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, payload, strong, seed, "plain")
            row["thread_cpu_seconds"] = thread_seconds[0]
        elif kind == "game":
            rng = random.Random(seed)
            probe.synchronize(probe.DEVICE)
            start, cpu, thread = time.perf_counter(), time.process_time(), time.thread_time()
            game = az.self_play_game(probe.NET, probe.CONFIG, probe.DEVICE, rng, strong=strong)
            probe.synchronize(probe.DEVICE)
            row = dict(wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu,
                       thread_cpu_seconds=time.thread_time()-thread)
            if game.outcome is None:
                raise AssertionError("unfinished game")
            row["exact"] = dict(seed=seed, strong=strong, moves=game.move_count, outcome=game.outcome,
                samples=len(game.samples), signature=probe.digest(asdict(game)), rng=probe.digest(rng.getstate()))
        else:
            raise ValueError("unknown task kind")
    row.update(kind=kind, mode=mode, pid=os.getpid(), stages=observer.finish() if observer else None)
    if profiler:
        path = DIRECTORY/f"cpu_trace_{os.getpid()}_{seed}_{int(strong)}.json"
        if path.exists():
            raise FileExistsError(path)
        profiler.export_chrome_trace(str(path))
        row["trace"] = str(path)
        row["host_operators"] = [dict(name=e.key, calls=e.count, self_cpu_seconds=e.self_cpu_time_total/1e6,
            cpu_seconds=e.cpu_time_total/1e6) for e in sorted(profiler.key_averages(),
                key=lambda e: -e.self_cpu_time_total)[:35]]
    return row


def graph_checks(weights, config, fixtures, logic=True):
    net = az.PolicyValueNet().to("cuda").eval()
    az.load_state_dict_padded(net, weights)
    original, rows = az.NeuralMCTS.search, []
    for fixture in fixtures:
        for strong in (False, True):
            outputs = []
            for mode in ("plain", "stages"):
                trace, retained = NetworkTrace(), []

                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original(search, *args, **kwargs)
                    retained.append(root)
                    return policy, move

                observer = Attribution(logic=logic) if mode == "stages" else None
                hook = net.register_forward_hook(trace.hook)
                try:
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(az.NeuralMCTS, "search", capture))
                        if observer:
                            observer.install(stack)
                        row = probe.run_case(net, config, torch.device("cuda"), fixture, strong,
                                             202610084+fixture["metadata"]["index"], "plain")
                    if observer:
                        observer.finish()
                    outputs.append(dict(exact=row["exact"], graph=canonical_graph(retained[0]), network=trace.rows))
                finally:
                    hook.remove()
            if outputs[0] != outputs[1]:
                raise AssertionError("attribution changed graph/network/strategy/RNG")
            rows.append(dict(index=fixture["metadata"]["index"], strong=strong,
                             network_requests=len(outputs[0]["network"]), exact=True))
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def validate_config(config):
    if (config.self_play_device not in ("cuda", "cuda:0") or config.leaf_batch_size != 64
            or config.virtual_loss != 3.0 or config.pcr_enabled or not config.chance_nodes
            or config.chance_k != 0 or not config.lazy_chance_allocation or not config.leaf_dedup
            or config.mcts_simulations != 800 or config.strong_mcts_simulations != 3200
            or config.strong_self_play_interval != 8):
        raise ValueError("require current CUDA/leaf64/virtual3/800+3200/chance-full-lazy/dedup/PCR-off protocol")


def paired_overhead(groups):
    pairs = sorted({row["repeat"] for row in groups})
    timing = {(r["repeat"], r["mode"]): r["wall_seconds"] for r in groups}
    values = [timing[p, "stages"]/timing[p, "plain"]-1 for p in pairs]
    return dict(pairs=len(pairs), cohort_overhead=values, mean_overhead=statistics.mean(values),
                aggregate_overhead=sum(timing[p, "stages"] for p in pairs)/sum(timing[p, "plain"] for p in pairs)-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"data/pilots/chance_support_20261008_iter28/frozen_inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=ROOT/"data/pilots/chance_support_20261008_iter28/measure.fixtures.pkl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--root-repeats", type=int, default=4)
    parser.add_argument("--game-repeats", type=int, default=2)
    parser.add_argument("--games-per-cohort", type=int, default=16)
    parser.add_argument("--measurement-level", choices=("full", "evaluation"), default="full")
    args = parser.parse_args()
    directory, source = args.output_dir.resolve(), args.run_dir.resolve()
    if directory.exists() or ROOT/"data/pilots" not in directory.parents or source == directory or source in directory.parents:
        parser.error("use a new output directory below data/pilots, outside source inputs")
    if (args.workers != 8 or args.root_repeats < 2 or (args.game_repeats < 2 and not (
            args.measurement_level == "evaluation" and args.game_repeats == 0))
            or args.games_per_cohort < 8 or args.games_per_cohort % 8):
        parser.error("require eight workers, >=2 paired cohorts, full groups of eight; evaluation may omit games")
    if (jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython"
            or az.SEARCH_BACKEND != "python" or not torch.cuda.is_available()):
        parser.error("require CUDA, rules/legal Cython and Python nodes")
    torch.set_num_threads(1)
    checkpoint_path = source/"models/training_checkpoint.pt"
    before = rr.file_record(checkpoint_path)["sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if rr.file_record(checkpoint_path)["sha256"] != before:
        raise RuntimeError("checkpoint changed while loading")
    config = az.AZConfig(**checkpoint["config"])
    validate_config(config)
    config.data_dir, config.model_dir, config.log_dir = (str(directory/n) for n in ("samples", "models", "logs"))
    with args.fixtures.open("rb") as handle:
        fixtures = pickle.load(handle)["fixtures"]
    if len(fixtures) != 18 or [f["metadata"]["index"] for f in fixtures] != list(range(18)):
        parser.error("require trusted current 18-position fixtures")
    production = ROOT/"data/train_rules5_20261004"
    paths = [production/n for n in ("models/training_checkpoint.pt", "models/latest.pt", "models/train_state.json", "logs/train.log")]
    paths += [production/"selfplay/iter_20261008_080731_0027.pkl.gz", checkpoint_path, args.fixtures.resolve()]
    paths += [ROOT/n for n in ("azero.py", "ismcts.py", "jieqi.py", "native/jieqi_rules.pyx", "native/jieqi_legal.pyx",
        "tools/profile_selfplay.py", "tools/benchmark_chance_allocation.py", "tools/measure_selfplay_round.py")]
    paths += [Path(__file__).resolve(), Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"])]
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    if hashes[str(checkpoint_path)] != before:
        raise RuntimeError("checkpoint changed before measurement")
    directory.mkdir(parents=True)
    origin = time.perf_counter()

    def record(event, value):
        append_event(directory/"events.jsonl", event, dict(elapsed=time.perf_counter()-origin, data=value))
        probe.emit(event, value)

    identity = runtime_identity()
    result = dict(args={k:str(v) for k,v in vars(args).items()}, source_hashes=hashes, identity=identity,
        iteration=checkpoint["train_state"]["iteration"], torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), config=asdict(config),
        profiler_supported_activities=[str(a) for a in torch.profiler.supported_activities()],
        limitations=["No native CPU stacks: Nsight/WPR require Windows Administrator, current token is not elevated",
            "Installed torch profiler supports CPU activities only; CPU trace has no CUDA kernel timeline",
            "CUDA event spans include queueing/preemption/host enqueue gaps, not pure kernel times",
            "Main-thread CPU in D2H is not proof of driver busy-wait without native stacks",
            "Exclusive scopes partition named boundaries; unlabeled work and instrumentation remain residual",
            "Frozen-root cohorts and short full games, not a new 1024-game or 4M-resident round",
            "Worker IPC returns compact audit summaries, not production full samples; no throughput claim"])
    record("metadata", result)
    result["graph_checks"] = graph_checks(checkpoint["model"], config, fixtures, args.measurement_level == "full")
    record("graph_checks", dict(pairs=len(result["graph_checks"]), exact=True))
    roots = [("root", "plain", f, strong, 202610083+i) for i,f in enumerate(fixtures) for strong in (False, True)]
    result["roots"], result["games"] = [], []
    with futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"), initializer=init_worker,
            initargs=(checkpoint["model"], asdict(config), identity, str(directory), args.measurement_level == "full")) as pool:
        list(pool.map(timed_task, roots))
        list(pool.map(timed_task, [(t[0], "stages", *t[2:]) for t in roots]))
        for kind, repeats in (("root", args.root_repeats), ("game", args.game_repeats)):
            for repeat in range(repeats):
                tasks = roots if kind == "root" else [("game", "plain", i, (i+1)%8 == 0,
                    202610085+repeat*100+i) for i in range(args.games_per_cohort)]
                exact = {}
                for mode in (("plain", "stages") if repeat%2 == 0 else ("stages", "plain")):
                    start = time.perf_counter()
                    rows = list(pool.map(timed_task, [(t[0], mode, *t[2:]) for t in tasks]))
                    group = dict(kind=kind, repeat=repeat, mode=mode, wall_seconds=time.perf_counter()-start, rows=rows)
                    result["roots" if kind == "root" else "games"].append(group)
                    exact[mode] = [r["exact"] for r in rows]
                    record("cohort", {k:v for k,v in group.items() if k != "rows"})
                    if kind == "root" and mode == "plain":
                        baseline_roots = rows
                if exact["plain"] != exact["stages"]:
                    raise AssertionError("instrumentation changed policy/RNG/counters or full game samples")
        # Two CPU operator traces amid an eight-search cohort; not a throughput experiment.
        trace_tasks = roots[::2][:7]+[roots[-1]]
        trace_tasks = [(t[0], "torchtrace" if i in (0,7) else "stages", *t[2:]) for i,t in enumerate(trace_tasks)]
        result["traces"] = list(pool.map(timed_task, trace_tasks))
        exact_roots = {(r["index"], r["strong"]): r["exact"] for r in baseline_roots}
        if any(r["exact"] != exact_roots[r["index"],r["strong"]] for r in result["traces"]):
            raise AssertionError("CPU operator profiler changed policy/RNG/counters")
    result["root_overhead"] = paired_overhead(result["roots"])
    result["game_overhead"] = paired_overhead(result["games"]) if result["games"] else None
    verify_hashes(hashes)
    result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-origin)
    with (directory/"result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    record("complete", dict(elapsed=result["elapsed_seconds"], root_overhead=result["root_overhead"],
        game_overhead=result["game_overhead"], production_unchanged=True))


if __name__ == "__main__":
    main()
