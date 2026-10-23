"""Read-only chance-allocation and exact neural-input reuse measurements."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import concurrent.futures as futures
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import azero as az
import jieqi as jq
import run_records as rr
from tools import profile_selfplay as probe


def shallow_node_bytes(node):
    # Excludes dict keys/values and scalar objects; not total graph memory.
    return sum(sys.getsizeof(item) for item in
               (node, node.children, node.public_outcomes, node.decision_nodes))


def count_graph(root):
    """Count reachable objects once, even when multiple edges share decisions."""
    categories = {name: dict(total=0, unvisited=0, shallow_bytes=0, unvisited_shallow_bytes=0)
                  for name in ("all_nodes", "chance_entries", "chance_layers",
                               "chance_results", "chance_children")}
    seen, chance_children, pending = set(), set(), [root]

    def add(name, node):
        row = categories[name]
        size = shallow_node_bytes(node)
        row["total"] += 1
        row["shallow_bytes"] += size
        if node.visits == 0:
            row["unvisited"] += 1
            row["unvisited_shallow_bytes"] += size

    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        add("all_nodes", node)
        if node.is_chance:
            add("chance_entries" if node.move is not None else "chance_layers", node)
            for child in node.children.values():
                if id(child) not in chance_children:
                    chance_children.add(id(child))
                    add("chance_children", child)
                    if not child.is_chance:
                        add("chance_results", child)
        pending.extend(node.children.values())
        pending.extend(node.public_outcomes.values())
        pending.extend(node.decision_nodes.values())
    return categories


class InputReuse:
    """Observe actual float32 inputs without substituting any network outputs.

    Lossless compressed bytes give collision-free input equality. Candidate keys
    and LRU tables are measured separately; their values are only audit records.
    A lifetime is one real game, never an unrelated repeated benchmark fixture.
    """

    def __init__(self, capacities=(1024, 4096)):
        self.game_seen = {}
        self.lrus = {capacity: OrderedDict() for capacity in capacities}
        self.start_search()

    def start_search(self):
        self.search_seen = {}
        self.candidates = {}
        self.call = 0
        names = ("rows", "root_rows", "leaf_rows", "calls", "first_game_inputs",
                 "within_batch_repeats", "search_prior_batch_hits", "game_additional_hits",
                 "candidate_prior_batch_hits", "candidate_false_hits", "packed_input_bytes",
                 "encoder_seconds", "exact_key_seconds", "exact_table_seconds",
                 "candidate_key_seconds", "candidate_lookup_seconds", "candidate_admit_seconds")
        self.stats = {name: 0 for name in names}
        for capacity in self.lrus:
            self.stats.update({f"lru_{capacity}_hits": 0, f"lru_{capacity}_seconds": 0,
                               f"lru_{capacity}_evictions": 0})

    def observe(self, batch, infos, histories, repetition_entries, role):
        if batch.dtype != np.float32 or batch.shape != (len(infos), az.STATE_CHANNELS, 10, 9):
            raise AssertionError("unexpected actual network input format")
        self.call += 1
        self.stats["calls"] += 1
        self.stats["rows"] += len(batch)
        self.stats[f"{role}_rows"] += len(batch)
        start = time.perf_counter()
        packed = [zlib.compress(row.tobytes(order="C"), level=1) for row in batch]
        self.stats["exact_key_seconds"] += time.perf_counter() - start
        self.stats["packed_input_bytes"] += sum(map(len, packed))

        start = time.perf_counter()
        entries = (repetition_entries if repetition_entries is not None else
                   [az._repetition_entries(info) for info in infos])
        keys = [az.leaf_input_signature(info, history[-az.HISTORY_PLIES:], repetition_entries=entry)
                for info, history, entry in zip(infos, histories, entries)]
        self.stats["candidate_key_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        for encoded in packed:
            previous_call = self.search_seen.get(encoded)
            if previous_call == self.call:
                self.stats["within_batch_repeats"] += 1
            elif previous_call is not None:
                self.stats["search_prior_batch_hits"] += 1
            elif encoded in self.game_seen:
                self.stats["game_additional_hits"] += 1
            else:
                self.stats["first_game_inputs"] += 1
            self.search_seen[encoded] = self.call
            self.game_seen[encoded] = True
        self.stats["exact_table_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        previous = [self.candidates.get(key) for key in keys]
        self.stats["candidate_lookup_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        for key, encoded, old in zip(keys, packed, previous):
            if old is None:
                old = self.candidates.get(key)
            if old is not None:
                if old[0] != encoded:
                    self.stats["candidate_false_hits"] += 1
                    raise AssertionError("candidate key merged different actual inputs")
                if old[1] < self.call:
                    self.stats["candidate_prior_batch_hits"] += 1
            self.candidates[key] = (encoded, self.call)
        self.stats["candidate_admit_seconds"] += time.perf_counter() - start

        for capacity, table in self.lrus.items():
            start = time.perf_counter()
            hits = evictions = 0
            for key, encoded in zip(keys, packed):
                old = table.get(key)
                if old is not None:
                    if old != encoded:
                        raise AssertionError("LRU candidate key merged different actual inputs")
                    hits += 1
                    table.move_to_end(key)
                else:
                    table[key] = encoded
                    if len(table) > capacity:
                        table.popitem(last=False)
                        evictions += 1
            self.stats[f"lru_{capacity}_seconds"] += time.perf_counter() - start
            self.stats[f"lru_{capacity}_hits"] += hits
            self.stats[f"lru_{capacity}_evictions"] += evictions


class SearchAudit:
    def __init__(self, measured, graph_hash=False):
        self.measured = measured
        self.graph_hash = graph_hash
        self.inputs = InputReuse() if measured else None
        self.rows = []
        self.active = False

    @contextmanager
    def install(self):
        original_search = az.NeuralMCTS.search
        original_batch = az._encode_batch_numpy
        original_root = az.encode_state_numpy

        def batch(infos, histories, *, repetition_entries=None):
            start = time.perf_counter()
            result = original_batch(infos, histories, repetition_entries=repetition_entries)
            if self.active:
                self.inputs.stats["encoder_seconds"] += time.perf_counter() - start
                self.inputs.observe(result, infos, histories, repetition_entries, "leaf")
            return result

        def root(info, history):
            start = time.perf_counter()
            result = original_root(info, history)
            if self.active:
                self.inputs.stats["encoder_seconds"] += time.perf_counter() - start
                self.inputs.observe(result[None], [info], [history], None, "root")
            return result

        def search(mcts, state, belief, history, *args, **kwargs):
            if self.active:
                raise AssertionError("nested search is unsupported by this audit")
            requested_root = kwargs.get("return_root", False)
            kwargs["return_root"] = True
            if self.measured:
                self.inputs.start_search()
            before = (mcts.leaf_evals_total, mcts.leaf_evals_unique)
            self.active = True
            start = time.perf_counter()
            try:
                policy, move, tree = original_search(mcts, state, belief, history, *args, **kwargs)
            finally:
                seconds = time.perf_counter() - start
                self.active = False
            pieces = sum(bool(p) for p in state.board)
            row = dict(ply=len(history), pieces=pieces,
                       dark_pieces=sum(bool(p) and jq.is_dark(p) for p in state.board),
                       stage="endgame" if pieces <= 12 else
                       ("opening" if len(history) < 10 and pieces >= 26 else "middlegame"),
                       simulations=kwargs.get("simulations") or mcts.config.mcts_simulations,
                       search_seconds=seconds,
                       leaf_total=mcts.leaf_evals_total-before[0], leaf_unique=mcts.leaf_evals_unique-before[1])
            if self.measured:
                row["inputs"] = self.inputs.stats.copy()
                start = time.perf_counter()
                row["graph"] = count_graph(tree)
                row["graph_scan_seconds"] = time.perf_counter() - start
            if self.graph_hash:
                row["graph_sha256"] = probe.digest(tree)
            self.rows.append(row)
            return (policy, move, tree) if requested_root else (policy, move)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(az.NeuralMCTS, "search", search))
            if self.measured:
                stack.enter_context(mock.patch.object(az, "_encode_batch_numpy", batch))
                stack.enter_context(mock.patch.object(az, "encode_state_numpy", root))
            yield self


def init_worker(weights, config, rules, search, legal):
    probe.init_worker(weights, config, rules, search_backend=search, legal_backend=legal)


def fixed_task(task):
    fixture, strong, seed, mode = task
    audit = SearchAudit(mode == "audit", graph_hash=True)
    with audit.install():
        result = probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")
    result.update(mode=mode, audit=audit.rows[0])
    return result


def game_task(task):
    index, seed, strong, mode = task
    audit = SearchAudit(mode == "audit")
    rng = random.Random(seed)
    probe.synchronize(probe.DEVICE)
    start = time.perf_counter()
    with audit.install():
        game = az.self_play_game(probe.NET, probe.CONFIG, probe.DEVICE, rng, strong=strong)
    probe.synchronize(probe.DEVICE)
    seconds = time.perf_counter() - start
    if game.outcome is None:
        raise AssertionError("unfinished game")
    exact = dict(index=index, seed=seed, strong=strong, moves=game.move_count, outcome=game.outcome,
                 samples=len(game.samples), leaf_total=game.leaf_evals_total, leaf_unique=game.leaf_evals_unique,
                 sample_sha256=probe.digest([asdict(s) for s in game.samples]), rng_sha256=probe.digest(rng.getstate()))
    return dict(mode=mode, seconds=seconds, exact=exact, searches=audit.rows)


def summary(searches):
    inputs, graph = {}, {}
    for row in searches:
        for name, value in row.get("inputs", {}).items():
            inputs[name] = inputs.get(name, 0) + value
        for category, counts in row.get("graph", {}).items():
            total = graph.setdefault(category, {name: 0 for name in counts})
            for name, value in counts.items():
                total[name] += value
    for counts in graph.values():
        counts["unvisited_fraction"] = counts["unvisited"] / counts["total"] if counts["total"] else None
    rows = inputs.get("rows", 0)
    inputs["search_cross_batch_fraction"] = inputs.get("search_prior_batch_hits", 0)/rows if rows else None
    inputs["game_additional_fraction"] = inputs.get("game_additional_hits", 0)/rows if rows else None
    for capacity in (1024, 4096):
        inputs[f"lru_{capacity}_hit_fraction"] = inputs.get(f"lru_{capacity}_hits", 0)/rows if rows else None
    return dict(searches=len(searches), search_seconds=sum(r["search_seconds"] for r in searches),
                leaf_total=sum(r["leaf_total"] for r in searches), leaf_unique=sum(r["leaf_unique"] for r in searches),
                graph_scan_seconds=sum(r.get("graph_scan_seconds", 0) for r in searches), inputs=inputs, graph=graph)


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if rr.file_record(path)["sha256"] != expected:
            raise AssertionError(f"protected file changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--trajectory-games", type=int, default=3)
    parser.add_argument("--roots-per-game", type=int, default=6)
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or not args.output.parent.is_dir():
        parser.error("new output/journal in an existing directory required")
    if min(args.workers, args.games, args.trajectory_games, args.roots_per_game) < 1:
        parser.error("positive sample sizes required")
    if (jq.RULES_BACKEND != "cython" or az.SEARCH_BACKEND != "python"
            or jq.LEGAL_BACKEND != "cython" or not torch.cuda.is_available()):
        parser.error("require CUDA, validated rules/legal Cython, and Python search backend")
    torch.set_num_threads(1)
    paths = [args.run_dir/"models/training_checkpoint.pt", args.run_dir/"models/latest.pt",
             args.run_dir/"models/train_state.json", args.run_dir/"logs/train.log",
             ROOT/"azero.py", ROOT/"jieqi.py", ROOT/"ismcts.py", ROOT/"run_records.py",
             ROOT/"native/jieqi_rules.pyx", ROOT/"native/jieqi_search.pyx", ROOT/"native/jieqi_legal.pyx",
             Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"]),
             Path(__file__), Path(probe.__file__)]
    hashes = {str(path): rr.file_record(path)["sha256"] for path in paths}
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir/"selfplay"/checkpoint["replay_files"][-1]["name"]
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    config = az.AZConfig(**checkpoint["config"])
    if config.pcr_enabled or not config.chance_nodes or config.chance_k != 0 or not config.leaf_dedup:
        parser.error("require existing fixed budgets, chance k=0 and leaf dedup on")
    if config.strong_self_play_interval > 0 and args.games % config.strong_self_play_interval:
        parser.error("games must contain whole strong-game intervals")
    result = dict(args={k: str(v) for k, v in vars(args).items()}, source_hashes=hashes,
                  checkpoint_iteration=checkpoint["train_state"]["iteration"], config=asdict(config),
                  runtime=dict(rules_backend=jq.RULES_BACKEND, rules_backend_native=jq.RULES_BACKEND_NATIVE,
                               search_backend=az.SEARCH_BACKEND, search_backend_native=az.SEARCH_BACKEND_NATIVE,
                               legal_backend=jq.LEGAL_BACKEND, legal_backend_native=jq.LEGAL_BACKEND_NATIVE),
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  input_equality="lossless compressed actual float32 input bytes; no digest-only equality",
                  candidate_key="leaf_input_signature with last 8 history plies; validated against actual bytes",
                  cache_scope="one independent search for fixtures; one real game for game/LRU tables",
                  limitations=["No network outputs reused and no visits/statistics merged",
                               "Identical encoded input does not prove batch-shape invariant floating-point outputs",
                               "Audit key compression, validation and graph scanning add overhead",
                               "Shallow bytes exclude dict contents and scalar objects; not total memory",
                               "Baseline and audit use unchanged CUDA batches/budgets/RNG; no optimizer/replay writes"])
    start = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle:
        def record(event, value, display=None):
            handle.write(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter()-start, value=value)) + "\n")
            handle.flush()
            probe.emit(event, value if display is None else display)

        record("metadata", result)
        net = az.PolicyValueNet().to(probe.DEVICE).eval()
        net.load_state_dict(checkpoint["model"])
        fixtures, trajectories = probe.collect_fixtures(net, config, probe.DEVICE, args.trajectory_games, args.roots_per_game, 0)
        result.update(fixtures=[f["metadata"] for f in fixtures], trajectories=trajectories)
        record("fixtures", result["fixtures"])
        del net
        gc.collect()
        torch.cuda.empty_cache()
        with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=init_worker, initargs=(checkpoint["model"], asdict(config), jq.RULES_BACKEND_NATIVE,
                                                   (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE),
                                                   (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))) as pool:
            # Alternate first mode, so audit overhead is not confounded with only cold runs.
            result["fixed_pairs"] = []
            for strong in (False, True):
                pairs = []
                tasks = [(fixture, strong, 20263000+i, mode) for i, fixture in enumerate(fixtures)
                         for mode in (("plain", "audit") if i % 2 == 0 else ("audit", "plain"))]
                for task in tasks:
                    pairs.append(pool.submit(fixed_task, task))
                outputs = [future.result() for future in pairs]
                for i in range(len(fixtures)):
                    a, b = outputs[2*i:2*i+2]
                    if a["exact"] != b["exact"] or a["audit"]["graph_sha256"] != b["audit"]["graph_sha256"]:
                        raise AssertionError("instrumented fixed search changed policy/RNG/tree")
                    pair = dict(index=i, strong=strong, modes={r["mode"]: r for r in (a, b)})
                    result["fixed_pairs"].append(pair)
                    record("fixed_pair", pair, dict(index=i, strong=strong, identical=True,
                           measured=summary([pair["modes"]["audit"]["audit"]])))
            verify_hashes(hashes)
            result["game_pairs"] = []
            # Each cohort supplies sustained work to all workers, rather than one game per worker.
            for offset in range(0, args.games, 16):
                indices = range(offset, min(offset+16, args.games))
                outputs = {}
                for mode in (("plain", "audit") if offset // 16 % 2 == 0 else ("audit", "plain")):
                    tasks = [(i, 20264000+i, config.strong_self_play_interval > 0 and
                              (i+1) % config.strong_self_play_interval == 0, mode) for i in indices]
                    group_start = time.perf_counter()
                    pending = [pool.submit(game_task, task) for task in tasks]
                    rows = []
                    for future in futures.as_completed(pending):
                        row = future.result()
                        rows.append(row)
                        record("game", row, dict(mode=mode, **row["exact"], measured=summary(row["searches"])))
                    outputs[mode] = sorted(rows, key=lambda r: r["exact"]["index"])
                    record("game_group", dict(offset=offset, mode=mode, games=len(rows), seconds=time.perf_counter()-group_start))
                for plain, audit in zip(outputs["plain"], outputs["audit"]):
                    if plain["exact"] != audit["exact"]:
                        raise AssertionError("instrumented game changed samples/RNG/outcome")
                    result["game_pairs"].append(dict(plain=plain, audit=audit))
                record("game_pairs_exact", dict(offset=offset, games=len(outputs["plain"]), identical=True))
        verify_hashes(hashes)
        measured = [row for pair in result["game_pairs"] for row in pair["audit"]["searches"]]
        plain = [row for pair in result["game_pairs"] for row in pair["plain"]["searches"]]
        result.update(elapsed_seconds=time.perf_counter()-start, production_unchanged=True,
                      game_summary=summary(measured), baseline_game_summary=summary(plain),
                      game_by_stage={stage: summary([r for r in measured if r["stage"] == stage])
                                     for stage in ("opening", "middlegame", "endgame")},
                      game_by_budget={str(sims): summary([r for r in measured if r["simulations"] == sims])
                                      for sims in sorted({r["simulations"] for r in measured})},
                      fixed_summary=summary([p["modes"]["audit"]["audit"] for p in result["fixed_pairs"]]))
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        record("complete", dict(output=str(args.output), elapsed_seconds=result["elapsed_seconds"],
                                production_unchanged=True, summary=result["game_summary"]))


if __name__ == "__main__":
    main()
