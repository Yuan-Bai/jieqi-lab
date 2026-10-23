"""Observe cross-batch cache opportunities and output equality; never reuse outputs."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import concurrent.futures as futures
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import struct
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
from tools.measure_search_reuse import verify_hashes


class CacheObserver:
    def __init__(self, capacities=(256, 1024, 4096, 16384), output_capacity=4096):
        self.lrus = {n: OrderedDict() for n in capacities}
        self.output_capacity = output_capacity
        self.outputs = {name: OrderedDict() for name in ("input_only", "batch_size")}
        self.game_seen, self.batch_seen = {}, set()
        self.search_id, self.call = 0, 0
        self.examples = []
        self.start_search()

    def start_search(self):
        self.search_id += 1
        self.search_seen = {}
        self.stats = {}
        self.histograms = {n: {} for n in self.lrus}

    def add(self, key, value=1):
        self.stats[key] = self.stats.get(key, 0) + value

    def prepare(self, batch, infos, histories, entries, role):
        if batch.dtype != np.float32 or batch.shape != (len(infos), az.STATE_CHANNELS, 10, 9):
            raise AssertionError("unexpected actual input format")
        if role not in ("root", "leaf"):
            raise ValueError("unknown evaluation role")
        self.call += 1
        n = len(batch)
        self.add("rows", n)
        self.add(role+"_rows", n)
        self.add(role+"_calls")
        start = time.perf_counter()
        packed = [zlib.compress(row.tobytes(order="C"), 1) for row in batch]
        self.add("exact_key_seconds", time.perf_counter()-start)
        start = time.perf_counter()
        entries = entries if entries is not None else [az._repetition_entries(info) for info in infos]
        keys = [az.leaf_input_signature(info, history[-az.HISTORY_PLIES:], repetition_entries=entry)
                for info, history, entry in zip(infos, histories, entries)]
        self.add("candidate_key_seconds", time.perf_counter()-start)

        for encoded in packed:
            old = self.search_seen.get(encoded)
            kind = ("within_batch_repeats" if old == self.call else "search_prior_batch_hits" if old is not None
                    else "game_additional_hits" if encoded in self.game_seen else "first_game_inputs")
            self.add(kind)
            self.add(role+"_"+kind)
            self.search_seen[encoded] = self.call
            self.game_seen[encoded] = True
        batch_key = tuple(packed)
        identical_batch = batch_key in self.batch_seen
        if identical_batch:
            self.add("identical_full_batch_repeats")
            self.add(role+"_identical_full_batch_repeats")
        self.batch_seen.add(batch_key)

        hits_by_capacity = {}
        for capacity, table in self.lrus.items():
            start = time.perf_counter()
            previous = [table.get(key) for key in keys]
            hits = evictions = 0
            for key, encoded, old in zip(keys, packed, previous):
                if old is not None:
                    if old != encoded:
                        raise AssertionError("candidate key merged different actual inputs")
                    hits += 1
                current = table.get(key)
                if current is not None:
                    if current != encoded:
                        raise AssertionError("candidate key collision within batch")
                    table.move_to_end(key)
                else:
                    table[key] = encoded
                while len(table) > capacity:
                    table.popitem(last=False)
                    evictions += 1
            self.add(f"lru_{capacity}_seconds", time.perf_counter()-start)
            self.add(f"lru_{capacity}_{role}_hits", hits)
            self.add(f"lru_{capacity}_evictions", evictions)
            if hits:
                self.add(f"lru_{capacity}_{role}_hit_calls")
            if hits == n:
                self.add(f"lru_{capacity}_{role}_all_hit_calls")
            bucket = f"{role}:{n}:{hits}"
            self.histograms[capacity][bucket] = self.histograms[capacity].get(bucket, 0) + 1
            hits_by_capacity[capacity] = hits
        return dict(role=role, n=n, packed=packed, hits=hits_by_capacity,
                    identical_batch=identical_batch,
                    call=self.call)

    def finish(self, context, policies, values, evaluation_seconds):
        role, n = context["role"], context["n"]
        policies = np.asarray(policies)
        if policies.dtype != np.float32 or policies.shape != (n, az.MOVE_SIZE) or len(values) != n:
            raise AssertionError("unexpected actual output format")
        self.add(role+"_evaluation_seconds", evaluation_seconds)
        if context["identical_batch"]:
            self.add(role+"_identical_full_batch_evaluation_seconds", evaluation_seconds)
        for capacity, hits in context["hits"].items():
            if hits == n:
                self.add(f"lru_{capacity}_{role}_all_hit_evaluation_seconds", evaluation_seconds)
        for name, table in self.outputs.items():
            keys = [(n, encoded) if name == "batch_size" else encoded for encoded in context["packed"]]
            previous = [table.get(key) for key in keys]
            for i, (key, old) in enumerate(zip(keys, previous)):
                policy, value = policies[i], float(values[i])
                if old is not None:
                    old_policy, old_value, old_role, old_n, old_i, old_search = old
                    pair = old_role+"_to_"+role
                    self.add(name+"_output_comparisons")
                    self.add(name+"_"+pair+"_comparisons")
                    p_equal = old_policy.tobytes() == policy.tobytes()
                    v_equal = struct.pack("d", old_value) == struct.pack("d", value)
                    if not p_equal or not v_equal:
                        self.add(name+"_output_mismatches")
                        self.add(name+"_"+pair+"_mismatches")
                        self.add(name+"_policy_mismatches", int(not p_equal))
                        self.add(name+"_value_mismatches", int(not v_equal))
                        self.add(name+"_same_size_mismatches", int(old_n == n))
                        self.add(name+"_same_size_position_mismatches", int(old_n == n and old_i == i))
                        dp = float(np.max(np.abs(old_policy-policy)))
                        dv = abs(old_value-value)
                        for suffix, difference in (("max_policy_abs_diff", dp), ("max_value_abs_diff", dv)):
                            metric = name+"_"+suffix
                            self.stats[metric] = max(self.stats.get(metric, 0), difference)
                        if len(self.examples) < 12:
                            self.examples.append(dict(cache=name, prior_role=old_role, role=role,
                                prior_batch_size=old_n, batch_size=n, prior_position=old_i, position=i,
                                prior_search=old_search, search=self.search_id, call=context["call"],
                                input_sha256=hashlib.sha256(context["packed"][i]).hexdigest(),
                                policy_equal=p_equal, value_equal=v_equal, max_policy_abs_diff=dp,
                                value_abs_diff=dv, prior_value=old_value, value=value))
                if key in table:
                    table.move_to_end(key)
                else:
                    table[key] = (policy.copy(), value, role, n, i, self.search_id)
                while len(table) > self.output_capacity:
                    table.popitem(last=False)


class EvaluationAudit:
    def __init__(self):
        self.observer = CacheObserver()
        self.rows = []
        self.context = None

    @contextmanager
    def install(self):
        original_search = az.NeuralMCTS.search
        original_root, original_leaf = az.NeuralMCTS._evaluate, az.NeuralMCTS._evaluate_batch
        encode_root, encode_leaf = az.encode_state_numpy, az._encode_batch_numpy

        def observe(batch, infos, histories, entries, role):
            if self.context is not None:
                start = time.perf_counter()
                self.context["prepared"] = self.observer.prepare(batch, infos, histories, entries, role)
                self.context["overhead"] += time.perf_counter()-start

        def root_encode(info, history):
            result = encode_root(info, history)
            observe(result[None], [info], [history], None, "root")
            return result

        def leaf_encode(infos, histories, *, repetition_entries=None):
            result = encode_leaf(infos, histories, repetition_entries=repetition_entries)
            observe(result, infos, histories, repetition_entries, "leaf")
            return result

        def evaluate(function, mcts, *args, **kwargs):
            if self.context is not None:
                raise AssertionError("nested evaluation")
            self.context = dict(overhead=0.0)
            start = time.perf_counter()
            try:
                result = function(mcts, *args, **kwargs)
                seconds = time.perf_counter()-start-self.context["overhead"]
                context = self.context["prepared"]
                policies, values = result
                if context["role"] == "root":
                    policies, values = policies.numpy()[None], [values]
                self.observer.finish(context, policies, values, seconds)
                return result
            finally:
                self.context = None

        def root(mcts, *args, **kwargs):
            return evaluate(original_root, mcts, *args, **kwargs)

        def leaf(mcts, *args, **kwargs):
            return evaluate(original_leaf, mcts, *args, **kwargs)

        def search(mcts, state, belief, history, *args, **kwargs):
            self.observer.start_search()
            result = original_search(mcts, state, belief, history, *args, **kwargs)
            self.rows.append(dict(ply=len(history), simulations=kwargs.get("simulations") or mcts.config.mcts_simulations,
                                  stats=self.observer.stats.copy(), histograms=self.observer.histograms))
            return result

        with ExitStack() as stack:
            for owner, name, replacement in ((az.NeuralMCTS, "search", search), (az.NeuralMCTS, "_evaluate", root),
                    (az.NeuralMCTS, "_evaluate_batch", leaf), (az, "encode_state_numpy", root_encode),
                    (az, "_encode_batch_numpy", leaf_encode)):
                stack.enter_context(mock.patch.object(owner, name, replacement))
            yield self


def summary(rows):
    stats, histograms = {}, {}
    for row in rows:
        for key, value in row["stats"].items():
            stats[key] = max(stats.get(key, 0), value) if "_max_" in key else stats.get(key, 0) + value
        for capacity, buckets in row["histograms"].items():
            target = histograms.setdefault(str(capacity), {})
            for key, value in buckets.items():
                target[key] = target.get(key, 0) + value
    return dict(stats=stats, histograms=histograms, searches=len(rows))


def init_worker(weights, config, rules, search, legal):
    probe.init_worker(weights, config, rules, search_backend=search, legal_backend=legal)


def game_task(task):
    index, mode = task
    seed = 20264000+index
    strong = probe.CONFIG.strong_self_play_interval > 0 and (index+1) % probe.CONFIG.strong_self_play_interval == 0
    rng, audit = random.Random(seed), EvaluationAudit() if mode == "audit" else None
    probe.synchronize(probe.DEVICE)
    start = time.perf_counter()
    with audit.install() if audit is not None else ExitStack():
        game = az.self_play_game(probe.NET, probe.CONFIG, probe.DEVICE, rng, strong=strong)
    probe.synchronize(probe.DEVICE)
    seconds = time.perf_counter()-start
    if game.outcome is None:
        raise AssertionError("unfinished game")
    exact = dict(index=index, seed=seed, strong=strong, moves=game.move_count, outcome=game.outcome,
                 samples=len(game.samples), leaf_total=game.leaf_evals_total, leaf_unique=game.leaf_evals_unique,
                 sample_sha256=probe.digest([asdict(s) for s in game.samples]), rng_sha256=probe.digest(rng.getstate()))
    return dict(mode=mode, seconds=seconds, exact=exact, searches=audit.rows if audit else [],
                examples=audit.observer.examples if audit else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lazy-chance-allocation", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or not args.output.parent.is_dir():
        parser.error("new output/journal in existing directory required")
    if args.games < 1 or args.workers < 1 or args.games % 16:
        parser.error("positive workers and games in whole 16-game cohorts required")
    if jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython" or az.SEARCH_BACKEND != "python" or not torch.cuda.is_available():
        parser.error("require CUDA, rules/legal Cython, Python nodes")
    torch.set_num_threads(1)
    paths = [args.run_dir/"models/training_checkpoint.pt", args.run_dir/"models/latest.pt",
             args.run_dir/"models/train_state.json", args.run_dir/"logs/train.log", ROOT/"azero.py",
             ROOT/"jieqi.py", ROOT/"ismcts.py", ROOT/"run_records.py", Path(__file__), Path(probe.__file__),
             ROOT/"tools/measure_search_reuse.py", ROOT/"native/jieqi_rules.pyx", ROOT/"native/jieqi_legal.pyx",
             Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"])]
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    saved = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir/"selfplay"/saved["replay_files"][-1]["name"]
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    config = az.AZConfig(**saved["config"])
    config.lazy_chance_allocation = args.lazy_chance_allocation
    if config.pcr_enabled or not config.chance_nodes or config.chance_k != 0 or not config.leaf_dedup:
        parser.error("require fixed budgets, chance k=0 and leaf dedup")
    if config.strong_self_play_interval > 0 and args.games % config.strong_self_play_interval:
        parser.error("whole strong-game intervals required")
    result = dict(args={k: str(v) for k, v in vars(args).items()}, source_hashes=hashes, config=asdict(config),
                  checkpoint_iteration=saved["train_state"]["iteration"], gpu=torch.cuda.get_device_name(),
                  torch=str(torch.__version__), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32, cache_scope="one game; new observer per game",
                  runtime=dict(rules=jq.RULES_BACKEND_NATIVE, search=az.SEARCH_BACKEND,
                               legal=jq.LEGAL_BACKEND_NATIVE), game_pairs=[])
    start = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle:
        def record(event, value, display=None):
            handle.write(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter()-start, value=value))+"\n")
            handle.flush()
            probe.emit(event, display if display is not None else value)

        record("metadata", result)
        with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=init_worker, initargs=(saved["model"], asdict(config), jq.RULES_BACKEND_NATIVE,
                    (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE), (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))) as pool:
            for offset in range(0, args.games, 16):
                groups = {}
                for mode in (("plain", "audit") if offset//16 % 2 == 0 else ("audit", "plain")):
                    group_start = time.perf_counter()
                    pending = [pool.submit(game_task, (i, mode)) for i in range(offset, offset+16)]
                    rows = []
                    for future in futures.as_completed(pending):
                        row = future.result()
                        rows.append(row)
                        record("game", row, dict(mode=mode, **row["exact"], measured=summary(row["searches"])["stats"]))
                    groups[mode] = sorted(rows, key=lambda r: r["exact"]["index"])
                    record("cohort", dict(mode=mode, offset=offset, seconds=time.perf_counter()-group_start))
                for plain, audit in zip(groups["plain"], groups["audit"]):
                    if plain["exact"] != audit["exact"]:
                        raise AssertionError("measurement changed game samples/RNG/leaf counts")
                    result["game_pairs"].append(dict(plain=plain, audit=audit))
                record("paired_exact", dict(offset=offset, games=16, identical=True))
        verify_hashes(hashes)
        rows = [row for pair in result["game_pairs"] for row in pair["audit"]["searches"]]
        result.update(production_unchanged=True, elapsed_seconds=time.perf_counter()-start, summary=summary(rows),
                      by_budget={str(n): summary([r for r in rows if r["simulations"] == n])
                                 for n in sorted({r["simulations"] for r in rows})},
                      limitations=["No outputs reused, no batch compaction, no statistics merged, no training/replay writes",
                                   "Exact compressed input keys are audit-only; candidate costs timed separately",
                                   "Output comparisons use a 4096-entry per-game LRU; batch-size keys do not imply compaction is safe",
                                   "Evaluation timings subtract observer preparation but include encoding, H2D, network, softmax and CPU transfer",
                                   "Audit overhead/CPU contention means timings are not actual cache savings",
                                   "No across-game/worker/model-version reuse or production replay-buffer residency measured"])
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        record("complete", dict(output=str(args.output), elapsed_seconds=result["elapsed_seconds"],
                                production_unchanged=True, summary=result["summary"]["stats"]))


if __name__ == "__main__":
    main()
