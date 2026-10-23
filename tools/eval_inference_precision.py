"""Gated frozen-model precision matches, using the existing paired evaluation protocol."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

import azero as az
import jieqi as jq
import run_records as rr
from tools import benchmark_inference_precision as bench
from tools import central_inference as central
from tools import eval as evaluation
from tools import paired_stats
from tools import profile_selfplay as probe
from tools.measure_search_reuse import verify_hashes

WEIGHT_HASH = None


class CheckedOutputs(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        logits, values = self.net(inputs)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(values).all()):
            raise AssertionError("nonfinite output during precision match")
        return logits, values


class PrecisionOpponent(evaluation.NeuralOpponent):
    def __init__(self, net, config, device, rng, mode, settings):
        super().__init__(net, config, device, rng)
        self.mode, self.settings = mode, settings
        self.moves = []

    def select_move(self, state, belief, history):
        # Each player's scope must end before the other player or environment acts.
        with bench.precision_scope(self.mode, self.settings):
            move = super().select_move(state, belief, history)
        self.moves.append(move)
        return move

    def audit(self):
        return dict(mode=self.mode, moves_sha256=probe.digest(self.moves),
                    rng_sha256=probe.digest(self.mcts.rng.getstate()),
                    leaf_total=self.mcts.leaf_evals_total,
                    leaf_unique=self.mcts.leaf_evals_unique)


def paired_tasks(pairs, seed, max_plies):
    return [("amp_fp16", pair, seed+pair, side, max_plies)
            for pair in range(pairs) for side in (jq.RED, jq.BLACK)]


def validate_throughput(evidence):
    minimum = evidence.get("args", {}).get("pilot_min_time_saving")
    if minimum is None or minimum < 0.05 or not evidence.get("production_unchanged"):
        raise ValueError("requires an audited, predeclared >=5% throughput pilot")
    gate = bench.throughput_gate(evidence["game_trials"], minimum)
    if not gate["passed"] or not evidence.get("throughput_gate", {}).get("passed"):
        raise ValueError("throughput screening gate did not pass; do not run strength matches")
    verify_hashes(evidence["source_hashes"])
    if (evidence["baseline_precision"] != central.precision_settings()
            or evidence["backends"] != dict(rules=jq.RULES_BACKEND_NATIVE,
                                            legal=jq.LEGAL_BACKEND_NATIVE, search=az.SEARCH_BACKEND)):
        raise ValueError("precision flags or backend provenance differs from throughput evidence")
    return gate


def init_worker(weights, architecture, config, settings, modes, backends):
    global WEIGHT_HASH
    bench.init_worker(weights, architecture, config, settings, modes, backends)
    bench.NET = CheckedOutputs(bench.NET).eval()
    WEIGHT_HASH = rr.model_hash(bench.NET)


def game_task(task):
    mode, pair, seed, side, max_plies = task
    agents = []

    def factory(precision):
        def create(rng):
            agent = PrecisionOpponent(bench.NET, bench.CONFIG, bench.DEVICE, rng,
                                      precision, bench.SETTINGS)
            agents.append(agent)
            return agent
        return create

    probe.synchronize(bench.DEVICE)
    started = time.perf_counter()
    record = evaluation.play_one_game(factory(mode), factory("baseline"), side, seed, max_plies)
    probe.synchronize(bench.DEVICE)
    record.update(type="game", pair=pair, opponent="baseline", candidate_precision=mode,
                  seconds=time.perf_counter()-started, agents=[agent.audit() for agent in agents])
    if rr.model_hash(bench.NET) != WEIGHT_HASH:
        raise AssertionError("frozen network weights changed during evaluation")
    exact = {key: record[key] for key in ("seed", "neural_side", "outcome", "reason", "plies", "score", "agents")}
    record["exact_sha256"] = probe.digest(exact)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throughput-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20282008)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=600)
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or not args.output.parent.is_dir():
        parser.error("new output and journal in an existing directory required")
    if min(args.pairs, args.sims, args.workers, args.max_plies) < 1:
        parser.error("positive experiment sizes required")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    torch.set_num_threads(1)
    with args.throughput_result.open(encoding="utf-8") as handle:
        evidence = json.load(handle)
    try:
        gate = validate_throughput(evidence)
    except ValueError as error:
        parser.error(str(error))
    hashes = dict(evidence["source_hashes"])
    for path in (Path(__file__), args.throughput_result, Path(evaluation.__file__),
                 Path(paired_stats.__file__), ROOT/"run_records.py"):
        hashes[str(path)] = rr.file_record(path)["sha256"]
    checkpoint = Path(evidence["args"]["run_dir"])/"models/training_checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = az.AZConfig(**evidence["config"])
    config.mcts_simulations = args.sims
    metadata = dict(type="run", experiment="frozen_inference_precision_strength_pilot",
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    checkpoint_iteration=saved["train_state"]["iteration"], source_hashes=hashes,
                    config=asdict(config), baseline_precision=evidence["baseline_precision"],
                    candidate_precision="amp_fp16", opponent_precision="baseline",
                    temperature=1e-6, root_noise=False, prune_policy=False,
                    finite_outputs_checked_every_call=True,
                    eval_protocol_version=az.EVAL_PROTOCOL_VERSION,
                    environment=dict(python=sys.version, torch=str(torch.__version__),
                                     numpy=np.__version__, cuda=torch.version.cuda,
                                     gpu=torch.cuda.get_device_name()),
                    throughput_gate=gate,
                    started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    started = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle:
        def record(row):
            handle.write(json.dumps(row)+"\n")
            handle.flush()
            probe.emit(row["type"], {k: v for k, v in row.items() if k not in ("source_hashes", "config", "agents")})

        record(metadata)
        with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=init_worker,
                initargs=(saved["model"], evidence["architecture"], asdict(config),
                          evidence["baseline_precision"], ("baseline", "amp_fp16"),
                          (jq.RULES_BACKEND_NATIVE, jq.LEGAL_BACKEND_NATIVE, az.SEARCH_BACKEND_NATIVE))) as pool:
            task = ("baseline", -1, args.seed-1, jq.RED, args.max_plies)
            control = list(pool.map(game_task, (task, task)))
            if control[0]["exact_sha256"] != control[1]["exact_sha256"]:
                raise AssertionError("baseline full-game repeatability control failed")
            record(dict(type="control", identical=True, games=control))
            pending = [pool.submit(game_task, task) for task in paired_tasks(args.pairs, args.seed, args.max_plies)]
            games = []
            for future in futures.as_completed(pending):
                row = future.result()
                games.append(row)
                record(row)
        verify_hashes(hashes)
        games.sort(key=lambda row: (row["pair"], row["neural_side"]))
        stats = paired_stats.analyze_games(str(journal), metadata, games, 20000, 20000, 20261008)
        result = dict(run=metadata, control=control, games=games, stats=stats,
                      production_unchanged=True, elapsed_seconds=time.perf_counter()-started,
                      strength_acceptance=False,
                      limitations=["Frozen-model pilot only; no noninferiority or convergence acceptance",
                          "Same weights and 800-by-default simulation budgets on both sides; no root noise",
                          "Existing eval protocol supplies shared shuffle and isolated environment/player RNGs",
                          "AMP applies only to candidate search; scopes and weight caches close between moves",
                          "Incomplete pairs are excluded from pair_mean/bootstrap CI, never scored as draws",
                          "All-game Wilson score may include a finished half of an incomplete pair; pair CI is primary",
                          "Throughput was screened on earlier seeds; these independent match seeds were not selected by outcomes",
                          "Production defaults, updates and replay remain untouched"])
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        record(dict(type="summary", **{k: v for k, v in stats.items() if k != "run"},
                    strength_acceptance=False, production_unchanged=True))


if __name__ == "__main__":
    main()
