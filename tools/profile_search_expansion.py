"""Confirm expansion/selection costs without per-node cProfile callbacks."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import azero as az
import jieqi as jq
import run_records as rr
from tools import profile_selfplay as probe


class ExpansionBoundaries(probe.Boundaries):
    def install(self, stack):
        for owner, attribute, name in [
            (az.PUCTNode, "expand", "expand"),
            (az.PUCTNode, "select_child", "select_child"),
            (az.NeuralMCTS, "_evaluate_batch", "evaluate_batch"),
            (az.NeuralMCTS, "_evaluate", "evaluate_root"),
        ]:
            stack.enter_context(mock.patch.object(owner, attribute, self.wrapper(name, getattr(owner, attribute))))


def init_worker(weights, config, native, search_backend, legal_backend):
    probe.init_worker(weights, config, native, search_backend, legal_backend)
    probe.Boundaries = ExpansionBoundaries


def normalise_exact(row):
    return dict(row, move=tuple(row["move"]) if row["move"] is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    if args.pairs < 1 or args.output.exists() or not args.output.parent.is_dir():
        parser.error("positive pairs and new output in an existing directory required")
    if jq.RULES_BACKEND != "cython" or not torch.cuda.is_available():
        parser.error("current Cython backend and CUDA required")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    hashes = dict(reference["source_hashes"])
    hashes[str(args.reference)] = rr.file_record(args.reference)["sha256"]
    hashes[str(Path(__file__))] = rr.file_record(__file__)["sha256"]
    assert all(rr.file_record(path)["sha256"] == expected for path, expected in hashes.items()), "reference sources changed"
    assert jq.RULES_BACKEND_NATIVE == reference["native"], "native identity changed"
    assert az.SEARCH_BACKEND == reference.get("search_backend", "python"), "node backend changed"
    assert az.SEARCH_BACKEND_NATIVE == reference.get("native_search"), "node binary identity changed"
    assert jq.LEGAL_BACKEND == reference.get("legal_backend", "python"), "legal backend changed"
    assert jq.LEGAL_BACKEND_NATIVE == reference.get("native_legal"), "legal binary identity changed"
    torch.set_num_threads(1)
    assert torch.backends.cuda.matmul.allow_tf32 == reference["tf32_matmul"]
    assert torch.backends.cudnn.allow_tf32 == reference["tf32_cudnn"]
    run_dir = Path(reference["args"]["run_dir"])
    checkpoint = torch.load(run_dir / "models/training_checkpoint.pt", map_location="cpu", weights_only=True)
    config = az.AZConfig(**reference["config"])
    net = az.PolicyValueNet().to(probe.DEVICE).eval()
    net.load_state_dict(checkpoint["model"])
    fixtures, trajectories = probe.collect_fixtures(net, config, probe.DEVICE,
        int(reference["args"]["games"]), int(reference["args"]["roots_per_game"]),
        int(reference["args"]["trajectory_sims"]))
    assert [f["metadata"] for f in fixtures] == reference["fixtures"], "fixture metadata changed"
    del net
    gc.collect()
    torch.cuda.empty_cache()
    expected = {(r["index"], r["strong"]): normalise_exact(r["exact"])
                for r in reference["serial"] if r["mode"] == "plain"}
    trials = []
    workers = int(reference["args"]["workers"])
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(checkpoint["model"], asdict(config), jq.RULES_BACKEND_NATIVE,
                                             (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE),
                                             (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))) as pool:
        for strong in (False, True):
            warm = list(pool.map(probe.worker_task, [(f, strong, 20261025+i, "plain") for i, f in enumerate(fixtures)]))
            assert all(r["exact"] == expected[(r["index"], strong)] for r in warm), "warm search differs from reference"
        for pair in range(args.pairs):
            for strong in (False, True):
                for mode in (("plain", "boundaries") if pair % 2 == 0 else ("boundaries", "plain")):
                    tasks = [(f, strong, 20261025+i, mode) for i, f in enumerate(fixtures)]
                    start = time.perf_counter()
                    rows = list(pool.map(probe.worker_task, tasks))
                    trial = dict(pair=pair, strong=strong, mode=mode, wall_seconds=time.perf_counter()-start,
                                 worker_seconds=sum(r["wall_seconds"] for r in rows), searches=rows)
                    assert all(r["exact"] == expected[(r["index"], strong)] for r in rows), "search differs from reference"
                    trials.append(trial)
                    probe.emit("expansion", {k: v for k, v in trial.items() if k != "searches"})
    assert all(rr.file_record(path)["sha256"] == expected for path, expected in hashes.items()), "source/production changed"
    result = dict(source_hashes=hashes, production_unchanged=True, reference=str(args.reference),
                  native=jq.RULES_BACKEND_NATIVE, native_search=az.SEARCH_BACKEND_NATIVE,
                  search_backend=az.SEARCH_BACKEND, config=asdict(config), workers=workers, pairs=args.pairs,
                  legal_backend=jq.LEGAL_BACKEND, native_legal=jq.LEGAL_BACKEND_NATIVE,
                  fixtures=reference["fixtures"], trajectories=trajectories, trials=trials,
                  exact_searches=len(fixtures)*2*2*args.pairs,
                  limitations=["Expansion includes construction and chance enumeration; it is not pure allocation time",
                               "Whole-function wall timers include scheduling/preemption; not pure CPU resource percentages",
                               "No per-constructor timer, no cProfile, no inference/precision/budget change",
                               "Separate budget groups and small diagnostic corpus, not full production throughput"])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    probe.emit("complete", dict(output=str(args.output), exact_searches=result["exact_searches"], production_unchanged=True))


if __name__ == "__main__":
    main()
