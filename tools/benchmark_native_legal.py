"""Read-only legal-fast-path comparison with real CUDA searches and full games."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from dataclasses import asdict
import gc
import json
import multiprocessing as mp
from pathlib import Path
import random
import statistics
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
try:
    import _jieqi_legal as native
except ImportError:
    native = None


def set_mode(mode):
    if mode not in ("python", "cython"):
        raise ValueError("unknown legal benchmark mode")
    jq.GameState._native_legal_module = jq
    jq.GameState._iter_legal_moves_fast = (jq.GameState._iter_legal_moves_fast_python
                                         if mode == "python" else native.iter_legal_moves_fast)


def micro_probe(fixtures, pairs):
    states = [f["state"] for f in fixtures]
    rows = []
    for state in states:
        outputs = []
        for mode in ("python", "cython"):
            set_mode(mode)
            before = probe.digest(state)
            outputs.append((state.legal_moves(), state.has_legal_moves(), state._move_scan(), state.game_outcome()))
            assert probe.digest(state) == before
        assert outputs[0] == outputs[1], "legal query differs"
    for pair in range(pairs):
        for mode in (("python", "cython") if pair % 2 == 0 else ("cython", "python")):
            set_mode(mode)
            for kind in ("legal_moves", "has_legal_moves", "game_outcome"):
                gc.collect()
                start = time.perf_counter()
                for _ in range(128):
                    for state in states:
                        getattr(state, kind)()
                rows.append(dict(pair=pair, mode=mode, kind=kind, seconds=time.perf_counter()-start,
                                 operations=128*len(states)))
    probe.emit("micro", {kind: {mode: statistics.mean(r["seconds"] for r in rows if r["kind"]==kind and r["mode"]==mode)
                               for mode in ("python", "cython")}
                         for kind in ("legal_moves", "has_legal_moves", "game_outcome")})
    return rows


def full_graph_checks(net, config, fixtures):
    rows = []
    indices = sorted(set([0, 1, 2, len(fixtures)//3, len(fixtures)//2-1,
                          len(fixtures)//2+5, len(fixtures)-6, len(fixtures)-2]))
    original = az.NeuralMCTS.search
    for index in indices:
        for strong in (False, True):
            outputs = []
            for mode in ("python", "cython"):
                set_mode(mode)
                saved = []
                def capture(search, *args, **kwargs):
                    kwargs["return_root"] = True
                    policy, move, root = original(search, *args, **kwargs)
                    saved.append(root)
                    return policy, move
                with mock.patch.object(az.NeuralMCTS, "search", capture):
                    result = probe.run_case(net, config, probe.DEVICE, fixtures[index], strong, 20261025+index, "plain")
                graph = probe.digest(saved[0])
                rows.append(dict(index=index, strong=strong, mode=mode, exact=result["exact"], graph_sha256=graph))
                outputs.append((result["exact"], graph))
                del saved, result
                gc.collect()
            assert outputs[0] == outputs[1], f"full tree differs at {index}/{strong}"
            probe.emit("graph_exact", dict(index=index, strong=strong, identical=True))
    return rows


def init_worker(weights, config, rule_record, node_record, legal_record):
    probe.init_worker(weights, config, rule_record, node_record, legal_record)


def search_task(task):
    mode, fixture, strong, seed = task
    set_mode(mode)
    return probe.run_case(probe.NET, probe.CONFIG, probe.DEVICE, fixture, strong, seed, "plain")


def parallel_probe(weights, config, fixtures, workers, pairs, blocks):
    trials = []
    with futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(weights, asdict(config), jq.RULES_BACKEND_NATIVE,
                                             (az.SEARCH_BACKEND, az.SEARCH_BACKEND_NATIVE),
                                             (jq.LEGAL_BACKEND, jq.LEGAL_BACKEND_NATIVE))) as pool:
        for mode in ("python", "cython"):
            list(pool.map(search_task, [(mode, f, False, 20261025+i) for i, f in enumerate(fixtures)]))
        for pair in range(pairs):
            outputs = {}
            for mode in (("python", "cython") if pair % 2 == 0 else ("cython", "python")):
                tasks = [(mode, f, config.strong_self_play_interval > 0 and (i+block)%config.strong_self_play_interval==0,
                          20261025+block*len(fixtures)+i)
                         for block in range(blocks) for i, f in enumerate(fixtures)]
                start = time.perf_counter()
                searches = list(pool.map(search_task, tasks))
                trial = dict(pair=pair, mode=mode, wall_seconds=time.perf_counter()-start,
                             worker_seconds=sum(r["wall_seconds"] for r in searches), searches=searches)
                trials.append(trial)
                outputs[mode] = searches
                probe.emit("search_timing", {k:v for k,v in trial.items() if k != "searches"})
            assert [r["exact"] for r in outputs["python"]] == [r["exact"] for r in outputs["cython"]], "CUDA search differs"
    return trials


def complete_games(weights, config, count):
    net = az.PolicyValueNet().to(probe.DEVICE).eval()
    net.load_state_dict(weights)
    rows = []
    for index in range(count):
        outputs = []
        for mode in (("python", "cython") if index % 2 == 0 else ("cython", "python")):
            set_mode(mode)
            rng = random.Random(20261010+index)
            probe.synchronize(probe.DEVICE)
            start = time.perf_counter()
            game = az.self_play_game(net, config, probe.DEVICE, rng, strong=index%2==1)
            probe.synchronize(probe.DEVICE)
            assert game.outcome is not None
            row = dict(index=index, mode=mode, strong=index%2==1, seconds=time.perf_counter()-start,
                       outcome=game.outcome, moves=game.move_count, samples=len(game.samples),
                       total=game.leaf_evals_total, unique=game.leaf_evals_unique,
                       sample_sha256=probe.digest([asdict(s) for s in game.samples]), rng_sha256=probe.digest(rng.getstate()))
            rows.append(row)
            outputs.append({k:v for k,v in row.items() if k not in ("seconds", "mode")})
            probe.emit("complete_game", row)
        assert outputs[0] == outputs[1], "complete game/sample differs"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--complete-games", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir() or min(args.pairs,args.blocks,args.workers)<1 or args.complete_games<0:
        parser.error("positive sizes and a new output in an existing directory required")
    if not torch.cuda.is_available() or jq.LEGAL_BACKEND_NATIVE is None or jq.RULES_BACKEND_NATIVE is None:
        parser.error("current legal/rules binaries and CUDA required; set JIEQI_LEGAL_BACKEND=cython")
    if az.SEARCH_BACKEND != "python":
        parser.error("keep the node backend Python for this isolated legal experiment")
    torch.set_num_threads(1)
    paths = [args.run_dir / "models/training_checkpoint.pt", args.run_dir / "models/latest.pt",
             args.run_dir / "models/train_state.json", args.run_dir / "logs/train.log",
             ROOT / "azero.py", ROOT / "jieqi.py", ROOT / "ismcts.py", ROOT / "run_records.py",
             ROOT / "native/jieqi_rules.pyx", ROOT / "native/jieqi_legal.pyx",
             Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"]),
             Path(__file__), Path(probe.__file__)]
    hashes = {str(p):rr.file_record(p)["sha256"] for p in paths}
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    replay = args.run_dir / "selfplay" / checkpoint["replay_files"][-1]["name"]
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    config = az.AZConfig(**checkpoint["config"])
    if config.pcr_enabled:
        parser.error("fixed-budget probe requires PCR off")
    net = az.PolicyValueNet().to(probe.DEVICE).eval()
    net.load_state_dict(checkpoint["model"])
    set_mode("python")
    fixtures, trajectories = probe.collect_fixtures(net, config, probe.DEVICE, 3, 6, 0)
    micro = micro_probe(fixtures, args.pairs)
    graphs = full_graph_checks(net, config, fixtures)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    searches = parallel_probe(checkpoint["model"],config,fixtures,args.workers,args.pairs,args.blocks)
    games = complete_games(checkpoint["model"],config,args.complete_games)
    assert all(rr.file_record(p)["sha256"]==h for p,h in hashes.items()), "production/source changed"
    savings = []
    for pair in range(args.pairs):
        timing = {r["mode"]:r["wall_seconds"] for r in searches if r["pair"]==pair}
        savings.append(1-timing["cython"]/timing["python"])
    result = dict(source_hashes=hashes, production_unchanged=True, checkpoint_iteration=checkpoint["train_state"]["iteration"],
                  args={k:str(v) for k,v in vars(args).items()}, config=asdict(config),
                  native_rules=jq.RULES_BACKEND_NATIVE, native_legal=jq.LEGAL_BACKEND_NATIVE,
                  search_backend=az.SEARCH_BACKEND, native_search=az.SEARCH_BACKEND_NATIVE,
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  fixtures=[f["metadata"] for f in fixtures], trajectories=trajectories, micro_trials=micro,
                  full_graph_checks=graphs, search_trials=searches, complete_games=games,
                  paired_savings=savings, mean_paired_saving=statistics.mean(savings),
                  exact_paired_searches=args.pairs*args.blocks*len(fixtures),
                  limitations=["Only no-ban legal path differs; attack/node/encoding/sampling/network settings unchanged",
                               "Full graph checks retain trees; their timing is not used for throughput",
                               "Repeated fixed roots and two serial games, not full 1024-game throughput or learning-speed proof"])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    probe.emit("complete", dict(output=str(args.output), paired_savings=savings,
                               exact_paired_searches=result["exact_paired_searches"], production_unchanged=True))


if __name__ == "__main__":
    main()
