"""Read-only, root-local graph-search feasibility audit; never merge search nodes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import azero as az
import ismcts as im
import jieqi as jq
import run_records as rr
import torch
from tools import profile_selfplay as probe
from tools.benchmark_central_inference import EvaluationTrace
from tools.measure_search_reuse import verify_hashes

# Audit encodings must not be mistaken for actual network requests by trace hooks.
REFERENCE_ENCODER = az.encode_state_numpy
LAYERS = ("board", "rules", "input_state", "recall", "conservative")


def rule_key(info, entries):
    if entries is None:
        counts = None if info.position_counts is None else tuple(sorted(
            (side, bytes(board), count) for (side, board), count in info.position_counts.items()))
    else:
        counts = tuple(sorted(entries))
    return (info.rules, info.side, bytes(info.board), info.no_capture_plies,
            tuple(info.consecutive_checks), tuple(info.check_anchor), tuple(info.chase_streak),
            tuple(info.chase_target), tuple(info.chase_target_code), tuple(info.chase_anchor), counts)


def clean_public_state(info, private_path, entries):
    """No hidden pieces or residual private-pool features for future actor views."""
    return (entries is not None and info.repetition_history_complete and not private_path
            and not any(piece and jq.is_dark(piece) for piece in info.board)
            and not any(info.pool_total_known) and not any(info.unknown_removals)
            and not any(any(row) for row in info.pool_known)
            and info.public_pool_known is not None
            and not any(any(row) for row in info.public_pool_known))


def decision_keys(info, history, public_path, private_path):
    entries = az._repetition_entries(info)
    rules = rule_key(info, entries)
    pools = None if info.public_pool_known is None else tuple(tuple(row) for row in info.public_pool_known)
    state = (len(public_path), rules, pools, info.repetition_history_complete,
             az.leaf_input_signature(info, history[-az.HISTORY_PLIES:], repetition_entries=entries))
    recall = (state, tuple(public_path), tuple(private_path))
    clean = clean_public_state(info, private_path, entries)
    return dict(board=(info.side, bytes(info.board)), rules=rules, input_state=state,
                recall=recall, conservative=state if clean else recall), clean


def observable_continuation(info):
    # Dedicated RNG and actor-only data: no access to the real opponent belief slot.
    state = info.determinize(random.Random(717))
    return tuple(state.legal_moves()), state.game_outcome()


def layer_summary(records, name):
    groups = defaultdict(list)
    for record in records:
        groups[record["keys"][name]].append(record)
    collisions = [rows for rows in groups.values() if len({r["public"] for r in rows}) > 1]
    extra = sum(len(rows)-1 for rows in collisions)
    cross_root_action = sum(len(rows)-1 for rows in collisions
                            if len({r["public"][0][0] if r["public"] else None for r in rows}) > 1)
    return dict(groups=len(groups), transposition_groups=len(collisions), extra_decisions=extra,
                fraction=extra/len(records) if records else None, cross_root_action_extra=cross_root_action)


class TranspositionAudit:
    def __init__(self, root_history):
        self.root_history = list(root_history)
        self.records = {}
        self.stats = Counter()
        self.first_key = None

    def observe(self, info, public_path, private_path, original_key):
        self.stats["decision_encounters"] += 1
        start = time.perf_counter()
        prior = self.records.get(original_key)
        self.stats["existing_lookup_seconds"] += time.perf_counter()-start
        if prior is not None:
            self.stats["same_existing_decision_revisits"] += 1
            return
        start = time.perf_counter()
        history = self.root_history+[item[0] for item in public_path]
        keys, clean = decision_keys(info, history, public_path, private_path)
        self.stats["key_build_seconds"] += time.perf_counter()-start
        self.records[original_key] = dict(info=info, history=history[-az.HISTORY_PLIES:], keys=keys,
                                         public=tuple(public_path), clean=clean)
        if self.first_key is None:
            self.first_key = original_key

    @contextmanager
    def install(self):
        original = im.information_set_key
        original_direct = im.BeliefModel.information_set_key

        def key(info, public_path, private_path):
            result = original(info, public_path, private_path)
            self.observe(info, public_path, private_path, result)
            return result

        def direct_key(view, state, observer, public_path, private_path):
            result = original_direct(view, state, observer, public_path, private_path)
            # Audit-only snapshots retain full rule/input evidence, never search statistics.
            self.observe(view.info_state(state, observer), public_path, private_path, result)
            return result

        with mock.patch.object(im, "information_set_key", key), mock.patch.object(
                im.BeliefModel, "information_set_key", direct_key):
            yield self

    def finish(self, root):
        start = time.perf_counter()
        nodes, incoming, seen, pending = {self.first_key: root}, Counter(), set(), [root]
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            for key, child in node.decision_nodes.items():
                if key in nodes and nodes[key] is not child:
                    raise AssertionError("existing decision key maps to distinct nodes")
                nodes[key] = child
                incoming[id(child)] += 1
            pending.extend(node.children.values())
            pending.extend(node.public_outcomes.values())
            pending.extend(node.decision_nodes.values())
        if nodes.keys() != self.records.keys():
            raise AssertionError("audit missed decision objects")
        # Terminal/unexpanded objects are not evidence of saved neural expansion work.
        expanded = [record for key, record in self.records.items() if nodes[key].children]
        layers = {name: layer_summary(expanded, name) for name in LAYERS}
        groups = defaultdict(list)
        for record in expanded:
            groups[record["keys"]["input_state"]].append(record)
        checks, examples = 0, []
        for rows in groups.values():
            if len({record["public"] for record in rows}) < 2:
                continue
            first = rows[0]
            expected = REFERENCE_ENCODER(first["info"], first["history"]).tobytes()
            legal = observable_continuation(first["info"])
            for record in rows[1:]:
                actual = REFERENCE_ENCODER(record["info"], record["history"]).tobytes()
                if actual != expected or observable_continuation(record["info"]) != legal:
                    raise AssertionError("candidate failed actual-input/legal/terminal equality")
                checks += 1
            if len(examples) < 3:
                examples.append(dict(depth=len(first["public"]), clean_public=first["clean"],
                                     input_sha256=hashlib.sha256(expected).hexdigest(),
                                     paths=[r["public"] for r in rows[:2]]))
        self.stats["graph_scan_and_validation_seconds"] = time.perf_counter()-start
        return dict(**self.stats, distinct_decisions=len(self.records), expanded_decisions=len(expanded),
                    clean_public_expanded=sum(r["clean"] for r in expanded),
                    expanded_depth_counts=dict(sorted(Counter(len(r["public"]) for r in expanded).items())),
                    already_shared_decisions=sum(count > 1 for count in incoming.values()),
                    existing_shared_extra_links=sum(max(0, count-1) for count in incoming.values()),
                    validated_input_state_pairs=checks, layers=layers, examples=examples)


def run_search(net, config, device, fixture, simulations, seed, measured):
    state, belief, history = (fixture[name] for name in ("state", "belief", "history"))
    before = probe.digest((fixture, config))
    audit, trace = TranspositionAudit(history), EvaluationTrace(True)
    search = az.NeuralMCTS(net, config, device, random.Random(seed))
    probe.synchronize(device)
    start = time.perf_counter()
    with trace.install():
        if measured:
            with audit.install():
                policy, move, root = search.search(state, belief, history, 1e-6, simulations=simulations,
                                                  add_dirichlet=False, return_root=True)
        else:
            policy, move, root = search.search(state, belief, history, 1e-6, simulations=simulations,
                                              add_dirichlet=False, return_root=True)
    probe.synchronize(device)
    seconds = time.perf_counter()-start
    if before != probe.digest((fixture, config)):
        raise AssertionError("audit changed fixture/config")
    exact = dict(policy_sha256=probe.digest(policy), move=move, graph_sha256=probe.digest(root),
                 rng_sha256=probe.digest(search.rng.getstate()), inputs_sha256=trace.inputs.hexdigest(),
                 outputs_sha256=trace.outputs.hexdigest(), input_rows=trace.metrics["rows"],
                 root_calls=trace.metrics["root_calls"], leaf_calls=trace.metrics["leaf_calls"],
                 leaf_total=search.leaf_evals_total, leaf_unique=search.leaf_evals_unique,
                 chance_applied=search.chance_applied, chance_nav_miss=search.chance_nav_miss,
                 chance_outcomes_added=search.chance_outcomes_added)
    return dict(seconds=seconds, exact=exact, audit=audit.finish(root) if measured else None)


def summarize_rows(rows):
    total = sum(row["audit"]["expanded_decisions"] for row in rows)
    layers = {}
    for name in LAYERS:
        extra = sum(row["audit"]["layers"][name]["extra_decisions"] for row in rows)
        layers[name] = dict(extra_decisions=extra, fraction=extra/total if total else None,
                            roots_with_hits=sum(row["audit"]["layers"][name]["extra_decisions"] > 0 for row in rows))
    depths = Counter()
    for row in rows:
        depths.update(row["audit"]["expanded_depth_counts"])
    return dict(pairs=len(rows), expanded_decisions=total, layers=layers,
        clean_public_expanded=sum(row["audit"]["clean_public_expanded"] for row in rows),
        expanded_depth_counts=dict(sorted(depths.items())),
        existing_shared_extra_links=sum(row["audit"]["existing_shared_extra_links"] for row in rows),
        validated_input_state_pairs=sum(row["audit"]["validated_input_state_pairs"] for row in rows),
        plain_seconds=sum(row["plain_seconds"] for row in rows),
        measured_seconds=sum(row["measured_seconds"] for row in rows),
        key_build_seconds=sum(row["audit"]["key_build_seconds"] for row in rows),
        existing_lookup_seconds=sum(row["audit"]["existing_lookup_seconds"] for row in rows))


def summarize(pairs):
    summaries = {}
    for budget in sorted({pair["simulations"] for pair in pairs}):
        rows = [pair for pair in pairs if pair["simulations"] == budget]
        summaries[str(budget)] = summarize_rows(rows)
        summaries[str(budget)]["by_fixture_kind"] = {
            kind: summarize_rows([row for row in rows if row["metadata"]["kind"] == kind])
            for kind in sorted({row["metadata"]["kind"] for row in rows})}
        summaries[str(budget)]["by_seed_block"] = {
            str(block): summarize_rows([row for row in rows if row["block"] == block])
            for block in sorted({row["block"] for row in rows})}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-games", type=int, default=2)
    parser.add_argument("--roots-per-game", type=int, default=4)
    parser.add_argument("--seed-blocks", type=int, default=2)
    args = parser.parse_args()
    journal = args.output.with_suffix(".jsonl")
    if args.output.exists() or journal.exists() or not args.output.parent.is_dir():
        parser.error("new output/journal in an existing directory required")
    if min(args.trajectory_games, args.roots_per_game, args.seed_blocks) < 1:
        parser.error("positive corpus sizes required")
    if not torch.cuda.is_available():
        parser.error("CUDA required for this fixed-model pilot")
    torch.set_num_threads(1)
    paths = [ROOT/name for name in ("azero.py", "ismcts.py", "jieqi.py", "run_records.py",
                                   "tools/profile_selfplay.py", "tools/benchmark_central_inference.py",
                                   "tools/measure_search_reuse.py")]
    paths += [Path(__file__), args.run_dir/"models/training_checkpoint.pt", args.run_dir/"models/latest.pt",
              args.run_dir/"models/train_state.json", args.run_dir/"logs/train.log"]
    for native in (jq.RULES_BACKEND_NATIVE, jq.LEGAL_BACKEND_NATIVE, az.SEARCH_BACKEND_NATIVE):
        if native:
            paths.append(Path(native["path"]))
    hashes = {str(path): rr.file_record(path)["sha256"] for path in paths}
    saved = torch.load(args.run_dir/"models/training_checkpoint.pt", map_location="cpu", weights_only=True)
    if (saved["ruleset_version"], saved["data_version"], saved["search_version"]) != (
            jq.RULESET_VERSION, az.DATA_VERSION, az.SEARCH_VERSION):
        parser.error("incompatible checkpoint versions")
    config = az.AZConfig(**saved["config"])
    if config.pcr_enabled or not config.chance_nodes or config.chance_k != 0:
        parser.error("fixed-budget on/k=0 checkpoint required")
    replay = args.run_dir/"selfplay"/saved["replay_files"][-1]["name"]
    hashes[str(replay)] = rr.file_record(replay)["sha256"]
    device = torch.device("cuda")
    net = az.PolicyValueNet().to(device).eval()
    net.load_state_dict(saved["model"])
    result = dict(checkpoint_iteration=saved["train_state"]["iteration"], source_hashes=hashes,
                  config=asdict(config), args={k: str(v) for k, v in vars(args).items()},
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  runtime=dict(rules=jq.RULES_BACKEND, legal=jq.LEGAL_BACKEND, nodes=az.SEARCH_BACKEND),
                  precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                 cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                                 cudnn_deterministic=torch.backends.cudnn.deterministic), pairs=[])
    start = time.perf_counter()
    with journal.open("x", encoding="utf-8") as handle:
        def record(event, value):
            handle.write(json.dumps(dict(event=event, elapsed_seconds=time.perf_counter()-start, value=value))+"\n")
            handle.flush()
            probe.emit(event, value if event != "pair" else dict(index=value["index"],
                       simulations=value["simulations"], block=value["block"], layers=value["audit"]["layers"]))

        record("metadata", {k: v for k, v in result.items() if k != "pairs"})
        fixtures, trajectories = probe.collect_fixtures(net, config, device, args.trajectory_games,
                                                        args.roots_per_game, config.mcts_simulations)
        result.update(fixtures=[fixture["metadata"] for fixture in fixtures], trajectories=trajectories)
        record("fixtures", dict(fixtures=result["fixtures"], trajectories=trajectories))
        for index, fixture in enumerate(fixtures):
            for simulations in (config.mcts_simulations, config.strong_mcts_simulations):
                for block in range(args.seed_blocks):
                    seed = 20267000+block*1000+index
                    modes = (False, True) if (index+block) % 2 == 0 else (True, False)
                    runs = {mode: run_search(net, config, device, fixture, simulations, seed, mode) for mode in modes}
                    if runs[False]["exact"] != runs[True]["exact"]:
                        raise AssertionError(f"instrumentation changed search: {index}/{simulations}/{block}")
                    pair = dict(index=index, metadata=fixture["metadata"], simulations=simulations, block=block,
                                seed=seed, exact=runs[False]["exact"], plain_seconds=runs[False]["seconds"],
                                measured_seconds=runs[True]["seconds"], audit=runs[True]["audit"])
                    result["pairs"].append(pair)
                    record("pair", pair)
        verify_hashes(hashes)
        result.update(summary=summarize(result["pairs"]), protected_files_unchanged=True,
                      elapsed_seconds=time.perf_counter()-start)
        record("summary", result["summary"])
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    probe.emit("complete", dict(output=str(args.output), pairs=len(result["pairs"]),
                                seconds=result["elapsed_seconds"]))


if __name__ == "__main__":
    main()
