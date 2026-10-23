"""Offline numerical gate for merging original leaf requests; no search uses merged outputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
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


def forward(net, batch, device):
    with torch.no_grad():
        logits, values = net(torch.from_numpy(batch).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy(), np.asarray(values.tolist(), dtype=np.float64)


def compare_group(net, requests, device):
    reference = [forward(net, batch, device) for batch in requests]
    merged = np.concatenate(requests, axis=0)
    policies, values = forward(net, merged, device)
    rows, offset = [], 0
    for batch, (old_policies, old_values) in zip(requests, reference):
        n = len(batch)
        new_policies, new_values = policies[offset:offset+n], values[offset:offset+n]
        rows.append(dict(original_batch=n, merged_batch=len(merged), offset=offset,
            input_sha256=hashlib.sha256(batch.tobytes()).hexdigest(),
            policy_equal=old_policies.tobytes() == new_policies.tobytes(),
            value_equal=old_values.tobytes() == new_values.tobytes(),
            policy_max_abs_diff=float(np.max(np.abs(old_policies-new_policies))),
            value_max_abs_diff=float(np.max(np.abs(old_values-new_values)))))
        offset += n
    return rows


def collect_requests(net, config, fixtures, device):
    captured = []
    original = az._encode_batch_numpy
    for fixture in fixtures:
        batches = []

        def capture(*args, **kwargs):
            batch = original(*args, **kwargs)
            if len(batches) < 4:
                batches.append(batch.copy())
            return batch

        with mock.patch.object(az, "_encode_batch_numpy", capture):
            result = probe.run_case(net, config, device, fixture, False,
                                    20266000+fixture["metadata"]["index"], "plain")
        if batches:
            captured.append(dict(fixture=fixture["metadata"], batches=batches,
                                 search_exact=result["exact"]))
    # Interleave sources to combine independent requests instead of adjacent calls only.
    requests = [(row["fixture"]["index"], row["batches"][i]) for i in range(4)
                for row in captured if i < len(row["batches"])]
    return captured, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"data/train_rules5_20261004")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("new output in existing directory required")
    if (jq.RULES_BACKEND != "cython" or jq.LEGAL_BACKEND != "cython" or az.SEARCH_BACKEND != "python"
            or not torch.cuda.is_available()):
        parser.error("require CUDA, rules/legal Cython and Python nodes")
    torch.set_num_threads(1)
    checkpoint = args.run_dir/"models/training_checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = az.AZConfig(**saved["config"])
    config.lazy_chance_allocation = True
    device = torch.device("cuda")
    paths = [checkpoint, args.run_dir/"models/latest.pt", args.run_dir/"models/train_state.json",
             args.run_dir/"logs/train.log", args.run_dir/"selfplay"/saved["replay_files"][-1]["name"],
             ROOT/"azero.py", ROOT/"jieqi.py", ROOT/"ismcts.py", ROOT/"run_records.py", Path(__file__),
             Path(central.__file__), Path(probe.__file__), ROOT/"tools/measure_search_reuse.py",
             ROOT/"native/jieqi_rules.pyx", ROOT/"native/jieqi_legal.pyx",
             Path(jq.RULES_BACKEND_NATIVE["path"]), Path(jq.LEGAL_BACKEND_NATIVE["path"])]
    hashes = {str(p): rr.file_record(p)["sha256"] for p in paths}
    net = az.PolicyValueNet(**{k: saved["network"][k] for k in ("channels", "blocks")}).to(device).eval()
    net.load_state_dict(saved["model"])
    start = time.perf_counter()
    fixtures, trajectories = probe.collect_fixtures(net, config, device, 2, 4, 0)
    corpus, requests = collect_requests(net, config, fixtures, device)
    result = dict(source_hashes=hashes, checkpoint_iteration=saved["train_state"]["iteration"],
                  config=asdict(config), precision=central.precision_settings(), trajectories=trajectories,
                  fixtures=[row["fixture"] for row in corpus], comparisons=[])
    for width in (2, 4):
        for begin in range(0, len(requests)-width+1, width):
            selected = requests[begin:begin+width]
            rows = compare_group(net, [batch for _, batch in selected], device)
            result["comparisons"].append(dict(requests=width, sources=[index for index, _ in selected], rows=rows))
    verify_hashes(hashes)
    rows = [row for group in result["comparisons"] for row in group["rows"]]
    if not rows:
        raise AssertionError("empty merge probe")
    result.update(elapsed_seconds=time.perf_counter()-start, production_unchanged=True,
                  summary=dict(request_comparisons=len(rows),
                    mismatches=sum(not r["policy_equal"] or not r["value_equal"] for r in rows),
                    policy_max_abs_diff=max(r["policy_max_abs_diff"] for r in rows),
                    value_max_abs_diff=max(r["value_max_abs_diff"] for r in rows)))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    probe.emit("complete", dict(output=str(args.output), **result["summary"], production_unchanged=True))


if __name__ == "__main__":
    main()
