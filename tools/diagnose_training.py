"""Read fixed diagnostic curves with checkpoint/attempt commit status."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import run_records as rr
import training_diagnostics as td


def inspect_diagnostics(data_dir, checkpoint=None):
    data_dir = Path(data_dir)
    lineage = state = None
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != "jieqi-training":
            raise ValueError("--checkpoint must be a full training checkpoint")
        lineage, state = saved.get("provenance"), saved.get("diagnostics")
        if lineage is not None:
            rr.validate_lineage(lineage)
        td.validate_state(state, saved["train_state"]["iteration"],
                          lineage["model_sha256"] if lineage else None)
    reports = {path.stem: json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((data_dir / "diagnostics" / "reports").glob("*.json"))}
    if state and state.get("latest"):
        latest = state["latest"]
        key = latest["provenance"]["attempt_id"]
        if key in reports and reports[key] != latest:
            raise ValueError("diagnostic report differs from committed checkpoint")
        reports.setdefault(key, latest)
    rows = []
    for key, report in reports.items():
        source = report["provenance"]
        if report["version"] != td.VERSION or key != source["attempt_id"]:
            raise ValueError("diagnostic report version/attempt mismatch")
        if report["completed_iterations"] != source["iteration"] + 1:
            raise ValueError("diagnostic report iteration mismatch")
        status = rr.attempt_status(data_dir, dict(**source, evaluated_model_sha256=report["evaluated_model_sha256"]),
                                   lineage)
        if status == "committed" and lineage and lineage.get("attempt") == source:
            if report["evaluated_model_sha256"] != lineage["model_sha256"]:
                raise ValueError("diagnostic evaluated model differs from checkpoint")
        rows.append(dict(status=status, **report))
    rows.sort(key=lambda row: (row["completed_iterations"], row["provenance"]["attempt_id"]))
    dataset = state["dataset"] if state else None
    if dataset:
        for record in dataset["sets"].values():
            path = td.dataset_path(data_dir, record["path"])
            if rr.file_record(path)["sha256"] != record["sha256"]:
                raise ValueError("fixed diagnostic data changed: " + str(path))
    return dict(dataset=dataset, reports=rows,
                note="Use committed rows with the same dataset_sha256. Targets are fixed self-play targets; metrics do not measure playing strength.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读固定留出诊断；区分已提交轮次与中断尝试")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", help="完整检查点；校验固定数据，并补充未写出提交标记的最新轮次")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_diagnostics(args.data_dir, args.checkpoint), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
