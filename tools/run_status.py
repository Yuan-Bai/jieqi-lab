"""Read run manifests and commit markers; never modify training artifacts."""
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


def inspect_runs(data_dir: str | Path, checkpoint: str | Path | None = None) -> dict:
    data_dir = Path(data_dir)
    lineage = None
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != "jieqi-training":
            raise ValueError("--checkpoint must be a full training checkpoint")
        lineage = saved.get("provenance")
        if lineage is not None:
            rr.validate_lineage(lineage)
            if lineage["completed_iterations"] != saved["train_state"]["iteration"]:
                raise ValueError("checkpoint/lineage iteration mismatch")
    manifests = {}
    for path in sorted((data_dir / "runs").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if rr.run_dir(data_dir, manifest["run_id"]) != path.parent:
            raise ValueError(f"manifest path/ID mismatch: {path}")
        manifests[manifest["run_id"]] = manifest
    # A commit-marker write can be interrupted. The checkpoint still proves its own attempt.
    if lineage is not None:
        manifests.setdefault(lineage["manifest"]["run_id"], lineage["manifest"])
    runs = []
    for run_id, manifest in manifests.items():
        folder = rr.run_dir(data_dir, run_id)
        entry = dict(run_id=run_id, kind=manifest["kind"], code_sha256=manifest["code"]["sha256"],
                     parent=manifest["parent"], attempts=[])
        attempts = {path.stem: json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted((folder / "attempts").glob("*.json"))}
        if lineage is not None and lineage["manifest"]["run_id"] == run_id and lineage.get("attempt"):
            attempts.setdefault(lineage["attempt"]["attempt_id"], lineage["attempt"])
        for attempt in attempts.values():
            if attempt["manifest_sha256"] != rr.record_hash(manifest):
                raise ValueError("attempt/manifest hash mismatch")
            entry["attempts"].append(dict(iteration=attempt["iteration"], attempt_id=attempt["attempt_id"],
                                          status=rr.attempt_status(data_dir, attempt, lineage)))
        if manifest["kind"] == "evaluation":
            complete = folder / "complete.json"
            entry["status"] = "unconfirmed"
            if complete.exists():
                marker = json.loads(complete.read_text(encoding="utf-8"))
                if any(marker[key] != value for key, value in rr.source_ref(manifest).items()):
                    raise ValueError("evaluation completion/manifest mismatch")
                output = marker["output"]
                if not Path(output["path"]).exists():
                    entry["status"] = "output_missing"
                elif rr.file_record(output["path"])["sha256"] != output["sha256"]:
                    entry["status"] = "output_changed"
                else:
                    entry["status"] = "complete"
        runs.append(entry)
    return dict(runs=runs, note="unconfirmed means no commit evidence was found; historical provenance cannot be reconstructed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="只读检查运行来源及轮次提交状态；未确认结果不用于训练验收")
    parser.add_argument("--data-dir", default="data", help="训练 data-dir；独立评估使用输出文件所在目录")
    parser.add_argument("--checkpoint", help="可选完整检查点，用于识别提交成功但标记尚未落盘的轮次")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_runs(args.data_dir, args.checkpoint), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
