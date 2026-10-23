"""A/B 实验串行 runner（Stage 2A / v4 调参）。

每个 arm 在独立目录 ``data/ab/<name>/`` 下从指定起点模型热启动、清空重放缓冲
（不复用旧自博弈数据），跑固定轮数后分别评估 h2h 与外部基线；结果汇总到
``data/ab/results.jsonl``，结束时打印对比表。中断后重跑会自动跳过已完成的 arm。

用法：
    python tools/ab_runner.py --config tools/ab_experiments.json
    python tools/ab_runner.py --config tools/ab_experiments.json --only chance-k4
    python tools/ab_runner.py --config tools/ab_experiments.json --dry-run

配置文件字段见 ``tools/ab_experiments.json``；所有相对路径都相对项目根目录。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import azero  # noqa: E402
import jieqi as jq  # noqa: E402


def resolve(path: str) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="A/B 实验串行 runner")
    parser.add_argument("--config", default="tools/ab_experiments.json")
    parser.add_argument("--only", action="append", default=[], help="只跑指定 arm（可重复或逗号分隔）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force", action="store_true", help="忽略 .done 标记重新执行")
    return parser.parse_args(argv)


def tee_command(cmd: list[str], log_path: Path) -> None:
    """执行命令并把输出同时写入日志与终端。"""
    printable = " ".join(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"命令失败 (exit={code}): {printable}")


def prepare_arm(arm_dir: Path, start_model: Path) -> None:
    """准备 arm 目录：复制起点模型并写入 iteration=0 的训练状态（可续跑）。"""
    models_dir = arm_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    state_path = models_dir / azero.STATE_FILE_NAME
    latest_path = models_dir / azero.MODEL_LATEST_NAME
    if state_path.exists() and latest_path.exists():
        state = azero.load_training_state(str(state_path))
        print(f"[准备] 检测到已有进度 iteration={state.iteration}，继续该 arm。")
        return
    if not start_model.exists():
        raise FileNotFoundError(f"起点模型不存在: {start_model}")
    shutil.copyfile(start_model, latest_path)
    azero.save_training_state(
        str(state_path),
        azero.TrainingState(iteration=0, best_score=0.0, total_games=0),
    )
    print(f"[准备] 已从 {start_model.name} 热启动（iteration=0，重放缓冲清空）。")


def read_train_iteration(arm_dir: Path) -> int:
    state_path = arm_dir / "models" / azero.STATE_FILE_NAME
    if not state_path.exists():
        return -1
    return azero.load_training_state(str(state_path)).iteration


def read_summaries(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "summary":
                rows.append(obj)
    return rows


def append_result(result_path: Path, row: dict) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_results(result_path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not result_path.exists():
        return latest
    with open(result_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                latest[row["arm"]] = row
    return latest


def print_table(results: dict[str, dict]) -> None:
    if not results:
        print("（暂无结果）")
        return
    header = f"{'arm':<16}{'训练(s)':>9}  " + "  ".join(f"{name:>22}" for name in ("evals",))
    print("\n===== A/B 汇总 =====")
    for arm, row in results.items():
        pieces = []
        for name, summary in row.get("evals", {}).items():
            pieces.append(
                f"{name}: {summary['score']:.3f} "
                f"({summary['wins']}W/{summary['draws']}D/{summary['losses']}L)"
            )
        train_sec = row.get("train_seconds", 0)
        print(f"{arm:<16}{train_sec:>9.0f}  " + " | ".join(pieces))


def main(argv=None) -> int:
    args = parse_args(argv)
    config_path = resolve(args.config)
    with open(config_path, encoding="utf-8") as handle:
        cfg = json.load(handle)

    start_model = resolve(cfg["start_model"])
    iterations = int(cfg.get("iterations", 32))
    base_args = [str(item) for item in cfg.get("base_args", [])]
    arms = cfg.get("arms", [])
    evals = cfg.get("evals", [])

    only: set[str] = set()
    for item in args.only:
        only.update(part.strip() for part in item.split(",") if part.strip())

    ab_dir = resolve(cfg.get("work_dir", "data/ab"))
    result_path = ab_dir / "results.jsonl"
    results = load_results(result_path)

    for arm in arms:
        name = arm["name"]
        if only and name not in only:
            continue
        arm_dir = ab_dir / name
        done_marker = arm_dir / ".done"
        if done_marker.exists() and not args.force:
            print(f"[跳过] {name} 已完成（存在 .done）")
            continue

        arm_args = [str(item) for item in arm.get("args", [])]
        models_dir = arm_dir / "models"
        train_cmd = [
            sys.executable,
            str(ROOT / "azero.py"),
            "--iterations", str(iterations),
            "--data-dir", str(arm_dir),
            "--model-dir", str(models_dir),
            "--log-dir", str(arm_dir / "logs"),
            *base_args,
            *arm_args,
        ]
        print(f"\n===== arm: {name} =====")
        if args.dry_run:
            print("[dry-run] $", " ".join(train_cmd))
            for spec in evals:
                print("[dry-run] $", " ".join(build_eval_cmd(arm_dir, spec)))
            continue

        train_seconds = 0.0
        if not args.skip_train:
            prepare_arm(arm_dir, start_model)
            done_iteration = read_train_iteration(arm_dir)
            if done_iteration >= iterations:
                print(f"[训练] 已完成 iteration={done_iteration} >= {iterations}，跳过训练。")
            else:
                started = time.time()
                tee_command(train_cmd, arm_dir / "logs" / "ab_train.log")
                train_seconds = time.time() - started
                print(f"[训练] {name} 完成，用时 {train_seconds / 3600:.2f} h")

        arm_row = results.get(name, {"arm": name, "evals": {}})
        arm_row["train_seconds"] = train_seconds
        if not args.skip_eval:
            for spec in evals:
                eval_name = spec["name"]
                out_path = arm_dir / f"eval_{eval_name}.json"
                eval_cmd = build_eval_cmd(arm_dir, spec, out_path)
                if out_path.exists() and not args.force:
                    print(f"[评估] {eval_name} 已有 {out_path.name}，直接读取。")
                else:
                    tee_command(eval_cmd, arm_dir / "logs" / f"ab_eval_{eval_name}.log")
                summaries = read_summaries(out_path)
                if not summaries:
                    print(
                        f"[错误] {eval_name} 未产出有效 summary: {out_path}；"
                        f"arm={name} 不写 .done，请修复后重跑。",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
                arm_row["evals"][eval_name] = summaries[0]
            append_result(result_path, arm_row)
            results[name] = arm_row
            done_marker.write_text("done\n", encoding="utf-8")
            print(f"[完成] arm={name}")
        else:
            print(f"[警告] --skip-eval: arm={name} 未评估，不写 .done。")

    print_table(results)
    return 0


def build_eval_cmd(arm_dir: Path, spec: dict, out_path: Path | None = None) -> list[str]:
    out_path = out_path or (arm_dir / f"eval_{spec['name']}.json")
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "eval.py"),
        "--model", str(arm_dir / "models" / azero.MODEL_LATEST_NAME),
        "--opponents", str(spec["opponents"]),
        "--games", str(int(spec.get("games", 120))),
        "--sims", str(int(spec.get("sims", 400))),
        # 历史实验保持原 leaf64；新评估标准不隐式改变已有 A/B 配置。
        "--leaf-batch-size", str(int(spec.get("leaf_batch_size", 64))),
        "--seed", str(int(spec.get("seed", 20260911))),
        "--workers", str(int(spec.get("workers", 8))),
        "--out", str(out_path),
    ]
    if "chance_nodes" in spec:
        cmd.append("--chance-nodes" if spec["chance_nodes"] else "--no-chance-nodes")
    if "chance_k" in spec:
        cmd.extend(["--chance-k", str(int(spec["chance_k"]))])
    return cmd


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
