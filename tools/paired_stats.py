# -*- coding: utf-8 -*-
"""配对评估统计：读取 tools/eval.py 的 JSONL，输出 Wilson CI 与配对检验。

统计口径（ADR-0007）：

- 单局得分 w=1 / d=0.5 / l=0，未终局（ply_limit）不计分；
- 配对设计（同初始洗牌红黑互换）：每 pair 两局得分均值 d_i ∈ [0,1]，
  H0: E[d] = 0.5（无差异）；配对检验用 t（正态近似）与符号翻转置换检验，
  bootstrap 给出配对均值 CI；红黑分层报告换色均衡性。

用法::

    python tools/paired_stats.py data/eval/chance_ab/s400_n1600.jsonl \
        [--out data/eval/chance_ab/s400_n1600_stats.json] [--perm 20000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_jsonl(path: str) -> tuple[dict | None, list[dict], list[dict]]:
    """读取评估 JSONL，返回 (run, games, summaries)。"""
    run = None
    games: list[dict] = []
    summaries: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("type")
            if kind == "run":
                if run is not None:
                    raise ValueError("multiple runs in one file; analyze each run separately")
                run = record
            elif kind == "game":
                games.append(record)
            elif kind == "summary":
                summaries.append(record)
    return run, games, summaries


def game_value(record: dict) -> float | None:
    """单局得分映射到 [0,1]：w=1 / d=0.5 / l=0；未终局返回 None。"""
    if record.get("score") is None:
        return None
    return (float(record["score"]) + 1.0) / 2.0


def wilson_score_interval(wins: int, draws: int, losses: int, z: float = 1.96) -> tuple[float, float, float]:
    """按“胜=1、和=0.5”的得分计算 Wilson 区间（与 tools/eval.py 同口径）。"""
    games = wins + draws + losses
    if games <= 0:
        return 0.0, 0.0, 0.0
    point = (wins + 0.5 * draws) / games
    successes = wins + 0.5 * draws
    n = float(games)
    denom = 1.0 + z * z / n
    center = (successes / n + z * z / (2 * n)) / denom
    half = z * math.sqrt(point * (1.0 - point) / n + z * z / (4 * n * n)) / denom
    return point, max(center - half, 0.0), min(center + half, 1.0)


def sign_flip_p(delta: np.ndarray, rng: np.random.Generator, permutations: int, chunk: int = 2000) -> float:
    """符号翻转置换检验（双尾）：H0 下 d_i-0.5 关于 0 对称。"""
    n = len(delta)
    if n == 0:
        return 1.0
    stat = abs(float(np.mean(delta)))
    count = 0
    done = 0
    choices = np.array([-1.0, 1.0])
    while done < permutations:
        size = min(chunk, permutations - done)
        signs = rng.choice(choices, size=(size, n))
        stats = np.abs(signs @ delta) / n
        count += int(np.sum(stats >= stat - 1e-12))
        done += size
    return (count + 1) / (permutations + 1)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, resamples: int, chunk: int = 2000) -> tuple[float, float]:
    """配对均值的 bootstrap 95% CI。"""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    means = []
    done = 0
    while done < resamples:
        size = min(chunk, resamples - done)
        indices = rng.integers(0, n, size=(size, n))
        means.append(values[indices].mean(axis=1))
        done += size
    pooled = np.concatenate(means)
    return float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))


def pair_summary(pair_scores, resamples: int = 20000, seed: int = 1) -> dict:
    """Independent bootstrap of complete two-game means; never consumes game RNG.

    Percentile bootstrap is uninformative for an empty/single/constant sample;
    expose that limitation instead of interpreting a zero-width interval as proof.
    """
    values = np.asarray(pair_scores, dtype=np.float64)
    if resamples < 1 or values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("invalid bootstrap input")
    if np.any((values < 0) | (values > 1)) or np.any(values * 4 != np.floor(values * 4)):
        raise ValueError("pair scores must be 0, .25, .5, .75 or 1")
    counts = [int(np.sum(values == value)) for value in (0, .25, .5, .75, 1)]
    interval = bootstrap_ci(values, np.random.default_rng(seed), resamples) if len(values) else (None, None)
    return {
        "pair_mean": float(values.mean()) if len(values) else None,
        "pentanomial": counts,
        "bootstrap_ci95": list(interval),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "bootstrap_degenerate": len(values) < 2 or len(set(values.tolist())) < 2,
        "bootstrap_method": "paired_percentile_bootstrap",
    }


def analyze(path: str, permutations: int, bootstrap: int, seed: int) -> dict:
    """对单个评估 JSONL 做整体 + 配对统计，返回结果字典。"""
    run, games, summaries = load_jsonl(path)
    opponents = sorted({record.get("opponent", "unknown") for record in games})
    if len(opponents) > 1:
        return dict(file=path, run=run, games=len(games), opponents={
            opponent: analyze_games(path, run, [r for r in games if r.get("opponent", "unknown") == opponent],
                                    permutations, bootstrap, seed)
            for opponent in opponents})
    return analyze_games(path, run, games, permutations, bootstrap, seed)


def analyze_games(path, run, games, permutations, bootstrap, seed) -> dict:
    rng = np.random.default_rng(seed)

    unfinished = sum(1 for record in games if record.get("score") is None)

    by_pair: dict = {}
    for record in games:
        by_pair.setdefault(record.get("pair"), []).append(record)
    deltas: list[float] = []
    valid_games = []
    unfinished_pairs = invalid_pairs = 0
    for pair, records in by_pair.items():
        if (pair is None or len(records) != 2
                or {record.get("neural_side") for record in records} != {"red", "black"}
                or records[0].get("seed") is None
                or records[0].get("seed") != records[1].get("seed")):
            invalid_pairs += 1
            continue
        values = [game_value(record) for record in records]
        if any(value is None for value in values):
            unfinished_pairs += 1
            continue
        deltas.append(sum(values) / 2)
        valid_games.extend(records)
    delta_arr = np.asarray(deltas, dtype=np.float64)
    wins = sum(record.get("score") == 1.0 for record in valid_games)
    draws = sum(record.get("score") == 0.0 for record in valid_games)
    losses = sum(record.get("score") == -1.0 for record in valid_games)
    point, low, high = wilson_score_interval(wins, draws, losses)

    if len(delta_arr) > 1:
        mean_d = float(delta_arr.mean())
        sd_d = float(delta_arr.std(ddof=1))
        se_d = sd_d / math.sqrt(len(delta_arr))
        t_stat = (mean_d - 0.5) / se_d if se_d > 0 else None
        p_normal = math.erfc(abs(t_stat) / math.sqrt(2.0)) if t_stat is not None else None
    else:
        mean_d = float(delta_arr.mean()) if len(delta_arr) else 0.0
        sd_d = se_d = 0.0
        t_stat = p_normal = None
    p_perm = sign_flip_p(delta_arr - 0.5, rng, permutations) if len(delta_arr) else 1.0
    paired = pair_summary(delta_arr, bootstrap, seed)

    def side_score(side: str) -> dict:
        values = [game_value(record) for record in valid_games if record.get("neural_side") == side]
        valid = [value for value in values if value is not None]
        return {
            "games": len(valid),
            "score": float(np.mean(valid)) if valid else 0.0,
        }

    red = side_score("red")
    black = side_score("black")
    result = {
        "file": path,
        "run": run,
        "games": len(games),
        "unfinished": unfinished,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": point,
        "ci95_low": low,
        "ci95_high": high,
        "pairs_total": len(by_pair),
        "pairs_used": len(delta_arr),
        "pairs_excluded_unfinished": unfinished_pairs,
        "pairs_excluded_invalid": invalid_pairs,
        "pair_mean": mean_d,
        "pair_sd": sd_d,
        "pair_se": se_d,
        "t_stat": t_stat,
        "p_normal": p_normal,
        "p_permutation": p_perm,
        **paired,
        "red": red,
        "black": black,
    }
    return result


def print_report(result: dict) -> None:
    """打印单个文件的分析结果。"""
    if "opponents" in result:
        for opponent, group in result["opponents"].items():
            print(f"对手: {opponent}")
            print_report(group)
        return
    run = result["run"] or {}
    print(f"===== {os.path.basename(result['file'])} =====")
    print(
        f"run: sims={run.get('mcts_simulations')} chance_nodes={run.get('chance_nodes')} "
        f"opponent_chance_nodes={run.get('opponent_chance_nodes')} seed={run.get('seed')}"
    )
    print(
        f"整体: W/D/L={result['wins']}/{result['draws']}/{result['losses']} "
        f"未终局={result['unfinished']} score={result['score']:.4f} "
        f"CI95=[{result['ci95_low']:.4f},{result['ci95_high']:.4f}]"
    )
    interval = result["bootstrap_ci95"]
    ci_label = f"[{interval[0]:.4f},{interval[1]:.4f}]" if interval[0] is not None else "NA"
    mean_label = f"{result['pair_mean']:.4f}" if result['pair_mean'] is not None else "NA"
    print(
        f"配对: n={result['pairs_used']}/{result['pairs_total']}（未终局 {result['pairs_excluded_unfinished']}，"
        f"不完整/无效 {result['pairs_excluded_invalid']}），"
        f"d̄={mean_label} SD={result['pair_sd']:.4f} "
        f"bootstrap CI95={ci_label}"
    )
    if result["bootstrap_degenerate"]:
        print("注意：配对样本不足或零方差，bootstrap 区间不能用于确认精度或优势。")
    t_label = f"{result['t_stat']:+.3f}" if result["t_stat"] is not None else "NA (zero variance)"
    p_label = f"{result['p_normal']:.4f}" if result["p_normal"] is not None else "NA"
    print(
        f"检验: t={t_label} 正态 p={p_label} "
        f"符号翻转置换 p={result['p_permutation']:.4f}"
    )
    print(
        f"换色: red={result['red']['score']:.4f}（n={result['red']['games']}） "
        f"black={result['black']['score']:.4f}（n={result['black']['games']}）"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="配对评估统计（tools/eval.py JSONL）")
    parser.add_argument("files", nargs="+", help="评估 JSONL 文件（可多个，逐个统计）")
    parser.add_argument("--out", default=None, help="统计结果 JSON 输出路径（缺省=同名 .stats.json）")
    parser.add_argument("--perm", type=int, default=20000, help="符号翻转置换次数")
    parser.add_argument("--bootstrap", type=int, default=20000, help="bootstrap 重采样次数")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    if args.perm < 1 or args.bootstrap < 1:
        parser.error("--perm and --bootstrap must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    results = []
    for path in args.files:
        result = analyze(path, args.perm, args.bootstrap, args.seed)
        results.append(result)
        print_report(result)
        print()
    out_path = args.out or (os.path.splitext(args.files[0])[0] + ".stats.json")
    with open(out_path, "x", encoding="utf-8") as handle:
        json.dump(results if len(results) > 1 else results[0], handle, ensure_ascii=False, indent=2)
    print(f"统计结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
