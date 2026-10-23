"""模型评估协议：配对双色、冻结多对手、配对 bootstrap 区间、JSONL 记录。

用法示例::

    python tools/eval.py --quick
    python tools/eval.py --model data/models/latest.pt --opponents ris:800,random \
        --games 200 --sims 800 --seed 20260911 --out data/eval/acceptance.jsonl --workers 8
    python tools/eval.py --opponents random --games 4 --sims 8 --workers 1  # 串行调试
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

import azero
import evaluation_runtime
import ismcts as im
import jieqi as jq
import run_records as rr
from tools.paired_stats import pair_summary

DEFAULT_SEED = 20260911
DEFAULT_MAX_PLIES = 600


def load_opponent_pool(path: str) -> dict:
    """读取固定模型池；相对权重路径以清单目录为基准，哈希/版本不符即拒绝。"""
    source = rr.file_record(path)
    pool = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if rr.file_record(path) != source:
        raise RuntimeError("读取期间对手池清单改变")
    if (not isinstance(pool, dict) or pool.get("version") != 1 or pool.get("ruleset_version") != jq.RULESET_VERSION
            or not isinstance(pool.get("id"), str) or not pool["id"].strip()
            or not isinstance(pool.get("models"), list) or not pool["models"]):
        raise ValueError("对手池需要 version=1、当前 ruleset_version、id 和非空 models")
    search = pool.get("search")
    if search is not None and (not isinstance(search, dict)
            or set(search) != {"mcts_simulations", "leaf_batch_size"}
            or any(type(v) is not int or v < 1 for v in search.values())):
        raise ValueError("对手池 search 需要正整数 mcts_simulations 和 leaf_batch_size")
    opponents, sources, identities = [], {}, set()
    for entry in pool["models"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError("对手池模型需要非空 path")
        sims, sha = entry.get("sims"), entry.get("sha256")
        if (type(sims) is not int or sims < 1 or not isinstance(sha, str)
                or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("对手池模型需要正整数 sims 和小写 SHA256")
        if search is not None and sims != search["mcts_simulations"]:
            raise ValueError("固定评估基准要求候选与池中模型使用相同模拟预算")
        model_path = (Path(path).resolve().parent / entry["path"]).resolve()
        record = rr.file_record(model_path)
        if record["sha256"] != sha:
            raise ValueError(f"对手池模型 SHA256 不匹配: {model_path}")
        identity = (sha, sims)
        if identity in identities:
            raise ValueError("对手池不能重复同一权重/预算")
        identities.add(identity)
        spec = f"model:{model_path}:{sims}"
        opponents.append(spec)
        sources[spec] = record
    return dict(id=pool["id"], source=source, opponents=opponents, sources=sources, search=search)


def validate_pool_search(pool: dict, sims: int, leaf_batch_size: int) -> None:
    """新池可固定双方基准，防止命令行覆盖后仍沿用同一基准名。"""
    expected = pool.get("search")
    if expected is not None and expected != dict(mcts_simulations=sims, leaf_batch_size=leaf_batch_size):
        raise ValueError(f"评估搜索配置与冻结池不一致，要求 {expected}；改变基准请新建池清单")


def wilson_score_interval(wins: int, draws: int, losses: int, z: float = 1.96) -> tuple[float, float, float]:
    """按“胜=1、和=0.5”的得分计算 Wilson 区间。"""
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


class RandomOpponent:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def select_move(self, state: jq.GameState, belief: im.BeliefModel, history) -> tuple[int, int] | None:
        moves = state.legal_moves()
        if not moves:
            return None
        return self.rng.choice(moves)


class RisOpponent:
    def __init__(self, iterations: int, rng: random.Random) -> None:
        self.agent = im.ISMCTSAgent(im.ISMCTSConfig(iterations=iterations), rng)

    def select_move(self, state: jq.GameState, belief: im.BeliefModel, history) -> tuple[int, int] | None:
        move, _ = self.agent.select_move(state, belief)
        return move


class NeuralOpponent:
    def __init__(self, net, config: azero.AZConfig, device: torch.device, rng: random.Random) -> None:
        self.mcts = azero.NeuralMCTS(net, config, device, rng)

    def select_move(self, state: jq.GameState, belief: im.BeliefModel, history) -> tuple[int, int] | None:
        _, move = self.mcts.search(
            state,
            belief,
            history,
            temperature=1e-6,
            simulations=self.mcts.config.mcts_simulations,
            add_dirichlet=False,
        )
        return move


def parse_opponent_spec(spec: str) -> tuple[str, str | None, int | None]:
    """解析对手描述，返回 ``(kind, path, sims)``。

    路径允许包含冒号（如 Windows 盘符 ``model:C:\\models\\best.pt:400``）：
    仅当最后一个冒号后是纯数字时才视为模拟次数后缀。
    """
    name, _, rest = spec.partition(":")
    name = name.strip().lower()
    rest = rest.strip()
    if name == "model":
        if not rest:
            raise ValueError("model opponent requires a path: model:<path>[:sims]")
        path, sep, sims_text = rest.rpartition(":")
        if sep and sims_text.isdigit():
            return name, path, int(sims_text)
        return name, rest, None
    if name in ("ris", "pika"):
        return name, None, int(rest) if rest else None
    return name, rest or None, None


def build_search_config(sims: int, chance_nodes: bool = True, chance_k: int = 0,
                        leaf_batch_size: int = azero.DEFAULT_EVAL_LEAF_BATCH_SIZE) -> azero.AZConfig:
    """构造评估用搜索配置；机会节点开关默认对双方一致，保证模型比较公平。"""
    if leaf_batch_size < 1:
        raise ValueError("leaf_batch_size must be positive")
    return azero.AZConfig(mcts_simulations=sims, chance_nodes=chance_nodes, chance_k=chance_k,
                         leaf_batch_size=leaf_batch_size)


def resolve_opponent_chance(
    chance_nodes: bool,
    chance_k: int,
    opponent_mode: str,
    opponent_k: int | None,
) -> tuple[bool, int]:
    """把 CLI 的对手机会节点设置解析为具体值（inherit=跟随己方，用于搜索 A/B 时显式 on/off）。"""
    nodes = chance_nodes if opponent_mode == "inherit" else opponent_mode == "on"
    k = chance_k if opponent_k is None else opponent_k
    return nodes, k


def check_model_source(path: str, expected_source: dict | None) -> None:
    if expected_source is not None and rr.file_record(path)["sha256"] != expected_source["sha256"]:
        raise RuntimeError(f"评估期间对手模型改变，拒绝混用来源: {path}")


def build_opponent_factory(spec: str, device: torch.device, net, chance_nodes: bool = True,
                           chance_k: int = 0, expected_source: dict | None = None,
                           pika_config: dict | None = None,
                           leaf_batch_size: int = azero.DEFAULT_EVAL_LEAF_BATCH_SIZE,
                           default_model_sims: int = azero.DEFAULT_EVAL_SIMULATIONS):
    name, path, sims = parse_opponent_spec(spec)
    if name == "random":
        return lambda rng: RandomOpponent(rng)
    if name == "ris":
        return lambda rng: RisOpponent(sims if sims is not None else 800, rng)
    if name == "pika":
        from pika_engine import PikaOpponent
        return lambda rng: PikaOpponent(pika_config, sims if sims is not None else 10000)
    if name == "model":
        if net is None:
            raise ValueError("model opponent requires --model to be loaded")
        other = azero.PolicyValueNet()
        check_model_source(path, expected_source)
        if not azero.load_model_if_exists(other, path, device):
            raise FileNotFoundError(path)
        check_model_source(path, expected_source)
        other.to(device)
        other.eval()
        config = build_search_config(sims if sims is not None else default_model_sims,
                                     chance_nodes, chance_k, leaf_batch_size)
        return lambda rng: NeuralOpponent(other, config, device, rng)
    raise ValueError(f"unknown opponent spec: {spec}")


def play_one_game(
    neural_factory,
    opponent_factory,
    neural_side: int,
    seed: int,
    max_plies: int,
) -> dict:
    """单局对弈；返回可写入 JSONL 的记录。"""
    rng, neural_rng, opponent_rng = azero.evaluation_rngs(seed)
    neural = opponent = None
    # 同一 seed 预物化同一副初始洗牌：pair 内换色两局共享暗子真值分配（ADR-0007 §1）。
    dark_types = jq.materialize_dark_types(random.Random(f"jieqi-pair-shuffle:{seed}"))
    state = jq.GameState.initial(rng=rng, dark_types=dark_types)
    belief = im.init_belief_for_game(state)
    history: list[tuple[int, int]] = []
    stop_reason = "no_move"
    elapsed = {"neural": 0.0, "opponent": 0.0}
    moves_played = {"neural": 0, "opponent": 0}
    try:
        neural = neural_factory(neural_rng)
        opponent = opponent_factory(opponent_rng)
        while state.game_outcome()[0] is None and len(history) < max_plies:
            role = "neural" if state.side == neural_side else "opponent"
            agent = neural if role == "neural" else opponent
            started = time.perf_counter()
            move = agent.select_move(state, belief, history)
            elapsed[role] += time.perf_counter() - started
            if move is None:
                stop_reason = "no_move"
                break
            if move not in state.legal_moves():
                raise ValueError(f"{role} returned illegal move: {move}")
            im.apply_move_and_update_belief(state, belief, move, rng=rng)
            history.append(move)
            moves_played[role] += 1
    finally:
        for agent in (opponent, neural):
            close = getattr(agent, "close", None)
            if close is not None:
                close()
    outcome, reason = state.game_outcome()
    if outcome is None:
        reason = "ply_limit" if len(history) >= max_plies else stop_reason
    # 未终局（ply_limit）不计分：score 置 None，统计时单独记为 unfinished（ADR-0007）。
    score = azero.outcome_value_for_player(outcome, neural_side) if outcome is not None else None
    return {
        "seed": seed,
        "neural_side": "red" if neural_side == jq.RED else "black",
        "outcome": outcome,
        "reason": reason,
        "plies": len(history),
        "score": score,
        "move_seconds": elapsed,
        "moves_played": moves_played,
        "opponent_metadata": opponent.metadata() if hasattr(opponent, "metadata") else None,
    }


def tally_game(record: dict) -> str:
    """把单局记录归类为 win/draw/loss/unfinished；ply_limit 未终局不计分。"""
    outcome = record["outcome"]
    if outcome is None:
        return "unfinished"
    if outcome == jq.RESULT_DRAW:
        return "draw"
    neural_side = jq.RED if record["neural_side"] == "red" else jq.BLACK
    return "win" if azero.outcome_value_for_player(outcome, neural_side) > 0 else "loss"


_WORKER_STATE: dict = {}


def _init_eval_worker(model_state: dict, sims: int, device_str: str, chance_nodes: bool = True,
                      chance_k: int = 0, opponent_chance_nodes: bool | None = None,
                      opponent_chance_k: int | None = None,
                      opponent_sources: dict | None = None,
                      pika_config: dict | None = None,
                      leaf_batch_size: int = azero.DEFAULT_EVAL_LEAF_BATCH_SIZE) -> None:
    """评估子进程初始化：加载模型并缓存设备/配置（含对手独立机会节点设置）。"""
    torch.set_num_threads(1)
    device = torch.device(device_str)
    net = azero.PolicyValueNet()
    azero.load_state_dict_padded(net, model_state)
    net.to(device)
    net.eval()
    _WORKER_STATE["net"] = net
    _WORKER_STATE["device"] = device
    _WORKER_STATE["config"] = build_search_config(sims, chance_nodes, chance_k, leaf_batch_size)
    _WORKER_STATE["opponent_chance"] = (
        chance_nodes if opponent_chance_nodes is None else opponent_chance_nodes,
        chance_k if opponent_chance_k is None else opponent_chance_k,
    )
    _WORKER_STATE["opponents"] = {}
    _WORKER_STATE["opponent_sources"] = opponent_sources or {}
    _WORKER_STATE["pika_config"] = pika_config


def _start_eval_worker(*args) -> None:
    evaluation_runtime.isolate_worker()
    _init_eval_worker(*args)


def _build_opponent_factory_worker(spec: str, device: torch.device):
    """子进程内的对手工厂（缓存 model: 对手网络）。"""
    name, path, sims = parse_opponent_spec(spec)
    if name == "random":
        return lambda rng: RandomOpponent(rng)
    if name == "ris":
        return lambda rng: RisOpponent(sims if sims is not None else 800, rng)
    if name == "pika":
        return build_opponent_factory(spec, device, None, pika_config=_WORKER_STATE.get("pika_config"))
    if name == "model":
        sims = sims if sims is not None else _WORKER_STATE["config"].mcts_simulations
        cache = _WORKER_STATE["opponents"]
        key = (path, sims)
        if key not in cache:
            other = azero.PolicyValueNet()
            expected_source = _WORKER_STATE.get("opponent_sources", {}).get(spec)
            check_model_source(path, expected_source)
            if not azero.load_model_if_exists(other, path, device):
                raise FileNotFoundError(path)
            check_model_source(path, expected_source)
            other.to(device)
            other.eval()
            cache[key] = other
        other = cache[key]
        # 对手侧机会节点设置独立于己方（默认 inherit 时两者相同，用于搜索 A/B）。
        opp_nodes, opp_k = _WORKER_STATE.get(
            "opponent_chance",
            (_WORKER_STATE["config"].chance_nodes, _WORKER_STATE["config"].chance_k),
        )
        config = build_search_config(sims, opp_nodes, opp_k, _WORKER_STATE["config"].leaf_batch_size)
        return lambda rng: NeuralOpponent(other, config, device, rng)
    raise ValueError(f"unknown opponent spec: {spec}")


def _eval_game_task(task: tuple) -> dict:
    """子进程任务：跑一局并返回记录。"""
    spec, pair, pair_seed, neural_side, max_plies = task
    net = _WORKER_STATE["net"]
    device = _WORKER_STATE["device"]
    config = _WORKER_STATE["config"]
    neural_factory = lambda rng: NeuralOpponent(net, config, device, rng)
    opponent_factory = _build_opponent_factory_worker(spec, device)
    record = play_one_game(neural_factory, opponent_factory, neural_side, pair_seed, max_plies)
    record["type"] = "game"
    record["opponent"] = spec
    record["pair"] = pair
    return record


def _ordered_eval_results(executor, tasks):
    """Notice any failed game promptly, but retain deterministic output/pair order."""
    pending = {executor.submit(_eval_game_task, task): index for index, task in enumerate(tasks)}
    ready = {}
    next_index = 0
    while pending:
        done, _ = futures.wait(pending, timeout=0.1, return_when=futures.FIRST_COMPLETED)
        for future in done:
            ready[pending.pop(future)] = future.result()
        while next_index in ready:
            yield ready.pop(next_index)
            next_index += 1


def evaluate(
    net,
    device: torch.device,
    opponents: list[str],
    games: int,
    sims: int,
    seed: int,
    max_plies: int,
    out_path: str,
    quiet: bool = False,
    workers: int = 1,
    chance_nodes: bool = True,
    chance_k: int = 0,
    opponent_chance_nodes: bool | None = None,
    opponent_chance_k: int | None = None,
    model_file: dict | None = None,
    pika_config: dict | None = None,
    bootstrap: int = 20000,
    expected_opponent_sources: dict | None = None,
    context: dict | None = None,
    leaf_batch_size: int = azero.DEFAULT_EVAL_LEAF_BATCH_SIZE,
):
    if games <= 0:
        raise ValueError("games must be positive")
    if games % 2 != 0:
        games += 1
    if workers < 1 or sims < 1 or max_plies < 1 or bootstrap < 1 or not opponents:
        raise ValueError("positive budgets/workers and at least one opponent required")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    # Python 调用与 CLI 一样默认继承己方设置；显式覆盖用于搜索 A/B。
    opponent_chance_nodes = chance_nodes if opponent_chance_nodes is None else opponent_chance_nodes
    opponent_chance_k = chance_k if opponent_chance_k is None else opponent_chance_k
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(out_path):
        raise FileExistsError(f"评估输出已存在，请使用新路径: {out_path}")
    config = build_search_config(sims, chance_nodes, chance_k, leaf_batch_size)
    if context and context.get("pool"):
        validate_pool_search(context["pool"], sims, leaf_batch_size)
    pair_count = games // 2
    opponent_sources = {}
    for spec in opponents:
        name, path, budget = parse_opponent_spec(spec)
        if name not in ("random", "ris", "model", "pika") or (budget is not None and budget < 1):
            raise ValueError(f"invalid opponent spec: {spec}")
        if name == "model":
            opponent_sources[spec] = rr.file_record(path)
    if expected_opponent_sources is not None and opponent_sources != expected_opponent_sources:
        raise RuntimeError("评估对手来源与冻结清单不一致")
    if any(parse_opponent_spec(spec)[0] == "pika" for spec in opponents):
        from pika_engine import prepare_config
        pika_config = prepare_config(pika_config)
    records_dir = os.path.dirname(os.path.abspath(out_path))
    weight_sha256 = rr.model_hash(net)
    manifest = rr.create_run(
        records_dir, "evaluation",
        dict(model_games=games, mcts_simulations=sims, seed=seed, max_plies=max_plies,
             leaf_batch_size=leaf_batch_size, leaf_dedup=config.leaf_dedup,
             model_opponent_default_simulations=sims,
             workers=workers, chance_nodes=chance_nodes, chance_k=chance_k,
             opponent_chance_nodes=opponent_chance_nodes, opponent_chance_k=opponent_chance_k,
             opponents=opponents, pika=pika_config, bootstrap=bootstrap,
             statistics_version=2, output=os.path.abspath(out_path)),
        azero.network_spec(net) if net is not None and hasattr(net, "conv") else None,
        dict(ruleset=jq.RULESET_VERSION, data=azero.DATA_VERSION, search=azero.SEARCH_VERSION),
        dict(model_file=model_file, model_sha256=weight_sha256, opponent_files=opponent_sources,
             context=context), device,
    )
    provenance = dict(**rr.source_ref(manifest), model_sha256=weight_sha256)

    executor = None
    if workers > 1:
        ctx = mp.get_context("spawn")
        model_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        executor = futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_start_eval_worker,
            initargs=(model_state, sims, str(device), chance_nodes, chance_k,
                      opponent_chance_nodes, opponent_chance_k, opponent_sources, pika_config, leaf_batch_size),
        )
    neural_factory = lambda rng: NeuralOpponent(net, config, device, rng)

    summaries = []
    try:
        with open(out_path, "x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "run",
                        "eval_protocol_version": azero.EVAL_PROTOCOL_VERSION,
                        "model_games": games,
                        "mcts_simulations": sims,
                        "leaf_batch_size": leaf_batch_size,
                        "leaf_dedup": config.leaf_dedup,
                        "seed": seed,
                        "workers": workers,
                        "chance_nodes": chance_nodes,
                        "chance_k": chance_k,
                        "opponent_chance_nodes": opponent_chance_nodes,
                        "opponent_chance_k": opponent_chance_k,
                        "pika": pika_config,
                        "statistics_version": 2,
                        "bootstrap_resamples": bootstrap,
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "provenance": provenance,
                        "manifest": manifest,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for spec in opponents:
                if executor is None:
                    opponent_factory = build_opponent_factory(
                        spec, device, net,
                        chance_nodes=opponent_chance_nodes, chance_k=opponent_chance_k,
                        expected_source=opponent_sources.get(spec),
                        pika_config=pika_config,
                        leaf_batch_size=leaf_batch_size, default_model_sims=sims,
                    )
                    records = (
                        _with_meta(
                            play_one_game(neural_factory, opponent_factory, neural_side, seed + pair, max_plies),
                            spec,
                            pair,
                        )
                        for pair in range(pair_count)
                        for neural_side in (jq.RED, jq.BLACK)
                    )
                else:
                    tasks = [
                        (spec, pair, seed + pair, neural_side, max_plies)
                        for pair in range(pair_count)
                        for neural_side in (jq.RED, jq.BLACK)
                    ]
                    records = _ordered_eval_results(executor, tasks)

                wins = draws = losses = unfinished = 0
                pairs_used = pairs_excluded = 0
                pair_results = {}
                pair_scores = []
                move_seconds = {"neural": 0.0, "opponent": 0.0}
                move_counts = {"neural": 0, "opponent": 0}
                for index, record in enumerate(records):
                    record["provenance"] = provenance
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    for role in move_seconds:
                        move_seconds[role] += record.get("move_seconds", {}).get(role, 0)
                        move_counts[role] += record.get("moves_played", {}).get(role, 0)
                    result = tally_game(record)
                    if result == "unfinished":
                        unfinished += 1
                    pair = pair_results.setdefault(record["pair"], [])
                    pair.append(result)
                    if len(pair) == 2:
                        if "unfinished" in pair:
                            pairs_excluded += 1
                        else:
                            pairs_used += 1
                            wins += pair.count("win")
                            draws += pair.count("draw")
                            losses += pair.count("loss")
                            pair_scores.append((pair.count("win") + .5 * pair.count("draw")) / 2)
                    if not quiet and (index + 1) % 20 == 0:
                        print(
                            f"[{spec}] {index + 1}/{games} wins={wins} draws={draws} "
                            f"losses={losses} unfinished={unfinished}",
                            flush=True,
                        )
                if unfinished and not quiet:
                    print(f"[{spec}] 未终局 {unfinished} 局，剔除 {pairs_excluded} 个配对", flush=True)
                point, low, high = wilson_score_interval(wins, draws, losses)
                paired = pair_summary(pair_scores, bootstrap, seed)
                summary = {
                    "type": "summary",
                    "opponent": spec,
                    "games": wins + draws + losses,
                    "unfinished": unfinished,
                    "pairs": pair_count,
                    "pairs_used": pairs_used,
                    "pairs_excluded": pairs_excluded,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "score": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "wilson_ci95_approx": [low, high],
                    **paired,
                    "move_seconds": move_seconds,
                    "moves_played": move_counts,
                    "mcts_simulations": sims,
                    "leaf_batch_size": leaf_batch_size,
                    "leaf_dedup": config.leaf_dedup,
                    "seed": seed,
                    "provenance": provenance,
                }
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
                handle.flush()
                summaries.append(summary)
                if not quiet:
                    score_text = f"{point:.3f}" if pairs_used else "无有效配对"
                    print(
                        f"[{spec}] score={score_text} paired CI95={paired['bootstrap_ci95']} "
                        f"W/D/L={wins}/{draws}/{losses}",
                        flush=True,
                    )
                    if paired["bootstrap_degenerate"]:
                        print("配对样本不足或零方差，bootstrap 区间不能用于确认精度或优势。", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        if executor is not None:
            executor.shutdown()
    except BaseException as error:
        if executor is not None:
            evaluation_runtime.abort_pool(executor, error)
        raise
    rr.write_record(rr.run_dir(records_dir, manifest["run_id"]) / "complete.json",
                    dict(**provenance, status="complete", output=rr.file_record(out_path)))
    return summaries


def _with_meta(record: dict, spec: str, pair: int) -> dict:
    record["type"] = "game"
    record["opponent"] = spec
    record["pair"] = pair
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="揭棋模型配对评估协议")
    parser.add_argument("--model", default=os.path.join("data", "models", "latest.pt"))
    opponents_group = parser.add_mutually_exclusive_group()
    opponents_group.add_argument("--opponents", default="ris:800", help="逗号分隔: ris:N / random / model:path[:sims] / pika:budget")
    opponents_group.add_argument("--pool", help="冻结历史模型池 JSON（校验规则版本和权重 SHA256）")
    parser.add_argument("--pika-exe", help="Pikafish jieqi_old executable (required with pika)")
    parser.add_argument("--pika-limit", choices=("nodes", "movetime"), default="nodes",
                        help="pika:budget 的单位：节点或每步毫秒")
    parser.add_argument("--pika-threads", type=int, default=1)
    parser.add_argument("--pika-hash-mb", type=int, default=32)
    parser.add_argument("--pika-timeout", type=float, default=30, help="每次协议响应最长秒数；超时中止评估")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=azero.DEFAULT_EVAL_SIMULATIONS, help="评估时模型每步 MCTS 模拟数，默认1600")
    parser.add_argument("--leaf-batch-size", type=int, default=azero.DEFAULT_EVAL_LEAF_BATCH_SIZE,
                        help="评估双方模型的叶批大小，默认16")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=8, help="并行评估进程数（1 为串行）")
    parser.add_argument("--out", default="")
    parser.add_argument("--quick", action="store_true", help="快评预设: 50 局 @400 sims vs RIS-400")
    parser.add_argument("--chance-nodes", action=argparse.BooleanOptionalAction, default=True,
                        help="己方（模型侧）机会节点默认开启；--no-chance-nodes 改为惰性建树，对手默认继承同一设置"
                             "（可用 --opponent-chance-nodes 覆盖，用于搜索 A/B）")
    parser.add_argument("--chance-k", type=int, default=0, help="己方机会分支初始预建上限 k：0=全枚举（推荐），遗漏结果按需补建")
    parser.add_argument("--opponent-chance-nodes", choices=("inherit", "on", "off"), default="inherit",
                        help="对手机会节点设置：inherit=跟随己方（默认，公平比较）；on/off 显式覆盖")
    parser.add_argument("--opponent-chance-k", type=int, default=None,
                        help="对手机会分支初始预建上限 k；缺省=跟随 --chance-k")
    args = parser.parse_args(argv)
    if args.quick and args.pool:
        parser.error("--quick 只用于 RIS 回归，不能与 --pool 合用")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.sims < 1:
        parser.error("--sims 必须不小于 1")
    if args.games < 1:
        parser.error("--games 必须不小于 1")
    if args.leaf_batch_size < 1:
        parser.error("--leaf-batch-size 必须不小于 1")
    if args.chance_k < 0:
        parser.error("--chance-k 必须不小于 0")
    if args.opponent_chance_k is not None and args.opponent_chance_k < 0:
        parser.error("--opponent-chance-k 必须不小于 0")
    if min(args.workers, args.max_plies, args.bootstrap, args.pika_threads, args.pika_hash_mb) < 1:
        parser.error("workers/max-plies/bootstrap/pika-threads/pika-hash-mb must be positive")
    if not math.isfinite(args.pika_timeout) or args.pika_timeout <= 0:
        parser.error("--pika-timeout must be finite and positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    pool = load_opponent_pool(args.pool) if args.pool else None
    if pool:
        validate_pool_search(pool, args.sims, args.leaf_batch_size)
    torch.set_num_threads(1)
    if args.quick:
        args.games = 50
        args.sims = 400
        args.opponents = "ris:400"
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    net = azero.PolicyValueNet()
    model_file = rr.file_record(args.model) if os.path.exists(args.model) else None
    if not azero.load_model_if_exists(net, args.model, device):
        print(
            f"错误: 未找到模型 {args.model}，拒绝以随机初始化权重评估（fail-fast）。",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if rr.file_record(args.model) != model_file:
        raise RuntimeError("加载期间评估模型改变，拒绝记录不准确的来源")
    net.to(device)
    net.eval()
    out_path = args.out or os.path.join("data", "eval", f"eval_{time.strftime('%Y%m%d_%H%M%S')}_{rr.new_id()[:8]}.jsonl")
    opponents = pool["opponents"] if pool else [spec.strip() for spec in args.opponents.split(",") if spec.strip()]
    opponent_chance_nodes, opponent_chance_k = resolve_opponent_chance(
        args.chance_nodes, args.chance_k, args.opponent_chance_nodes, args.opponent_chance_k
    )
    evaluate(net, device, opponents, args.games, args.sims, args.seed, args.max_plies, out_path,
             workers=args.workers, chance_nodes=args.chance_nodes, chance_k=args.chance_k,
             opponent_chance_nodes=opponent_chance_nodes, opponent_chance_k=opponent_chance_k,
             model_file=model_file, bootstrap=args.bootstrap,
             leaf_batch_size=args.leaf_batch_size,
             expected_opponent_sources=pool["sources"] if pool else None,
             context=dict(pool=pool) if pool else None,
             pika_config=dict(executable=args.pika_exe, limit=args.pika_limit,
                              threads=args.pika_threads, hash_mb=args.pika_hash_mb,
                              timeout=args.pika_timeout))
    print(f"结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
