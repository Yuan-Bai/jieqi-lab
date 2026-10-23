from __future__ import annotations

import training_runtime as tr
tr.bootstrap(__name__)  # Spawn workers ignore Ctrl+C before importing torch.

import argparse
import concurrent.futures as futures
import copy
import gc
import gzip
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import random
import signal
import struct
import sys
import tempfile
import time
import traceback
import warnings
import zlib
from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

import ismcts as im
import jieqi as jq
import numpy as np
import run_records as rr
import training_diagnostics as td


BOARD_H = jq.BOARD_H
BOARD_W = jq.BOARD_W
BOARD_SIZE = jq.BOARD_SIZE
HISTORY_PLIES = 8
# 保留旧 71 通道前缀；新增锚点、目标码与仍可能重复的公共局面。
REPETITION_HISTORY_SIZE = jq.NO_CAPTURE_LIMIT_TIANTIAN_PLIES + 1
REPETITION_HISTORY_COMPLETE_CHANNEL = 75
REPETITION_HISTORY_OFFSET = 76
STATE_CHANNELS = REPETITION_HISTORY_OFFSET + 2 * REPETITION_HISTORY_SIZE

DATA_VERSION = 3
SELFPLAY_EXT = ".pkl.gz"
MODEL_LATEST_NAME = "latest.pt"
MODEL_BEST_NAME = "best.pt"
STATE_FILE_NAME = "train_state.json"
TRAINING_CHECKPOINT_NAME = "training_checkpoint.pt"
TRAINING_CHECKPOINT_VERSION = 1
SEARCH_VERSION = 3  # v2 合法根噪声；v3 有界策略裁剪及统一终局优先级。
EVAL_PROTOCOL_VERSION = 2  # Same-shuffle pairs, independent RNGs, incomplete pairs excluded.
DEFAULT_EVAL_SIMULATIONS = 1600
DEFAULT_EVAL_LEAF_BATCH_SIZE = 16


def build_legal_move_table() -> list[tuple[int, int]]:
    """
    构造基于“位移向量”的动作映射表，长度应为 2550。
    注意：这是“几何可达”的动作集合，并未检查棋子规则。
    """
    # 合法动作向量列表
    legal_move_vector: list[tuple[int, int]] = []
    for i in range(1, 9):
        legal_move_vector.append((i, 0))
        legal_move_vector.append((-i, 0))
    for i in range(1, 10):
        legal_move_vector.append((0, i))
        legal_move_vector.append((0, -i))
    legal_move_vector.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    legal_move_vector.extend([(2, 2), (2, -2), (-2, 2), (-2, -2)])
    legal_move_vector.extend(
        [(1, 2), (2, 1), (2, -1), (1, -2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
    )

    table: list[tuple[int, int]] = []
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            for move_vector in legal_move_vector:
                to_point = (r + move_vector[1], c + move_vector[0])
                if jq.in_bounds(to_point[0], to_point[1]):
                    table.append((jq.index(r, c), jq.index(to_point[0], to_point[1])))
    if len(table) != 2550:
        raise ValueError(f"unexpected legal move table size: {len(table)}")
    return table


LEGAL_MOVE_TABLE = build_legal_move_table()
MOVE_TO_ACTION = [-1] * (BOARD_SIZE * BOARD_SIZE)
for action_id, move in enumerate(LEGAL_MOVE_TABLE):
    MOVE_TO_ACTION[move[0] * BOARD_SIZE + move[1]] = action_id
MOVE_SIZE = len(LEGAL_MOVE_TABLE)

def move_to_index(move: tuple[int, int]) -> int:
    """动作索引编码：from/to -> action_id（2550 动作空间）。"""
    idx = move[0] * BOARD_SIZE + move[1]
    action_id = MOVE_TO_ACTION[idx]
    if action_id < 0:
        raise ValueError(f"move not in legal move table: {move}")
    return action_id


def index_to_move(index: int) -> tuple[int, int]:
    """动作索引解码为 (from, to)。"""
    return LEGAL_MOVE_TABLE[index]


def mirror_index(index: int) -> int:
    """左右镜像棋盘坐标，用于数据增强。"""
    r = index // BOARD_W
    c = index % BOARD_W
    return r * BOARD_W + (BOARD_W - 1 - c)


def mirror_move(move: tuple[int, int]) -> tuple[int, int]:
    """左右镜像后的走子。"""
    return mirror_index(move[0]), mirror_index(move[1])




def outcome_value_for_player(outcome: int | None, player: int) -> float:
    """将胜负结果转成当前视角的价值。"""
    if outcome is None or outcome == jq.RESULT_DRAW:
        return 0.0
    if outcome == jq.RESULT_RED_WIN:
        return 1.0 if player == jq.RED else -1.0
    if outcome == jq.RESULT_BLACK_WIN:
        return 1.0 if player == jq.BLACK else -1.0
    raise ValueError("invalid outcome")


def outcome_label(outcome: int | None) -> str:
    """人类可读的胜负标签。"""
    if outcome == jq.RESULT_RED_WIN:
        return "红胜"
    if outcome == jq.RESULT_BLACK_WIN:
        return "黑胜"
    if outcome == jq.RESULT_DRAW:
        return "和棋"
    return "未结束"


def temperature_for_move(
    move_index: int,
    t_initial: float,
    t_final: float,
    decay_moves: int,
) -> float:
    """线性温度退火：从 t_initial 逐步衰减到 t_final。"""
    if decay_moves <= 0:
        return t_final
    ratio = min(1.0, move_index / decay_moves)
    return t_initial + (t_final - t_initial) * ratio


def apply_dirichlet_noise(
    priors: torch.Tensor,
    legal_moves: list[tuple[int, int]],
    alpha: float,
    epsilon: float,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """在根节点对先验加入 Dirichlet 噪声，增加探索。

    先在合法动作上归一化（零质量用均匀回退），保证 epsilon 是实际混合比例。
    传入 ``rng`` 时由其派生种子采样，保证同一 seed 的自博弈可复现（ADR-0005）；
    不传时退回 torch 全局 RNG（仅用于测试/评估等非复现场景）。
    """
    if epsilon <= 0.0 or alpha <= 0.0 or not legal_moves:
        return priors
    if rng is not None:
        np_rng = np.random.default_rng(rng.getrandbits(63))
        noise = torch.from_numpy(
            np_rng.dirichlet([alpha] * len(legal_moves)).astype(np.float32)
        )
    else:
        noise = torch.distributions.Dirichlet(
            torch.full((len(legal_moves),), alpha, dtype=torch.float32)
        ).sample()
    indices = [move_to_index(move) for move in legal_moves]
    legal_priors = priors[indices]
    total = float(legal_priors.sum())
    if total > 0.0:
        legal_priors = legal_priors / total
    else:
        legal_priors = torch.full_like(legal_priors, 1.0 / len(legal_moves))
    mixed = priors.clone()
    mixed[indices] = (1.0 - epsilon) * legal_priors + epsilon * noise.to(legal_priors)
    return mixed


def _piece_plane_index(is_dark: bool, side: int, p_type: int) -> int:
    base = 0 if not is_dark else 14
    return base + side * 7 + (p_type - 1)


def _pool_plane_index(side: int, offset: int) -> int:
    return offset + side * len(jq.POOL_TYPES)


def _repetition_count(info_state: im.InformationState) -> int:
    """当前公共局面的重复计数（position_counts 缺失时按首次出现记 1）。"""
    if not info_state.position_counts:
        return 1
    return int(info_state.position_counts.get((info_state.side, tuple(info_state.board)), 1))


@lru_cache(maxsize=4096)
def _public_material_signature(board: tuple[int, ...]) -> tuple[int, int]:
    return sum(piece != 0 for piece in board), sum(jq.is_dark(piece) for piece in board if piece)


def _repetition_entries(info: im.InformationState) -> tuple | None:
    """完整的、未来仍可能出现的重复键，按首次出现顺序保存。

    吃子和暗子翻开不可逆：更多棋子或更多暗子的旧局面不可能再出现。
    默认 80 ply 无吃子即终局，所以剩余最多 81 个键（含当前局面）。
    自定义规则/手造历史超过容量时显式拒绝，禁止把截断历史标成完整。
    """
    if info.position_counts is None or not info.repetition_history_complete:
        return None
    current = tuple(info.board)
    occupied, dark = _public_material_signature(current)
    entries = []
    for (side, board), count in info.position_counts.items():
        old_occupied, old_dark = _public_material_signature(board)
        if old_occupied > occupied or old_dark > dark:
            continue
        entries.append((int(side), bytes(board), int(count)))
    if (info.side, current) not in info.position_counts:
        entries.append((info.side, bytes(current), 1))
    if len(entries) > REPETITION_HISTORY_SIZE:
        raise ValueError(f"repetition history exceeds {REPETITION_HISTORY_SIZE} positions: {len(entries)}")
    return tuple(entries)


_REPETITION_RECORD = struct.Struct(f"<BI{BOARD_SIZE}s")


def _pack_repetition_entries(entries: tuple) -> bytes:
    # 每格一个公共棋子码，计数原值保留；避免重放缓冲保存大量 Python 整数元组。
    return zlib.compress(b"\x01" + b"".join(_REPETITION_RECORD.pack(side, count, board)
                                           for side, board, count in entries), level=1)


def pack_repetition_history(info: im.InformationState) -> bytes | None:
    entries = _repetition_entries(info)
    return None if entries is None else _pack_repetition_entries(entries)


def unpack_repetition_history(blob: bytes) -> dict:
    data = zlib.decompress(blob)
    size = _REPETITION_RECORD.size
    if (not data or data[0] != 1 or (len(data) - 1) % size
            or (len(data) - 1) // size > REPETITION_HISTORY_SIZE):
        raise ValueError("invalid repetition history payload")
    counts = {}
    for side, count, board in _REPETITION_RECORD.iter_unpack(data[1:]):
        if side not in (jq.RED, jq.BLACK) or count < 1 or any(piece > 31 or (piece and not piece & 7) for piece in board):
            raise ValueError("invalid repetition history record")
        key = (side, tuple(board))
        if key in counts:
            raise ValueError("duplicate repetition history key")
        counts[key] = count
    return counts


def _transform_repetition_history(blob: bytes | None, board_transform, swap_sides: bool = False) -> bytes | None:
    if blob is None:
        return None
    entries = tuple((1 - side if swap_sides else side, bytes(board_transform(list(board))), count)
                    for (side, board), count in unpack_repetition_history(blob).items())
    return _pack_repetition_entries(entries)


# Match the original uint8 / float64 division followed by float32 assignment.
_REPETITION_PIECE_VALUES = (np.arange(256, dtype=np.uint8) / 31.0).astype(np.float32)
_REPETITION_PIECE_VALUES.setflags(write=False)
_REPETITION_ENTRIES_UNSET = object()


def _encode_state_into(
    info_state: im.InformationState,
    history: list[tuple[int, int]],
    planes: np.ndarray,
    *,
    repetition_entries=_REPETITION_ENTRIES_UNSET,
) -> None:
    """填充已清零的 float32 平面；批次构建时无需逐样本分配再堆叠。"""
    for idx, piece in enumerate(info_state.board):
        if not piece:
            continue
        planes[
            _piece_plane_index(jq.is_dark(piece), jq.piece_color(piece), jq.piece_type(piece)),
            jq.ROWS[idx],
            jq.COLS[idx],
        ] = 1.0

    constants = [1.0 if info_state.side == jq.RED else 0.0]
    for side in (jq.BLACK, jq.RED):
        for p_type in jq.POOL_TYPES:
            denom = jq.POOL_INIT[p_type]
            constants.append(info_state.pool_known[side][p_type] / denom if denom else 0.0)
    constants.extend(info_state.pool_total_known[side] / jq.POOL_TOTAL for side in (jq.BLACK, jq.RED))
    constants.extend(info_state.unknown_removals[side] / jq.POOL_TOTAL for side in (jq.BLACK, jq.RED))
    constants.append(info_state.no_capture_plies / jq.NO_CAPTURE_LIMIT_TIANTIAN_PLIES)
    denom = jq.LONG_CHECK_ROUNDS + 1
    constants.extend(info_state.consecutive_checks[side] / denom for side in (jq.RED, jq.BLACK))
    planes[28:48] = np.asarray(constants, dtype=np.float32)[:, None, None]

    offset = 48
    recent = history[-HISTORY_PLIES:]
    for i, move in enumerate(recent):
        from_idx, to_idx = move
        planes[offset + i * 2, jq.ROWS[from_idx], jq.COLS[from_idx]] = 1.0
        planes[offset + i * 2 + 1, jq.ROWS[to_idx], jq.COLS[to_idx]] = 1.0
    offset += 2 * HISTORY_PLIES

    # 64：当前公共局面重复计数（长将/长捉/重复判定的公开状态）。
    planes[offset, :, :] = _repetition_count(info_state) / jq.REPETITION_LIMIT
    # 65~66：连续追击计数（红/黑）。
    planes[offset + 1, :, :] = info_state.chase_streak[jq.RED] / jq.LONG_CHASE_ROUNDS
    planes[offset + 2, :, :] = info_state.chase_streak[jq.BLACK] / jq.LONG_CHASE_ROUNDS
    # 67~68：追击目标所在格（红/黑，单点 1.0）。
    # 69~70：追击发起子所在格（红/黑，单点 1.0）。
    for i, side in enumerate((jq.RED, jq.BLACK)):
        target = info_state.chase_target[side]
        if target >= 0:
            planes[offset + 3 + i, jq.ROWS[target], jq.COLS[target]] = 1.0
        anchor = info_state.chase_anchor[side]
        if anchor >= 0:
            planes[offset + 5 + i, jq.ROWS[anchor], jq.COLS[anchor]] = 1.0

    # 71~72：上轮将军发起子；73~74：长捉目标公共棋子码。
    for i, side in enumerate((jq.RED, jq.BLACK)):
        anchor = info_state.check_anchor[side]
        if anchor >= 0:
            planes[71 + i, jq.ROWS[anchor], jq.COLS[anchor]] = 1.0
        planes[73 + i, :, :] = info_state.chase_target_code[side] / 31.0
    entries = (_repetition_entries(info_state) if repetition_entries is _REPETITION_ENTRIES_UNSET
               else repetition_entries)
    if entries is not None:
        planes[REPETITION_HISTORY_COMPLETE_CHANNEL, :, :] = 1.0
        if entries:
            end = REPETITION_HISTORY_OFFSET + 2 * len(entries)
            boards = np.frombuffer(b"".join(board for _, board, _ in entries), dtype=np.uint8)
            boards = boards.reshape(len(entries), BOARD_H, BOARD_W)
            planes[REPETITION_HISTORY_OFFSET:end:2] = _REPETITION_PIECE_VALUES[boards]
            # (计数,轮次) 单射；0 留给空槽。超过判和阈值的计数仍保留。
            counts = [(2 * count + side) / (2 * jq.REPETITION_LIMIT + 1) for side, _, count in entries]
            planes[REPETITION_HISTORY_OFFSET + 1:end:2] = np.asarray(counts, dtype=np.float32)[:, None, None]


def encode_state_numpy(
    info_state: im.InformationState,
    history: list[tuple[int, int]],
) -> np.ndarray:
    """将信息集编码为 (STATE_CHANNELS, 10, 9) 的 float32 numpy 平面。"""
    planes = np.zeros((STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.float32)
    _encode_state_into(info_state, history, planes)
    return planes


def _encode_batch_numpy(infos, histories, *, repetition_entries=None) -> np.ndarray:
    pairs = list(zip(infos, histories))
    if not pairs:
        raise ValueError("need at least one state to encode")
    if repetition_entries is not None and len(repetition_entries) != len(pairs):
        raise ValueError("repetition entries size mismatch")
    batch = np.zeros((len(pairs), STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.float32)
    for i, (row, (info, history)) in enumerate(zip(batch, pairs)):
        entries = _REPETITION_ENTRIES_UNSET if repetition_entries is None else repetition_entries[i]
        _encode_state_into(info, history, row, repetition_entries=entries)
    return batch


def encode_state(
    info_state: im.InformationState,
    history: list[tuple[int, int]],
    observer: int,
) -> torch.Tensor:
    """
    将信息集编码为固定 238 通道张量：
    - 0~27：明子/暗子 × 阵营 × 类型
    - 28：轮到红方（全图 1 or 0）
    - 29~40：暗盒中各类棋子剩余比例（双方）
    - 41~42：暗盒剩余总数比例（双方）
    - 43~44：未知被吃数比例（双方）
    - 45：无吃子步数比例
    - 46~47：连续将军次数比例（双方）
    - 48~63：最近 8 步的 from/to 位置
    - 64：当前公共局面重复计数 / 3
    - 65~66：连续追击同一目标计数 / 6（红/黑）
    - 67~68：追击目标所在格（红/黑，单点 1.0）
    - 69~70：追击发起子所在格（红/黑，单点 1.0）
    - 71~72：将军发起子所在格（红/黑，单点 1.0）
    - 73~74：追击目标公共棋子码 / 31（红/黑）
    - 75：重复历史完整标记（旧 v2 样本为 0）
    - 76~237：81 个仍可能重复的公共局面；每个两通道：
      公共棋盘码 / 31、(2 * 重复计数 + 轮次) / 7；未用槽填零
    """
    _ = observer
    return torch.from_numpy(encode_state_numpy(info_state, history))


@dataclass(slots=True)
class TrainingState:
    iteration: int = 0
    best_score: float = -1.0
    total_games: int = 0
    ruleset_version: int = jq.RULESET_VERSION


@dataclass(slots=True)
class RawSample:
    """自博弈生成的“未标注 value”的中间样本。"""
    board: list[int]
    side: int
    pool_known: list[list[int]]
    pool_total_known: list[int]
    unknown_removals: list[int]
    no_capture_plies: int
    consecutive_checks: list[int]
    # 公开规则状态（长将/长捉/重复判定的输入；见 encode_state）。
    check_anchor: list[int]
    chase_streak: list[int]
    chase_target: list[int]
    chase_target_code: list[int]
    chase_anchor: list[int]
    rep_count: int
    history: list[tuple[int, int]]
    policy_idx: list[int]
    policy_prob: list[float]
    player: int
    # 策略目标权重：1.0=full/strong 搜索（策略目标参与训练），
    # 0.0=PCR fast 搜索（策略目标噪声大，仅 value 参与训练）。
    policy_weight: float = 1.0
    repetition_history: bytes | None = None


@dataclass(slots=True)
class CompactSample:
    """紧凑样本：可无损还原为训练张量。"""
    board: list[int]
    side: int
    pool_known: list[list[int]]
    pool_total_known: list[int]
    unknown_removals: list[int]
    no_capture_plies: int
    consecutive_checks: list[int]
    # 公开规则状态（长将/长捉/重复判定的输入；见 encode_state）。
    check_anchor: list[int]
    chase_streak: list[int]
    chase_target: list[int]
    chase_target_code: list[int]
    chase_anchor: list[int]
    rep_count: int
    history: list[tuple[int, int]]
    policy_idx: list[int]
    policy_prob: list[float]
    value: float
    policy_weight: float = 1.0
    repetition_history: bytes | None = None


@dataclass(slots=True)
class SelfPlayResult:
    samples: list[CompactSample]
    outcome: int | None
    move_count: int
    leaf_evals_total: int = 0
    leaf_evals_unique: int = 0


def setup_logger(log_dir: str) -> logging.Logger:
    """初始化日志系统，输出到控制台与文件。"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("azero")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_training_state(path: str, logger: logging.Logger | None = None) -> TrainingState:
    """读取训练状态（断点续训用）。

    文件缺失时返回默认状态；文件存在但损坏（截断/格式错误）时显式告警，
    并返回默认状态（不静默吞掉解析失败）。
    """
    if not os.path.exists(path):
        return TrainingState()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TrainingState(
            iteration=int(data.get("iteration", 0)),
            best_score=float(data.get("best_score", -1.0)),
            total_games=int(data.get("total_games", 0)),
            ruleset_version=int(data.get("ruleset_version", -1)),
        )
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        (logger or logging.getLogger("azero")).warning(
            "训练状态读取失败，按 iteration=0 的全新状态处理: %s (%s)", path, exc
        )
        return TrainingState()


def save_training_state(path: str, state: TrainingState) -> None:
    """保存训练状态（写入临时文件后原子替换，避免截断产物）。"""
    data = {
        "iteration": state.iteration,
        "best_score": state.best_score,
        "total_games": state.total_games,
        "ruleset_version": state.ruleset_version,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_state_dict_padded(
    net: PolicyValueNet,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    """加载权重；旧 64/71 通道编码模型的检查点自动 pad-load。

    仅当首层 ``conv.weight`` 的输出通道/卷积核一致、输入通道更少时，把新增输入
    通道权重置零（其余权重逐元素保持原值）；其他形状不兼容直接抛 ``RuntimeError``，
    避免静默加载错误权重。返回是否发生了通道补零。
    """
    target = net.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    padded = False
    old_channels = new_channels = 0
    for key, value in state_dict.items():
        if key not in target:
            raise RuntimeError(f"模型权重包含未知参数: {key}")
        if tuple(value.shape) == tuple(target[key].shape):
            adapted[key] = value
            continue
        if (
            key == "conv.weight"
            and value.dim() == 4
            and value.shape[0] == target[key].shape[0]
            and value.shape[2:] == target[key].shape[2:]
            and value.shape[1] < target[key].shape[1]
        ):
            old_channels, new_channels = value.shape[1], target[key].shape[1]
            new_value = torch.zeros_like(target[key])
            new_value[:, :old_channels] = value
            adapted[key] = new_value
            padded = True
            continue
        raise RuntimeError(
            f"模型参数形状不兼容: {key} {tuple(value.shape)} -> {tuple(target[key].shape)}"
        )
    net.load_state_dict(adapted)
    if padded:
        logging.getLogger("azero").info(
            "旧编码模型 pad-load：首层 conv 输入通道 %d -> %d，新增通道权重置零",
            old_channels,
            new_channels,
        )
    return padded


def load_model_if_exists(net: PolicyValueNet, model_path: str, device: torch.device) -> bool:
    """加载已存在模型权重（旧编码检查点自动 pad-load），返回是否成功加载。"""
    if not os.path.exists(model_path):
        return False
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    load_state_dict_padded(net, state_dict)
    return True


def save_model(net: PolicyValueNet, model_path: str) -> None:
    """原子保存纯权重，保持 GUI/评估及旧工具的文件格式。"""
    atomic_torch_save(net.state_dict(), model_path)


def atomic_torch_save(payload: dict, path: str) -> None:
    """写入独立临时文件后提交；写入/替换失败保留之前的检查点。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sparse_policy_from_dense(policy: list[float] | torch.Tensor) -> tuple[list[int], list[float]]:
    """将 dense policy 压缩成 (index, prob) 列表，减少存储。"""
    if isinstance(policy, torch.Tensor):
        data = policy.detach().cpu().tolist()
    else:
        data = policy
    indices: list[int] = []
    probs: list[float] = []
    for i, p in enumerate(data):
        if p > 0.0:
            indices.append(i)
            probs.append(float(p))
    return indices, probs


def dense_policy_from_sparse(indices: list[int], probs: list[float]) -> torch.Tensor:
    """将稀疏 policy 还原为 dense。"""
    policy = torch.zeros(MOVE_SIZE, dtype=torch.float32)
    for idx, prob in zip(indices, probs):
        policy[idx] = float(prob)
    return policy


def raw_sample_from_info(
    info_state: im.InformationState,
    history: list[tuple[int, int]],
    policy: list[float],
    policy_weight: float = 1.0,
) -> RawSample:
    """从信息集 + policy 生成 RawSample（尚未标注 value）。"""
    policy_idx, policy_prob = sparse_policy_from_dense(policy)
    return RawSample(
        board=info_state.board.copy(),
        side=info_state.side,
        pool_known=[info_state.pool_known[0].copy(), info_state.pool_known[1].copy()],
        pool_total_known=[info_state.pool_total_known[0], info_state.pool_total_known[1]],
        unknown_removals=[info_state.unknown_removals[0], info_state.unknown_removals[1]],
        no_capture_plies=info_state.no_capture_plies,
        consecutive_checks=info_state.consecutive_checks.copy(),
        check_anchor=info_state.check_anchor.copy(),
        chase_streak=info_state.chase_streak.copy(),
        chase_target=info_state.chase_target.copy(),
        chase_target_code=info_state.chase_target_code.copy(),
        chase_anchor=info_state.chase_anchor.copy(),
        rep_count=_repetition_count(info_state),
        history=history[-HISTORY_PLIES:].copy(),
        policy_idx=policy_idx,
        policy_prob=policy_prob,
        player=info_state.side,
        policy_weight=policy_weight,
        repetition_history=pack_repetition_history(info_state),
    )


def finalize_samples(raw_samples: list[RawSample], outcome: int | None) -> list[CompactSample]:
    """将 RawSample 标注 value，生成 CompactSample。"""
    if outcome not in (jq.RESULT_RED_WIN, jq.RESULT_BLACK_WIN, jq.RESULT_DRAW):
        raise ValueError("self-play labels require a terminal outcome")
    samples: list[CompactSample] = []
    for raw in raw_samples:
        value = outcome_value_for_player(outcome, raw.player)
        samples.append(
            CompactSample(
                board=raw.board,
                side=raw.side,
                pool_known=raw.pool_known,
                pool_total_known=raw.pool_total_known,
                unknown_removals=raw.unknown_removals,
                no_capture_plies=raw.no_capture_plies,
                consecutive_checks=raw.consecutive_checks,
                check_anchor=raw.check_anchor,
                chase_streak=raw.chase_streak,
                chase_target=raw.chase_target,
                chase_target_code=raw.chase_target_code,
                chase_anchor=raw.chase_anchor,
                rep_count=raw.rep_count,
                history=raw.history,
                policy_idx=raw.policy_idx,
                policy_prob=raw.policy_prob,
                value=value,
                policy_weight=raw.policy_weight,
                repetition_history=getattr(raw, "repetition_history", None),
            )
        )
    return samples


def _compact_information_state(sample: CompactSample) -> im.InformationState:
    blob = getattr(sample, "repetition_history", None)
    return im.InformationState(
        board=sample.board.copy(),
        side=sample.side,
        pool_known=[sample.pool_known[0].copy(), sample.pool_known[1].copy()],
        pool_total_known=[sample.pool_total_known[0], sample.pool_total_known[1]],
        unknown_removals=[sample.unknown_removals[0], sample.unknown_removals[1]],
        no_capture_plies=sample.no_capture_plies,
        consecutive_checks=sample.consecutive_checks.copy(),
        check_anchor=sample.check_anchor.copy(),
        chase_streak=sample.chase_streak.copy(),
        chase_target=sample.chase_target.copy(),
        chase_target_code=sample.chase_target_code.copy(),
        chase_anchor=sample.chase_anchor.copy(),
        position_counts=(unpack_repetition_history(blob) if blob is not None else
                         {(sample.side, tuple(sample.board)): int(sample.rep_count)}),
        repetition_history_complete=blob is not None,
    )


def compact_to_tensors(sample: CompactSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """还原训练张量；旧样本缺失的重复历史明确编码为未知。"""
    info_state = _compact_information_state(sample)
    state_tensor = encode_state(info_state, sample.history, info_state.side)
    policy_tensor = dense_policy_from_sparse(sample.policy_idx, sample.policy_prob)
    value_tensor = torch.tensor(sample.value, dtype=torch.float32)
    return state_tensor, policy_tensor, value_tensor


def compact_batch_to_tensors(batch: list[CompactSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """一次分配连续 NCHW 批次；顺序、FP32 舍入和重复策略索引的覆盖顺序不变。"""
    if not batch:
        raise ValueError("need at least one sample to encode")
    states = np.zeros((len(batch), STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.float32)
    policies = np.zeros((len(batch), MOVE_SIZE), dtype=np.float32)
    values = np.empty(len(batch), dtype=np.float32)
    for i, sample in enumerate(batch):
        _encode_state_into(_compact_information_state(sample), sample.history, states[i])
        for idx, prob in zip(sample.policy_idx, sample.policy_prob):
            policies[i, idx] = float(prob)
        values[i] = sample.value
    return torch.from_numpy(states), torch.from_numpy(policies), torch.from_numpy(values)


def mirror_board(board: list[int]) -> list[int]:
    """左右镜像棋盘上的棋子。"""
    mirrored = [0] * BOARD_SIZE
    for idx, piece in enumerate(board):
        mirrored[mirror_index(idx)] = piece
    return mirrored


def mirror_history(history: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """左右镜像历史走子。"""
    return [mirror_move(move) for move in history]


def mirror_policy(indices: list[int], probs: list[float]) -> tuple[list[int], list[float]]:
    """左右镜像 policy 的动作索引。"""
    mirrored_indices: list[int] = []
    for idx in indices:
        from_idx, to_idx = index_to_move(idx)
        mirrored_move = (mirror_index(from_idx), mirror_index(to_idx))
        mirrored_indices.append(move_to_index(mirrored_move))
    return mirrored_indices, probs.copy()


def mirror_sample(sample: CompactSample) -> CompactSample:
    """生成镜像增强样本（信息集随棋盘镜像）。"""
    mirrored_indices, mirrored_probs = mirror_policy(sample.policy_idx, sample.policy_prob)
    return CompactSample(
        board=mirror_board(sample.board),
        side=sample.side,
        pool_known=[sample.pool_known[0].copy(), sample.pool_known[1].copy()],
        pool_total_known=[sample.pool_total_known[0], sample.pool_total_known[1]],
        unknown_removals=[sample.unknown_removals[0], sample.unknown_removals[1]],
        no_capture_plies=sample.no_capture_plies,
        consecutive_checks=sample.consecutive_checks.copy(),
        check_anchor=[mirror_index(idx) if idx >= 0 else -1 for idx in sample.check_anchor],
        chase_streak=sample.chase_streak.copy(),
        chase_target=[mirror_index(idx) if idx >= 0 else -1 for idx in sample.chase_target],
        chase_target_code=sample.chase_target_code.copy(),
        chase_anchor=[mirror_index(idx) if idx >= 0 else -1 for idx in sample.chase_anchor],
        rep_count=int(sample.rep_count),
        history=mirror_history(sample.history),
        policy_idx=mirrored_indices,
        policy_prob=mirrored_probs,
        value=sample.value,
        policy_weight=float(getattr(sample, "policy_weight", 1.0)),
        repetition_history=_transform_repetition_history(getattr(sample, "repetition_history", None), mirror_board),
    )


def mirror_samples(samples: list[CompactSample]) -> list[CompactSample]:
    """批量镜像样本。"""
    return [mirror_sample(sample) for sample in samples]


def augment_mirror(samples: list[CompactSample]) -> list[CompactSample]:
    """批量生成镜像增强样本。"""
    return [mirror_sample(sample) for sample in samples]


def flip_vertical_index(index: int) -> int:
    """楚河汉界翻面的棋盘坐标（行 r -> BOARD_H-1-r，列不变）。"""
    r = index // BOARD_W
    c = index % BOARD_W
    return (BOARD_H - 1 - r) * BOARD_W + c


def flip_vertical_move(move: tuple[int, int]) -> tuple[int, int]:
    """翻面后的走子。"""
    return flip_vertical_index(move[0]), flip_vertical_index(move[1])


def flip_vertical_square(index: int) -> int:
    """翻面后的格子索引；-1（无）保持 -1。"""
    return flip_vertical_index(index) if index >= 0 else -1


def swap_piece_color(piece: int) -> int:
    """翻转棋子颜色位（红黑互换）；空格 0 保持不变。"""
    return (piece ^ (1 << jq.COLOR_SHIFT)) if piece else 0


def flip_vertical_board(board: list[int]) -> list[int]:
    """楚河汉界翻面棋盘并互换红黑。"""
    flipped = [0] * BOARD_SIZE
    for idx, piece in enumerate(board):
        if piece:
            flipped[flip_vertical_index(idx)] = swap_piece_color(piece)
    return flipped


def flip_vertical_history(history: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """翻面历史走子。"""
    return [flip_vertical_move(move) for move in history]


def flip_vertical_policy(indices: list[int], probs: list[float]) -> tuple[list[int], list[float]]:
    """翻面 policy 的动作索引。"""
    flipped_indices: list[int] = []
    for idx in indices:
        from_idx, to_idx = index_to_move(idx)
        flipped_indices.append(
            move_to_index((flip_vertical_index(from_idx), flip_vertical_index(to_idx)))
        )
    return flipped_indices, probs.copy()


def flip_vertical_sample(sample: CompactSample) -> CompactSample:
    """生成楚河汉界翻面增强样本：棋盘行翻面 + 红黑互换（暗池/规则状态随之互换）。

    该变换等于“左右镜像 ∘ (180°旋转+红黑互换)”，是棋盘+颜色对称群（Klein 四元
    群）中被遗漏的第三个元素：初始阵型在变换下自对应，棋子走法、九宫/河界、照面、
    观察结构随位置与颜色对应；规则同构性必须另检查规则转移，不能由字段搬运推出。

    注意：长为 2 的颜色数组按 [黑, 红]（jq.BLACK/jq.RED）存放，互换即交换槽位；
    ``chase_target_code`` 存的是含颜色位的完整棋子码（见 jieqi._update_chase），
    故除交换槽位外还需翻转颜色位，否则翻转样本会立即断捉。

    新长捉最高价值目标并列时清空追击；受害子按公开移动追踪，双闪将不按
    格子顺序指定锚点（RULESET_VERSION=5，ADR-0020）。字段及规则转移的
    确定性/随机回归通过，不将有限测试宣称为所有可达局面的形式证明。
    """
    flipped_indices, flipped_probs = flip_vertical_policy(sample.policy_idx, sample.policy_prob)
    return CompactSample(
        board=flip_vertical_board(sample.board),
        side=1 - sample.side,
        pool_known=[sample.pool_known[1].copy(), sample.pool_known[0].copy()],
        pool_total_known=[sample.pool_total_known[1], sample.pool_total_known[0]],
        unknown_removals=[sample.unknown_removals[1], sample.unknown_removals[0]],
        no_capture_plies=sample.no_capture_plies,
        consecutive_checks=[sample.consecutive_checks[1], sample.consecutive_checks[0]],
        check_anchor=[
            flip_vertical_square(sample.check_anchor[1]),
            flip_vertical_square(sample.check_anchor[0]),
        ],
        chase_streak=[sample.chase_streak[1], sample.chase_streak[0]],
        chase_target=[
            flip_vertical_square(sample.chase_target[1]),
            flip_vertical_square(sample.chase_target[0]),
        ],
        chase_target_code=[
            swap_piece_color(sample.chase_target_code[1]),
            swap_piece_color(sample.chase_target_code[0]),
        ],
        chase_anchor=[
            flip_vertical_square(sample.chase_anchor[1]),
            flip_vertical_square(sample.chase_anchor[0]),
        ],
        rep_count=int(sample.rep_count),
        history=flip_vertical_history(sample.history),
        policy_idx=flipped_indices,
        policy_prob=flipped_probs,
        value=sample.value,
        policy_weight=float(getattr(sample, "policy_weight", 1.0)),
        repetition_history=_transform_repetition_history(getattr(sample, "repetition_history", None), flip_vertical_board, True),
    )


def flip_vertical_samples(samples: list[CompactSample]) -> list[CompactSample]:
    """批量生成翻面增强样本。"""
    return [flip_vertical_sample(sample) for sample in samples]


def augment_flip_vertical(samples: list[CompactSample]) -> list[CompactSample]:
    """批量生成翻面增强样本（与 augment_mirror 对称的入口）。"""
    return [flip_vertical_sample(sample) for sample in samples]


def augment_samples(
    samples: list[CompactSample],
    mirror: bool = True,
    flip: bool = False,
) -> list[CompactSample]:
    """按开关组合镜像/翻面增强，便于训练、续训加载与合并共用同一套语义。

    两个开关都启用时返回全部 4 个变换（恰好覆盖棋盘+颜色对称群的 Klein 四元群）：
    原样、左右镜像、翻面+换色、二者复合（180°旋转+换色）；只开启一个时返回 2 倍。
    """
    augmented = list(samples)
    if mirror:
        augmented += mirror_samples(samples)
    if flip:
        augmented += flip_vertical_samples(augmented)
    return augmented



class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class PolicyValueNet(nn.Module):
    """AlphaZero 风格的策略-价值网络。"""

    def __init__(self, channels: int = 64, blocks: int = 15) -> None:
        super().__init__()
        self.conv = nn.Conv2d(STATE_CHANNELS, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.residual = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * BOARD_H * BOARD_W, MOVE_SIZE)

        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * BOARD_H * BOARD_W, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.residual(out)

        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.reshape(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value.squeeze(1)


_WORKER_NET: PolicyValueNet | None = None
_WORKER_CONFIG: "AZConfig" | None = None
_WORKER_DEVICE: torch.device | None = None


def _init_selfplay_worker(model_state: dict, config: "AZConfig", device_str: str) -> None:
    """多进程自博弈初始化：每个子进程各自加载一次模型。"""
    global _WORKER_NET, _WORKER_CONFIG, _WORKER_DEVICE
    # Ctrl+C 由主进程统一处理（终止子进程 + 硬退出）；worker 忽略 SIGINT，
    # 避免控制台信号在 worker 里抛 KeyboardInterrupt 打出无意义的堆栈。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    device = torch.device(device_str)
    net = PolicyValueNet().to(device)
    load_state_dict_padded(net, model_state)
    net.eval()
    _WORKER_NET = net
    _WORKER_CONFIG = config
    _WORKER_DEVICE = device


def _selfplay_worker(args: tuple[int, bool]) -> "SelfPlayResult":
    """子进程执行一局自博弈。"""
    if _WORKER_NET is None or _WORKER_CONFIG is None or _WORKER_DEVICE is None:
        raise RuntimeError("self-play worker not initialized")
    seed, strong = args
    rng = random.Random(seed)
    return self_play_game(_WORKER_NET, _WORKER_CONFIG, _WORKER_DEVICE, rng, strong=strong)


def _sample_weighted_without_replacement(candidates: list[tuple], k: int, rng) -> list[tuple]:
    """按权重无放回采样至多 k 个 (item, weight) 候选。"""
    pool = list(candidates)
    chosen = []
    for _ in range(min(k, len(pool))):
        total = sum(weight for _, weight in pool)
        roll = rng.random() * total
        acc = 0.0
        picked = len(pool) - 1
        for i, (item, weight) in enumerate(pool):
            acc += weight
            if roll <= acc:
                picked = i
                break
        chosen.append(pool.pop(picked))
    return chosen


def sample_reveal_outcomes(counts: list[int], k: int, rng) -> list[tuple[int, float]]:
    """选取预建的揭示分支，保留完整信念分布中的概率质量。

    参数：
        counts: 该方暗池按棋子类型索引的剩余计数。
        k: 初始预建上限；``k <= 0`` 表示全枚举（推荐），
           否则按权重无放回选取至多 k 类；后续导航按需补建遗漏结果。
    返回：
        [(揭示类型, 原始边际概率)]；预建子集的概率和可能小于 1。
        这不是环境采样分布；实际结果始终从模拟世界的完整暗池抽取。
    """
    candidates = [(t, counts[t]) for t in jq.POOL_TYPES if counts[t] > 0]
    if not candidates:
        return []
    picked = (
        _sample_weighted_without_replacement(candidates, k, rng)
        if k > 0 and len(candidates) > k
        else candidates
    )
    total = sum(weight for _, weight in candidates)
    if total <= 0:
        return []
    return [(t, weight / total) for t, weight in picked]


def sample_playout_cap(config: "AZConfig", rng: random.Random) -> tuple[int, float]:
    """PCR 档位采样：返回 (本步搜索次数, 策略目标权重)。

    - full 档：``mcts_simulations``，策略目标权重 1.0；
    - fast 档：``pcr_fast_sims``，权重 0.0（策略目标不训练，value 目标照常）。

    探索（Dirichlet/温度）与回合统计与档位无关，避免 PCR 降低探索。
    """
    if rng.random() < config.pcr_full_prob:
        return config.mcts_simulations, 1.0
    return config.pcr_fast_sims, 0.0


def chance_spec_for_move(state: jq.GameState, info: im.InformationState, move) -> dict | None:
    """若该着法会触发暗子揭示，给出揭示类型的计数分布；否则返回 None。"""
    from_idx, to_idx = move
    moved = state.board[from_idx]
    if not moved:
        return None
    moved_counts = None
    if jq.is_dark(moved):
        moved_counts = info.pool_known[jq.piece_color(moved)]
        if sum(moved_counts) <= 0:
            return None
    capture_counts = None
    target = state.board[to_idx]
    if target and jq.is_dark(target):
        captured_side = jq.piece_color(target)
        capture_counts = info.pool_known[captured_side]
        if sum(capture_counts) <= 0:
            return None
    if moved_counts is None and capture_counts is None:
        return None
    return {"moved_counts": moved_counts, "capture_counts": capture_counts}


def leaf_input_signature(
    info: im.InformationState,
    history: list[tuple[int, int]],
    *,
    repetition_entries=_REPETITION_ENTRIES_UNSET,
) -> tuple:
    """叶评估去重用的完整输入签名：覆盖编码使用的全部信息与 history。

    两个叶节点只有签名完全一致才允许复用同一次网络前向（policy/value）。
    因此这里宁可保守多带上字段（如 ``check_anchor``），也不漏掉任何可能
    进入 ``encode_state_numpy`` 的信息。
    """
    rep_count = 1
    if info.position_counts:
        rep_count = int(info.position_counts.get((info.side, tuple(info.board)), 1))
    return (
        bytes(info.board),
        info.side,
        tuple(info.pool_known[jq.BLACK]),
        tuple(info.pool_known[jq.RED]),
        tuple(info.pool_total_known),
        tuple(info.unknown_removals),
        info.no_capture_plies,
        tuple(info.consecutive_checks),
        tuple(info.check_anchor),
        tuple(info.chase_streak),
        tuple(info.chase_target),
        tuple(info.chase_target_code),
        tuple(info.chase_anchor),
        rep_count,
        (_repetition_entries(info) if repetition_entries is _REPETITION_ENTRIES_UNSET else repetition_entries),
        tuple(history),
    )


class PUCTNode:
    """PUCT 搜索节点。

    普通节点的 children 以着法为键；机会节点（is_chance=True）的 children
    以揭示结果序号为键，其子节点上记录 reveal_moved/reveal_captured。
    双揭示（走暗子吃暗子）拆成两层机会节点：第一层翻移动子，第二层翻被吃子。
    这些是环境边；走后通过 decision_nodes 连接行动者信息集，共享决策统计。
    """

    __slots__ = (
        "prior",
        "player_to_move",
        "visits",
        "value_sum",
        "children",
        "public_outcomes",
        "decision_nodes",
        "move",
        "is_chance",
        "reveal_moved",
        "reveal_captured",
        "chance_counts",
    )

    def __init__(self, prior: float, player_to_move: int, move=None, is_chance: bool = False) -> None:
        self.prior = prior
        self.player_to_move = player_to_move
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict = {}
        # 非枚举路径也必须按公开翻子结果隔离决策子树（Issue 0007）。
        # 仅 moved_reveal 是公开信息；捕获暗子的私有类型不进入此键。
        self.public_outcomes: dict[int, PUCTNode] = {}
        # 环境结果与决策统计分离；不同环境边可引用同一行动者信息集节点。
        self.decision_nodes: dict[tuple, PUCTNode] = {}
        self.move = move
        self.is_chance = is_chance
        self.reveal_moved = 0
        self.reveal_captured = 0
        # 仅入口机会节点保存完整边际计数快照，供后续补建分支设置先验。
        # 导航抽样使用当前模拟世界的池，绝不按这个子集或边际先验抽样。
        self.chance_counts: tuple | None = None

    def public_outcome(self, reveal_moved: int) -> "PUCTNode":
        """按实际采样到的公开揭示惰性建树，动作节点仍汇总全部结果价值。"""
        outcome = self.public_outcomes.get(reveal_moved)
        if outcome is None:
            outcome = PUCTNode(1.0, self.player_to_move)
            outcome.reveal_moved = reveal_moved
            self.public_outcomes[reveal_moved] = outcome
        return outcome

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def select_child(self, c_puct: float) -> tuple:
        total_visits = sum(child.visits for child in self.children.values())
        best_score = -1e9
        best_move = None
        best_child = None
        sqrt_total = math.sqrt(total_visits + 1.0)
        for move, child in self.children.items():
            u = c_puct * child.prior * sqrt_total / (1 + child.visits)
            # child.q_value() is stored from the child's player perspective.
            # Convert to parent perspective for selection.
            score = -child.q_value() + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def sample_outcome(self, rng) -> tuple:
        """在已建分支中按存储先验归一化采样（仅辅助工具）。

        注意（Issue 0005）：搜索路径已改为**世界条件采样**（`NeuralMCTS._navigate_chance`
        直接从当前模拟世界的暗池计数采样），本方法保留作为边际分布采样工具与
        概率测试使用，不再参与搜索导航。预建子集不完整时，本方法只是
        条件于该子集的分布，不能作为完整环境采样器。
        """
        if isinstance(self.chance_counts, _DeferredChanceCounts):
            materialize_chance_outcomes(self)
        items = list(self.children.items())
        if not items:
            raise RuntimeError("chance node has no outcomes")
        total = sum(child.prior for _, child in items)
        if total <= 0.0:
            return items[0]
        roll = rng.random() * total
        acc = 0.0
        for key, child in items:
            acc += child.prior
            if roll <= acc:
                return key, child
        return items[-1]

    def expand(
        self,
        legal_moves: list[tuple[int, int]],
        priors: torch.Tensor | np.ndarray | list[float],
        chance_specs: dict | None = None,
        chance_k: int = 0,
        rng=None,
    ) -> None:
        if not legal_moves:
            return
        total = 0.0
        for move in legal_moves:
            total += float(priors[move_to_index(move)])
        if total <= 0.0:
            move_priors = {move: 1.0 / len(legal_moves) for move in legal_moves}
        else:
            move_priors = {
                move: float(priors[move_to_index(move)]) / total for move in legal_moves
            }
        for move in legal_moves:
            spec = chance_specs.get(move) if chance_specs else None
            moved_outcomes = (
                (sample_reveal_outcomes(spec["moved_counts"], chance_k, rng)
                 if spec["moved_counts"] is not None else [(0, 1.0)])
                if spec is not None and rng is not None
                else []
            )
            if moved_outcomes:
                child = PUCTNode(move_priors[move], 1 - self.player_to_move, move=move, is_chance=True)
                child.chance_counts = tuple(
                    tuple(counts) if counts is not None else None
                    for counts in (spec["moved_counts"], spec["capture_counts"])
                )
                capture_outcomes = (
                    sample_reveal_outcomes(spec["capture_counts"], chance_k, rng)
                    if spec["capture_counts"] is not None
                    else []
                )
                for index, (reveal_moved, moved_prob) in enumerate(moved_outcomes):
                    if capture_outcomes:
                        # 双揭示拆两层：先翻移动子，再按被吃方池采样；每层最多 6 类。
                        layer2 = PUCTNode(moved_prob, 1 - self.player_to_move, is_chance=True)
                        layer2.reveal_moved = reveal_moved
                        for sub_index, (reveal_captured, captured_prob) in enumerate(capture_outcomes):
                            outcome_node = PUCTNode(captured_prob, 1 - self.player_to_move)
                            outcome_node.reveal_moved = reveal_moved
                            outcome_node.reveal_captured = reveal_captured
                            layer2.children[sub_index] = outcome_node
                        child.children[index] = layer2
                    else:
                        outcome_node = PUCTNode(moved_prob, 1 - self.player_to_move)
                        outcome_node.reveal_moved = reveal_moved
                        child.children[index] = outcome_node
                self.children[move] = child
            else:
                self.children[move] = PUCTNode(move_priors[move], 1 - self.player_to_move, move=move)


def _node_methods_sha256() -> str:
    import marshal
    import types

    def normalized(code):
        return code.replace(co_filename="azero.py", co_consts=tuple(
            normalized(value) if isinstance(value, types.CodeType) else value for value in code.co_consts))

    methods = (PUCTNode._init_python, PUCTNode._expand_python)
    return hashlib.sha256(marshal.dumps(tuple(
        (normalized(method.__code__), method.__defaults__, method.__kwdefaults__) for method in methods))).hexdigest()


def _load_search_backend():
    from pathlib import Path

    requested = os.environ.get("JIEQI_SEARCH_BACKEND", "python").lower()
    if requested not in ("auto", "python", "cython"):
        raise ValueError("JIEQI_SEARCH_BACKEND must be auto, python, or cython")
    PUCTNode.__init__, PUCTNode.expand = PUCTNode._init_python, PUCTNode._expand_python
    if requested == "python":
        return "python", None, None
    try:
        import _jieqi_search as native
        if getattr(native, "API_VERSION", None) != 1 or getattr(native, "SEARCH_VERSION", None) != SEARCH_VERSION:
            raise ImportError("incompatible native search API; rebuild with tools/build_search.py")
        source = Path(__file__).with_name("native") / "jieqi_search.pyx"
        if getattr(native, "SOURCE_SHA256", None) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ImportError("stale native search source; rebuild with tools/build_search.py")
        if getattr(native, "METHODS_SHA256", None) != _node_methods_sha256():
            raise ImportError("stale native node methods; rebuild with tools/build_search.py")
        binary = Path(native.__file__).resolve()
        with binary.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        record = dict(path=str(binary), sha256=digest, bytes=binary.stat().st_size,
                      api_version=native.API_VERSION, search_version=native.SEARCH_VERSION,
                      source_sha256=native.SOURCE_SHA256, methods_sha256=native.METHODS_SHA256,
                      methods=["expand"])
        module = sys.modules[__name__]
        native.initialize(module)
    except (ImportError, OSError) as error:
        if requested == "cython":
            raise ImportError("Cython search backend unavailable") from error
        return "python", str(error), None
    PUCTNode._native_search_module = module
    PUCTNode.expand = native.expand
    return "cython", None, record


PUCTNode._init_python, PUCTNode._expand_python = PUCTNode.__init__, PUCTNode.expand
SEARCH_BACKEND, SEARCH_BACKEND_ERROR, SEARCH_BACKEND_NATIVE = _load_search_backend()


class _DeferredChanceCounts(tuple):
    """Full k=0 support whose outcome objects have not all been allocated."""

    __slots__ = ()


def _full_chance_outcomes_uncached(counts):
    moved, captured = counts
    return (sample_reveal_outcomes(moved, 0, None) if moved is not None else [(0, 1.0)],
            sample_reveal_outcomes(captured, 0, None) if captured is not None else [])


_FULL_CHANCE_SAMPLE = sample_reveal_outcomes
_FULL_CHANCE_POOL_TYPES = jq.POOL_TYPES


@lru_cache(maxsize=4096)
def _cached_full_chance_outcomes(counts):
    return tuple(tuple(row) for row in _full_chance_outcomes_uncached(counts))


def _full_chance_outcomes(counts):
    # Equal int/float keys can have different arithmetic; retain the original
    # path for numeric duck types, mutable inputs, and overridden enumerators.
    if (sample_reveal_outcomes is _FULL_CHANCE_SAMPLE and jq.POOL_TYPES is _FULL_CHANCE_POOL_TYPES
            and jq._immutable_integer_pools is not None
            and type(counts) in (tuple, _DeferredChanceCounts)
            and jq._immutable_integer_pools(counts)):
        return _cached_full_chance_outcomes(counts)
    return _full_chance_outcomes_uncached(counts)


def _deferred_moved_child(entry, index, reveal, prior, captures):
    child = entry.children.get(index)
    if child is None:
        child = PUCTNode(prior, entry.player_to_move, is_chance=bool(captures))
        child.reveal_moved = reveal
        if captures:
            child.chance_counts = _DeferredChanceCounts((None, entry.chance_counts[1]))
        entry.children[index] = child
    return child


def _deferred_captured_child(layer, index, reveal, prior):
    child = layer.children.get(index)
    if child is None:
        child = PUCTNode(prior, layer.player_to_move)
        child.reveal_moved = layer.reveal_moved
        child.reveal_captured = reveal
        layer.children[index] = child
    return child


def materialize_chance_outcomes(node: PUCTNode) -> None:
    """Restore eager k=0 layout without RNG or changes to existing node statistics.

    Search never needs this for a valid world. Auxiliary marginal sampling and
    exact graph audits use it to include the entire logical support in order.
    """
    if not isinstance(node.chance_counts, _DeferredChanceCounts):
        return
    moved, captures = _full_chance_outcomes(node.chance_counts)
    if node.move is not None:
        for index, (reveal, prior) in enumerate(moved):
            child = _deferred_moved_child(node, index, reveal, prior, captures)
            if captures:
                materialize_chance_outcomes(child)
        node.chance_counts = tuple(node.chance_counts)
    else:
        for index, (reveal, prior) in enumerate(captures):
            _deferred_captured_child(node, index, reveal, prior)
        node.chance_counts = None
    ordered = sorted(node.children.items())
    node.children.clear()
    node.children.update(ordered)


def pruned_policy_from_root(root: "PUCTNode", c_puct: float) -> torch.Tensor:
    """Bounded KataGo-inspired target pruning using final, noised PUCT priors.

    Protect all most-visited moves. Remove at most floor(sqrt(2 * P * N))
    visits from another move while its PUCT stays below the protected threshold.
    A move reduced to one visit is then dropped entirely.
    No forced playouts are implemented, so this remains a heuristic, not the full
    KataGo procedure or an exact removal of Dirichlet noise from targets.
    """
    policy = torch.zeros(MOVE_SIZE)
    if not root.children:
        return policy
    total_visits = sum(child.visits for child in root.children.values())
    if total_visits <= 0:
        return policy
    sqrt_total = math.sqrt(total_visits)
    most_visits = max(child.visits for child in root.children.values())
    protected = [child for child in root.children.values() if child.visits == most_visits]
    threshold = max(-child.q_value() + c_puct * child.prior * sqrt_total / (1 + child.visits)
                    for child in protected)
    pruned: dict[tuple[int, int], int] = {}
    for move, child in root.children.items():
        count = child.visits
        if count and count < most_visits:
            limit = math.floor(math.sqrt(2 * max(0.0, child.prior) * total_visits))
            removed = 0
            while removed < limit and count > 0:
                next_score = -child.q_value() + c_puct * child.prior * sqrt_total / count
                if next_score >= threshold:
                    break
                count -= 1
                removed += 1
            if count == 1 and removed:
                count = 0
        pruned[move] = count
    total_pruned = sum(pruned.values())
    if total_pruned <= 0:
        for move, child in root.children.items():
            policy[move_to_index(move)] = child.visits / total_visits
        return policy
    for move, count in pruned.items():
        policy[move_to_index(move)] = count / total_pruned
    return policy


@dataclass(slots=True)
class AZConfig:
    # 自博弈 / 训练超参
    self_play_games: int = 256
    self_play_workers: int = 8
    self_play_device: str = "cuda"
    self_play_log_interval: int = 0
    iterations: int = 200
    run_iterations: int = 0  # 本次最多新增几轮；0=训练到 iterations，不改变 cosine 总计划。
    mcts_simulations: int = 800
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    c_puct: float = 1.5
    leaf_batch_size: int = 64
    virtual_loss: float = 3.0
    # 批内叶评估去重：同一批 pending 中完整输入签名一致时只做一次网络前向。
    # 合并完整相同输入；CUDA 批次形状改变不保证逐位一致，保持既有默认配置。
    # chance_k>0 时，机会预建的采样 RNG 流也会因去重而改变。
    leaf_dedup: bool = True
    # Stage 2A：暗子揭示机会节点（把“走暗子揭出什么”作为显式随机分支）
    chance_nodes: bool = True
    chance_k: int = 0  # 0=完整支持（推荐）；>0 只限制预建数量，遗漏结果按需补建
    lazy_chance_allocation: bool | None = None  # None=on/k=0 时默认开启；实例中解析为 bool。
    temperature_initial: float = 1.0
    temperature_final: float = 1e-3
    temperature_decay_moves: int = 20
    batch_size: int = 1024
    train_steps: int = 1000
    learning_rate: float = 5e-4
    lr_min: float = 1e-5
    lr_schedule: str = "cosine"
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    replay_buffer_size: int = 4_000_000
    diagnostic_games: int = 0  # >0 从下一轮新对局一次性留出整局；已有固定集自动恢复。
    diagnostic_batch_size: int = 128

    # 日常主指标用冻结历史池；未指定池时仅跑 ISMCTS 回归。
    eval_games: int = 32
    eval_interval: int = 4
    eval_mcts_simulations: int = DEFAULT_EVAL_SIMULATIONS
    eval_leaf_batch_size: int = DEFAULT_EVAL_LEAF_BATCH_SIZE
    ris_iterations: int = 800
    eval_workers: int = 8
    eval_device: str = "cuda"
    eval_pool: str = ""
    eval_seed: int = 2026102202  # 固定日常种子；正式晋升另用留出种子。
    eval_max_plies: int = 600
    legacy_ris_best: bool = False  # 仅显式兼容旧 RIS 最高分导出，不代表综合冠军。

    # 数据与日志
    data_dir: str = "./data"
    model_dir: str = "./data/models"
    log_dir: str = "./data/logs"
    resume: bool = True
    reset_optimizer: bool = False  # 显式改变 LR/调度时，保留模型/计数但重建优化状态。
    allow_buffer_growth: bool = False  # 显式扩容：精确保留旧 FIFO，由后续新数据填满。
    augment_mirror: bool = True
    # 楚河汉界翻面增强（行翻面 + 红黑互换 + 暗池互换）。与 augment_mirror 同开时
    # 每个样本产出 4 份（原样/镜像/翻面/镜像∘翻面），即完整棋盘+颜色对称群。
    augment_flip: bool = True
    # 续训加载上限：0=按 replay_buffer_size 自适应（默认，从最新文件回填到容量）；
    # >0=只加载最近 N 个文件（旧语义，可能与连续训练的缓冲状态不一致）
    max_load_files: int = 0
    strong_self_play_interval: int = 8
    strong_mcts_simulations: int = 1600
    strong_temperature_final: float = 1e-6
    strong_temperature_decay_moves: int = 8
    strong_dirichlet_epsilon: float = 0.0

    # PCR（playout cap randomization，KataGo 2019 §3.1）：普通自博弈每步按概率在
    # fast / full 两档搜索间随机（把算力从“每步搜深”移到“同预算下多产对局”）。
    # fast 档的策略目标噪声大，只参与 value 训练（policy_weight=0.0）；
    # strong 局（每 strong_self_play_interval 局）保持整局 strong_mcts_simulations 不变。
    pcr_enabled: bool = False
    pcr_fast_sims: int = 100
    pcr_full_prob: float = 0.25

    def __post_init__(self) -> None:
        if self.lazy_chance_allocation is None:
            self.lazy_chance_allocation = self.chance_nodes and self.chance_k == 0


class NeuralMCTS:
    """神经网络驱动的 MCTS（PUCT + 根视角单次确定化）。

    每次模拟按根玩家信息确定化一次，整条路径复用该世界；依据其中的捕获结果
    重建合成行动者观测，不读取真实对手私有槽。环境机会结果连接共享信息集
    决策节点，叶值仍从行动方视角输出。根 Q 使用根条件样本，根前私有历史仍
    是计数抽象，不代表策略条件后验或均衡搜索（见 ADR-0012、Issue 0003）。
    """

    def __init__(self, net: PolicyValueNet, config: AZConfig, device: torch.device, rng=None) -> None:
        if config.lazy_chance_allocation and (not config.chance_nodes or config.chance_k != 0):
            raise ValueError("lazy chance allocation requires chance_nodes=True and chance_k=0")
        self.net = net
        self.config = config
        self.device = device
        self.rng = rng or random.Random()
        # 叶评估计数（去重收益观测）：total=收集的叶数，unique=实际前向的输入数。
        self.leaf_evals_total = 0
        self.leaf_evals_unique = 0
        # 机会链计数（Issue 0005）：applied=按当前模拟世界池成功导航，
        # nav_miss=无有效池/节点结构不一致等异常兜底；遗漏缓存分支直接补建。
        self.chance_applied = 0
        self.chance_nav_miss = 0
        self.chance_outcomes_added = 0

    def _evaluate_batch(
        self,
        infos: list[im.InformationState],
        histories: list[list[tuple[int, int]]],
        *,
        repetition_entries=None,
    ) -> tuple[np.ndarray, list[float]]:
        """原形状完整前向；policy 用只读 CPU 数组，value 保持 Python float 回传。"""
        self.net.eval()
        with torch.no_grad():
            batch = _encode_batch_numpy(infos, histories, repetition_entries=repetition_entries)
            x = torch.from_numpy(batch).to(self.device)
            policy_logits, values = self.net(x)
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
            policies.setflags(write=False)
        # NumPy float32 参与 Python 累加会改变舍入，不能把 value 也改成 NumPy 标量。
        return policies, values.tolist()

    def _evaluate(self, info_state: im.InformationState, history: list[tuple[int, int]]) -> tuple[torch.Tensor, float]:
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(encode_state_numpy(info_state, history)).unsqueeze(0).to(self.device)
            policy_logits, value = self.net(x)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu()
        return policy, float(value.item())

    @staticmethod
    def _backup(path: list["PUCTNode"], value: float, virtual_loss: float) -> None:
        """回传真实值并扣除选择阶段添加的虚拟损失。

        沿路径逐层翻转视角；机会节点链（移动子揭示→被吃子揭示）与其揭示
        结果节点属于同一行动方（同视角），因此这些层之间不翻转。
        """
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            if i == 0:
                node.visits += 1
                node.value_sum += value
            else:
                node.value_sum += value - virtual_loss
                parent = path[i - 1]
                if parent.player_to_move != node.player_to_move:
                    value = -value

    def _chance_specs(
        self,
        state: jq.GameState,
        info: im.InformationState,
        legal_moves: list[tuple[int, int]],
    ) -> dict | None:
        """收集本节点全部暗子揭示机会着法的分布；未启用时返回 None。"""
        if not self.config.chance_nodes:
            return None
        specs = {}
        for move in legal_moves:
            spec = chance_spec_for_move(state, info, move)
            if spec is not None:
                specs[move] = spec
        return specs or None

    def _expand_node(self, node, legal_moves, priors, specs):
        if not self.config.lazy_chance_allocation or not specs:
            node.expand(legal_moves, priors, specs, self.config.chance_k, self.rng)
            return
        # Reuse the active Python/native normalizer and allocate identical action edges.
        node.expand(legal_moves, priors)
        for move in legal_moves:
            spec = specs.get(move)
            if spec is None:
                continue
            counts = _DeferredChanceCounts(tuple(
                tuple(row) if row is not None else None
                for row in (spec["moved_counts"], spec["capture_counts"])))
            moved, _ = _full_chance_outcomes(counts)
            if moved:
                child = node.children[move]
                child.is_chance = True
                child.chance_counts = counts

    def _apply_chance_move(
        self,
        state: jq.GameState,
        move: tuple[int, int],
        reveal_moved: int,
        reveal_captured: int,
    ):
        """应用带强制揭示的着法。

        修复（Issue 0005）后 reveal 类型由 `_navigate_chance` 从本模拟世界池采样，
        世界必然含该类型，因此可用性检查只是防御性兜底；退回自然抽样的路径仅用于
        无效状态/直接传入不可用揭示等异常情形，不用于候选遗漏。
        """
        available = True
        if reveal_moved and state.pool[state.side][reveal_moved] <= 0:
            available = False
        if available and reveal_captured:
            target = state.board[move[1]]
            if target and jq.is_dark(target):
                captured_side = jq.piece_color(target)
                if state.pool[captured_side][reveal_captured] <= 0:
                    available = False
        if available:
            return state.apply_move(
                move,
                rng=self.rng,
                reveal_choice=reveal_moved or None,
                capture_reveal_choice=reveal_captured or None,
            )
        return state.apply_move(move, rng=self.rng)

    @staticmethod
    def _sample_weighted(candidates: list[tuple], rng) -> object:
        """按正权重采样一个候选项；权重来自当前模拟世界的池计数。"""
        total = sum(weight for _, weight in candidates)
        roll = rng.randrange(total)
        acc = 0
        for item, weight in candidates:
            acc += weight
            if roll < acc:
                return item
        return candidates[-1][0]

    def _navigate_chance(self, chance_node: PUCTNode, sim_state: jq.GameState, rng):
        """世界条件采样（Issue 0005）：按本模拟世界的暗池计数采样揭示分支。

        先从完整世界池抽取类型，再查找或补建对应缓存分支；预建的候选集合不得
        限制概率支持。单/双揭示均只抽一次，不因缓存缺失重新抽样。
        本方法返回 ``(outcome_node, visited)``：visited 为入口机会节点**之后**的
        路径节点（单揭示时为结果节点本身，双揭示时为中间层与结果节点），入口机会
        节点已由选择阶段计入访问，不在 visited 中重复。
        无有效池或公共节点结构不一致时才返回 ``(None, [])``，作为异常兜底。
        """
        if chance_node.move is None:
            return None, []
        moved_piece = sim_state.board[chance_node.move[0]]
        if not moved_piece:
            return None, []
        moved_pool = sim_state.pool[jq.piece_color(moved_piece)]
        reveal_moved = 0
        if jq.is_dark(moved_piece):
            candidates = [(t, moved_pool[t]) for t in jq.POOL_TYPES if moved_pool[t] > 0]
            if not candidates:
                return None, []
            reveal_moved = self._sample_weighted(candidates, rng)
        target = sim_state.board[chance_node.move[1]]
        captured_pool = None
        reveal_captured = 0
        if target and jq.is_dark(target):
            captured_pool = sim_state.pool[jq.piece_color(target)]
            candidates = [(t, captured_pool[t]) for t in jq.POOL_TYPES if captured_pool[t] > 0]
            if not candidates:
                return None, []
            reveal_captured = self._sample_weighted(candidates, rng)

        if isinstance(chance_node.chance_counts, _DeferredChanceCounts):
            moved, captures = _full_chance_outcomes(chance_node.chance_counts)
            moved_item = next(((i, p) for i, (r, p) in enumerate(moved) if r == reveal_moved), None)
            captured_item = next(((i, p) for i, (r, p) in enumerate(captures) if r == reveal_captured), None)
            if (moved_item is not None and bool(captures) == (captured_pool is not None)
                    and (captured_pool is None or captured_item is not None)):
                index, prior = moved_item
                child = _deferred_moved_child(chance_node, index, reveal_moved, prior, captures)
                if captured_pool is None:
                    return child, [child]
                sub_index, sub_prior = captured_item
                outcome = _deferred_captured_child(child, sub_index, reveal_captured, sub_prior)
                return outcome, [child, outcome]
            # Preserve the eager path's defensive behavior for inconsistent external states.
            materialize_chance_outcomes(chance_node)

        moved_counts, captured_counts = chance_node.chance_counts or (None, None)
        child = next((node for node in chance_node.children.values() if node.reveal_moved == reveal_moved), None)
        if child is None:
            counts = moved_counts if moved_counts is not None else moved_pool
            prior = counts[reveal_moved] / sum(counts) if reveal_moved else 1.0
            child = PUCTNode(prior, chance_node.player_to_move, is_chance=captured_pool is not None)
            child.reveal_moved = reveal_moved
            chance_node.children[max(chance_node.children, default=-1) + 1] = child
            self.chance_outcomes_added += 1
        visited = [child]
        if child.is_chance != (captured_pool is not None):
            return None, []
        if captured_pool is None:
            return child, visited
        outcome = next((node for node in child.children.values() if node.reveal_captured == reveal_captured), None)
        if outcome is None:
            counts = captured_counts if captured_counts is not None else captured_pool
            outcome = PUCTNode(counts[reveal_captured] / sum(counts), chance_node.player_to_move)
            outcome.reveal_moved = reveal_moved
            outcome.reveal_captured = reveal_captured
            child.children[max(child.children, default=-1) + 1] = outcome
            self.chance_outcomes_added += 1
        visited.append(outcome)
        return outcome, visited

    def search(
        self,
        root_state: jq.GameState,
        belief: im.BeliefModel,
        history: list[tuple[int, int]],
        temperature: float,
        simulations: int | None = None,
        add_dirichlet: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        prune_policy: bool = False,
        return_root: bool = False,
    ) -> tuple[list[float], tuple[int, int] | None] | tuple[list[float], tuple[int, int] | None, PUCTNode]:
        sims = simulations or self.config.mcts_simulations
        batch_size = max(1, min(int(self.config.leaf_batch_size), sims))
        virtual_loss = float(self.config.virtual_loss)
        root = PUCTNode(1.0, root_state.side)
        root_info = belief.info_state(root_state, root_state.side)
        # 仅根信息生成粒子；后续行动者视图来自该粒子的模拟观测，不能读取
        # 真实 belief 的对手槽。环境揭示边各自记账，决策按行动者信息集共享。
        decision_table = {im.information_set_key(root_info, (), ()): root}
        priors, _ = self._evaluate(root_info, history)
        legal_moves = root_state.legal_moves()
        if add_dirichlet:
            priors = apply_dirichlet_noise(
                priors, legal_moves, dirichlet_alpha, dirichlet_epsilon, rng=self.rng
            )
        root_specs = self._chance_specs(root_state, root_info, legal_moves)
        self._expand_node(root, legal_moves, priors, root_specs)

        completed = 0
        while completed < sims:
            n_batch = min(batch_size, sims - completed)
            pending: list[tuple] = []
            for _ in range(n_batch):
                node = root
                # 单次确定化（根玩家视角）：整条模拟复用同一世界。
                sim_state = root_info.determinize(self.rng)
                sim_view = im.BeliefModel.for_simulation(root_info, root_state.side, sim_state)
                sim_history = history.copy()
                public_path = []
                private_paths = [[], []]
                actor_info = root_info
                path = [node]

                outcome, _ = sim_state.outcome_checked()
                while outcome is None and node.children:
                    move, child = node.select_child(self.config.c_puct)
                    child.visits += 1
                    child.value_sum += virtual_loss
                    node = child
                    if node.is_chance:
                        # 世界条件采样（Issue 0005）：按本模拟世界的池计数采样揭示分支，
                        # 保证 (世界, 揭示) 联合分布与真实规则一致，不再出现回退污染。
                        outcome_node, visited = self._navigate_chance(node, sim_state, self.rng)
                        if outcome_node is not None:
                            self.chance_applied += 1
                            for chance_node in visited:
                                chance_node.visits += 1
                                chance_node.value_sum += virtual_loss
                            path.append(node)
                            path.extend(visited)
                            node = outcome_node
                            undo = self._apply_chance_move(
                                sim_state, move, node.reveal_moved, node.reveal_captured
                            )
                        else:
                            # 无效池/节点结构的异常兜底，再进入当前行动者的信息集；
                            # 不得把揭示序号当作着法继续选择（Issue 0005）。
                            self.chance_nav_miss += 1
                            undo = sim_state.apply_move(move, rng=self.rng)
                            path.append(node)
                    else:
                        undo = sim_state.apply_move(move, rng=self.rng)
                        path.append(node)
                        if undo.moved_reveal:
                            # 关闭机会枚举时仍隔离公开结果，不能把首次翻出的棋子
                            # 对应的 legal_moves/prior 复用到另一个公开棋盘。
                            node = node.public_outcome(undo.moved_reveal)
                            node.visits += 1
                            node.value_sum += virtual_loss
                            path.append(node)
                    sim_view.observe_move(undo)
                    sim_history.append(move)
                    im.record_observation(move, undo, public_path, private_paths)
                    actor_info = None
                    key = sim_view.information_set_key(
                        sim_state, sim_state.side, public_path, private_paths[sim_state.side]
                    )
                    decision = decision_table.get(key)
                    if decision is None:
                        decision = PUCTNode(1.0, sim_state.side)
                        decision_table[key] = decision
                    node.decision_nodes[key] = decision
                    node = decision
                    node.visits += 1
                    node.value_sum += virtual_loss
                    path.append(node)
                    outcome, _ = sim_state.outcome_checked()

                if outcome is not None:
                    value = outcome_value_for_player(outcome, sim_state.side)
                    self._backup(path, value, virtual_loss)
                    continue

                leaf_moves = sim_state.legal_moves()
                if not leaf_moves:
                    winner = 1 - sim_state.side
                    result = jq.RESULT_RED_WIN if winner == jq.RED else jq.RESULT_BLACK_WIN
                    value = outcome_value_for_player(result, sim_state.side)
                    self._backup(path, value, virtual_loss)
                    continue

                # 网络训练语义是行动者自己的观测。这里使用合成观测，根不可见的
                # 对手知识来自根一致采样；真实对手 observer 槽始终不进入搜索。
                leaf_info = (actor_info if actor_info is not None else
                             sim_view.info_state(sim_state, sim_state.side))
                leaf_specs = self._chance_specs(sim_state, leaf_info, leaf_moves)
                pending.append((path, leaf_moves, leaf_info, sim_history, leaf_specs))

            if pending:
                self.leaf_evals_total += len(pending)
                if self.config.leaf_dedup:
                    # 快照只活在当前批次内，签名和编码共用；不缓存可变 InformationState。
                    repetition_entries = [_repetition_entries(item[2]) for item in pending]
                    signatures = [leaf_input_signature(item[2], item[3], repetition_entries=entries)
                                  for item, entries in zip(pending, repetition_entries)]
                    unique_slots: dict[tuple, int] = {}
                    first_indices: list[int] = []
                    for index, signature in enumerate(signatures):
                        if signature not in unique_slots:
                            unique_slots[signature] = len(first_indices)
                            first_indices.append(index)
                    if len(first_indices) == len(pending):
                        policies, values = self._evaluate_batch(
                            [item[2] for item in pending],
                            [item[3] for item in pending],
                            repetition_entries=repetition_entries,
                        )
                    else:
                        unique_policies, unique_values = self._evaluate_batch(
                            [pending[index][2] for index in first_indices],
                            [pending[index][3] for index in first_indices],
                            repetition_entries=[repetition_entries[index] for index in first_indices],
                        )
                        slots = [unique_slots[signature] for signature in signatures]
                        policies = [unique_policies[slot] for slot in slots]
                        values = [unique_values[slot] for slot in slots]
                    self.leaf_evals_unique += len(first_indices)
                else:
                    policies, values = self._evaluate_batch(
                        [item[2] for item in pending],
                        [item[3] for item in pending],
                    )
                    self.leaf_evals_unique += len(pending)
                for (path, leaf_moves, _leaf_info, _sim_history, leaf_specs), policy_row, value in zip(
                    pending, policies, values
                ):
                    leaf = path[-1]
                    # 同一叶节点在同批内被多次到达时只展开一次：二次 expand 会重造
                    # children，并额外消耗机会节点采样的 RNG。
                    if not leaf.children:
                        self._expand_node(leaf, leaf_moves, policy_row, leaf_specs)
                    self._backup(path, value, virtual_loss)
            completed += n_batch

        policy = [0.0] * MOVE_SIZE
        if not root.children:
            if return_root:
                return policy, None, root
            return policy, None
        total_visits = sum(child.visits for child in root.children.values())
        if total_visits <= 0:
            # sims=0 或根已终局：没有任何访问回填，返回空策略（与无子节点同路径）。
            if return_root:
                return policy, None, root
            return policy, None
        if prune_policy:
            pruned = pruned_policy_from_root(root, self.config.c_puct)
            policy = pruned.tolist()
        else:
            for move, child in root.children.items():
                policy[move_to_index(move)] = child.visits / total_visits

        if temperature <= 1e-3:
            move = max(root.children.items(), key=lambda item: item[1].visits)[0]
            if return_root:
                return policy, move, root
            return policy, move

        weights = []
        moves = []
        log_weights = []
        for move, child in root.children.items():
            if child.visits > 0:
                log_w = math.log(child.visits) / temperature
            else:
                log_w = float("-inf")
            log_weights.append(log_w)
            moves.append(move)
        max_log = max(log_weights)
        for log_w in log_weights:
            weight = math.exp(log_w - max_log) if log_w != float("-inf") else 0.0
            weights.append(weight)
        total = sum(weights)
        if total <= 0:
            move = self.rng.choice(moves)
        else:
            roll = self.rng.random() * total
            acc = 0.0
            move = moves[-1]
            for w, m in zip(weights, moves):
                acc += w
                if roll <= acc:
                    move = m
                    break
        if return_root:
            return policy, move, root
        return policy, move


class ReplayBuffer:
    """自博弈样本缓冲区。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: list[CompactSample] = []

    def add_game(self, samples: list[CompactSample]) -> None:
        self.data.extend(samples)
        if len(self.data) > self.capacity:
            # 删除最旧前缀，避免逐局复制并增减整个 buffer 的对象引用。
            del self.data[: -self.capacity]

    def sample(self, batch_size: int) -> list[CompactSample]:
        return random.sample(self.data, batch_size)

    def __len__(self) -> int:
        return len(self.data)


def save_compact_samples(
    samples: list[CompactSample],
    path: str,
    meta: dict | None = None,
) -> str:
    """保存样本（gzip + pickle，节省空间；临时文件 + 原子替换）。"""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    payload = {"version": DATA_VERSION, "samples": samples, "meta": meta or {}}
    tmp_path = f"{path}.tmp"
    with gzip.open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    return path


def save_compact_game(samples: list[CompactSample], data_dir: str, meta: dict | None = None) -> str:
    """按单局生成文件。"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(data_dir, f"game_{stamp}_{random.randint(0, 9999):04d}{SELFPLAY_EXT}")
    return save_compact_samples(samples, path, meta)


def save_compact_iteration(
    samples: list[CompactSample],
    data_dir: str,
    iteration: int,
    meta: dict | None = None,
) -> str:
    """按迭代保存样本（每个 iteration 一个文件）。"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(data_dir, f"iter_{stamp}_{iteration:04d}{SELFPLAY_EXT}")
    while os.path.exists(path):  # 快速重启同一未提交轮次也保留已有样本文件。
        path = os.path.join(data_dir, f"iter_{stamp}_{time.time_ns()}_{iteration:04d}{SELFPLAY_EXT}")
    return save_compact_samples(samples, path, meta)


_PICKLE_SAMPLE_NAMES = {"RawSample", "CompactSample", "SelfPlayResult"}


class _CompatUnpickler(pickle.Unpickler):
    """兼容以 ``__main__`` / ``__mp_main__`` 形式保存的样本类引用。

    ``python azero.py`` 直接运行时，类定义在 ``__main__`` 中，pickle 会记录
    ``__main__.CompactSample``；从其他模块 ``import azero`` 读取这类文件时
    需要把引用重定向回本模块。
    """

    def find_class(self, module: str, name: str):
        if module in ("__main__", "__mp_main__") and name in _PICKLE_SAMPLE_NAMES:
            module = __name__
        return super().find_class(module, name)


def load_compact_payload(path: str) -> tuple[list[CompactSample], dict]:
    """读取紧凑样本文件，返回 (样本, 元数据)。

    仅加载自己生成或可信来源的 Pickle；版本检查发生在反序列化之后，
    不能阻止恶意文件执行代码。详见 SECURITY.md。
    接受 v3 与 v2；v2 缺失的完整重复历史显式标为 None，保留原有当前计数。
    其他版本抛 ValueError（调用方跳过并告警）。
    """
    with gzip.open(path, "rb") as f:
        payload = _CompatUnpickler(f).load()
    if isinstance(payload, dict) and "samples" in payload:
        version = int(payload.get("version", -1))
        if version not in (2, DATA_VERSION):
            raise ValueError(f"data version mismatch: {version} not in (2, {DATA_VERSION})")
        samples, meta = payload["samples"], payload.get("meta") or {}
    elif isinstance(payload, list):
        samples, meta = payload, {}
    else:
        raise ValueError("invalid compact data format")
    for sample in samples:
        if not hasattr(sample, "repetition_history"):
            sample.repetition_history = None
    return samples, meta


def load_compact_game(path: str) -> list[CompactSample]:
    """读取紧凑样本文件。"""
    samples, _ = load_compact_payload(path)
    return samples


def saved_game_lengths(samples: list[CompactSample], meta: dict) -> list[int]:
    lengths = meta.get("game_lengths")
    if lengths is None:  # 历史文件按当时的整文件增强顺序重建。
        lengths = [len(samples)]
    if not isinstance(lengths, list) or any(type(n) is not int or n < 0 for n in lengths) or sum(lengths) != len(samples):
        raise ValueError("自博弈 game_lengths 与样本数不一致")
    mirror, flip = saved_augmentation(meta)
    factor = (2 if mirror else 1) * (2 if flip else 1)
    if any(length % factor for length in lengths):
        raise ValueError("game_lengths incompatible with saved augmentation")
    return lengths


def saved_augmentation(meta: dict) -> tuple[bool, bool]:
    flags = (meta.get("augment", False), meta.get("augment_flip", False))
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("invalid saved augmentation flags")
    return flags


def iter_saved_games(samples: list[CompactSample], meta: dict, mirror: bool, flip: bool):
    """Recover each original block, then produce the union of requested/stored symmetries."""
    stored_mirror, stored_flip = saved_augmentation(meta)
    factor = (2 if stored_mirror else 1) * (2 if stored_flip else 1)
    start = 0
    for length in saved_game_lengths(samples, meta):
        if length % factor:
            raise ValueError("game_lengths incompatible with saved augmentation")
        originals = samples[start:start + length // factor]
        yield augment_samples(originals, mirror=mirror or stored_mirror, flip=flip or stored_flip)
        start += length


def add_saved_samples(buffer: ReplayBuffer, samples: list[CompactSample], meta: dict,
                      mirror: bool, flip: bool, skip_augmented: int = 0) -> int:
    """按每局增强顺序回填，可跳过已淘汰的前缀；返回文件的完整增强样本数。"""
    if type(skip_augmented) is not int or skip_augmented < 0:
        raise ValueError("无效重放前缀裁剪数")
    total = 0
    pending: list[CompactSample] = []
    # 有界批量插入，减少满 buffer 的移动次数；不积累整份大文件。
    batch_limit = min(buffer.capacity, 65536) if buffer.capacity > 0 else 65536
    for augmented in iter_saved_games(samples, meta, mirror, flip):
        total += len(augmented)
        if skip_augmented:
            skipped = min(skip_augmented, len(augmented))
            augmented = augmented[skipped:]
            skip_augmented -= skipped
        if not augmented:
            continue
        if buffer.capacity <= 0 or (not pending and len(augmented) >= batch_limit):
            buffer.add_game(augmented)
        else:
            pending.extend(augmented)
            if len(pending) >= batch_limit:
                buffer.add_game(pending)
                pending.clear()
    if pending:
        buffer.add_game(pending)
    if skip_augmented:
        raise ValueError("重放前缀裁剪数超过文件样本数")
    return total


@contextmanager
def _replay_restore_gc():
    """批量重建无环样本时暂停自动扫描；结束后统一回收并恢复调用方设置。"""
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            try:
                # 将长期保留的 buffer 对象移到老年代，避免重新启用后的首次小分配扫描全量。
                gc.collect(2)
            finally:
                gc.enable()


def replay_file_record(path: str, augmented_count: int) -> dict:
    with open(path, "rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(name=os.path.basename(path), sha256=checksum, augmented_count=augmented_count)


def replay_record_size(record: dict) -> int:
    """文件完整计数用于校验；可选前缀偏移用于保持扩容前的 FIFO 边界。"""
    count = record["augmented_count"]
    skipped = record.get("skip_augmented", 0)
    if type(count) is not int or type(skipped) is not int or not 0 <= skipped <= count:
        raise ValueError("无效重放样本数/前缀裁剪数")
    return count - skipped


def retained_replay_files(records: list[dict], capacity: int) -> list[dict]:
    """只保留重建当前 FIFO 所需的文件，完整样本仍留在原目录。"""
    if capacity <= 0:
        return list(records)
    count = 0
    for index in range(len(records) - 1, -1, -1):
        count += replay_record_size(records[index])
        if count >= capacity:
            return records[index:]
    return list(records)


def replay_files_for_buffer(records: list[dict], buffer_size: int) -> list[dict]:
    """显式保存当前 FIFO 的起点，扩容后不重新引入首文件中已淘汰的样本。"""
    if buffer_size == 0:
        return []
    retained = retained_replay_files(records, buffer_size)
    excess = sum(replay_record_size(record) for record in retained) - buffer_size
    if excess < 0:
        raise ValueError("重放清单不足以重建缓冲")
    if excess:
        first = dict(retained[0])
        first["skip_augmented"] = first.get("skip_augmented", 0) + excess
        retained = [first, *retained[1:]]
    return retained


def load_selfplay_data(
    data_dir: str,
    buffer: ReplayBuffer,
    logger: logging.Logger | None = None,
    max_files: int = 0,
    augment: bool = False,
    augment_flip: bool = False,
    ruleset_version: int | None = None,
    loaded_files: list[dict] | None = None,
) -> int:
    """加载已有自博弈数据到缓冲区；规则版本不符的文件会被跳过。

    加载语义：

    - ``max_files <= 0``（默认）：按 ``buffer.capacity`` 自适应。按文件 mtime
      从新到旧扫描，累计样本数（增强按每个开关 ×2 计）达到容量即停止，
      再按旧→新顺序灌入缓冲。``ReplayBuffer`` 超容量时 FIFO 淘汰最旧，
      因此最终保留的是"最新的容量内样本"，与连续训练（全部数据按时间序灌入、
      超出即淘汰）的结果一致。
    - ``max_files > 0``：保留旧语义（仅加载最近 ``max_files`` 个文件，不做
      容量裁剪），供历史实验配置复现使用；可能只重建部分缓冲。

    ``augment`` 为左右镜像、``augment_flip`` 为楚河汉界翻面（含红黑互换）；
    两者同开时原始样本 4 倍；已增强文件只补缺失变换，已有变换不会重复扩增。
    返回实际加载的样本数（含增强副本）；自适应模式下不超过 ``capacity``。
    """
    if not os.path.isdir(data_dir):
        return 0
    files = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(SELFPLAY_EXT)
    ]
    files.sort(key=lambda p: os.path.getmtime(p))
    skipped = 0
    if max_files > 0:
        if len(files) > max_files:
            files = files[-max_files:]
        total = 0
        for path in files:
            try:
                samples, meta = load_compact_payload(path)
                if ruleset_version is not None and int(meta.get("ruleset_version", -1)) != ruleset_version:
                    skipped += 1
                    continue
                count = add_saved_samples(buffer, samples, meta, augment, augment_flip)
                total += count
                if loaded_files is not None:
                    loaded_files.append(replay_file_record(path, count))
            except (
                OSError,
                pickle.UnpicklingError,
                ValueError,
                EOFError,
                zlib.error,
                AttributeError,
                TypeError,
            ) as exc:
                if logger:
                    logger.warning("自博弈数据读取失败: %s (%s)", path, exc)
        if skipped and logger:
            logger.warning("跳过 %d 个规则版本不符的自博弈文件。", skipped)
        return total

    capacity = max(0, int(buffer.capacity))
    selected: list[tuple[str, list[CompactSample], dict]] = []
    estimated = 0
    for path in reversed(files):
        if capacity > 0 and estimated >= capacity:
            break
        try:
            samples, meta = load_compact_payload(path)
            saved_game_lengths(samples, meta)
        except (
            OSError,
            pickle.UnpicklingError,
            ValueError,
            EOFError,
            zlib.error,
            AttributeError,
            TypeError,
        ) as exc:
            if logger:
                logger.warning("自博弈数据读取失败: %s (%s)", path, exc)
            continue
        if ruleset_version is not None and int(meta.get("ruleset_version", -1)) != ruleset_version:
            skipped += 1
            continue
        selected.append((path, samples, meta))
        stored_mirror, stored_flip = saved_augmentation(meta)
        estimated += len(samples) * (2 if augment and not stored_mirror else 1) * (2 if augment_flip and not stored_flip else 1)
    total = 0
    for path, samples, meta in reversed(selected):
        count = add_saved_samples(buffer, samples, meta, augment, augment_flip)
        total += count
        if loaded_files is not None:
            loaded_files.append(replay_file_record(path, count))
    if skipped and logger:
        logger.warning("跳过 %d 个规则版本不符的自博弈文件。", skipped)
    return min(total, capacity) if capacity > 0 else total


def merge_selfplay_files(
    input_dir: str,
    output_path: str,
    augment: bool = False,
    augment_flip: bool = False,
    logger: logging.Logger | None = None,
    ruleset_version: int | None = None,
) -> int:
    """合并 selfplay 文件，保留逐局边界；拒绝混合规则版本及覆盖已有输出。

    ``augment`` 为左右镜像、``augment_flip`` 为楚河汉界翻面（含红黑互换）。
    已存变换取并集，按每局原样/镜像/翻面/复合顺序生成，不重复增强。
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(input_dir)
    if os.path.exists(output_path):
        raise FileExistsError(output_path)
    files = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.endswith(SELFPLAY_EXT)
    ]
    output_abs = os.path.abspath(output_path)
    files = [path for path in files if os.path.abspath(path) != output_abs]
    files.sort(key=lambda p: os.path.getmtime(p))
    sources = []
    source_versions: set[int] = set()
    unscoped_files = 0
    for path in files:
        try:
            samples, meta = load_compact_payload(path)
        except (OSError, pickle.UnpicklingError, ValueError) as exc:
            if logger:
                logger.warning("合并失败: %s (%s)", path, exc)
            continue
        version = meta.get("ruleset_version")
        if ruleset_version is not None and (version is None or int(version) != ruleset_version):
            raise ValueError(f"ruleset_version mismatch: {path} ({version} != {ruleset_version})")
        if version is None:
            unscoped_files += 1
        else:
            source_versions.add(int(version))
        stored_mirror, stored_flip = saved_augmentation(meta)
        augment = augment or stored_mirror
        augment_flip = augment_flip or stored_flip
        sources.append((samples, meta))
    if len(source_versions) > 1 or (source_versions and unscoped_files):
        raise ValueError("cannot merge mixed/unknown ruleset_version sources")
    if ruleset_version is None and len(source_versions) == 1 and unscoped_files == 0:
        ruleset_version = next(iter(source_versions))
    merged: list[CompactSample] = []
    game_lengths = []
    for samples, meta in sources:
        for game in iter_saved_games(samples, meta, augment, augment_flip):
            merged.extend(game)
            game_lengths.append(len(game))
    output_meta = {
        "merged_from": len(sources),
        "samples": len(merged),
        "augment": augment,
        "augment_flip": augment_flip,
        "game_lengths": game_lengths,
    }
    if ruleset_version is not None:
        output_meta["ruleset_version"] = ruleset_version
    save_compact_samples(merged, output_path, meta=output_meta)
    return len(merged)


def self_play_game(
    net: PolicyValueNet,
    config: AZConfig,
    device: torch.device,
    rng: random.Random | None = None,
    strong: bool = False,
) -> SelfPlayResult:
    """单局自博弈，返回紧凑样本与局面信息。"""
    rng = rng or random.Random()
    state = jq.GameState.initial(rng=rng)
    belief = im.init_belief_for_game(state)
    history: list[tuple[int, int]] = []
    mcts = NeuralMCTS(net, config, device, rng)
    raw_samples: list[RawSample] = []
    if strong:
        base_sims = config.strong_mcts_simulations
        dirichlet_eps = config.strong_dirichlet_epsilon
        temp_final = config.strong_temperature_final
        temp_decay = config.strong_temperature_decay_moves
    else:
        base_sims = config.mcts_simulations
        dirichlet_eps = config.dirichlet_epsilon
        temp_final = config.temperature_final
        temp_decay = config.temperature_decay_moves
    pcr_active = config.pcr_enabled and not strong

    while state.game_outcome()[0] is None:
        info = belief.info_state(state, state.side)
        temperature = temperature_for_move(
            len(history),
            config.temperature_initial,
            temp_final,
            temp_decay,
        )
        if pcr_active:
            sims, policy_weight = sample_playout_cap(config, rng)
        else:
            sims, policy_weight = base_sims, 1.0
        policy, move = mcts.search(
            state,
            belief,
            history,
            temperature,
            simulations=sims,
            add_dirichlet=True,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=dirichlet_eps,
            prune_policy=True,
        )
        raw_samples.append(raw_sample_from_info(info, history, policy, policy_weight))
        if move is None:
            break
        im.apply_move_and_update_belief(state, belief, move, rng=rng)
        history.append(move)

    outcome, _ = state.game_outcome()
    samples = finalize_samples(raw_samples, outcome)
    return SelfPlayResult(
        samples=samples,
        outcome=outcome,
        move_count=len(history),
        leaf_evals_total=mcts.leaf_evals_total,
        leaf_evals_unique=mcts.leaf_evals_unique,
    )


def stop_child_processes() -> int:
    """终止本进程的全部 spawn 子进程（Ctrl+C 中断时清理 worker，避免残留）。"""
    procs = mp.active_children()
    # During Windows start(), the new Popen may not yet be in active_children().
    spawning = mp.context.get_spawning_popen()
    if spawning is not None:
        procs.append(spawning)
    killed = 0
    for proc in procs:
        try:
            proc.terminate()
            killed += 1
        except (OSError, ValueError):
            continue  # A dying child must not prevent cleanup of the remaining workers.
    return killed


def _sigint_handler(signum, frame) -> None:
    """Ctrl+C：终止子进程后立即退出，绕过 ProcessPoolExecutor 的阻塞式关闭。

    Windows 上 Ctrl+C 会同时发给主进程与 spawn 子进程；若走默认的
    KeyboardInterrupt → ``with`` 退出路径，``shutdown(wait=True)`` 可能与
    已死亡的 worker 死锁（实测主进程卡在关闭流程、永不退出）。这里直接
    ``os._exit``，保证进程立即结束；当前未完成的迭代不保存。
    """
    tr.terminate_owned_job(130)
    try:
        killed = stop_child_processes()
        sys.stderr.write(
            f"\n训练被中断（Ctrl+C）：已终止 {killed} 个残留子进程；"
            "从最近完整检查点继续，当前未提交的轮次会重新执行。\n"
        )
        sys.stderr.flush()
    except Exception:
        pass
    finally:
        os._exit(130)


def install_interrupt_handler() -> None:
    """把 SIGINT 绑定为硬退出处理器（默认 KeyboardInterrupt 路径在 Windows 上可能挂起）。"""
    signal.signal(signal.SIGINT, _sigint_handler)
    tr.refresh_console_handler()


def generate_selfplay_results(
    net: PolicyValueNet,
    config: AZConfig,
    device: torch.device,
    rng: random.Random,
    games: int,
    logger: logging.Logger | None = None,
    strong_flags: list[bool] | None = None,
    iteration: int | None = None,
) -> list[SelfPlayResult]:
    """根据配置选择串行或多进程自博弈。"""
    if games <= 0:
        return []
    flags = strong_flags or [False] * games
    if len(flags) != games:
        raise ValueError("strong_flags size mismatch")
    if config.self_play_workers <= 1:
        return [
            self_play_game(net, config, device, rng, strong=flags[i])
            for i in range(games)
        ]

    # 多进程自博弈要求 GPU 预测，否则按要求回退为单进程。
    if device.type != "cuda" or not torch.cuda.is_available():
        if logger:
            logger.info("GPU 不可用，已回退为单进程自博弈。")
        return [
            self_play_game(net, config, device, rng, strong=flags[i])
            for i in range(games)
        ]
    if not str(config.self_play_device).startswith("cuda"):
        if logger:
            logger.info("自博弈设备非 GPU，已回退为单进程自博弈。")
        return [
            self_play_game(net, config, device, rng, strong=flags[i])
            for i in range(games)
        ]

    max_workers = min(config.self_play_workers, games)
    ctx = mp.get_context("spawn")
    model_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    worker_device = config.self_play_device or "cpu"
    results: dict[int, SelfPlayResult] = {}
    started = time.time()
    completed = 0
    executor = futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_selfplay_worker,
        initargs=(model_state, config, worker_device),
    )
    try:
        seeds = [rng.randrange(1 << 30) for _ in range(games)]
        tasks = list(zip(seeds, flags))
        future_map = {executor.submit(_selfplay_worker, task): index for index, task in enumerate(tasks)}
        pending = set(future_map)
        # 带 1s 超时轮询而非 as_completed 的无超时等待：主线程会被周期性唤醒并
        # 检查信号，保证 Ctrl+C 在 ~1s 内触发处理器（而不是等下一局完成）。
        while pending:
            done, pending = futures.wait(
                pending, timeout=1.0, return_when=futures.FIRST_COMPLETED
            )
            for fut in done:
                try:
                    results[future_map[fut]] = fut.result()
                except Exception as exc:
                    if logger:
                        logger.warning("自博弈进程失败: %s", exc)
                completed += 1
                if (
                    logger
                    and config.self_play_log_interval > 0
                    and completed % config.self_play_log_interval == 0
                ):
                    logger.info(
                        "自博弈进度: iter=%s game=%d/%d elapsed=%.0fs",
                        iteration if iteration is not None else "?",
                        completed,
                        games,
                        time.time() - started,
                    )
    except BaseException:
        # Ctrl+C/异常：不做阻塞式关闭（Windows 上可能与死 worker 死锁），
        # 取消排队任务并终止 worker 后抛出。
        executor.shutdown(wait=False, cancel_futures=True)
        stop_child_processes()
        raise
    else:
        executor.shutdown(wait=True)
    # GPU worker 的完成顺序不稳定；固定任务顺序后，续训采样不受调度时序影响。
    return [results[index] for index in sorted(results)]


_EVAL_NET: PolicyValueNet | None = None
_EVAL_CONFIG: "AZConfig" | None = None
_EVAL_DEVICE: torch.device | None = None


def _init_eval_worker(model_state: dict, config: "AZConfig", device_str: str) -> None:
    """多进程评估初始化：每个子进程各自加载一次模型。"""
    global _EVAL_NET, _EVAL_CONFIG, _EVAL_DEVICE
    # 同自博弈 worker：Ctrl+C 由主进程统一处理。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    device = torch.device(device_str)
    net = PolicyValueNet().to(device)
    load_state_dict_padded(net, model_state)
    net.eval()
    _EVAL_NET = net
    _EVAL_CONFIG = config
    _EVAL_DEVICE = device


def _play_eval_game(
    net: PolicyValueNet,
    config: AZConfig,
    device: torch.device,
    neural_side: int,
    seed: int,
) -> float | None:
    """评估用单局对弈：神经网络 vs RIS-ISMCTS，返回神经网络视角得分。"""
    rng, neural_rng, opponent_rng = evaluation_rngs(seed)
    mcts = NeuralMCTS(net, replace(config, leaf_batch_size=config.eval_leaf_batch_size), device, neural_rng)
    ris_agent = im.ISMCTSAgent(im.ISMCTSConfig(iterations=config.ris_iterations), opponent_rng)
    dark_types = jq.materialize_dark_types(random.Random(f"jieqi-pair-shuffle:{seed}"))
    state = jq.GameState.initial(rng=rng, dark_types=dark_types)
    belief = im.init_belief_for_game(state)
    history: list[tuple[int, int]] = []
    while state.game_outcome()[0] is None:
        if state.side == neural_side:
            _, move = mcts.search(
                state,
                belief,
                history,
                temperature=1e-6,
                simulations=config.eval_mcts_simulations,
                add_dirichlet=False,
            )
        else:
            move, _ = ris_agent.select_move(state, belief)
        if move is None:
            break
        im.apply_move_and_update_belief(state, belief, move, rng=rng)
        history.append(move)
    outcome, _ = state.game_outcome()
    return outcome_value_for_player(outcome, neural_side) if outcome is not None else None


def evaluation_rngs(seed: int) -> tuple[random.Random, random.Random, random.Random]:
    """Independent environment and agent streams, shared by both evaluation entry points."""
    return tuple(random.Random(f"jieqi-eval-{role}:{seed}")
                 for role in ("environment", "neural", "opponent"))


def _eval_pairs(net, config, device, pairs: int, base_seed: int, start_pair: int = 0) -> dict:
    stats = dict(wins=0, draws=0, losses=0, unfinished=0, pairs_used=0, pairs_excluded=0)
    for pair in range(pairs):
        seed = base_seed + start_pair + pair
        scores = [_play_eval_game(net, config, device, side, seed) for side in (jq.RED, jq.BLACK)]
        if any(score is None for score in scores):
            stats["unfinished"] += sum(score is None for score in scores)
            stats["pairs_excluded"] += 1
            continue
        stats["pairs_used"] += 1
        for score in scores:
            if score > 0:
                stats["wins"] += 1
            elif score < 0:
                stats["losses"] += 1
            else:
                stats["draws"] += 1
    return stats


def _eval_worker(args: tuple[int, int, int]) -> dict:
    """Evaluate complete same-shuffle pairs; exclude a pair if either game is unfinished."""
    if _EVAL_NET is None or _EVAL_CONFIG is None or _EVAL_DEVICE is None:
        raise RuntimeError("eval worker not initialized")
    return _eval_pairs(_EVAL_NET, _EVAL_CONFIG, _EVAL_DEVICE, *args)


def _resolve_eval_device(config_device: str | None, runtime_device: torch.device) -> str:
    """Resolve eval device with CUDA availability fallback."""
    requested = str(config_device).strip() if config_device else ""
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return runtime_device.type if runtime_device.type != "cuda" else "cpu"
        return requested
    if runtime_device.type.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return runtime_device.type


def evaluate_vs_ris_mp(
    net: PolicyValueNet,
    config: AZConfig,
    device: torch.device,
    games: int,
    rng: random.Random,
) -> dict:
    """多进程评估当前模型强度（同种子双色配对）。"""
    if games <= 0:
        return {"wins": 0, "draws": 0, "losses": 0, "score": 0.0}
    pairs = (games + 1) // 2
    max_workers = min(config.eval_workers, pairs)
    ctx = mp.get_context("spawn")
    model_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    device_str = _resolve_eval_device(config.eval_device, device)

    base = pairs // max_workers
    remainder = pairs % max_workers
    base_seed = rng.randrange(1 << 30)
    tasks: list[tuple[int, int, int]] = []
    start_pair = 0
    for i in range(max_workers):
        count = base + (1 if i < remainder else 0)
        if count <= 0:
            continue
        tasks.append((count, base_seed, start_pair))
        start_pair += count

    stats = dict(wins=0, draws=0, losses=0, unfinished=0, pairs_used=0, pairs_excluded=0)
    executor = futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_eval_worker,
        initargs=(model_state, config, device_str),
    )
    try:
        pending = {executor.submit(_eval_worker, task) for task in tasks}
        # 同自博弈：1s 超时轮询，保证 Ctrl+C 及时响应。
        while pending:
            done, pending = futures.wait(
                pending, timeout=1.0, return_when=futures.FIRST_COMPLETED
            )
            for fut in done:
                result = fut.result()
                for key in stats:
                    stats[key] += result[key]
    except BaseException:
        # Ctrl+C/异常：不做阻塞式关闭，取消排队任务并终止 worker 后抛出。
        executor.shutdown(wait=False, cancel_futures=True)
        stop_child_processes()
        raise
    else:
        executor.shutdown(wait=True)

    total = stats["wins"] + stats["draws"] + stats["losses"]
    stats.update(score=(stats["wins"] + 0.5 * stats["draws"]) / total if total else 0.0,
                 base_seed=base_seed, pairs=pairs)
    return stats


def evaluate_vs_ris(
    net: PolicyValueNet,
    config: AZConfig,
    device: torch.device,
    games: int,
    rng: random.Random,
    jsonl_path: str | None = None,
    iteration: int | None = None,
    provenance: dict | None = None,
) -> dict:
    """用 RIS-ISMCTS 作为对手评估当前模型强度（同种子双色配对）。"""
    if games <= 0:
        return {"wins": 0, "draws": 0, "losses": 0, "score": 0.0}
    if config.eval_workers > 1:
        stats = evaluate_vs_ris_mp(net, config, device, games, rng)
    else:
        pairs = (games + 1) // 2
        base_seed = rng.randrange(1 << 30)
        stats = _eval_pairs(net, config, device, pairs, base_seed)
        total = stats["wins"] + stats["draws"] + stats["losses"]
        stats.update(score=(stats["wins"] + 0.5 * stats["draws"]) / total if total else 0.0,
                     base_seed=base_seed, pairs=pairs)

    if jsonl_path:
        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
        record = {
            "type": "train_eval",
            "eval_protocol_version": EVAL_PROTOCOL_VERSION,
            "iteration": iteration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mcts_simulations": config.eval_mcts_simulations,
            "leaf_batch_size": config.eval_leaf_batch_size,
            "leaf_dedup": config.leaf_dedup,
            "ris_iterations": config.ris_iterations,
            "ruleset_version": jq.RULESET_VERSION,
            "data_version": DATA_VERSION,
            "search_version": SEARCH_VERSION,
            "chance_nodes": config.chance_nodes,
            "chance_k": config.chance_k,
            "lazy_chance_allocation": config.lazy_chance_allocation,
            "provenance": provenance,
            "status_at_write": "unconfirmed" if provenance else "unrecorded",
            **stats,
        }
        with open(jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return stats


def train_step(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: list[CompactSample],
    device: torch.device,
    grad_clip_norm: float,
) -> tuple[float, float, float]:
    """单个 batch 的优化步骤。"""
    # PCR：fast 档样本策略目标权重为 0；旧格式样本按 1.0 处理。
    policy_weights = [float(getattr(sample, "policy_weight", 1.0)) for sample in batch]
    states_tensor, policy_tensor, value_tensor = [tensor.to(device) for tensor in compact_batch_to_tensors(batch)]

    policy_logits, values = net(states_tensor)
    per_sample_policy = -(policy_tensor * F.log_softmax(policy_logits, dim=1)).sum(dim=1)
    weight_tensor = torch.tensor(policy_weights, dtype=torch.float32, device=device)
    weight_sum = float(weight_tensor.sum())
    if weight_sum > 0.0:
        policy_loss = (per_sample_policy * weight_tensor).sum() / weight_tensor.sum()
    else:
        # 整个 batch 全部来自 fast 档：本步不产生策略梯度（value 照常回传）。
        policy_loss = torch.zeros((), dtype=torch.float32, device=device)
    value_loss = F.mse_loss(values, value_tensor)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def compute_train_steps(
    steps: int,
    batch_size: int,
    buffer_size: int,
) -> int:
    """固定步数制：每轮对 replay buffer 均匀采样训练 ``steps`` 步（AlphaZero 风格）。

    历史：S1.4 曾按“新增样本 1 epoch”折算步数（``added/batch``），在本项目数据率
    （64 局/轮 ≈ 9k 样本/轮）与 1M buffer 下每轮只有 ~15 步，单个样本整个生命周期
    平均仅被训练约 1 次，等价于不训练（Gate 320 轮 policy loss 仅 6.9→3.0）。
    详见 ADR-0010。
    """
    if steps <= 0 or batch_size <= 0 or buffer_size < batch_size:
        return 0
    return int(steps)


def fast_forward_scheduler(scheduler, steps: int) -> None:
    """断点续训时把 LR 调度器快进到已完成步数，避免从初值重新起跑。"""
    count = max(0, int(steps))
    if count == 0:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")
        for _ in range(count):
            scheduler.step()


def network_spec(net: PolicyValueNet) -> dict:
    return dict(channels=net.conv.out_channels, blocks=len(net.residual),
                state_channels=net.conv.in_channels, move_size=net.policy_fc.out_features)


def capture_training_rng(rng: random.Random) -> dict:
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), selfplay=rng.getstate(), torch=torch.get_rng_state(),
                numpy=[numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                       int(numpy_state[3]), float(numpy_state[4])],
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else [])


def restore_training_rng(saved: dict, rng: random.Random, logger: logging.Logger) -> None:
    random.setstate(saved["python"])
    rng.setstate(saved["selfplay"])
    state = saved["numpy"]
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    torch.set_rng_state(saved["torch"].cpu())
    cuda_states = saved["cuda"]
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != devices:
        logger.warning("CUDA RNG 设备数量改变 (%d -> %d)，跨硬件续训不保证逐位一致。", len(cuda_states), devices)
    for index, state in enumerate(cuda_states[:devices]):
        torch.cuda.set_rng_state(state.cpu(), index)


def save_training_checkpoint(path: str, net: PolicyValueNet, optimizer, scheduler,
                             state: TrainingState, rng: random.Random, config: AZConfig,
                             replay_files: list[dict], buffer_size: int, best_updated: bool = False,
                             provenance: dict | None = None, diagnostics: dict | None = None) -> None:
    """轮末 scheduler 已推进后提交；此文件是完整续训的唯一提交点。"""
    retained = retained_replay_files(replay_files, config.replay_buffer_size)
    replay_count = sum(replay_record_size(record) for record in retained)
    expected_size = min(replay_count, config.replay_buffer_size) if config.replay_buffer_size > 0 else replay_count
    if buffer_size != expected_size or state.ruleset_version != jq.RULESET_VERSION:
        raise ValueError("检查点缓冲/规则状态与重放清单不一致，拒绝提交")
    if scheduler is not None and scheduler.last_epoch != state.iteration:
        raise ValueError("调度器与已完成轮计数不一致，拒绝提交")
    if provenance is not None:
        rr.validate_training_lineage(provenance, asdict(config), network_spec(net),
                                     dict(ruleset=jq.RULESET_VERSION, data=DATA_VERSION, search=SEARCH_VERSION),
                                     state.iteration, retained)
        if provenance["model_sha256"] != rr.model_hash(net):
            raise ValueError("检查点来源与模型/轮计数不一致，拒绝提交")
    td.validate_state(diagnostics, state.iteration,
                      provenance["model_sha256"] if provenance else rr.model_hash(net) if diagnostics else None)
    if (diagnostics and diagnostics.get("latest") and provenance and provenance.get("attempt") and
            diagnostics["latest"]["provenance"] != provenance["attempt"]):
        raise ValueError("诊断报告与检查点尝试来源不一致，拒绝提交")
    payload = dict(format="jieqi-training", version=TRAINING_CHECKPOINT_VERSION,
                   ruleset_version=jq.RULESET_VERSION, data_version=DATA_VERSION,
                   search_version=SEARCH_VERSION, network=network_spec(net), config=asdict(config),
                   model=net.state_dict(), optimizer_type=type(optimizer).__name__,
                   optimizer=optimizer.state_dict(), scheduler_type=type(scheduler).__name__ if scheduler else None,
                   scheduler=scheduler.state_dict() if scheduler else None, train_state=asdict(state),
                   rng=capture_training_rng(rng), replay_files=retained,
                   buffer_size=buffer_size, best_updated=best_updated, provenance=provenance,
                   diagnostics=diagnostics)
    atomic_torch_save(payload, path)


def load_training_checkpoint(path: str, net: PolicyValueNet, optimizer, scheduler,
                             rng: random.Random, config: AZConfig, buffer: ReplayBuffer,
                             selfplay_dir: str, logger: logging.Logger,
                             provenance_out: dict | None = None,
                             diagnostics_out: dict | None = None,
                             resume_out: dict | None = None) -> tuple[TrainingState, list[dict], bool]:
    """完整检查点存在却损坏/不兼容时拒绝降级；从 CPU 加载以保留 RNG tensor 类型。"""
    restore_started = time.perf_counter()
    logger.info("开始恢复完整检查点: %s", path)
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        expected = dict(format="jieqi-training", version=TRAINING_CHECKPOINT_VERSION,
                        ruleset_version=jq.RULESET_VERSION, data_version=DATA_VERSION,
                        search_version=SEARCH_VERSION, network=network_spec(net))
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"{key} 不兼容: {saved.get(key)} != {value}")
        previous = saved["config"]
        old_capacity = previous["replay_buffer_size"]
        new_capacity = config.replay_buffer_size
        growing = new_capacity != old_capacity
        if growing:
            if not (0 < old_capacity < new_capacity):
                raise ValueError("replay_buffer_size 只支持从正容量扩容；缩容请在新目录热启动")
            if not config.allow_buffer_growth:
                raise ValueError("完整续训要求 replay_buffer_size 不变；扩容需显式 --allow-buffer-growth")
        for key in ("augment_mirror", "augment_flip"):
            if previous[key] != getattr(config, key):
                raise ValueError(f"完整续训要求 {key} 不变；改变缓冲/增强请在新目录热启动")
        optimizer_keys = ["learning_rate", "weight_decay", "lr_schedule"]
        if config.lr_schedule == "cosine" or previous["lr_schedule"] == "cosine":
            optimizer_keys += ["iterations", "lr_min"]
        changed = [key for key in optimizer_keys if previous[key] != getattr(config, key)]
        if changed and not config.reset_optimizer:
            raise ValueError(f"优化/调度配置已改变 {changed}；显式 --reset-optimizer 才会使用新配置")
        state = TrainingState(**saved["train_state"])
        if state.ruleset_version != jq.RULESET_VERSION or state.iteration < 0 or state.total_games < 0:
            raise ValueError("训练计数/规则版本无效")
        if saved["scheduler"] is not None and saved["scheduler"]["last_epoch"] != state.iteration:
            raise ValueError("调度器与已完成轮计数不一致")
        lineage = saved.get("provenance")  # Earlier full checkpoints remain compatible.
        if lineage is not None:
            rr.validate_training_lineage(lineage, previous, saved["network"],
                                         dict(ruleset=jq.RULESET_VERSION, data=DATA_VERSION, search=SEARCH_VERSION),
                                         state.iteration, saved["replay_files"])
        diagnostics = saved.get("diagnostics")  # Earlier checkpoints have no fixed holdout.
        diagnostic_samples = {}
        if diagnostics is not None:
            td.validate_state(diagnostics, state.iteration, lineage["model_sha256"] if lineage else None)
            if (lineage and lineage.get("attempt") and diagnostics.get("latest") and
                    diagnostics["latest"]["provenance"] != lineage["attempt"]):
                raise ValueError("诊断报告与检查点尝试来源不一致")
            if config.diagnostic_games not in (0, diagnostics["dataset"]["requested_games"]):
                raise ValueError("已有固定留出局数不可改变；请保留 --diagnostic-games 或省略该参数自动恢复")
            diagnostic_samples = td.load_fixed_sets(diagnostics, config.data_dir, load_compact_payload,
                                                    jq.RULESET_VERSION)
        records = saved["replay_files"]
        # 先按旧容量还原并校验，验证完成后才扩容，不能补回历史已淘汰的前缀。
        if growing:
            buffer.capacity = old_capacity
        # 只读已提交轮次引用的文件；同目录中断轮次/外部文件不会混入。
        replay_started = time.perf_counter()
        logger.info("开始重建训练缓冲: 文件=%d 目标样本=%d", len(records), saved["buffer_size"])
        with _replay_restore_gc():
            for index, record in enumerate(records, 1):
                replay_record_size(record)
                name = record["name"]
                if not name or os.path.basename(name) != name or not name.endswith(SELFPLAY_EXT):
                    raise ValueError("重放清单含无效文件名")
                logger.info("恢复缓冲 [%d/%d]: %s", index, len(records), name)
                file_started = time.perf_counter()
                source = os.path.join(selfplay_dir, name)
                if replay_file_record(source, record["augmented_count"])["sha256"] != record["sha256"]:
                    raise ValueError(f"重放文件内容改变: {name}")
                samples, meta = load_compact_payload(source)
                if int(meta.get("ruleset_version", -1)) != jq.RULESET_VERSION:
                    raise ValueError(f"重放规则版本不兼容: {name}")
                loaded_at = time.perf_counter()
                count = add_saved_samples(buffer, samples, meta, config.augment_mirror, config.augment_flip,
                                          skip_augmented=record.get("skip_augmented", 0))
                if count != record["augmented_count"]:
                    raise ValueError(f"重放样本数不匹配: {name}")
                logger.info("恢复缓冲 [%d/%d] 完成: 新增=%d buffer=%d/%d 读取校验=%.1fs 增强插入=%.1fs",
                            index, len(records), count, len(buffer), buffer.capacity,
                            loaded_at - file_started, time.perf_counter() - loaded_at)
                del samples, meta  # 回收已淘汰的原始块，不带入下一份文件或统一 GC。
            logger.info("训练缓冲重建完毕，整理内存。")
        logger.info("训练缓冲恢复完成: 样本=%d 用时=%.1fs", len(buffer), time.perf_counter() - replay_started)
        if len(buffer) != saved["buffer_size"]:
            raise ValueError("重建缓冲大小与检查点不一致")
        if growing:
            records = replay_files_for_buffer(records, len(buffer))
            buffer.capacity = new_capacity
        net.load_state_dict(saved["model"])  # 完整续训要求形状一致；旧编码走纯权重兼容路径。
        if lineage is not None and lineage["model_sha256"] != rr.model_hash(net):
            raise ValueError("来源记录与模型内容不一致")
        if diagnostics is not None:
            td.validate_state(diagnostics, state.iteration, rr.model_hash(net))
        if not config.reset_optimizer:
            if saved["optimizer_type"] != type(optimizer).__name__ or saved["scheduler_type"] != (type(scheduler).__name__ if scheduler else None):
                raise ValueError("优化器/调度器类型不一致")
            optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
        else:
            logger.warning("显式重置优化器/调度器：保留模型、轮计数、缓冲与 RNG，不是完整优化状态续训。")
        # 构造网络、加载数据会发生在恢复 RNG 之前，避免吞掉已保存的下一次随机抽样。
        restore_training_rng(saved["rng"], rng, logger)
        if provenance_out is not None and lineage is not None:
            provenance_out.update(lineage)
        if diagnostics_out is not None:
            diagnostics_out.update(state=diagnostics, samples=diagnostic_samples)
        if growing:
            growth = dict(old_capacity=old_capacity, new_capacity=new_capacity, retained_samples=len(buffer))
            if resume_out is not None:
                resume_out["buffer_growth"] = growth
            logger.warning("显式扩容训练缓冲: %d -> %d，保留原 FIFO 样本=%d；不回填更旧数据，后续新数据逐步填满。"
                           "已恢复 RNG；扩容后的训练轨迹会改变。", old_capacity, new_capacity, len(buffer))
    except Exception as exc:
        raise RuntimeError(f"完整检查点恢复失败: {path} ({exc})；拒绝静默降级到纯权重。") from exc
    logger.info("完整检查点已恢复: iter=%d buffer=%d lr=%.6g (%s) 用时=%.1fs",
                state.iteration, len(buffer), optimizer.param_groups[0]["lr"], path,
                time.perf_counter() - restore_started)
    other_changes = [key for key in previous if key not in optimizer_keys and key not in
                     ("model_dir", "data_dir", "log_dir", "resume", "reset_optimizer", "run_iterations",
                      "diagnostic_batch_size", "allow_buffer_growth", "replay_buffer_size")
                     and previous[key] != getattr(config, key, None)]
    if other_changes:
        logger.warning("续训运行配置改变: %s；后续轨迹不与原配置逐位等价。", other_changes)
    return state, records, bool(saved["best_updated"])


def prepare_training_eval(config: AZConfig) -> dict | None:
    """在生成数据/修复兼容导出之前校验配置和冻结池，错误不回退 RIS。"""
    if (config.eval_games < 0 or min(config.eval_interval, config.eval_mcts_simulations,
                                   config.eval_leaf_batch_size, config.eval_workers,
                                   config.eval_max_plies, config.ris_iterations) < 1):
        raise ValueError("评估局数不能为负，间隔/预算/worker/最大步数必须为正")
    if config.eval_pool and config.legacy_ris_best:
        raise ValueError("--eval-pool 不能与 --legacy-ris-best 合用")
    if config.eval_seed < 0:
        raise ValueError("--eval-seed 不能为负")
    if config.eval_pool:
        from tools.eval import load_opponent_pool, validate_pool_search
        pool = load_opponent_pool(config.eval_pool)
        validate_pool_search(pool, config.eval_mcts_simulations, config.eval_leaf_batch_size)
        return pool
    return None


def evaluate_training_pool(net, config: AZConfig, device, rng, pool: dict, iteration: int,
                           provenance: dict, logger) -> dict:
    """复用独立配对协议；隔离网络/全局 RNG，记录当前未提交训练 attempt。"""
    saved_rng = capture_training_rng(rng)
    threads = torch.get_num_threads()
    try:
        from tools import eval as evaluator
        if rr.file_record(pool["source"]["path"]) != pool["source"]:
            raise RuntimeError("训练期间对手池清单改变，请用新运行记录切换基准")
        eval_device = torch.device(config.eval_device) if torch.cuda.is_available() else torch.device("cpu")
        # 不移动训练网络或其 optimizer 引用；串行/多进程均使用相同评估设备。
        eval_net = copy.deepcopy(net).to(eval_device).eval()
        torch.set_num_threads(1)
        out_path = os.path.join(config.data_dir, "eval", f"pool_iter{iteration:04d}_{rr.new_id()}.jsonl")
        summaries = evaluator.evaluate(
            eval_net, eval_device, pool["opponents"], config.eval_games,
            config.eval_mcts_simulations, config.eval_seed, config.eval_max_plies, out_path,
            quiet=True, workers=config.eval_workers, chance_nodes=config.chance_nodes,
            chance_k=config.chance_k, leaf_batch_size=config.eval_leaf_batch_size,
            expected_opponent_sources=pool["sources"],
            context=dict(pool=pool, training_attempt=provenance),
        )
        record = dict(type="train_eval_pool", iteration=iteration, pool_id=pool["id"],
                      eval_protocol_version=EVAL_PROTOCOL_VERSION, statistics_version=2,
                      selection_policy="report_only", seed=config.eval_seed,
                      mcts_simulations=config.eval_mcts_simulations, leaf_batch_size=config.eval_leaf_batch_size,
                      output=rr.file_record(out_path), opponents=summaries, provenance=provenance)
        with open(os.path.join(config.data_dir, "eval", "train_eval.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        for stats in summaries:
            logger.info("历史池评估: iter=%d opponent=%s W/D/L=%d/%d/%d score=%s 配对CI95=%s 有效对=%d 剔除对=%d 退化=%s",
                        iteration, stats["opponent"], stats["wins"], stats["draws"], stats["losses"],
                        f'{stats["score"]:.3f}' if stats["pairs_used"] else "无有效配对",
                        stats["bootstrap_ci95"], stats["pairs_used"],
                        stats["pairs_excluded"], stats["bootstrap_degenerate"])
        logger.info("历史池逐局结果: %s；仅报告，不自动晋升 best.pt", out_path)
        return record
    finally:
        torch.set_num_threads(threads)
        restore_training_rng(saved_rng, rng, logger)


def training_loop(config: AZConfig) -> None:
    """
    训练入口：
    - 使用最新模型自博弈生成数据
    - 断点续训：优先完整检查点；旧纯权重按兼容路径加载并明确告警
    - 评估：冻结历史池或 RIS 回归，默认仅报告；显式选项兼容旧 RIS 最高分导出
    """
    eval_pool = prepare_training_eval(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(config.log_dir)
    os.makedirs(config.model_dir, exist_ok=True)
    selfplay_dir = os.path.join(config.data_dir, "selfplay")
    state_path = os.path.join(config.model_dir, STATE_FILE_NAME)
    checkpoint_path = os.path.join(config.model_dir, TRAINING_CHECKPOINT_NAME)
    if config.allow_buffer_growth and (not config.resume or not os.path.exists(checkpoint_path)):
        raise ValueError("--allow-buffer-growth 仅用于恢复已有完整检查点，不能与 --no-resume 或纯权重热启动合用")

    net = PolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.iterations),
            eta_min=config.lr_min,
        )
    elif config.lr_schedule == "none":
        scheduler = None
    else:
        raise ValueError("unsupported lr_schedule")
    buffer = ReplayBuffer(config.replay_buffer_size)
    rng = random.Random()

    train_state = TrainingState()
    replay_files: list[dict] = []
    full_resume = False
    previous_lineage: dict = {}
    diagnostic_restore: dict = {}
    resume_changes: dict = {}
    parent_checkpoint = None
    parent_model = None
    if config.resume:
        if os.path.exists(checkpoint_path):
            parent_checkpoint = rr.file_record(checkpoint_path)
            train_state, replay_files, best_updated = load_training_checkpoint(
                checkpoint_path, net, optimizer, scheduler, rng, config, buffer, selfplay_dir, logger,
                provenance_out=previous_lineage,
                diagnostics_out=diagnostic_restore,
                resume_out=resume_changes,
            )
            if rr.file_record(checkpoint_path) != parent_checkpoint:
                raise RuntimeError("恢复期间检查点已改变，拒绝记录不准确的父模型来源")
            full_resume = True
            if previous_lineage:
                # Checkpoint commit may have succeeded immediately before a marker/export failure.
                rr.commit_record(config.data_dir, previous_lineage, parent_checkpoint)
            # 提交后若兼容导出被中断，在开始下一轮前从权威检查点修复。
            if best_updated:
                save_model(net, os.path.join(config.model_dir, MODEL_BEST_NAME))
            save_model(net, os.path.join(config.model_dir, MODEL_LATEST_NAME))
            save_training_state(state_path, train_state)
        else:
            train_state = load_training_state(state_path, logger)
            if train_state.ruleset_version != jq.RULESET_VERSION:
                raise RuntimeError(
                    f"训练状态规则版本不兼容: {train_state.ruleset_version} != {jq.RULESET_VERSION}；"
                    "拒绝在旧目录从头训练。请保留旧产物，在新目录热启动旧权重并生成当前规则数据。"
                )
            else:
                latest_path = os.path.join(config.model_dir, MODEL_LATEST_NAME)
                parent_model = rr.file_record(latest_path) if os.path.exists(latest_path) else None
                if load_model_if_exists(net, latest_path, device):
                    if rr.file_record(latest_path) != parent_model:
                        raise RuntimeError("加载期间父模型已改变，拒绝记录不准确的来源")
                    logger.warning("旧纯权重兼容续训: %s；AdamW 动量/RNG 不可恢复，不是完整续训。", latest_path)
                elif train_state.iteration > 0:
                    raise RuntimeError(
                        f"断点续训失败：{latest_path} 不存在，但训练状态记录 iteration="
                        f"{train_state.iteration}；拒绝以随机权重续训（请恢复模型文件，"
                        f"或在新目录从零重跑）。"
                    )
                loaded = load_selfplay_data(
                    selfplay_dir, buffer, logger, config.max_load_files,
                    augment=config.augment_mirror, augment_flip=config.augment_flip,
                    ruleset_version=jq.RULESET_VERSION, loaded_files=replay_files,
                )
                if loaded > 0:
                    logger.info("已加载自博弈样本: %d", loaded)

    diagnostics = diagnostic_restore.get("state")
    diagnostic_samples = diagnostic_restore.get("samples", {})
    if diagnostics is not None:
        if diagnostics.get("latest") is not None:
            td.write_report(config.data_dir, diagnostics["latest"])
        logger.info("固定诊断已恢复: heldout=%d局/%d样本 reference=%d样本，不重新划分",
                    diagnostics["dataset"]["requested_games"], len(diagnostic_samples["heldout"]),
                    len(diagnostic_samples["training_reference"]))
    elif config.diagnostic_games > 0:
        if config.self_play_games < 2 * config.diagnostic_games:
            raise ValueError("固定留出需要 --self-play-games >= 2 * --diagnostic-games")
        logger.info("固定诊断待建立: 从下一轮新对局按整局留出%d局，旧缓冲全部保持训练用途",
                    config.diagnostic_games)
    end_iteration = min(config.iterations, train_state.iteration + config.run_iterations) if config.run_iterations else config.iterations
    if config.run_iterations:
        logger.info("本次训练范围: iter=%d..%d（最多新增%d轮；cosine 总计划仍为%d）",
                    train_state.iteration, end_iteration - 1, config.run_iterations, config.iterations)
    if train_state.iteration >= config.iterations:
        logger.warning(
            "--iterations=%d 不大于已完成迭代数 %d，本次不会训练新迭代。",
            config.iterations,
            train_state.iteration,
        )
    if scheduler is not None and train_state.iteration > 0 and (not full_resume or config.reset_optimizer):
        fast_forward_scheduler(scheduler, train_state.iteration)
        logger.info(
            "断点续训：LR 调度快进 %d 步至 lr=%.6g",
            train_state.iteration,
            optimizer.param_groups[0]["lr"],
        )

    current_model_sha256 = rr.model_hash(net)
    resume_mode = ("reset_optimizer" if full_resume and config.reset_optimizer else
                   "full" if full_resume else "legacy_weights" if parent_model else "fresh")
    if resume_changes:
        resume_mode = "buffer_growth_reset_optimizer" if config.reset_optimizer else "buffer_growth"
    manifest = rr.create_run(
        config.data_dir, "training", asdict(config), network_spec(net),
        dict(ruleset=jq.RULESET_VERSION, data=DATA_VERSION, search=SEARCH_VERSION),
        dict(resume_mode=resume_mode, checkpoint=parent_checkpoint, model_file=parent_model,
             model_sha256=current_model_sha256,
             run_id=previous_lineage.get("manifest", {}).get("run_id"),
             history="recorded" if previous_lineage else "unrecorded", evaluation_pool=eval_pool,
             **resume_changes),
        device, start_state=asdict(train_state), replay_files=replay_files,
    )
    if config.eval_games > 0:
        logger.info("评估策略: %s；模型选择=%s", eval_pool["id"] if eval_pool else "RIS 固定回归（未配置历史池）",
                    "旧 RIS 最高分导出" if config.legacy_ris_best else "仅报告，保留已有 best.pt/best_score")
    logger.info("运行来源: run_id=%s (%s)", manifest["run_id"], resume_mode)
    logger.info("节点热点后端: %s%s", SEARCH_BACKEND,
                (" (" + SEARCH_BACKEND_ERROR + ")") if SEARCH_BACKEND_ERROR else "")
    logger.info("合法着快路后端: %s%s", jq.LEGAL_BACKEND,
                (" (" + jq.LEGAL_BACKEND_ERROR + ")") if jq.LEGAL_BACKEND_ERROR else "")
    committed_checkpoint = parent_checkpoint
    if train_state.iteration < config.iterations and (not full_resume or config.reset_optimizer or resume_changes):
        # 首次迁移也建立轮前提交点；中断第一轮后不会误读其未提交的数据。
        replay_files = retained_replay_files(replay_files, buffer.capacity)
        bootstrap = dict(manifest=manifest, manifest_sha256=rr.record_hash(manifest), attempt=None,
                         completed_iterations=train_state.iteration, model_sha256=current_model_sha256)
        save_training_checkpoint(checkpoint_path, net, optimizer, scheduler, train_state,
                                 rng, config, replay_files, len(buffer), provenance=bootstrap,
                                 diagnostics=diagnostics)
        committed_checkpoint = rr.file_record(checkpoint_path)
        rr.commit_record(config.data_dir, bootstrap, committed_checkpoint)

    if config.chance_nodes:
        if config.lazy_chance_allocation:
            logger.info("搜索：暗子揭示机会节点已启用 (完整揭示支持，按需分配对象)")
        elif config.chance_k > 0:
            logger.info("搜索：暗子揭示机会节点已启用 (chance_k=%d，初始预建上限，遗漏按需补建)", config.chance_k)
        else:
            logger.info("搜索：暗子揭示机会节点已启用 (全枚举揭示候选)")
    logger.info("搜索：批内叶评估去重%s", "已启用" if config.leaf_dedup else "已关闭")
    if config.pcr_enabled:
        logger.info(
            "搜索：PCR 已启用 (fast=%d / full=%d, p_full=%.2f；fast 类样本策略目标权重=0)",
            config.pcr_fast_sims,
            config.mcts_simulations,
            config.pcr_full_prob,
        )

    for iteration in range(train_state.iteration, end_iteration):
        logger.info("迭代开始: iter=%d", iteration)
        attempt = rr.start_attempt(config.data_dir, manifest, iteration, current_model_sha256,
                                   committed_checkpoint)
        data_file = None
        net.eval()
        selfplay_start = time.time()
        if config.strong_self_play_interval > 0:
            strong_flags = [
                (idx + 1) % config.strong_self_play_interval == 0
                for idx in range(config.self_play_games)
            ]
        else:
            strong_flags = [False] * config.self_play_games
        results = generate_selfplay_results(
            net,
            config,
            device,
            rng,
            config.self_play_games,
            logger,
            strong_flags=strong_flags,
            iteration=iteration,
        )
        if len(results) != config.self_play_games or any(result.outcome is None for result in results):
            raise RuntimeError("自博弈结果缺失或未结束；本轮不提交训练检查点。")
        selfplay_time = time.time() - selfplay_start
        excluded_games: set[int] = set()
        if diagnostics is None and config.diagnostic_games > 0:
            diagnostics, diagnostic_samples = td.create_fixed_sets(
                results, strong_flags, config.diagnostic_games, attempt, config.data_dir,
                save_compact_samples, jq.RULESET_VERSION,
            )
            excluded_games = set(diagnostics["dataset"]["sets"]["heldout"]["game_indices"])
            logger.info("固定诊断已建立: 留出%d整局/%d原始样本（仅本轮排除）；参照%d样本参与训练",
                        len(excluded_games), len(diagnostic_samples["heldout"]),
                        len(diagnostic_samples["training_reference"]))
        outcome_counts = {
            jq.RESULT_RED_WIN: 0,
            jq.RESULT_BLACK_WIN: 0,
            jq.RESULT_DRAW: 0,
            None: 0,
        }
        total_moves = 0
        iteration_samples: list[CompactSample] = []
        added_samples = 0
        training_indices = [i for i in range(len(results)) if i not in excluded_games]
        for i, result in enumerate(results):
            if i not in excluded_games:
                iteration_samples.extend(result.samples)
                samples = augment_samples(
                    result.samples,
                    mirror=config.augment_mirror,
                    flip=config.augment_flip,
                )
                buffer.add_game(samples)
                added_samples += len(samples)
            train_state.total_games += 1
            total_moves += result.move_count
            outcome_counts[result.outcome] = outcome_counts.get(result.outcome, 0) + 1
        strong_games = sum(1 for flag in strong_flags if flag)
        leaf_evals_total = sum(result.leaf_evals_total for result in results)
        leaf_evals_unique = sum(result.leaf_evals_unique for result in results)

        if results:
            avg_moves = total_moves / len(results)
        else:
            avg_moves = 0.0
        logger.info(
            "自博弈汇总: iter=%d games=%d strong=%d avg_moves=%.1f 红=%d 黑=%d 和=%d 未结束=%d time=%.2fs buffer=%d 新增=%d",
            iteration,
            len(results),
            strong_games,
            avg_moves,
            outcome_counts.get(jq.RESULT_RED_WIN, 0),
            outcome_counts.get(jq.RESULT_BLACK_WIN, 0),
            outcome_counts.get(jq.RESULT_DRAW, 0),
            outcome_counts.get(None, 0),
            selfplay_time,
            len(buffer),
            added_samples,
        )
        if config.leaf_dedup and leaf_evals_total > 0:
            saved_ratio = (1.0 - leaf_evals_unique / leaf_evals_total) * 100.0
            logger.info(
                "叶评估去重: iter=%d 唯一=%d 总计=%d 节省=%.2f%%",
                iteration,
                leaf_evals_unique,
                leaf_evals_total,
                saved_ratio,
            )
        if config.pcr_enabled and results:
            policy_full = sum(
                1
                for result in results
                for sample in result.samples
                if float(getattr(sample, "policy_weight", 1.0)) > 0.0
            )
            policy_all = sum(len(result.samples) for result in results)
            logger.info(
                "PCR: iter=%d fast=%d full=%d 策略目标样本 full 占比=%.1f%%",
                iteration,
                config.pcr_fast_sims,
                config.mcts_simulations,
                policy_full / policy_all * 100.0 if policy_all else 0.0,
            )
        if iteration_samples:
            saved_moves = sum(results[i].move_count for i in training_indices)
            saved_outcomes = {outcome: sum(results[i].outcome == outcome for i in training_indices)
                              for outcome in outcome_counts}
            saved_path = save_compact_iteration(
                iteration_samples,
                selfplay_dir,
                iteration,
                meta={
                    "iteration": iteration,
                    "games": len(training_indices),
                    "generated_games": len(results),
                    "game_indices": training_indices,
                    "excluded_holdout_game_indices": sorted(excluded_games),
                    "game_lengths": [len(results[i].samples) for i in training_indices],
                    "moves": saved_moves,
                    "avg_moves": saved_moves / len(training_indices),
                    "generated_moves": total_moves,
                    "ruleset_version": jq.RULESET_VERSION,
                    "provenance": attempt,
                    "outcomes": {
                        "red": saved_outcomes.get(jq.RESULT_RED_WIN, 0),
                        "black": saved_outcomes.get(jq.RESULT_BLACK_WIN, 0),
                        "draw": saved_outcomes.get(jq.RESULT_DRAW, 0),
                        "none": saved_outcomes.get(None, 0),
                    },
                },
            )
            data_file = replay_file_record(saved_path, added_samples)
            replay_files.append(data_file)
            replay_files = retained_replay_files(replay_files, buffer.capacity)

        diagnostic_before = (td.evaluate_sets(net, diagnostic_samples, device, compact_to_tensors,
                                             config.diagnostic_batch_size) if diagnostics else None)
        net.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        steps = 0
        steps_target = compute_train_steps(
            config.train_steps,
            config.batch_size,
            len(buffer),
        )
        if steps_target <= 0:
            logger.warning(
                "训练步数为 0（buffer=%d < batch=%d），本轮跳过训练。",
                len(buffer),
                config.batch_size,
            )
        for _ in range(steps_target):
            if len(buffer) < config.batch_size:
                break
            batch = buffer.sample(config.batch_size)
            loss, policy_loss, value_loss = train_step(net, optimizer, batch, device, config.grad_clip_norm)
            total_loss += loss
            total_policy += policy_loss
            total_value += value_loss
            steps += 1

        if steps > 0:
            logger.info(
                "训练: iter=%d steps=%d loss=%.4f policy=%.4f value=%.4f",
                iteration,
                steps,
                total_loss / steps,
                total_policy / steps,
                total_value / steps,
            )

        # 评估与模型晋升分离；RIS 最高分不是综合棋力证明。
        current_model_sha256 = rr.model_hash(net)
        if diagnostics is not None:
            diagnostic_after = td.evaluate_sets(net, diagnostic_samples, device, compact_to_tensors,
                                                config.diagnostic_batch_size)
            report = td.make_report(diagnostics, attempt, current_model_sha256, diagnostic_before,
                                    diagnostic_after, steps, config.batch_size, len(buffer), added_samples)
            diagnostics["latest"] = report
            td.write_report(config.data_dir, report)
            for role, label in (("training_reference", "参照"), ("heldout", "留出")):
                old, new = diagnostic_before[role], diagnostic_after[role]
                logger.info("固定诊断: iter=%d %s policy_KL=%s -> %s value_MSE=%.6f -> %.6f 目标熵=%s 校准MSE=%.6f",
                            iteration, label,
                            f'{old["policy_kl"]:.6f}' if old["policy_kl"] is not None else "无策略目标",
                            f'{new["policy_kl"]:.6f}' if new["policy_kl"] is not None else "无策略目标",
                            old["value_mse"], new["value_mse"],
                            f'{new["target_entropy"]:.6f}' if new["target_entropy"] is not None else "无策略目标",
                            new["calibration_mse"])
            logger.info("样本复用: iter=%d draws=%d buffer=%d 每个缓冲样本本轮期望抽样=%.3f",
                        iteration, report["training"]["update_draws"], len(buffer),
                        report["training"]["expected_draws_per_buffer_sample"])
        best_updated = False
        evaluation = None
        if config.eval_games > 0 and (iteration + 1) % config.eval_interval == 0:
            net.eval()
            if eval_pool:
                evaluation = evaluate_training_pool(
                    net, config, device, rng, eval_pool, iteration,
                    dict(**attempt, evaluated_model_sha256=current_model_sha256), logger,
                )
            else:
                stats = evaluate_vs_ris(
                    net, config, device, config.eval_games, rng,
                    jsonl_path=os.path.join(config.data_dir, "eval", "train_eval.jsonl"),
                    iteration=iteration,
                    provenance=dict(**attempt, evaluated_model_sha256=current_model_sha256),
                )
                evaluation = dict(**stats, selection_policy="legacy_ris_best" if config.legacy_ris_best else "report_only")
                logger.info("RIS 回归: iter=%d wins=%d draws=%d losses=%d score=%.3f",
                            iteration, stats["wins"], stats["draws"], stats["losses"], stats["score"])
                if config.legacy_ris_best and stats["score"] > train_state.best_score:
                    train_state.best_score = stats["score"]
                    best_updated = True

        # 先提交单文件完整检查点；纯权重/JSON 是兼容导出，失败时不决定恢复轮次。
        train_state.iteration = iteration + 1
        if scheduler is not None:
            scheduler.step()
        lineage = dict(manifest=manifest, manifest_sha256=rr.record_hash(manifest), attempt=attempt,
                       completed_iterations=train_state.iteration, model_sha256=current_model_sha256,
                       data_file=data_file, evaluation=evaluation)
        save_training_checkpoint(checkpoint_path, net, optimizer, scheduler, train_state,
                                 rng, config, replay_files, len(buffer), best_updated=best_updated,
                                 provenance=lineage, diagnostics=diagnostics)
        committed_checkpoint = rr.file_record(checkpoint_path)
        rr.commit_record(config.data_dir, lineage, committed_checkpoint)
        if best_updated:
            save_model(net, os.path.join(config.model_dir, MODEL_BEST_NAME))
            logger.info("更新旧 RIS 最高分模型: score=%.3f", train_state.best_score)
        save_model(net, os.path.join(config.model_dir, MODEL_LATEST_NAME))
        save_training_state(state_path, train_state)
        logger.info(
            "学习率: iter=%d lr=%.6f",
            iteration,
            optimizer.param_groups[0]["lr"],
        )
        logger.info(
            "迭代结束: iter=%d total_games=%d legacy_ris_best_score=%.3f",
            iteration,
            train_state.total_games,
            train_state.best_score,
        )


def parse_args(argv: list[str] | None = None) -> AZConfig:
    """解析命令行参数并覆盖 AZConfig 默认值。"""
    parser = argparse.ArgumentParser(description="揭棋 AlphaZero 训练入口")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--run-iterations", type=int,
                        help="本次最多新增几轮；0=训练到 --iterations，不改变 cosine 总计划")
    parser.add_argument("--diagnostic-games", type=int,
                        help="从下一轮新对局一次性按整局留出 N 局；默认0，已有固定集自动恢复且不重划")
    parser.add_argument("--diagnostic-batch-size", type=int, help="固定诊断推理 batch，默认128")
    parser.add_argument("--self-play-games", type=int)
    parser.add_argument("--self-play-workers", type=int)
    parser.add_argument("--self-play-log-interval", type=int)
    parser.add_argument("--mcts-simulations", type=int)
    parser.add_argument("--leaf-batch-size", type=int)
    parser.add_argument("--virtual-loss", type=float)
    parser.add_argument(
        "--no-leaf-dedup",
        action="store_true",
        help="关闭批内叶评估去重（默认开启；关闭后同输入会重复做网络前向）",
    )
    parser.add_argument(
        "--chance-nodes",
        dest="chance_nodes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="暗子揭示机会节点（默认开启；--no-chance-nodes 改为按采样公开结果惰性建树）",
    )
    parser.add_argument(
        "--chance-k",
        dest="chance_k",
        type=int,
        help="机会分支初始预建上限；0=完整支持（默认按需分配对象），>0 预建至多 k 类并按需补建遗漏结果",
    )
    parser.add_argument("--lazy-chance-allocation", action=argparse.BooleanOptionalAction, default=None,
                        help="on/k=0 时默认按需分配揭示节点对象；--no-lazy-chance-allocation 恢复全量预建；"
                        "显式开启要求 --chance-nodes --chance-k 0，完整支持/采样不变")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--replay-buffer-size", dest="replay_buffer_size", type=int)
    parser.add_argument("--allow-buffer-growth", action="store_true",
                        help="完整检查点续训时显式允许增大 replay buffer；精确保留旧样本与优化状态，由新数据填满")
    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        help="基础学习率（默认 5e-4）；完整检查点续训改变此值需 --reset-optimizer",
    )
    parser.add_argument(
        "--lr-schedule",
        dest="lr_schedule",
        choices=("cosine", "none"),
        help="学习率调度：cosine（默认，T_max=iterations）或 none（恒定 --lr）",
    )
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-games", type=int)
    parser.add_argument("--eval-mcts-simulations", type=int)
    parser.add_argument("--eval-leaf-batch-size", type=int, help="评估双方模型的叶批大小，默认16；独立于自博弈 --leaf-batch-size")
    parser.add_argument("--ris-iterations", type=int)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--eval-pool", help="冻结历史模型池 JSON；每个对手各跑 --eval-games 局")
    parser.add_argument("--eval-seed", type=int, help="历史池固定日常种子；不要用作独立晋升验收种子")
    parser.add_argument("--eval-max-plies", type=int, help="历史池每局最大步数；未终局整对剔除，默认600")
    parser.add_argument("--legacy-ris-best", action="store_true",
                        help="显式恢复旧 RIS 最高分更新 best.pt；不可与 --eval-pool 合用")
    parser.add_argument("--strong-self-play-interval", type=int)
    parser.add_argument("--strong-mcts-simulations", type=int)
    parser.add_argument(
        "--pcr",
        dest="pcr_enabled",
        action="store_true",
        default=None,
        help="启用 PCR（playout cap randomization）：普通局面每步按概率用 fast/full 两档搜索",
    )
    parser.add_argument(
        "--pcr-fast-sims",
        dest="pcr_fast_sims",
        type=int,
        help="PCR fast 档搜索次数（默认 100；策略目标不参与训练，仅 value）",
    )
    parser.add_argument(
        "--pcr-full-prob",
        dest="pcr_full_prob",
        type=float,
        help="PCR 每步使用 full 档的概率（默认 0.25；建议不低于 0.2）",
    )
    parser.add_argument(
        "--max-load-files",
        type=int,
        help="续训加载自博弈文件数上限：默认 0=按 replay buffer 容量自适应"
        "（从最新文件回填）；>0=仅加载最近 N 个文件（旧语义，兼容历史配置）",
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--model-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--self-play-device")
    parser.add_argument("--eval-device")
    parser.add_argument("--no-resume", action="store_true", help="不加载已有模型/数据")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="完整检查点续训时显式重建 AdamW/调度（改变 --lr 或 cosine 总轮数时使用）；保留模型/计数/缓冲/RNG")
    parser.add_argument("--no-augment-mirror", action="store_true")
    parser.add_argument(
        "--no-augment-flip",
        action="store_true",
        help="关闭楚河汉界翻面增强（行翻面+红黑互换+暗池互换）",
    )
    args = parser.parse_args(argv)
    config = AZConfig()
    for name in (
        "iterations",
        "run_iterations",
        "diagnostic_games",
        "diagnostic_batch_size",
        "self_play_games",
        "self_play_workers",
        "self_play_log_interval",
        "mcts_simulations",
        "leaf_batch_size",
        "virtual_loss",
        "chance_nodes",
        "chance_k",
        "lazy_chance_allocation",
        "replay_buffer_size",
        "train_steps",
        "learning_rate",
        "lr_schedule",
        "eval_interval",
        "eval_games",
        "eval_mcts_simulations",
        "eval_leaf_batch_size",
        "ris_iterations",
        "eval_workers",
        "eval_pool",
        "eval_seed",
        "eval_max_plies",
        "strong_self_play_interval",
        "strong_mcts_simulations",
        "pcr_enabled",
        "pcr_fast_sims",
        "pcr_full_prob",
        "max_load_files",
        "data_dir",
        "model_dir",
        "log_dir",
        "self_play_device",
        "eval_device",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.no_resume:
        config.resume = False
    config.legacy_ris_best = args.legacy_ris_best
    if config.eval_seed < 0:
        parser.error("--eval-seed 不能为负")
    if config.eval_pool and config.legacy_ris_best:
        parser.error("--eval-pool 不能与 --legacy-ris-best 合用")
    if config.eval_games < 0 or min(config.eval_mcts_simulations, config.eval_leaf_batch_size, config.eval_workers,
                                   config.eval_max_plies, config.ris_iterations) < 1:
        parser.error("评估局数不能为负，模拟/worker/最大步数/RIS预算必须为正")
    if args.reset_optimizer:
        config.reset_optimizer = True
    if args.allow_buffer_growth:
        config.allow_buffer_growth = True
    if config.replay_buffer_size <= 0:
        parser.error("--replay-buffer-size 必须为正整数")
    if config.allow_buffer_growth and not config.resume:
        parser.error("--allow-buffer-growth 不能与 --no-resume 合用")
    if args.no_augment_mirror:
        config.augment_mirror = False
    if args.no_augment_flip:
        config.augment_flip = False
    if args.no_leaf_dedup:
        config.leaf_dedup = False
    if config.lr_schedule == "cosine" and config.learning_rate < config.lr_min:
        parser.error("--lr 必须不小于 lr_min (=1e-5)，否则 cosine 曲线会反向爬升")
    if config.chance_k < 0:
        parser.error("--chance-k 不能为负（0 表示全枚举）")
    if args.lazy_chance_allocation is None:
        config.lazy_chance_allocation = config.chance_nodes and config.chance_k == 0
    if config.lazy_chance_allocation and (not config.chance_nodes or config.chance_k != 0):
        parser.error("--lazy-chance-allocation 要求 --chance-nodes --chance-k 0")
    if config.pcr_enabled:
        if config.pcr_fast_sims < 1:
            parser.error("--pcr-fast-sims 必须不小于 1")
        if config.pcr_fast_sims >= config.mcts_simulations:
            parser.error("--pcr-fast-sims 必须小于 --mcts-simulations（full 档）")
        if not 0.0 < config.pcr_full_prob <= 1.0:
            parser.error("--pcr-full-prob 必须在 (0, 1] 区间")
    if config.eval_interval < 1:
        parser.error("--eval-interval 必须不小于 1")
    if config.run_iterations < 0 or config.diagnostic_games < 0:
        parser.error("--run-iterations / --diagnostic-games 不能为负")
    if config.diagnostic_batch_size < 1:
        parser.error("--diagnostic-batch-size 必须不小于1")
    if config.mcts_simulations < 1:
        parser.error("--mcts-simulations 必须不小于 1")
    return config


if __name__ == "__main__":
    install_interrupt_handler()
    try:
        training_loop(parse_args())
    except KeyboardInterrupt:
        # 处理器未生效时的兜底路径（如非主线程收到中断）。
        _sigint_handler(signal.SIGINT, None)
    except SystemExit:
        raise
    except BaseException:
        # 其他异常同样避免在 executor 关闭阶段挂起。
        traceback.print_exc()
        stop_child_processes()
        os._exit(1)
