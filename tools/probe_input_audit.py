"""只读审计 238 通道输入编码的算术正确性与归一化边界。

python tools/probe_input_audit.py --games 30 --pluses 200 [--data data/train_fixed_20261001/selfplay]
逐局面校验：棋盘平面、标量平面、历史平面、锚点平面、81 槽重复历史
（槽位集合、顺序、棋盘码、计数解码、空槽为零、完整性标记），并统计
各字段的实际取值范围（重点：重复计数能否超过 REPETITION_LIMIT=3）。
不写任何产物。
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import azero as az
import ismcts as im
import jieqi as jq


def const_value(planes, ch, expected, name, tol=1e-6):
    v = float(planes[ch, 0, 0])
    lo, hi = float(planes[ch].min()), float(planes[ch].max())
    if abs(hi - v) > tol or abs(lo - v) > tol:
        raise AssertionError(f"{name}: plane not constant min={lo} max={hi}")
    if abs(v - expected) > tol:
        raise AssertionError(f"{name}: value {v} != expected {expected}")


def validate_encoded(info, history, ctx, stats):
    planes = az.encode_state_numpy(info, history)
    H, W = jq.BOARD_H, jq.BOARD_W
    assert planes.shape == (az.STATE_CHANNELS, H, W), planes.shape
    assert planes.dtype == np.float32 and np.isfinite(planes).all()

    # 0~27：棋子平面，逐子对应、无重叠。
    count = 0
    for idx, piece in enumerate(info.board):
        if not piece:
            continue
        count += 1
        p_idx = az._piece_plane_index(jq.is_dark(piece), jq.piece_color(piece), jq.piece_type(piece))
        assert planes[p_idx, jq.ROWS[idx], jq.COLS[idx]] == 1.0, ctx
    assert int((planes[:28] > 0).sum()) == count, (ctx, count)
    assert bool((planes[:28].sum(axis=0) <= 1.0).all()), ctx

    # 28：轮到红方。
    const_value(planes, 28, 1.0 if info.side == jq.RED else 0.0, "side")
    # 29~40：暗盒各类剩余；41~42 总数；43~44 未知被吃；45 无吃子；46~47 长将。
    for s in (jq.BLACK, jq.RED):
        for i, p_type in enumerate(jq.POOL_TYPES):
            if jq.POOL_INIT[p_type]:
                const_value(planes, 29 + s * 6 + i, info.pool_known[s][p_type] / jq.POOL_INIT[p_type], f"pool{s},{p_type}")
    const_value(planes, 41, info.pool_total_known[jq.BLACK] / jq.POOL_TOTAL, "pool_total0")
    const_value(planes, 42, info.pool_total_known[jq.RED] / jq.POOL_TOTAL, "pool_total1")
    const_value(planes, 43, info.unknown_removals[jq.BLACK] / jq.POOL_TOTAL, "unk0")
    const_value(planes, 44, info.unknown_removals[jq.RED] / jq.POOL_TOTAL, "unk1")
    const_value(planes, 45, info.no_capture_plies / jq.NO_CAPTURE_LIMIT_TIANTIAN_PLIES, "no_capture")
    const_value(planes, 46, info.consecutive_checks[jq.RED] / (jq.LONG_CHECK_ROUNDS + 1), "checksR")
    const_value(planes, 47, info.consecutive_checks[jq.BLACK] / (jq.LONG_CHECK_ROUNDS + 1), "checksB")

    # 48~63：最近 8 步 from/to。
    recent = history[-az.HISTORY_PLIES:]
    for i, move in enumerate(recent):
        f, t = move
        assert planes[48 + 2 * i, jq.ROWS[f], jq.COLS[f]] == 1.0 and planes[48 + 2 * i].sum() == 1.0, ctx
        assert planes[49 + 2 * i, jq.ROWS[t], jq.COLS[t]] == 1.0 and planes[49 + 2 * i].sum() == 1.0, ctx

    # 64：当前重复计数；65~66 追击计数；67~70 追击目标/发起子；71~72 将军发起子；73~74 目标码。
    rep = az._repetition_count(info)
    const_value(planes, 64, rep / jq.REPETITION_LIMIT, "rep_count")
    const_value(planes, 65, info.chase_streak[jq.RED] / jq.LONG_CHASE_ROUNDS, "chaseR")
    const_value(planes, 66, info.chase_streak[jq.BLACK] / jq.LONG_CHASE_ROUNDS, "chaseB")
    const_value(planes, 73, info.chase_target_code[jq.RED] / 31.0, "target_codeR")
    const_value(planes, 74, info.chase_target_code[jq.BLACK] / 31.0, "target_codeB")
    for i, s in enumerate((jq.RED, jq.BLACK)):
        for ch, field in ((67, info.chase_target), (69, info.chase_anchor)):
            v = field[s]
            if v >= 0:
                assert planes[ch + i].sum() == 1.0 and planes[ch + i, jq.ROWS[v], jq.COLS[v]] == 1.0, ctx
            else:
                assert planes[ch + i].sum() == 0.0, ctx
        v = info.check_anchor[s]
        if v >= 0:
            assert planes[71 + i].sum() == 1.0 and planes[71 + i, jq.ROWS[v], jq.COLS[v]] == 1.0, ctx
        else:
            assert planes[71 + i].sum() == 0.0, ctx

    # 75~237：完整重复历史（81 槽 × 2 通道）。
    entries = az._repetition_entries(info)
    const_value(planes, 75, 1.0 if entries is not None else 0.0, "complete_flag")
    if entries is not None:
        cur = tuple(info.board)
        cur_occ, cur_dark = az._public_material_signature(cur)
        expected = [
            (s, b, c) for (s, b), c in info.position_counts.items()
            if az._public_material_signature(b)[0] <= cur_occ and az._public_material_signature(b)[1] <= cur_dark
        ]
        if (info.side, cur) not in info.position_counts:
            expected.append((info.side, cur, 1))
        got = [(e[0], tuple(e[1]), e[2]) for e in entries]
        assert got == expected, (ctx, "entry set/order mismatch")
        assert len(entries) <= az.REPETITION_HISTORY_SIZE, ctx
        for i, (s, b, c) in enumerate(entries):
            base = az.REPETITION_HISTORY_OFFSET + 2 * i
            codes = np.rint(planes[base] * 31).astype(np.int64)
            assert np.array_equal(codes, np.frombuffer(b, dtype=np.uint8).reshape(H, W)), ctx
            const_value(planes, base + 1, (2 * c + s) / (2 * jq.REPETITION_LIMIT + 1), f"slot_cnt{i}")
            stats["max_slot_count"] = max(stats["max_slot_count"], c)
        for i in range(len(entries), az.REPETITION_HISTORY_SIZE):
            base = az.REPETITION_HISTORY_OFFSET + 2 * i
            assert planes[base].max() == 0.0 and planes[base + 1].max() == 0.0, (ctx, "unused slot not zero")
        for i in range(az.REPETITION_HISTORY_SIZE):
            base = az.REPETITION_HISTORY_OFFSET + 2 * i
            assert (planes[base].max() > 0) == (planes[base + 1].max() > 0), (ctx, "slot occupancy mismatch")
    else:
        assert planes[az.REPETITION_HISTORY_OFFSET:].max() == 0.0, ctx

    # 取值范围统计（重点关注可能 >1 的字段）。
    stats["max_rep_count"] = max(stats["max_rep_count"], rep)
    stats["max_checks"] = max(stats["max_checks"], max(info.consecutive_checks))
    stats["max_chase"] = max(stats["max_chase"], max(info.chase_streak))
    stats["max_no_capture"] = max(stats["max_no_capture"], info.no_capture_plies)
    stats["max_entries"] = max(stats["max_entries"], len(entries or ()))
    stats["max_value"] = max(stats["max_value"], float(planes.max()))
    stats["over_one"] = max(stats["over_one"], float((planes > 1.0 + 1e-6).sum()))
    nonzero = int((planes.any(axis=(1, 2))).sum())
    ply = len(history)
    key = "early" if ply < 20 else ("mid" if ply < 60 else "late")
    stats["nonzero_channels"][key].append(nonzero)
    return planes


def random_game_audit(games: int, max_plies: int, seed: int):
    rng = random.Random(seed)
    stats = {
        "states": 0, "max_rep_count": 0, "max_checks": 0, "max_chase": 0,
        "max_no_capture": 0, "max_entries": 0, "max_slot_count": 0,
        "max_value": 0.0, "over_one": 0,
        "nonzero_channels": {"early": [], "mid": [], "late": []},
    }
    tensors = []
    for _ in range(games):
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        history = []
        for _ in range(max_plies):
            if state.game_outcome()[0] is not None:
                break
            info = belief.info_state(state, state.side)
            planes = validate_encoded(info, history, f"game ply {len(history)}", stats)
            if len(tensors) < 32:
                tensors.append(torch.from_numpy(planes))
            stats["states"] += 1
            move = rng.choice(state.legal_moves())
            undo = state.apply_move(move)
            belief.observe_move(undo)
            history.append(move)
    return stats, tensors


def network_forward_check(tensors):
    net = az.PolicyValueNet()
    net.eval()
    with torch.no_grad():
        x = torch.stack(tensors)
        logits, values = net(x)
    assert logits.shape == (len(tensors), az.MOVE_SIZE), logits.shape
    assert values.shape == (len(tensors),), values.shape
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    assert float(values.abs().max()) <= 1.0 + 1e-5
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(len(tensors)), atol=1e-5)
    return {
        "batch": len(tensors),
        "policy_shape": list(logits.shape),
        "value_shape": list(values.shape),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
        "conv1_params": net.conv.weight.numel(),
        "input_bytes_per_sample": x.element_size() * x[0].numel(),
    }


def scan_training_data(data_dir: str):
    """扫描真实 v3 样本的重复计数与归一化越界情况（只读）。"""
    result = {"files": 0, "samples": 0, "rep_count_over_limit": 0, "max_rep_count": 0,
              "history_count_over_limit": 0, "max_history_count": 0, "max_entries": 0,
              "missing_history": 0}
    for path in sorted(Path(data_dir).glob("**/*.pkl.gz")):
        try:
            samples, _ = az.load_compact_payload(str(path))
        except Exception as exc:  # 旧版本文件直接跳过。
            continue
        result["files"] += 1
        for sample in samples:
            result["samples"] += 1
            rep = int(getattr(sample, "rep_count", 0))
            if rep > jq.REPETITION_LIMIT:
                result["rep_count_over_limit"] += 1
            result["max_rep_count"] = max(result["max_rep_count"], rep)
            blob = getattr(sample, "repetition_history", None)
            if blob is None:
                result["missing_history"] += 1
                continue
            counts = az.unpack_repetition_history(blob)
            result["max_entries"] = max(result["max_entries"], len(counts))
            for count in counts.values():
                if count > jq.REPETITION_LIMIT:
                    result["history_count_over_limit"] += 1
                result["max_history_count"] = max(result["max_history_count"], count)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--pluses", type=int, default=200)
    parser.add_argument("--data", default="")
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    torch.set_num_threads(1)
    stats, tensors = random_game_audit(args.games, args.pluses, args.seed)
    forward = network_forward_check(tensors)
    report = {"channel_layout": {
        "STATE_CHANNELS": az.STATE_CHANNELS,
        "HISTORY_PLIES": az.HISTORY_PLIES,
        "REPETITION_HISTORY_SIZE": az.REPETITION_HISTORY_SIZE,
        "piece_planes": 28, "rule_scalar_planes": 48, "repetition_planes": 2 * az.REPETITION_HISTORY_SIZE,
        "nonzero_by_phase": {k: {
            "mean": round(sum(v) / len(v), 1) if v else None,
            "max": max(v) if v else None,
        } for k, v in stats["nonzero_channels"].items()},
    }, "bounds": {
        "max_rep_count": stats["max_rep_count"],
        "max_checks": stats["max_checks"],
        "max_chase": stats["max_chase"],
        "max_no_capture": stats["max_no_capture"],
        "max_entries": stats["max_entries"],
        "max_encoded_value": round(stats["max_value"], 6),
        "cells_over_one": stats["over_one"],
    }, "forward": forward}
    if args.data:
        report["training_data"] = scan_training_data(args.data)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
