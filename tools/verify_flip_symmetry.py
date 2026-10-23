# -*- coding: utf-8 -*-
"""楚河汉界翻面数据增强的合法性检验（ADR-0011 证据）：行翻面 σ(r,c)=(9-r,c) + 红黑互换 + 暗池互换。

做法：成对随机对局（原局 s 与翻转局 s2 同步走映射着法），每步对比：

1. `legal_moves` 的 σ 映射集合；
2. `game_outcome`（结果颜色互换、原因不变）；
3. `T_state(s)` 与 `s2` 的全字段（棋盘/轮次/暗池/王位/规则状态/`position_counts`）；
4. `T_info(info(s))` 与 `info(s2)`（BeliefModel 观察结构）；
5. 样本级：`azero.flip_vertical_sample`（正式实现）与翻转局样本逐字段一致；
6. `encode_state` 当前全部通道张量一致（含完整重复历史）。

预期：所有比较均零偏差。棋盘、历史和非均匀策略取自同一走子前时刻。
固定长捉路径参与验收，任何规则字段分歧都判失败。

用法（项目根目录）::

    python tools/verify_flip_symmetry.py --games 200

退出码：0=通过，1=任何层级存在分歧。
"""
import argparse
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import azero
import ismcts as im
import jieqi as jq

BW, BH = jq.BOARD_W, jq.BOARD_H
FIELD_NAMES = [
    "board",
    "side",
    "pool",
    "pool_total",
    "king_pos",
    "no_capture",
    "checks",
    "check_anchor",
    "chase_streak",
    "chase_target",
    "chase_target_code",
    "chase_anchor",
    "position_key",
    "position_counts",
]


def fidx(index: int) -> int:
    """行翻面后的格子索引（列不变）。"""
    r, c = divmod(index, BW)
    return (BH - 1 - r) * BW + c


def fmap(index: int) -> int:
    return fidx(index) if index >= 0 else -1


def swap_color_code(piece: int) -> int:
    return (piece ^ (1 << jq.COLOR_SHIFT)) if piece else 0


def swap_color_slot(values: list) -> list:
    return [values[1], values[0]]


def t_board(board: list[int]) -> list[int]:
    out = [0] * jq.BOARD_SIZE
    for idx, piece in enumerate(board):
        if piece:
            out[fidx(idx)] = swap_color_code(piece)
    return out


def t_moves(moves):
    return [(fidx(a), fidx(b)) for a, b in moves]


def t_state(state: jq.GameState) -> jq.GameState:
    """把引擎局面整体变换到翻转局（仅用于检验对照，不用于训练数据）。"""
    board = t_board(state.board)
    dark_types = None
    if state.dark_types is not None:
        dark_types = {fidx(idx): ptype for idx, ptype in state.dark_types.items()}
    out = jq.GameState(
        board,
        1 - state.side,
        [state.pool[1].copy(), state.pool[0].copy()],
        [state.pool_total[1], state.pool_total[0]],
        im.infer_king_pos(board),
        random.Random(0),
        state.rules,
        dark_types,
    )
    out.no_capture_plies = state.no_capture_plies
    out.consecutive_checks = swap_color_slot(state.consecutive_checks)
    out.check_anchor = [fmap(state.check_anchor[1]), fmap(state.check_anchor[0])]
    out.chase_streak = swap_color_slot(state.chase_streak)
    out.chase_target = [fmap(state.chase_target[1]), fmap(state.chase_target[0])]
    out.chase_target_code = [
        swap_color_code(state.chase_target_code[1]),
        swap_color_code(state.chase_target_code[0]),
    ]
    out.chase_anchor = [fmap(state.chase_anchor[1]), fmap(state.chase_anchor[0])]
    out.position_counts = {}
    for (side, board_tuple), count in state.position_counts.items():
        out.position_counts[(1 - side, tuple(t_board(list(board_tuple))))] = count
    out.position_key = out._position_key()
    return out


def t_info(info: im.InformationState) -> im.InformationState:
    """把观察者信息态变换到翻转局对应信息态（仅用于检验对照）。"""
    out = im.InformationState(
        board=t_board(info.board),
        side=1 - info.side,
        pool_known=[info.pool_known[1].copy(), info.pool_known[0].copy()],
        pool_total_known=[info.pool_total_known[1], info.pool_total_known[0]],
        unknown_removals=[info.unknown_removals[1], info.unknown_removals[0]],
        no_capture_plies=info.no_capture_plies,
        consecutive_checks=swap_color_slot(info.consecutive_checks),
        rules=info.rules,
        check_anchor=[fmap(info.check_anchor[1]), fmap(info.check_anchor[0])],
        chase_streak=swap_color_slot(info.chase_streak),
        chase_target=[fmap(info.chase_target[1]), fmap(info.chase_target[0])],
        chase_target_code=[
            swap_color_code(info.chase_target_code[1]),
            swap_color_code(info.chase_target_code[0]),
        ],
        chase_anchor=[fmap(info.chase_anchor[1]), fmap(info.chase_anchor[0])],
        position_counts=None,
        public_pool_known=([info.public_pool_known[1].copy(), info.public_pool_known[0].copy()]
                           if info.public_pool_known is not None else None),
        repetition_history_complete=info.repetition_history_complete,
    )
    if info.position_counts is not None:
        out.position_counts = {}
        for (side, board_tuple), count in info.position_counts.items():
            out.position_counts[(1 - side, tuple(t_board(list(board_tuple))))] = count
    return out


def state_fields(state: jq.GameState):
    return (
        tuple(state.board),
        state.side,
        (tuple(state.pool[0]), tuple(state.pool[1])),
        tuple(state.pool_total),
        tuple(state.king_pos),
        state.no_capture_plies,
        tuple(state.consecutive_checks),
        tuple(state.check_anchor),
        tuple(state.chase_streak),
        tuple(state.chase_target),
        tuple(state.chase_target_code),
        tuple(state.chase_anchor),
        state.position_key,
        tuple(sorted((k[0], k[1], v) for k, v in state.position_counts.items())),
    )


def info_fields(info: im.InformationState):
    position_counts = None
    if info.position_counts is not None:
        position_counts = tuple(sorted((k[0], k[1], v) for k, v in info.position_counts.items()))
    return (
        tuple(info.board),
        info.side,
        (tuple(info.pool_known[0]), tuple(info.pool_known[1])),
        tuple(info.pool_total_known),
        tuple(info.unknown_removals),
        info.no_capture_plies,
        tuple(info.consecutive_checks),
        tuple(info.check_anchor),
        tuple(info.chase_streak),
        tuple(info.chase_target),
        tuple(info.chase_target_code),
        tuple(info.chase_anchor),
        position_counts,
        info.public_pool_known,
        info.repetition_history_complete,
    )


def map_outcome(outcome):
    if outcome is None:
        return None
    result, reason = outcome
    if result == jq.RESULT_RED_WIN:
        result = jq.RESULT_BLACK_WIN
    elif result == jq.RESULT_BLACK_WIN:
        result = jq.RESULT_RED_WIN
    return (result, reason)


def sample_fields_equal(transformed, reference) -> bool:
    return (
        len(transformed.policy_idx) == len(transformed.policy_prob)
        and len(reference.policy_idx) == len(reference.policy_prob)
        and transformed.board == reference.board
        and transformed.side == reference.side
        and transformed.pool_known == reference.pool_known
        and transformed.pool_total_known == reference.pool_total_known
        and transformed.unknown_removals == reference.unknown_removals
        and transformed.no_capture_plies == reference.no_capture_plies
        and transformed.consecutive_checks == reference.consecutive_checks
        and transformed.check_anchor == reference.check_anchor
        and transformed.chase_streak == reference.chase_streak
        and transformed.chase_target == reference.chase_target
        and transformed.chase_target_code == reference.chase_target_code
        and transformed.chase_anchor == reference.chase_anchor
        and transformed.rep_count == reference.rep_count
        and transformed.repetition_history == reference.repetition_history
        and transformed.history == reference.history
        and sorted(zip(transformed.policy_idx, transformed.policy_prob))
        == sorted(zip(reference.policy_idx, reference.policy_prob))
        and abs(transformed.value - reference.value) <= 1e-9
        and transformed.policy_weight == reference.policy_weight
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=200, help="随机对局数（默认 200）")
    parser.add_argument("--max-plies", type=int, default=150, help="每局最大步数（默认 150）")
    parser.add_argument("--seed", type=int, default=1000, help="首个对局种子（默认 1000）")
    args = parser.parse_args(argv)

    stats = {
        "games": 0,
        "plies": 0,
        "legal_mismatch": 0,
        "outcome_mismatch": 0,
        "state_mismatch": {},
        "state_mismatch_total": 0,
        "info_mismatch": 0,
        "sample_mismatch": 0,
        "tensor_mismatch": 0,
        "value_mismatch": 0,
    }
    first = {}

    def note(category: str, seed: int, ply: int, detail: str = "") -> None:
        stats[category] = stats.get(category, 0) + 1
        first.setdefault(category, (seed, ply, detail))

    from tools.audit_augmentation import chase_fixture

    cases = [(f"chase-{mode}", *chase_fixture(mode)) for mode in ("mirror", "flip")]
    cases.extend((args.seed + game, None, None) for game in range(args.games))
    for seed, fixture, path in cases:
        random_seed = seed if isinstance(seed, int) else 0
        play_rng = random.Random(random_seed * 7 + 3)
        state = fixture or jq.GameState.initial(rng=random.Random(random_seed * 2 + 1))
        if fixture is None:
            state.dark_types = jq.materialize_dark_types(random.Random(random_seed * 2 + 2))
        flipped = t_state(state)
        belief = im.BeliefModel.from_game_state(state)
        flipped_belief = im.BeliefModel.from_game_state(flipped)
        history: list[tuple[int, int]] = []
        flipped_history: list[tuple[int, int]] = []
        stats["games"] += 1
        limit = len(path) + 1 if path is not None else args.max_plies
        for ply in range(limit):
            stats["plies"] += 1
            moves = state.legal_moves()
            flipped_moves = flipped.legal_moves()
            if set(t_moves(moves)) != set(flipped_moves):
                note("legal_mismatch", seed, ply, f"moves={len(moves)} flipped={len(flipped_moves)}")
            outcome = state.game_outcome()
            flipped_outcome = flipped.game_outcome()
            if map_outcome(outcome) != flipped_outcome:
                note("outcome_mismatch", seed, ply, f"{outcome} vs {flipped_outcome}")
            if azero.outcome_value_for_player(
                jq.RESULT_RED_WIN, jq.RED
            ) != azero.outcome_value_for_player(jq.RESULT_BLACK_WIN, jq.BLACK):
                stats["value_mismatch"] += 1
            transformed_fields = state_fields(t_state(state))
            flipped_fields = state_fields(flipped)
            if transformed_fields != flipped_fields:
                for name, left, right in zip(FIELD_NAMES, transformed_fields, flipped_fields):
                    if left != right:
                        stats["state_mismatch"][name] = stats["state_mismatch"].get(name, 0) + 1
                note(
                    "state_mismatch_total",
                    seed,
                    ply,
                    next(n for n, a, b in zip(FIELD_NAMES, transformed_fields, flipped_fields) if a != b),
                )
            info = belief.info_state(state, state.side)
            flipped_info = flipped_belief.info_state(flipped, flipped.side)
            if info_fields(t_info(info)) != info_fields(flipped_info):
                note("info_mismatch", seed, ply)

            policy = [0.0] * azero.MOVE_SIZE
            flipped_policy = [0.0] * azero.MOVE_SIZE
            total_weight = len(moves) * (len(moves) + 1) / 2
            for weight, legal_move in enumerate(moves, 1):
                probability = weight / total_weight
                policy[azero.move_to_index(legal_move)] = probability
                flipped_policy[azero.move_to_index(tuple(fidx(sq) for sq in legal_move))] = probability
            sample = azero.finalize_samples(
                [azero.raw_sample_from_info(info, history, policy)], jq.RESULT_RED_WIN
            )[0]
            flipped_sample = azero.finalize_samples(
                [azero.raw_sample_from_info(flipped_info, flipped_history, flipped_policy)],
                jq.RESULT_BLACK_WIN,
            )[0]
            transformed = azero.flip_vertical_sample(sample)
            if not sample_fields_equal(transformed, flipped_sample):
                note("sample_mismatch", seed, ply)

            tensor = azero.encode_state(t_info(info), t_moves(history), flipped_info.side)
            flipped_tensor = azero.encode_state(flipped_info, flipped_history, flipped_info.side)
            if not (tensor == flipped_tensor).all():
                note("tensor_mismatch", seed, ply)

            if outcome[0] is not None or flipped_outcome[0] is not None or not moves:
                break
            if path is not None and ply == len(path):
                break
            move = path[ply] if path is not None else play_rng.choice(moves)
            flipped_move = tuple(fidx(sq) for sq in move)
            if move not in moves or flipped_move not in flipped_moves:
                note("legal_mismatch", seed, ply, "mapped path move is illegal")
                break
            undo = state.apply_move(move, rng=random.Random(0))
            flipped_undo = flipped.apply_move(flipped_move, rng=random.Random(0))
            belief.observe_move(undo)
            flipped_belief.observe_move(flipped_undo)
            history.append(move)
            flipped_history.append(flipped_move)

    print("=== 楚河汉界翻面同构性检验（ADR-0011） ===")
    print(f"对局: {stats['games']}  总步数: {stats['plies']}")
    print(f"legal_moves 不匹配: {stats['legal_mismatch']}")
    print(f"game_outcome 不匹配: {stats['outcome_mismatch']}")
    print(f"value 语义不匹配: {stats['value_mismatch']}")
    print(f"状态字段不匹配: {stats['state_mismatch']}（共 {stats['state_mismatch_total']} ply）")
    print(f"信息态不匹配: {stats['info_mismatch']}")
    print(f"样本不匹配: {stats['sample_mismatch']}")
    print(f"张量不匹配: {stats['tensor_mismatch']}")
    for key, value in first.items():
        print(f"首个 {key}: {value}")

    if any(stats[key] for key in ("legal_mismatch", "outcome_mismatch", "value_mismatch",
                                  "state_mismatch_total", "info_mismatch",
                                  "sample_mismatch", "tensor_mismatch")):
        print("结果: 失败（存在分歧）")
        return 1
    print("结果: 通过（所有层级零偏差）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
