"""Read-only augmentation and root-prior audit; exit 1 on semantic failures.

Run from the project root: python tools/audit_augmentation.py
Synthetic tensor fixtures test encoding only. Chase fixtures use legal moves
from explicit study positions, with the default rules and no injected streaks.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import azero as az
import ismcts as im
import jieqi as jq


def square_map(square: int, mode: str) -> int:
    if square < 0:
        return -1
    row, col = divmod(square, jq.BOARD_W)
    if mode == "mirror":
        col = jq.BOARD_W - 1 - col
    else:
        row = jq.BOARD_H - 1 - row
    return jq.index(row, col)


def board_map(board, mode):
    mapped = [0] * jq.BOARD_SIZE
    for square, piece in enumerate(board):
        mapped[square_map(square, mode)] = piece ^ 8 if piece and mode == "flip" else piece
    return mapped


def move_map(move, mode):
    return tuple(square_map(square, mode) for square in move)


def state_map(state, mode):
    # Chase fixtures start with empty rule history and contain no dark pieces.
    board = board_map(state.board, mode)
    return jq.GameState(
        board, 1 - state.side if mode == "flip" else state.side,
        [row.copy() for row in (state.pool[::-1] if mode == "flip" else state.pool)],
        state.pool_total[::-1] if mode == "flip" else state.pool_total.copy(),
        im.infer_king_pos(board), random.Random(0), state.rules,
    )


def tensor_fixture():
    codes = [jq.make_piece(kind, side, dark)
             for dark in (0, 1) for side in (jq.BLACK, jq.RED)
             for kind in range(1 if not dark else 2, 8)]
    board = [0] * jq.BOARD_SIZE
    for square, piece in zip(random.Random(42).sample(range(jq.BOARD_SIZE), len(codes)), codes):
        board[square] = piece
    info = im.InformationState(
        board=board, side=jq.RED,
        pool_known=[[0, 0, 2, 1, 2, 1, 2, 4], [0, 0, 1, 2, 1, 2, 1, 3]],
        pool_total_known=[10, 8], unknown_removals=[2, 2],
        no_capture_plies=27, consecutive_checks=[2, 5],
        check_anchor=[3, 61], chase_streak=[4, 1],
        chase_target=[8, 72], chase_target_code=[10, 19], chase_anchor=[13, 64],
        position_counts={(jq.RED, tuple(board)): 2,
                         (jq.BLACK, tuple(board_map(board, "mirror"))): 1,
                         (jq.RED, tuple(board_map(board, "flip"))): 3},
    )
    weights = np.arange(1, az.MOVE_SIZE + 1, dtype=np.float64)
    weights /= weights.sum()
    raw = az.raw_sample_from_info(info, az.LEGAL_MOVE_TABLE[:11], weights.tolist(), .375)
    return az.finalize_samples([raw], jq.RESULT_BLACK_WIN)[0]


def expected_tensor(encoded, mode):
    out = (encoded[:, :, ::-1] if mode == "mirror" else encoded[:, ::-1, :]).copy()
    if mode == "mirror":
        return out
    for start, width in ((0, 7), (14, 7), (29, 6), (41, 1), (43, 1),
                         (46, 1), (65, 1), (67, 1), (69, 1), (71, 1), (73, 1)):
        out[start:start + 2 * width] = out[start:start + 2 * width].reshape(
            2, width, jq.BOARD_H, jq.BOARD_W)[::-1].reshape(2 * width, jq.BOARD_H, jq.BOARD_W)
    out[28] = 1 - out[28]
    for channel in [73, 74, *range(az.REPETITION_HISTORY_OFFSET, az.STATE_CHANNELS, 2)]:
        codes = np.rint(out[channel] * 31).astype(np.uint8)
        out[channel] = np.where(codes, codes ^ 8, 0).astype(np.float32) / 31
    for channel in range(az.REPETITION_HISTORY_OFFSET + 1, az.STATE_CHANNELS, 2):
        codes = np.rint(out[channel] * 7).astype(np.uint32)
        out[channel] = np.where(codes, codes ^ 1, 0).astype(np.float32) / 7
    return out


def audit_mapping():
    sample = tensor_fixture()
    encoded, policy, value = az.compact_to_tensors(sample)
    failures = []
    results = {}
    for mode, transform in (("mirror", az.mirror_sample), ("flip", az.flip_vertical_sample)):
        changed = transform(sample)
        actual, changed_policy, changed_value = az.compact_to_tensors(changed)
        permutation = [az.move_to_index(move_map(move, mode)) for move in az.LEGAL_MOVE_TABLE]
        expected_policy = torch.zeros_like(policy)
        expected_policy[permutation] = policy
        checks = {
            "action_bijection_2550": len(set(permutation)) == az.MOVE_SIZE,
            "action_involution": all(permutation[permutation[i]] == i for i in range(az.MOVE_SIZE)),
            "sample_involution": transform(changed) == sample,
            "all_238_tensor_channels": bool(np.allclose(actual.numpy(), expected_tensor(encoded.numpy(), mode), atol=1e-7, rtol=0)),
            "nonuniform_policy": torch.equal(changed_policy, expected_policy),
            "actor_value_unchanged": torch.equal(changed_value, value),
            "policy_weight_unchanged": changed.policy_weight == sample.policy_weight,
        }
        results[mode] = checks
        failures.extend(f"{mode}.{name}" for name, passed in checks.items() if not passed)
    results["commuting_transforms"] = az.mirror_sample(az.flip_vertical_sample(sample)) == az.flip_vertical_sample(az.mirror_sample(sample))
    results["four_group_elements"] = az.augment_samples([sample], True, True) == [
        sample, az.mirror_sample(sample), az.flip_vertical_sample(sample),
        az.flip_vertical_sample(az.mirror_sample(sample)),
    ]
    failures.extend(name for name in ("commuting_transforms", "four_group_elements") if not results[name])
    return dict(checks=results, failures=failures)


def chase_fixture(mode):
    state = jq.GameState.empty(rng=random.Random(0), with_pools=False)
    if mode == "flip":
        pieces = [(9, 4, 1, jq.RED), (0, 3, 1, jq.BLACK), (4, 3, 2, jq.RED),
                  (2, 4, 3, jq.BLACK), (6, 4, 6, jq.BLACK)]
        path = [((4, 3), (4, 4)), ((6, 4), (6, 5))]
        rook_cells, king_cells = ((4, 4), (3, 4)), ((0, 3), (1, 3))
    else:
        pieces = [(9, 4, 1, jq.RED), (0, 5, 1, jq.BLACK), (4, 3, 2, jq.RED),
                  (5, 1, 3, jq.BLACK), (5, 5, 6, jq.BLACK)]
        path = [((4, 3), (5, 3)), ((5, 5), (6, 5))]
        rook_cells, king_cells = ((5, 3), (5, 2)), ((0, 5), (1, 5))
    for row, col, kind, side in pieces:
        state.place_piece(row, col, kind, side, 0)
    state.side = jq.RED
    state.reset_history()
    for turn in range(6):
        a, b = turn % 2, 1 - turn % 2
        path.extend([(rook_cells[a], rook_cells[b]), (king_cells[a], king_cells[b])])
    return state, [(jq.index(*start), jq.index(*end)) for start, end in path]


def audit_chase(mode):
    state, moves = chase_fixture(mode)
    mapped = state_map(state, mode)
    actor = jq.BLACK if mode == "flip" else jq.RED
    trace = []
    first_rule_mismatch = None
    for ply, move in enumerate(moves, 1):
        other_move = move_map(move, mode)
        if (state.game_outcome()[0] is not None or mapped.game_outcome()[0] is not None
                or move not in state.legal_moves() or other_move not in mapped.legal_moves()):
            return dict(fixture_error=f"path stops being legal at ply {ply}")
        state.apply_move(move)
        mapped.apply_move(other_move)
        expected_target = square_map(state.chase_target[jq.RED], mode)
        if first_rule_mismatch is None and (expected_target != mapped.chase_target[actor]
                or state.chase_streak[jq.RED] != mapped.chase_streak[actor]):
            first_rule_mismatch = ply
        trace.append(dict(ply=ply, move=jq.format_move_uci(move),
                          original_streak=state.chase_streak[jq.RED], mapped_streak=mapped.chase_streak[actor]))
    expected_moves = {move_map(move, mode) for move in state.legal_moves()}
    actual_moves = set(mapped.legal_moves())
    return dict(
        all_moves_legal=True, path_plies=len(moves), first_rule_mismatch_ply=first_rule_mismatch,
        original_streak=state.chase_streak[jq.RED], mapped_streak=mapped.chase_streak[actor],
        extra_moves_in_original_coordinates=sorted(jq.format_move_uci(move_map(move, mode)) for move in actual_moves - expected_moves),
        missing_moves_in_original_coordinates=sorted(jq.format_move_uci(move_map(move, mode)) for move in expected_moves - actual_moves),
        legal_sets_equal=expected_moves == actual_moves, trace=trace,
    )


def audit_noise():
    state = jq.GameState.initial(rng=random.Random(0))
    moves = state.legal_moves()
    indices = [az.move_to_index(move) for move in moves]
    prior = torch.full((az.MOVE_SIZE,), 1 / az.MOVE_SIZE)
    epsilon = .25
    mixed = az.apply_dirichlet_noise(prior, moves, .3, epsilon, random.Random(7))
    noise = az.apply_dirichlet_noise(prior, moves, .3, 1., random.Random(7))[indices]
    root = az.PUCTNode(1., state.side)
    root.expand(moves, mixed)
    actual = torch.tensor([root.children[move].prior for move in moves])
    legal_mass = float(prior[indices].sum())
    expected = (1 - epsilon) * prior[indices] / legal_mass + epsilon * noise
    return dict(
        legal_moves=len(moves), legal_prior_mass=legal_mass, configured_epsilon=epsilon,
        old_normalize_after_mix_epsilon=epsilon / ((1 - epsilon) * legal_mass + epsilon),
        maximum_prior_error=float((actual - expected).abs().max()),
        matches_normalize_then_mix=bool(torch.allclose(actual, expected, atol=1e-7, rtol=0)),
    )


def main():
    report = dict(
        ruleset_version=jq.RULESET_VERSION, data_version=az.DATA_VERSION,
        augmentation_mapping=audit_mapping(),
        chase_symmetry={mode: audit_chase(mode) for mode in ("mirror", "flip")},
        root_noise=audit_noise(),
    )
    print(json.dumps(report, indent=2))
    passed = (not report["augmentation_mapping"]["failures"]
              and all(item.get("legal_sets_equal", False) and item.get("first_rule_mismatch_ply") is None
                      for item in report["chase_symmetry"].values())
              and report["root_noise"]["matches_normalize_then_mix"])
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
