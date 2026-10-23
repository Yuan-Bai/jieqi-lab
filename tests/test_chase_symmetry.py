"""Chase ambiguity and rule transitions under all augmentation transforms."""
import random
import unittest

import jieqi as jq


def transform_square(square, mode):
    if square < 0:
        return -1
    row, col = divmod(square, jq.BOARD_W)
    if mode in ("flip", "rotate"):
        row = jq.BOARD_H - 1 - row
    if mode in ("mirror", "rotate"):
        col = jq.BOARD_W - 1 - col
    return jq.index(row, col)


def transform_move(move, mode):
    return tuple(transform_square(square, mode) for square in move)


def transform_state(state, mode):
    swap = mode in ("flip", "rotate")
    order = [1, 0] if swap else [0, 1]

    def board_map(board):
        result = [0] * jq.BOARD_SIZE
        for square, piece in enumerate(board):
            result[transform_square(square, mode)] = piece ^ 8 if piece and swap else piece
        return result

    result = state.clone()
    result.board = board_map(state.board)
    result.side = 1 - state.side if swap else state.side
    result.pool = [state.pool[side].copy() for side in order]
    result.pool_total = [state.pool_total[side] for side in order]
    result.king_pos = [transform_square(state.king_pos[side], mode) for side in order]
    for field in ("consecutive_checks", "chase_streak"):
        setattr(result, field, [getattr(state, field)[side] for side in order])
    for field in ("check_anchor", "chase_anchor", "chase_target"):
        setattr(result, field, [transform_square(getattr(state, field)[side], mode) for side in order])
    result.chase_target_code = [code ^ 8 if code and swap else code
                                for code in (state.chase_target_code[side] for side in order)]
    result.position_counts = {(1 - side if swap else side, tuple(board_map(board))): count
                              for (side, board), count in state.position_counts.items()}
    result.position_key = result._position_key()
    if state.dark_types is not None:
        result.dark_types = {transform_square(square, mode): kind for square, kind in state.dark_types.items()}
    return result


def chase_position(horizontal=False):
    state = jq.GameState.empty(rng=random.Random(0), with_pools=False)
    if horizontal:
        pieces = [(9, 4, 1, jq.RED), (0, 5, 1, jq.BLACK), (4, 3, 2, jq.RED),
                  (5, 1, 3, jq.BLACK), (5, 5, 6, jq.BLACK)]
        first = [((4, 3), (5, 3)), ((5, 5), (6, 5))]
        rook, king = ((5, 3), (5, 2)), ((0, 5), (1, 5))
    else:
        pieces = [(9, 4, 1, jq.RED), (0, 3, 1, jq.BLACK), (4, 3, 2, jq.RED),
                  (2, 4, 3, jq.BLACK), (6, 4, 6, jq.BLACK)]
        first = [((4, 3), (4, 4)), ((6, 4), (6, 5))]
        rook, king = ((4, 4), (3, 4)), ((0, 3), (1, 3))
    for row, col, kind, side in pieces:
        state.place_piece(row, col, kind, side, 0)
    state.reset_history()
    path = list(first)
    for turn in range(6):
        a, b = turn % 2, 1 - turn % 2
        path.extend([(rook[a], rook[b]), (king[a], king[b])])
    moves = [(jq.index(*start), jq.index(*end)) for start, end in path]
    return state, moves, (jq.index(*rook[0]), jq.index(*rook[1]))


class TestChaseSymmetry(unittest.TestCase):
    def assert_state_equal(self, expected, actual):
        for field in ("board", "side", "pool", "pool_total", "king_pos", "no_capture_plies",
                      "consecutive_checks", "check_anchor", "chase_streak", "chase_target",
                      "chase_target_code", "chase_anchor", "position_counts", "position_key"):
            self.assertEqual(getattr(expected, field), getattr(actual, field), field)

    def test_equal_value_new_targets_clear_chase_for_all_transforms(self):
        state, moves, _ = chase_position()
        for mode in ("mirror", "flip", "rotate"):
            with self.subTest(mode=mode):
                original = state.clone()
                mapped = transform_state(state, mode)
                original.apply_move(moves[0])
                mapped.apply_move(transform_move(moves[0], mode))
                self.assertEqual(original.chase_streak[jq.RED], 0)
                self.assertEqual(original.chase_target[jq.RED], -1)
                self.assertEqual(original.chase_target_code[jq.RED], 0)
                self.assertEqual(original.chase_anchor[jq.RED], -1)
                self.assert_state_equal(transform_state(original, mode), mapped)

    def test_unique_highest_target_beats_lower_value_targets(self):
        state, moves, _ = chase_position()
        victim = jq.index(2, 4)
        state.set_piece(victim, jq.make_piece(jq.TYPE_ROOK, jq.BLACK, 0))
        state.reset_history()
        state.apply_move(moves[0])
        self.assertEqual(state.chase_streak[jq.RED], 1)
        self.assertEqual(state.chase_target[jq.RED], victim)

    def test_existing_target_continues_despite_new_equal_value_target(self):
        state, moves, _ = chase_position()
        state.chase_streak[jq.RED] = jq.LONG_CHASE_ROUNDS - 1
        state.chase_anchor[jq.RED] = moves[0][0]
        state.chase_target[jq.RED] = jq.index(2, 4)
        state.chase_target_code[jq.RED] = state.board[jq.index(2, 4)]
        for mode in ("mirror", "flip", "rotate"):
            with self.subTest(mode=mode):
                original = state.clone()
                mapped = transform_state(state, mode)
                original.apply_move(moves[0])
                mapped.apply_move(transform_move(moves[0], mode))
                self.assertEqual(original.chase_streak[jq.RED], jq.LONG_CHASE_ROUNDS)
                self.assertEqual(original.chase_target[jq.RED], jq.index(2, 4))
                self.assert_state_equal(transform_state(original, mode), mapped)

    def test_counterexamples_remain_symmetric_through_actual_forbidden_moves(self):
        for horizontal in (False, True):
            for mode in ("mirror", "flip", "rotate"):
                with self.subTest(horizontal=horizontal, mode=mode):
                    state, path, continuing = chase_position(horizontal)
                    mapped = transform_state(state, mode)
                    for ply, move in enumerate(path, 1):
                        self.assertIsNone(state.game_outcome()[0])
                        self.assertIsNone(mapped.game_outcome()[0])
                        self.assertIn(move, state.legal_moves())
                        other_move = transform_move(move, mode)
                        self.assertIn(other_move, mapped.legal_moves())
                        before, other_before = state.clone(), mapped.clone()
                        undo = state.apply_move(move)
                        other_undo = mapped.apply_move(other_move)
                        self.assert_state_equal(transform_state(state, mode), mapped)
                        self.assertEqual({transform_move(m, mode) for m in state.legal_moves()}, set(mapped.legal_moves()))
                        expected_outcome, reason = state.game_outcome()
                        if expected_outcome is not None and mode in ("flip", "rotate"):
                            expected_outcome = -expected_outcome
                        self.assertEqual((expected_outcome, reason), mapped.game_outcome())
                        after, other_after = state.clone(), mapped.clone()
                        state.undo_move(undo)
                        mapped.undo_move(other_undo)
                        self.assert_state_equal(before, state)
                        self.assert_state_equal(other_before, mapped)
                        self.assert_state_equal(transform_state(state, mode), mapped)
                        state, mapped = after, other_after
                        if ply == 12:
                            self.assertEqual(state.chase_streak[jq.RED], 5)
                    self.assertEqual(state.chase_streak[jq.RED], jq.LONG_CHASE_ROUNDS)
                    self.assertNotIn(continuing, state.legal_moves())
                    self.assertNotIn(transform_move(continuing, mode), mapped.legal_moves())

    def test_double_discovered_check_has_no_arbitrary_anchor(self):
        state = jq.GameState.empty(with_pools=False)
        for row, col, kind, side in [(0, 4, 1, jq.BLACK), (9, 4, 1, jq.RED),
                                     (4, 4, 2, jq.RED), (6, 4, 6, jq.RED),
                                     (2, 4, 3, jq.RED)]:
            state.place_piece(row, col, kind, side, 0)
        state.reset_history()
        move = (jq.index(2, 4), jq.index(4, 3))
        self.assertIn(move, state.legal_moves())
        for mode in ("mirror", "flip", "rotate"):
            with self.subTest(mode=mode):
                original, mapped = state.clone(), transform_state(state, mode)
                undo = original.apply_move(move)
                mapped.apply_move(transform_move(move, mode))
                self.assertEqual(original.consecutive_checks[jq.RED], 1)
                self.assertEqual(original.check_anchor[jq.RED], -1)
                self.assert_state_equal(transform_state(original, mode), mapped)
                original.undo_move(undo)
                self.assert_state_equal(state, original)

    def test_chase_does_not_transfer_to_identical_victim(self):
        state = jq.GameState.empty(with_pools=False)
        for row, col, kind, side in [(9, 4, 1, jq.RED), (0, 3, 1, jq.BLACK),
                                     (1, 5, 2, jq.RED), (2, 4, 2, jq.BLACK),
                                     (4, 5, 2, jq.BLACK), (7, 4, 7, jq.RED)]:
            state.place_piece(row, col, kind, side, 0)
        state.reset_history()
        path = [(jq.index(1, 5), jq.index(1, 4)),
                (jq.index(2, 4), jq.index(2, 3)),
                (jq.index(1, 4), jq.index(1, 5))]
        for mode in ("mirror", "flip", "rotate"):
            original, mapped = state.clone(), transform_state(state, mode)
            for move in path:
                self.assertIn(move, original.legal_moves())
                before, other_before = original.clone(), mapped.clone()
                undo = original.apply_move(move)
                other_undo = mapped.apply_move(transform_move(move, mode))
                after, other_after = original.clone(), mapped.clone()
                self.assert_state_equal(transform_state(original, mode), mapped)
                original.undo_move(undo)
                mapped.undo_move(other_undo)
                self.assert_state_equal(before, original)
                self.assert_state_equal(other_before, mapped)
                original, mapped = after, other_after
            self.assertEqual(original.chase_streak[jq.RED], 1)
            self.assertEqual(original.chase_target[jq.RED], jq.index(4, 5))

    def test_target_move_updates_exact_identity_and_undo(self):
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 4, 1, jq.RED, 0)
        state.place_piece(0, 3, 1, jq.BLACK, 0)
        state.place_piece(1, 4, 2, jq.RED, 0)
        state.place_piece(2, 4, 2, jq.BLACK, 0)
        state.side = jq.BLACK
        state.reset_history()
        state.chase_streak[jq.RED] = 3
        state.chase_anchor[jq.RED] = jq.index(1, 4)
        state.chase_target[jq.RED] = jq.index(2, 4)
        state.chase_target_code[jq.RED] = state.board[jq.index(2, 4)]
        before = state.clone()
        undo = state.apply_move((jq.index(2, 4), jq.index(2, 5)))
        self.assertEqual(state.chase_target[jq.RED], jq.index(2, 5))
        state.undo_move(undo)
        self.assert_state_equal(before, state)

    def test_capturing_chaser_clears_chase_and_undo(self):
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 4, 1, jq.RED, 0)
        state.place_piece(0, 3, 1, jq.BLACK, 0)
        state.place_piece(1, 4, 2, jq.RED, 0)
        state.place_piece(2, 4, 2, jq.BLACK, 0)
        state.side = jq.BLACK
        state.reset_history()
        state.chase_streak[jq.RED] = 3
        state.chase_anchor[jq.RED] = jq.index(1, 4)
        state.chase_target[jq.RED] = jq.index(2, 4)
        state.chase_target_code[jq.RED] = state.board[jq.index(2, 4)]
        before = state.clone()
        undo = state.apply_move((jq.index(2, 4), jq.index(1, 4)))
        self.assertEqual(state.chase_streak[jq.RED], 0)
        self.assertEqual(state.chase_target[jq.RED], -1)
        state.undo_move(undo)
        self.assert_state_equal(before, state)

    def test_revealing_target_updates_code_or_clears_ineligible_chase(self):
        for revealed in (jq.TYPE_CANNON, jq.TYPE_PAWN):
            for mode in ("mirror", "flip", "rotate"):
                with self.subTest(revealed=revealed, mode=mode):
                    state = jq.GameState.empty(with_pools=False)
                    state.place_piece(9, 4, 1, jq.RED, 0)
                    state.place_piece(0, 3, 1, jq.BLACK, 0)
                    state.place_piece(1, 4, 2, jq.RED, 0)
                    state.place_piece(2, 4, 2, jq.BLACK, 1)
                    state.place_piece(7, 4, 7, jq.RED, 0)
                    state.pool[jq.BLACK][revealed] = 1
                    state.pool_total[jq.BLACK] = 1
                    state.side = jq.BLACK
                    state.reset_history()
                    state.chase_streak[jq.RED] = 3
                    state.chase_anchor[jq.RED] = jq.index(1, 4)
                    state.chase_target[jq.RED] = jq.index(2, 4)
                    state.chase_target_code[jq.RED] = state.board[jq.index(2, 4)]
                    move = (jq.index(2, 4), jq.index(2, 5))
                    before, mapped = state.clone(), transform_state(state, mode)
                    other_before = mapped.clone()
                    self.assertIn(move, state.legal_moves())
                    undo = state.apply_move(move, reveal_choice=revealed)
                    other_undo = mapped.apply_move(transform_move(move, mode), reveal_choice=revealed)
                    if revealed == jq.TYPE_PAWN:
                        self.assertEqual(state.chase_streak[jq.RED], 0)
                    else:
                        self.assertEqual(state.chase_target_code[jq.RED], jq.make_piece(revealed, jq.BLACK, 0))
                        self.assertEqual(state.chase_streak[jq.RED], 3)
                    self.assert_state_equal(transform_state(state, mode), mapped)
                    state.undo_move(undo)
                    mapped.undo_move(other_undo)
                    self.assert_state_equal(before, state)
                    self.assert_state_equal(other_before, mapped)


if __name__ == "__main__":
    unittest.main()
