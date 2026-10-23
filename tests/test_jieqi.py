import random
import unittest

import ismcts as im
import jieqi as jq


class TestJieQi(unittest.TestCase):
    def test_initial_setup_pools_and_kings(self):
        state = jq.GameState.initial()
        self.assertEqual(state.side, jq.RED)
        self.assertEqual(state.king_pos[jq.BLACK], jq.index(0, 4))
        self.assertEqual(state.king_pos[jq.RED], jq.index(9, 4))
        self.assertEqual(state.pool_total, [jq.POOL_TOTAL, jq.POOL_TOTAL])

        p = state.board[jq.index(0, 0)]
        self.assertEqual(jq.piece_type(p), jq.TYPE_ROOK)
        self.assertEqual(jq.is_dark(p), 1)

        k = state.board[jq.index(9, 4)]
        self.assertEqual(jq.piece_type(k), jq.TYPE_KING)
        self.assertEqual(jq.is_dark(k), 0)

    def test_dark_piece_reveal_on_move(self):
        state = jq.GameState.initial()
        from_idx = jq.index(6, 0)
        to_idx = jq.index(5, 0)
        undo = state.apply_move((from_idx, to_idx), reveal_choice=jq.TYPE_ROOK)
        piece = state.board[to_idx]
        self.assertEqual(jq.piece_type(piece), jq.TYPE_ROOK)
        self.assertEqual(jq.is_dark(piece), 0)
        self.assertEqual(state.pool[jq.RED][jq.TYPE_ROOK], 1)

        state.undo_move(undo)
        self.assertEqual(state.pool[jq.RED][jq.TYPE_ROOK], 2)

    def test_capture_dark_piece_reveals_pool(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)

        state.place_piece(5, 4, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(5, 6, jq.TYPE_PAWN, jq.BLACK, 1)

        move = (jq.index(5, 4), jq.index(5, 6))
        undo = state.apply_move(move, capture_reveal_choice=jq.TYPE_CANNON)
        self.assertTrue(state.board[jq.index(5, 6)])
        self.assertEqual(state.pool[jq.BLACK][jq.TYPE_CANNON], 1)

        state.undo_move(undo)
        self.assertEqual(state.pool[jq.BLACK][jq.TYPE_CANNON], 2)

    def test_elephant_advisor_cross_river_moves(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)

        e_idx = state.place_piece(6, 2, jq.TYPE_ELEPHANT, jq.RED, 0)
        dark_a_idx = state.place_piece(7, 3, jq.TYPE_ADVISOR, jq.RED, 1)
        bright_a_idx = state.place_piece(5, 3, jq.TYPE_ADVISOR, jq.RED, 0)

        moves = set(state.iter_pseudo_moves(jq.RED))
        self.assertIn((e_idx, jq.index(4, 0)), moves)
        self.assertIn((dark_a_idx, jq.index(8, 4)), moves)
        self.assertNotIn((dark_a_idx, jq.index(6, 2)), moves)
        self.assertIn((bright_a_idx, jq.index(4, 2)), moves)

    def test_cannon_screen_rule(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)

        c_idx = state.place_piece(5, 0, jq.TYPE_CANNON, jq.RED, 0)
        target_idx = state.place_piece(5, 3, jq.TYPE_ROOK, jq.BLACK, 0)

        moves = set(state.iter_pseudo_moves(jq.RED))
        self.assertNotIn((c_idx, target_idx), moves)

        state.place_piece(5, 2, jq.TYPE_PAWN, jq.RED, 0)
        moves = set(state.iter_pseudo_moves(jq.RED))
        self.assertIn((c_idx, target_idx), moves)

    def test_checkmate_result(self):
        state = jq.GameState.empty()
        state.side = jq.BLACK
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)

        state.place_piece(1, 4, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(1, 3, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(1, 5, jq.TYPE_ROOK, jq.RED, 0)

        self.assertEqual(state.game_result(), jq.RED)

    def test_no_capture_draw(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(4, 4, jq.TYPE_PAWN, jq.RED, 0)

        state.no_capture_plies = jq.NO_CAPTURE_LIMIT_TIANTIAN_PLIES
        outcome, reason = state.game_outcome()
        self.assertEqual(outcome, jq.RESULT_DRAW)
        self.assertEqual(reason, jq.REASON_NO_CAPTURE)

    def test_legacy_no_capture_draw(self):
        state = jq.GameState.empty(rules=jq.RULESET_LEGACY)
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(4, 4, jq.TYPE_PAWN, jq.RED, 0)

        state.no_capture_plies = jq.NO_CAPTURE_LIMIT_PLIES
        outcome, reason = state.game_outcome()
        self.assertEqual(outcome, jq.RESULT_DRAW)
        self.assertEqual(reason, jq.REASON_NO_CAPTURE)

    def test_long_check_legacy_loss(self):
        state = jq.GameState.empty(rules=jq.RULESET_LEGACY)
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 4, jq.TYPE_ROOK, jq.RED, 0)

        moves = [
            (jq.index(5, 4), jq.index(1, 4)),
            (jq.index(0, 4), jq.index(0, 3)),
            (jq.index(1, 4), jq.index(1, 3)),
            (jq.index(0, 3), jq.index(0, 4)),
            (jq.index(1, 3), jq.index(1, 4)),
            (jq.index(0, 4), jq.index(0, 3)),
            (jq.index(1, 4), jq.index(1, 3)),
        ]
        for move in moves:
            state.apply_move(move)

        outcome, reason = state.game_outcome()
        self.assertEqual(outcome, jq.RESULT_BLACK_WIN)
        self.assertEqual(reason, jq.REASON_LONG_CHECK)

    def test_checkmate_priority_over_long_check(self):
        state = jq.GameState.empty()
        state.side = jq.BLACK
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(1, 4, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(1, 3, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(1, 5, jq.TYPE_ROOK, jq.RED, 0)
        state.consecutive_checks[jq.RED] = jq.LONG_CHECK_ROUNDS + 1

        outcome, reason = state.game_outcome()
        self.assertEqual(outcome, jq.RESULT_RED_WIN)
        self.assertEqual(reason, jq.REASON_CHECKMATE)

    def test_long_check_forbidden_but_alternative_exists(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 4, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(9, 0, jq.TYPE_PAWN, jq.RED, 0)
        state.reset_history()
        state.consecutive_checks[jq.RED] = jq.LONG_CHECK_ROUNDS
        state.check_anchor[jq.RED] = jq.index(5, 4)

        legal = set(state.legal_moves())
        self.assertNotIn((jq.index(5, 4), jq.index(5, 3)), legal)
        self.assertIn((jq.index(9, 0), jq.index(8, 0)), legal)
        outcome, _ = state.game_outcome()
        self.assertIsNone(outcome)

    def test_long_check_no_alternative_loss(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(9, 0, jq.TYPE_ROOK, jq.BLACK, 0)
        state.place_piece(8, 3, jq.TYPE_PAWN, jq.BLACK, 0)
        state.place_piece(8, 5, jq.TYPE_PAWN, jq.BLACK, 0)
        state.place_piece(0, 2, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 2, jq.TYPE_ROOK, jq.RED, 0)
        state.reset_history()
        state.consecutive_checks[jq.RED] = jq.LONG_CHECK_ROUNDS
        state.check_anchor[jq.RED] = jq.index(5, 2)

        self.assertEqual(state.legal_moves(), [])
        outcome, reason = state.game_outcome()
        self.assertEqual(outcome, jq.RESULT_BLACK_WIN)
        self.assertEqual(reason, jq.REASON_LONG_CHECK)
        state.no_capture_plies = jq.NO_CAPTURE_LIMIT_TIANTIAN_PLIES
        self.assertEqual(state.outcome_checked(), state.game_outcome())

    def test_long_chase_forbidden_moves(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 5, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 6, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(5, 7, jq.TYPE_KNIGHT, jq.BLACK, 0)
        state.reset_history()

        moves = [
            (jq.index(5, 6), jq.index(5, 5)),
            (jq.index(0, 3), jq.index(0, 4)),
            (jq.index(5, 5), jq.index(5, 6)),
            (jq.index(0, 4), jq.index(0, 3)),
            (jq.index(5, 6), jq.index(5, 5)),
            (jq.index(0, 3), jq.index(0, 4)),
            (jq.index(5, 5), jq.index(5, 6)),
            (jq.index(0, 4), jq.index(0, 3)),
            (jq.index(5, 6), jq.index(5, 5)),
            (jq.index(0, 3), jq.index(0, 4)),
            (jq.index(5, 5), jq.index(5, 6)),
            (jq.index(0, 4), jq.index(0, 3)),
        ]
        for move in moves:
            state.apply_move(move)

        self.assertEqual(state.chase_streak[jq.RED], jq.LONG_CHASE_ROUNDS)
        legal = set(state.legal_moves())
        self.assertNotIn((jq.index(5, 6), jq.index(5, 5)), legal)
        self.assertIn((jq.index(5, 6), jq.index(4, 6)), legal)
        self.assertIn((jq.index(5, 6), jq.index(5, 7)), legal)

    def test_knight_attack_leg_direction(self):
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        knight_idx = state.place_piece(3, 3, jq.TYPE_KNIGHT, jq.BLACK, 0)
        target = jq.index(5, 4)
        state.place_piece(4, 3, jq.TYPE_PAWN, jq.RED, 0)

        self.assertFalse(state.is_square_attacked(jq.BLACK, target))
        self.assertNotIn((knight_idx, target), set(state.iter_pseudo_moves(jq.BLACK)))

        state.set_piece(jq.index(4, 3), 0)
        state.place_piece(4, 4, jq.TYPE_PAWN, jq.RED, 0)

        self.assertTrue(state.is_square_attacked(jq.BLACK, target))
        self.assertIn((knight_idx, target), set(state.iter_pseudo_moves(jq.BLACK)))

    def test_knight_leg_false_negative_check(self):
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(3, 3, jq.TYPE_KNIGHT, jq.BLACK, 0)
        state.place_piece(4, 4, jq.TYPE_PAWN, jq.RED, 0)
        state.place_piece(5, 4, jq.TYPE_KING, jq.RED, 0)
        state.side = jq.BLACK

        self.assertTrue(state.is_in_check(jq.RED))

    def test_long_check_requires_same_piece(self):
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 0, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(7, 8, jq.TYPE_ROOK, jq.RED, 0)
        state.reset_history()

        state.apply_move((jq.index(5, 0), jq.index(0, 0)))
        state.apply_move((jq.index(0, 3), jq.index(1, 3)))
        state.apply_move((jq.index(7, 8), jq.index(7, 3)))

        self.assertEqual(state.consecutive_checks[jq.RED], 1)
        self.assertEqual(state.check_anchor[jq.RED], jq.index(7, 3))

    def test_long_chase_requires_same_piece(self):
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 5, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 6, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(1, 7, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(5, 7, jq.TYPE_KNIGHT, jq.BLACK, 0)
        state.reset_history()

        moves = [
            (jq.index(5, 6), jq.index(5, 5)),
            (jq.index(0, 3), jq.index(0, 4)),
            (jq.index(5, 5), jq.index(5, 6)),
            (jq.index(0, 4), jq.index(0, 3)),
            (jq.index(1, 7), jq.index(2, 7)),
        ]
        for move in moves:
            state.apply_move(move)

        self.assertEqual(state.chase_target[jq.RED], jq.index(5, 7))
        self.assertEqual(state.chase_streak[jq.RED], 1)
        self.assertEqual(state.chase_anchor[jq.RED], jq.index(2, 7))

    def test_fen_restores_rule_state_and_repetition(self):
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(5, 0, jq.TYPE_ROOK, jq.RED, 0)
        state.place_piece(7, 8, jq.TYPE_ROOK, jq.RED, 0)
        state.reset_history()
        state.apply_move((jq.index(5, 0), jq.index(0, 0)))
        state.apply_move((jq.index(0, 3), jq.index(1, 3)))
        state.position_counts[state.position_key] = 2

        fen = jq.to_fen(state)
        clone = jq.from_fen(fen, rules=state.rules)

        self.assertEqual(clone.consecutive_checks, state.consecutive_checks)
        self.assertEqual(clone.check_anchor, state.check_anchor)
        self.assertEqual(clone.chase_streak, state.chase_streak)
        self.assertEqual(clone.chase_target, state.chase_target)
        self.assertEqual(clone.chase_target_code, state.chase_target_code)
        self.assertEqual(clone.chase_anchor, state.chase_anchor)
        self.assertEqual(clone.position_counts.get(clone.position_key), 2)
        self.assertEqual(jq.to_fen(clone), fen)

    def test_repetition_draw_tiantian(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(4, 4, jq.TYPE_PAWN, jq.RED, 0)
        state.reset_history()

        cycle = [
            (jq.index(9, 4), jq.index(9, 3)),
            (jq.index(0, 4), jq.index(0, 3)),
            (jq.index(9, 3), jq.index(9, 4)),
            (jq.index(0, 3), jq.index(0, 4)),
        ]
        for _ in range(2):
            for move in cycle:
                state.apply_move(move)

        outcome, reason = state.game_outcome()
        self.assertEqual(outcome, jq.RESULT_DRAW)
        self.assertEqual(reason, jq.REASON_REPETITION)

    def test_repetition_legacy_loss(self):
        state = jq.GameState.empty(rules=jq.RULESET_LEGACY)
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(4, 4, jq.TYPE_PAWN, jq.RED, 0)
        state.reset_history()

        cycle = [
            (jq.index(9, 4), jq.index(9, 3)),
            (jq.index(0, 4), jq.index(0, 3)),
            (jq.index(9, 3), jq.index(9, 4)),
            (jq.index(0, 3), jq.index(0, 4)),
        ]
        for _ in range(3):
            for move in cycle:
                state.apply_move(move)

        outcome, reason = state.game_outcome()
        self.assertEqual(outcome, jq.RESULT_RED_WIN)
        self.assertEqual(reason, jq.REASON_REPETITION)

    def test_fast_outcome_no_capture(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.no_capture_plies = jq.NO_CAPTURE_LIMIT_TIANTIAN_PLIES
        outcome, reason = state.fast_outcome()
        self.assertEqual(outcome, jq.RESULT_DRAW)
        self.assertEqual(reason, jq.REASON_NO_CAPTURE)

    def test_fast_outcome_long_check(self):
        state = jq.GameState.empty(rules=jq.RULESET_LEGACY)
        state.side = jq.BLACK
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.consecutive_checks[jq.RED] = jq.MAX_CONSECUTIVE_CHECKS + 1
        outcome, reason = state.fast_outcome()
        self.assertEqual(outcome, jq.RESULT_BLACK_WIN)
        self.assertEqual(reason, jq.REASON_LONG_CHECK)

    def test_fast_outcome_repetition(self):
        state = jq.GameState.empty(rules=jq.RULESET_LEGACY)
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.reset_history()
        state.position_counts[state.position_key] = jq.MAX_REPETITIONS + 1
        outcome, reason = state.fast_outcome()
        self.assertEqual(outcome, jq.RESULT_RED_WIN)
        self.assertEqual(reason, jq.REASON_REPETITION)

    def test_fast_outcome_king_missing(self):
        state = jq.GameState.empty()
        state.side = jq.RED
        state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
        state.king_pos[jq.BLACK] = -1
        outcome, reason = state.fast_outcome()
        self.assertEqual(outcome, jq.RESULT_RED_WIN)
        self.assertEqual(reason, jq.REASON_CHECKMATE)

    def test_fen_initial_position(self):
        state = jq.GameState.initial()
        self.assertEqual(
            jq.to_fen(state),
            "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w - - 0 1",
        )

    def test_fen_roundtrip_midgame(self):
        state = jq.GameState.initial()
        state.apply_move((jq.index(6, 0), jq.index(5, 0)), reveal_choice=jq.TYPE_ROOK)
        state.apply_move((jq.index(3, 0), jq.index(4, 0)), reveal_choice=jq.TYPE_PAWN)
        state.apply_move((jq.index(5, 0), jq.index(4, 0)), capture_reveal_choice=jq.TYPE_CANNON)
        fen = jq.to_fen(state)
        clone = jq.from_fen(fen, rules=state.rules)
        self.assertEqual(clone.side, state.side)
        self.assertEqual(clone.board, state.board)
        self.assertEqual(clone.pool, state.pool)
        self.assertEqual(clone.pool_total, state.pool_total)
        self.assertEqual(clone.king_pos, state.king_pos)
        self.assertEqual(clone.no_capture_plies, state.no_capture_plies)
        self.assertEqual(clone.consecutive_checks, state.consecutive_checks)
        self.assertEqual(clone.check_anchor, state.check_anchor)
        self.assertEqual(clone.chase_streak, state.chase_streak)
        self.assertEqual(clone.chase_target, state.chase_target)
        self.assertEqual(clone.chase_target_code, state.chase_target_code)
        self.assertEqual(clone.chase_anchor, state.chase_anchor)
        self.assertEqual(jq.to_fen(clone), fen)

    def test_square_and_move_notation(self):
        self.assertEqual(jq.square_to_name(jq.index(0, 1)), "b9")
        self.assertEqual(jq.name_to_square("c7"), jq.index(2, 2))
        self.assertEqual(
            jq.format_move_uci((jq.index(0, 1), jq.index(2, 2)), jq.TYPE_CANNON),
            "b9c7c",
        )
        self.assertEqual(jq.format_move_uci((jq.index(5, 4), jq.index(7, 6))), "e4g2")

    def test_long_chase_tracks_fleeing_victim(self):
        """P1-6：受害子换格逃跑后，同一追子的续捉计数应继续累加。"""
        state = jq.GameState.empty(with_pools=False)
        state.place_piece(9, 0, jq.TYPE_KING, jq.RED, 0)
        state.place_piece(0, 8, jq.TYPE_KING, jq.BLACK, 0)
        state.place_piece(1, 5, jq.TYPE_ROOK, jq.RED, 0)  # 追子
        state.place_piece(2, 4, jq.TYPE_ROOK, jq.BLACK, 0)  # 逃跑的受害车
        state.side = jq.RED
        state.reset_history()

        seq = [
            (jq.index(1, 5), jq.index(1, 4)),  # 红车追 (2,4)
            (jq.index(2, 4), jq.index(2, 5)),  # 黑车逃
            (jq.index(1, 4), jq.index(1, 5)),  # 红车追 (2,5)
            (jq.index(2, 5), jq.index(2, 4)),  # 黑车逃回
        ]
        for i in range(12):
            state.apply_move(seq[i % 4])

        self.assertEqual(state.chase_streak[jq.RED], jq.LONG_CHASE_ROUNDS)
        legal = set(state.legal_moves())
        # 再追同一枚逃回 (2,4) 的黑车应被禁；换目标/离开不追则放行
        self.assertNotIn((jq.index(1, 5), jq.index(1, 4)), legal)
        self.assertIn((jq.index(1, 5), jq.index(0, 5)), legal)

    def test_outcome_checked_matches_game_outcome_priority(self):
        """P2-7：无吃子和棋与困毙同触时，校正版结果应与 game_outcome 一致。"""
        fen = "5x2e/1e3k3/4p4/9/9/4p3p/9/4K1c2/3r5/9[A2E2R2C2H2P5a2r1c1h2p1] w - - 80 1"
        state = jq.from_fen(fen)
        self.assertEqual(state.legal_moves(), [])
        self.assertEqual(state.game_outcome(), (jq.RESULT_BLACK_WIN, jq.REASON_STALEMATE))
        self.assertEqual(state.outcome_checked(), state.game_outcome())
        self.assertEqual(im.outcome_from_moves(state, state.legal_moves()), state.game_outcome())
        # fast_outcome 仍是廉价近似（保留旧语义）
        self.assertEqual(state.fast_outcome(), (jq.RESULT_DRAW, jq.REASON_NO_CAPTURE))

    def test_queries_do_not_consume_rng(self):
        """P2-8：legal_moves/game_outcome 查询不得推进 state.rng。"""

        def build_banned_state():
            st = jq.GameState.empty()
            st.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
            st.place_piece(0, 3, jq.TYPE_KING, jq.BLACK, 0)
            st.place_piece(5, 4, jq.TYPE_ROOK, jq.RED, 0)
            st.place_piece(6, 0, jq.TYPE_PAWN, jq.RED, 1)
            st.side = jq.RED
            st.reset_history()
            st.consecutive_checks[jq.RED] = jq.LONG_CHECK_ROUNDS
            st.check_anchor[jq.RED] = jq.index(5, 4)
            return st

        move = (jq.index(6, 0), jq.index(5, 0))
        control = build_banned_state()
        control.rng = random.Random(123)
        queried = build_banned_state()
        queried.rng = random.Random(123)

        before = queried.rng.getstate()
        queried.legal_moves()
        queried.game_outcome()
        self.assertEqual(queried.rng.getstate(), before)

        undo_control = control.apply_move(move)
        undo_queried = queried.apply_move(move)
        self.assertEqual(undo_control.moved_reveal, undo_queried.moved_reveal)

    def test_materialize_dark_types_deterministic_and_consistent(self):
        """P1-5：预物化洗牌对同 seed 确定，且每方类型计数与初始暗池一致。"""
        board, _ = jq.initial_board()
        first = jq.materialize_dark_types(random.Random("pair-seed"), board)
        second = jq.materialize_dark_types(random.Random("pair-seed"), board)
        self.assertEqual(first, second)

        counts = {jq.RED: [0] * 8, jq.BLACK: [0] * 8}
        for idx, p_type in first.items():
            counts[jq.piece_color(board[idx])][p_type] += 1
        self.assertEqual(counts[jq.RED], jq.POOL_INIT)
        self.assertEqual(counts[jq.BLACK], jq.POOL_INIT)

    def test_preassigned_shuffle_reveals_same_type_per_cell(self):
        """P1-5：带预分配洗牌时，同一格暗子揭示的真值固定且池计数正确扣减。"""
        board, _ = jq.initial_board()
        assignment = jq.materialize_dark_types(random.Random("shared"), board)
        move = (jq.index(6, 0), jq.index(5, 0))  # 红方暗兵前进
        reveals = []
        for _ in range(2):
            state = jq.GameState.initial(rng=random.Random(1), dark_types=assignment)
            undo = state.apply_move(move)
            reveals.append(undo.moved_reveal)
            self.assertEqual(undo.moved_reveal, assignment[move[0]])
            self.assertEqual(state.pool[jq.RED][undo.moved_reveal], jq.POOL_INIT[undo.moved_reveal] - 1)
        self.assertEqual(reveals[0], reveals[1])


if __name__ == "__main__":
    unittest.main()
