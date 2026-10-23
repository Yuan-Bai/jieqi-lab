# cython: language_level=3, infer_types=False, annotation_typing=False, boundscheck=True, wraparound=True
"""Compile only the no-ban legal-move path; retain Python state and callbacks."""

API_VERSION = 1
RULESET_VERSION = 5
SOURCE_SHA256 = NATIVE_SOURCE_SHA256
METHODS_SHA256 = NATIVE_METHODS_SHA256


def initialize(rules):
    if rules.RULESET_VERSION != RULESET_VERSION or rules.BOARD_SIZE != 90:
        raise ImportError("incompatible native legal rules version")


cdef inline object color(object piece, object rules):
    cdef int code
    if type(piece) is int:
        code = piece & 15
        return code >> 3
    return rules.piece_color(piece)


cdef inline object kind(object piece, object rules):
    cdef int code
    if type(piece) is int:
        code = piece & 7
        return code
    return rules.piece_type(piece)


cdef inline object dark_flag(object piece, object rules):
    cdef int code
    if type(piece) is int:
        code = piece & 16
        return code >> 4
    return rules.is_dark(piece)


def _pseudo(self, side, rules):
    cdef list board = self.board
    cdef int i, j, leg, eye, fwd, left, right, d
    cdef object p, target, p_type, dark
    cdef bint screen
    for i, p in enumerate(board):
        if not p or color(p, rules) != side:
            continue
        p_type = kind(p, rules)
        if p_type == 2:
            for ray in rules.RAYS[i]:
                for j in ray:
                    target = board[j]
                    if not target:
                        yield (i, j)
                        continue
                    if color(target, rules) != side:
                        yield (i, j)
                    break
        elif p_type == 6:
            for ray in rules.RAYS[i]:
                screen = False
                for j in ray:
                    target = board[j]
                    if not screen:
                        if not target:
                            yield (i, j)
                        else:
                            screen = True
                    else:
                        if target:
                            if color(target, rules) != side:
                                yield (i, j)
                            break
        elif p_type == 3:
            for j, leg in rules.KNIGHT_MOVES[i]:
                if board[leg]:
                    continue
                target = board[j]
                if not target or color(target, rules) != side:
                    yield (i, j)
        elif p_type == 4:
            for j, eye in rules.ELEPHANT_MOVES[i]:
                if board[eye]:
                    continue
                target = board[j]
                if not target or color(target, rules) != side:
                    yield (i, j)
        elif p_type == 5:
            dark = dark_flag(p, rules)
            if dark and not rules.PALACE[side][i]:
                continue
            for j in rules.ADVISOR_MOVES[i]:
                if dark and not rules.PALACE[side][j]:
                    continue
                target = board[j]
                if not target or color(target, rules) != side:
                    yield (i, j)
        elif p_type == 1:
            for j in rules.KING_ADJ[i]:
                if not rules.PALACE[side][j]:
                    continue
                target = board[j]
                if not target or color(target, rules) != side:
                    yield (i, j)
            for d in (0, 1):
                for j in rules.RAYS[i][d]:
                    target = board[j]
                    if target:
                        if color(target, rules) != side and kind(target, rules) == 1:
                            yield (i, j)
                        break
        elif p_type == 7:
            fwd = rules.PAWN_FWD[side][i]
            if fwd != -1:
                target = board[fwd]
                if not target or color(target, rules) != side:
                    yield (i, fwd)
            if rules.CROSSED_RIVER[side][i]:
                left = rules.PAWN_LEFT[i]
                if left != -1:
                    target = board[left]
                    if not target or color(target, rules) != side:
                        yield (i, left)
                right = rules.PAWN_RIGHT[i]
                if right != -1:
                    target = board[right]
                    if not target or color(target, rules) != side:
                        yield (i, right)


def iter_legal_moves_fast(self, side):
    cdef object rules = self._native_legal_module
    cdef list board
    cdef int frm, to
    cdef object king_side, move, moving, captured, moved_king, safe
    if (type(self) is not rules.GameState or type(self.board) is not list
            or len(self.board) != 90 or type(side) is not int or side not in (0, 1)
            or type(self.king_pos) is not list
            or getattr(self.iter_pseudo_moves, "__func__", None) is not rules.GameState._iter_pseudo_moves_python
            or (rules.piece_color, rules.piece_type, rules.is_dark) != self._legal_helpers_python
            or (rules.TYPE_MASK, rules.COLOR_SHIFT, rules.DARK_SHIFT) != (7, 3, 4)
            or (rules.TYPE_KING, rules.TYPE_ROOK, rules.TYPE_KNIGHT, rules.TYPE_ELEPHANT,
                rules.TYPE_ADVISOR, rules.TYPE_CANNON, rules.TYPE_PAWN, rules.UP, rules.DOWN)
               != (1, 2, 3, 4, 5, 6, 7, 0, 1)):
        yield from self._iter_legal_moves_fast_python(side)
        return
    board = self.board
    king_side = self.king_pos[side]
    if king_side < 0:
        return
    for move in _pseudo(self, side, rules):
        frm, to = move
        moving = board[frm]
        captured = board[to]
        board[to] = moving
        board[frm] = 0
        moved_king = color(moving, rules) == side and kind(moving, rules) == 1
        if moved_king:
            self.king_pos[side] = to
        safe = not self.is_square_attacked(1 - side, self.king_pos[side])
        if moved_king:
            self.king_pos[side] = king_side
        board[frm] = moving
        board[to] = captured
        if safe:
            yield move
