# cython: language_level=3, boundscheck=True, wraparound=True
"""Read-only attack detection and integer snapshot guards; RNG stays in Python."""

from cpython.long cimport PyLong_CheckExact
from cpython.tuple cimport PyTuple_CheckExact

API_VERSION = 1
RULESET_VERSION = 5
SOURCE_SHA256 = NATIVE_SOURCE_SHA256

cdef list _rays, _knights, _elephants, _advisors, _palace, _rows, _cols, _crossed


def immutable_integer_pools(counts):
    """Check immutable pool rows without converting arbitrary Python integers."""
    cdef object row
    cdef tuple pool
    cdef Py_ssize_t index
    for row in counts:
        if row is None:
            continue
        if not PyTuple_CheckExact(row):
            return False
        pool = row
        for index in range(len(pool)):
            if not PyLong_CheckExact(pool[index]):
                return False
    return True


def initialize(rules):
    global _rays, _knights, _elephants, _advisors, _palace, _rows, _cols, _crossed
    if rules.RULESET_VERSION != RULESET_VERSION or rules.BOARD_SIZE != 90:
        raise ImportError("incompatible native rules version")
    _rays = rules.RAYS
    _knights = rules.KNIGHT_ATTACKS
    _elephants = rules.ELEPHANT_MOVES
    _advisors = rules.ADVISOR_MOVES
    _palace = rules.PALACE
    _rows = rules.ROWS
    _cols = rules.COLS
    _crossed = rules.CROSSED_RIVER


def is_square_attacked(self, by_side, idx):
    cdef list board, ray
    cdef int side, cell, j, leg, eye, code, r, c, p_idx
    cdef bint screen
    cdef object target
    # Preserve Python indexing/errors for callers outside the normal game domain.
    if (type(self.board) is not list or not isinstance(idx, int)
            or not isinstance(by_side, int) or idx < 0 or idx >= 90
            or by_side not in (0, 1)):
        return self._is_square_attacked_python(by_side, idx)
    board = self.board
    cell = idx
    side = by_side
    for ray in _rays[cell]:
        screen = False
        for j in ray:
            target = board[j]
            if not target:
                continue
            # Mask before conversion: arbitrary Python integers cannot overflow C int.
            code = target & 15
            if not screen:
                if (code >> 3) == side and (code & 7) in (2, 1):
                    return True
                screen = True
            else:
                if (code >> 3) == side and (code & 7) == 6:
                    return True
                break

    for j, leg in _knights[cell]:
        if board[leg]:
            continue
        target = board[j]
        if target:
            code = target & 15
            if (code >> 3) == side and (code & 7) == 3:
                return True

    for j, eye in _elephants[cell]:
        if board[eye]:
            continue
        target = board[j]
        if target:
            code = target & 15
            if (code >> 3) == side and (code & 7) == 4:
                return True

    for j in _advisors[cell]:
        target = board[j]
        if not target:
            continue
        code = target & 15
        if (code >> 3) != side or (code & 7) != 5:
            continue
        if target & 16:
            if not (_palace[side][cell] and _palace[side][j]):
                continue
        return True

    r = _rows[cell]
    c = _cols[cell]
    if side == 1:
        if r + 1 < 10:
            target = board[(r + 1) * 9 + c]
            if target and (target & 15) == 15:
                return True
        if c - 1 >= 0:
            p_idx = r * 9 + c - 1
            target = board[p_idx]
            if target and (target & 15) == 15 and _crossed[1][p_idx]:
                return True
        if c + 1 < 9:
            p_idx = r * 9 + c + 1
            target = board[p_idx]
            if target and (target & 15) == 15 and _crossed[1][p_idx]:
                return True
    else:
        if r - 1 >= 0:
            target = board[(r - 1) * 9 + c]
            if target and (target & 15) == 7:
                return True
        if c - 1 >= 0:
            p_idx = r * 9 + c - 1
            target = board[p_idx]
            if target and (target & 15) == 7 and _crossed[0][p_idx]:
                return True
        if c + 1 < 9:
            p_idx = r * 9 + c + 1
            target = board[p_idx]
            if target and (target & 15) == 7 and _crossed[0][p_idx]:
                return True
    return False
