# cython: language_level=3, infer_types=False, annotation_typing=False, boundscheck=True, wraparound=True
"""Isolated encoder candidate; never imported by production."""

import numpy as np

API_VERSION = 1
SOURCE_SHA256 = NATIVE_SOURCE_SHA256
CORE_SHA256 = NATIVE_CORE_SHA256
CODE_SHA256 = NATIVE_CODE_SHA256
MODULE = ORIGINAL = HELPERS = None
UNSET = object()


def initialize(module, fingerprint):
    global MODULE, ORIGINAL, HELPERS
    if fingerprint != CODE_SHA256 or module.STATE_CHANNELS != 238:
        raise ImportError("experimental encoder source/function fingerprint mismatch")
    if MODULE is not None:
        if MODULE is not module:
            raise ImportError("experimental encoder cannot serve another module")
        return
    MODULE = module
    ORIGINAL = module._encode_state_into
    HELPERS = (module._piece_plane_index, module.jq.is_dark,
               module.jq.piece_color, module.jq.piece_type)


def supports(info, planes):
    cdef object board, piece, jq
    if MODULE is None:
        raise RuntimeError("initialize experimental encoder first")
    jq = MODULE.jq
    board = info.board
    if (type(board) is not list or len(board) != 90
            or type(planes) is not np.ndarray or planes.dtype != np.float32
            or (MODULE.BOARD_SIZE, jq.TYPE_MASK, jq.COLOR_SHIFT, jq.DARK_SHIFT) != (90, 7, 3, 4)
            or (MODULE.HISTORY_PLIES, MODULE.BOARD_H, MODULE.BOARD_W, MODULE.REPETITION_HISTORY_SIZE) != (8, 10, 9, 81)
            or (MODULE._piece_plane_index, jq.is_dark, jq.piece_color, jq.piece_type) != HELPERS):
        return False
    for piece in board:
        if type(piece) is not int or piece < 0 or piece > 31:
            return False
    return True


def _supported_history(history):
    if type(history) is not list:
        return False
    for move in history[-MODULE.HISTORY_PLIES:]:
        if type(move) not in (tuple, list) or len(move) != 2:
            return False
        for square in move:
            if type(square) is not int or square < -90 or square >= 90:
                return False
    return True


def _supported_entries(entries):
    if entries is MODULE._REPETITION_ENTRIES_UNSET or entries is None:
        return True
    if type(entries) is not tuple or len(entries) > MODULE.REPETITION_HISTORY_SIZE:
        return False
    for entry in entries:
        if (type(entry) is not tuple or len(entry) != 3 or type(entry[0]) is not int
                or type(entry[1]) is not bytes or len(entry[1]) != 90 or type(entry[2]) is not int):
            return False
    return True


def encode_into(info_state, history, planes, *, repetition_entries=UNSET):
    cdef int idx, code, plane
    cdef object module = MODULE
    cdef object jq
    if module is None:
        raise RuntimeError("initialize experimental encoder first")
    if repetition_entries is UNSET:
        repetition_entries = module._REPETITION_ENTRIES_UNSET
    if (not supports(info_state, planes) or not _supported_history(history)
            or not _supported_entries(repetition_entries)):
        return ORIGINAL(info_state, history, planes, repetition_entries=repetition_entries)
    jq = module.jq
    # Only this integer loop bypasses helpers; all floating operations remain Python objects.
    for idx in range(90):
        code = info_state.board[idx]
        if not code:
            continue
        plane = (14 if code >> 4 else 0) + ((code >> 3) & 1) * 7 + (code & 7) - 1
        planes[plane, jq.ROWS[idx], jq.COLS[idx]] = 1.0

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
    recent = history[-module.HISTORY_PLIES:]
    for i, move in enumerate(recent):
        from_idx, to_idx = move
        planes[offset + i * 2, jq.ROWS[from_idx], jq.COLS[from_idx]] = 1.0
        planes[offset + i * 2 + 1, jq.ROWS[to_idx], jq.COLS[to_idx]] = 1.0
    offset += 2 * module.HISTORY_PLIES

    planes[offset, :, :] = module._repetition_count(info_state) / jq.REPETITION_LIMIT
    planes[offset + 1, :, :] = info_state.chase_streak[jq.RED] / jq.LONG_CHASE_ROUNDS
    planes[offset + 2, :, :] = info_state.chase_streak[jq.BLACK] / jq.LONG_CHASE_ROUNDS
    for i, side in enumerate((jq.RED, jq.BLACK)):
        target = info_state.chase_target[side]
        if target >= 0:
            planes[offset + 3 + i, jq.ROWS[target], jq.COLS[target]] = 1.0
        anchor = info_state.chase_anchor[side]
        if anchor >= 0:
            planes[offset + 5 + i, jq.ROWS[anchor], jq.COLS[anchor]] = 1.0

    for i, side in enumerate((jq.RED, jq.BLACK)):
        anchor = info_state.check_anchor[side]
        if anchor >= 0:
            planes[71 + i, jq.ROWS[anchor], jq.COLS[anchor]] = 1.0
        planes[73 + i, :, :] = info_state.chase_target_code[side] / 31.0
    entries = (module._repetition_entries(info_state) if repetition_entries is module._REPETITION_ENTRIES_UNSET
               else repetition_entries)
    if entries is not None:
        planes[module.REPETITION_HISTORY_COMPLETE_CHANNEL, :, :] = 1.0
        if entries:
            end = module.REPETITION_HISTORY_OFFSET + 2 * len(entries)
            boards = np.frombuffer(b"".join(board for _, board, _ in entries), dtype=np.uint8)
            boards = boards.reshape(len(entries), module.BOARD_H, module.BOARD_W)
            planes[module.REPETITION_HISTORY_OFFSET:end:2] = module._REPETITION_PIECE_VALUES[boards]
            counts = [(2 * count + side) / (2 * jq.REPETITION_LIMIT + 1) for side, _, count in entries]
            planes[module.REPETITION_HISTORY_OFFSET + 1:end:2] = np.asarray(counts, dtype=np.float32)[:, None, None]
