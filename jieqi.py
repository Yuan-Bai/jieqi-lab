from __future__ import annotations

import random
import re
from dataclasses import dataclass

BOARD_W = 9
BOARD_H = 10
BOARD_SIZE = BOARD_W * BOARD_H

BLACK = 0
RED = 1

TYPE_KING = 1
TYPE_ROOK = 2
TYPE_KNIGHT = 3
TYPE_ELEPHANT = 4
TYPE_ADVISOR = 5
TYPE_CANNON = 6
TYPE_PAWN = 7

TYPE_MASK = 0x7
COLOR_SHIFT = 3
DARK_SHIFT = 4

POOL_TYPES = (TYPE_ROOK, TYPE_KNIGHT, TYPE_ELEPHANT, TYPE_ADVISOR, TYPE_CANNON, TYPE_PAWN)
FULL_WIDTH_SPACE = "　"

PIECE_NAMES = {
    RED: {
        TYPE_KING: "帅",
        TYPE_ROOK: "车",
        TYPE_KNIGHT: "马",
        TYPE_ELEPHANT: "相",
        TYPE_ADVISOR: "仕",
        TYPE_CANNON: "炮",
        TYPE_PAWN: "兵",
    },
    BLACK: {
        TYPE_KING: "将",
        TYPE_ROOK: "车",
        TYPE_KNIGHT: "马",
        TYPE_ELEPHANT: "象",
        TYPE_ADVISOR: "士",
        TYPE_CANNON: "炮",
        TYPE_PAWN: "卒",
    },
}
NO_CAPTURE_LIMIT_FULL_MOVES = 25
NO_CAPTURE_LIMIT_PLIES = NO_CAPTURE_LIMIT_FULL_MOVES * 2
MAX_CONSECUTIVE_CHECKS = 3
MAX_REPETITIONS = 3

LONG_CHECK_ROUNDS = 6
LONG_CHASE_ROUNDS = 6
REPETITION_LIMIT = 3
NO_CAPTURE_LIMIT_TIANTIAN_FULL_MOVES = 40
NO_CAPTURE_LIMIT_TIANTIAN_PLIES = NO_CAPTURE_LIMIT_TIANTIAN_FULL_MOVES * 2
# v4: tied new chase targets clear; v5: symmetric check anchors and exact victim tracking.
RULESET_VERSION = 5

RESULT_RED_WIN = 1
RESULT_BLACK_WIN = -1
RESULT_DRAW = 0

REASON_LONG_CHECK = "long_check"
REASON_LONG_CHASE = "long_chase"
REASON_REPETITION = "repetition"
REASON_NO_CAPTURE = "no_capture"
REASON_CHECKMATE = "checkmate"
REASON_STALEMATE = "stalemate"


@dataclass(frozen=True, slots=True)
class RulesConfig:
    """对局规则配置：默认天天象棋揭棋规则，legacy 保留旧行为。"""

    name: str = "tiantian"
    long_check_rounds: int = LONG_CHECK_ROUNDS
    long_chase_rounds: int = LONG_CHASE_ROUNDS
    repetition_limit: int = REPETITION_LIMIT
    no_capture_plies: int = NO_CAPTURE_LIMIT_TIANTIAN_PLIES
    forbidden_moves: bool = True
    repetition_is_loss: bool = False
    ruleset_version: int = RULESET_VERSION


RULESET_TIANTIAN = RulesConfig()
RULESET_LEGACY = RulesConfig(
    name="legacy",
    long_check_rounds=MAX_CONSECUTIVE_CHECKS + 1,
    long_chase_rounds=0,
    repetition_limit=MAX_REPETITIONS + 1,
    no_capture_plies=NO_CAPTURE_LIMIT_PLIES,
    forbidden_moves=False,
    repetition_is_loss=True,
    ruleset_version=0,
)

CHASE_VALUES = {
    TYPE_ROOK: 9,
    TYPE_CANNON: 4,
    TYPE_KNIGHT: 4,
    TYPE_ELEPHANT: 2,
    TYPE_ADVISOR: 2,
}


def index(row: int, col: int) -> int:
    return row * BOARD_W + col


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_H and 0 <= col < BOARD_W


def make_piece(piece_type: int, side: int, dark: int) -> int:
    return piece_type | (side << COLOR_SHIFT) | (dark << DARK_SHIFT)


def piece_type(piece: int) -> int:
    return piece & TYPE_MASK


def piece_color(piece: int) -> int:
    return (piece >> COLOR_SHIFT) & 1


def is_dark(piece: int) -> int:
    return (piece >> DARK_SHIFT) & 1


ROWS = [i // BOARD_W for i in range(BOARD_SIZE)]
COLS = [i % BOARD_W for i in range(BOARD_SIZE)]

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

RAYS = [[[] for _ in range(4)] for _ in range(BOARD_SIZE)]
for i in range(BOARD_SIZE):
    r = ROWS[i]
    c = COLS[i]
    for rr in range(r - 1, -1, -1):
        RAYS[i][UP].append(index(rr, c))
    for rr in range(r + 1, BOARD_H):
        RAYS[i][DOWN].append(index(rr, c))
    for cc in range(c - 1, -1, -1):
        RAYS[i][LEFT].append(index(r, cc))
    for cc in range(c + 1, BOARD_W):
        RAYS[i][RIGHT].append(index(r, cc))

PALACE = [[False] * BOARD_SIZE for _ in range(2)]
for r in range(0, 3):
    for c in range(3, 6):
        PALACE[BLACK][index(r, c)] = True
for r in range(7, 10):
    for c in range(3, 6):
        PALACE[RED][index(r, c)] = True

KING_ADJ = [[] for _ in range(BOARD_SIZE)]
for i in range(BOARD_SIZE):
    r = ROWS[i]
    c = COLS[i]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r2 = r + dr
        c2 = c + dc
        if in_bounds(r2, c2):
            KING_ADJ[i].append(index(r2, c2))

ADVISOR_MOVES = [[] for _ in range(BOARD_SIZE)]
for i in range(BOARD_SIZE):
    r = ROWS[i]
    c = COLS[i]
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        r2 = r + dr
        c2 = c + dc
        if in_bounds(r2, c2):
            ADVISOR_MOVES[i].append(index(r2, c2))

ELEPHANT_MOVES = [[] for _ in range(BOARD_SIZE)]
for i in range(BOARD_SIZE):
    r = ROWS[i]
    c = COLS[i]
    for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
        r2 = r + dr
        c2 = c + dc
        if in_bounds(r2, c2):
            eye = index(r + dr // 2, c + dc // 2)
            ELEPHANT_MOVES[i].append((index(r2, c2), eye))

KNIGHT_MOVES = [[] for _ in range(BOARD_SIZE)]
for i in range(BOARD_SIZE):
    r = ROWS[i]
    c = COLS[i]
    for dr, dc, lr, lc in (
        (-2, -1, -1, 0),
        (-2, 1, -1, 0),
        (2, -1, 1, 0),
        (2, 1, 1, 0),
        (-1, -2, 0, -1),
        (1, -2, 0, -1),
        (-1, 2, 0, 1),
        (1, 2, 0, 1),
    ):
        r2 = r + dr
        c2 = c + dc
        if in_bounds(r2, c2):
            leg = index(r + lr, c + lc)
            KNIGHT_MOVES[i].append((index(r2, c2), leg))

# 反向索引：KNIGHT_ATTACKS[目标格] = [(攻击方马的位置, 该马的马腿格), ...]
# 马腿必须按“攻击方”计算，否则与 iter_pseudo_moves 的走法判定不一致。
KNIGHT_ATTACKS = [[] for _ in range(BOARD_SIZE)]
for attacker in range(BOARD_SIZE):
    for target, leg in KNIGHT_MOVES[attacker]:
        KNIGHT_ATTACKS[target].append((attacker, leg))

PAWN_FWD = [[-1] * BOARD_SIZE for _ in range(2)]
PAWN_LEFT = [-1] * BOARD_SIZE
PAWN_RIGHT = [-1] * BOARD_SIZE
CROSSED_RIVER = [[False] * BOARD_SIZE for _ in range(2)]
for i in range(BOARD_SIZE):
    r = ROWS[i]
    c = COLS[i]
    if r + 1 < BOARD_H:
        PAWN_FWD[BLACK][i] = index(r + 1, c)
    if r - 1 >= 0:
        PAWN_FWD[RED][i] = index(r - 1, c)
    if c - 1 >= 0:
        PAWN_LEFT[i] = index(r, c - 1)
    if c + 1 < BOARD_W:
        PAWN_RIGHT[i] = index(r, c + 1)
    if r >= 5:
        CROSSED_RIVER[BLACK][i] = True
    if r <= 4:
        CROSSED_RIVER[RED][i] = True

POOL_INIT = [0] * 8
POOL_INIT[TYPE_ROOK] = 2
POOL_INIT[TYPE_KNIGHT] = 2
POOL_INIT[TYPE_ELEPHANT] = 2
POOL_INIT[TYPE_ADVISOR] = 2
POOL_INIT[TYPE_CANNON] = 2
POOL_INIT[TYPE_PAWN] = 5
POOL_TOTAL = sum(POOL_INIT)

BACK_RANK = (
    TYPE_ROOK,
    TYPE_KNIGHT,
    TYPE_ELEPHANT,
    TYPE_ADVISOR,
    TYPE_KING,
    TYPE_ADVISOR,
    TYPE_ELEPHANT,
    TYPE_KNIGHT,
    TYPE_ROOK,
)


def initial_board():
    board = [0] * BOARD_SIZE
    king_pos = [-1, -1]

    for c, t in enumerate(BACK_RANK):
        dark = 0 if t == TYPE_KING else 1
        idx = index(0, c)
        board[idx] = make_piece(t, BLACK, dark)
        if t == TYPE_KING:
            king_pos[BLACK] = idx
    for c in (1, 7):
        board[index(2, c)] = make_piece(TYPE_CANNON, BLACK, 1)
    for c in (0, 2, 4, 6, 8):
        board[index(3, c)] = make_piece(TYPE_PAWN, BLACK, 1)

    for c, t in enumerate(BACK_RANK):
        dark = 0 if t == TYPE_KING else 1
        idx = index(9, c)
        board[idx] = make_piece(t, RED, dark)
        if t == TYPE_KING:
            king_pos[RED] = idx
    for c in (1, 7):
        board[index(7, c)] = make_piece(TYPE_CANNON, RED, 1)
    for c in (0, 2, 4, 6, 8):
        board[index(6, c)] = make_piece(TYPE_PAWN, RED, 1)

    return board, king_pos


def materialize_dark_types(rng, board=None) -> dict[int, int]:
    """把双方暗池一次性洗牌分配到每个暗子格，返回 {格索引: 真实类型}。

    同一 rng 种子产生同一分配，供配对评估等需要跨局复用"同一初始洗牌"的
    场景使用。返回的映射视为只读；未包含的暗子格仍走 ``apply_move`` 的
    懒抽（``_draw_from_pool``）逻辑。
    """
    board = board if board is not None else initial_board()[0]
    cells: dict[int, list[int]] = {RED: [], BLACK: []}
    for idx, piece in enumerate(board):
        if piece and is_dark(piece):
            cells[piece_color(piece)].append(idx)
    assignment: dict[int, int] = {}
    for side in (RED, BLACK):
        types: list[int] = []
        for p_type in POOL_TYPES:
            types.extend([p_type] * POOL_INIT[p_type])
        rng.shuffle(types)
        for idx, p_type in zip(cells[side], types):
            assignment[idx] = p_type
    return assignment


@dataclass(slots=True)
class Undo:
    from_idx: int
    to_idx: int
    moved_before: int
    captured_before: int
    moved_reveal: int
    captured_reveal: int
    prev_side: int
    prev_king0: int
    prev_king1: int
    prev_no_capture_plies: int
    prev_consecutive_checks_red: int
    prev_consecutive_checks_black: int
    prev_position_key: tuple
    prev_chase_streak: tuple[int, int]
    prev_chase_target: tuple[int, int]
    prev_chase_target_code: tuple[int, int]
    prev_check_anchor: int
    prev_chase_anchor: tuple[int, int]


class GameState:
    __slots__ = (
        "board",
        "side",
        "pool",
        "pool_total",
        "king_pos",
        "rng",
        "no_capture_plies",
        "consecutive_checks",
        "check_anchor",
        "position_key",
        "position_counts",
        "rules",
        "chase_streak",
        "chase_target",
        "chase_target_code",
        "chase_anchor",
        "dark_types",
    )

    def __init__(self, board, side, pool, pool_total, king_pos, rng=None, rules=None, dark_types=None):
        self.board = board
        self.side = side
        self.pool = pool
        self.pool_total = pool_total
        self.king_pos = king_pos
        self.rng = rng or random.Random()
        self.no_capture_plies = 0
        self.consecutive_checks = [0, 0]
        self.check_anchor = [-1, -1]
        self.rules = rules or RULESET_TIANTIAN
        self.chase_streak = [0, 0]
        self.chase_target = [-1, -1]
        self.chase_target_code = [0, 0]
        self.chase_anchor = [-1, -1]
        self.dark_types = dark_types
        self.position_key = self._position_key()
        self.position_counts = {self.position_key: 1}

    @classmethod
    def initial(cls, rng=None, rules=None, dark_types=None):
        board, king_pos = initial_board()
        pool = [POOL_INIT.copy(), POOL_INIT.copy()]
        pool_total = [POOL_TOTAL, POOL_TOTAL]
        return cls(board, RED, pool, pool_total, king_pos, rng, rules, dark_types)

    @classmethod
    def empty(cls, rng=None, with_pools=True, rules=None):
        board = [0] * BOARD_SIZE
        king_pos = [-1, -1]
        if with_pools:
            pool = [POOL_INIT.copy(), POOL_INIT.copy()]
            pool_total = [POOL_TOTAL, POOL_TOTAL]
        else:
            pool = [[0] * 8, [0] * 8]
            pool_total = [0, 0]
        return cls(board, RED, pool, pool_total, king_pos, rng, rules)

    def clone(self):
        state = GameState(
            self.board.copy(),
            self.side,
            [self.pool[0].copy(), self.pool[1].copy()],
            [self.pool_total[0], self.pool_total[1]],
            [self.king_pos[0], self.king_pos[1]],
            self.rng,
            self.rules,
            self.dark_types,
        )
        state.no_capture_plies = self.no_capture_plies
        state.consecutive_checks = self.consecutive_checks.copy()
        state.check_anchor = self.check_anchor.copy()
        state.chase_streak = self.chase_streak.copy()
        state.chase_target = self.chase_target.copy()
        state.chase_target_code = self.chase_target_code.copy()
        state.chase_anchor = self.chase_anchor.copy()
        state.position_key = self.position_key
        state.position_counts = self.position_counts.copy()
        return state

    def _position_key(self):
        return (self.side, tuple(self.board))

    def reset_history(self):
        self.no_capture_plies = 0
        self.consecutive_checks[0] = 0
        self.consecutive_checks[1] = 0
        self.check_anchor = [-1, -1]
        self.chase_streak = [0, 0]
        self.chase_target = [-1, -1]
        self.chase_target_code = [0, 0]
        self.chase_anchor = [-1, -1]
        self.position_key = self._position_key()
        self.position_counts = {self.position_key: 1}

    def set_piece(self, idx: int, piece: int) -> None:
        old = self.board[idx]
        if old and piece_type(old) == TYPE_KING:
            self.king_pos[piece_color(old)] = -1
        self.board[idx] = piece
        if piece and piece_type(piece) == TYPE_KING:
            self.king_pos[piece_color(piece)] = idx

    def place_piece(self, row: int, col: int, p_type: int, side: int, dark: int) -> int:
        idx = index(row, col)
        self.set_piece(idx, make_piece(p_type, side, dark))
        return idx

    def _draw_from_pool(self, side: int, rng, choice: int | None) -> int:
        if self.pool_total[side] <= 0:
            raise ValueError("pool empty")
        if choice is not None:
            if self.pool[side][choice] <= 0:
                raise ValueError("choice not available")
            self.pool[side][choice] -= 1
            self.pool_total[side] -= 1
            return choice
        roll = rng.randrange(self.pool_total[side])
        acc = 0
        for t in POOL_TYPES:
            acc += self.pool[side][t]
            if roll < acc:
                self.pool[side][t] -= 1
                self.pool_total[side] -= 1
                return t
        raise RuntimeError("pool draw failed")

    def _reveal_from_pool(self, side: int, idx: int, rng, choice: int | None) -> int:
        """揭示某格暗子的真实类型：有预分配洗牌则用（格索引），否则懒抽。

        显式 ``choice``（搜索机会节点）优先于预分配，只作用于确定化世界。
        """
        if choice is None and self.dark_types is not None:
            p_type = self.dark_types.get(idx)
            if p_type is not None and 0 < p_type < len(self.pool[side]) and self.pool[side][p_type] > 0:
                self.pool[side][p_type] -= 1
                self.pool_total[side] -= 1
                return p_type
        return self._draw_from_pool(side, rng, choice)

    def apply_move(self, move, rng=None, reveal_choice=None, capture_reveal_choice=None):
        from_idx, to_idx = move
        board = self.board
        moved_before = board[from_idx]
        if not moved_before:
            raise ValueError("no piece on from square")
        if piece_color(moved_before) != self.side:
            raise ValueError("wrong side to move")
        captured_before = board[to_idx]
        if captured_before and piece_color(captured_before) == self.side:
            raise ValueError("cannot capture own piece")

        prev_side = self.side
        prev_king0, prev_king1 = self.king_pos[0], self.king_pos[1]
        prev_no_capture_plies = self.no_capture_plies
        prev_checks_red = self.consecutive_checks[RED]
        prev_checks_black = self.consecutive_checks[BLACK]
        prev_position_key = self.position_key
        prev_chase_streak = tuple(self.chase_streak)
        prev_chase_target = tuple(self.chase_target)
        prev_chase_target_code = tuple(self.chase_target_code)
        prev_check_anchor = self.check_anchor[prev_side]
        prev_chase_anchor = tuple(self.chase_anchor)
        moved_reveal = 0
        captured_reveal = 0

        rng = rng or self.rng
        moved_piece = moved_before
        if is_dark(moved_piece):
            moved_reveal = self._reveal_from_pool(self.side, from_idx, rng, reveal_choice)
            moved_piece = make_piece(moved_reveal, self.side, 0)

        board[from_idx] = 0

        if captured_before:
            captured_side = piece_color(captured_before)
            if is_dark(captured_before):
                captured_reveal = self._reveal_from_pool(captured_side, to_idx, rng, capture_reveal_choice)
            if piece_type(captured_before) == TYPE_KING:
                self.king_pos[captured_side] = -1

        board[to_idx] = moved_piece

        # Public movement tracks the exact victim; equal piece codes are not identities.
        chaser = 1 - prev_side
        if captured_before and to_idx == self.chase_anchor[chaser]:
            self._clear_chase(chaser)
        elif from_idx == self.chase_target[chaser]:
            if (moved_before == self.chase_target_code[chaser]
                    and piece_type(moved_piece) not in (TYPE_KING, TYPE_PAWN)):
                self.chase_target[chaser] = to_idx
                self.chase_target_code[chaser] = moved_piece
            else:
                self._clear_chase(chaser)

        if piece_type(moved_piece) == TYPE_KING:
            self.king_pos[self.side] = to_idx

        self.side = 1 - self.side
        if captured_before:
            self.no_capture_plies = 0
        else:
            self.no_capture_plies = prev_no_capture_plies + 1
        gave_check = self.is_in_check(self.side)
        self._update_check(prev_side, from_idx, to_idx, gave_check)
        self._update_chase(prev_side, from_idx, to_idx, captured_before)
        self.position_key = self._position_key()
        self.position_counts[self.position_key] = self.position_counts.get(self.position_key, 0) + 1

        return Undo(
            from_idx,
            to_idx,
            moved_before,
            captured_before,
            moved_reveal,
            captured_reveal,
            prev_side,
            prev_king0,
            prev_king1,
            prev_no_capture_plies,
            prev_checks_red,
            prev_checks_black,
            prev_position_key,
            prev_chase_streak,
            prev_chase_target,
            prev_chase_target_code,
            prev_check_anchor,
            prev_chase_anchor,
        )

    def undo_move(self, undo: Undo) -> None:
        cur_key = self.position_key
        cur_count = self.position_counts[cur_key] - 1
        if cur_count:
            self.position_counts[cur_key] = cur_count
        else:
            del self.position_counts[cur_key]
        self.position_key = undo.prev_position_key
        self.no_capture_plies = undo.prev_no_capture_plies
        self.consecutive_checks[RED] = undo.prev_consecutive_checks_red
        self.consecutive_checks[BLACK] = undo.prev_consecutive_checks_black
        self.chase_streak[:] = undo.prev_chase_streak
        self.chase_target[:] = undo.prev_chase_target
        self.chase_target_code[:] = undo.prev_chase_target_code
        self.check_anchor[undo.prev_side] = undo.prev_check_anchor
        self.chase_anchor[:] = undo.prev_chase_anchor
        self.side = undo.prev_side
        self.king_pos[0] = undo.prev_king0
        self.king_pos[1] = undo.prev_king1
        self.board[undo.from_idx] = undo.moved_before
        self.board[undo.to_idx] = undo.captured_before

        if undo.moved_reveal:
            side = piece_color(undo.moved_before)
            self.pool[side][undo.moved_reveal] += 1
            self.pool_total[side] += 1
        if undo.captured_reveal and undo.captured_before:
            side = piece_color(undo.captured_before)
            self.pool[side][undo.captured_reveal] += 1
            self.pool_total[side] += 1

    def _piece_threat_targets(self, origin: int):
        board = self.board
        piece = board[origin]
        if not piece:
            return
        p_type = piece_type(piece)
        side = piece_color(piece)
        if p_type == TYPE_ROOK:
            for ray in RAYS[origin]:
                for j in ray:
                    if board[j]:
                        yield j
                        break
        elif p_type == TYPE_CANNON:
            for ray in RAYS[origin]:
                screen = False
                for j in ray:
                    if not screen:
                        if board[j]:
                            screen = True
                    elif board[j]:
                        yield j
                        break
        elif p_type == TYPE_KNIGHT:
            for j, leg in KNIGHT_MOVES[origin]:
                if not board[leg] and board[j]:
                    yield j
        elif p_type == TYPE_ELEPHANT:
            for j, eye in ELEPHANT_MOVES[origin]:
                if not board[eye] and board[j]:
                    yield j
        elif p_type == TYPE_ADVISOR:
            for j in ADVISOR_MOVES[origin]:
                if board[j]:
                    yield j
        elif p_type == TYPE_KING:
            for j in KING_ADJ[origin]:
                if PALACE[side][j] and board[j]:
                    yield j
            # 照面（飞将）：同列无遮挡时的对方将/帅。
            for d in (UP, DOWN):
                for j in RAYS[origin][d]:
                    target = board[j]
                    if target:
                        if piece_color(target) != side and piece_type(target) == TYPE_KING:
                            yield j
                        break
        elif p_type == TYPE_PAWN:
            fwd = PAWN_FWD[side][origin]
            if fwd != -1 and board[fwd]:
                yield fwd
            if CROSSED_RIVER[side][origin]:
                left = PAWN_LEFT[origin]
                if left != -1 and board[left]:
                    yield left
                right = PAWN_RIGHT[origin]
                if right != -1 and board[right]:
                    yield right

    def _checker_squares(self, side: int, king_idx: int) -> list[int]:
        """返回 side 方正在攻击 king_idx 的棋子所在格列表（用于按子追踪长将）。"""
        if king_idx < 0:
            return []
        checkers = []
        for idx, piece in enumerate(self.board):
            if not piece or piece_color(piece) != side:
                continue
            for target in self._piece_threat_targets(idx):
                if target == king_idx:
                    checkers.append(idx)
                    break
        return checkers

    def _update_check(self, side: int, from_idx: int, to_idx: int, gave_check: bool) -> None:
        """按“同一枚棋子连续将军”更新长将计数（ADR-0003）。

        续将条件（二选一）：
        - 上一轮将军的棋子本轮自己走动后仍在将军（``from_idx == check_anchor``）；
        - 该棋子未移动（``check_anchor`` 格上仍是它）且仍攻击对方将/帅（闪露将）。
        换子接手将军时计数重置为 1；双闪将无可辨认的唯一锚点时不指定棋子。
        """
        if not gave_check:
            self.consecutive_checks[side] = 0
            self.check_anchor[side] = -1
            return
        checkers = self._checker_squares(side, self.king_pos[1 - side])
        anchor = self.check_anchor[side]
        same_piece = (
            anchor >= 0
            and (
                (from_idx == anchor and to_idx in checkers)
                or (from_idx != anchor and to_idx != anchor and anchor in checkers)
            )
        )
        if same_piece:
            self.consecutive_checks[side] += 1
            self.check_anchor[side] = to_idx if from_idx == anchor else anchor
            return
        self.consecutive_checks[side] = 1
        if to_idx in checkers:
            self.check_anchor[side] = to_idx
        elif len(checkers) == 1:
            self.check_anchor[side] = checkers[0]
        else:
            self.check_anchor[side] = -1

    def _clear_chase(self, side: int) -> None:
        self.chase_streak[side] = 0
        self.chase_target[side] = -1
        self.chase_target_code[side] = 0
        self.chase_anchor[side] = -1

    def _update_chase(self, side: int, from_idx: int, to_idx: int, captured: int) -> None:
        rules = self.rules
        if rules.long_chase_rounds <= 0:
            self._clear_chase(side)
            return
        if not captured:
            target = self.chase_target[side]
            code = self.chase_target_code[side]
            # 续捉要求是同一枚棋子：它上一轮追完停在 chase_anchor，本轮必须从该格再走。
            # 受害子的公开移动已在 apply_move 中更新目标格，不猜测同码棋子的身份。
            if target >= 0 and code and from_idx == self.chase_anchor[side]:
                threats = list(self._piece_threat_targets(to_idx))
                if self.board[target] == code and target in threats:
                    self.chase_streak[side] += 1
                    self.chase_anchor[side] = to_idx
                    return
        best_idx = -1
        best_value = -1
        best_count = 0
        for j in self._piece_threat_targets(to_idx):
            victim = self.board[j]
            if piece_color(victim) == side:
                continue
            v_type = piece_type(victim)
            if v_type in (TYPE_KING, TYPE_PAWN):
                continue
            value = CHASE_VALUES.get(v_type, 0)
            if value > best_value:
                best_idx = j
                best_value = value
                best_count = 1
            elif value == best_value:
                best_count += 1
        # 与续捉身份歧义一致：新目标不唯一时不依赖绝对扫描方向指定一枚。
        if best_count == 1:
            self.chase_streak[side] = 1
            self.chase_target[side] = best_idx
            self.chase_target_code[side] = self.board[best_idx]
            self.chase_anchor[side] = to_idx
        else:
            self._clear_chase(side)

    def _banned_for_side(self):
        rules = self.rules
        if not rules.forbidden_moves:
            return False, False
        side = self.side
        ban_check = rules.long_check_rounds > 0 and self.consecutive_checks[side] >= rules.long_check_rounds
        ban_chase = (
            rules.long_chase_rounds > 0
            and self.chase_streak[side] >= rules.long_chase_rounds
            and self.chase_target[side] >= 0
        )
        return ban_check, ban_chase

    def iter_pseudo_moves(self, side=None):
        if side is None:
            side = self.side
        board = self.board
        for i, p in enumerate(board):
            if not p or piece_color(p) != side:
                continue
            p_type = piece_type(p)
            if p_type == TYPE_ROOK:
                for ray in RAYS[i]:
                    for j in ray:
                        target = board[j]
                        if not target:
                            yield (i, j)
                            continue
                        if piece_color(target) != side:
                            yield (i, j)
                        break
            elif p_type == TYPE_CANNON:
                for ray in RAYS[i]:
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
                                if piece_color(target) != side:
                                    yield (i, j)
                                break
            elif p_type == TYPE_KNIGHT:
                for j, leg in KNIGHT_MOVES[i]:
                    if board[leg]:
                        continue
                    target = board[j]
                    if not target or piece_color(target) != side:
                        yield (i, j)
            elif p_type == TYPE_ELEPHANT:
                for j, eye in ELEPHANT_MOVES[i]:
                    if board[eye]:
                        continue
                    target = board[j]
                    if not target or piece_color(target) != side:
                        yield (i, j)
            elif p_type == TYPE_ADVISOR:
                dark = is_dark(p)
                if dark and not PALACE[side][i]:
                    continue
                for j in ADVISOR_MOVES[i]:
                    if dark and not PALACE[side][j]:
                        continue
                    target = board[j]
                    if not target or piece_color(target) != side:
                        yield (i, j)
            elif p_type == TYPE_KING:
                for j in KING_ADJ[i]:
                    if not PALACE[side][j]:
                        continue
                    target = board[j]
                    if not target or piece_color(target) != side:
                        yield (i, j)
                for d in (UP, DOWN):
                    for j in RAYS[i][d]:
                        target = board[j]
                        if target:
                            if piece_color(target) != side and piece_type(target) == TYPE_KING:
                                yield (i, j)
                            break
            elif p_type == TYPE_PAWN:
                fwd = PAWN_FWD[side][i]
                if fwd != -1:
                    target = board[fwd]
                    if not target or piece_color(target) != side:
                        yield (i, fwd)
                if CROSSED_RIVER[side][i]:
                    left = PAWN_LEFT[i]
                    if left != -1:
                        target = board[left]
                        if not target or piece_color(target) != side:
                            yield (i, left)
                    right = PAWN_RIGHT[i]
                    if right != -1:
                        target = board[right]
                        if not target or piece_color(target) != side:
                            yield (i, right)

    def is_square_attacked(self, by_side: int, idx: int) -> bool:
        board = self.board
        for ray in RAYS[idx]:
            screen = False
            for j in ray:
                target = board[j]
                if not target:
                    continue
                t_type = piece_type(target)
                if not screen:
                    if piece_color(target) == by_side and (t_type == TYPE_ROOK or t_type == TYPE_KING):
                        return True
                    screen = True
                else:
                    if piece_color(target) == by_side and t_type == TYPE_CANNON:
                        return True
                    break

        for j, leg in KNIGHT_ATTACKS[idx]:
            if board[leg]:
                continue
            target = board[j]
            if target and piece_color(target) == by_side and piece_type(target) == TYPE_KNIGHT:
                return True

        for j, eye in ELEPHANT_MOVES[idx]:
            if board[eye]:
                continue
            target = board[j]
            if target and piece_color(target) == by_side and piece_type(target) == TYPE_ELEPHANT:
                return True

        for j in ADVISOR_MOVES[idx]:
            target = board[j]
            if not target or piece_color(target) != by_side or piece_type(target) != TYPE_ADVISOR:
                continue
            if is_dark(target):
                if not (PALACE[by_side][idx] and PALACE[by_side][j]):
                    continue
            return True

        r = ROWS[idx]
        c = COLS[idx]
        if by_side == RED:
            if r + 1 < BOARD_H:
                target = board[index(r + 1, c)]
                if target and piece_color(target) == RED and piece_type(target) == TYPE_PAWN:
                    return True
            if c - 1 >= 0:
                p_idx = index(r, c - 1)
                target = board[p_idx]
                if (
                    target
                    and piece_color(target) == RED
                    and piece_type(target) == TYPE_PAWN
                    and CROSSED_RIVER[RED][p_idx]
                ):
                    return True
            if c + 1 < BOARD_W:
                p_idx = index(r, c + 1)
                target = board[p_idx]
                if (
                    target
                    and piece_color(target) == RED
                    and piece_type(target) == TYPE_PAWN
                    and CROSSED_RIVER[RED][p_idx]
                ):
                    return True
        else:
            if r - 1 >= 0:
                target = board[index(r - 1, c)]
                if target and piece_color(target) == BLACK and piece_type(target) == TYPE_PAWN:
                    return True
            if c - 1 >= 0:
                p_idx = index(r, c - 1)
                target = board[p_idx]
                if (
                    target
                    and piece_color(target) == BLACK
                    and piece_type(target) == TYPE_PAWN
                    and CROSSED_RIVER[BLACK][p_idx]
                ):
                    return True
            if c + 1 < BOARD_W:
                p_idx = index(r, c + 1)
                target = board[p_idx]
                if (
                    target
                    and piece_color(target) == BLACK
                    and piece_type(target) == TYPE_PAWN
                    and CROSSED_RIVER[BLACK][p_idx]
                ):
                    return True

        return False

    def is_in_check(self, side: int) -> bool:
        king_idx = self.king_pos[side]
        if king_idx < 0:
            return True
        return self.is_square_attacked(1 - side, king_idx)

    def _iter_legal_moves_fast(self, side):
        """无长将/长捉禁着时的快速合法性过滤（不触发抽子与追踪更新）。"""
        board = self.board
        king_side = self.king_pos[side]
        if king_side < 0:
            return
        for move in self.iter_pseudo_moves(side):
            frm, to = move
            moving = board[frm]
            captured = board[to]
            board[to] = moving
            board[frm] = 0
            moved_king = piece_color(moving) == side and piece_type(moving) == TYPE_KING
            if moved_king:
                self.king_pos[side] = to
            safe = not self.is_square_attacked(1 - side, self.king_pos[side])
            if moved_king:
                self.king_pos[side] = king_side
            board[frm] = moving
            board[to] = captured
            if safe:
                yield move

    def _iter_legal_moves(self):
        side = self.side
        ban_check, ban_chase = self._banned_for_side()
        if not ban_check and not ban_chase:
            yield from self._iter_legal_moves_fast(side)
            return
        # 试走用临时 rng：查询（legal_moves/game_outcome）不得消耗 self.rng，
        # 否则同 seed 的后续揭示会随查询次数改变（P2-8）。
        probe_rng = random.Random(0)
        for move in self.iter_pseudo_moves(side):
            prev_check = self.consecutive_checks[side]
            prev_streak = self.chase_streak[side]
            undo = self.apply_move(move, rng=probe_rng)
            legal = not self.is_in_check(side)
            if legal and ban_check and self.consecutive_checks[side] > prev_check:
                legal = False
            # 续捉按次数增加判定，不比较目标格：受害子逃跑会合法换格（P1-6）。
            if legal and ban_chase and self.chase_streak[side] > prev_streak:
                legal = False
            self.undo_move(undo)
            if legal:
                yield move

    def _move_scan(self):
        side = self.side
        ban_check, ban_chase = self._banned_for_side()
        if not ban_check and not ban_chase:
            count = 0
            for _ in self._iter_legal_moves_fast(side):
                count += 1
            return count, count, False, False
        self_legal_count = 0
        legal_count = 0
        banned_long_check = False
        banned_long_chase = False
        probe_rng = random.Random(0)  # 同上：终局查询不得推进 self.rng
        for move in self.iter_pseudo_moves(side):
            prev_check = self.consecutive_checks[side]
            prev_streak = self.chase_streak[side]
            undo = self.apply_move(move, rng=probe_rng)
            legal = not self.is_in_check(side)
            if legal:
                self_legal_count += 1
                if ban_check and self.consecutive_checks[side] > prev_check:
                    banned_long_check = True
                elif ban_chase and self.chase_streak[side] > prev_streak:
                    banned_long_chase = True
                else:
                    legal_count += 1
            self.undo_move(undo)
        return self_legal_count, legal_count, banned_long_check, banned_long_chase

    def legal_moves(self):
        return list(self._iter_legal_moves())

    def has_legal_moves(self) -> bool:
        for _ in self._iter_legal_moves():
            return True
        return False

    def game_result(self):
        outcome, _ = self.game_outcome()
        if outcome is None:
            return None
        if outcome == RESULT_DRAW:
            return RESULT_DRAW
        return RED if outcome == RESULT_RED_WIN else BLACK

    def game_outcome(self):
        rules = self.rules
        self_legal, legal, banned_check, banned_chase = self._move_scan()
        if self_legal == 0:
            winner = 1 - self.side
            result = RESULT_RED_WIN if winner == RED else RESULT_BLACK_WIN
            reason = REASON_CHECKMATE if self.is_in_check(self.side) else REASON_STALEMATE
            return result, reason
        if legal == 0 and (banned_check or banned_chase):
            result = RESULT_BLACK_WIN if self.side == RED else RESULT_RED_WIN
            reason = REASON_LONG_CHECK if banned_check else REASON_LONG_CHASE
            return result, reason
        if not rules.forbidden_moves:
            offender = 1 - self.side
            if rules.long_check_rounds > 0 and self.consecutive_checks[offender] >= rules.long_check_rounds:
                result = RESULT_BLACK_WIN if offender == RED else RESULT_RED_WIN
                return result, REASON_LONG_CHECK
        if rules.repetition_limit > 0:
            count = self.position_counts.get(self.position_key, 0)
            if count >= rules.repetition_limit:
                if rules.repetition_is_loss:
                    offender = 1 - self.side
                    result = RESULT_BLACK_WIN if offender == RED else RESULT_RED_WIN
                    return result, REASON_REPETITION
                if max(self.consecutive_checks) < 2 and max(self.chase_streak) < 2:
                    return RESULT_DRAW, REASON_REPETITION
        if rules.no_capture_plies > 0 and self.no_capture_plies >= rules.no_capture_plies:
            return RESULT_DRAW, REASON_NO_CAPTURE
        return None, None

    def fast_outcome(self):
        rules = self.rules
        if self.king_pos[RED] < 0:
            return RESULT_BLACK_WIN, REASON_CHECKMATE
        if self.king_pos[BLACK] < 0:
            return RESULT_RED_WIN, REASON_CHECKMATE
        if not rules.forbidden_moves:
            offender = 1 - self.side
            if rules.long_check_rounds > 0 and self.consecutive_checks[offender] >= rules.long_check_rounds:
                result = RESULT_BLACK_WIN if offender == RED else RESULT_RED_WIN
                return result, REASON_LONG_CHECK
        if rules.repetition_limit > 0:
            count = self.position_counts.get(self.position_key, 0)
            if count >= rules.repetition_limit:
                if rules.repetition_is_loss:
                    offender = 1 - self.side
                    result = RESULT_BLACK_WIN if offender == RED else RESULT_RED_WIN
                    return result, REASON_REPETITION
                if max(self.consecutive_checks) < 2 and max(self.chase_streak) < 2:
                    return RESULT_DRAW, REASON_REPETITION
        if rules.no_capture_plies > 0 and self.no_capture_plies >= rules.no_capture_plies:
            return RESULT_DRAW, REASON_NO_CAPTURE
        return None, None

    def outcome_checked(self):
        """fast_outcome 的语义校正版：终局候选交由完整裁判确认优先级。

        ``fast_outcome`` 是廉价近似，不检查无合法着法；当和棋阈值与将死/困毙
        或长将/长捉禁着恰好同触时可能遮蔽判负。搜索/回放等需要与 ``game_outcome``
        优先级一致的场合用本方法；fast 返回 None 时仍由调用方走 leaf_moves
        检查。
        """
        outcome, reason = self.fast_outcome()
        if outcome is not None:
            return self.game_outcome()
        return outcome, reason


def board_to_text(state: GameState, show_dark: bool = True) -> str:
    empty_cell = FULL_WIDTH_SPACE * 2
    rows = []
    for r in range(BOARD_H):
        cells = []
        for c in range(BOARD_W):
            piece = state.board[index(r, c)]
            if not piece:
                cells.append(empty_cell)
                continue
            name = PIECE_NAMES[piece_color(piece)][piece_type(piece)]
            if is_dark(piece):
                cell = f"暗{name}" if show_dark else f"暗{FULL_WIDTH_SPACE}"
            else:
                cell = f"{name}{FULL_WIDTH_SPACE}"
            cells.append(cell)
        rows.append(FULL_WIDTH_SPACE.join(cells))
    return "\n".join(rows)


def print_board(state: GameState, show_dark: bool = True) -> None:
    print(board_to_text(state, show_dark))


FEN_LETTERS = {
    TYPE_KING: "K",
    TYPE_ADVISOR: "A",
    TYPE_ELEPHANT: "E",
    TYPE_ROOK: "R",
    TYPE_CANNON: "C",
    TYPE_KNIGHT: "H",
    TYPE_PAWN: "P",
}
FEN_LETTER_TO_TYPE = {letter: ptype for ptype, letter in FEN_LETTERS.items()}
FEN_CAPTURE_ORDER = (TYPE_ADVISOR, TYPE_ELEPHANT, TYPE_ROOK, TYPE_CANNON, TYPE_KNIGHT, TYPE_PAWN)
FEN_CAPTURE_PATTERN = re.compile(r"([AERCHP])(\d+)", re.IGNORECASE)
# FEN 第 7 字段（可选）的默认值，顺序为：
# 连续将军（红/黑）、将军锚点（红/黑）、追击计数（红/黑）、
# 追击目标格（红/黑）、追击目标棋子码（红/黑）、追击锚点（红/黑）。
RULE_STATE_DEFAULT = [0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1]


def square_to_name(idx: int) -> str:
    return f"{chr(ord('a') + COLS[idx])}{9 - ROWS[idx]}"


def name_to_square(name: str) -> int:
    col = ord(name[0].lower()) - ord("a")
    row = 9 - int(name[1])
    return index(row, col)


def format_move_uci(move, reveal_type: int | None = None) -> str:
    text = f"{square_to_name(move[0])}{square_to_name(move[1])}"
    if reveal_type:
        text += FEN_LETTERS[reveal_type].lower()
    return text


def fake_type_for_square(idx: int) -> int:
    row, col = ROWS[idx], COLS[idx]
    if row in (0, BOARD_H - 1):
        return BACK_RANK[col]
    if (row, col) in ((2, 1), (2, 7), (7, 1), (7, 7)):
        return TYPE_CANNON
    if row in (3, 6) and col in (0, 2, 4, 6, 8):
        return TYPE_PAWN
    raise ValueError(f"dark piece on invalid square {square_to_name(idx)}")


def _captured_counts(state: GameState):
    counts = {RED: {}, BLACK: {}}
    for color in (RED, BLACK):
        for t in POOL_TYPES:
            bright = 0
            for p in state.board:
                if p and piece_color(p) == color and not is_dark(p) and piece_type(p) == t:
                    bright += 1
            lost = POOL_INIT[t] - state.pool[color][t] - bright
            if lost > 0:
                counts[color][t] = lost
    return counts


def _format_captured(counts) -> str:
    parts = []
    for color, letter_case in ((RED, str.upper), (BLACK, str.lower)):
        for t in FEN_CAPTURE_ORDER:
            n = counts[color].get(t, 0)
            if n:
                parts.append(f"{letter_case(FEN_LETTERS[t])}{n}")
    return "".join(parts)


def _parse_captured(text: str):
    counts = {RED: {}, BLACK: {}}
    for letter, digits in FEN_CAPTURE_PATTERN.findall(text):
        color = RED if letter.isupper() else BLACK
        t = FEN_LETTER_TO_TYPE[letter.upper()]
        counts[color][t] = counts[color].get(t, 0) + int(digits)
    return counts


def to_fen(state: GameState) -> str:
    rows = []
    for r in range(BOARD_H):
        parts = []
        empty = 0
        for c in range(BOARD_W):
            piece = state.board[index(r, c)]
            if not piece:
                empty += 1
                continue
            if empty:
                parts.append(str(empty))
                empty = 0
            if is_dark(piece):
                parts.append("X" if piece_color(piece) == RED else "x")
            else:
                letter = FEN_LETTERS[piece_type(piece)]
                parts.append(letter if piece_color(piece) == RED else letter.lower())
        if empty:
            parts.append(str(empty))
        rows.append("".join(parts))
    placement = "/".join(rows)
    captured = _captured_counts(state)
    captured_text = _format_captured(captured)
    if captured_text:
        placement += f"[{captured_text}]"
    side = "w" if state.side == RED else "b"
    fen = f"{placement} {side} - - {state.no_capture_plies} 1"
    # 可选第 7 字段：规则判定状态（长将/长捉的按子追踪 + 当前局面重复次数）。
    # 仅在非默认时输出，保持旧 FEN 兼容；更早局面的重复历史无法用 FEN 表达。
    rule_state = (
        list(state.consecutive_checks)
        + list(state.check_anchor)
        + list(state.chase_streak)
        + list(state.chase_target)
        + list(state.chase_target_code)
        + list(state.chase_anchor)
    )
    rep_count = state.position_counts.get(state.position_key, 1)
    if rule_state != RULE_STATE_DEFAULT or rep_count != 1:
        fen += " " + ",".join(str(value) for value in rule_state + [rep_count])
    return fen


def from_fen(fen: str, rng=None, rules=None) -> GameState:
    parts = fen.split()
    if len(parts) < 2:
        raise ValueError("invalid Jeiqi FEN")
    field = parts[0]
    captured = {RED: {}, BLACK: {}}
    if "[" in field:
        head, rest = field.split("[", 1)
        captured = _parse_captured(rest.split("]", 1)[0])
        field = head
    side = BLACK if parts[1].lower().startswith("b") else RED
    clock = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
    rule_state = list(RULE_STATE_DEFAULT)
    rep_count = 1
    if len(parts) > 6:
        values = [int(value) for value in parts[6].split(",")]
        if len(values) != 13:
            raise ValueError("invalid rule state field")
        rule_state = values[:12]
        rep_count = max(1, values[12])
    board = [0] * BOARD_SIZE
    king_pos = [-1, -1]
    rows = field.split("/")
    if len(rows) != BOARD_H:
        raise ValueError("invalid board rows")
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
                continue
            if c >= BOARD_W:
                raise ValueError("rank overflow")
            idx = index(r, c)
            if ch in ("X", "x"):
                color = RED if ch == "X" else BLACK
                fake = fake_type_for_square(idx)
                if fake == TYPE_KING:
                    raise ValueError("dark king")
                board[idx] = make_piece(fake, color, 1)
            else:
                upper = ch.upper()
                if upper not in FEN_LETTER_TO_TYPE:
                    raise ValueError(f"unknown piece letter {ch}")
                color = RED if ch.isupper() else BLACK
                ptype = FEN_LETTER_TO_TYPE[upper]
                board[idx] = make_piece(ptype, color, 0)
                if ptype == TYPE_KING:
                    king_pos[color] = idx
            c += 1
        if c != BOARD_W:
            raise ValueError("invalid rank width")
    pool = [[0] * 8, [0] * 8]
    pool_total = [0, 0]
    for color in (RED, BLACK):
        for t in POOL_TYPES:
            bright = 0
            for p in board:
                if p and piece_color(p) == color and not is_dark(p) and piece_type(p) == t:
                    bright += 1
            remain = POOL_INIT[t] - bright - captured[color].get(t, 0)
            if remain < 0:
                raise ValueError("pool reconstruction failed")
            pool[color][t] = remain
            pool_total[color] += remain
    state = GameState(board, side, pool, pool_total, king_pos, rng, rules)
    state.no_capture_plies = clock
    state.consecutive_checks = rule_state[0:2]
    state.check_anchor = rule_state[2:4]
    state.chase_streak = rule_state[4:6]
    state.chase_target = rule_state[6:8]
    state.chase_target_code = rule_state[8:10]
    state.chase_anchor = rule_state[10:12]
    state.position_key = state._position_key()
    state.position_counts = {state.position_key: rep_count}
    return state


_immutable_integer_pools = None


def _load_rules_backend():
    global _immutable_integer_pools
    import hashlib
    import os
    from pathlib import Path
    import sys

    requested = os.environ.get("JIEQI_RULES_BACKEND", "auto").lower()
    if requested not in ("auto", "python", "cython"):
        raise ValueError("JIEQI_RULES_BACKEND must be auto, python, or cython")
    if requested == "python":
        return "python", None, None
    try:
        import _jieqi_rules as native
        if getattr(native, "API_VERSION", None) != 1 or getattr(native, "RULESET_VERSION", None) != RULESET_VERSION:
            raise ImportError("incompatible native rules API; rebuild with tools/build_rules.py")
        source = Path(__file__).with_name("native") / "jieqi_rules.pyx"
        if getattr(native, "SOURCE_SHA256", None) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ImportError("stale native rules binary; rebuild with tools/build_rules.py")
        guard = getattr(native, "immutable_integer_pools", None)
        if not callable(guard):
            raise ImportError("missing native snapshot guard; rebuild with tools/build_rules.py")
        binary = Path(native.__file__).resolve()
        with binary.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        native_record = dict(path=str(binary), sha256=digest, bytes=binary.stat().st_size,
                             api_version=native.API_VERSION, ruleset_version=native.RULESET_VERSION,
                             source_sha256=native.SOURCE_SHA256)
        native.initialize(sys.modules[__name__])
    except (ImportError, OSError) as error:
        if requested == "cython":
            raise ImportError("Cython rules backend unavailable") from error
        return "python", str(error), None
    GameState.is_square_attacked = native.is_square_attacked
    _immutable_integer_pools = guard
    return "cython", None, native_record


GameState._is_square_attacked_python = GameState.is_square_attacked
RULES_BACKEND, RULES_BACKEND_ERROR, RULES_BACKEND_NATIVE = _load_rules_backend()


def _legal_methods_sha256():
    import hashlib
    import marshal
    import types

    def normalized(code):
        constants = tuple(normalized(c) if isinstance(c, types.CodeType) else c for c in code.co_consts)
        return code.replace(co_filename="jieqi.py", co_consts=constants)

    methods = (GameState._iter_pseudo_moves_python, GameState._iter_legal_moves_fast_python,
               *GameState._legal_helpers_python)
    return hashlib.sha256(marshal.dumps(tuple(
        (normalized(method.__code__), method.__defaults__, method.__kwdefaults__) for method in methods
    ))).hexdigest()


def _load_legal_backend():
    import hashlib
    import os
    from pathlib import Path
    import sys

    requested = os.environ.get("JIEQI_LEGAL_BACKEND", "python").lower()
    if requested not in ("auto", "python", "cython"):
        raise ValueError("JIEQI_LEGAL_BACKEND must be auto, python, or cython")
    GameState._iter_legal_moves_fast = GameState._iter_legal_moves_fast_python
    if requested == "python":
        return "python", None, None
    try:
        import _jieqi_legal as native
        if getattr(native, "API_VERSION", None) != 1 or getattr(native, "RULESET_VERSION", None) != RULESET_VERSION:
            raise ImportError("incompatible native legal API; rebuild with tools/build_legal.py")
        source = Path(__file__).with_name("native") / "jieqi_legal.pyx"
        if getattr(native, "SOURCE_SHA256", None) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ImportError("stale native legal source; rebuild with tools/build_legal.py")
        if getattr(native, "METHODS_SHA256", None) != _legal_methods_sha256():
            raise ImportError("stale native legal methods; rebuild with tools/build_legal.py")
        binary = Path(native.__file__).resolve()
        with binary.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        record = dict(path=str(binary), sha256=digest, bytes=binary.stat().st_size,
                      api_version=native.API_VERSION, ruleset_version=native.RULESET_VERSION,
                      source_sha256=native.SOURCE_SHA256, methods_sha256=native.METHODS_SHA256,
                      methods=["_iter_legal_moves_fast"])
        module = sys.modules[__name__]
        native.initialize(module)
    except (ImportError, OSError) as error:
        if requested == "cython":
            raise ImportError("Cython legal backend unavailable") from error
        return "python", str(error), None
    GameState._native_legal_module = module
    GameState._iter_legal_moves_fast = native.iter_legal_moves_fast
    return "cython", None, record


GameState._iter_pseudo_moves_python = GameState.iter_pseudo_moves
GameState._iter_legal_moves_fast_python = GameState._iter_legal_moves_fast
GameState._legal_helpers_python = (piece_color, piece_type, is_dark)
LEGAL_BACKEND, LEGAL_BACKEND_ERROR, LEGAL_BACKEND_NATIVE = _load_legal_backend()
