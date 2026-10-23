from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

import jieqi as jq


# =============================================================================
# 常量与辅助函数
# =============================================================================

# 预先缓存九宫格索引，用于快速查找将帅位置
# PALACE_INDEX[0] 为黑方九宫，PALACE_INDEX[1] 为红方九宫
# 这样可以避免每次遍历整个棋盘寻找将帅
PALACE_INDEX = [
    [i for i in range(jq.BOARD_SIZE) if jq.PALACE[side][i]]
    for side in (jq.BLACK, jq.RED)
]

# 根据棋盘推断双方将帅位置，用于构造 GameState。
def infer_king_pos(board: list[int]) -> list[int]:
    king_pos = [-1, -1]
    for side in (jq.BLACK, jq.RED):
        for idx in PALACE_INDEX[side]:
            piece = board[idx]
            if piece and jq.piece_type(piece) == jq.TYPE_KING and jq.piece_color(piece) == side:
                king_pos[side] = idx
                break
    # 保险：若出现非法局面（将帅不在九宫），回退全盘扫描。
    if king_pos[jq.BLACK] == -1 or king_pos[jq.RED] == -1:
        for idx, piece in enumerate(board):
            if piece and jq.piece_type(piece) == jq.TYPE_KING:
                side = jq.piece_color(piece)
                if king_pos[side] == -1:
                    king_pos[side] = idx
    return king_pos


# 从给定暗盒统计中“随机移除”未知被吃的棋子，用于确定化。
def sample_unknown_removals(counts: list[int], remove: int, rng: random.Random) -> list[int]:
    counts = counts.copy()
    total = sum(counts)
    if remove < 0 or remove > total:
        raise ValueError("invalid removal count")
    for _ in range(remove):
        roll = rng.randrange(total)
        acc = 0
        for p_type in jq.POOL_TYPES:
            acc += counts[p_type]
            if roll < acc:
                counts[p_type] -= 1
                total -= 1
                break
    return counts


@dataclass(slots=True)
class InformationState:
    # 棋盘上“暗子/明子”的位置是已知的，但暗子的真实类型未知。
    board: list[int]
    # 当前轮到哪一方走子。
    side: int
    # 观察者认为双方暗盒中“各类棋子”的剩余上界。
    pool_known: list[list[int]]
    # 观察者认为双方暗盒剩余总数。
    pool_total_known: list[int]
    # 观察者认为“未知被吃的暗子数量”（只扣总数，不扣类型）。
    unknown_removals: list[int]
    # 无吃子步数（用于和棋判定）。
    no_capture_plies: int
    # 连续将军次数（用于长将判负）。
    consecutive_checks: list[int]
    # 以下字段用于把“规则判定所需的公开历史”带入确定化后的局面，
    # 否则搜索看不到长将/长捉禁着与已有的重复局面（ADR-0003/0004）。
    rules: jq.RulesConfig = jq.RULESET_TIANTIAN
    # 上一轮将军棋子所在格（每色一个，-1 表示无）。
    check_anchor: list[int] = field(default_factory=lambda: [-1, -1])
    # 连续追击同一目标的计数与目标信息（每色一个）。
    chase_streak: list[int] = field(default_factory=lambda: [0, 0])
    chase_target: list[int] = field(default_factory=lambda: [-1, -1])
    chase_target_code: list[int] = field(default_factory=lambda: [0, 0])
    chase_anchor: list[int] = field(default_factory=lambda: [-1, -1])
    # 公共局面重复计数（(side, board) -> count）；None 表示仅当前局面记 1。
    position_counts: dict | None = None
    # 只扣公开移动揭示，不扣任何私有捕获类型；用于重建模拟对手的知识。
    public_pool_known: list[list[int]] | None = None
    # v2 训练样本只有当前重复计数；不能把缺失历史冒充为完整公开历史。
    repetition_history_complete: bool = True

    # 基于“观察者视角”的信息做一次确定化，生成可模拟的 GameState。
    # 注意：棋盘上的暗子类型在棋盘里依然以“初始站位类型”表示，走法由 jieqi.py 控制。
    def determinize(self, rng: random.Random) -> jq.GameState:
        pool = []
        pool_total = []
        for side in (jq.BLACK, jq.RED):
            counts = self.pool_known[side].copy()
            total_known = self.pool_total_known[side]
            missing = sum(counts) - total_known
            if missing < 0:
                raise ValueError("pool totals inconsistent")
            if missing:
                # 对“未知少了哪些子”的情况进行随机采样。
                counts = sample_unknown_removals(counts, missing, rng)
            pool.append(counts)
            pool_total.append(sum(counts))
        king_pos = infer_king_pos(self.board)
        state = jq.GameState(self.board.copy(), self.side, pool, pool_total, king_pos, rng, self.rules)
        # 同步对局的公共历史信息，以保持和棋/长将/长捉/重复判定一致。
        state.no_capture_plies = self.no_capture_plies
        state.consecutive_checks = self.consecutive_checks.copy()
        state.check_anchor = self.check_anchor.copy()
        state.chase_streak = self.chase_streak.copy()
        state.chase_target = self.chase_target.copy()
        state.chase_target_code = self.chase_target_code.copy()
        state.chase_anchor = self.chase_anchor.copy()
        state.position_key = state._position_key()
        if self.position_counts is not None:
            state.position_counts = dict(self.position_counts)
            state.position_counts.setdefault(state.position_key, 1)
        else:
            state.position_counts = {state.position_key: 1}
        return state


# BeliefModel 保存“每个观察者”的暗盒认知。
# pool_counts[observer][side][type]：观察者认为该方该类棋子的剩余上界（未扣未知被吃）。
# pool_total[observer][side]：观察者认为该方暗盒剩余总数（已扣未知被吃）。
# unknown_removals[observer][side]：观察者认为该方暗盒中“未知被吃”的数量。
class BeliefModel:
    __slots__ = ("pool_counts", "pool_total", "unknown_removals", "public_pool_counts")

    def __init__(self, pool_counts, pool_total, unknown_removals, public_pool_counts=None) -> None:
        self.pool_counts = pool_counts
        self.pool_total = pool_total
        self.unknown_removals = unknown_removals
        # 兼容旧构造器：每方对自己暗池的类型上界只由公开揭示改变。
        # 正常游戏/克隆显式传入独立公共账本，搜索不读取真实对手 observer 槽。
        self.public_pool_counts = [
            row.copy() for row in (public_pool_counts if public_pool_counts is not None else
                                  (pool_counts[jq.BLACK][jq.BLACK], pool_counts[jq.RED][jq.RED]))
        ]

    @classmethod
    def from_game_state(cls, state: jq.GameState) -> "BeliefModel":
        # 起始局面：双方对自己/对手暗盒都“完全知道”，未知被吃数量为 0。
        pool_counts = [
            [state.pool[jq.BLACK].copy(), state.pool[jq.RED].copy()],
            [state.pool[jq.BLACK].copy(), state.pool[jq.RED].copy()],
        ]
        pool_total = [
            [state.pool_total[jq.BLACK], state.pool_total[jq.RED]],
            [state.pool_total[jq.BLACK], state.pool_total[jq.RED]],
        ]
        unknown_removals = [
            [0, 0],
            [0, 0],
        ]
        return cls(pool_counts, pool_total, unknown_removals, state.pool)

    @classmethod
    def for_simulation(cls, root_info: InformationState, root_player: int,
                       world: jq.GameState) -> "BeliefModel":
        """只从根信息和根一致粒子重建双方观测，不读取真实对手秘密。

        对手知道它先前捕获的根方暗子，所以对根方池的认知等于本粒子的池；
        对手不知道自身被吃类型，所以对自身池只保留公共计数上界。
        先前私有捕获采用组成计数抽象；搜索内新观测另保留逐事件历史。
        """
        if root_info.public_pool_known is None:
            raise ValueError("simulation requires public pool counts")
        opponent = 1 - root_player
        counts = [[row.copy() for row in root_info.pool_known] for _ in (jq.BLACK, jq.RED)]
        counts[opponent][opponent] = root_info.public_pool_known[opponent].copy()
        counts[opponent][root_player] = world.pool[root_player].copy()
        totals = [root_info.pool_total_known.copy() for _ in (jq.BLACK, jq.RED)]
        unknown = [root_info.unknown_removals.copy() for _ in (jq.BLACK, jq.RED)]
        unknown[opponent][root_player] = 0
        unknown[opponent][opponent] = sum(counts[opponent][opponent]) - totals[opponent][opponent]
        if unknown[opponent][opponent] < 0:
            raise ValueError("public pool totals inconsistent")
        return cls(counts, totals, unknown, root_info.public_pool_known)

    # 深拷贝，用于每次模拟独立维护 belief。
    def clone(self) -> "BeliefModel":
        pool_counts = [
            [self.pool_counts[0][0].copy(), self.pool_counts[0][1].copy()],
            [self.pool_counts[1][0].copy(), self.pool_counts[1][1].copy()],
        ]
        pool_total = [
            [self.pool_total[0][0], self.pool_total[0][1]],
            [self.pool_total[1][0], self.pool_total[1][1]],
        ]
        unknown_removals = [
            [self.unknown_removals[0][0], self.unknown_removals[0][1]],
            [self.unknown_removals[1][0], self.unknown_removals[1][1]],
        ]
        return BeliefModel(pool_counts, pool_total, unknown_removals, self.public_pool_counts)

    # 从真实局面 state 生成“观察者视角”的信息态。
    def info_state(self, state: jq.GameState, observer: int) -> InformationState:
        return InformationState(
            board=state.board.copy(),
            side=state.side,
            pool_known=[self.pool_counts[observer][0].copy(), self.pool_counts[observer][1].copy()],
            pool_total_known=[self.pool_total[observer][0], self.pool_total[observer][1]],
            unknown_removals=[
                self.unknown_removals[observer][0],
                self.unknown_removals[observer][1],
            ],
            no_capture_plies=state.no_capture_plies,
            consecutive_checks=state.consecutive_checks.copy(),
            rules=state.rules,
            check_anchor=state.check_anchor.copy(),
            chase_streak=state.chase_streak.copy(),
            chase_target=state.chase_target.copy(),
            chase_target_code=state.chase_target_code.copy(),
            chase_anchor=state.chase_anchor.copy(),
            position_counts=state.position_counts.copy(),
            public_pool_known=[row.copy() for row in self.public_pool_counts],
        )

    def information_set_key(self, state: jq.GameState, observer: int,
                            public_path, private_path) -> tuple:
        """Freeze the actor's key without allocating a full observation snapshot."""
        return _information_set_key(
            state, self.pool_counts[observer],
            (self.pool_total[observer][0], self.pool_total[observer][1]),
            (self.unknown_removals[observer][0], self.unknown_removals[observer][1]),
            public_path, private_path,
        )

    # 观察者知道明确类型的情况下减少暗盒计数。
    # allow_underflow=True 用于确定化模拟阶段的信息矛盾容错。
    def _dec_known(self, observer: int, side: int, p_type: int, allow_underflow: bool) -> None:
        if self.pool_counts[observer][side][p_type] <= 0:
            if not allow_underflow:
                raise ValueError("belief pool underflow")
            if self.pool_total[observer][side] <= 0:
                return
            self.pool_total[observer][side] -= 1
            self.unknown_removals[observer][side] += 1
            return
        self.pool_counts[observer][side][p_type] -= 1
        self.pool_total[observer][side] -= 1

    # 观察者只知道“少了一个暗子”，但不知道类型。
    # allow_underflow=True 时允许跳过异常以保证模拟继续。
    def _dec_total_only(self, observer: int, side: int, allow_underflow: bool) -> None:
        if self.pool_total[observer][side] <= 0:
            if not allow_underflow:
                raise ValueError("belief pool total underflow")
            return
        self.pool_total[observer][side] -= 1
        self.unknown_removals[observer][side] += 1

    # 用一次走子（undo 里含揭示信息）更新双方的观测 belief。
    # - 暗子走动揭示：双方都知道类型
    # - 暗子被吃揭示：吃子者知道类型，被吃者只知道总数少 1
    # strict=False 用于确定化模拟阶段，允许信息矛盾时柔性修正。
    def observe_move(self, undo: jq.Undo, strict: bool = True) -> None:
        mover = undo.prev_side
        allow_underflow = not strict
        if undo.moved_reveal:
            if self.public_pool_counts[mover][undo.moved_reveal] > 0:
                self.public_pool_counts[mover][undo.moved_reveal] -= 1
            elif strict:
                raise ValueError("public belief pool underflow")
            for obs in (jq.BLACK, jq.RED):
                self._dec_known(obs, mover, undo.moved_reveal, allow_underflow)
        if undo.captured_before and jq.is_dark(undo.captured_before):
            captured_side = jq.piece_color(undo.captured_before)
            if undo.captured_reveal <= 0:
                raise ValueError("missing capture reveal")
            self._dec_known(mover, captured_side, undo.captured_reveal, allow_underflow)
            other = 1 - mover
            self._dec_total_only(other, captured_side, allow_underflow)


def record_observation(move, undo, public_path: list, private_paths: list[list]) -> None:
    """记录公开走法/移动揭示；捕获类型只进入吃方的私有观测链。"""
    public_path.append((move, undo.moved_reveal))
    if undo.captured_before and jq.is_dark(undo.captured_before):
        private_paths[undo.prev_side].append((len(public_path) - 1, undo.captured_reveal))


def information_set_key(info: InformationState, public_path, private_path) -> tuple:
    """当前行动者的决策键，不含世界真池或另一方私有捕获结果。

    保留从搜索根开始的完整观测序列，不合并不同公开历史；根前的私有
    信息沿用现有池组成抽象。本键仅在一次 search 的根先验下共享。
    """
    return _information_set_key(info, info.pool_known, info.pool_total_known,
                                info.unknown_removals, public_path, private_path)


def _information_set_key(state, pool_known, pool_total_known, unknown_removals,
                         public_path, private_path) -> tuple:
    return (tuple(public_path), tuple(private_path), state.side, bytes(state.board),
            tuple(pool_known[jq.BLACK]), tuple(pool_known[jq.RED]),
            tuple(pool_total_known), tuple(unknown_removals),
            state.no_capture_plies, tuple(state.consecutive_checks), tuple(state.check_anchor),
            tuple(state.chase_streak), tuple(state.chase_target), tuple(state.chase_target_code),
            tuple(state.chase_anchor))


class Node:
    __slots__ = ("parent", "move", "player_to_move", "children", "public_outcomes", "decision_nodes", "untried_moves", "visits", "value")

    def __init__(self, parent, move, player_to_move) -> None:
        # 共享决策图的 parent 仅记录首次入边；回传必须沿实际模拟 path。
        self.parent = parent
        self.move = move
        self.player_to_move = player_to_move
        self.children = []
        self.public_outcomes: dict[int, Node] = {}
        self.decision_nodes: dict[tuple, Node] = {}
        self.untried_moves = None
        self.visits = 0
        self.value = 0.0

    def public_outcome(self, reveal_moved: int) -> "Node":
        """公开翻子结果分别保存决策子树，不读取捕获暗子的私有类型。"""
        outcome = self.public_outcomes.get(reveal_moved)
        if outcome is None:
            outcome = Node(self, None, self.player_to_move)
            self.public_outcomes[reveal_moved] = outcome
        return outcome

    # 标准 UCT 选择：Q/N + c * sqrt(log N / n)
    def uct_select_child(self, exploration: float) -> "Node":
        log_visits = math.log(self.visits)

        def score(child: Node) -> float:
            if child.visits == 0:
                return float("inf")
            # child.value is stored from the child's player perspective.
            # Convert to parent perspective for selection.
            exploit = -(child.value / child.visits)
            explore = exploration * math.sqrt(log_visits / child.visits)
            return exploit + explore

        return max(self.children, key=score)

    # 扩展一个未尝试的动作。
    def expand(self, state: jq.GameState, rng: random.Random) -> tuple["Node", tuple[int, int]]:
        if self.untried_moves is None:
            self.untried_moves = state.legal_moves()
        move = rng.choice(self.untried_moves)
        self.untried_moves.remove(move)
        child = Node(self, move, 1 - state.side)
        self.children.append(child)
        return child, move

    # 返回访问次数最多的子节点作为最终落子。
    def best_child(self) -> "Node":
        return max(self.children, key=lambda c: c.visits)


def outcome_value_for_player(outcome: int | None, player: int) -> float:
    if outcome is None or outcome == jq.RESULT_DRAW:
        return 0.0
    if outcome == jq.RESULT_RED_WIN:
        return 1.0 if player == jq.RED else -1.0
    if outcome == jq.RESULT_BLACK_WIN:
        return 1.0 if player == jq.BLACK else -1.0
    raise ValueError("invalid outcome")


def outcome_from_moves(state: jq.GameState, moves: list[tuple[int, int]]):
    """由已知合法着法列表判定终局；无合法着法时优先将死/困毙（与 game_outcome 一致）。"""
    if not moves:
        winner = 1 - state.side
        result = jq.RESULT_RED_WIN if winner == jq.RED else jq.RESULT_BLACK_WIN
        reason = jq.REASON_CHECKMATE if state.is_in_check(state.side) else jq.REASON_STALEMATE
        return result, reason
    outcome, reason = state.fast_outcome()
    if outcome is not None:
        return outcome, reason
    return None, None


PIECE_VALUES = {
    jq.TYPE_PAWN: 1,
    jq.TYPE_ADVISOR: 2,
    jq.TYPE_ELEPHANT: 2,
    jq.TYPE_KNIGHT: 3,
    jq.TYPE_CANNON: 4,
    jq.TYPE_ROOK: 5,
    jq.TYPE_KING: 100,
}

# rollout 截断深度（ply，0 = 跑到终局）与子力评估的 tanh 刻度。
# 纯 MCTS 原先只回传终局胜负：rollout 双方都优先吃子，白吃一个无威胁的
# 暗子/明子并不会改变终局分布，搜索于是"看不见"子力。截断后用子力评估
# 把该信号补进 Q（暗子按剩余暗池期望计）。
ROLLOUT_MAX_PLIES = 60
LEAF_EVAL_SCALE = 10.0


def pool_expected_value(state: jq.GameState, side: int) -> float:
    """该方全部未翻开暗子的期望子力总值（按当前暗池计数）。

    确定化世界里的暗池就是暗子真实类型的分布，因此单枚暗子的期望价值
    = ``pool_expected_value() / state.pool_total[side]``；池空时返回 0。
    """
    if state.pool_total[side] <= 0:
        return 0.0
    return float(
        sum(state.pool[side][p_type] * PIECE_VALUES[p_type] for p_type in jq.POOL_TYPES)
    )


def material_value(state: jq.GameState, side: int) -> float:
    """某方子力：明子按面值计，暗子按剩余暗池期望计（将帅不计）。"""
    value = pool_expected_value(state, side)
    for piece in state.board:
        if not piece or jq.piece_color(piece) != side:
            continue
        p_type = jq.piece_type(piece)
        if p_type == jq.TYPE_KING or jq.is_dark(piece):
            continue
        value += PIECE_VALUES[p_type]
    return value


def evaluate_for_player(state: jq.GameState, player: int, scale: float = LEAF_EVAL_SCALE) -> float:
    """叶子评估：子力差经 tanh 压缩到 (-1, 1)，与终局值同量纲。"""
    diff = material_value(state, player) - material_value(state, 1 - player)
    return math.tanh(diff / scale)


def select_rollout_move(
    state: jq.GameState,
    moves: list[tuple[int, int]],
    rng: random.Random,
    policy: str,
) -> tuple[int, int]:
    """rollout 走子选择：默认优先吃子，否则随机。

    吃暗子时真实类型未知，按其所属方剩余暗池的单子期望计分（而非假类型
    面值）；暗池为空时回退假类型。
    """
    if not moves:
        raise ValueError("empty moves")
    if policy == "random":
        return rng.choice(moves)
    best_score = -1.0
    best_moves: list[tuple[int, int]] = []
    for move in moves:
        target = state.board[move[1]]
        if target and jq.piece_color(target) != state.side:
            target_side = jq.piece_color(target)
            if jq.is_dark(target) and state.pool_total[target_side] > 0:
                score = pool_expected_value(state, target_side) / state.pool_total[target_side]
            else:
                score = float(PIECE_VALUES.get(jq.piece_type(target), 1))
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
    if best_moves:
        return rng.choice(best_moves)
    return rng.choice(moves)


@dataclass(slots=True)
class ISMCTSConfig:
    # 每步搜索的模拟次数。
    iterations: int = 1000
    # UCT 探索系数。
    exploration: float = 1.4
    rollout_policy: str = "heuristic"
    # rollout 截断深度（ply）；0 = 不截断、跑到终局（旧行为）。
    rollout_max_plies: int = ROLLOUT_MAX_PLIES
    # 子力评估的 tanh 刻度：值 = tanh(子力差 / eval_scale)。
    eval_scale: float = LEAF_EVAL_SCALE


class ISMCTSAgent:
    __slots__ = ("config", "rng")

    def __init__(self, config: ISMCTSConfig | None = None, rng: random.Random | None = None) -> None:
        self.config = config or ISMCTSConfig()
        self.rng = rng or random.Random()

    # 用根视角单次确定化的 ISMCTS（原 RIS-ISMCTS）选择当前玩家的一步棋。
    def select_move(self, state: jq.GameState, belief: BeliefModel):
        return ris_ismcts(
            root_state=state,
            belief_model=belief,
            iterations=self.config.iterations,
            exploration=self.config.exploration,
            rollout_policy=self.config.rollout_policy,
            rollout_max_plies=self.config.rollout_max_plies,
            eval_scale=self.config.eval_scale,
            rng=self.rng,
        )


def init_belief_for_game(state: jq.GameState) -> BeliefModel:
    """
    实战用的 belief 初始化：
    - 起手局面：双方对“对方暗盒”完全了解；对“自己暗盒”也完全了解；
    - 后续用 observe_move 更新即可。
    """
    return BeliefModel.from_game_state(state)


def apply_move_and_update_belief(
    state: jq.GameState,
    belief: BeliefModel,
    move: tuple[int, int],
    rng: random.Random | None = None,
    reveal_choice: int | None = None,
    capture_reveal_choice: int | None = None,
) -> jq.Undo:
    """
    实战回合调用：
    1) 在真实 state 上执行走子；
    2) 使用 undo 更新 belief（包含公开信息与私有信息）。
    """
    undo = state.apply_move(
        move,
        rng=rng,
        reveal_choice=reveal_choice,
        capture_reveal_choice=capture_reveal_choice,
    )
    belief.observe_move(undo)
    return undo


def info_state_for_player(state: jq.GameState, belief: BeliefModel, observer: int) -> InformationState:
    """便捷函数：返回指定玩家视角的信息态。"""
    return belief.info_state(state, observer)


def auto_play_turn(
    state: jq.GameState,
    belief: BeliefModel,
    agent: ISMCTSAgent,
    rng: random.Random | None = None,
):
    """
    实战回合自动走子：
    - 调用根视角单次确定化的 ISMCTS（原 RIS-ISMCTS）选招；
    - 执行走子并同步更新 belief。
    """
    move, root = agent.select_move(state, belief)
    if move is None:
        return None, root
    apply_move_and_update_belief(state, belief, move, rng=rng or agent.rng)
    return move, root


class AutoMatch:
    """
    简易对局控制器：
    - 自动维护 belief；
    - 每步用根视角单次确定化的 ISMCTS（原 RIS-ISMCTS）选招。
    """

    __slots__ = ("state", "belief", "agents", "rng")

    def __init__(
        self,
        state: jq.GameState | None = None,
        rng: random.Random | None = None,
        config: ISMCTSConfig | None = None,
        agents: dict[int, ISMCTSAgent] | None = None,
    ) -> None:
        self.rng = rng or random.Random()
        self.state = state or jq.GameState.initial(rng=self.rng)
        self.belief = init_belief_for_game(self.state)
        if agents is None:
            cfg = config or ISMCTSConfig()
            self.agents = {jq.RED: ISMCTSAgent(cfg, self.rng), jq.BLACK: ISMCTSAgent(cfg, self.rng)}
        else:
            self.agents = agents

    # 执行一步自动走子并更新 belief。
    def step(self):
        agent = self.agents.get(self.state.side)
        if agent is None:
            raise ValueError("no agent for side to move")
        return auto_play_turn(self.state, self.belief, agent, rng=self.rng)


def ris_ismcts(
    root_state: jq.GameState,
    belief_model: BeliefModel,
    iterations: int = 1000,
    exploration: float = 1.4,
    rollout_policy: str = "heuristic",
    rng: random.Random | None = None,
    rollout_max_plies: int = ROLLOUT_MAX_PLIES,
    eval_scale: float = LEAF_EVAL_SCALE,
):
    """根一致采样、按行动者观测共享统计的 ISMCTS（保留旧 RIS 函数名）。

    每条迭代只在根按**根玩家**的信息确定化一次，整条模拟复用同一个世界；
    不读取真实对手的私有观测；合成对手知道本粒子中它吃过什么，但不知道
    自己被吃类型。不同环境边可以连接同一信息集决策节点，所以回传用路径，
    不能沿节点唯一 parent 回传。仍是根先验下的近似搜索，无均衡保证。

    rollout 超过 ``rollout_max_plies`` 步未终局时，回传子力评估
    （``evaluate_for_player``，暗子按剩余暗池期望计）；``0`` 表示不截断。
    和棋同理改用子力评估：随机 rollout 走到的无吃子/重复和棋不代表真和棋，
    否则子力优势会被"和棋=0"稀释。
    """
    rng = rng or random.Random()
    root_player = root_state.side
    root_node = Node(parent=None, move=None, player_to_move=root_state.side)
    root_node.untried_moves = root_state.legal_moves()
    root_info = belief_model.info_state(root_state, root_player)
    decision_table = {information_set_key(root_info, (), ()): root_node}

    # 主循环：选择 -> 扩展 -> 随机模拟 -> 回传（单次确定化）
    for _ in range(iterations):
        node = root_node
        # 每条模拟只在根按根玩家视角确定化一次，整条模拟复用该世界。
        state = root_info.determinize(rng)
        sim_view = BeliefModel.for_simulation(root_info, root_player, state)
        path = [root_node]
        public_path = []
        private_paths = [[], []]

        def advance(action, move):
            path.append(action)
            undo = state.apply_move(move, rng=rng)
            edge = action
            if undo.moved_reveal:
                edge = action.public_outcome(undo.moved_reveal)
                path.append(edge)
            sim_view.observe_move(undo)
            record_observation(move, undo, public_path, private_paths)
            info = sim_view.info_state(state, state.side)
            key = information_set_key(info, public_path, private_paths[state.side])
            decision = decision_table.get(key)
            if decision is None:
                decision = Node(edge, None, state.side)
                decision_table[key] = decision
            edge.decision_nodes[key] = decision
            path.append(decision)
            return decision

        outcome, _ = state.outcome_checked()
        # 选择阶段：沿着已扩展的树走到未完全展开的位置。
        while outcome is None and node.untried_moves == [] and node.children:
            action = node.uct_select_child(exploration)
            node = advance(action, action.move)
            outcome, _ = state.outcome_checked()

        # 扩展阶段：从未尝试动作中扩展一个。
        if outcome is None:
            if node.untried_moves is None:
                node.untried_moves = state.legal_moves()
                outcome, _ = outcome_from_moves(state, node.untried_moves)
            if node.untried_moves:
                action, move = node.expand(state, rng)
                node = advance(action, move)
                outcome, _ = state.outcome_checked()

        # 随机模拟：优先吃子，否则随机，直到终局或到达截断深度。
        plies = 0
        while outcome is None:
            moves = state.legal_moves()
            outcome, _ = outcome_from_moves(state, moves)
            if outcome is not None:
                break
            if rollout_max_plies and plies >= rollout_max_plies:
                break
            move = select_rollout_move(state, moves, rng, rollout_policy)
            state.apply_move(move, rng=rng)
            plies += 1
            outcome, _ = state.outcome_checked()

        # 回传阶段：从叶子节点执棋方视角开始，逐层翻转。
        # 注意必须用 node.player_to_move 而不是 state.side：rollout 会随机走子并翻转
        # state.side，而 node 仍是扩展节点，其值必须按该节点轮次的视角记账。
        # 未分胜负时（截断或和棋）用子力评估补信号，否则子力在 Q 里不可见。
        if outcome is None or outcome == jq.RESULT_DRAW:
            value = evaluate_for_player(state, node.player_to_move, eval_scale)
        else:
            value = outcome_value_for_player(outcome, node.player_to_move)
        for i in range(len(path) - 1, -1, -1):
            visited = path[i]
            visited.visits += 1
            visited.value += value
            # 动作与公开结果之间没有换手，只有行动方改变时翻转价值。
            if i > 0 and path[i - 1].player_to_move != visited.player_to_move:
                value = -value

    if not root_node.children:
        return None, root_node
    return root_node.best_child().move, root_node


# 用法示例（伪代码）：
# state = jq.GameState.initial()
# belief = init_belief_for_game(state)
# agent = ISMCTSAgent(ISMCTSConfig(iterations=1000))
# while state.game_outcome()[0] is None:
#     move, _ = auto_play_turn(state, belief, agent)
#     if move is None:
#         break
