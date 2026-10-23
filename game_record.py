"""GUI 棋谱树：保留每条变例的揭示、规则历史与双方私有信念。

v2 保存完整起始局面及按父节点排列的节点表；v1 线性棋谱仍可读取。
暗子结果写在边上，重放不重新随机抽子；添加变例不会改写已有节点。
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field

import ismcts as im
import jieqi as jq

FORMAT = "jieqi-replay"
VERSION = 2
RULE_FIELDS = ("consecutive_checks", "check_anchor", "chase_streak", "chase_target",
               "chase_target_code", "chase_anchor")
MIGRATABLE_RULESETS = (3, 4)


def _prepare_rules_migration(data):
    """准备内存迁移；调用方仍须逐步按当前规则重演，成功后才能使用/保存。"""
    source = data.get("ruleset_version")
    if type(source) is not int or source not in MIGRATABLE_RULESETS or source >= jq.RULESET_VERSION:
        raise ValueError(f"不支持从规则 v{source} 迁移到 v{jq.RULESET_VERSION}")
    migrated = copy.deepcopy(data)
    migration = {"source_ruleset_version": source, "target_ruleset_version": jq.RULESET_VERSION,
                 "original_end": copy.deepcopy(data.get("end")), "initial_rule_history_reset": False}
    if migrated.get("version", 1) == VERSION:
        initial = migrated["initial"]["state"]
        source_rules = jq.RulesConfig(**initial.get("rules", {"ruleset_version": source}))
        expected = asdict(jq.RULESET_TIANTIAN)
        expected["ruleset_version"] = source
        if asdict(source_rules) != expected:
            raise ValueError("初始规则配置与棋谱来源版本不符，不能自动迁移")
        initial["rules"] = asdict(jq.RULESET_TIANTIAN)
        unpack_state(initial)  # 重置旧跟踪字段前也必须验证起点，不能掩盖损坏的数据。
        # 起点之前的旧将/捉身份不能用新算法凭空补算。保留原字段供追溯，
        # 从起点重新建立这些计数；公开重复历史和无吃子计数仍可沿用。
        defaults = jq.GameState.empty(with_pools=False)
        original = {name: copy.deepcopy(initial.get(name, getattr(defaults, name))) for name in RULE_FIELDS}
        if any(original[name] != getattr(defaults, name) for name in RULE_FIELDS):
            migration["initial_rule_history_reset"] = True
            migration["original_rule_history"] = original
            for name in RULE_FIELDS:
                initial[name] = list(getattr(defaults, name))
    migrated["ruleset_version"] = jq.RULESET_VERSION
    history = migrated.setdefault("rules_migrations", [])
    if not isinstance(history, list):
        raise ValueError("规则迁移记录无效")
    history.append(migration)
    return migrated


def _validate_rules_migrations(history):
    if not isinstance(history, list):
        raise ValueError("规则迁移记录无效")
    for item in history:
        if not isinstance(item, dict):
            raise ValueError("规则迁移记录无效")
        source, target = item.get("source_ruleset_version"), item.get("target_ruleset_version")
        if (type(source) is not int or type(target) is not int or
                not 1 <= source < target <= jq.RULESET_VERSION or
                type(item.get("initial_rule_history_reset", False)) is not bool):
            raise ValueError("规则迁移记录无效")


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label}必须为 {low}～{high} 的整数")
    return value


def validate_position(state: jq.GameState, *, require_kings=True) -> None:
    """校验可用于对弈/分析的摆棋局面；允许终局少一个将帅。"""
    if len(state.board) != jq.BOARD_SIZE or state.side not in (jq.RED, jq.BLACK):
        raise ValueError("棋盘或行动方无效")
    for piece in state.board:
        if type(piece) is not int or piece < 0 or piece > 31:
            raise ValueError("棋子编码无效")
        if piece and (jq.piece_type(piece) not in range(1, 8) or
                      (jq.is_dark(piece) and jq.piece_type(piece) == jq.TYPE_KING)):
            raise ValueError("棋子类型无效，将帅不能是暗子")
    for side in (jq.BLACK, jq.RED):
        name = "红方" if side == jq.RED else "黑方"
        kings = [i for i, p in enumerate(state.board) if p and
                 jq.piece_color(p) == side and jq.piece_type(p) == jq.TYPE_KING]
        if len(kings) > 1 or (require_kings and len(kings) != 1):
            raise ValueError(f"{name}必须恰有一个将帅")
        if kings:
            r, c = divmod(kings[0], jq.BOARD_W)
            if c not in (3, 4, 5) or r not in ((7, 8, 9) if side == jq.RED else (0, 1, 2)):
                raise ValueError(f"{name}将帅必须在九宫内")
        dark_count = sum(bool(p and jq.piece_color(p) == side and jq.is_dark(p)) for p in state.board)
        pool = state.pool[side]
        if len(pool) != 8 or pool[0] != 0 or pool[jq.TYPE_KING] != 0:
            raise ValueError(f"{name}暗池无效")
        for t in jq.POOL_TYPES:
            _integer(pool[t], 0, jq.POOL_INIT[t], f"{name}暗池")
            bright = sum(p == jq.make_piece(t, side, 0) for p in state.board)
            if bright + pool[t] > jq.POOL_INIT[t]:
                raise ValueError(f"{name}{jq.PIECE_NAMES[side][t]}的明子和暗池总数超出标准数量")
        if sum(pool) != dark_count or state.pool_total[side] != dark_count:
            raise ValueError(f"{name}暗池总数必须等于棋盘暗子数（{dark_count}）")


def pack_state(state: jq.GameState) -> dict:
    data = {"board": state.board.copy(), "side": state.side,
            "pool": copy.deepcopy(state.pool), "no_capture_plies": state.no_capture_plies,
            "rules": asdict(state.rules),
            "positions": [[side, list(board), count] for (side, board), count in state.position_counts.items()],
            "dark_types": None if state.dark_types is None else {str(i): t for i, t in state.dark_types.items()}}
    data.update({name: list(getattr(state, name)) for name in RULE_FIELDS})
    return data


def unpack_state(data: dict) -> jq.GameState:
    board, pool = list(data["board"]), copy.deepcopy(data["pool"])
    side = _integer(data["side"], 0, 1, "行动方")
    if len(pool) != 2:
        raise ValueError("暗池必须有红黑两方")
    rules = jq.RulesConfig(**data.get("rules", {}))
    if rules != jq.RULESET_TIANTIAN:
        raise ValueError("棋谱的规则配置与当前 GUI 不兼容")
    dark_types = data.get("dark_types")
    if dark_types is not None:
        dark_types = {int(i): _integer(t, 2, 7, "预设暗子类型") for i, t in dark_types.items()}
        if any(i < 0 or i >= jq.BOARD_SIZE for i in dark_types):
            raise ValueError("预设暗子坐标越界")
    state = jq.GameState(board, side, pool, [sum(row) for row in pool],
                         im.infer_king_pos(board), dark_types=dark_types)
    validate_position(state, require_kings=False)
    state.no_capture_plies = _integer(data.get("no_capture_plies", 0), 0, 100000, "无吃子步数")
    for name in RULE_FIELDS:
        values = data.get(name, getattr(state, name))
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"规则字段 {name} 无效")
        lower, upper = (-1, 89) if name.endswith("anchor") or name == "chase_target" else (0, 100000)
        setattr(state, name, [_integer(v, lower, upper, name) for v in values])
    if "positions" in data:
        counts = {}
        for p_side, p_board, count in data["positions"]:
            if len(p_board) != jq.BOARD_SIZE:
                raise ValueError("重复历史棋盘无效")
            counts[(_integer(p_side, 0, 1, "历史行动方"), tuple(p_board))] = _integer(count, 1, 100000, "重复次数")
        if state.position_key not in counts:
            raise ValueError("重复历史缺少当前局面")
        state.position_counts = counts
    return state


def pack_belief(belief: im.BeliefModel) -> dict:
    return {name: copy.deepcopy(getattr(belief, name)) for name in im.BeliefModel.__slots__}


def unpack_belief(data: dict, state: jq.GameState) -> im.BeliefModel:
    belief = im.BeliefModel(**copy.deepcopy(data))
    for obs in (jq.BLACK, jq.RED):
        for side in (jq.BLACK, jq.RED):
            counts = belief.pool_counts[obs][side]
            unknown = belief.unknown_removals[obs][side]
            total = belief.pool_total[obs][side]
            if len(counts) != 8 or any(type(v) is not int or v < 0 for v in counts):
                raise ValueError("初始信念计数无效")
            if total != state.pool_total[side] or type(unknown) is not int or unknown < 0 or sum(counts) != total + unknown:
                raise ValueError("初始信念与暗池数量不符")
            if any(counts[t] < state.pool[side][t] for t in range(8)):
                raise ValueError("初始信念排除了真实暗池")
    if len(belief.public_pool_counts) != 2:
        raise ValueError("公共暗池账本无效")
    for side in (jq.BLACK, jq.RED):
        public = belief.public_pool_counts[side]
        if len(public) != 8 or any(type(v) is not int or v < state.pool[side][t] for t, v in enumerate(public)):
            raise ValueError("公共暗池账本与真实暗池不符")
    return belief


def unpack_captured(data: dict) -> dict:
    """在棋谱替换当前对局前验证吃子条目，供 GUI 和 Web 共用。"""
    if not isinstance(data, dict):
        raise ValueError("初始吃子记录必须为对象")
    captured = {jq.BLACK: [], jq.RED: []}
    for side in (jq.BLACK, jq.RED):
        entries = data.get(str(side), [])
        if not isinstance(entries, list):
            raise ValueError("初始吃子记录必须为列表")
        for item in entries:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("吃子条目必须包含颜色、暗子标记和类型")
            color, dark, kind = item
            _integer(color, 0, 1, "吃子颜色")
            _integer(kind, 1, 7, "吃子类型")
            if type(dark) is not bool or (dark and kind == jq.TYPE_KING):
                raise ValueError("吃子暗子标记无效，将帅不能是暗子")
            captured[side].append((color, dark, kind))
    return captured


@dataclass
class RecordNode:
    id: int
    parent: int | None
    record: dict | None
    state: jq.GameState
    belief: im.BeliefModel
    captured: dict
    children: list[int] = field(default_factory=list)
    depth: int = 0


class GameRecord:
    def __init__(self, state=None, belief=None, captured=None, metadata=None):
        state = state if state is not None else jq.GameState.initial()
        belief = belief if belief is not None else im.init_belief_for_game(state)
        captured = captured if captured is not None else {jq.RED: [], jq.BLACK: []}
        self.nodes = {0: RecordNode(0, None, None, state.clone(), belief.clone(), copy.deepcopy(captured))}
        self.cursor = 0
        self.metadata = copy.deepcopy(metadata or {})

    @property
    def current(self):
        return self.nodes[self.cursor]

    def path(self, node_id=None):
        path = []
        node = self.nodes[self.cursor if node_id is None else node_id]
        while node.parent is not None:
            path.append(node.id)
            node = self.nodes[node.parent]
        return path[::-1]

    def add_move(self, move, *, record=None, rng=None, reuse=True):
        parent = self.current
        move = tuple(move)
        if len(move) != 2:
            raise ValueError("着法必须包含起点和落点")
        move = tuple(_integer(i, 0, jq.BOARD_SIZE - 1, "走子坐标") for i in move)
        # 同一走法优先进入原有分支，保留它原来记录的随机揭示。
        for child_id in parent.children:
            child = self.nodes[child_id]
            if reuse and (child.record["from"], child.record["to"]) == move and (record is None or
                    all(child.record[k] == record.get(k, 0) for k in ("reveal", "capture"))):
                self.cursor = child_id
                return child
        state, belief = parent.state.clone(), parent.belief.clone()
        if state.game_outcome()[0] is not None or move not in state.legal_moves():
            raise ValueError("该着法不合法，或局面已终局")
        src, dst = move
        captured_before = state.board[dst]
        if record is not None:
            for key, piece in (("reveal", state.board[src]), ("capture", captured_before)):
                value = record.get(key, 0)
                _integer(value, 0, 7, "揭示类型")
                if bool(piece and jq.is_dark(piece)) != bool(value) or value == jq.TYPE_KING:
                    raise ValueError(f"着法 {key} 与暗子状态不符，缺失揭示不能随机补造")
        undo = im.apply_move_and_update_belief(state, belief, move, rng=rng or random.Random(),
                      reveal_choice=record.get("reveal") or None if record else None,
                      capture_reveal_choice=record.get("capture") or None if record else None)
        captured = copy.deepcopy(parent.captured)
        if captured_before:
            dark = bool(jq.is_dark(captured_before))
            captured[undo.prev_side].append((jq.piece_color(captured_before), dark,
                          undo.captured_reveal if dark else jq.piece_type(captured_before)))
        result = {"from": src, "to": dst, "reveal": undo.moved_reveal, "capture": undo.captured_reveal,
                  "uci": jq.format_move_uci(move, undo.moved_reveal or jq.piece_type(state.board[dst]))}
        node_id = len(self.nodes)
        node = RecordNode(node_id, parent.id, result, state, belief, captured, depth=parent.depth + 1)
        self.nodes[node_id] = node
        parent.children.append(node_id)
        self.cursor = node_id
        return node

    def to_dict(self):
        root = self.nodes[0]
        data = copy.deepcopy(self.metadata)
        outcome, reason = self.current.state.game_outcome()
        data.update(format=FORMAT, version=VERSION, ruleset_version=jq.RULESET_VERSION,
                    initial={"state": pack_state(root.state), "belief": pack_belief(root.belief),
                             "captured": {str(s): copy.deepcopy(root.captured[s]) for s in (jq.BLACK, jq.RED)}},
                    nodes=[{"id": node.id, "parent": node.parent, "move": dict(node.record)}
                           for node in self.nodes.values() if node.parent is not None],
                    cursor=self.cursor, end={"outcome": outcome, "reason": reason})
        # 清除旧线性字段，v2 节点表是唯一权威走子记录。
        data.pop("moves", None)
        return data

    @classmethod
    def from_dict(cls, data, *, migrate_rules=False):
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError("不是有效的揭棋对局文件")
        if data.get("ruleset_version") != jq.RULESET_VERSION:
            if not migrate_rules:
                raise ValueError("棋谱规则版本与当前规则不符")
            data = _prepare_rules_migration(data)
        version = data.get("version", 1)
        if type(version) is not int:
            raise ValueError("棋谱版本必须是整数")
        _validate_rules_migrations(data.get("rules_migrations", []))
        metadata = {k: v for k, v in data.items() if k not in ("initial", "nodes", "moves", "cursor", "end")}
        human = metadata.get("human_side")
        if human is not None:
            _integer(human, 0, 1, "玩家颜色")
        if metadata.get("mode", "study") not in ("human", "aiai", "study"):
            raise ValueError("未知的对局模式")
        if "god_view" in metadata and type(metadata["god_view"]) is not bool:
            raise ValueError("观看视角必须是布尔值")
        if version == 1:
            if not isinstance(data.get("moves"), list):
                raise ValueError("旧棋谱缺少走子列表")
            game = cls(metadata=metadata)
            for i, record in enumerate(data["moves"], 1):
                try:
                    move = (_integer(record["from"], 0, 89, "起点"), _integer(record["to"], 0, 89, "落点"))
                    game.add_move(move, record=record, reuse=False)
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    raise ValueError(f"第 {i} 步记录错误：{exc}") from exc
            return game
        if version != VERSION:
            raise ValueError(f"不支持的棋谱版本：{version}")
        initial = data["initial"]
        state = unpack_state(initial["state"])
        belief = unpack_belief(initial["belief"], state)
        captured = unpack_captured(initial.get("captured", {}))
        game = cls(state, belief, captured, metadata)
        for item in data["nodes"]:
            node_id = _integer(item["id"], 1, 1000000, "节点编号")
            parent_id = _integer(item["parent"], 0, node_id - 1, "父节点编号")
            if node_id != len(game.nodes) or parent_id not in game.nodes:
                raise ValueError("棋谱节点重复、父节点缺失或顺序无效")
            game.cursor = parent_id
            try:
                record = item["move"]
                move = (_integer(record["from"], 0, 89, "起点"), _integer(record["to"], 0, 89, "落点"))
                game.add_move(move, record=record, reuse=False)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                raise ValueError(f"节点 {node_id}（第 {game.current.depth + 1} 手）记录错误：{exc}") from exc
        cursor = data.get("cursor", len(game.nodes) - 1)
        if type(cursor) is not int or cursor not in game.nodes:
            raise ValueError("当前节点无效")
        game.cursor = cursor
        return game
