"""引擎封装：复用项目既有 AlphaZero / RIS-ISMCTS 搜索给出推荐着法。

本模块**不修改任何算法**，只是把 ``azero.NeuralMCTS`` 与 ``ismcts.ISMCTSAgent``
按 GUI 里同样的方式调用，并把搜索结果整理成人类可读的建议：

* 推荐着法（访问次数最高的着法）；
* 前若干候选着法及各自访问次数占比；
* 根节点价值换算的胜率（AlphaZero 才有）。

用法：构造 :class:`AzEngine` 后调用
``analyse(state, belief, history)``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import jieqi as jq

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = HERE
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "data", "models", "latest.pt")

PIECE_NAMES = {
    jq.RED: {
        jq.TYPE_KING: "帅", jq.TYPE_ROOK: "车", jq.TYPE_KNIGHT: "马",
        jq.TYPE_ELEPHANT: "相", jq.TYPE_ADVISOR: "仕", jq.TYPE_CANNON: "炮",
        jq.TYPE_PAWN: "兵",
    },
    jq.BLACK: {
        jq.TYPE_KING: "将", jq.TYPE_ROOK: "车", jq.TYPE_KNIGHT: "马",
        jq.TYPE_ELEPHANT: "象", jq.TYPE_ADVISOR: "士", jq.TYPE_CANNON: "炮",
        jq.TYPE_PAWN: "卒",
    },
}

# 项目与 GUI 一致的暗池类型顺序（jieqi.POOL_TYPES 的索引即兵种编号）
AZ_CHANCE_NODES = True
AZ_CHANCE_K = 0


@dataclass
class MoveAdvice:
    """单个候选着法。"""

    move: tuple[int, int]
    visits: int
    share: float
    text: str

    def brief(self, rank: int) -> str:
        return f"{rank}. {self.text}  访问={self.visits} ({self.share:.0%})"


@dataclass
class Advice:
    """一次分析的完整结果。"""

    engine: str
    simulations: int
    mover: int
    moves: list[MoveAdvice] = field(default_factory=list)
    win_rate: float | None = None
    note: str = ""

    @property
    def best(self) -> MoveAdvice | None:
        return self.moves[0] if self.moves else None

    def report(self, topk: int = 3) -> str:
        lines = [f"引擎={self.engine} 模拟={self.simulations} "
                 f"行动方={'红' if self.mover == jq.RED else '黑'}"]
        if self.win_rate is not None:
            lines.append(f"胜率估计（模型）={self.win_rate:.1%}")
        if not self.moves:
            lines.append("没有可用着法")
        for i, mv in enumerate(self.moves[:topk], start=1):
            lines.append("  " + mv.brief(i))
        if self.note:
            lines.append(f"  注：{self.note}")
        return "\n".join(lines)


def describe_move(state: jq.GameState, move: tuple[int, int]) -> str:
    """把着法写成中文形式：``红方 暗车 (0,0)→(1,0)``。

    使用与 GUI 相同的记谱方式（``jq.square_to_name``），便于与界面上的
    "上一步/下一步"提示对照。
    """
    piece = state.board[move[0]]
    if not piece:
        return f"(空) {jq.square_to_name(move[0])}→{jq.square_to_name(move[1])}"
    color = jq.piece_color(piece)
    name = PIECE_NAMES[color][jq.piece_type(piece)]
    prefix = "暗" if jq.is_dark(piece) else ""
    side = "红" if color == jq.RED else "黑"
    return (f"{side}方 {prefix}{name} {jq.square_to_name(move[0])}"
            f"→{jq.square_to_name(move[1])}")


def _subtree_visits(node) -> int:
    """节点访问数；机会节点的访问数可由其子树求和得到。"""
    if getattr(node, "is_chance", False) and node.children:
        return int(sum(_subtree_visits(child) for child in node.children.values()))
    return int(node.visits)


class AzEngine:
    """AlphaZero（PUCT + 根信息采样）引擎。"""

    def __init__(self, model_path: str | None = None, simulations: int = 200,
                 device: str = "auto", rng=None) -> None:
        import torch

        import azero

        self._torch = torch
        self.model_path = model_path or DEFAULT_MODEL
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.simulations = simulations
        self.net = azero.PolicyValueNet()
        if not azero.load_model_if_exists(self.net, self.model_path, self.device):
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}")
        self.net.to(self.device)
        self.net.eval()
        self.config = azero.AZConfig(mcts_simulations=simulations,
                                     chance_nodes=AZ_CHANCE_NODES, chance_k=AZ_CHANCE_K)
        import random

        self.rng = rng or random.Random()
        self.mcts = azero.NeuralMCTS(self.net, self.config, self.device, self.rng)

    def analyse(self, state: jq.GameState, belief, history=(),
                topk: int = 3, simulations: int | None = None) -> Advice:
        """搜索并返回建议。"""
        sims = simulations or self.simulations
        _policy, _move, root = self.mcts.search(
            state, belief, list(history), temperature=1e-6,
            simulations=sims, add_dirichlet=False, return_root=True,
        )
        stats: list[tuple[tuple[int, int], int]] = []
        for move, child in root.children.items():
            stats.append((move, _subtree_visits(child)))
        stats.sort(key=lambda item: -item[1])
        total = sum(v for _m, v in stats) or 1
        moves = [
            MoveAdvice(move, visits, visits / total, describe_move(state, move))
            for move, visits in stats[:topk]
        ]
        visits = root.visits
        value = (root.value_sum / visits) if visits else 0.0
        return Advice("AlphaZero", sims, state.side, moves,
                      win_rate=(value + 1) / 2)


class RisEngine:
    """RIS-ISMCTS 引擎（无需神经网络，作为 AlphaZero 的对照/兜底）。"""

    def __init__(self, simulations: int = 800, rng=None) -> None:
        import random

        import ismcts as im

        self.simulations = simulations
        self.rng = rng or random.Random()
        self.agent = im.ISMCTSAgent(im.ISMCTSConfig(iterations=simulations), self.rng)

    def analyse(self, state: jq.GameState, belief, history=(),
                topk: int = 3, simulations: int | None = None) -> Advice:
        """搜索并返回建议（RIS 只给出最佳着法，访问占比不可得）。"""
        sims = simulations or self.simulations
        agent = self.agent
        if sims != self.simulations:
            import ismcts as im

            agent = im.ISMCTSAgent(im.ISMCTSConfig(iterations=sims), self.rng)
        move, _root = agent.select_move(state, belief)
        moves: list[MoveAdvice] = []
        if move is not None:
            moves.append(MoveAdvice(move, sims, 1.0, describe_move(state, move)))
        return Advice("RIS-ISMCTS", sims, state.side, moves,
                      note="RIS 不提供胜率与访问分布")


def build_engine(kind: str = "az", **kwargs):
    """按名称构造引擎：``az`` 或 ``ris``。

    两种引擎的参数不同（RIS 不需要模型与设备），这里按需过滤，
    便于上层用同一套关键字调用。
    """
    key = kind.lower()
    if key in ("az", "azero", "alphazero"):
        return AzEngine(**kwargs)
    if key in ("ris", "ismcts"):
        allowed = {"simulations", "rng"}
        return RisEngine(**{k: val for k, val in kwargs.items() if k in allowed})
    raise ValueError(f"未知引擎: {kind}")
