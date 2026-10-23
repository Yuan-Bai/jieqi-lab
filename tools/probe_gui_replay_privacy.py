"""只读检查 GUI 分析是否依赖行动方看不到的暗子捕获类型。

固定公开走法/移动揭示，仅替换 h7h0 的私有捕获结果（车/兵）。
走真实 AzAgent.analyse + SearchBelief 接口，同 seed 核对根输入、全部
网络评估输入、policy、推荐着法及根价值。不修改棋谱、模型或训练数据。
b0a2 两种移动揭示是受控构造，不声称存在于输入棋谱。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import azero as az
import gui
import jieqi as jq
from game_record import GameRecord
from gui_tasks import SearchBelief, SearchRandom


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_case(root, destination, moved_reveal, captured_reveal):
    game = GameRecord(root.state, root.belief, root.captured)
    game.add_move((82, destination), record={'reveal': moved_reveal, 'capture': 0})
    game.add_move((25, 88), record={'reveal': jq.TYPE_ADVISOR, 'capture': captured_reveal})
    return game


class AnalysisTrace:
    """记录真实前向的完整输入摘要；不替换网络、策略或搜索规则。"""
    def __init__(self, agent):
        self.agent = agent
        self.evaluate = agent.mcts._evaluate
        self.evaluate_batch = agent.mcts._evaluate_batch
        self.search = agent.mcts.search
        agent.mcts._evaluate = self.single
        agent.mcts._evaluate_batch = self.batch
        agent.mcts.search = self.run_search

    def reset(self):
        self.digest = hashlib.sha256()
        self.inputs = 0
        self.root_network_value = None
        self.root = None
        self.policy = None

    def record(self, infos, histories):
        self.digest.update(len(infos).to_bytes(8, 'little'))
        for info, history in zip(infos, histories):
            self.digest.update(az.encode_state_numpy(info, history).tobytes())
            self.inputs += 1

    def single(self, info, history):
        self.record([info], [history])
        result = self.evaluate(info, history)
        self.root_network_value = result[1]
        return result

    def batch(self, infos, histories, **kwargs):
        self.record(infos, histories)
        return self.evaluate_batch(infos, histories, **kwargs)

    def run_search(self, *args, **kwargs):
        result = self.search(*args, **kwargs)
        self.policy, _, self.root = result
        return result

    def analyse(self, game, seed, simulations):
        self.reset()
        self.agent.mcts.rng.seed(seed)
        node = game.current
        history = [(game.nodes[i].record['from'], game.nodes[i].record['to']) for i in game.path()]
        state, belief = node.state.clone(), node.belief.clone()
        root_info = belief.info_state(state, state.side)
        start = time.perf_counter()
        move, value = self.agent.analyse(state, SearchBelief(belief, self.agent.mcts.rng), history, simulations)
        return {
            'seed': seed, 'root_value': value, 'red_score': (1 + value) / 2,
            'root_network_value': self.root_network_value,
            'root_visits': self.root.visits,
            'move': jq.format_move_uci(move) if move else None,
            'root_input_sha256': hashlib.sha256(az.encode_state_numpy(root_info, history).tobytes()).hexdigest(),
            'evaluation_inputs': self.inputs, 'evaluation_trace_sha256': self.digest.hexdigest(),
            'policy_sha256': hashlib.sha256(np.asarray(self.policy, dtype=np.float64).tobytes()).hexdigest(),
            'seconds': round(time.perf_counter() - start, 3),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--sims', type=int, default=800)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', type=Path, help='可选的新报告文件；拒绝覆盖已有文件')
    args = parser.parse_args()
    if args.sims <= 0:
        parser.error('--sims 必须为正整数')
    if args.out is not None and args.out.exists():
        parser.error('--out 已存在，请选择新文件名')
    replay_hash, model_hash = file_hash(args.replay), file_hash(args.model)
    game = GameRecord.from_dict(json.loads(args.replay.read_text(encoding='utf-8')), migrate_rules=True)
    root = game.nodes[0]
    rng = SearchRandom()
    agent = gui.AzAgent(str(args.model.resolve()), args.sims, args.device, rng)
    trace = AnalysisTrace(agent)
    results = []
    invariant_fields = ('root_value', 'root_visits', 'move', 'root_input_sha256',
                        'evaluation_inputs', 'evaluation_trace_sha256', 'policy_sha256')
    for name, destination, reveal in [('b0c2_pawn', 65, jq.TYPE_PAWN),
                                       ('b0a2_elephant', 63, jq.TYPE_ELEPHANT),
                                       ('b0a2_pawn', 63, jq.TYPE_PAWN)]:
        cases = [build_case(root, destination, reveal, captured)
                 for captured in (jq.TYPE_ROOK, jq.TYPE_PAWN)]
        a, b = [case.current.belief.info_state(case.current.state, jq.RED) for case in cases]
        if asdict(a) != asdict(b):
            raise AssertionError(f'{name}: 红方完整观测不一致，不能用于私有信息对照')
        for seed in args.seeds:
            pair = []
            for captured, case in zip(('rook', 'pawn'), cases):
                result = trace.analyse(case, seed, args.sims)
                result.update(case=name, captured=captured)
                results.append(result)
                pair.append(result)
                print(f'{name} capture={captured} seed={seed}: red={result["red_score"]:.6%} '
                      f'move={result["move"]} inputs={result["evaluation_inputs"]} '
                      f'seconds={result["seconds"]}', flush=True)
            mismatches = [field for field in invariant_fields if pair[0][field] != pair[1][field]]
            if mismatches:
                raise AssertionError(f'{name}, seed={seed}: 私有捕获结果改变搜索 {mismatches}')
    if file_hash(args.replay) != replay_hash or file_hash(args.model) != model_hash:
        raise RuntimeError('运行期间来源文件发生变化，本次报告不能视为固定来源验证')
    report = {'replay': str(args.replay.resolve()), 'replay_sha256': replay_hash,
              'model': str(args.model.resolve()), 'model_sha256': model_hash,
              'ruleset_version': jq.RULESET_VERSION, 'search_version': az.SEARCH_VERSION,
              'device': str(agent.device), 'simulations': args.sims, 'seeds': args.seeds,
              'private_capture_invariance_passed': True, 'source_files_unchanged': True,
              'results': results}
    if args.out is not None:
        with args.out.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print('PASS: 同公开信息、同 seed 下，车/兵私有捕获结果的全部输入和搜索结果完全一致。')


if __name__ == '__main__':
    main()
