import json
import concurrent.futures
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jieqi as jq
from tools import ab_runner, eval as eval_mod, paired_stats
import azero as az
import torch


class TestEvalProtocol(unittest.TestCase):
    def test_agent_and_environment_rngs_are_isolated(self):
        streams = []

        class StopAgent:
            def __init__(self, rng):
                streams.append(rng)
                for _ in range(100):
                    rng.random()

            def select_move(self, state, belief, history):
                streams.append(state.rng)
                return None

        record = eval_mod.play_one_game(StopAgent, StopAgent, jq.RED, seed=11, max_plies=10)
        self.assertEqual(len({id(rng) for rng in streams}), 3)
        env, _, _ = az.evaluation_rngs(11)
        self.assertEqual(streams[-1].getstate(), env.getstate())
        self.assertIsNone(record["score"])
        self.assertEqual(record["reason"], "no_move")

    def test_train_eval_materializes_same_shuffle_for_both_colors(self):
        snapshots = []
        initial = jq.GameState.initial

        def spy_initial(*args, **kwargs):
            snapshots.append(kwargs["dark_types"])
            return initial(*args, **kwargs)

        with mock.patch.object(az, "NeuralMCTS") as neural, \
                mock.patch.object(az.im, "ISMCTSAgent") as opponent, \
                mock.patch.object(jq.GameState, "initial", side_effect=spy_initial):
            neural.return_value.search.return_value = ([], None)
            opponent.return_value.select_move.return_value = (None, {})
            for side in (jq.RED, jq.BLACK):
                self.assertIsNone(az._play_eval_game(None, az.AZConfig(), torch.device("cpu"), side, 13))
            self.assertEqual(snapshots[0], snapshots[1])
            self.assertEqual(snapshots[0], jq.materialize_dark_types(random.Random("jieqi-pair-shuffle:13")))
            self.assertIsNot(neural.call_args.args[3], opponent.call_args.args[1])

    def test_train_eval_excludes_entire_incomplete_pair_and_rounds_odd_games_up(self):
        with mock.patch.object(az, "_play_eval_game", side_effect=[1.0, None, -1.0, 0.0]):
            result = az.evaluate_vs_ris(None, az.AZConfig(eval_workers=1), torch.device("cpu"),
                                        3, random.Random(9))
        self.assertEqual((result["wins"], result["draws"], result["losses"]), (0, 1, 1))
        self.assertEqual((result["pairs"], result["pairs_used"], result["unfinished"]), (2, 1, 1))
        self.assertEqual(result["pairs_excluded"], 1)
        self.assertEqual(result["score"], 0.25)

    def test_external_eval_summary_uses_only_complete_pairs(self):
        games = [dict(outcome=jq.RESULT_RED_WIN, score=1, neural_side="red"),
                 dict(outcome=None, score=None, neural_side="black"),
                 dict(outcome=jq.RESULT_BLACK_WIN, score=-1, neural_side="red"),
                 dict(outcome=jq.RESULT_DRAW, score=0, neural_side="black")]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(eval_mod, "play_one_game", side_effect=games):
            result = eval_mod.evaluate(None, None, ["random"], 4, 4, 9, 10,
                                       str(Path(tmp) / "eval.jsonl"), quiet=True)[0]
        self.assertEqual((result["wins"], result["draws"], result["losses"]), (0, 1, 1))
        self.assertEqual((result["pairs_used"], result["pairs_excluded"], result["unfinished"]), (1, 1, 1))
        self.assertEqual(result["score"], 0.25)

    def test_train_eval_worker_partition_matches_serial_pairs(self):
        device = torch.device("cpu")
        config = az.AZConfig(eval_workers=3, eval_device="cpu")
        net = torch.nn.Linear(1, 1)

        def play(net, config, device, side, seed):
            if seed % 3 == 0 and side == jq.BLACK:
                return None
            return 0.0 if seed % 2 else (1.0 if side == jq.RED else -1.0)

        def submit(fn, task):
            future = concurrent.futures.Future()
            future.set_result(fn(task))
            return future

        with mock.patch.object(az, "_play_eval_game", side_effect=play), \
                mock.patch.object(az, "_EVAL_NET", net), \
                mock.patch.object(az, "_EVAL_CONFIG", config), \
                mock.patch.object(az, "_EVAL_DEVICE", device), \
                mock.patch.object(az.futures, "ProcessPoolExecutor") as pool:
            pool.return_value.submit.side_effect = submit
            mp_stats = az.evaluate_vs_ris_mp(net, config, device, 7, random.Random(19))
            config.eval_workers = 1
            serial_stats = az.evaluate_vs_ris(net, config, device, 7, random.Random(19))
        self.assertEqual(mp_stats, serial_stats)
        self.assertEqual(mp_stats["pairs"], 4)
        self.assertGreater(mp_stats["pairs_excluded"], 0)

    def analyze_records(self, records):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text("\n".join(json.dumps(dict(type="game", **record)) for record in records), encoding="utf-8")
            return paired_stats.analyze(str(path), 100, 100, 7)

    def test_paired_stats_separates_opponents(self):
        records = [dict(opponent=opponent, pair=0, seed=3, neural_side=side, score=score)
                   for opponent, score in (("a", 1), ("b", -1)) for side in ("red", "black")]
        result = self.analyze_records(records)
        for opponent, mean in (("a", 1), ("b", 0)):
            self.assertEqual(result["opponents"][opponent]["pairs_used"], 1)
            self.assertEqual(result["opponents"][opponent]["pair_mean"], mean)

    def test_paired_stats_rejects_missing_color_duplicate_and_mismatched_seed(self):
        good = lambda pair, side, **kw: dict(pair=pair, seed=2, neural_side=side, score=1, **kw)
        records = [good(0, "red"), good(1, "red"), good(1, "red"),
                   good(2, "red"), dict(pair=2, seed=3, neural_side="black", score=1),
                   good(3, "red"), good(3, "black")]
        result = self.analyze_records(records)
        self.assertEqual(result["pairs_used"], 1)
        self.assertEqual(result["pairs_excluded_invalid"], 3)

    def test_paired_stats_zero_variance_does_not_report_no_difference(self):
        records = [dict(pair=pair, seed=pair, neural_side=side, score=1)
                   for pair in range(3) for side in ("red", "black")]
        result = self.analyze_records(records)
        self.assertIsNone(result["p_normal"])
        self.assertIsNone(result["t_stat"])

    def test_wilson_interval_bounds(self):
        point, low, high = eval_mod.wilson_score_interval(0, 0, 0)
        self.assertEqual((point, low, high), (0.0, 0.0, 0.0))
        point, low, high = eval_mod.wilson_score_interval(10, 0, 0)
        self.assertEqual(point, 1.0)
        self.assertGreater(low, 0.5)
        self.assertLessEqual(high, 1.0)
        point, low, high = eval_mod.wilson_score_interval(50, 100, 50)
        self.assertAlmostEqual(point, 0.5)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertGreater(low, 0.4)
        self.assertLess(high, 0.6)

    def test_play_one_game_with_random_agents(self):
        factory = lambda rng: eval_mod.RandomOpponent(rng)
        record = eval_mod.play_one_game(factory, factory, jq.RED, seed=7, max_plies=40)
        self.assertEqual(record["seed"], 7)
        self.assertEqual(record["neural_side"], "red")
        self.assertGreater(record["plies"], 0)
        self.assertLessEqual(record["plies"], 40)
        self.assertIn(
            record["reason"],
            {
                jq.REASON_LONG_CHECK,
                jq.REASON_LONG_CHASE,
                jq.REASON_REPETITION,
                jq.REASON_NO_CAPTURE,
                jq.REASON_CHECKMATE,
                jq.REASON_STALEMATE,
                "ply_limit",
            },
        )

    def test_build_opponent_specs(self):
        with self.assertRaises(ValueError):
            eval_mod.build_opponent_factory("unknown", None, None)
        factory = eval_mod.build_opponent_factory("random", None, None)
        self.assertIsInstance(factory(random.Random(1)), eval_mod.RandomOpponent)
        factory = eval_mod.build_opponent_factory("ris:16", None, None)
        self.assertIsInstance(factory(random.Random(1)), eval_mod.RisOpponent)

    def test_parse_opponent_spec_windows_path(self):
        self.assertEqual(eval_mod.parse_opponent_spec("ris:16"), ("ris", None, 16))
        self.assertEqual(eval_mod.parse_opponent_spec("ris"), ("ris", None, None))
        self.assertEqual(eval_mod.parse_opponent_spec("random"), ("random", None, None))
        self.assertEqual(
            eval_mod.parse_opponent_spec(r"model:C:\models\best.pt:400"),
            ("model", r"C:\models\best.pt", 400),
        )
        self.assertEqual(
            eval_mod.parse_opponent_spec(r"model:C:\models\best.pt"),
            ("model", r"C:\models\best.pt", None),
        )
        self.assertEqual(
            eval_mod.parse_opponent_spec("model:data/x.pt"),
            ("model", "data/x.pt", None),
        )

    def test_tally_game_excludes_ply_limit(self):
        self.assertEqual(eval_mod.tally_game({"outcome": None, "neural_side": "red"}), "unfinished")
        self.assertEqual(
            eval_mod.tally_game({"outcome": jq.RESULT_DRAW, "neural_side": "black"}), "draw"
        )
        self.assertEqual(
            eval_mod.tally_game({"outcome": jq.RESULT_RED_WIN, "neural_side": "red"}), "win"
        )
        self.assertEqual(
            eval_mod.tally_game({"outcome": jq.RESULT_RED_WIN, "neural_side": "black"}), "loss"
        )

    def test_main_rejects_missing_model(self):
        """P0-2：主模型缺失必须非零退出且不产出评估文件（fail-fast）。"""
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "eval_missing.jsonl")
            code = eval_mod.main(
                [
                    "--model", os.path.join(tmp, "nope.pt"),
                    "--opponents", "random",
                    "--games", "2",
                    "--sims", "4",
                    "--workers", "1",
                    "--out", out_path,
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(out_path))

    def test_parse_args_rejects_non_positive_limits(self):
        with self.assertRaises(SystemExit):
            eval_mod.parse_args(["--sims", "0"])
        with self.assertRaises(SystemExit):
            eval_mod.parse_args(["--games", "0"])

    def _collect_reveals(self, neural_side: int, seed: int, max_plies: int = 60) -> dict:
        """跑一局并用 spy 记录 格索引->揭示真值（暗子移动或吃暗子）。

        两侧策略不对称（先手总是第一个合法着法、后手总是最后一个），换色后
        走子序列不同；lazy draw 下两局真值会分叉，预物化洗牌下不会。
        """
        seen: dict[int, int] = {}
        original = jq.GameState.apply_move

        def spy_apply(self, move, rng=None, reveal_choice=None, capture_reveal_choice=None):
            undo = original(
                self, move, rng=rng, reveal_choice=reveal_choice,
                capture_reveal_choice=capture_reveal_choice,
            )
            if undo.moved_reveal:
                seen[move[0]] = undo.moved_reveal
            if undo.captured_reveal:
                seen[move[1]] = undo.captured_reveal
            return undo

        class FirstLegal:
            def __init__(self, rng):
                pass

            def select_move(self, state, belief, history):
                moves = state.legal_moves()
                return moves[0] if moves else None

        class LastLegal:
            def __init__(self, rng):
                pass

            def select_move(self, state, belief, history):
                moves = state.legal_moves()
                return moves[-1] if moves else None

        with mock.patch.object(jq.GameState, "apply_move", spy_apply):
            eval_mod.play_one_game(FirstLegal, LastLegal, neural_side, seed, max_plies)
        return seen

    def test_paired_games_share_initial_shuffle(self):
        """P1-5：同 seed 的配对两局必须共享同一初始洗牌（共同格真值一致）。"""
        first = self._collect_reveals(jq.RED, seed=7)
        second = self._collect_reveals(jq.BLACK, seed=7)
        common = set(first) & set(second)
        self.assertTrue(common, "两局没有共同揭示的格子，测试无判别力")
        for cell in common:
            self.assertEqual(
                first[cell], second[cell], f"格子 {cell} 的揭示真值在配对两局间不一致"
            )

    def test_search_config_chance_flags(self):
        """评估搜索配置：默认开启全支持机会节点，与训练及 GUI 一致。"""
        config = eval_mod.build_search_config(400)
        self.assertTrue(config.chance_nodes)
        self.assertEqual(config.chance_k, 0)
        self.assertEqual(config.mcts_simulations, 400)
        self.assertEqual(config.leaf_batch_size, 16)
        config = eval_mod.build_search_config(400, chance_nodes=True, chance_k=0)
        self.assertTrue(config.chance_nodes)
        self.assertEqual(config.chance_k, 0)
        self.assertFalse(eval_mod.build_search_config(400, chance_nodes=False).chance_nodes)

    def test_parse_args_chance_flags(self):
        """CLI：--chance-nodes/--chance-k 可解析，非法 k 拒绝。"""
        args = eval_mod.parse_args([])
        self.assertEqual((args.sims, args.leaf_batch_size), (1600, 16))
        self.assertTrue(args.chance_nodes)
        self.assertEqual(args.chance_k, 0)
        self.assertFalse(eval_mod.parse_args(["--no-chance-nodes"]).chance_nodes)
        args = eval_mod.parse_args(["--chance-nodes", "--chance-k", "0"])
        self.assertTrue(args.chance_nodes)
        self.assertEqual(args.chance_k, 0)
        with self.assertRaises(SystemExit):
            eval_mod.parse_args(["--chance-k", "-1"])

    def test_model_opponent_inherits_chance_flags(self):
        """model 对手（串行与 worker 两条路径）默认继承机会节点设置，保证评估公平。"""
        device = mock.MagicMock()
        with mock.patch.object(eval_mod.azero, "PolicyValueNet", return_value=mock.MagicMock()), \
                mock.patch.object(eval_mod.azero, "load_model_if_exists", return_value=True), \
                mock.patch.object(eval_mod.azero, "NeuralMCTS") as mcts_cls:
            factory = eval_mod.build_opponent_factory(
                "model:fake.pt:8", device, mock.MagicMock(), chance_nodes=True, chance_k=0
            )
            factory(random.Random(1))
            serial_config = mcts_cls.call_args[0][1]
            self.assertTrue(serial_config.chance_nodes)
            self.assertEqual(serial_config.chance_k, 0)

            eval_mod._WORKER_STATE["config"] = eval_mod.build_search_config(8, True, 0)
            eval_mod._WORKER_STATE["opponents"] = {}
            worker_factory = eval_mod._build_opponent_factory_worker("model:fake.pt:8", device)
            worker_factory(random.Random(2))
            worker_config = mcts_cls.call_args[0][1]
            self.assertTrue(worker_config.chance_nodes)
            self.assertEqual(worker_config.chance_k, 0)
        eval_mod._WORKER_STATE.clear()

    def test_resolve_opponent_chance(self):
        """对手机会节点解析：inherit 跟随己方；on/off/k 显式覆盖。"""
        self.assertEqual(eval_mod.resolve_opponent_chance(True, 0, "inherit", None), (True, 0))
        self.assertEqual(eval_mod.resolve_opponent_chance(False, 0, "on", None), (True, 0))
        self.assertEqual(eval_mod.resolve_opponent_chance(True, 2, "off", 5), (False, 5))
        self.assertEqual(eval_mod.resolve_opponent_chance(False, 4, "off", None), (False, 4))

    def test_parse_args_opponent_chance_flags(self):
        """CLI：--opponent-chance-nodes/--opponent-chance-k 可解析，非法值拒绝。"""
        args = eval_mod.parse_args([])
        self.assertEqual(args.opponent_chance_nodes, "inherit")
        self.assertIsNone(args.opponent_chance_k)
        args = eval_mod.parse_args(["--opponent-chance-nodes", "off", "--opponent-chance-k", "3"])
        self.assertEqual(args.opponent_chance_nodes, "off")
        self.assertEqual(args.opponent_chance_k, 3)
        with self.assertRaises(SystemExit):
            eval_mod.parse_args(["--opponent-chance-k", "-1"])
        with self.assertRaises(SystemExit):
            eval_mod.parse_args(["--opponent-chance-nodes", "invalid"])

    def test_worker_opponent_chance_independent(self):
        """worker 路径：对手侧机会节点使用独立设置，不跟随己方 config。"""
        device = mock.MagicMock()
        with mock.patch.object(eval_mod.azero, "PolicyValueNet", return_value=mock.MagicMock()), \
                mock.patch.object(eval_mod.azero, "load_model_if_exists", return_value=True), \
                mock.patch.object(eval_mod.azero, "NeuralMCTS") as mcts_cls:
            eval_mod._WORKER_STATE["config"] = eval_mod.build_search_config(8, True, 0)
            eval_mod._WORKER_STATE["opponent_chance"] = (False, 0)
            eval_mod._WORKER_STATE["opponents"] = {}
            factory = eval_mod._build_opponent_factory_worker("model:fake.pt:8", device)
            factory(random.Random(3))
            config = mcts_cls.call_args[0][1]
            self.assertFalse(config.chance_nodes)
            self.assertEqual(config.chance_k, 0)
        eval_mod._WORKER_STATE.clear()

    def test_python_evaluation_and_worker_inherit_chance_settings(self):
        """直接调用 evaluate/worker 时关闭或截断设置也应默认传给对手。"""
        cases = ((True, 0, None, None), (False, 2, None, None), (False, 2, True, 0))
        for nodes, k, opp_nodes, opp_k in cases:
            expected = (nodes if opp_nodes is None else opp_nodes, k if opp_k is None else opp_k)
            with self.subTest(settings=(nodes, k, opp_nodes, opp_k)), tempfile.TemporaryDirectory() as tmp:
                def fake_game(neural_factory, opponent_factory, neural_side, seed, max_plies):
                    return {"outcome": jq.RESULT_DRAW, "neural_side": "red" if neural_side == jq.RED else "black"}

                with mock.patch.object(eval_mod, "build_opponent_factory") as factory, \
                        mock.patch.object(eval_mod, "play_one_game", side_effect=fake_game):
                    out_path = os.path.join(tmp, "eval.jsonl")
                    eval_mod.evaluate(None, None, ["random"], 2, 8, 1, 1, out_path,
                                      quiet=True, chance_nodes=nodes, chance_k=k,
                                      opponent_chance_nodes=opp_nodes, opponent_chance_k=opp_k)
                    self.assertEqual((factory.call_args.kwargs["chance_nodes"], factory.call_args.kwargs["chance_k"]), expected)
                    with open(out_path, encoding="utf-8") as handle:
                        run = json.loads(handle.readline())
                    self.assertEqual((run["opponent_chance_nodes"], run["opponent_chance_k"]), expected)

                with mock.patch.object(eval_mod.azero, "PolicyValueNet"), \
                        mock.patch.object(eval_mod.torch, "set_num_threads"), \
                        mock.patch.dict(eval_mod._WORKER_STATE, clear=True):
                    eval_mod._init_eval_worker({}, 8, "cpu", nodes, k, opp_nodes, opp_k)
                    self.assertEqual(eval_mod._WORKER_STATE["opponent_chance"], expected)

    def test_model_budgets_and_leaf_batch_reach_serial_worker_and_records(self):
        device = torch.device("cpu")
        net = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(az, "PolicyValueNet"), \
                mock.patch.object(az, "load_model_if_exists", return_value=True):
            model = Path(tmp) / "anchor.pt"
            model.write_bytes(b"config propagation fixture")

            def game(neural_factory, opponent_factory, side, seed, max_plies):
                for factory in (neural_factory, opponent_factory):
                    agent = factory(random.Random(seed))
                    self.assertEqual(agent.mcts.config.leaf_batch_size, 7)
                    self.assertEqual(agent.mcts.config.mcts_simulations, 11)
                return dict(outcome=jq.RESULT_DRAW, score=0, seed=seed,
                            neural_side="red" if side == jq.RED else "black")

            output = Path(tmp) / "eval.jsonl"
            with mock.patch.object(eval_mod, "play_one_game", side_effect=game):
                summaries = eval_mod.evaluate(net, device, [f"model:{model}"], 2, 11, 7, 2,
                                              str(output), quiet=True, leaf_batch_size=7)
            run = json.loads(output.read_text().splitlines()[0])
            for record in (run, run["manifest"]["config"], summaries[0]):
                self.assertEqual(record["leaf_batch_size"], 7)
                self.assertTrue(record["leaf_dedup"])
            with mock.patch.dict(eval_mod._WORKER_STATE, clear=True), \
                    mock.patch.object(eval_mod.torch, "set_num_threads"):
                eval_mod._init_eval_worker({}, 11, "cpu", leaf_batch_size=7)
                for spec, sims in ((f"model:{model}", 11), (f"model:{model}:13", 13)):
                    agent = eval_mod._build_opponent_factory_worker(spec, device)(random.Random(7))
                    self.assertEqual(agent.mcts.config.leaf_batch_size, 7)
                    self.assertEqual(agent.mcts.config.mcts_simulations, sims)

    def test_legacy_ris_uses_eval_leaf_batch_without_mutating_selfplay(self):
        config = az.AZConfig(leaf_batch_size=64, eval_leaf_batch_size=16)
        with mock.patch.object(az, "NeuralMCTS") as mcts, mock.patch.object(az.im, "ISMCTSAgent"):
            mcts.return_value.search.return_value = ([], None)
            az._play_eval_game(None, config, torch.device("cpu"), jq.RED, 8)
        self.assertEqual(mcts.call_args.args[1].leaf_batch_size, 16)
        self.assertEqual(config.leaf_batch_size, 64)
        with self.assertRaises(SystemExit):
            eval_mod.parse_args(["--leaf-batch-size", "0"])


class TestAbRunnerGating(unittest.TestCase):
    def test_eval_command_preserves_explicit_search_settings(self):
        spec = {"name": "test", "opponents": "random"}
        cmd = ab_runner.build_eval_cmd(Path("arm"), spec)
        self.assertTrue(eval_mod.parse_args(cmd[2:]).chance_nodes)
        self.assertEqual(eval_mod.parse_args(cmd[2:]).leaf_batch_size, 64)
        spec["leaf_batch_size"] = 16
        self.assertEqual(eval_mod.parse_args(ab_runner.build_eval_cmd(Path("arm"), spec)[2:]).leaf_batch_size, 16)
        spec.update(chance_nodes=False, chance_k=2)
        args = eval_mod.parse_args(ab_runner.build_eval_cmd(Path("arm"), spec)[2:])
        self.assertFalse(args.chance_nodes)
        self.assertEqual(args.chance_k, 2)
        spec["chance_nodes"] = True
        self.assertTrue(eval_mod.parse_args(ab_runner.build_eval_cmd(Path("arm"), spec)[2:]).chance_nodes)

    def test_historical_ab_configs_keep_distinct_chance_arms(self):
        for name in ("ab_experiments.json", "ab_chance128.json"):
            with open(ab_runner.ROOT / "tools" / name, encoding="utf-8") as handle:
                config = json.load(handle)
            for arm in config["arms"]:
                args = ab_runner.azero.parse_args(config["base_args"] + arm["args"])
                self.assertEqual(args.chance_nodes, arm["name"].startswith("chance-"))
            for spec in config["evals"]:
                cmd = ab_runner.build_eval_cmd(Path("arm"), spec)
                self.assertFalse(eval_mod.parse_args(cmd[2:]).chance_nodes)

    def _write_config(self, tmp: str) -> str:
        config_path = os.path.join(tmp, "ab_config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "start_model": "models/latest.pt",
                    "iterations": 1,
                    "work_dir": os.path.join(tmp, "ab"),
                    "arms": [{"name": "arm1", "args": []}],
                    "evals": [
                        {
                            "name": "h2h",
                            "opponents": "random",
                            "games": 2,
                            "sims": 4,
                            "seed": 1,
                            "workers": 1,
                        }
                    ],
                },
                handle,
            )
        return config_path

    def test_empty_summary_does_not_mark_done(self):
        """P0-2：评估文件没有有效 summary 时不得写 .done。"""
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(tmp)
            arm_dir = os.path.join(tmp, "ab", "arm1")
            os.makedirs(arm_dir)
            with open(os.path.join(arm_dir, "eval_h2h.json"), "w", encoding="utf-8") as handle:
                handle.write('{"type": "run"}\n')

            code = ab_runner.main(["--config", config_path, "--skip-train"])

            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(os.path.join(arm_dir, ".done")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "ab", "results.jsonl")))

    def test_skip_eval_does_not_mark_done(self):
        """P0-2：--skip-eval 没有评估结果，不应把 arm 标记为已完成。"""
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_config(tmp)
            arm_dir = os.path.join(tmp, "ab", "arm1")

            code = ab_runner.main(["--config", config_path, "--skip-train", "--skip-eval"])

            self.assertEqual(code, 0)
            self.assertFalse(os.path.exists(os.path.join(arm_dir, ".done")))


if __name__ == "__main__":
    unittest.main()
