import copy
import json
import random
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import ismcts as im
import jieqi as jq
import pika_engine as pika
from tools import eval as ev, paired_stats as ps


def initial():
    state = jq.GameState.initial(rng=random.Random(7))
    return state, im.init_belief_for_game(state)


def test_initial_fen_uses_pika_remaining_pool_and_piece_letters():
    state, belief = initial()
    fen = pika.information_fen(belief.info_state(state, jq.RED))
    assert fen == ("xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX "
                   "w R2N2B2A2C2P5r2n2b2a2c2p5 0 1")
    assert "[" not in fen


def test_bright_knight_elephant_orientation_clock():
    state, belief = initial()
    for square, kind in ((1, jq.TYPE_KNIGHT), (2, jq.TYPE_ELEPHANT)):
        state.board[square] = jq.make_piece(kind, jq.BLACK, 0)
        for observer in (jq.RED, jq.BLACK):
            belief._dec_known(observer, jq.BLACK, kind, False)
    state.side = jq.BLACK
    state.no_capture_plies = 15
    fen = pika.information_fen(belief.info_state(state, jq.BLACK), 23)
    assert fen.startswith("xnbxkxxxx/")
    assert fen.endswith(" b R2N2B2A2C2P5r2n1b1a2c2p5 15 12")


def test_unknown_capture_keeps_owner_uncertainty_and_captor_knowledge():
    state, belief = initial()
    state.board[0] = 0  # Black dark rook square was captured by red.
    state.pool[jq.BLACK][jq.TYPE_ROOK] -= 1
    state.pool_total[jq.BLACK] -= 1
    belief._dec_known(jq.RED, jq.BLACK, jq.TYPE_ROOK, False)
    belief._dec_total_only(jq.BLACK, jq.BLACK, False)
    owner_fen = pika.information_fen(belief.info_state(state, jq.BLACK))
    captor_fen = pika.information_fen(belief.info_state(state, jq.RED))
    assert "r2n2b2a2c2p5" in owner_fen
    assert "r1n2b2a2c2p5" in captor_fen
    # A different compatible hidden world/private opponent memory must serialize
    # identically for the owner. The adapter must never use jq.to_fen(state).
    state.pool[jq.BLACK][jq.TYPE_ROOK] += 1
    state.pool[jq.BLACK][jq.TYPE_CANNON] -= 1
    belief.pool_counts[jq.RED][jq.BLACK][jq.TYPE_ROOK] += 1
    belief.pool_counts[jq.RED][jq.BLACK][jq.TYPE_CANNON] -= 1
    assert owner_fen == pika.information_fen(belief.info_state(state, jq.BLACK))


def test_inconsistent_belief_and_nonrepresentable_dark_piece_fail():
    state, belief = initial()
    info = belief.info_state(state, jq.RED)
    bad = copy.deepcopy(info)
    bad.unknown_removals[jq.RED] = 1
    with pytest.raises(ValueError, match="unknown capture"):
        pika.information_fen(bad)
    bad = copy.deepcopy(info)
    bad.board[0] = jq.make_piece(jq.TYPE_CANNON, jq.BLACK, 1)
    with pytest.raises(ValueError, match="fake type"):
        pika.information_fen(bad)
    bad = copy.deepcopy(info)
    bad.pool_total_known[jq.RED] -= 1
    with pytest.raises(ValueError, match="pool total"):
        pika.information_fen(bad)


def fake_opponent():
    agent = object.__new__(pika.PikaOpponent)
    agent.config = dict(limit="nodes")
    agent.budget = 1234
    agent.search_nodes = agent.searches = 0
    agent.engine = mock.Mock()
    return agent


def test_searchmoves_matches_project_legal_set_and_never_exports_true_state():
    state, belief = initial()
    agent = fake_opponent()
    move = state.legal_moves()[0]
    agent.engine.wait_for.return_value = ["bestmove " + jq.format_move_uci(move)]
    with mock.patch.object(jq, "to_fen", side_effect=AssertionError("hidden export")):
        assert agent.select_move(state, belief, []) == move
    command = agent.engine.send.call_args_list[-1].args[0]
    assert command.startswith("go nodes 1234 searchmoves ")
    assert set(command.split()[4:]) == {jq.format_move_uci(m) for m in state.legal_moves()}


@pytest.mark.parametrize("reply", ["bestmove 0000", "bestmove (none)", "bestmove z0a0", "bestmove a0a0", "bestmove"])
def test_invalid_or_illegal_engine_move_is_error_not_loss(reply):
    state, belief = initial()
    agent = fake_opponent()
    agent.engine.wait_for.return_value = [reply]
    with pytest.raises(pika.PikaError):
        agent.select_move(state, belief, [])


def test_game_closes_external_opponent_on_error():
    agent = mock.Mock()
    agent.select_move.side_effect = pika.PikaError("timeout")
    with pytest.raises(pika.PikaError, match="timeout"):
        ev.play_one_game(ev.RandomOpponent, lambda rng: agent, jq.BLACK, 4, 10)
    agent.close.assert_called_once()


def test_game_closes_first_agent_if_second_factory_fails():
    agent = mock.Mock()
    with pytest.raises(pika.PikaError):
        ev.play_one_game(lambda rng: agent, mock.Mock(side_effect=pika.PikaError()), jq.RED, 4, 10)
    agent.close.assert_called_once()


def test_protocol_timeout_and_process_cleanup():
    process = pika.UciProcess([sys.executable, "-u", "-c", "import time; time.sleep(30)"], .2)
    try:
        with pytest.raises(pika.PikaError, match="timeout"):
            process.wait_for("readyok")
    finally:
        process.close()
    assert process.process.poll() is not None
    assert not process.reader.is_alive()
    process.close()  # Idempotent.


def test_protocol_premature_exit_and_error():
    for script in ("print('bye')", "print('info string ERROR bad network')"):
        process = pika.UciProcess([sys.executable, "-u", "-c", script], 5)
        try:
            with pytest.raises(pika.PikaError):
                process.wait_for("uciok")
        finally:
            process.close()


def test_config_hash_and_finite_budgets(tmp_path):
    exe = tmp_path / "dummy.exe"
    exe.write_bytes(b"test")
    config = pika.prepare_config(dict(executable=str(exe)))
    assert config["source"]["bytes"] == 4
    exe.write_bytes(b"changed")
    with pytest.raises(pika.PikaError, match="changed"):
        pika.prepare_config(config)
    for extra in (dict(timeout=float("nan")), dict(timeout=-1), dict(threads=0), dict(hash_mb=4096)):
        with pytest.raises(ValueError):
            pika.prepare_config(dict(executable=str(exe), **extra))


def test_pika_cli_and_worker_factory():
    args = ev.parse_args(["--opponents", "pika:10000,pika:50000", "--pika-exe", "fake.exe"])
    assert args.pika_limit == "nodes"
    assert ev.parse_opponent_spec("pika:50000") == ("pika", None, 50000)
    with mock.patch.object(pika, "PikaOpponent") as opponent:
        config = {"executable": "fake.exe"}
        with mock.patch.dict(ev._WORKER_STATE, {"pika_config": config}, clear=True):
            ev._build_opponent_factory_worker("pika:700", None)(random.Random(1))
        opponent.assert_called_once_with(config, 700)


def test_bootstrap_resamples_pairs_not_individual_games():
    # Every pair is either two losses or two wins; independent-game resampling
    # would underestimate the variance for this deliberately correlated fixture.
    scores = [0, 0, 1, 1]
    result = ps.pair_summary(scores, 5000, 17)
    assert result["pair_mean"] == .5
    assert result["pentanomial"] == [2, 0, 0, 0, 2]
    assert result["bootstrap_ci95"] == [0, 1]
    assert result == ps.pair_summary(scores, 5000, 17)
    assert not result["bootstrap_degenerate"]
    assert ps.pair_summary([1, 1])["bootstrap_degenerate"]
    assert ps.pair_summary([])["bootstrap_ci95"] == [None, None]
    with pytest.raises(ValueError):
        ps.pair_summary([.3])


def test_stats_exclude_entire_invalid_or_unfinished_pair(tmp_path):
    games = [dict(pair=0, seed=1, neural_side="red", score=1),
             dict(pair=0, seed=1, neural_side="black", score=None),
             dict(pair=1, seed=2, neural_side="red", score=-1),
             dict(pair=1, seed=2, neural_side="black", score=0),
             dict(pair=2, seed=3, neural_side="red", score=1)]
    result = ps.analyze_games("unused", {}, games, 100, 100, 9)
    assert (result["wins"], result["draws"], result["losses"]) == (0, 1, 1)
    assert result["score"] == result["pair_mean"] == .25
    assert result["red"]["score"] == 0
    assert result["black"]["score"] == .5
    assert result["pairs_excluded_invalid"] == 1
    empty = ps.analyze_games("unused", {}, [], 100, 100, 9)
    ps.print_report(empty)  # Empty files must be printable, not 0%-certainty.


def test_external_summary_bootstrap_matches_postprocessing(tmp_path):
    values = [1, 1, -1, -1, 1, 0, 0, -1]
    counter = iter(values)
    def game(n, o, side, seed, max_plies):
        score = next(counter)
        outcome = (jq.RESULT_DRAW if not score else
                   jq.RESULT_RED_WIN if (score > 0) == (side == jq.RED) else jq.RESULT_BLACK_WIN)
        return dict(outcome=outcome, score=score, neural_side="red" if side == jq.RED else "black", seed=seed)
    output = tmp_path / "games.jsonl"
    with mock.patch.object(ev, "play_one_game", side_effect=game):
        summary = ev.evaluate(None, None, ["random"], 8, 2, 17, 2, str(output), quiet=True)[0]
    stats = ps.analyze(str(output), 100, 20000, 17)
    for field in ("score", "pair_mean", "bootstrap_ci95", "pentanomial"):
        assert summary[field] == stats[field]
