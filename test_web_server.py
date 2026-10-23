"""Web boundary regressions: information privacy, stale writes, and search lifetime."""
import json
import threading
import time

import pytest

import jieqi as jq
from game_record import GameRecord
from web_server import WebGame, WebServer


@pytest.fixture
def game():
    value = WebGame()
    yield value
    value.close()


def action(game, kind, **data):
    game.act(kind, {"revision": game.revision, **data})


def wait_for_search(game):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = game.public_state()
        if not value["pending"] and not value["analysis_pending"]:
            return value
        time.sleep(.01)
    raise AssertionError("search did not finish")


def capture_position(game):
    game.game = GameRecord()
    game.game.add_move((58, 49), record={"reveal": 7, "capture": 0})
    game.game.add_move((19, 82), record={"reveal": 6, "capture": 7})
    game.head = game.game.cursor


def test_human_response_hides_opponents_private_capture_and_real_pools(game):
    capture_position(game)
    public = game.public_state()
    assert public["captured"][str(jq.BLACK)][0]["type"] is None
    assert public["captured"][str(jq.BLACK)][0]["dark"] is True
    for field in ("pool", "belief", "dark_types", "initial", "capture"):
        assert field not in public
    assert all("capture" not in node for node in public["history"])
    # The capturer sees the type, while both boards remain identical.
    game.human_side = jq.BLACK
    game.paused = True
    black = game.public_state()
    assert black["captured"][str(jq.BLACK)][0]["type"] == jq.TYPE_PAWN
    assert black["board"] == public["board"]


@pytest.mark.parametrize("mode", ["local", "aiai"])
def test_shared_screen_and_spectator_do_not_expose_either_private_capture(game, mode):
    capture_position(game)
    game.mode, game.paused = mode, True
    assert game.public_state()["captured"][str(jq.BLACK)][0]["type"] is None


def test_capture_sequence_has_safe_move_metadata_only_on_current_branch(game):
    capture_position(game)
    game.mode, game.paused = "local", True
    capture = game.public_state()["captured"][str(jq.BLACK)][0]
    assert capture == {"side": jq.RED, "dark": True, "type": None,
                       "node": 2, "depth": 2, "source": 19, "target": 82}
    action(game, "navigate", cursor=1)
    assert game.public_state()["captured"][str(jq.BLACK)] == []
    action(game, "continue")
    action(game, "move", **{"from": 0, "to": 9})
    assert game.game.cursor != 2
    assert game.public_state()["captured"][str(jq.BLACK)] == []
    action(game, "navigate", cursor=2)
    assert game.public_state()["captured"][str(jq.BLACK)][0] == capture
    game.human_side, game.mode = jq.BLACK, "human"
    assert game.public_state()["captured"][str(jq.BLACK)][0]["type"] == jq.TYPE_PAWN


def test_initial_captures_do_not_get_fabricated_move_numbers(game):
    game.game = GameRecord(captured={jq.BLACK: [(jq.RED, False, jq.TYPE_PAWN)], jq.RED: []})
    game.game.add_move((58, 49), record={"reveal": 7, "capture": 0})
    game.game.add_move((19, 82), record={"reveal": 6, "capture": 7})
    game.head = game.game.cursor
    captures = game.public_state()["captured"][str(jq.BLACK)]
    assert captures[0]["node"] is None and captures[0]["depth"] is None
    assert captures[1]["node"] == 2 and captures[1]["depth"] == 2


def test_record_identity_survives_updates_but_changes_on_new_or_import(game):
    original = game.public_state()["record_id"]
    action(game, "settings", simulations=13)
    assert game.public_state()["record_id"] == original
    action(game, "new", mode="local")
    new_id = game.public_state()["record_id"]
    assert new_id != original
    action(game, "import", record=GameRecord().to_dict())
    assert game.public_state()["record_id"] not in (original, new_id)


def test_running_export_is_public_and_finished_export_is_gui_compatible(game):
    capture_position(game)
    public = game.export()
    assert public["format"] == "jieqi-web-public"
    assert "initial" not in public and "nodes" not in public
    assert all("capture" not in node for node in public["history"])
    action(game, "resign")
    complete = game.export()
    assert complete["format"] == "jieqi-replay"
    restored = GameRecord.from_dict(json.loads(json.dumps(complete)))
    assert restored.current.state.board == game.game.current.state.board


def test_review_export_preserves_every_branch_cursor_live_head_and_resignation(game):
    capture_position(game)
    game.game.cursor = 0
    alternative = game.game.add_move((60, 51), record={"reveal": 7, "capture": 0})
    game.game.cursor = game.head
    action(game, "resign")
    action(game, "navigate", cursor=1)
    exported = json.loads(json.dumps(game.export()))
    assert exported["format"] == game.public_state()["export_format"] == "jieqi-replay"
    assert exported["cursor"] == 1 and exported["web_head"] == 2
    assert len(exported["nodes"]) == 3  # Includes later moves and the other branch.

    loaded = WebGame()
    try:
        action(loaded, "import", record=exported)
        assert loaded.reviewing and loaded.game.cursor == 1 and loaded.head == 2
        assert len(loaded.game.nodes) == len(game.game.nodes)
        for node_id, node in game.game.nodes.items():
            assert loaded.game.nodes[node_id].record == node.record
            assert loaded.game.nodes[node_id].state.board == node.state.board
            assert loaded.game.nodes[node_id].captured == node.captured
        assert loaded.export()["format"] == "jieqi-replay"
        action(loaded, "resume")
        assert loaded.game.cursor == 2 and loaded.outcome() == (-1, "resign")
        action(loaded, "navigate", cursor=alternative.id)
        action(loaded, "continue")
        assert loaded.resigned is None
        assert loaded.export()["format"] == "jieqi-web-public"
    finally:
        loaded.close()


def test_legacy_resignation_export_restores_live_head_without_new_metadata(game):
    capture_position(game)
    action(game, "resign")
    action(game, "navigate", cursor=0)
    legacy = game.export()
    legacy.pop("web_head")
    action(game, "import", record=legacy)
    assert game.game.cursor == 0 and game.head == 2 and game.resigned == jq.RED


def test_unfinished_review_export_is_explicitly_public_but_not_truncated(game):
    capture_position(game)
    action(game, "navigate", cursor=0)
    public = game.export()
    assert public["format"] == game.public_state()["export_format"] == "jieqi-web-public"
    assert public["cursor"] == 0 and len(public["history"]) == 2
    assert "initial" not in public and "web_resignation" not in public
    assert all("capture" not in item and "reveal" not in item for item in public["history"])


@pytest.mark.parametrize("metadata", [
    {"web_head": True}, {"web_head": 100}, {"web_resignation": []},
    {"web_resignation": {"side": True, "node": 0}},
    {"web_resignation": {"side": 1, "node": 100}},
    {"web_head": 1, "web_resignation": {"side": 1, "node": 0}},
])
def test_bad_saved_head_or_resignation_does_not_replace_current_game(game, metadata):
    capture_position(game)
    original, revision, record_id = game.game, game.revision, game.record_id
    record = original.to_dict()
    record.update(metadata)
    with pytest.raises(ValueError):
        action(game, "import", record=record)
    assert game.game is original and game.revision == revision and game.record_id == record_id


def test_stale_and_illegal_moves_never_change_game(game):
    revision = game.revision
    with pytest.raises(ValueError, match="局面已经更新"):
        game.act("move", {"revision": revision - 1, "from": 58, "to": 49})
    with pytest.raises(ValueError):
        action(game, "move", **{"from": 58, "to": 0})
    assert game.game.cursor == 0 and game.revision == revision
    for invalid in (True, "58", -1, 90):
        with pytest.raises(ValueError):
            action(game, "move", **{"from": invalid, "to": 49})
    assert len(game.game.nodes) == 1


def test_actual_ris_reply_and_hint_use_current_legal_moves(game):
    action(game, "new", simulations=8)
    action(game, "move", **{"from": 58, "to": 49})
    game.public_state()  # Schedules an actual search.
    value = wait_for_search(game)
    assert value["depth"] == 2 and value["side"] == jq.RED
    assert value["last_search"]["simulations"] == 8
    assert value["last_search"]["device"] == "cpu"
    assert value["last_search"]["result_wait_seconds"] >= 0
    action(game, "analysis-settings", engine="ris", simulations=8)
    action(game, "hint")
    hint = wait_for_search(game)
    assert hint["hint"] in hint["legal_moves"]
    assert hint["depth"] == 2
    assert hint["analysis"]["device"] == "cpu"


def test_az_adapter_passes_move_history_not_boards(game, monkeypatch, tmp_path):
    from types import SimpleNamespace
    import search_engine
    import web_server

    received = []

    class FakeAz:
        def __init__(self, *args, **kwargs):
            self.config = SimpleNamespace(leaf_batch_size=None)

        def analyse(self, state, belief, history, **kwargs):
            for src, dst in history:  # A 90-square board would fail unpacking here.
                assert 0 <= src < 90 and 0 <= dst < 90
            received.append(history)
            assert self.config.leaf_batch_size == 5
            return SimpleNamespace(best=SimpleNamespace(move=state.legal_moves()[0]), win_rate=.6)

    monkeypatch.setattr(search_engine, "AzEngine", FakeAz)
    model = tmp_path / "test.pt"
    model.write_bytes(b"fake-model")
    monkeypatch.setattr(web_server, "model_catalog", lambda: {"test.pt": model})
    action(game, "new", engine="az", model="test.pt", mode="local", simulations=8)
    action(game, "analysis-settings", engine="az", model="test.pt", simulations=8, leaf_batch_size=5)
    action(game, "move", **{"from": 58, "to": 49})
    action(game, "hint")
    value = wait_for_search(game)
    assert received == [[(58, 49)]]
    assert value["hint"] in value["legal_moves"]
    assert value["analysis"]["win_rate"] == .6
    assert value["analysis"]["red_win_rate"] == .4  # Black is the root player.
    assert value["analysis"]["black_win_rate"] == .6
    assert value["analysis"]["effective_leaf_batch_size"] == 5
    assert value["last_search"] is None


def test_resign_reveals_captures_and_undo_returns_to_human_turn(game):
    capture_position(game)
    action(game, "resign")
    ended = game.public_state()
    assert ended["outcome"] == jq.RESULT_BLACK_WIN
    assert ended["captured"][str(jq.BLACK)][0]["type"] == jq.TYPE_PAWN
    action(game, "undo")
    value = game.public_state()
    assert value["side"] == jq.RED and value["depth"] == 0
    assert value["outcome"] is None
    assert len(game.game.nodes) == 3  # Undo keeps the existing branch.


def test_review_and_branch_reuse_original_reveal(game):
    action(game, "new", mode="local")
    action(game, "move", **{"from": 58, "to": 49})
    original = game.game.current
    action(game, "navigate", cursor=0)
    assert game.public_state()["reviewing"]
    with pytest.raises(ValueError):
        action(game, "move", **{"from": 58, "to": 49})
    action(game, "continue")
    action(game, "move", **{"from": 58, "to": 49})
    assert game.game.current is original
    action(game, "navigate", cursor=0)
    action(game, "continue")
    alternative = next(m for m in game.game.current.state.legal_moves() if m != (58, 49))
    action(game, "move", **{"from": alternative[0], "to": alternative[1]})
    assert len(game.game.nodes[0].children) == 2
    assert game.game.nodes[original.id].record == original.record


def test_reset_invalidates_inflight_result(game):
    started, release = threading.Event(), threading.Event()

    def slow(rng):
        started.set()
        release.wait(2)
        rng.random()  # Cooperative cancellation boundary.
        return {"move": (58, 49), "value": None}

    token = game.runner.submit(slow)
    game.pending = {"kind": "move", "token": token, "node": 0, "started": time.monotonic()}
    assert started.wait(1)
    action(game, "new", mode="local")
    release.set()
    game.runner._thread.join(.05)
    assert game.public_state()["depth"] == 0
    assert game.pending is None


def test_import_preserves_tree_and_bad_import_does_not_replace_game(game):
    record = GameRecord()
    record.add_move((58, 49), record={"reveal": 7, "capture": 0})
    action(game, "import", record=record.to_dict())
    assert game.reviewing and game.mode == "local"
    assert game.game.current.state.board == record.current.state.board
    current = game.game
    invalid = record.to_dict()
    invalid["nodes"][0]["move"]["to"] = 0
    with pytest.raises(ValueError):
        action(game, "import", record=invalid)
    assert game.game is current
    with pytest.raises(ValueError, match="棋谱格式错误"):
        action(game, "import", record={"format": "jieqi-replay", "version": 2,
                                       "ruleset_version": jq.RULESET_VERSION, "initial": None, "nodes": []})
    assert game.game is current


@pytest.mark.parametrize("captured", [{"0": [[]]}, {"0": [[1, False, 8]]}, {"0": [[1, 1, 7]]}])
def test_bad_initial_captures_preserve_game_and_public_state(game, captured):
    action(game, "new", mode="local")
    capture_position(game)
    original = game.game
    before = game.public_state()
    invalid = original.to_dict()
    invalid["initial"]["captured"] = captured
    with pytest.raises(ValueError, match="棋谱格式错误.*吃子"):
        action(game, "import", record=invalid)
    assert game.game is original
    assert game.public_state() == before


def test_aiai_starts_paused_and_pause_cancels_pending_search(game):
    action(game, "new", mode="aiai", simulations=8)
    assert game.public_state()["pending"] is None
    action(game, "pause")
    assert game.public_state()["pending"]["kind"] == "move"
    action(game, "pause")
    assert game.paused and game.pending is None
    assert game.public_state()["depth"] == 0


def test_custom_play_settings_apply_without_replacing_record(game):
    action(game, "new", mode="local", simulations=1234, leaf_batch_size=300)
    action(game, "move", **{"from": 58, "to": 49})
    original = game.game
    action(game, "settings", simulations=2345, leaf_batch_size=128)
    state = game.public_state()
    assert state["simulations"] == 2345 and state["leaf_batch_size"] == 128
    assert state["depth"] == 1 and game.game is original
    for settings in ({"simulations": 0}, {"simulations": 1.2}, {"leaf_batch_size": 0},
                     {"leaf_batch_size": True}, {"leaf_batch_size": 4097}):
        with pytest.raises(ValueError):
            action(game, "settings", **settings)
        assert game.simulations == 2345 and game.leaf_batch_size == 128


def test_preferences_restore_marker_is_only_fresh_before_first_successful_action(game):
    assert game.public_state()["preferences_ready"] is False
    with pytest.raises(ValueError):
        action(game, "settings", simulations=0)
    assert game.public_state()["preferences_ready"] is False
    action(game, "settings", simulations=1234)
    assert game.public_state()["preferences_ready"] is True
    assert game.game.cursor == 0  # Even an untouched board can have deliberate settings.
    action(game, "new", mode="local")
    assert game.public_state()["preferences_ready"] is True


def test_analysis_does_not_block_moves_and_old_hint_is_cancelled(game, monkeypatch):
    import ismcts as im
    started, release = threading.Event(), threading.Event()

    def slow_search(agent, state, belief):
        started.set()
        release.wait(2)
        agent.rng.random()  # Same cancellation boundary used by the real search.
        return state.legal_moves()[0], None

    monkeypatch.setattr(im.ISMCTSAgent, "select_move", slow_search)
    action(game, "new", mode="local")
    action(game, "analysis-settings", engine="ris", simulations=17)
    action(game, "hint")
    assert started.wait(1)
    assert game.pending is None and game.analysis_pending is not None
    action(game, "move", **{"from": 58, "to": 49})
    release.set()
    game.analysis_runner._thread.join(.05)
    state = game.public_state()
    assert state["depth"] == 1
    assert state["analysis"] is None and state["hint"] is None
    assert state["analysis_pending"] is None


@pytest.fixture
def fake_az(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import search_engine
    import web_server

    class FakeAz:
        calls = []
        instances = 0

        def __init__(self, *args, **kwargs):
            FakeAz.instances += 1
            self.config = SimpleNamespace(leaf_batch_size=None)

        def analyse(self, state, belief, history, simulations):
            FakeAz.calls.append((simulations, self.config.leaf_batch_size, state.side))
            return SimpleNamespace(best=SimpleNamespace(move=state.legal_moves()[0]), win_rate=.7)

    model = tmp_path / "model.pt"
    model.write_bytes(b"fake-weights")
    monkeypatch.setattr(web_server, "model_catalog", lambda: {"fake.pt": model})
    monkeypatch.setattr(search_engine, "AzEngine", FakeAz)
    return FakeAz


def test_play_leaf_is_clamped_and_parameter_updates_reuse_model(game, fake_az):
    action(game, "new", engine="az", model="fake.pt", human_side=jq.BLACK,
           simulations=9, leaf_batch_size=64)
    game.public_state()
    wait_for_search(game)
    assert fake_az.calls == [(9, 9, jq.RED)]
    assert game.last_search["leaf_batch_size"] == 9
    action(game, "settings", simulations=11, leaf_batch_size=3)
    action(game, "move", **dict(zip(("from", "to"), game.game.current.state.legal_moves()[0])))
    game.public_state()
    wait_for_search(game)
    assert fake_az.calls[-1] == (11, 3, jq.RED)
    assert fake_az.instances == 1


def test_analysis_has_separate_budget_cache_and_red_perspective(game, fake_az):
    action(game, "new", mode="local", simulations=123)
    action(game, "analysis-settings", engine="az", model="fake.pt", simulations=7, leaf_batch_size=30)
    action(game, "evaluate")
    first = wait_for_search(game)
    assert fake_az.calls == [(7, 7, jq.RED)]
    assert first["analysis"]["red_win_rate"] == .7
    assert first["analysis"]["black_win_rate"] == pytest.approx(.3)
    assert first["hint"] is None  # Evaluation alone does not mark a move.
    action(game, "hint")
    assert game.hint == first["analysis"]["move"]
    assert game.analysis_pending is None and len(fake_az.calls) == 1  # Cached at this node.
    action(game, "move", **{"from": 58, "to": 49})
    assert game.analysis_result is None
    action(game, "evaluate")
    second = wait_for_search(game)
    assert second["analysis"]["red_win_rate"] == pytest.approx(.3)
    assert second["analysis"]["black_win_rate"] == .7
    assert game.simulations == 123  # Analysis does not change the playing budget.
    action(game, "navigate", cursor=0)
    action(game, "evaluate")
    assert game.analysis_result == first["analysis"]
    assert len(fake_az.calls) == 2


def test_automatic_analysis_and_settings_changes_cancel_old_results(game, fake_az):
    action(game, "new", mode="local")
    action(game, "analysis-settings", engine="az", model="fake.pt", simulations=5,
           leaf_batch_size=2, auto_hint=True, auto_evaluate=True)
    game.public_state()
    first = wait_for_search(game)
    assert first["hint"] in first["legal_moves"] and first["analysis"]["red_win_rate"] == .7
    action(game, "move", **{"from": 58, "to": 49})
    game.public_state()
    second = wait_for_search(game)
    assert second["analysis"]["node"] == game.game.cursor == 1
    assert len(fake_az.calls) == 2
    action(game, "analysis-settings", simulations=13, leaf_batch_size=4)
    assert game.analysis_result is None
    game.public_state()
    third = wait_for_search(game)
    assert third["analysis"]["simulations"] == 13
    assert fake_az.calls[-1] == (13, 4, jq.BLACK)
    assert fake_az.instances == 1


def test_analysis_failure_is_not_a_play_failure(game, fake_az, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("model error")

    monkeypatch.setattr(fake_az, "analyse", fail)
    action(game, "new", mode="local")
    action(game, "analysis-settings", engine="az", model="fake.pt", auto_evaluate=True)
    game.public_state()
    state = wait_for_search(game)
    assert "model error" in state["analysis_error"]
    assert state["error"] is None and not state["paused"]
    assert game.public_state()["analysis_pending"] is None  # No endless automatic retry.
    action(game, "move", **{"from": 58, "to": 49})
    assert game.game.current.depth == 1


def test_replaced_weights_invalidate_same_node_analysis_cache(game, fake_az):
    import web_server
    action(game, "new", mode="local")
    action(game, "analysis-settings", engine="az", model="fake.pt", simulations=5, auto_evaluate=True)
    game.public_state()
    first = wait_for_search(game)
    assert first["analysis"]["node"] == 0
    web_server.model_catalog()["fake.pt"].write_bytes(b"updated-weight-content")
    game.public_state()
    second = wait_for_search(game)
    assert second["analysis"]["node"] == 0
    assert fake_az.instances == 2 and len(fake_az.calls) == 2


def test_ris_evaluation_is_explicitly_unavailable(game):
    action(game, "analysis-settings", engine="ris")
    for kind in ("evaluate",):
        with pytest.raises(ValueError, match="AlphaZero"):
            action(game, kind)
    previous = dict(game.analysis_settings)
    with pytest.raises(ValueError, match="AlphaZero"):
        action(game, "analysis-settings", auto_evaluate=True)
    assert game.analysis_settings == previous
    assert game.analysis_result is None


def test_arbitrary_model_path_and_unknown_modes_are_rejected_atomically(game):
    original = game.game
    for settings in ({"engine": "az", "model": "../../secrets.pt"}, {"mode": "online"},
                     {"simulations": 999999}, {"human_side": True}):
        with pytest.raises(ValueError):
            action(game, "new", **settings)
        assert game.game is original


def test_http_boundary_sessions_assets_and_same_origin():
    # Socket-free HTTP integration: exercise the actual handler with in-memory streams.
    import io
    from web_server import Handler, trusted_http_hosts

    server = WebServer.__new__(WebServer)
    server.games, server.games_lock = {}, threading.Lock()
    server.trusted_hosts = trusted_http_hosts("127.0.0.1", ("127.0.0.1", 8765))

    class MemorySocket:
        def __init__(self, request):
            self.input = io.BytesIO(request)
            self.output = io.BytesIO()

        def makefile(self, mode, *args):
            return self.input

        def sendall(self, data):
            self.output.write(data)

    def request(path, *, cookie="", body=None, origin=None, host="127.0.0.1:8765", extra_headers=()):
        content = b"" if body is None else json.dumps(body).encode()
        headers = [f"{'GET' if body is None else 'POST'} {path} HTTP/1.0"]
        if host is not None:
            headers.append(f"Host: {host}")
        if cookie:
            headers.append(f"Cookie: {cookie}")
        if origin:
            headers.append(f"Origin: {origin}")
        if body is not None:
            headers += ["Content-Type: application/json", f"Content-Length: {len(content)}"]
        headers.extend(extra_headers)
        socket = MemorySocket(("\r\n".join(headers) + "\r\n\r\n").encode() + content)
        Handler(socket, ("127.0.0.1", 1234), server)
        raw_headers, payload = socket.output.getvalue().split(b"\r\n\r\n", 1)
        return raw_headers.decode(), payload

    try:
        for path, body in (("/", None), ("/app.js", None), ("/api/health", None),
                           ("/api/config", None), ("/api/state", None),
                           ("/api/export", None), ("/api/new", {})):
            for host in (None, "", "audit.invalid:8765", "127.0.0.1:9999",
                         "127.0.0.1:8765.audit.invalid", "localhost.:8765",
                         "user@127.0.0.1:8765", "0.0.0.0:8765"):
                assert "403" in request(path, body=body, host=host)[0]
            # An attacker-controlled Host and matching Origin must still be rejected.
            assert "403" in request(path, body=body, host="audit.invalid:8765",
                                    origin="http://audit.invalid:8765")[0]
            for origin in ("null", "http://localhost:8765", "https://127.0.0.1:8765",
                           "http://127.0.0.1:8765.audit.invalid", "http://127.0.0.1:8765/"):
                assert "403" in request(path, body=body, origin=origin)[0]
            for headers in (("Host: 127.0.0.1:8765",), ("Sec-Fetch-Site: cross-site",),
                            ("Origin: http://127.0.0.1:8765", "Origin: http://127.0.0.1:8765")):
                assert "403" in request(path, body=body, extra_headers=headers)[0]
        assert server.games == {}  # Rejected reads/writes do not allocate a session.
        for host in ("localhost:8765", "LOCALHOST:8765", "127.0.0.1:8765"):
            assert "200" in request("/api/health", host=host, origin=f"http://{host.lower()}")[0]
        headers, payload = request("/api/state")
        state = json.loads(payload)
        assert "200 OK" in headers and state["depth"] == 0
        cookie = next(line.split(": ", 1)[1].split(";", 1)[0] for line in headers.splitlines() if line.startswith("Set-Cookie:"))
        request("/api/new", cookie=cookie, body={"revision": state["revision"], "mode": "local"},
                origin="http://127.0.0.1:8765")
        state = json.loads(request("/api/state", cookie=cookie)[1])
        assert state["mode"] == "local"
        assert "403" in request("/api/new", cookie=cookie,
                                body={"revision": state["revision"], "mode": "human"},
                                host="audit.invalid:8765", origin="http://audit.invalid:8765")[0]
        assert json.loads(request("/api/state", cookie=cookie)[1]) == state
        invalid = GameRecord().to_dict()
        invalid["initial"]["captured"]["0"] = [[]]
        rejected, error = request("/api/import", cookie=cookie,
                                  body={"revision": state["revision"], "record": invalid})
        assert "400" in rejected and "吃子" in json.loads(error)["error"]
        assert json.loads(request("/api/state", cookie=cookie)[1]) == state
        assert json.loads(request("/api/state")[1])["mode"] == "human"
        assert "403" in request("/api/new", cookie=cookie, body={}, origin="https://elsewhere.example")[0]
        assert "404" in request("/../jieqi.py")[0]
        html_headers, html = request("/")
        assert "200 OK" in html_headers and "text/html" in html_headers
        assert "JIEQI LAB" in html.decode()
        assert "javascript" in request("/app.js")[0]
        assert "404" in request("/not-found")[0]
    finally:
        for item in server.games.values():
            item.close()


@pytest.mark.parametrize("configured,bound,expected", [
    ("localhost", ("127.0.0.1", 12345), {"localhost:12345", "127.0.0.1:12345"}),
    ("127.0.0.1", ("127.0.0.1", 80), {"localhost", "127.0.0.1", "localhost:80", "127.0.0.1:80"}),
    ("0.0.0.0", ("0.0.0.0", 8765), {"localhost:8765", "127.0.0.1:8765"}),
    ("", ("0.0.0.0", 8765), {"localhost:8765", "127.0.0.1:8765"}),
    ("192.0.2.10", ("192.0.2.10", 8765), {"192.0.2.10:8765"}),
    ("JIEQI.TEST", ("192.0.2.10", 8765), {"jieqi.test:8765", "192.0.2.10:8765"}),
])
def test_trusted_hosts_use_configured_address_and_actual_bound_port(configured, bound, expected):
    from web_server import trusted_http_hosts
    assert trusted_http_hosts(configured, bound) == expected
