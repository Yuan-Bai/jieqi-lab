"""The public sample is reproducible, importable, and respects Web privacy."""
import json
from pathlib import Path

import jieqi as jq
from game_record import GameRecord
from tools.generate_example_record import build_example
from web_server import WebGame


def test_public_example_matches_generator_and_roundtrips():
    path = Path(__file__).parent / "examples/synthetic-branches.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == build_example().to_dict()
    restored = GameRecord.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.nodes[1].children == [2, 3]
    assert restored.cursor == 2


def test_synthetic_private_capture_is_filtered_after_web_import():
    game = WebGame()
    try:
        game.act("import", {"revision": game.revision, "record": build_example().to_dict()})
        view = game.public_state()
        assert view["captured"][str(jq.BLACK)][0]["type"] is None
        assert view["captured"][str(jq.BLACK)][0]["dark"] is True
        exported = game.export()
        assert exported["format"] == "jieqi-web-public"
        assert "initial" not in exported and "nodes" not in exported
    finally:
        game.close()
