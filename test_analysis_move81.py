"""历史局面分析可使用外部棋谱与模型，无需导入训练依赖。"""
from unittest import mock

import pytest

import analysis_move81 as analysis


@pytest.mark.parametrize("mode,function", [
    ("position", "position_report"),
    ("gamble", "gamble_report"),
])
def test_reports_accept_external_replay(mode, function):
    with mock.patch.object(analysis, function) as report:
        analysis.cli([mode, "--replay", "games/my replay.json"])
    report.assert_called_once_with("games/my replay.json")


def test_search_accepts_external_replay_and_model():
    with mock.patch.object(analysis, "search_report") as search:
        analysis.cli(["search", "64", "--replay", "games/my replay.json",
                      "--model", "models/custom.pt"])
    search.assert_called_once_with(64, "games/my replay.json", "models/custom.pt")


def test_invalid_mode_does_not_read_replay():
    with mock.patch.object(analysis, "load_game") as load:
        with pytest.raises(SystemExit) as error:
            analysis.cli(["unknown"])
    assert error.value.code == 2
    load.assert_not_called()
