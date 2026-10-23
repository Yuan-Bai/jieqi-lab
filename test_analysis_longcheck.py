"""棋谱分析入口的模式与路径解析回归，不读取真实棋谱或模型。"""
from unittest import mock

import pytest

import analysis_longcheck as analysis


@pytest.mark.parametrize("argv,expected", [
    ([], (analysis.REPLAY, 59)),
    (["trace"], (analysis.REPLAY, 59)),
    (["replay.json"], ("replay.json", 59)),
    (["replay.json", "59"], ("replay.json", 59)),
    (["trace", "replay.json", "61"], ("replay.json", 61)),
    (["trace", "C:/games/my replay.json", "59"], ("C:/games/my replay.json", 59)),
])
def test_trace_accepts_documented_and_explicit_mode_calls(argv, expected):
    with mock.patch.object(analysis, "main") as trace, mock.patch.object(analysis, "eval_report") as evaluate:
        analysis.cli(argv)
    trace.assert_called_once_with(*expected)
    evaluate.assert_not_called()


@pytest.mark.parametrize("argv,expected", [
    (["eval"], (58, 400, False)),
    (["eval", "60", "800", "clear"], (60, 800, True)),
])
def test_eval_keeps_existing_arguments(argv, expected):
    with mock.patch.object(analysis, "main") as trace, mock.patch.object(analysis, "eval_report") as evaluate:
        analysis.cli(argv)
    trace.assert_not_called()
    evaluate.assert_called_once_with(*expected, path=analysis.REPLAY, model=analysis.MODEL)


def test_eval_accepts_external_replay_and_model_paths():
    with mock.patch.object(analysis, "eval_report") as evaluate:
        analysis.cli(["eval", "60", "800", "--replay", "games/my replay.json",
                      "--model", "models/custom.pt"])
    evaluate.assert_called_once_with(60, 800, False,
                                    path="games/my replay.json", model="models/custom.pt")


@pytest.mark.parametrize("argv", [
    ["trace", "replay.json", "invalid"],
    ["eval", "58", "invalid"],
    ["eval", "58", "400", "unexpected"],
])
def test_invalid_arguments_fail_before_reading_replay_or_model(argv, capsys):
    with mock.patch.object(analysis, "main") as trace, mock.patch.object(analysis, "eval_report") as evaluate:
        with pytest.raises(SystemExit) as error:
            analysis.cli(argv)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    trace.assert_not_called()
    evaluate.assert_not_called()
