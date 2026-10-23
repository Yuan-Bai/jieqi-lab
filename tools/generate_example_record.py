"""Generate a synthetic branch replay; never reads a model or starts a search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game_record import GameRecord


def build_example():
    game = GameRecord(metadata={
        "title": "合成示例：翻子、暗子被吃与分支",
        "source": "tools/generate_example_record.py",
        "synthetic": True,
        "note": "全部为人工构造的教学着法；完整棋谱含私有揭示，不代表真实对局或棋力。",
    })
    game.add_move((58, 49), record={"reveal": 7, "capture": 0})
    game.add_move((19, 82), record={"reveal": 6, "capture": 7})
    game.cursor = 1
    game.add_move((27, 36), record={"reveal": 7, "capture": 0})
    game.cursor = 2
    return game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write a NEW file; defaults to stdout")
    args = parser.parse_args()
    content = json.dumps(build_example().to_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.out is None:
        print(content, end="")
    else:
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)


if __name__ == "__main__":
    main()
