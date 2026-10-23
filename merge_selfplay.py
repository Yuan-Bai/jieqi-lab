import argparse
import os
from azero import CompactSample
import azero


def parse_args():
    parser = argparse.ArgumentParser(description="Merge self-play data into one file.")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(".", "data", "selfplay"),
        help="Self-play directory to scan.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(".", "data", "selfplay", "merged.pkl.gz"),
        help="Output merged file path.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Whether to add mirror-augmented samples during merge.",
    )
    parser.add_argument(
        "--augment-flip",
        action="store_true",
        help="Whether to add vertical-flip (river flip, red/black swap) samples during merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = azero.merge_selfplay_files(
        args.input_dir,
        args.output,
        augment=args.augment,
        augment_flip=args.augment_flip,
    )
    print(f"Merged {count} samples into {args.output}")


if __name__ == "__main__":
    main()
