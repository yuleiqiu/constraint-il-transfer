"""Extract deterministic SDP-style pairs from accepted correction episodes."""

import argparse
import json
from pathlib import Path

from paired_correction import extract_pair_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("outputs/targeted_finetuning/correction_raw.hdf5"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/targeted_finetuning/correction_pairs.hdf5"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = extract_pair_file(args.raw, args.out, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
