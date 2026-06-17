from __future__ import annotations

import argparse
import random

from frauddistill.data.schema import FraudDistillSample
from frauddistill.utils.io import read_jsonl, write_jsonl


def split_rows(rows: list[dict], seed: int = 42, train_ratio: float = 0.8, dev_ratio: float = 0.1) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    samples = [FraudDistillSample.model_validate(row) for row in rows]
    rng.shuffle(samples)
    n = len(samples)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    splits = {"train": samples[:train_end], "dev": samples[train_end:dev_end], "test": samples[dev_end:]}
    return {
        split: [sample.model_copy(update={"split": split}).model_dump(mode="json") for sample in values]
        for split, values in splits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    splits = split_rows(list(read_jsonl(args.input_file)), args.seed)
    for split, rows in splits.items():
        write_jsonl(f"{args.output_dir}/{split}.jsonl", rows)
        print(f"{split}: {len(rows)}")


if __name__ == "__main__":
    main()
