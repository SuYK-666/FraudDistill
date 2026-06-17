from __future__ import annotations

import argparse

from frauddistill.data.schema import FraudDistillSample
from frauddistill.utils.io import read_jsonl, write_jsonl


def merge_files(input_files: list[str], output_file: str) -> int:
    rows = []
    seen = set()
    for path in input_files:
        for row in read_jsonl(path):
            sample = FraudDistillSample.model_validate(row)
            if sample.id in seen:
                raise ValueError(f"duplicate sample id: {sample.id}")
            seen.add(sample.id)
            rows.append(sample.model_dump(mode="json"))
    return write_jsonl(output_file, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    print(f"wrote {merge_files(args.input_files, args.output_file)} rows")


if __name__ == "__main__":
    main()
