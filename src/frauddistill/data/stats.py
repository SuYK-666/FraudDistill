from __future__ import annotations

import argparse
import json
from collections import Counter

from frauddistill.data.schema import FraudDistillSample
from frauddistill.utils.io import read_jsonl


def dataset_stats(rows: list[dict]) -> dict:
    samples = [FraudDistillSample.model_validate(row) for row in rows]
    return {
        "num_samples": len(samples),
        "by_source": dict(Counter(sample.source.value for sample in samples)),
        "by_label": dict(Counter(sample.gold_label.value for sample in samples)),
        "by_risk_type": dict(Counter((sample.gold_risk_type.value if sample.gold_risk_type else "none") for sample in samples)),
        "by_language": dict(Counter(sample.language for sample in samples)),
        "by_split": dict(Counter(sample.split for sample in samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file")
    args = parser.parse_args()
    stats = dataset_stats(list(read_jsonl(args.input_file)))
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
