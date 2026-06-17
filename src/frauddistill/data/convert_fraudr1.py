from __future__ import annotations

import argparse

from frauddistill.data.common import first_present, map_binary_label, response_risk_type
from frauddistill.data.schema import FraudDistillSample, RiskType, Source
from frauddistill.utils.io import read_jsonl, write_jsonl


def convert_rows(rows: list[dict], split: str = "unspecified") -> list[dict]:
    output = []
    for idx, row in enumerate(rows):
        label = map_binary_label(first_present(row, ["gold_label", "official_label", "safe_label", "is_unsafe"]))
        sample = FraudDistillSample(
            id=str(first_present(row, ["id", "sample_id"], f"fraudr1_{idx:06d}")),
            source=Source.FRAUDR1,
            language=str(first_present(row, ["language"], "unknown")),
            user_query=str(first_present(row, ["prompt", "user_query", "question"])),
            target_model_answer=str(first_present(row, ["target_model_answer", "answer", "response"])),
            context=first_present(row, ["context"], None),
            gold_label=label,
            gold_risk_type=response_risk_type(label, RiskType.FRAUD_ASSISTANCE),
            split=split,
            metadata={
                "target_model": first_present(row, ["target_model", "model"], "unknown"),
                "fraud_category": first_present(row, ["fraud_category", "category"], "unknown"),
            },
        )
        output.append(sample.model_dump(mode="json"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--split", default="unspecified")
    args = parser.parse_args()
    print(f"wrote {write_jsonl(args.output_file, convert_rows(list(read_jsonl(args.input_file)), args.split))} rows")


if __name__ == "__main__":
    main()
