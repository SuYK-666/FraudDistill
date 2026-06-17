from __future__ import annotations

import argparse
from pathlib import Path

from frauddistill.data.common import first_present, map_binary_label, response_risk_type
from frauddistill.data.schema import FraudDistillSample, RiskType, Source
from frauddistill.utils.io import read_jsonl, write_jsonl


def convert_rows(rows: list[dict], split: str = "unspecified") -> list[dict]:
    converted = []
    for idx, row in enumerate(rows):
        label = map_binary_label(first_present(row, ["label", "gold_label", "is_hallucinated"]))
        sample = FraudDistillSample(
            id=str(first_present(row, ["id", "sample_id"], f"halubench_{idx:06d}")),
            source=Source.HALUBENCH,
            language=str(first_present(row, ["language"], "en")),
            user_query=str(first_present(row, ["question", "query", "user_query"])),
            target_model_answer=str(first_present(row, ["answer", "response", "target_model_answer"])),
            context=first_present(row, ["context", "passage", "source"]),
            gold_label=label,
            gold_risk_type=response_risk_type(label, RiskType.HALLUCINATION),
            split=split,
            metadata={"raw_source": "halubench"},
        )
        converted.append(sample.model_dump(mode="json"))
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--split", default="unspecified")
    args = parser.parse_args()
    input_path = Path(args.input_file)
    if input_path.suffix.lower() == ".parquet":
        import pandas as pd

        raw_rows = pd.read_parquet(input_path).to_dict(orient="records")
    else:
        raw_rows = list(read_jsonl(args.input_file))
    rows = convert_rows(raw_rows, args.split)
    print(f"wrote {write_jsonl(args.output_file, rows)} rows")


if __name__ == "__main__":
    main()
