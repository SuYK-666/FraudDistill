from __future__ import annotations

import argparse
import json

from frauddistill.data.common import first_present, map_binary_label, response_risk_type
from frauddistill.data.schema import FraudDistillSample, RiskType, Source
from frauddistill.utils.io import read_jsonl, write_jsonl


def _stringify_context(value):
    if isinstance(value, str) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def _label_from_ragtruth(row: dict) -> str:
    labels = row.get("labels", [])
    if isinstance(labels, str):
        labels = json.loads(labels)
    return "unsafe" if labels else "safe"


def convert_rows(rows: list[dict], split: str = "unspecified") -> list[dict]:
    output = []
    for idx, row in enumerate(rows):
        label_value = first_present(row, ["label", "gold_label", "is_hallucinated", "response_label"], None)
        label = map_binary_label(label_value if label_value is not None else _label_from_ragtruth(row))
        sample = FraudDistillSample(
            id=str(first_present(row, ["id", "sample_id", "response_id"], f"ragtruth_{idx:06d}")),
            source=Source.RAGTRUTH,
            language=str(first_present(row, ["language"], "en")),
            user_query=str(first_present(row, ["question", "query", "prompt", "user_query"])),
            target_model_answer=str(first_present(row, ["answer", "response", "target_model_answer"])),
            context=_stringify_context(first_present(row, ["context", "source", "source_info", "documents"])),
            gold_label=label,
            gold_risk_type=response_risk_type(label, RiskType.HALLUCINATION),
            split=split,
            metadata={"spans": first_present(row, ["spans", "hallucination_spans", "labels"], [])},
        )
        output.append(sample.model_dump(mode="json"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--source_info_file")
    parser.add_argument("--split", default="unspecified")
    args = parser.parse_args()
    rows = list(read_jsonl(args.input_file))
    if args.source_info_file:
        source_by_id = {str(row["source_id"]): row for row in read_jsonl(args.source_info_file)}
        joined = []
        for row in rows:
            source = source_by_id.get(str(row.get("source_id")), {})
            joined.append(
                {
                    **row,
                    "prompt": source.get("prompt"),
                    "context": source.get("source_info"),
                    "source_dataset": source.get("source"),
                    "task_type": source.get("task_type"),
                }
            )
        rows = joined
    print(f"wrote {write_jsonl(args.output_file, convert_rows(rows, args.split))} rows")


if __name__ == "__main__":
    main()
