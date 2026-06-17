from __future__ import annotations

import argparse

from frauddistill.data.common import first_present, map_binary_label, response_risk_type
from frauddistill.data.schema import FraudDistillSample, RiskType, Source
from frauddistill.utils.io import read_json_records, write_jsonl


def _label_from_segments(row: dict) -> str:
    segments = first_present(row, ["segments", "segment_labels"], [])
    if not segments:
        labels = first_present(row, ["labels"], [])
        if isinstance(labels, list) and labels:
            return "unsafe" if any(label is False or str(label).lower() == "false" for label in labels) else "safe"
    if isinstance(segments, list) and segments:
        has_error = any(str(seg.get("label", seg.get("factuality", ""))).lower() in {"false", "unsupported", "incorrect", "0"} for seg in segments if isinstance(seg, dict))
        return "unsafe" if has_error else "safe"
    return first_present(row, ["label", "gold_label", "is_factual"], "safe")


def convert_rows(rows: list[dict], split: str = "unspecified") -> list[dict]:
    output = []
    for idx, row in enumerate(rows):
        label = map_binary_label(_label_from_segments(row))
        sample = FraudDistillSample(
            id=str(first_present(row, ["id", "sample_id"], f"felm_{idx:06d}")),
            source=Source.FELM,
            language=str(first_present(row, ["language"], "en")),
            user_query=str(first_present(row, ["prompt", "question", "query", "user_query"])),
            target_model_answer=str(first_present(row, ["response", "answer", "target_model_answer"])),
            context=first_present(row, ["context", "reference", "source"]),
            gold_label=label,
            gold_risk_type=response_risk_type(label, RiskType.HALLUCINATION),
            split=split,
            metadata={
                "segments": first_present(row, ["segments", "segment_labels", "segmented_response"], []),
                "segment_labels": first_present(row, ["labels"], []),
                "comments": first_present(row, ["comment"], []),
                "refs": first_present(row, ["ref"], []),
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
    print(f"wrote {write_jsonl(args.output_file, convert_rows(read_json_records(args.input_file), args.split))} rows")


if __name__ == "__main__":
    main()
