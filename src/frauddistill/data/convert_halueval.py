from __future__ import annotations

import argparse
from pathlib import Path

from frauddistill.data.common import first_present, map_binary_label, response_risk_type
from frauddistill.data.schema import FraudDistillSample, RiskType, Source
from frauddistill.utils.io import read_json_records, write_jsonl


def convert_rows(rows: list[dict], split: str = "unspecified", id_prefix: str = "halueval") -> list[dict]:
    output = []
    for idx, row in enumerate(rows):
        query = str(first_present(row, ["question", "query", "user_query", "prompt", "dialogue_history"]))
        context = first_present(row, ["context", "knowledge", "passage", "document"])
        if "right_answer" in row and "hallucinated_answer" in row:
            answer_pairs = [("right", row["right_answer"], "safe"), ("hallucinated", row["hallucinated_answer"], "unsafe")]
        elif "right_response" in row and "hallucinated_response" in row:
            answer_pairs = [("right", row["right_response"], "safe"), ("hallucinated", row["hallucinated_response"], "unsafe")]
        elif "right_summary" in row and "hallucinated_summary" in row:
            query = "Summarize the provided document."
            answer_pairs = [("right", row["right_summary"], "safe"), ("hallucinated", row["hallucinated_summary"], "unsafe")]
        else:
            label_value = first_present(row, ["label", "gold_label", "hallucination", "is_hallucinated"])
            answer_pairs = [("answer", first_present(row, ["answer", "response", "target_model_answer", "chatgpt_response"]), map_binary_label(label_value).value)]
        for suffix, answer, label_text in answer_pairs:
            label = map_binary_label(label_text)
            sample = FraudDistillSample(
                id=f"{id_prefix}_{idx:06d}_{first_present(row, ['id', 'ID', 'sample_id'], 'row')}_{suffix}",
                source=Source.HALUEVAL,
                language=str(first_present(row, ["language"], "en")),
                user_query=query,
                target_model_answer=str(answer),
                context=context,
                gold_label=label,
                gold_risk_type=response_risk_type(label, RiskType.HALLUCINATION),
                split=split,
                metadata={"task": first_present(row, ["task"], "unknown")},
            )
            output.append(sample.model_dump(mode="json"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--id_prefix")
    args = parser.parse_args()
    id_prefix = args.id_prefix or Path(args.input_file).stem
    print(f"wrote {write_jsonl(args.output_file, convert_rows(read_json_records(args.input_file), args.split, id_prefix))} rows")


if __name__ == "__main__":
    main()
