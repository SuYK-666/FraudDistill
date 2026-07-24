from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from frauddistill.data.schema import FraudDistillSample, Label, RiskType, Source
from frauddistill.utils.io import write_jsonl


MODEL_PREFIXES = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]


def convert_rows(rows: list[dict[str, Any]], split: str = "unspecified") -> list[dict[str, Any]]:
    output = []
    for idx, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        for model_name in MODEL_PREFIXES:
            response = str(row.get(f"{model_name}_response", "")).strip()
            if not response or response.lower() == "nan":
                continue
            harmful = int(row.get(f"{model_name}_harmful", 0) or 0)
            action = int(row.get(f"{model_name}_action", 0) or 0)
            unsafe = harmful == 1 or action == 1
            sample = FraudDistillSample(
                id=f"dna_{row.get('id', idx)}_{_safe_model_id(model_name)}",
                source=Source.DO_NOT_ANSWER,
                language="English",
                user_query=question,
                target_model_answer=response,
                gold_label=Label.UNSAFE if unsafe else Label.SAFE,
                gold_risk_type=RiskType.REFUSAL_FAILURE if unsafe else RiskType.SAFE,
                split=split,
                metadata={
                    "risk_area": row.get("risk_area", "unknown"),
                    "types_of_harm": row.get("types_of_harm", "unknown"),
                    "specific_harms": row.get("specific_harms", "unknown"),
                    "target_model": model_name,
                    "harmful": harmful,
                    "action": action,
                    "gold_source": "do_not_answer_model_response_labels",
                },
            )
            output.append(sample.model_dump(mode="json"))
    return output


def build_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts = []
    for idx, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        prompts.append(
            {
                "id": f"dna_prompt_{row.get('id', idx)}",
                "source": "Do-Not-Answer",
                "source_split": "full",
                "language": "English",
                "fraud_category": "should_refuse",
                "fraud_subcategory": row.get("types_of_harm", "unknown"),
                "prompt_setting": "helpful",
                "user_query": question,
                "source_prior": "should_refuse",
                "expected_response_label": "requires_target_generation",
                "metadata": {
                    "risk_area": row.get("risk_area", "unknown"),
                    "types_of_harm": row.get("types_of_harm", "unknown"),
                    "specific_harms": row.get("specific_harms", "unknown"),
                },
            }
        )
    return prompts


def read_do_not_answer(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return pd.read_parquet(target).to_dict(orient="records")
    return pd.read_csv(target).to_dict(orient="records")


def _safe_model_id(value: str) -> str:
    return value.lower().replace("/", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/raw/do_not_answer/data/train-00000-of-00001-6ba0076b818accff.parquet")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--prompts_output_file")
    parser.add_argument("--split", default="unspecified")
    args = parser.parse_args()
    rows = read_do_not_answer(args.input_file)
    print(f"wrote {write_jsonl(args.output_file, convert_rows(rows, args.split))} q+y rows")
    if args.prompts_output_file:
        print(f"wrote {write_jsonl(args.prompts_output_file, build_prompt_rows(rows))} prompt rows")


if __name__ == "__main__":
    main()
