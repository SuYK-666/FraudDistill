from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from frauddistill.data.schema import FraudDistillSample, Label, RiskType, Source
from frauddistill.utils.io import write_jsonl


def convert_files(input_files: list[str], split: str = "unspecified") -> list[dict[str, Any]]:
    output = []
    for input_file in input_files:
        for row in read_aegis(input_file):
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if not prompt or not response:
                continue
            response_label = str(row.get("response_label", "safe")).lower()
            unsafe = response_label == "unsafe"
            sample = FraudDistillSample(
                id=f"aegis_{Path(input_file).stem}_{row.get('id')}",
                source=Source.AEGIS,
                language="English",
                user_query=prompt,
                target_model_answer=response,
                gold_label=Label.UNSAFE if unsafe else Label.SAFE,
                gold_risk_type=_risk_type(row, unsafe),
                split=split,
                metadata={
                    "prompt_label": row.get("prompt_label", "unknown"),
                    "response_label": response_label,
                    "violated_categories": row.get("violated_categories", ""),
                    "prompt_label_source": row.get("prompt_label_source", ""),
                    "response_label_source": row.get("response_label_source", ""),
                    "source_file": input_file,
                },
            )
            output.append(sample.model_dump(mode="json"))
    return output


def build_prompt_rows(input_files: list[str]) -> list[dict[str, Any]]:
    prompts = []
    seen = set()
    for input_file in input_files:
        split = Path(input_file).stem
        for row in read_aegis(input_file):
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            key = row.get("id") or prompt
            if key in seen:
                continue
            seen.add(key)
            prompt_label = str(row.get("prompt_label", "unknown")).lower()
            prompts.append(
                {
                    "id": f"aegis_prompt_{split}_{row.get('id')}",
                    "source": "Aegis",
                    "source_split": split,
                    "language": "English",
                    "fraud_category": "general_safety",
                    "fraud_subcategory": row.get("violated_categories", "unknown"),
                    "prompt_setting": "helpful",
                    "user_query": prompt,
                    "source_prior": "should_refuse" if prompt_label == "unsafe" else "benign_or_contextual",
                    "expected_response_label": "requires_target_generation",
                    "metadata": {
                        "prompt_label": prompt_label,
                        "violated_categories": row.get("violated_categories", ""),
                        "prompt_label_source": row.get("prompt_label_source", ""),
                        "source_file": input_file,
                    },
                }
            )
    return prompts


def read_aegis(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def _risk_type(row: dict[str, Any], unsafe: bool) -> RiskType:
    if not unsafe:
        return RiskType.SAFE
    categories = str(row.get("violated_categories", "")).lower()
    if "criminal" in categories or "privacy" in categories:
        return RiskType.FRAUD_ASSISTANCE
    return RiskType.REFUSAL_FAILURE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--prompts_output_file")
    parser.add_argument("--split", default="unspecified")
    args = parser.parse_args()
    print(f"wrote {write_jsonl(args.output_file, convert_files(args.input_files, args.split))} q+y rows")
    if args.prompts_output_file:
        print(f"wrote {write_jsonl(args.prompts_output_file, build_prompt_rows(args.input_files))} prompt rows")


if __name__ == "__main__":
    main()
