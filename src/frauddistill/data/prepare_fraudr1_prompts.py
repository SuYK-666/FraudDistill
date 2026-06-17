from __future__ import annotations

import argparse
import json
from pathlib import Path

from frauddistill.data.common import first_present
from frauddistill.utils.io import write_jsonl


def load_json_array(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def prepare_rows(input_files: list[str], output_language: str | None = None) -> list[dict]:
    rows = []
    for input_file in input_files:
        for row in load_json_array(input_file):
            language = str(first_present(row, ["language"], output_language or "unknown"))
            prompt = first_present(row, ["generated text", "generated_text", "prompt", "raw_data"])
            rows.append(
                {
                    "id": f"fraudr1_{language.lower()}_{row.get('id', len(rows))}",
                    "language": language,
                    "fraud_category": first_present(row, ["category"], "unknown"),
                    "fraud_subcategory": first_present(row, ["subcategory"], "unknown"),
                    "user_query": prompt,
                    "raw_data": first_present(row, ["raw_data"], ""),
                    "data_type": first_present(row, ["data_type"], "unknown"),
                    "source_file": str(input_file),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--language")
    args = parser.parse_args()
    rows = prepare_rows(args.input_files, args.language)
    print(f"wrote {write_jsonl(args.output_file, rows)} rows")


if __name__ == "__main__":
    main()
