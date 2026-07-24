from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from frauddistill.utils.io import read_json_records, write_jsonl

CATEGORY_MAP = {
    "phishing": "phishing_scams",
    "impersonation": "impersonation",
    "fake job posting": "fake_job_postings",
    "fraudulent service": "fraudulent_services",
    "network friendship": "online_relationships",
}


def build_exp1_rows(input_files: list[str], include_categories: set[str] | None = None) -> list[dict]:
    """Build Fraud-R1 prompt/context assets, not q-y gold pairs.

    Fraud-R1 `generated text` is source scenario material. It is not a
    target-model answer, so this adapter leaves `target_model_answer` empty and
    requires a later target generation pass before q-y evaluation.
    """
    rows: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for input_file in input_files:
        variant = _variant_from_path(input_file)
        for raw in read_json_records(input_file):
            category = CATEGORY_MAP.get(str(raw.get("category", "")).strip().lower())
            if not category or (include_categories and category not in include_categories):
                continue
            query = str(raw.get("raw_data") or "").strip()
            generated_text = str(raw.get("generated text") or raw.get("generated_text") or "").strip()
            if not query or not generated_text:
                continue
            language = str(raw.get("language", "unknown"))
            raw_id = str(raw.get("id", _short_hash(query + generated_text)))
            key = (variant, category, language, raw_id)
            if key in seen:
                continue
            seen.add(key)
            base_id = f"fraudr1_{variant}_{category}_{language.lower()}_{raw_id}"
            rows.append(
                {
                    "id": base_id,
                    "source": "Fraud-R1",
                    "language": language,
                    "user_query": _prompt_from_raw(query, generated_text, language),
                    "context": generated_text,
                    "target_model_answer": "",
                    "prompt_risk_label": "unsafe",
                    "prompt_risk_type": "fraud_request",
                    "response_harm_label": None,
                    "pair_fraud_label": None,
                    "label_provenance": "unlabeled_prompt_requires_target_generation",
                    "gold_label": None,
                    "gold_risk_type": None,
                    "split": "unspecified",
                    "metadata": {
                        "fraud_category": category,
                        "fraud_subcategory": raw.get("subcategory", "unknown"),
                        "fraudr1_variant": variant,
                        "fraudr1_raw_id": raw_id,
                        "raw_data_sha256": _sha256(query),
                        "generated_text_sha256": _sha256(generated_text),
                        "generated_text_role": "prompt_context_not_target_answer",
                        "source_file": str(input_file),
                    },
                }
            )
    return rows


def _prompt_from_raw(raw_data: str, generated_text: str, language: str) -> str:
    if str(language).lower().startswith("chinese"):
        return f"{raw_data}\n\n参考场景材料：{generated_text}"
    return f"{raw_data}\n\nReference scenario material: {generated_text}"


def _variant_from_path(path: str) -> str:
    text = str(Path(path)).lower()
    if "levelup" in text:
        return "levelup"
    if "base" in text:
        return "base"
    return "unknown"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_file", default="data/unified/exp1_fraudr1_prompt_assets.jsonl")
    parser.add_argument("--categories", nargs="*", default=sorted(CATEGORY_MAP.values()))
    args = parser.parse_args()
    rows = build_exp1_rows(args.input_files, set(args.categories))
    print(f"wrote {write_jsonl(args.output_file, rows)} rows")


if __name__ == "__main__":
    main()
