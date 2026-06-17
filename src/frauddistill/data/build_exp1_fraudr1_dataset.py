from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from frauddistill.data.schema import FraudDistillSample, Label, RiskType, Source
from frauddistill.utils.io import read_json_records, write_jsonl

CATEGORY_MAP = {
    "phishing": "phishing_scams",
    "impersonation": "impersonation",
    "fake job posting": "fake_job_postings",
    "fraudulent service": "fraudulent_services",
    "network friendship": "online_relationships",
}


def build_exp1_rows(input_files: list[str], include_categories: set[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for input_file in input_files:
        variant = _variant_from_path(input_file)
        for raw in read_json_records(input_file):
            category = CATEGORY_MAP.get(str(raw.get("category", "")).strip().lower())
            if not category or (include_categories and category not in include_categories):
                continue
            query = str(raw.get("raw_data") or "").strip()
            unsafe_answer = str(raw.get("generated text") or raw.get("generated_text") or "").strip()
            if not query or not unsafe_answer:
                continue
            language = str(raw.get("language", "unknown"))
            raw_id = str(raw.get("id", _short_hash(query + unsafe_answer)))
            key = (variant, category, language, raw_id)
            if key in seen:
                continue
            seen.add(key)
            base_id = f"fraudr1_{variant}_{category}_{language.lower()}_{raw_id}"
            meta = {
                "fraud_category": category,
                "fraud_subcategory": raw.get("subcategory", "unknown"),
                "fraudr1_variant": variant,
                "fraudr1_raw_id": raw_id,
                "gold_source": "fraudr1_generated_text_protocol",
                "source_file": str(input_file),
            }
            rows.append(
                _sample(
                    sample_id=f"{base_id}_unsafe",
                    language=language,
                    query=query,
                    answer=unsafe_answer,
                    label=Label.UNSAFE,
                    risk_type=RiskType.FRAUD_ASSISTANCE,
                    metadata={**meta, "answer_role": "fraudr1_generated_unsafe"},
                )
            )
            rows.append(
                _sample(
                    sample_id=f"{base_id}_safe",
                    language=language,
                    query=query,
                    answer=_safe_answer(language),
                    label=Label.SAFE,
                    risk_type=RiskType.SAFE,
                    metadata={**meta, "answer_role": "safe_refusal_control"},
                )
            )
    return rows


def _sample(
    sample_id: str,
    language: str,
    query: str,
    answer: str,
    label: Label,
    risk_type: RiskType,
    metadata: dict,
) -> dict:
    return FraudDistillSample(
        id=sample_id,
        source=Source.FRAUDR1,
        language=language,
        user_query=query,
        target_model_answer=answer,
        context=None,
        gold_label=label,
        gold_risk_type=risk_type,
        split="unspecified",
        metadata=metadata,
    ).model_dump(mode="json")


def _safe_answer(language: str) -> str:
    if language.lower().startswith("chinese"):
        return "我不能帮助撰写或优化欺诈、钓鱼、冒充身份或虚假招聘内容。可以提供反诈识别要点、风险提醒、举报渠道和安全教育建议。"
    return (
        "I can't help create or optimize fraud, phishing, impersonation, or fake job content. "
        "I can help with fraud-awareness signs, safety guidance, reporting channels, and prevention education."
    )


def _variant_from_path(path: str) -> str:
    text = str(Path(path)).lower()
    if "levelup" in text:
        return "levelup"
    if "base" in text:
        return "base"
    return "unknown"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_file", default="data/unified/exp1_fraudr1_full.jsonl")
    parser.add_argument("--categories", nargs="*", default=["phishing_scams", "impersonation", "fake_job_postings"])
    args = parser.parse_args()
    rows = build_exp1_rows(args.input_files, set(args.categories))
    print(f"wrote {write_jsonl(args.output_file, rows)} rows")


if __name__ == "__main__":
    main()
