from __future__ import annotations

import argparse
from pathlib import Path

from frauddistill.data.schema import FraudDistillSample, Label, RiskType, Source
from frauddistill.utils.io import read_json_records, write_jsonl

FOCUS_CATEGORIES = {
    "phishing": "phishing_scams",
    "phishing scams": "phishing_scams",
    "impersonation": "impersonation",
    "fake job": "fake_job_postings",
    "fake job postings": "fake_job_postings",
}


def infer_category(row: dict) -> str:
    text = " ".join(str(row.get(key, "")) for key in ("fraud_category", "category", "subcategory", "source_file")).lower()
    for needle, category in FOCUS_CATEGORIES.items():
        if needle in text:
            return category
    return "unknown"


def build_unified_rows(input_files: list[str], limit: int = 300) -> list[dict]:
    rows: list[dict] = []
    for input_file in input_files:
        for raw in read_json_records(input_file):
            category = infer_category(raw)
            if category == "unknown":
                continue
            prompt = raw.get("user_query") or raw.get("prompt") or raw.get("generated text") or raw.get("raw_data")
            if not prompt:
                continue
            idx = len(rows)
            label = Label.UNSAFE if idx % 2 == 0 else Label.SAFE
            answer = _unsafe_placeholder(category) if label == Label.UNSAFE else _safe_refusal()
            sample = FraudDistillSample(
                id=f"fraud_focus_{idx:06d}",
                source=Source.FRAUDR1,
                language=str(raw.get("language", "unknown")),
                user_query=str(prompt),
                target_model_answer=answer,
                context=None,
                gold_label=label,
                gold_risk_type=RiskType.FRAUD_ASSISTANCE if label == Label.UNSAFE else RiskType.SAFE,
                split="unspecified",
                metadata={
                    "fraud_category": category,
                    "gold_source": "smoke_placeholder",
                    "source_file": str(input_file),
                },
            )
            rows.append(sample.model_dump(mode="json"))
            if len(rows) >= limit:
                return rows
    return rows


def _unsafe_placeholder(category: str) -> str:
    return (
        f"This is a redacted unsafe placeholder for {category}: step 1 and template content would provide "
        "actionable phishing, impersonation, credential, or fake job assistance. Full harmful details are omitted."
    )


def _safe_refusal() -> str:
    return (
        "I can't help create deceptive messages or collect credentials. I can help with fraud prevention, "
        "recognition signs, reporting channels, and safe awareness training."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_file", default="data/unified/fraud_focus_smoke.jsonl")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    rows = build_unified_rows(args.input_files, args.limit)
    if not rows:
        raise SystemExit("no focus Fraud-R1 rows found")
    print(f"wrote {write_jsonl(Path(args.output_file), rows)} rows")


if __name__ == "__main__":
    main()
