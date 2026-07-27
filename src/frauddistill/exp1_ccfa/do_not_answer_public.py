from __future__ import annotations

import csv
from pathlib import Path

from frauddistill.exp1_ccfa.semantic_components import attach_semantic_components
from frauddistill.utils.io import read_jsonl


def do_not_answer_rows(root: Path) -> list[dict]:
    prepared = root / "data" / "prepared" / "full" / "evaluation_qy" / "do_not_answer_qy.jsonl"
    if prepared.exists():
        rows = []
        for idx, row in enumerate(read_jsonl(prepared)):
            item = dict(row)
            item.setdefault("id", f"dna_qy_{idx}")
            item.setdefault("source", "Do-Not-Answer")
            item.setdefault("metadata", {})
            item["metadata"] = {**dict(item.get("metadata") or {}), "official_split": "response_assessment", "license_note": "public research dataset"}
            item.setdefault("label_provenance", "public_official_response_assessment")
            if "pair_fraud_label" in item and "exp1_label" not in item:
                item["exp1_label"] = item["pair_fraud_label"]
            if "gold_label" in item and "exp1_label" not in item:
                item["exp1_label"] = item["gold_label"]
            rows.append(item)
        return attach_semantic_components(rows)
    csv_path = root / "data" / "raw" / "do_not_answer" / "data_en.csv"
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            prompt = row.get("question") or row.get("prompt") or row.get("instruction") or ""
            answer = row.get("response") or row.get("answer") or ""
            if not prompt or not answer:
                continue
            label = str(row.get("label") or row.get("risk_label") or "").lower()
            exp_label = "unsafe" if "unsafe" in label or "do not answer" in label else "safe"
            rows.append(
                {
                    "id": f"dna_raw_{idx}",
                    "source": "Do-Not-Answer",
                    "source_prompt_id": f"dna_raw_prompt_{idx}",
                    "user_query": prompt,
                    "target_model_answer": answer,
                    "pair_fraud_label": exp_label,
                    "exp1_label": exp_label,
                    "gold_label": exp_label,
                    "label_provenance": "public_official_response_assessment",
                    "metadata": {"official_split": "response_assessment", "license_note": "public research dataset"},
                }
            )
    return attach_semantic_components(rows)
