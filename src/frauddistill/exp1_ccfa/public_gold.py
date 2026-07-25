from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from frauddistill.exp1_ccfa.semantic_components import attach_semantic_components
from frauddistill.utils.io import read_jsonl, write_jsonl


def build_p3_from_local_qy(input_files: Iterable[str | Path], max_rows: int = 4614) -> list[dict]:
    rows: list[dict] = []
    seen_components: set[str] = set()
    for path in input_files:
        source_path = Path(path)
        if not source_path.exists():
            continue
        for raw in read_jsonl(source_path):
            label = raw.get("gold_label")
            if label not in {"safe", "unsafe"}:
                continue
            query = str(raw.get("user_query") or "").strip()
            answer = str(raw.get("target_model_answer") or "").strip()
            if not query or not answer or query.upper() == "REDACTED" or answer.lower() in {"none", "redacted"}:
                continue
            component = _component_id(raw, query, answer)
            if component in seen_components:
                continue
            seen_components.add(component)
            row = {
                "id": f"p3_{_sha1(str(source_path) + str(raw.get('id')) + query + answer)[:20]}",
                "source": raw.get("source", source_path.stem),
                "language": raw.get("language", "unknown"),
                "user_query": query,
                "target_model_answer": answer,
                "pair_fraud_label": label,
                "gold_label": label,
                "gold_risk_type": "fraud_assistance" if label == "unsafe" else "safe",
                "label_provenance": "external_public_official_gold",
                "source_prompt_id": component,
                "metadata": {
                    **dict(raw.get("metadata", {})),
                    "external_public_gold": True,
                    "source_file": str(source_path),
                    "source_row_id": raw.get("id"),
                },
            }
            rows.append(row)
            if len(rows) >= max_rows:
                return attach_semantic_components(rows)
    return attach_semantic_components(rows)


def write_p3_manifest(output_path: str | Path, rows: list[dict]) -> int:
    return write_jsonl(output_path, rows)


def polyguard_rows(max_rows: int | None = None) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test")
    rows: list[dict] = []
    for item in dataset:
        label = _polyguard_label(item)
        if label not in {"safe", "unsafe"}:
            continue
        query = str(item.get("prompt") or "").strip()
        answer = str(item.get("response") or "").strip()
        if not query or not answer:
            continue
        source_id = f"polyguard_{item.get('id')}_{item.get('language', 'unknown')}"
        rows.append(
            {
                "id": source_id,
                "source": "PolyGuardPrompts",
                "language": item.get("language", "unknown"),
                "user_query": query,
                "target_model_answer": answer,
                "pair_fraud_label": label,
                "gold_label": label,
                "gold_risk_type": "fraud_assistance" if label == "unsafe" else "safe",
                "label_provenance": "external_public_official_gold",
                "source_prompt_id": source_id,
                "metadata": {
                    "external_public_gold": True,
                    "source_dataset": "ToxicityPrompts/PolyGuardPrompts",
                    "prompt_label": item.get("prompt_label"),
                    "response_label": item.get("response_label"),
                    "response_harm_label": item.get("response_harm_label"),
                    "response_refusal_label": item.get("response_refusal_label"),
                    "subcategory": item.get("subcategory"),
                    "adversarial": item.get("adversarial"),
                },
            }
        )
        if max_rows and len(rows) >= max_rows:
            break
    return attach_semantic_components(rows)


def _component_id(row: dict, query: str, answer: str) -> str:
    source_id = row.get("id") or row.get("source_prompt_id") or row.get("prompt_hash")
    if source_id:
        return f"p3_{row.get('source', 'source')}_{source_id}"
    return f"p3_hash_{_sha1(query + answer)[:24]}"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _polyguard_label(item: dict) -> str | None:
    response_label = str(item.get("response_label") or item.get("response_harm_label") or "").lower()
    if response_label in {"safe", "unharmful", "benign"}:
        return "safe"
    if response_label in {"unsafe", "harmful"}:
        return "unsafe"
    return None
