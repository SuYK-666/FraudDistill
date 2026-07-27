from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_taxonomy(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def risk_type_for_row(row: dict, taxonomy: dict[str, Any]) -> str:
    label = row.get("exp1_label") or row.get("pair_fraud_label") or row.get("gold_label")
    if label == "safe":
        return "safe"
    source = str(row.get("source") or "")
    metadata = dict(row.get("metadata") or {})
    if "BeaverTails" in source:
        category = metadata.get("category") or {}
        fraud_keys = set(taxonomy.get("beavertails", {}).get("fraud_primary_categories", []))
        if any(bool(category.get(key)) for key in fraud_keys):
            return "fraud_primary"
        return "general_robustness"
    if "Aegis" in source:
        haystack = " ".join(str(value) for value in _flatten(metadata.get("violated_categories"))).lower()
        if any(keyword.lower() in haystack for keyword in taxonomy.get("aegis", {}).get("fraud_primary_keywords", [])):
            return "fraud_primary"
        return "general_robustness"
    if "Fraud-R1" in source:
        return "fraud_primary"
    if "silver" in str(row.get("label_provenance") or "").lower() or "project" in source.lower():
        return str(taxonomy.get("project_silver", {}).get("default_risk_type", "fraud_primary"))
    return str(taxonomy.get("unknown", {}).get("default_risk_type", "general_robustness"))


def annotate_risk_type(row: dict, taxonomy: dict[str, Any]) -> dict:
    result = dict(row)
    result["gold_risk_type"] = risk_type_for_row(result, taxonomy)
    metadata = dict(result.get("metadata") or {})
    metadata["gold_risk_type"] = result["gold_risk_type"]
    result["metadata"] = metadata
    return result


def fraud_primary_rate(rows: list[dict]) -> float:
    unsafe = [row for row in rows if (row.get("exp1_label") or row.get("gold_label")) == "unsafe"]
    if not unsafe:
        return 0.0
    return sum(1 for row in unsafe if row.get("gold_risk_type") == "fraud_primary") / len(unsafe)


def _flatten(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: list[object] = []
        for key, item in value.items():
            out.append(key)
            out.extend(_flatten(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]
