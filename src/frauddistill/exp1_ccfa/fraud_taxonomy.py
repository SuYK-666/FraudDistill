from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_taxonomy(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def risk_type_for_row(row: dict, taxonomy: dict[str, Any]) -> str:
    domain = prompt_risk_domain_for_row(row, taxonomy)
    label = row.get("exp1_label") or row.get("pair_fraud_label") or row.get("gold_label")
    if label == "safe":
        return "safe_fraud_prompt" if domain == "fraud_core" else "safe"
    if domain == "fraud_core":
        return "fraud_primary"
    if domain == "fraud_adjacent":
        return "fraud_adjacent"
    return "general_robustness"


def prompt_risk_domain_for_row(row: dict, taxonomy: dict[str, Any]) -> str:
    source = str(row.get("source") or "")
    metadata = dict(row.get("metadata") or {})
    text_domain = _prompt_text_domain(row)
    if text_domain:
        return text_domain
    if "BeaverTails" in source:
        category = metadata.get("category") or {}
        if any(bool(category.get(key)) for key in taxonomy.get("beavertails", {}).get("fraud_core_categories", [])):
            return "fraud_core"
        if any(bool(category.get(key)) for key in taxonomy.get("beavertails", {}).get("fraud_adjacent_categories", [])):
            return "fraud_adjacent"
        return "general_safety"
    if "Aegis" in source:
        haystack = " ".join(str(value) for value in _flatten(metadata.get("violated_categories"))).lower()
        if any(keyword.lower() in haystack for keyword in taxonomy.get("aegis", {}).get("fraud_core_keywords", [])):
            return "fraud_core"
        if any(keyword.lower() in haystack for keyword in taxonomy.get("aegis", {}).get("fraud_adjacent_keywords", [])):
            return "fraud_adjacent"
        return "general_safety"
    if "Fraud-R1" in source:
        return "fraud_core"
    if "PKU-SafeRLHF" in source:
        category = metadata.get("harm_category_union") or metadata.get("harm_category") or {}
        if any(bool(category.get(key)) for key in taxonomy.get("saferlhf", {}).get("fraud_core_categories", [])):
            return "fraud_core"
        if any(bool(category.get(key)) for key in taxonomy.get("saferlhf", {}).get("fraud_adjacent_categories", [])):
            return "fraud_adjacent"
        return "general_safety"
    if "silver" in str(row.get("label_provenance") or "").lower() or "project" in source.lower():
        return str(taxonomy.get("project_silver", {}).get("default_prompt_risk_domain", "fraud_core"))
    return str(taxonomy.get("unknown", {}).get("default_prompt_risk_domain", "general_safety"))


def _prompt_text_domain(row: dict) -> str | None:
    text = str(row.get("user_query") or "").lower()
    core_terms = (
        "fraud",
        "scam",
        "phishing",
        "credential",
        "password",
        "otp",
        "credit card",
        "bank account",
        "wire transfer",
        "social security",
        "ssn",
        "impersonat",
        "identity theft",
        "steal money",
        "financial institution",
        "market manipulation",
        "privacy",
        "personal data",
        "pii",
    )
    adjacent_terms = ("deceive", "deception", "manipulat", "bypass", "evade", "illegal")
    if any(term in text for term in core_terms):
        return "fraud_core"
    if any(term in text for term in adjacent_terms):
        return "fraud_adjacent"
    return None


def annotate_risk_type(row: dict, taxonomy: dict[str, Any]) -> dict:
    result = dict(row)
    result["response_safety_label"] = result.get("exp1_label") or result.get("pair_fraud_label") or result.get("gold_label")
    result["prompt_risk_domain"] = prompt_risk_domain_for_row(result, taxonomy)
    result["gold_risk_type"] = risk_type_for_row(result, taxonomy)
    metadata = dict(result.get("metadata") or {})
    metadata["prompt_risk_domain"] = result["prompt_risk_domain"]
    metadata["response_safety_label"] = result["response_safety_label"]
    metadata["gold_risk_type"] = result["gold_risk_type"]
    result["metadata"] = metadata
    return result


def fraud_primary_rate(rows: list[dict]) -> float:
    unsafe = [row for row in rows if (row.get("exp1_label") or row.get("gold_label")) == "unsafe"]
    if not unsafe:
        return 0.0
    return sum(1 for row in unsafe if row.get("gold_risk_type") == "fraud_primary") / len(unsafe)


def prompt_domain_rate(rows: list[dict], domain: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("prompt_risk_domain") == domain) / len(rows)


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
