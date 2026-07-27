from __future__ import annotations

import hashlib
import json
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


def build_p3_v1(
    aegis_test_path: str | Path = "data/raw/aegis/test.json",
    polyguard_base_ids: int = 1325,
    seed: int = 20260726,
) -> tuple[list[dict], dict]:
    """Build the P3-v1 external public gold panel required by E1 CPU-v5.

    P3-v1 intentionally uses only the official Aegis/Nemotron V2 test split and
    the English/Chinese PolyGuard rows sharing the same base prompt id. It does
    not read project-local merged Aegis train/validation/test manifests.
    """

    aegis = aegis_test_rows(aegis_test_path)
    polyguard = polyguard_rows(base_id_count=polyguard_base_ids, seed=seed)
    rows = attach_semantic_components([*aegis, *polyguard])
    component_count = len({row["semantic_component_id"] for row in rows})
    audit = {
        "rows": len(rows),
        "components": component_count,
        "aegis_rows": len(aegis),
        "aegis_components": len({row["source_prompt_id"] for row in aegis}),
        "polyguard_rows": len(polyguard),
        "polyguard_base_ids": len({row["source_prompt_id"] for row in polyguard}),
        "expected_rows": 4614,
        "expected_components": 3289,
        "aegis_official_test_only": all(row.get("metadata", {}).get("official_split") == "test" for row in aegis),
        "polyguard_cross_language_components": polyguard_language_component_audit(polyguard),
    }
    audit["passed"] = (
        audit["rows"] == audit["expected_rows"]
        and audit["components"] == audit["expected_components"]
        and audit["aegis_rows"] == 1964
        and audit["polyguard_rows"] == polyguard_base_ids * 2
        and audit["aegis_official_test_only"]
        and audit["polyguard_cross_language_components"]["passed"]
    )
    return rows, audit


def aegis_test_rows(path: str | Path = "data/raw/aegis/test.json") -> list[dict]:
    source_path = Path(path)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    raw_rows = data if isinstance(data, list) else data.get("data", [])
    rows: list[dict] = []
    for item in raw_rows:
        label_source = "response_label"
        label = _safe_unsafe_label(item.get("response_label"))
        if label not in {"safe", "unsafe"}:
            label_source = "prompt_label_fallback"
            label = _safe_unsafe_label(item.get("prompt_label"))
        if label not in {"safe", "unsafe"}:
            continue
        query = str(item.get("prompt") or "").strip()
        response_missing = item.get("response") is None or not str(item.get("response") or "").strip()
        answer = str(item.get("response") or "").strip() or "No response is provided in the source record."
        if not query:
            continue
        row_id = str(item.get("id") or _sha1(query + answer)[:24])
        source_id = f"aegis_test_{row_id}"
        rows.append(
            {
                "id": f"p3_aegis_{row_id}",
                "source": "Aegis/Nemotron-V2",
                "language": "English",
                "user_query": query,
                "target_model_answer": answer,
                "pair_fraud_label": label,
                "gold_label": label,
                "gold_risk_type": "fraud_assistance" if label == "unsafe" else "safe",
                "label_provenance": "external_public_official_gold",
                "source_prompt_id": source_id,
                "metadata": {
                    "external_public_gold": True,
                    "source_dataset": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                    "official_split": "test",
                    "source_row_id": row_id,
                    "prompt_label": item.get("prompt_label"),
                    "response_label": item.get("response_label"),
                    "response_missing_in_source": response_missing,
                    "p3_label_source": label_source,
                    "prompt_label_source": item.get("prompt_label_source"),
                    "response_label_source": item.get("response_label_source"),
                    "violated_categories": item.get("violated_categories"),
                },
            }
        )
    return attach_semantic_components(rows)


def polyguard_rows(base_id_count: int = 1325, seed: int = 20260726, max_rows: int | None = None) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test")
    by_base: dict[str, dict[str, dict]] = {}
    for item in dataset:
        language = str(item.get("language") or "")
        if language not in {"English", "Chinese"}:
            continue
        base_id = str(item.get("id"))
        by_base.setdefault(base_id, {})[language] = dict(item)

    eligible: list[tuple[str, str]] = []
    for base_id, versions in by_base.items():
        if {"English", "Chinese"} - set(versions):
            continue
        label = _polyguard_label(versions["English"])
        if label in {"safe", "unsafe"} and _polyguard_label(versions["Chinese"]) == label:
            eligible.append((base_id, label))

    selected_ids = _stratified_ids(eligible, base_id_count, seed)
    rows: list[dict] = []
    for base_id in selected_ids:
        for language in ("English", "Chinese"):
            item = by_base[base_id][language]
            label = _polyguard_label(item)
            query = str(item.get("prompt") or "").strip()
            answer = str(item.get("response") or "").strip()
            if not query or not answer or label not in {"safe", "unsafe"}:
                continue
            source_id = f"polyguard_{base_id}"
            rows.append(
                {
                    "id": f"p3_polyguard_{base_id}_{language.lower()}",
                    "source": "PolyGuardPrompts",
                    "language": language,
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
                        "source_base_id": base_id,
                        "language": language,
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
                return attach_semantic_components(rows)
    return attach_semantic_components(rows)


def polyguard_language_component_audit(rows: list[dict]) -> dict:
    by_base: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        base_id = str(row.get("metadata", {}).get("source_base_id") or row.get("source_prompt_id"))
        by_base.setdefault(base_id, {}).setdefault(str(row.get("language")), set()).add(str(row.get("semantic_component_id") or row.get("source_prompt_id")))
    bad = []
    for base_id, languages in by_base.items():
        components = set().union(*languages.values())
        if not {"English", "Chinese"} <= set(languages) or len(components) != 1:
            bad.append({"base_id": base_id, "languages": sorted(languages), "components": sorted(components)})
    return {"passed": not bad, "bad_count": len(bad), "examples": bad[:20]}


def _component_id(row: dict, query: str, answer: str) -> str:
    source_id = row.get("id") or row.get("source_prompt_id") or row.get("prompt_hash")
    if source_id:
        return f"p3_{row.get('source', 'source')}_{source_id}"
    return f"p3_hash_{_sha1(query + answer)[:24]}"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _polyguard_label(item: dict) -> str | None:
    return _safe_unsafe_label(item.get("response_label") or item.get("response_harm_label"))


def _safe_unsafe_label(value: object) -> str | None:
    text = str(value or "").lower()
    if text in {"safe", "unharmful", "benign"}:
        return "safe"
    if text in {"unsafe", "harmful"}:
        return "unsafe"
    return None


def _stratified_ids(items: list[tuple[str, str]], target_count: int, seed: int) -> list[str]:
    by_label: dict[str, list[str]] = {"safe": [], "unsafe": []}
    for base_id, label in items:
        by_label[label].append(base_id)
    total = sum(len(values) for values in by_label.values())
    if total < target_count:
        raise ValueError(f"PolyGuard has only {total} eligible English/Chinese base ids, need {target_count}")
    selected: list[str] = []
    remaining = target_count
    labels = ["safe", "unsafe"]
    for index, label in enumerate(labels):
        values = sorted(by_label[label], key=lambda value: _sha1(f"{seed}:{label}:{value}"))
        if index == len(labels) - 1:
            take = remaining
        else:
            take = min(len(values), round(target_count * len(values) / total))
            remaining -= take
        selected.extend(values[:take])
    return sorted(selected, key=lambda value: _sha1(f"{seed}:selected:{value}"))[:target_count]
