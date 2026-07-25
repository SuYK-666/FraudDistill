from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from frauddistill.utils.io import write_jsonl


REQUIRED_TEXT_FIELDS = ("user_query", "target_model_answer")
LABEL_FIELD_CANDIDATES = ("pair_fraud_label", "gold_label")
EXPLICIT_CONTEXT_TOKENS = (
    "SAFE CONTEXT",
    "UNSAFE INTENT",
    "BENIGN CONTEXT",
    "MALICIOUS INTENT",
    "安全上下文",
    "危险意图",
)


def normalize_label(row: dict) -> str:
    for field in LABEL_FIELD_CANDIDATES:
        value = row.get(field)
        if value in {"safe", "unsafe"}:
            return str(value)
        if value == "fraud_assistance":
            return "unsafe"
    raise ValueError(f"row {row.get('id', '<missing id>')} has no valid safe/unsafe label")


def component_id_for_row(row: dict) -> str:
    raw = (
        row.get("source_prompt_id")
        or row.get("pair_family_id")
        or row.get("metadata", {}).get("source_prompt_id")
        or row.get("prompt_hash")
    )
    if raw:
        return f"sc_{_slug(str(raw))[:80]}_{_sha1(str(raw))[:10]}"
    query = str(row.get("user_query") or "")
    return f"sc_query_{_sha1(query)[:16]}"


def attach_semantic_components(rows: Iterable[dict]) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        checked = dict(row)
        for field in REQUIRED_TEXT_FIELDS:
            if checked.get(field) is None:
                raise ValueError(f"row {checked.get('id', '<missing id>')} has null {field}")
        checked["exp1_label"] = normalize_label(checked)
        checked["semantic_component_id"] = component_id_for_row(checked)
        output.append(checked)
    return output


def split_by_component(
    rows: list[dict],
    seed: int,
    train_ratio: float = 0.7,
    model_dev_ratio: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    components = sorted({str(row["semantic_component_id"]) for row in rows})
    shuffled = sorted(components, key=lambda value: _sha1(f"{seed}:{value}"))
    train_cut = int(len(shuffled) * train_ratio)
    model_dev_cut = train_cut + int(len(shuffled) * model_dev_ratio)
    train_components = set(shuffled[:train_cut])
    model_dev_components = set(shuffled[train_cut:model_dev_cut])
    threshold_components = set(shuffled[model_dev_cut:])
    train = [row for row in rows if row["semantic_component_id"] in train_components]
    model_dev = [row for row in rows if row["semantic_component_id"] in model_dev_components]
    threshold_dev = [row for row in rows if row["semantic_component_id"] in threshold_components]
    audit = leakage_audit({"train": train, "model_dev": model_dev, "threshold_dev": threshold_dev})
    return train, model_dev, threshold_dev, audit


def leakage_audit(splits: dict[str, list[dict]]) -> dict:
    memberships: dict[str, set[str]] = {}
    for split_name, rows in splits.items():
        for row in rows:
            memberships.setdefault(str(row["semantic_component_id"]), set()).add(split_name)
    leaked = {component: sorted(names) for component, names in memberships.items() if len(names) > 1}
    return {
        "passed": not leaked,
        "leaked_components": leaked,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "component_counts": {name: len({row["semantic_component_id"] for row in rows}) for name, rows in splits.items()},
    }


def explicit_label_token_audit(rows: Iterable[dict]) -> dict:
    hits: list[dict] = []
    for row in rows:
        text = f"{row.get('user_query', '')}\n{row.get('target_model_answer', '')}"
        for token in EXPLICIT_CONTEXT_TOKENS:
            if token in text:
                hits.append({"id": row.get("id"), "token": token})
    return {"passed": not hits, "hits": hits[:200], "hit_count": len(hits)}


def write_component_manifest(path: str | Path, rows: list[dict]) -> int:
    counter = Counter(str(row["semantic_component_id"]) for row in rows)
    manifest = [{"semantic_component_id": cid, "row_count": count} for cid, count in sorted(counter.items())]
    return write_jsonl(path, manifest)


def write_schema(path: str | Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    schema = {
        "required_fields": ["id", "user_query", "target_model_answer", "semantic_component_id", "exp1_label"],
        "observed_fields": fields,
        "label_policy": "missing labels raise ValueError; public gold must not be overwritten by guard labels",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "component"

