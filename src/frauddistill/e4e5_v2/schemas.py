# -*- coding: utf-8 -*-
"""E4/E5 v2 shared schemas, hashing utilities and label policy."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

LABEL_TO_ID = {"safe": 0, "fraud_assistance": 1, "refusal_failure": 2, "over_refusal": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# canonical binary mapping used by the Final Student (guide 2.2)
UNSAFE_TYPES = {"fraud_assistance", "refusal_failure", "over_refusal"}


def norm_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text)).lower()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def q_hash(q: str) -> str:
    return sha256_hex(norm_text(q))


def y_hash(y: str) -> str:
    return sha256_hex(norm_text(y))


def qy_hash(q: str, y: str) -> str:
    return sha256_hex(f"{norm_text(q)}\n{norm_text(y)}")


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def family_key_of(row: dict, default: str = "") -> str:
    """Unified family key: template_family_id > group_id > q-norm prefix."""
    for k in ("template_family_id", "family_id", "family", "group_id", "group"):
        v = row.get(k)
        if v:
            return str(v)
    q = str(row.get("user_query") or row.get("query") or "")
    if q:
        return "q_" + norm_text(q)[:48]
    return default


def template_key_of(row: dict) -> str:
    for k in ("template_id", "group_id", "group"):
        v = row.get(k)
        if v:
            return str(v)
    return family_key_of(row)


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def canonical_row(id_: str, q: str, y: str, gold_label: str, gold_type: str,
                  source: str, language: str, primary_shift: str, family_id: str,
                  gold_source: str, target_model: str = "", context: Any = None,
                  **extra) -> dict:
    return {
        "id": id_,
        "user_query": q,
        "target_model_answer": y,
        "context": context,
        "gold_label": gold_label,
        "gold_type": gold_type,
        "gold_binary": 1 if gold_label == "unsafe" else 0,
        "source": source,
        "source_version": extra.pop("source_version", ""),
        "language": language,
        "target_model": target_model,
        "primary_shift": primary_shift,
        "fraud_category": extra.pop("fraud_category", ""),
        "family_id": family_id,
        "pair_id": extra.pop("pair_id", None),
        "template_id": extra.pop("template_id", family_id),
        "gold_source": gold_source,
        "license": extra.pop("license", ""),
        "exposure_level": "L3",
        "qy_hash": qy_hash(q, y),
        "q_hash": q_hash(q),
        "y_hash": y_hash(y),
        "metadata": extra,
    }


def gold_binary_of(label: str) -> int:
    return 1 if str(label) == "unsafe" else 0


def risk_type_of(gold_label: str, gold_type: str) -> str:
    if str(gold_label) == "safe":
        return "safe"
    return str(gold_type) if str(gold_type) in LABEL_TO_ID else "fraud_assistance"
