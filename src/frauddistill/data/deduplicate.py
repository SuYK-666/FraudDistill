from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def text_hash(value: object) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def enrich_dedup_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach deterministic exact and near-duplicate cluster identifiers."""
    buckets: dict[str, list[int]] = defaultdict(list)
    output = [dict(row) for row in rows]
    for index, row in enumerate(output):
        query = normalize_text(row.get("user_query"))
        answer = normalize_text(row.get("target_model_answer"))
        row["normalized_query"] = query
        row["normalized_answer"] = answer
        row["query_hash"] = text_hash(query)
        row["answer_hash"] = text_hash(answer)
        row["pair_hash"] = text_hash(query + "\n" + answer)
        # Stable lexical signature catches template variants without relying on row order.
        tokens = sorted(set(re.findall(r"[\w\u4e00-\u9fff]+", query)))[:24]
        buckets[" ".join(tokens)].append(index)
    for signature, indices in buckets.items():
        cluster = "cluster:" + hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
        for index in indices:
            output[index]["near_duplicate_cluster_id"] = cluster
    return output


def build_group_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    case_id = row.get("fraud_case_id") or metadata.get("fraud_case_id") or metadata.get("fraudr1_raw_id")
    prompt_id = row.get("original_prompt_id") or metadata.get("base_prompt_id")
    if case_id:
        return f"case:{case_id}"
    if prompt_id:
        return f"prompt:{prompt_id}"
    return str(row.get("near_duplicate_cluster_id") or "cluster:" + text_hash(row.get("user_query"))[:16])
