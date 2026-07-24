from __future__ import annotations

from collections import defaultdict
from typing import Any

from .deduplicate import normalize_text


def build_context_pairs(rows: list[dict[str, Any]], max_pairs: int = 5000) -> list[dict[str, Any]]:
    """Automatically surface same/near answer contexts with differing gold behavior."""
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        answer = normalize_text(row.get("target_model_answer"))
        if answer:
            by_answer[answer[:300]].append(row)
    pairs = []
    for answer_key, bucket in by_answer.items():
        unsafe = next((row for row in bucket if row.get("gold_label") == "unsafe"), None)
        safe = next((row for row in bucket if row.get("gold_label") == "safe"), None)
        if unsafe and safe:
            pairs.append({"unsafe_id": unsafe["id"], "safe_id": safe["id"], "answer_signature": answer_key, "source": "exact_answer_context"})
            if len(pairs) >= max_pairs:
                break
    return pairs
