from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_wrong_q_map(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_split_cat = defaultdict(list)
    for row in rows:
        by_split_cat[(row["split_role"], row["fraud_category"])].append(row)
    mapping = {}
    misses = 0
    for row in rows:
        candidates = [
            r
            for r in by_split_cat[(row["split_role"], row["fraud_category"])]
            if r["pair_id"] != row["pair_id"] and r["gold"] == row["gold"]
        ]
        if not candidates:
            misses += 1
            continue
        target = sorted(candidates, key=lambda r: r["response_id"])[0]
        mapping[row["response_id"]] = {
            "response_id": row["response_id"],
            "wrong_q_response_id": target["response_id"],
            "wrong_q_private": target["q_private"],
            "split_role": row["split_role"],
            "same_split": target["split_role"] == row["split_role"],
            "same_category": target["fraud_category"] == row["fraud_category"],
        }
    return mapping, {"coverage": len(mapping) / max(1, len(rows)), "matched_rows": len(mapping), "misses": misses, "same_split_required": True}
