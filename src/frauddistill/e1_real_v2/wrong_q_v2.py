from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def build_wrong_q(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[(row.get("split_role"), row.get("language"), row.get("fraud_category"))].append(row)
    reuse = Counter()
    mapping = {}
    for row in rows:
        bucket = by_bucket[(row.get("split_role"), row.get("language"), row.get("fraud_category"))]
        candidates = [
            r
            for r in bucket
            if r["canonical_case_id"] != row["canonical_case_id"]
            and r["normalized_q_hash"] != row["normalized_q_hash"]
            and reuse[r["response_id"]] < 2
        ]
        if not candidates:
            continue
        target = sorted(candidates, key=lambda r: (reuse[r["response_id"]], abs(len(r["q_private"]) - len(row["q_private"]))))[0]
        reuse[target["response_id"]] += 1
        mapping[row["response_id"]] = {"response_id": row["response_id"], "wrong_q_response_id": target["response_id"], "wrong_q_private": target["q_private"]}
    audit = {
        "coverage": len(mapping) / max(1, len(rows)),
        "max_reuse": max(reuse.values()) if reuse else 0,
        "self_map": sum(1 for k, v in mapping.items() if k == v["wrong_q_response_id"]),
        "uses_gold": False,
    }
    return mapping, audit
