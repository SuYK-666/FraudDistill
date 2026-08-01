from __future__ import annotations

from typing import Any


def build_wrong_q(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        candidates = [
            c
            for c in rows
            if c["response_id"] != row["response_id"]
            and c["language"] == row["language"]
            and c["fraud_category"] == row["fraud_category"]
            and c["actor_role"] == row["actor_role"]
            and c["stage_id"] == row["stage_id"]
            and c["semantic_q_component"] != row["semantic_q_component"]
            and c["source_case_id"] != row["source_case_id"]
            and 0.80 <= len(c["q_private"]) / max(1, len(row["q_private"])) <= 1.25
        ]
        if candidates:
            best = min(candidates, key=lambda c: abs(len(c["q_private"]) - len(row["q_private"])))
            out[row["response_id"]] = {"wrong_response_id": best["response_id"], "wrong_q_private": best["q_private"], "wrong_exact_q_hash": best["exact_q_hash"], "length_ratio": len(best["q_private"]) / max(1, len(row["q_private"]))}
    return out
