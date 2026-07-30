from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any


def build_probe_panel(rows: list[dict[str, Any]], target_n: int = 400, min_balanced_n: int = 240, min_failure_to_run: int = 50) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [r for r in rows if r.get("x_consensus_state") in {"SUCCESS", "FAILURE"}]
    failures = [r for r in valid if r["x_consensus_state"] == "FAILURE"]
    successes = [r for r in valid if r["x_consensus_state"] == "SUCCESS"]
    if len(failures) < min_failure_to_run or not successes:
        return [], {"decision": "PROBE_CAPACITY_STOP", "failure_count": len(failures), "success_count": len(successes)}
    per_class = 200 if len(failures) >= 200 and len(successes) >= 200 else 120 if len(failures) >= 120 and len(successes) >= 120 else min(len(failures), len(successes))
    if per_class * 2 > target_n:
        per_class = target_n // 2
    if per_class * 2 < min_balanced_n and len(failures) >= min_failure_to_run:
        per_class = min(len(failures), len(successes))
    failure_order = sorted(failures, key=lambda r: _rank(r, "risk"))
    success_order = sorted(successes, key=lambda r: _rank(r, "safe"))
    panel = []
    for label, chosen in [(1, failure_order[:per_class]), (0, success_order[:per_class])]:
        for idx, row in enumerate(chosen):
            panel.append(
                {
                    "probe_id": f"probe_{label}_{idx:04d}",
                    "gold": label,
                    "canonical_id": row["canonical_id"],
                    "q_stage_group_id": row["q_stage_group_id"],
                    "q": row["prompt"],
                    "y": row["text"],
                    "language": row["language"],
                    "category": row["category"],
                    "stage_id": row["stage_id"],
                    "stage_name": row["stage_name"],
                    "target_model_family": row["target_model"],
                    "response_id": row["response_id"],
                    "split": "PILOT_DIAGNOSTIC_PROBE_ONLY",
                }
            )
    panel = sorted(panel, key=lambda r: _rank(r, "panel"))
    audit = {
        "decision": "PROBE_PANEL_READY",
        "n": len(panel),
        "positive": sum(r["gold"] for r in panel),
        "negative": len(panel) - sum(r["gold"] for r in panel),
        "q_group_majority_oracle_acc": q_group_majority_oracle_accuracy(panel),
        "language_counts": dict(Counter(r["language"] for r in panel)),
        "stage_counts": dict(Counter(str(r["stage_id"]) for r in panel)),
        "category_counts": dict(Counter(r["category"] for r in panel)),
        "model_counts": dict(Counter(r["target_model_family"] for r in panel)),
    }
    return panel, audit


def q_group_majority_oracle_accuracy(panel: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in panel:
        grouped[row["q_stage_group_id"]].append(int(row["gold"]))
    if not panel:
        return 0.0
    return sum(max(sum(labels), len(labels) - sum(labels)) for labels in grouped.values()) / len(panel)


def _rank(row: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(f"{salt}|{row.get('q_stage_group_id')}|{row.get('response_id')}".encode("utf-8")).hexdigest()
