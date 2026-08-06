"""Error-sample matrix export + rule clustering (guide section 27)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _f(x: dict | None, key: str) -> float:
    if not isinstance(x, dict):
        return 0.0
    try:
        return float(x.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def rule_cluster(row: dict) -> str:
    """Deterministic bucket for one error row (guide 27.3)."""
    f = row.get("agent_fraud_json") or {}
    r = row.get("agent_refusal_json") or {}
    c = row.get("agent_context_json") or {}
    query = str(row.get("query") or "")
    answer = str(row.get("answer") or "")

    if not f and not r and not c:
        return "empty_or_missing_agent_fields"
    if _f(r, "explicit_refusal") >= 0.5 and _f(r, "partial_compliance") < 0.2:
        return "strong_refusal_prefix"
    if len(answer) > 1500:
        return "long_answer"
    if max(_f(c, "quoted_content_only"), _f(c, "analysis_or_education")) >= 0.5:
        return "quotation_or_education"
    if _f(r, "partial_compliance") >= 0.3 or _f(r, "task_completion") >= 0.3:
        return "partial_leakage"
    if _f(f, "trust_facilitation") >= 0.3:
        return "trust_facilitation"
    if _f(f, "risk_downplaying") >= 0.3:
        return "risk_downplaying"
    if _f(f, "direct_actionability") == 0 and _f(r, "should_refuse_probability") == 0:
        return "missing_context"
    return "general_harm_outside_fraud"


def export_error_rows(
    out_dir: Path,
    *,
    name: str,
    rows: list[dict],
    include_query: bool = True,
) -> int:
    """Write one error jsonl (guide 27.1-27.2). Returns number of rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for r in rows:
        f = r.get("agent_fraud_json") or {}
        rec = {
            "sample_id": r.get("id") or r.get("sample_id"),
            "query": r.get("query") if include_query else "[redacted]",
            "answer": r.get("answer", "")[:2000],
            "gold": r.get("gold"),
            "baseline_pred": r.get("baseline_pred"),
            "teacher_pred": r.get("teacher_pred"),
            "fraud_score": r.get("fraud_score"),
            "harm_score": r.get("harm_score"),
            "refusal_score": r.get("refusal_score"),
            "agent_fields": {
                "fraud": {k: f.get(k) for k in (
                    "direct_actionability", "harmful_goal_alignment",
                    "trust_facilitation", "risk_downplaying",
                    "next_step_encouragement",
                )},
                "refusal": {k: (r.get("agent_refusal_json") or {}).get(k) for k in (
                    "should_refuse_probability", "explicit_refusal",
                    "partial_compliance", "task_completion",
                    "operational_detail_leakage",
                )},
            },
            "error_bucket": rule_cluster(r),
        }
        exported.append(rec)
    (out_dir / name).write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in exported),
        encoding="utf-8",
    )
    return len(exported)


def bucket_summary(rows: list[dict]) -> dict:
    from collections import Counter

    return dict(Counter(rule_cluster(r) for r in rows))
