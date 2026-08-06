"""Canonical evidence table + feature matrix (guide section 23).

Flattens agent_fraud_json / agent_refusal_json / agent_context_json into a
fixed, versioned feature vector with interaction terms for the adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BASE_FEATURES = [
    "fraud.direct_actionability",
    "fraud.harmful_goal_alignment",
    "fraud.trust_facilitation",
    "fraud.risk_downplaying",
    "fraud.next_step_encouragement",
    "fraud.fraud_cost_reduction",
    "refusal.should_refuse_probability",
    "refusal.explicit_refusal",
    "refusal.safe_redirection",
    "refusal.risk_warning",
    "refusal.partial_compliance",
    "refusal.task_completion",
    "refusal.operational_detail_leakage",
    "refusal.leakage_before_refusal",
    "refusal.leakage_after_refusal",
    "refusal.prompt_is_benign",
    "refusal.unnecessary_refusal",
    "refusal.answer_delivers_safe_utility",
    "context.analysis_or_education",
    "context.quoted_content_only",
    "context.active_endorsement",
    "context.query_answer_alignment",
]

INTERACTION_FEATURES = [
    "direct_x_harmful_alignment",
    "trust_x_next_step",
    "should_refuse_x_compliance",
    "explicit_refusal_x_leakage_after",
    "benign_x_unnecessary_refusal",
    "quoted_x_analysis",
]


def feature_names() -> list[str]:
    return BASE_FEATURES + INTERACTION_FEATURES


def _num(x: dict | None, key: str) -> float:
    if not isinstance(x, dict):
        return 0.0
    v = x.get(key, 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def row_features(row: dict) -> np.ndarray:
    f = row.get("agent_fraud_json") or {}
    r = row.get("agent_refusal_json") or {}
    c = row.get("agent_context_json") or {}

    base = [
        _num(f, "direct_actionability"),
        _num(f, "harmful_goal_alignment"),
        _num(f, "trust_facilitation"),
        _num(f, "risk_downplaying"),
        _num(f, "next_step_encouragement"),
        _num(f, "fraud_cost_reduction"),
        _num(r, "should_refuse_probability"),
        _num(r, "explicit_refusal"),
        _num(r, "safe_redirection"),
        _num(r, "risk_warning"),
        _num(r, "partial_compliance"),
        _num(r, "task_completion"),
        _num(r, "operational_detail_leakage"),
        _num(r, "leakage_before_refusal"),
        _num(r, "leakage_after_refusal"),
        _num(r, "prompt_is_benign"),
        _num(r, "unnecessary_refusal"),
        _num(r, "answer_delivers_safe_utility"),
        _num(c, "analysis_or_education"),
        _num(c, "quoted_content_only"),
        _num(c, "active_endorsement"),
        _num(c, "query_answer_alignment"),
    ]
    d, h, tr, ns, sr, comp = base[0], base[1], base[2], base[4], base[6], base[10]
    ex_ref, leak_after, benign, unnec, quoted, analysis = (
        base[7], base[14], base[15], base[16], base[19], base[18],
    )
    interactions = [
        d * h,
        tr * ns,
        sr * comp,
        ex_ref * leak_after,
        benign * unnec,
        quoted * analysis,
    ]
    return np.asarray(base + interactions, dtype=float)


def build_evidence_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names) for a list of prediction rows."""
    X = np.vstack([row_features(r) for r in rows]) if rows else np.empty((0, len(feature_names())))
    return X, feature_names()


def evidence_digest(rows: list[dict]) -> str:
    """Content hash over the canonical evidence columns."""
    X, _ = build_evidence_matrix(rows)
    h = hashlib.sha256()
    h.update(X.astype(np.float32).tobytes())
    return h.hexdigest()[:16]


def write_canonical_evidence(rows: list[dict], out_dir: Path) -> dict:
    """Write canonical_evidence.parquet / evidence_schema.json / digest."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    X, names = build_evidence_matrix(rows)
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "sample_id", [r.get("id", "") for r in rows])
    df.insert(1, "benchmark", [r.get("benchmark", "") for r in rows])
    df.insert(2, "group_id", [r.get("group_id", "") for r in rows])
    df.to_parquet(out_dir / "canonical_evidence.parquet", index=False)

    schema = {
        "version": "exp2-static-v1",
        "feature_names": names,
        "n_rows": len(rows),
        "generated_by": "build_exp2_evidence_table.py",
    }
    (out_dir / "evidence_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    digest = {
        "evidence_digest": evidence_digest(rows),
        "n_rows": len(rows),
        "n_columns": len(names),
    }
    (out_dir / "evidence_digest.json").write_text(
        json.dumps(digest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return digest
