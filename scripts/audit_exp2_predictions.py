# -*- coding: utf-8 -*-
"""Exp2 prediction integrity + schema audit (guide sections 6.7, 7).

Zero-API. Checks every T6 prediction file for completeness, schema-field
availability, degenerate score distributions and evidence coverage.

Outputs:
  experiments/exp2_prior_work_comparison/audit/schema_integrity_summary.json
  experiments/exp2_prior_work_comparison/audit/suspicious_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_static_repair.offline_guard import require_offline  # noqa: E402
from frauddistill.exp2_static_repair.schemas import (  # noqa: E402
    REQUIRED_CONTEXT_FIELDS,
    REQUIRED_FRAUD_FIELDS,
    REQUIRED_REFUSAL_FIELDS,
)

EXPERIMENT_DIR = REPO / "experiments" / "exp2_prior_work_comparison"
TEACHER_T6_DIR = EXPERIMENT_DIR / "teacher_predictions_t6"
AUDIT_DIR = EXPERIMENT_DIR / "audit"

FILES = {
    "fraudr1": "fraudr1_t6_predictions.jsonl",
    "orbench": "orbench_t6_predictions.jsonl",
    "do_not_answer": "do_not_answer_t6_predictions.jsonl",
    "aegis2_response": "aegis2_t6_predictions.jsonl",
    "aegis2_prompt": "aegis2_t6_prompt_predictions.jsonl",
    "aegis2_validation": "aegis_validation_t6_predictions.jsonl",
}

EXPECTED_N = {
    "fraudr1": 8564,
    "orbench": 3000,
    "do_not_answer": 5634,
    "aegis2_response": 813,
    "aegis2_prompt": 1151,
    "aegis2_validation": 300,
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def score_distribution_checks(scores: list[float]) -> dict:
    if not scores:
        return {"unique_count": 0, "std": 0.0, "p01": None, "p50": None, "p99": None, "degenerate": True}
    s = np.asarray(scores, dtype=float)
    unique_count = int(len(np.unique(np.round(s, 6))))
    std = float(np.std(s))
    checks = {
        "unique_count": unique_count,
        "std": round(std, 6),
        "p01": float(np.quantile(s, 0.01)),
        "p50": float(np.quantile(s, 0.50)),
        "p99": float(np.quantile(s, 0.99)),
        "degenerate": (unique_count < 20) or (std < 0.03),
    }
    return checks


def audit_file(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    ids = {r.get("id") or r.get("sample_id") for r in rows}
    parse_failed = sum(1 for r in rows if r.get("parse_status") == "parse_failed" or r.get("parse_failed"))
    abstain = sum(1 for r in rows if r.get("abstain"))
    missing_score = sum(1 for r in rows if r.get("risk_score") is None)
    missing_raw = sum(1 for r in rows if not r.get("rationale") and not r.get("raw_content"))

    fraud_missing = sum(1 for r in rows if not isinstance(r.get("agent_fraud_json"), dict))
    refusal_missing = sum(1 for r in rows if not isinstance(r.get("agent_refusal_json"), dict))
    context_missing = sum(1 for r in rows if not isinstance(r.get("agent_context_json"), dict))
    arbiter_missing = sum(1 for r in rows if r.get("arbiter_json") is None and "arbiter_json" in r)

    fraud_field_missing = 0
    refusal_field_missing = 0
    context_field_missing = 0
    all_zero_specialist = 0
    empty_evidence = 0
    weak_reason = 0

    suspicious = []
    for r in rows:
        f = r.get("agent_fraud_json") or {}
        rf = r.get("agent_refusal_json") or {}
        c = r.get("agent_context_json") or {}
        if f:
            fraud_field_missing += len(REQUIRED_FRAUD_FIELDS - set(f))
            nums = [f.get(k) for k in ("fraud_probability", "direct_actionability", "harmful_goal_alignment", "trust_facilitation", "risk_downplaying", "fraud_cost_reduction")]
            nums = [v for v in nums if isinstance(v, (int, float))]
            if nums and max(nums) == 0.0:
                all_zero_specialist += 1
        if rf:
            refusal_field_missing += len(REQUIRED_REFUSAL_FIELDS - set(rf))
        if c:
            context_field_missing += len(REQUIRED_CONTEXT_FIELDS - set(c))
        if not f.get("evidence_spans") and not f.get("counter_evidence_spans"):
            empty_evidence += 1
        if len(str(f.get("concise_reason") or "").strip()) < 8:
            weak_reason += 1

        flags = []
        if r.get("parse_status") == "parse_failed" or r.get("parse_failed"):
            flags.append("parse_failed")
        if r.get("abstain"):
            flags.append("abstain")
        if f and nums and max(nums) == 0.0 and not f.get("evidence_spans") and not f.get("counter_evidence_spans") and len(str(f.get("concise_reason") or "").strip()) < 8:
            flags.append("suspicious_empty_fraud_output")
        if flags:
            suspicious.append({"sample_id": r.get("id") or r.get("sample_id"), "flags": flags})

    expected = EXPECTED_N.get(name)
    integrity_ok = (
        (expected is None or n == expected)
        and len(ids) == n
        and parse_failed == 0
        and missing_score == 0
    )
    return {
        "file": name,
        "n": n,
        "expected_n": expected,
        "unique_ids": len(ids),
        "parse_failed": parse_failed,
        "abstain": abstain,
        "missing_score": missing_score,
        "missing_raw_output": missing_raw,
        "fraud_specialist_missing": fraud_missing,
        "refusal_specialist_missing": refusal_missing,
        "context_specialist_missing": context_missing,
        "arbiter_missing": arbiter_missing,
        "fraud_required_field_missing": fraud_field_missing,
        "refusal_required_field_missing": refusal_field_missing,
        "context_required_field_missing": context_field_missing,
        "all_zero_specialist": all_zero_specialist,
        "empty_evidence": empty_evidence,
        "weak_reason": weak_reason,
        "score_distribution": score_distribution_checks(
            [r.get("risk_score") for r in rows if r.get("risk_score") is not None]
        ),
        "integrity_ok": bool(integrity_ok),
        "suspicious_count": len(suspicious),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="enable offline guard")
    args = ap.parse_args()
    if args.offline:
        import os
        os.environ["FRAUDDISTILL_OFFLINE"] = "1"
    require_offline()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    all_suspicious = []
    for name, fname in FILES.items():
        rows = read_jsonl(TEACHER_T6_DIR / fname)
        res = audit_file(name, rows)
        summary[name] = res
        # collect real suspicious rows
        for r in rows:
            flags = []
            if r.get("parse_status") == "parse_failed" or r.get("parse_failed"):
                flags.append("parse_failed")
            if r.get("abstain"):
                flags.append("abstain")
            f = r.get("agent_fraud_json") or {}
            nums = [f.get(k) for k in ("fraud_probability", "direct_actionability", "harmful_goal_alignment", "trust_facilitation", "risk_downplaying", "fraud_cost_reduction")]
            nums = [v for v in nums if isinstance(v, (int, float))]
            if nums and max(nums) == 0.0 and not f.get("evidence_spans") and not f.get("counter_evidence_spans") and len(str(f.get("concise_reason") or "").strip()) < 8:
                flags.append("suspicious_empty_fraud_output")
            if flags:
                all_suspicious.append({
                    "benchmark": name,
                    "sample_id": r.get("id") or r.get("sample_id"),
                    "flags": flags,
                })

    overall = all(v["integrity_ok"] for v in summary.values())
    report = {
        "guide": "exp2-static-fix-20260806",
        "offline": True,
        "overall_integrity_ok": bool(overall),
        "per_benchmark": summary,
    }
    (AUDIT_DIR / "schema_integrity_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (AUDIT_DIR / "suspicious_predictions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in all_suspicious),
        encoding="utf-8",
    )
    print(json.dumps({k: {"n": v["n"], "integrity_ok": v["integrity_ok"], "suspicious_count": v["suspicious_count"]} for k, v in summary.items()}, ensure_ascii=False, indent=1))
    print("outputs ->", AUDIT_DIR)


if __name__ == "__main__":
    main()
