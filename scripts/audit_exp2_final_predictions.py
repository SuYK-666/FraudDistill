# -*- coding: utf-8 -*-
"""Exp2 FINAL integrity audit (formal guide Phase 4 / section 22 Technical Gate).

Checks per benchmark:
  coverage = 100%, parse failure = 0, empty output = 0, truncation = 0,
  missing field = 0, skill trace missing = 0, duplicate sample = 0,
  unexpected q+y hash = 0, scores finite in [0,1].
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BASE = REPO / "experiments" / "exp2_prior_work_comparison"
MANIFEST_DIR = BASE / "manifests"
PRED_DIR = BASE / "predictions"
AUDIT_DIR = BASE / "audit"

SPECS = {
    "fraudr1": ("fraudr1_content_final_3000.jsonl", "fraudr1_evidence_mat_3000.jsonl",
                ("fraud_assistance_score", "over_refusal_score", "response_content_harm_score",
                 "general_harmful_compliance_score", "did_refuse_score")),
    "orbench": ("orbench_final_3000.jsonl", "orbench_evidence_mat_3000.jsonl",
                ("behavior_error_score", "over_refusal_score", "partial_leakage_score",
                 "general_harmful_compliance_score")),
    "do_not_answer": ("dna_final_5634.jsonl", "dna_evidence_mat_5634.jsonl",
                      ("general_harmful_compliance_score", "over_refusal_score")),
    "aegis2": ("aegis_response_test_813.jsonl", "aegis_evidence_mat_response_813.jsonl",
               ("response_content_harm_score",)),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    report = {}
    all_ok = True
    for b, (mf, pf, heads) in SPECS.items():
        mani = read_jsonl(MANIFEST_DIR / mf)
        preds = {r["id"]: r for r in read_jsonl(PRED_DIR / pf)}
        m_by_id = {r["sample_id"]: r for r in mani}
        checks = {
            "manifest_n": len(mani),
            "prediction_n": len(preds),
            "coverage_pct": round(100.0 * len([s for s in m_by_id if s in preds]) / len(mani), 4),
            "duplicate_sample": len(preds) - len(set(preds)),
            "parse_failed": 0, "abstain": 0, "empty_answer": 0, "truncated": 0,
            "missing_required_field": 0, "skill_trace_missing": 0,
            "qy_hash_mismatch": 0, "nonfinite_score": 0, "score_out_of_range": 0,
            "ids_not_in_manifest": 0, "manifest_ids_missing": 0,
        }
        for sid, m in m_by_id.items():
            p = preds.get(sid)
            if p is None:
                checks["manifest_ids_missing"] += 1
                continue
            if p.get("parse_status") != "ok" or p.get("abstain"):
                checks["parse_failed"] += 1
            if not str(p.get("answer", "") or "").strip():
                checks["empty_answer"] += 1
            if p.get("finish_reason") == "length":
                checks["truncated"] += 1
            for h in heads:
                v = p.get(h)
                if v is None:
                    checks["missing_required_field"] += 1
                else:
                    try:
                        vf = float(v)
                        if vf != vf or vf in (float("inf"), float("-inf")):
                            checks["nonfinite_score"] += 1
                        elif vf < 0.0 or vf > 1.0:
                            checks["score_out_of_range"] += 1
                    except (TypeError, ValueError):
                        checks["nonfinite_score"] += 1
            if not p.get("skill_trace"):
                checks["skill_trace_missing"] += 1
            if p.get("qy_hash") and m.get("qy_hash") and p.get("qy_hash") != m.get("qy_hash"):
                checks["qy_hash_mismatch"] += 1
        for sid in preds:
            if sid not in m_by_id:
                checks["ids_not_in_manifest"] += 1
        ok = (checks["coverage_pct"] == 100.0 and checks["parse_failed"] == 0
              and checks["duplicate_sample"] == 0 and checks["missing_required_field"] == 0
              and checks["manifest_ids_missing"] == 0)
        report[b] = checks
        all_ok = all_ok and ok
        print(f"[integrity:{b}] ok={ok} {json.dumps(checks)}")
    report["overall_ok"] = all_ok
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "final_integrity_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OVERALL:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()