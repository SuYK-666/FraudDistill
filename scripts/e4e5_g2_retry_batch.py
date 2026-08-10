# -*- coding: utf-8 -*-
"""Batch retry of ALL api_fail rows across G2 judgment files (120 concurrent)."""
from __future__ import annotations
import concurrent.futures as cf, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
from frauddistill.e4e5_v2.deepseek_fallback import deepseek_judge
from frauddistill.e4e5_v2.selective_policy import BudgetLedger

BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"
LEDGER = REPO / "outputs/exp4_unseen_student_v2/e5_budget_ledger.jsonl"

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def main():
    # collect api_fail ids from all judgment files
    fails = {}
    for fname in ("g2_full_rejudge", "g2_u2_candidates", "g2_u3_safe", "g2_u3_refuse_remaining",
                  "g2_u1_leftover", "g2_u3_leftover", "g2_u2_pku_leftover"):
        fp = BASE / "audits" / f"{fname}.jsonl"
        if not fp.exists(): continue
        for r in load(fp):
            if r.get("status") == "api_fail":
                fails[r["id"]] = fname
    # already retried
    for r in load(BASE / "audits" / "g2_retry_results.jsonl"):
        fails.pop(r["id"], None)
    print("api_fail to retry:", len(fails))

    # source row map
    src = {}
    pools = ["manifests/frozen_test.jsonl", "manifests/calibration.jsonl",
             "manifests/u2_candidates_v2.jsonl", "manifests/u3_safe_candidates_v2.jsonl",
             "manifests/u1_leftover_candidates_v2.jsonl", "manifests/u3_leftover_candidates_v2.jsonl",
             "manifests/u2_pku_leftover_candidates_v2.jsonl", "manifests/u3_refuse_remaining_v2.jsonl"]
    for p in pools:
        if (BASE / p).exists():
            for r in load(BASE / p):
                src[r["id"]] = r
    missing = [i for i in fails if i not in src]
    print("missing source rows:", missing[:10], "n=", len(missing))

    ledger = BudgetLedger(LEDGER, hard_stop_cny=10.0, soft_stop_cny=8.0)
    print(f"ledger spent: {ledger.spent():.4f}")

    def one(rid):
        r = src.get(rid)
        if not r: return {"id": rid, "status": "no_source"}
        q = str(r.get("user_query") or "")[:2500]
        y = str(r.get("target_model_answer") or "")[:2500]
        ok, why = ledger.can_call(0.002)
        if not ok: return {"id": rid, "status": "budget_stop", "why": why}
        res, cost = deepseek_judge(q, y, max_tokens=96)
        if res.get("status") != "ok":
            return {"id": rid, "status": "api_fail2", "raw": str(res)[:120]}
        ledger.record(rid, r.get("qy_hash", ""), "G2_RETRY", "deepseek", "deepseek-chat",
                      cost.get("input_tokens", 0), cost.get("output_tokens", 0),
                      cost.get("estimated_cost_cny", 0.0))
        return {"id": rid, "status": "ok", "judge_label": res.get("label"),
                "judge_type": res.get("type"), "confidence": res.get("confidence"),
                "evidence": res.get("evidence", "")[:150]}

    out = BASE / "audits" / "g2_retry_batch.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        with cf.ThreadPoolExecutor(max_workers=120) as ex:
            for res in ex.map(one, list(fails.keys())):
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                if res["status"] == "ok":
                    print("ok:", res["id"], res["judge_label"], flush=True)
                else:
                    print("fail:", res["id"], res["status"], flush=True)
    print(f"DONE spent={ledger.spent():.4f}")

if __name__ == "__main__":
    main()
