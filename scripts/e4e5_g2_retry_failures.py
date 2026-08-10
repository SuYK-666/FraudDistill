# -*- coding: utf-8 -*-
"""Retry api_fail rows from previous G2 judgments (map id -> source row -> rejudge)."""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from frauddistill.e4e5_v2.deepseek_fallback import deepseek_judge
from frauddistill.e4e5_v2.selective_policy import BudgetLedger

BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"
LEDGER = REPO / "outputs/exp4_unseen_student_v2/e5_budget_ledger.jsonl"

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def main():
    retries = load(BASE / "manifests" / "g2_retry_list.jsonl")
    print("retry rows:", len(retries))
    # source manifests by tag
    sources = {
        "G2_U2": "manifests/u2_candidates_v2.jsonl",
        "G2_U3safe": "manifests/u3_safe_candidates_v2.jsonl",
    }
    rowmap = {}
    for tag, path in sources.items():
        for r in load(BASE / path):
            rowmap[r["id"]] = r
    ledger = BudgetLedger(LEDGER, hard_stop_cny=10.0, soft_stop_cny=8.0)
    print(f"ledger spent: {ledger.spent():.4f}")
    for rr in retries:
        src = rowmap.get(rr["id"])
        if not src:
            print("no source row for", rr["id"]); continue
        q = str(src.get("user_query") or "")[:2500]
        y = str(src.get("target_model_answer") or "")[:2500]
        ok, why = ledger.can_call(0.002)
        if not ok:
            print("budget stop:", why); break
        res, cost = deepseek_judge(q, y, max_tokens=96)
        if res.get("status") != "ok":
            print("FAIL again", rr["id"], str(res)[:150]); continue
        ledger.record(rr["id"], src.get("qy_hash", ""), rr["tag"] + "_retry", "deepseek",
                      "deepseek-chat", cost.get("input_tokens", 0), cost.get("output_tokens", 0),
                      cost.get("estimated_cost_cny", 0.0))
        out = BASE / "audits" / "g2_retry_results.jsonl"
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": rr["id"], "status": "ok", "judge_label": res.get("label"),
                                "judge_type": res.get("type"), "confidence": res.get("confidence"),
                                "evidence": res.get("evidence", "")[:150]}, ensure_ascii=False) + "\n")
        print("retry ok:", rr["id"], res.get("label"), res.get("type"))
    print(f"total spent: {ledger.spent():.4f}")

if __name__ == "__main__":
    main()
