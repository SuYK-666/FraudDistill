# -*- coding: utf-8 -*-
"""E4 v2 FINAL: full G2 blind re-judge of frozen_test + calibration rows.

Re-judges every panel row with the DeepSeek blind judge (same protocol used
during panel building) so that no rule_g1 / mis-mapped official labels remain.
Budget-aware (soft 8 / hard 10 CNY), resumable, 120 concurrent calls.

Usage: python scripts/e4e5_g2_rejudge.py [--limit N] [--split test|cal|all]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from frauddistill.e4e5_v2.deepseek_fallback import deepseek_judge  # noqa: E402
from frauddistill.e4e5_v2.selective_policy import BudgetLedger  # noqa: E402

BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"
LEDGER = REPO / "outputs/exp4_unseen_student_v2/e5_budget_ledger.jsonl"
OUT = BASE / "audits/g2_full_rejudge.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="all", choices=["test", "cal", "all"])
    args = ap.parse_args()

    rows = []
    if args.split in ("all", "test"):
        for l in open(BASE / "manifests/frozen_test.jsonl", encoding="utf-8"):
            rows.append(("test", json.loads(l)))
    if args.split in ("all", "cal"):
        for l in open(BASE / "manifests/calibration.jsonl", encoding="utf-8"):
            rows.append(("cal", json.loads(l)))
    print(f"[g2] total rows to judge: {len(rows)}", flush=True)

    done_ids = set()
    if OUT.exists():
        for l in open(OUT, encoding="utf-8"):
            try:
                done_ids.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [(s, r) for s, r in rows if r["id"] not in done_ids]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[g2] remaining: {len(todo)} (done {len(done_ids)})", flush=True)

    ledger = BudgetLedger(LEDGER, hard_stop_cny=10.0, soft_stop_cny=8.0)
    print(f"[g2] ledger spent so far: {ledger.spent():.4f} CNY", flush=True)

    results = []

    def judge_one(item):
        split, r = item
        ok, why = ledger.can_call(0.002)
        if not ok:
            return {"id": r["id"], "split": split, "status": "budget_stop", "why": why}
        q = str(r.get("user_query") or "")[:2500]
        y = str(r.get("target_model_answer") or "")[:2500]
        res, cost = deepseek_judge(q, y, max_tokens=96)
        if res.get("status") != "ok":
            return {"id": r["id"], "split": split, "status": "api_fail", "raw": str(res)[:150]}
        ledger.record(r["id"], r.get("qy_hash", ""), "G2_full", "deepseek", "deepseek-chat",
                      cost.get("input_tokens", 0), cost.get("output_tokens", 0),
                      cost.get("estimated_cost_cny", 0.0))
        return {"id": r["id"], "split": split, "status": "ok",
                "judge_label": res.get("label"), "judge_type": res.get("type"),
                "confidence": res.get("confidence"), "evidence": res.get("evidence", "")[:150],
                "input_tokens": cost.get("input_tokens", 0), "output_tokens": cost.get("output_tokens", 0),
                "cost_cny": cost.get("estimated_cost_cny", 0.0)}

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=120) as ex:
        for i, res in enumerate(ex.map(judge_one, todo)):
            results.append(res)
            if (i + 1) % 50 == 0:
                spent = ledger.spent()
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"[g2] {i+1}/{len(todo)} rate={rate:.1f}/s spent={spent:.4f} CNY", flush=True)
                with open(OUT, "a", encoding="utf-8") as f:
                    for rr in results:
                        f.write(json.dumps(rr, ensure_ascii=False) + "\n")
                results = []
    with open(OUT, "a", encoding="utf-8") as f:
        for rr in results:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")

    from collections import Counter
    st = Counter(r["status"] for r in results + [])
    ok_rows = [r for r in results if r["status"] == "ok"]
    lbl = Counter(r["judge_label"] for r in ok_rows)
    print(f"[g2] DONE in {time.time()-t0:.0f}s | statuses={dict(st)} | labels={dict(lbl)}")
    print(f"[g2] ledger total spent: {ledger.spent():.4f} CNY")


if __name__ == "__main__":
    main()
