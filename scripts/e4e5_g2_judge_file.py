# -*- coding: utf-8 -*-
"""Generic G2 blind-judge runner for a candidate manifest (resumable, budgeted)."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from frauddistill.e4e5_v2.deepseek_fallback import deepseek_judge  # noqa: E402
from frauddistill.e4e5_v2.selective_policy import BudgetLedger  # noqa: E402

LEDGER = REPO / "outputs/exp4_unseen_student_v2/e5_budget_ledger.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag", default="G2_v2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=120)
    ap.add_argument("--hard-stop", type=float, default=10.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    done = {}
    out_path = Path(args.output)
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(l)
                done[r["id"]] = r
            except Exception:
                pass
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[g2f] total={len(rows)} done={len(done)} todo={len(todo)}", flush=True)

    ledger = BudgetLedger(LEDGER, hard_stop_cny=args.hard_stop, soft_stop_cny=args.hard_stop - 2)
    print(f"[g2f] ledger spent: {ledger.spent():.4f}", flush=True)

    def judge_one(r):
        ok, why = ledger.can_call(0.002)
        if not ok:
            return {"id": r["id"], "status": "budget_stop", "why": why}
        q = str(r.get("user_query") or "")[:2500]
        y = str(r.get("target_model_answer") or "")[:2500]
        res, cost = deepseek_judge(q, y, max_tokens=96)
        if res.get("status") != "ok":
            return {"id": r["id"], "status": "api_fail", "raw": str(res)[:150]}
        ledger.record(r["id"], r.get("qy_hash", ""), args.tag, "deepseek", "deepseek-chat",
                      cost.get("input_tokens", 0), cost.get("output_tokens", 0),
                      cost.get("estimated_cost_cny", 0.0))
        return {"id": r["id"], "status": "ok", "judge_label": res.get("label"),
                "judge_type": res.get("type"), "confidence": res.get("confidence"),
                "evidence": res.get("evidence", "")[:150]}

    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, res in enumerate(ex.map(judge_one, todo)):
            results.append(res)
            if (i + 1) % 100 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"[g2f] {i+1}/{len(todo)} rate={rate:.1f}/s spent={ledger.spent():.4f}", flush=True)
                with open(out_path, "a", encoding="utf-8") as f:
                    for rr in results:
                        f.write(json.dumps(rr, ensure_ascii=False) + "\n")
                results = []
    with open(out_path, "a", encoding="utf-8") as f:
        for rr in results:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")

    st = Counter(r["status"] for r in done.values()) + Counter(r["status"] for r in results)
    print(f"[g2f] DONE {time.time()-t0:.0f}s | {dict(st)} | total spent {ledger.spent():.4f} CNY")


if __name__ == "__main__":
    main()
