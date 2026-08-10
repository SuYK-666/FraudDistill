# -*- coding: utf-8 -*-
"""E5 P3: Student -> DeepSeek audit on the most ambiguous test rows.
- Ranks test rows by |risk_score - 0.5| (smallest first = most ambiguous).
- Audits top-K with a single DeepSeek structured judge (temp=0, max_tokens<=96).
- qy-hash cache + resume; BudgetLedger with per-phase ledger; concurrency 120.
- Output human-readable jsonl (query, answer, student_score, ds_label, ...).
Usage: python scripts/e4e5_p3_audit.py [--k 300] [--workers 120] [--max-cny 3.0]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from frauddistill.e4e5_v2.deepseek_fallback import deepseek_judge
from frauddistill.e4e5_v2.schemas import qy_hash, read_jsonl, write_jsonl
from frauddistill.e4e5_v2.selective_policy import BudgetLedger

BASE = REPO / "outputs" / "exp4_unseen_student_v2" / "e4v2_FINAL"


def load_target_rows(k: int) -> list[dict]:
    manifest = read_jsonl(BASE / "manifests" / "frozen_test.jsonl")
    preds = {p["id"]: p for p in read_jsonl(BASE / "predictions" / "final_student.jsonl")}
    rows = []
    for r in manifest:
        p = preds.get(r["id"])
        if p is None:
            continue
        rows.append({
            "id": r["id"],
            "user_query": r["user_query"],
            "target_model_answer": r["target_model_answer"],
            "gold_label": r["gold_label"],
            "gold_binary": r["gold_binary"],
            "student_label": p["label"],
            "student_type": p["risk_type"],
            "risk_score": float(p["risk_score"]),
            "primary_shift": r.get("primary_shift", ""),
        })
    rows.sort(key=lambda x: abs(x["risk_score"] - 0.5))
    return rows[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--max-cny", type=float, default=3.0)
    ap.add_argument("--soft-cny", type=float, default=2.5)
    args = ap.parse_args()

    e5_dir = BASE / "e5"
    e5_dir.mkdir(parents=True, exist_ok=True)
    cache_path = e5_dir / "p3_audit_cache.jsonl"
    ledger_path = e5_dir / "p3_audit_budget_ledger.jsonl"
    out_path = e5_dir / "p3_audit_results.jsonl"
    state_path = e5_dir / "p3_audit_state.json"

    budget = BudgetLedger(ledger_path, hard_stop_cny=args.max_cny, soft_stop_cny=args.soft_cny)

    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                cache[r["qy_hash"]] = r
            except Exception:
                pass

    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass

    rows = load_target_rows(args.k)
    todo = [r for r in rows if r["id"] not in done and qy_hash(r["user_query"], r["target_model_answer"]) not in cache]
    print(f"[p3] k={args.k} rows={len(rows)} cache_hits={len(rows) - len(todo)} todo={len(todo)} "
          f"ledger_spent={budget.spent():.4f}", flush=True)

    lock = threading.Lock()
    results = []
    ok_count = 0
    fail_count = 0
    budget_skipped = 0
    start = time.time()

    def work(r: dict) -> None:
        nonlocal ok_count, fail_count, budget_skipped
        h = qy_hash(r["user_query"], r["target_model_answer"])
        c = cache.get(h)
        if c is not None:
            with lock:
                results.append({**r, "ds_label": c["label"], "ds_type": c["type"],
                                "ds_confidence": c.get("confidence"), "ds_evidence": c.get("evidence"),
                                "ds_source": "cache"})
            return
        with lock:
            ok, msg = budget.can_call(0.002, retry_reserve=0.004)
            if not ok:
                budget_skipped += 1
                results.append({**r, "ds_label": None, "ds_type": None, "ds_source": f"budget:{msg}"})
                return
        res, cost = deepseek_judge(r["user_query"], r["target_model_answer"], max_tokens=96,
                                   temperature=0, max_retries=2, model="deepseek-chat",
                                   price_in_cny=(0.3, 0.6))
        with lock:
            if res.get("status") == "ok":
                ok_count += 1
                budget.record(r["id"], h, "p3_audit", "deepseek", "deepseek-chat",
                              cost.get("input_tokens", 0), cost.get("output_tokens", 0),
                              cost.get("estimated_cost_cny", 0.0))
                entry = {"qy_hash": h, "id": r["id"], "label": res["label"], "type": res["type"],
                         "confidence": res.get("confidence"), "evidence": res.get("evidence")}
                cache[h] = entry
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                results.append({**r, "ds_label": res["label"], "ds_type": res["type"],
                                "ds_confidence": res.get("confidence"), "ds_evidence": res.get("evidence"),
                                "ds_source": "deepseek"})
            else:
                fail_count += 1
                results.append({**r, "ds_label": None, "ds_type": None, "ds_source": "failed",
                                "ds_error": res.get("error", "")})

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        n = len(futs)
        last_print = 0
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            now = time.time()
            if now - last_print >= 5 or i == n:
                last_print = now
                el = now - start
                rate = i / el if el > 0 else 0
                eta = (n - i) / rate if rate > 0 else 0
                spent = budget.spent()
                print(f"[p3] {i}/{n} ok={ok_count} fail={fail_count} skip={budget_skipped} "
                      f"spent={spent:.4f} rate={rate:.1f}/s eta={eta:.0f}s", flush=True)

    # merge cache-hit/done rows from previous runs (they are not in `results`)
    by_id = {r["id"]: r for r in results}
    full = []
    for r in rows:
        if r["id"] in by_id:
            full.append(by_id[r["id"]])
        else:
            h = qy_hash(r["user_query"], r["target_model_answer"])
            c = cache.get(h)
            if c is not None:
                full.append({**r, "ds_label": c["label"], "ds_type": c["type"],
                             "ds_confidence": c.get("confidence"), "ds_evidence": c.get("evidence"),
                             "ds_source": "cache"})
            else:
                full.append({**r, "ds_label": None, "ds_type": None, "ds_source": "missing"})

    write_jsonl(out_path, full)
    n_ds = sum(1 for r in full if r.get("ds_source") == "deepseek")
    n_cache = sum(1 for r in full if r.get("ds_source") == "cache")
    n_fail = sum(1 for r in full if r.get("ds_source") in ("failed", "missing"))
    state = {
        "k": args.k, "n_rows": len(full), "n_deepseek": n_ds, "n_cache": n_cache,
        "n_failed": n_fail, "spent_cny": round(budget.spent(), 6),
        "elapsed_s": round(time.time() - start, 1),
        "unlabeled_count": sum(1 for r in full if r.get("ds_label") is None),
    }
    Path(state_path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[p3] done: deepseek={n_ds} cache={n_cache} failed={n_fail} spent={budget.spent():.4f} "
          f"elapsed={state['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
