"""Auditor-model selection on calibration rows only; never uses the 384-row test."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .audit import AUDIT_PROMPT_V2, _client, _judge_one, prompt_fingerprint
from .common import REPO, binary_label, binary_metrics, read_jsonl, row_key, write_json, write_jsonl

SUPP_DIR = REPO / "experiments" / "lighteval_supplementary"
RAW_DIR = SUPP_DIR / "prepared" / "raw"
OUT_DIR = SUPP_DIR / "audit" / "calibration"
PROMPT_V2 = AUDIT_PROMPT_V2


def run(model: str = "deepseek-reasoner", limit: int | None = None,
        workers: int = 4, max_cny: float = 5.0, max_output_tokens: int = 512,
        prompt_version: str = "v1") -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rel = read_jsonl(RAW_DIR / "relation_calibration.jsonl")
    hard = read_jsonl(RAW_DIR / "hard_safe_calibration.jsonl")
    rows = rel + hard
    if limit is not None:
        rows = rows[:limit]
    tasks = [{"id": row_key(row), "rank": i + 1, "user_query": row["user_query"],
              "target_model_answer": row["target_model_answer"]} for i, row in enumerate(rows)]
    stem = f"{model.replace('/', '_')}_{prompt_version}"
    cache_path = OUT_DIR / f"{stem}_cache.jsonl"
    result_path = OUT_DIR / f"{stem}_results.jsonl"
    ledger_path = OUT_DIR / f"{stem}_ledger.jsonl"
    cfg = {
        "model": model, "temperature": 0.0, "omit_temperature": model == "deepseek-reasoner",
        "max_output_tokens": max_output_tokens, "max_input_chars": 16000,
        "prompt": PROMPT_V2 if prompt_version == "v2" else None,
        "model_snapshot": model,
        "prompt_version": prompt_version,
    }
    fingerprint = hashlib.sha256(json.dumps({**prompt_fingerprint(), "model": model,
                                              "prompt_version": prompt_version,
                                              "prompt": cfg["prompt"] or "v1-constant",
                                              "model_snapshot": model,
                                              "max_output_tokens": max_output_tokens},
                                             sort_keys=True).encode("utf-8")).hexdigest()
    cache = {}
    for row in read_jsonl(cache_path):
        if row.get("config_fingerprint") == fingerprint and row.get("audit_label") in {"safe", "unsafe"}:
            cache[row["task_id"]] = row
    todo = [task for task in tasks if task["id"] not in cache]
    plan = {
        "model": model, "rows": len(rows), "cached": len(cache), "todo": len(todo),
        "prompt": prompt_fingerprint(), "config_fingerprint": fingerprint,
        "scope": "calibration_only",
    }
    write_json(OUT_DIR / f"{stem}_plan.json", plan)
    if not todo:
        print(json.dumps({**plan, "status": "CACHE_ONLY"}, ensure_ascii=False, indent=2))

    results = dict(cache)
    if todo:
        from frauddistill.e4e5_v2.selective_policy import BudgetLedger
        budget = BudgetLedger(ledger_path, hard_stop_cny=max_cny, soft_stop_cny=max_cny * 0.8)
        client = _client(model)
        lock = threading.Lock()

        def work(task: dict) -> dict:
            ok, msg = budget.can_call(0.001, retry_reserve=0.005)
            if not ok:
                return {"id": task["id"], "rank": task["rank"], "audit_status": "budget_skipped",
                        "audit_label": None, "audit_type": None, "error": msg,
                        "input_tokens": 0, "output_tokens": 0, "estimated_cost_cny": 0.0}
            row = _judge_one(client, task, cfg, max_retries=1,
                             input_cost_per_m=2.0,
                             output_cost_per_m=8.0)
            with lock:
                if row.get("input_tokens", 0) or row.get("output_tokens", 0):
                    budget.record(row["id"], task["id"], "auditor_calibration", "deepseek", model,
                                  row["input_tokens"], row["output_tokens"],
                                  row.get("estimated_cost_cny", 0.0), row.get("retry_count", 0))
                if row.get("audit_status") == "ok":
                    native = {"task_id": task["id"], "rank": task["rank"],
                              "config_fingerprint": fingerprint, "prompt_version": "lighteval-audit-v1",
                              **row}
                    with cache_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(native, ensure_ascii=False) + "\n")
            return row

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, task) for task in todo]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                results[row["id"]] = row
                if i % 10 == 0 or i == len(futures):
                    print(f"[cal-audit] {i}/{len(futures)}", flush=True)

    ordered = []
    for row in rows:
        key = row_key(row)
        result = results.get(key, {})
        ordered.append({**row, "audit_label": result.get("audit_label"),
                        "audit_type": result.get("audit_type"),
                        "audit_status": result.get("audit_status"),
                        "input_tokens": result.get("input_tokens", 0),
                        "output_tokens": result.get("output_tokens", 0),
                        "estimated_cost_cny": result.get("estimated_cost_cny", 0.0)})
    write_jsonl(result_path, ordered)
    valid = [row for row in ordered if row.get("audit_label") in {"safe", "unsafe"}]
    metrics = binary_metrics([binary_label(row) for row in valid],
                             [float(row["audit_label"] == "unsafe") for row in valid],
                             pred=[int(row["audit_label"] == "unsafe") for row in valid],
                             threshold=0.5) if valid else {}
    summary = {**plan, "valid": len(valid), "invalid": len(ordered) - len(valid),
               "metrics_on_calibration": metrics,
               "actual_cost_cny": round(sum(float(row.get("estimated_cost_cny", 0) or 0) for row in ordered), 6),
               "results": str(result_path), "status": "DONE"}
    write_json(OUT_DIR / f"{stem}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cny", type=float, default=5.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args(argv)
    run(args.model, args.limit, args.workers, args.max_cny, args.max_output_tokens, args.prompt_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
