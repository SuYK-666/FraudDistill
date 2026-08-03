"""Annotate Fraud-R1 Online-Relationships rows for the balanced diagnostic set.

Two blinded LLM annotators + adjudicator on disagreement (same protocol as the
existing exp2 human_audit files). Writes per-annotator jsonl + adjudicated file
under fraudr1/diagnostic/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frauddistill.exp2_cross_benchmark.api_client import ApiConfig, CostLedger, complete_json, make_client
from frauddistill.exp2_cross_benchmark.audit import ANNOTATOR_PROMPT, ANNOTATOR_SYSTEM, ADJUDICATOR_SYSTEM

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "exp2_prior_work_comparison", "fraudr1", "diagnostic", "_romance_work"))
MODEL_A = "deepseek-v4-flash"
MODEL_B = "deepseek-v4-pro"


def _load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


async def annotate_rows(rows, annotator, out_path, concurrency=120, temp=0.2):
    cfg = ApiConfig("deepseek", annotator)
    client = make_client(cfg)
    ledger = CostLedger()
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    pending = [r for r in rows if r["id"] not in done]
    print(f"[annotate:{annotator}] pending={len(pending)}")

    async def one(row):
        async with sem:
            prompt = ANNOTATOR_PROMPT.format(query=row["query"], answer=row["answer"])
            rec = await complete_json(client, cfg, ledger, system_prompt=ANNOTATOR_SYSTEM, user_prompt=prompt, max_tokens=160, temperature=temp, json_mode=True)
        p = rec.get("parsed", {})
        result = {
            "id": row["id"], "group_id": row["group_id"], "benchmark": "fraudr1_romance",
            "annotator": annotator,
            "binary": p.get("binary"), "four_type": p.get("four_type"),
            "q1_should_refuse": p.get("q1_should_refuse"), "q2_did_refuse": p.get("q2_did_refuse"),
            "q3_still_helped": p.get("q3_still_helped"), "q4_reduced_execution_cost": p.get("q4_reduced_execution_cost"),
            "q5_unnecessary_refusal": p.get("q5_unnecessary_refusal"),
            "rationale": str(p.get("rationale", ""))[:300],
            "parse_status": "ok" if rec["parse_ok"] else "parse_failed",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        async with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    for i in range(0, len(pending), 400):
        part = pending[i : i + 400]
        await asyncio.gather(*[one(r) for r in part])
        print(f"[annotate:{annotator}] {min(i+400, len(pending))}/{len(pending)} cost={ledger.snapshot(cfg.prices)}", flush=True)
    print(f"[annotate:{annotator}] done cost={ledger.snapshot(cfg.prices)}")


async def adjudicate_rows(rows, a_path, b_path, out_path, concurrency=120):
    a = {json.loads(l)["id"]: json.loads(l) for l in open(a_path, encoding="utf-8")}
    b = {json.loads(l)["id"]: json.loads(l) for l in open(b_path, encoding="utf-8")}
    ids = sorted(set(a) & set(b))
    cfg = ApiConfig("deepseek", MODEL_A)
    client = make_client(cfg)
    ledger = CostLedger()
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    to_do = [sid for sid in ids if sid not in done]
    print(f"[adjudicate] pending={len(to_do)}")

    async def one(sid):
        ra, rb = a[sid], b[sid]
        if ra.get("binary") == rb.get("binary") and ra.get("four_type") == rb.get("four_type"):
            final = {"binary": ra["binary"], "four_type": ra["four_type"], "adjudicated": False, "rationale": "agreement"}
        else:
            async with sem:
                prompt = (
                    f"Annotator A: binary={ra.get('binary')} four_type={ra.get('four_type')} rationale={ra.get('rationale')}\n"
                    f"Annotator B: binary={rb.get('binary')} four_type={rb.get('four_type')} rationale={rb.get('rationale')}"
                )
                rec = await complete_json(client, cfg, ledger, system_prompt=ADJUDICATOR_SYSTEM, user_prompt=prompt, max_tokens=120, temperature=0.0, json_mode=True)
            p = rec.get("parsed", {})
            final = {"binary": p.get("binary"), "four_type": p.get("four_type"), "adjudicated": True, "rationale": str(p.get("rationale", ""))[:300]}
        result = {
            "id": sid, "group_id": ra["group_id"], "benchmark": "fraudr1_romance",
            "annotator_a_binary": ra.get("binary"), "annotator_b_binary": rb.get("binary"),
            "annotator_a_type": ra.get("four_type"), "annotator_b_type": rb.get("four_type"),
            **final,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        async with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    for i in range(0, len(to_do), 400):
        part = to_do[i : i + 400]
        await asyncio.gather(*[one(sid) for sid in part])
    print(f"[adjudicate] done cost={ledger.snapshot(cfg.prices)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--adjudicate", action="store_true")
    ap.add_argument("--concurrency", type=int, default=120)
    ap.add_argument("--input", default="romance_annotation_input.jsonl")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = _load(os.path.join(BASE, args.input))
    tag = args.tag
    if args.annotate:
        asyncio.run(annotate_rows(rows, MODEL_A, os.path.join(BASE, f"romance_audit_a{tag}.jsonl"), args.concurrency))
        asyncio.run(annotate_rows(rows, MODEL_B, os.path.join(BASE, f"romance_audit_b{tag}.jsonl"), args.concurrency))
    if args.adjudicate:
        asyncio.run(adjudicate_rows(rows, os.path.join(BASE, f"romance_audit_a{tag}.jsonl"), os.path.join(BASE, f"romance_audit_b{tag}.jsonl"), os.path.join(BASE, f"romance_audit_adjudicated{tag}.jsonl"), args.concurrency))


if __name__ == "__main__":
    main()
