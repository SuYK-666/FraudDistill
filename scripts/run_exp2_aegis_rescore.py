# -*- coding: utf-8 -*-
"""Run the Aegis task-specific content-harm rescore (Exp2 balanced final round).

Pipeline: frozen Evidence-MAT specialists (refusal + context + arbiter) with the
aegis content-harm audit composed into the refusal and arbiter prompts. The
context agent stays on the frozen prompt so its cached outputs are reused.

Modes:
  --smoke N   : N pilot rows (cheap sanity check, ~0.2 RMB)
  --pilot     : 180 official-validation aegis rows with gold (C2 pilot subset)
  --test      : 813 official aegis response-test rows (final artifact)

Usage:
  python scripts/run_exp2_aegis_rescore.py --pilot --budget 3
  python scripts/run_exp2_aegis_rescore.py --test --budget 15 --used-rmb 0
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher
from frauddistill.providers.deepseek_client import DeepSeekClient
from frauddistill.runtime.budget import BudgetState
from frauddistill.runtime.cache import RequestCache
from frauddistill.teacher.evidence_table import build_evidence_table

from aegis_rescore_prompts import build_prompts

BASE = REPO / "experiments" / "exp2_prior_work_comparison" / "balanced_design"
TEST_MANIFEST = BASE / "manifests" / "balanced_test_manifest.jsonl"
PILOT_MANIFEST = REPO / "experiments" / "exp2_prior_work_comparison" / "archive" / "prefinal_20260806" / "pilot" / "skill_gate_manifest.jsonl"
CACHE_DIR = REPO / "outputs" / "api_cache_exp2"
MODEL = "deepseek-v4-flash"
CONCURRENCY = 120
CHECKPOINT_EVERY = 25
MAX_TOKENS = {"fraud": 560, "refusal": 900, "context": 420, "arbiter": 640}

_spec = importlib.util.spec_from_file_location("run_exp2_teacher", REPO / "scripts" / "run_exp2_teacher.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)
normalize_arbiter = rt.normalize_arbiter
sample_for_agents = rt.sample_for_agents
prompt_digest = rt.prompt_digest

PROMPTS = build_prompts()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_existing(path: Path) -> dict[str, dict]:
    out = {}
    for r in read_jsonl(path):
        out[str(r.get("id") or r.get("sample_id"))] = r
    return out


def build_teacher(client) -> MultiAgentTeacher:
    return MultiAgentTeacher(
        client,
        use_fraud_agent=False,
        use_refusal_agent=True,
        use_context_agent=True,
        arbiter_mode="evidence",
        use_correction=False,
        threshold=0.5,
        max_tokens=dict(MAX_TOKENS),
    )


async def process_row_aegis(teacher: MultiAgentTeacher, row: dict) -> dict:
    sample = sample_for_agents(row)
    refusal_env = await teacher.refusal_agent.run_async(
        sample, teacher.client, system_prompt_override=PROMPTS["refusal"],
    )
    context_env = await teacher.context_agent.run_async(sample, teacher.client)
    table = build_evidence_table(None, refusal_env, context_env)
    arb = await teacher.arbiter_agent.run_async(
        sample, table, threshold=teacher.threshold, client=teacher.client,
        system_prompt_override=PROMPTS["arbiter"],
    )
    out = normalize_arbiter(arb, row, "aegis2", base_prompts=PROMPTS)
    out["evidence_table"] = table
    out["agent_fraud_json"] = None
    out["agent_refusal_json"] = refusal_env.get("parsed")
    out["agent_context_json"] = context_env.get("parsed")
    out["arbiter_json"] = arb if (isinstance(arb, dict) and arb.get("teacher_label") is not None) else (arb or {}).get("parsed")
    out["agent_agreement"] = arb.get("agent_agreement")
    out["contradiction_flags"] = arb.get("contradiction_flags")
    out["latency_ms"] = round(
        float(arb.get("latency_ms", 0))
        + sum(float(env.get("latency_ms", 0)) for env in (refusal_env, context_env)),
        1,
    )
    arb_usage = arb.get("usage") or {}
    out["input_tokens"] = sum(
        int((env.get("usage") or {}).get("input_miss", 0) + (env.get("usage") or {}).get("input_hit", 0))
        for env in (refusal_env, context_env)
    ) + int(arb_usage.get("input_miss", 0) + arb_usage.get("input_hit", 0))
    out["output_tokens"] = sum(int((env.get("usage") or {}).get("output", 0)) for env in (refusal_env, context_env)) + int(arb_usage.get("output", 0))
    return out


async def run_rows(rows: list[dict], out_path: Path, budget: BudgetState, teacher) -> dict:
    existing = load_existing(out_path)
    todo = [r for r in rows if str(r["sample_id"]) not in existing]
    print(f"[aegis-rescore] rows={len(rows)} done={len(rows) - len(todo)} todo={len(todo)}", flush=True)
    if not todo:
        return {"todo": 0, "done": 0}
    sem = asyncio.Semaphore(CONCURRENCY)
    failures = 0
    parse_failed = 0
    results = dict(existing)

    async def worker(row: dict):
        async with sem:
            try:
                return str(row["sample_id"]), await process_row_aegis(teacher, row), ""
            except Exception as exc:  # noqa: BLE001
                return str(row["sample_id"]), None, f"{type(exc).__name__}: {exc}"

    started = time.perf_counter()
    done_count = 0
    batch = []
    budget_stop = False
    for row in todo:
        if not teacher.client.budget.can_spend(0.05):
            print(f"[aegis-rescore] BUDGET STOP used={teacher.client.budget.used_rmb:.2f}", flush=True)
            budget_stop = True
            break
        batch.append(asyncio.create_task(worker(row)))
        if len(batch) >= CONCURRENCY:
            for rid, rec, err in await asyncio.gather(*batch):
                done_count += 1
                if rec is not None:
                    results[rid] = rec
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if rec.get("abstain") or rec.get("parse_status") == "parse_failed":
                        parse_failed += 1
                else:
                    failures += 1
                    if err.startswith("BUDGET_EXCEEDED"):
                        budget_stop = True
                    print(f"[aegis-rescore] FAILED {rid}: {err}", flush=True)
            batch = []
            if done_count % CHECKPOINT_EVERY == 0:
                rate = done_count / max(time.perf_counter() - started, 1e-9)
                print(f"[aegis-rescore] progress {done_count}/{len(todo)} rows/s={rate:.2f} "
                      f"used_rmb={teacher.client.budget.used_rmb:.4f} failures={failures} parse_failed={parse_failed}", flush=True)
    if batch:
        for rid, rec, err in await asyncio.gather(*batch):
            done_count += 1
            if rec is not None:
                results[rid] = rec
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                failures += 1
    return {"todo": len(todo), "done": done_count, "failures": failures, "parse_failed": parse_failed, "budget_stop": budget_stop}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="run N pilot rows as smoke test")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--used-rmb", type=float, default=0.0)
    ap.add_argument("--out", default="", help="output path relative to balanced_design")
    args = ap.parse_args()

    if args.test:
        rows = [r for r in read_jsonl(TEST_MANIFEST) if r.get("source") == "aegis2"]
        out_rel = args.out or "predictions/aegis_fd_predictions_v3.jsonl"
        tag = "test"
    else:
        rows = [r for r in read_jsonl(PILOT_MANIFEST) if r.get("source") == "aegis2"]
        if args.smoke:
            rows = rows[: args.smoke]
            out_rel = args.out or "predictions/aegis_rescore_smoke.jsonl"
            tag = "smoke"
        else:
            out_rel = args.out or "predictions/aegis_rescore_pilot.jsonl"
            tag = "pilot"
    print(f"[aegis-rescore] mode={tag} rows={len(rows)} budget={args.budget}")

    budget = BudgetState(max_rmb=args.budget, reserved_rmb=0.0, used_rmb=args.used_rmb)
    cache = RequestCache(str(CACHE_DIR))
    teacher_client = DeepSeekClient(model=MODEL, json_mode=True, thinking_disabled=True,
                                    timeout_seconds=90, max_retries=1, cache=cache, budget=budget,
                                    concurrency=CONCURRENCY)
    teacher = build_teacher(teacher_client)
    out_path = BASE / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = asyncio.run(run_rows(rows, out_path, budget, teacher))
    print(f"[aegis-rescore] {res} used_rmb={budget.used_rmb:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
