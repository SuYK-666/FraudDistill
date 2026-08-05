# -*- coding: utf-8 -*-
"""Run the frozen T6 Evidence MAT teacher on the Exp2 task-aligned test manifest.

Frozen method (guide 2.1 / 13): Fraud + Refusal + Context specialists +
Evidence Arbiter, conflict correction OFF, model deepseek-v4-flash,
thinking disabled, JSON mode, temperature 0, caps 160/160/140/160.
Agents see only query + answer (no benchmark metadata).

Aegis 2.0: 794/813 test samples already have frozen Exp3 T6 agent predictions
(identical prompts, same model; correction never fired). Those are reused and
marked; the remaining ~19 samples are run live.

Budget: shared BudgetState (hard cap 36, reserve 4) persisted to
experiments/exp2_prior_work_comparison/audit/budget_state.json.
Cache: outputs/api_cache_exp2. Checkpointing by id.

Usage:
  python scripts/run_exp2_teacher.py --pilot        # 25 rows/benchmark sanity
  python scripts/run_exp2_teacher.py                # full test manifest
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher  # noqa: E402
from frauddistill.providers.deepseek_client import DeepSeekClient  # noqa: E402
from frauddistill.runtime.budget import BudgetState  # noqa: E402
from frauddistill.runtime.cache import RequestCache  # noqa: E402
from frauddistill.teacher.evidence_table import build_evidence_table  # noqa: E402
from frauddistill.exp2_cross_benchmark.paths import (  # noqa: E402
    BUDGET_HARD_CAP_RMB,
    BUDGET_RESERVE_RMB,
    EXPERIMENT_DIR,
    EXP3_AGENT_DIR,
    MANIFEST_DIR,
    MODEL_TEACHER,
    T6_MAX_TOKENS,
    TEACHER_T6_DIR,
)

BUDGET_FILE = EXPERIMENT_DIR / "audit" / "budget_state.json"
CACHE_DIR = REPO / "outputs" / "api_cache_exp2"
ROW_CONCURRENCY = 60
CHECKPOINT_EVERY = 25
PILOT_PER_BENCHMARK = 25


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_existing(path: Path) -> dict[str, dict]:
    return {str(r["id"]): r for r in read_jsonl(path)}


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO)
        return out.stdout.strip()[:12] or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def prompt_digest() -> str:
    from frauddistill.agents import (  # noqa: F401
        fraud_assistance_agent as fa,
        refusal_quality_agent as rq,
        relevance_agent as rl,
        arbiter_agent as ar,
    )
    parts = [fa.SYSTEM_PROMPT, rq.SYSTEM_PROMPT, rl.SYSTEM_PROMPT, ar.SYSTEM_PROMPT]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def sample_for_agents(row: dict) -> dict:
    return {
        "id": row["sample_id"],
        "group_id": row["group_id"],
        "user_query": row["query"],
        "target_model_answer": row["answer"],
    }


def normalize_arbiter(arb: dict, row: dict, benchmark: str) -> dict:
    """Map an arbiter/signal dict into the exp2 prediction-row schema."""
    label = str(arb.get("teacher_label", "safe")).lower()
    rtype = str(arb.get("teacher_type", "safe")).lower()
    if rtype not in {"fraud_assistance", "refusal_failure", "over_refusal", "safe"}:
        rtype = "safe" if label == "safe" else "fraud_assistance"
    binary = 1 if label == "unsafe" else 0
    status = str(arb.get("status", "ok"))
    abstain = status == "parse_failed" or not arb.get("parsed")
    if "parsed" not in arb:
        # exp3 signal rows carry fields directly
        abstain = status == "parse_failed"
    score = float(arb.get("teacher_score", 0.0) or 0.0)
    return {
        "id": row["sample_id"],
        "benchmark": benchmark,
        "group_id": row["group_id"],
        "query": row["query"],
        "answer": row["answer"],
        "method": "frauddistill_evidence_mat_t6",
        "teacher_model": MODEL_TEACHER,
        "teacher_prompt_hash": prompt_digest(),
        "prediction_binary": binary,
        "prediction_type": rtype,
        "risk_score": score,
        "confidence": float(arb.get("confidence", 0.0) or 0.0),
        "abstain": abstain,
        "parse_status": status,
        "rationale": str(arb.get("rationale", "") or "")[:400],
        "decision_basis": arb.get("decision_basis") or [],
        "evidence_spans": (arb.get("unsafe_evidence_spans") or [])[:5],
        "overlap_exp3": bool(row.get("overlap_exp3")),
        "reused_from_exp3": bool(row.get("_reused")),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def build_reused_aegis() -> dict[str, dict]:
    """Reuse frozen Exp3 T6 predictions (signal == pre-correction arbiter)."""
    out: dict[str, dict] = {}
    for split in ("train", "dev", "test"):
        for r in read_jsonl(EXP3_AGENT_DIR / f"{split}.jsonl"):
            if not str(r["id"]).startswith("aegis_"):
                continue
            sig = r.get("signal") or {}
            out[str(r["id"])] = {
                "teacher_label": sig.get("teacher_label", "safe"),
                "teacher_type": sig.get("teacher_type", "safe"),
                "teacher_score": sig.get("teacher_score", 0.0),
                "confidence": sig.get("confidence", 0.0),
                "rationale": sig.get("rationale", ""),
                "decision_basis": sig.get("decision_basis") or [],
                "unsafe_evidence_spans": sig.get("unsafe_evidence_spans") or [],
                "status": sig.get("status", "ok"),
            }
    return out


async def process_row(teacher: MultiAgentTeacher, row: dict) -> dict:
    sample = sample_for_agents(row)
    env: dict[str, dict] = {}
    tasks = [
        ("fraud", teacher.fraud_agent.run_async(sample, teacher.client)),
        ("refusal", teacher.refusal_agent.run_async(sample, teacher.client)),
        ("context", teacher.context_agent.run_async(sample, teacher.client)),
    ]
    for name, task in tasks:
        env[name] = await task
    table = build_evidence_table(env.get("fraud"), env.get("refusal"), env.get("context"))
    arb = await teacher.arbiter_agent.run_async(sample, table, threshold=teacher.threshold, client=teacher.client)
    out = normalize_arbiter(arb, row, row["source"])
    out["evidence_table"] = table
    out["agent_fraud_json"] = env.get("fraud", {}).get("parsed")
    out["agent_refusal_json"] = env.get("refusal", {}).get("parsed")
    out["agent_context_json"] = env.get("context", {}).get("parsed")
    out["arbiter_json"] = arb.get("parsed")
    out["agent_agreement"] = arb.get("agent_agreement")
    out["contradiction_flags"] = arb.get("contradiction_flags")
    out["latency_ms"] = round(
        float(arb.get("latency_ms", 0))
        + sum(float(env[k].get("latency_ms", 0)) for k in ("fraud", "refusal", "context")),
        1,
    )
    out["input_tokens"] = sum(int((env[k].get("usage") or {}).get("input_miss", 0) + (env[k].get("usage") or {}).get("input_hit", 0)) for k in ("fraud", "refusal", "context"))
    out["output_tokens"] = sum(int((env[k].get("usage") or {}).get("output", 0)) for k in ("fraud", "refusal", "context"))
    return out


def build_teacher(client) -> MultiAgentTeacher:
    return MultiAgentTeacher(
        client,
        use_fraud_agent=True,
        use_refusal_agent=True,
        use_context_agent=True,
        arbiter_mode="evidence",
        use_correction=False,
        threshold=0.5,
        max_tokens=dict(T6_MAX_TOKENS),
    )


async def run_benchmark(benchmark: str, rows: list[dict], teacher: MultiAgentTeacher, existing: dict, out_path: Path, reused: dict[str, dict], limit: int = 0) -> dict:
    if limit:
        rows = rows[:limit]
    results: dict[str, dict] = dict(existing)
    reused_ids: list[str] = []
    todo: list[dict] = []
    for row in rows:
        rid = str(row["sample_id"])
        if rid in results:
            continue
        if rid in reused:
            rec = normalize_arbiter(reused[rid], row, benchmark)
            rec["_reused"] = True
            results[rid] = rec
            reused_ids.append(rid)
        else:
            todo.append(row)
    print(f"[{benchmark}] total={len(rows)} done={len(results) - len(reused_ids)} reused_exp3={len(reused_ids)} todo={len(todo)}")

    failures = 0
    sem = asyncio.Semaphore(ROW_CONCURRENCY)

    async def worker(row: dict):
        async with sem:
            try:
                return str(row["sample_id"]), await process_row(teacher, row), ""
            except Exception as exc:  # noqa: BLE001
                return str(row["sample_id"]), None, f"{type(exc).__name__}: {exc}"

    started = time.perf_counter()
    done_count = 0
    batch: list[asyncio.Task] = []
    for row in todo:
        batch.append(asyncio.create_task(worker(row)))
        if len(batch) >= ROW_CONCURRENCY:
            for rid, rec, err in await asyncio.gather(*batch):
                done_count += 1
                if rec is not None:
                    results[rid] = rec
                else:
                    failures += 1
                    print(f"[{benchmark}] FAILED {rid}: {err}")
            batch = []
            if done_count % CHECKPOINT_EVERY == 0:
                rate = done_count / max(time.perf_counter() - started, 1e-9)
                print(f"[{benchmark}] progress {done_count}/{len(todo)} rows/s={rate:.2f} used_rmb={teacher.client.budget.used_rmb:.4f} failures={failures}")
                write_rows(out_path, results, [str(r["sample_id"]) for r in rows])
    if batch:
        for rid, rec, err in await asyncio.gather(*batch):
            done_count += 1
            if rec is not None:
                results[rid] = rec
            else:
                failures += 1
                print(f"[{benchmark}] FAILED {rid}: {err}")
    write_rows(out_path, results, [str(r["sample_id"]) for r in rows])
    print(f"[{benchmark}] finished done={done_count} failures={failures} reused={len(reused_ids)} used_rmb={teacher.client.budget.used_rmb:.4f}")
    return results


def write_rows(out_path: Path, results: dict[str, dict], order: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rid in order:
            rec = results.get(rid)
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_budget(budget) -> None:
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_FILE.write_text(
        json.dumps(
            {"used_rmb": round(budget.used_rmb, 6), "cap_rmb": budget.effective_cap, "reserved_rmb": budget.reserved_rmb, "owner": "exp2_20260805"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_budget() -> BudgetState:
    budget = BudgetState(max_rmb=BUDGET_HARD_CAP_RMB, reserved_rmb=BUDGET_RESERVE_RMB)
    if BUDGET_FILE.exists():
        st = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
        budget.used_rmb = float(st.get("used_rmb", 0.0))
        print(f"[budget] resumed used_rmb={budget.used_rmb:.4f} cap={budget.effective_cap:.2f}")
    return budget


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="run 25 rows per benchmark sanity pilot only")
    ap.add_argument("--benchmark", choices=["fraudr1", "orbench", "do_not_answer", "aegis2"], default=None)
    ap.add_argument("--concurrency", type=int, default=120)
    args = ap.parse_args()

    manifest = read_jsonl(MANIFEST_DIR / "test_manifest.jsonl")
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for r in manifest:
        by_benchmark[r["source"]].append(r)

    budget = load_budget()
    cache = RequestCache(str(CACHE_DIR))
    client = DeepSeekClient(
        model=MODEL_TEACHER,
        json_mode=True,
        thinking_disabled=True,
        timeout_seconds=90,
        max_retries=1,
        cache=cache,
        budget=budget,
        concurrency=args.concurrency,
    )
    teacher = build_teacher(client)
    reused = build_reused_aegis() if args.benchmark in (None, "aegis2") else {}
    print(f"[teacher] commit={git_commit()} prompt_digest={prompt_digest()}")

    for b, rows in by_benchmark.items():
        if args.benchmark and b != args.benchmark:
            continue
        out_path = TEACHER_T6_DIR / f"{b}_t6_predictions.jsonl"
        existing = load_existing(out_path)
        asyncio.run(run_benchmark(b, rows, teacher, existing, out_path, reused, limit=PILOT_PER_BENCHMARK if args.pilot else 0))

    save_budget(budget)
    ledger = client.ledger.snapshot(client.prices)
    cost_dir = EXPERIMENT_DIR / "metrics"
    cost_dir.mkdir(parents=True, exist_ok=True)
    tag = "pilot" if args.pilot else "test"
    cost_file = cost_dir / f"cost_teacher_t6_{tag}.json"
    cost_file.write_text(
        json.dumps(
            {**ledger, "used_rmb": round(budget.used_rmb, 6), "cache": cache.stats(), "commit": git_commit(), "prompt_digest": prompt_digest(), "tag": tag},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[cost] used_rmb={budget.used_rmb:.4f} -> {cost_file}")


if __name__ == "__main__":
    main()
