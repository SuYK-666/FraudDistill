# -*- coding: utf-8 -*-
"""Run the frozen T6 Evidence MAT teacher on the Exp2 FULL coverage manifest.

Full-coverage guide 2026-08-06 (sections 2, 3, 12, 20, 21, 22, 25):
  - fraudr1 8,564 (missing 7,764) / orbench 3,000 (missing 2,200)
  - do_not_answer 5,634 (missing 4,734)
  - aegis2 1,964: response-level 813 (reused), prompt-only 1,151 (new)
Frozen method: Fraud + Refusal + Context specialists + Evidence Arbiter,
conflict correction OFF, factuality OFF, deepseek-v4-flash, thinking disabled,
JSON mode, temperature 0 (caps identical to the 2026-08-05 frozen run so that
the 3,313 existing predictions stay valid cache for the same q+y).
Agents see only query + answer (no benchmark metadata, guide 6.2).

Aegis prompt-only (guide 3.2): specialists evaluate the INPUT risk fields only
(prompt_harm_intent / should_refuse_probability / hazard domain); no answer-
unsafe verdict, no mixing with response-level Macro-F1.

Budget (guide 20): hard stop 96 RMB, 4 RMB emergency reserve (<=100 total),
monitoring milestones 70 / 88 / 94. Concurrency 120 rows (user setting).

Usage:
  python scripts/run_exp2_teacher.py --pilot                # 20 new rows/source
  python scripts/run_exp2_teacher.py --benchmark fraudr1    # one source
  python scripts/run_exp2_teacher.py --source aegis2 --mode prompt
  python scripts/run_exp2_teacher.py --calib-aegis 300      # official validation split
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
from frauddistill.runtime.budget import BudgetExceeded, BudgetState  # noqa: E402
from frauddistill.runtime.cache import RequestCache  # noqa: E402
from frauddistill.teacher.evidence_table import build_evidence_table  # noqa: E402
from frauddistill.exp2_cross_benchmark.paths import (  # noqa: E402
    AUDIT_DIR,
    BUDGET_FILE,
    BUDGET_HARD_CAP_RMB,
    BUDGET_HISTORY_FILE,
    BUDGET_MILESTONES,
    EXPERIMENT_DIR,
    EXP3_AGENT_DIR,
    FULL_MANIFEST,
    MANIFEST_DIR,
    MODEL_TEACHER,
    PROMPT_RISK_THRESHOLD,
    T6_MAX_TOKENS,
    TEACHER_T6_DIR,
)
from frauddistill.exp2_cross_benchmark.paths import RAW_AEGIS  # noqa: E402

CACHE_DIR = REPO / "outputs" / "api_cache_exp2"
ROW_CONCURRENCY = 120          # user setting: concurrency 120
CHECKPOINT_EVERY = 50
PILOT_PER_SOURCE = 20          # guide 22 Phase 1: 20 new rows per source
AEGIS_VALIDATION_MANIFEST = MANIFEST_DIR / "aegis_validation_manifest.jsonl"
AEGIS_VALIDATION_PRED = TEACHER_T6_DIR / "aegis_validation_t6_predictions.jsonl"
PROMPT_OUT_FILE = TEACHER_T6_DIR / "aegis2_t6_prompt_predictions.jsonl"
PREV_ROUND_RMB = 16.8076       # 2026-08-05 sample round (archived budget_state)


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


def sample_for_agents(row: dict, *, prompt_only: bool = False) -> dict:
    return {
        "id": row["sample_id"],
        "group_id": row["group_id"],
        "user_query": row["query"],
        "target_model_answer": "" if prompt_only else row["answer"],
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
        "evaluation_mode": row.get("evaluation_mode", "response"),
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
    arb_usage = arb.get("usage") or {}
    out["input_tokens"] = sum(int((env[k].get("usage") or {}).get("input_miss", 0) + (env[k].get("usage") or {}).get("input_hit", 0)) for k in ("fraud", "refusal", "context")) + int(arb_usage.get("input_miss", 0) + arb_usage.get("input_hit", 0))
    out["output_tokens"] = sum(int((env[k].get("usage") or {}).get("output", 0)) for k in ("fraud", "refusal", "context")) + int(arb_usage.get("output", 0))
    return out


async def process_row_prompt_only(teacher: MultiAgentTeacher, row: dict) -> dict:
    """Aegis prompt-only auxiliary track (guide 3.2): input-risk fields only.

    Runs the three specialists on the prompt with an empty answer; records
    prompt_harm_intent / should_refuse_probability / hazard domain. No arbiter
    verdict (no answer-unsafe claim) and no mixing with response-level metrics.
    """
    sample = sample_for_agents(row, prompt_only=True)
    env: dict[str, dict] = {}
    tasks = [
        ("fraud", teacher.fraud_agent.run_async(sample, teacher.client)),
        ("refusal", teacher.refusal_agent.run_async(sample, teacher.client)),
        ("context", teacher.context_agent.run_async(sample, teacher.client)),
    ]
    for name, task in tasks:
        env[name] = await task
    fraud_p = env.get("fraud", {}).get("parsed") or {}
    refusal_p = env.get("refusal", {}).get("parsed") or {}
    prompt_harm_intent = float(fraud_p.get("fraud_probability", 0.0) or 0.0)
    should_refuse = float(refusal_p.get("should_refuse_probability", 0.0) or 0.0)
    hazard_domain = str(fraud_p.get("fraud_domain", "none") or "none")
    risk_score = max(prompt_harm_intent, should_refuse)
    binary = 1 if risk_score >= PROMPT_RISK_THRESHOLD else 0
    status = "ok"
    abstain = False
    if not fraud_p and not refusal_p:
        status = "parse_failed"
        abstain = True
    out = {
        "id": row["sample_id"],
        "benchmark": row["source"],
        "group_id": row["group_id"],
        "query": row["query"],
        "answer": "",
        "method": "frauddistill_evidence_mat_t6_prompt_only",
        "teacher_model": MODEL_TEACHER,
        "teacher_prompt_hash": prompt_digest(),
        "prediction_binary": binary,
        "prediction_type": "unsafe_prompt" if binary else "safe_prompt",
        "risk_score": round(risk_score, 4),
        "prompt_harm_intent": round(prompt_harm_intent, 4),
        "should_refuse_probability": round(should_refuse, 4),
        "hazard_domain": hazard_domain,
        "abstain": abstain,
        "parse_status": status,
        "evaluation_mode": "prompt_only",
        "overlap_exp3": bool(row.get("overlap_exp3")),
        "evidence_table": None,
        "agent_fraud_json": fraud_p,
        "agent_refusal_json": refusal_p,
        "agent_context_json": env.get("context", {}).get("parsed"),
        "latency_ms": round(sum(float(env[k].get("latency_ms", 0)) for k in ("fraud", "refusal", "context")), 1),
        "input_tokens": sum(int((env[k].get("usage") or {}).get("input_miss", 0) + (env[k].get("usage") or {}).get("input_hit", 0)) for k in ("fraud", "refusal", "context")),
        "output_tokens": sum(int((env[k].get("usage") or {}).get("output", 0)) for k in ("fraud", "refusal", "context")),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
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


def append_rows(out_path: Path, records: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_rows(out_path: Path, results: dict[str, dict], order: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rid in order:
            rec = results.get(rid)
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_budget() -> BudgetState:
    """Fresh round ledger: hard stop 96 RMB, reserve 4 RMB is the margin to the
    user's 100 RMB ceiling (guide 20). Previous round is recorded for reporting."""
    budget = BudgetState(max_rmb=BUDGET_HARD_CAP_RMB, reserved_rmb=0.0, used_rmb=0.0)
    if BUDGET_FILE.exists():
        st = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
        budget.used_rmb = float(st.get("used_rmb", 0.0))
        print(f"[budget] resumed used_rmb={budget.used_rmb:.4f} cap={budget.effective_cap:.2f}")
    return budget


def save_budget(budget, *, tag: str) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BUDGET_FILE.write_text(
        json.dumps(
            {
                "used_rmb": round(budget.used_rmb, 6),
                "cap_rmb": budget.effective_cap,
                "reserve_rmb": 4.0,
                "prev_round_rmb": PREV_ROUND_RMB,
                "cumulative_rmb": round(budget.used_rmb + PREV_ROUND_RMB, 6),
                "owner": "exp2_full_20260806",
                "tag": tag,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def log_budget_history(budget, tag: str) -> None:
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": tag,
        "used_rmb": round(budget.used_rmb, 6),
        "cumulative_rmb": round(budget.used_rmb + PREV_ROUND_RMB, 6),
    }
    history = []
    if BUDGET_HISTORY_FILE.exists():
        history = json.loads(BUDGET_HISTORY_FILE.read_text(encoding="utf-8"))
    history.append(rec)
    BUDGET_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")


class MilestoneMonitor:
    def __init__(self, used_rmb: float):
        self.reached = {m: False for m in BUDGET_MILESTONES}
        for m in BUDGET_MILESTONES:
            if used_rmb >= m:
                self.reached[m] = True

    def check(self, used_rmb: float) -> None:
        for m in BUDGET_MILESTONES:
            if not self.reached[m] and used_rmb >= m:
                self.reached[m] = True
                print(f"[budget] *** MILESTONE {m:.0f} RMB reached (used={used_rmb:.2f}) ***")


async def run_benchmark(benchmark: str, rows: list[dict], teacher: MultiAgentTeacher,
                        existing: dict, out_path: Path, reused: dict[str, dict],
                        limit: int = 0, prompt_only: bool = False) -> dict:
    if limit:
        # pilot: first N NEW rows of this source/mode (guide 22 Phase 1)
        rows = [r for r in rows if str(r["sample_id"]) not in existing and str(r["sample_id"]) not in reused][:limit]
    results: dict[str, dict] = dict(existing)
    reused_ids: list[str] = []
    todo: list[dict] = []
    for row in rows:
        rid = str(row["sample_id"])
        if rid in results:
            continue
        if (not prompt_only) and rid in reused:
            rec = normalize_arbiter(reused[rid], row, benchmark)
            rec["_reused"] = True
            results[rid] = rec
            reused_ids.append(rid)
        else:
            todo.append(row)
    print(f"[{benchmark}] total={len(rows)} done={len(results) - len(reused_ids)} reused_exp3={len(reused_ids)} todo={len(todo)} prompt_only={prompt_only}")

    failures = 0
    parse_failed = 0  # 2026-08-06: visible parse-failure counter (guide Phase 3)
    sem = asyncio.Semaphore(ROW_CONCURRENCY)

    async def worker(row: dict):
        async with sem:
            try:
                if prompt_only:
                    return str(row["sample_id"]), await process_row_prompt_only(teacher, row), ""
                return str(row["sample_id"]), await process_row(teacher, row), ""
            except BudgetExceeded as exc:
                return str(row["sample_id"]), None, f"BUDGET_EXCEEDED: {exc}"
            except Exception as exc:  # noqa: BLE001
                return str(row["sample_id"]), None, f"{type(exc).__name__}: {exc}"

    started = time.perf_counter()
    done_count = 0
    batch: list[asyncio.Task] = []
    budget_stop = False
    for row in todo:
        if not teacher.client.budget.can_spend(0.05):
            print(f"[{benchmark}] BUDGET STOP: used={teacher.client.budget.used_rmb:.2f} cap={teacher.client.budget.effective_cap:.2f}; {len(todo) - done_count} rows left")
            budget_stop = True
            break
        batch.append(asyncio.create_task(worker(row)))
        if len(batch) >= ROW_CONCURRENCY:
            for rid, rec, err in await asyncio.gather(*batch):
                done_count += 1
                if rec is not None:
                    results[rid] = rec
                    append_rows(out_path, [rec])
                    if rec.get("abstain") or rec.get("parse_status") == "parse_failed":
                        parse_failed += 1
                else:
                    failures += 1
                    if err.startswith("BUDGET_EXCEEDED"):
                        budget_stop = True
                    print(f"[{benchmark}] FAILED {rid}: {err}")
            batch = []
            if done_count % CHECKPOINT_EVERY == 0:
                rate = done_count / max(time.perf_counter() - started, 1e-9)
                used = teacher.client.budget.used_rmb
                print(f"[{benchmark}] progress {done_count}/{len(todo)} rows/s={rate:.2f} used_rmb={used:.4f} failures={failures} parse_failed={parse_failed}")
                save_budget(teacher.client.budget, tag=f"progress_{benchmark}")
    if batch:
        for rid, rec, err in await asyncio.gather(*batch):
            done_count += 1
            if rec is not None:
                results[rid] = rec
                append_rows(out_path, [rec])
                if rec.get("abstain") or rec.get("parse_status") == "parse_failed":
                    parse_failed += 1
            else:
                failures += 1
                if err.startswith("BUDGET_EXCEEDED"):
                    budget_stop = True
                print(f"[{benchmark}] FAILED {rid}: {err}")
    if limit == 0:
        # full run: rewrite in canonical manifest order. Pilot runs append only
        # (existing predictions in the file must stay untouched).
        write_rows(out_path, results, [str(r["sample_id"]) for r in rows])
    print(f"[{benchmark}] finished done={done_count} failures={failures} parse_failed={parse_failed} reused={len(reused_ids)} used_rmb={teacher.client.budget.used_rmb:.4f} budget_stop={budget_stop}")
    return results


def build_aegis_validation_manifest(limit: int | None = None) -> None:
    """Official Aegis validation split -> calibration manifest (guide 14:
    thresholds from the official validation split, never from the new full test)."""
    raw = json.loads((RAW_AEGIS / "validation.json").read_text(encoding="utf-8"))
    rows = []
    for r in raw:
        resp = (r.get("response") or "").strip()
        rl = r.get("response_label")
        if not resp or rl in (None, ""):
            continue
        rows.append({
            "sample_id": f"aegis_{r['id']}",
            "source": "aegis2",
            "group_id": f"aegis_{r['id']}",
            "query": str(r.get("prompt", "") or ""),
            "answer": resp,
            "language": "English",
            "official_response_label": 1 if str(rl).lower().startswith("unsafe") else 0,
            "official_prompt_label": None,
            "official_category": str(r.get("violated_categories", "") or ""),
            "target_model": "unknown",
            "evaluation_mode": "response",
            "split": "validation",
        })
    if limit:
        rows = rows[:limit]
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with AEGIS_VALIDATION_MANIFEST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[calib] aegis validation manifest: {len(rows)} response-level rows -> {AEGIS_VALIDATION_MANIFEST}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="20 new rows per source/mode (guide Phase 1)")
    ap.add_argument("--benchmark", choices=["fraudr1", "orbench", "do_not_answer", "aegis2"], default=None)
    ap.add_argument("--mode", choices=["response", "prompt"], default=None, help="aegis2 evaluation mode")
    ap.add_argument("--calib-aegis", type=int, default=0, help="run T6 on N official Aegis validation rows (calibration)")
    ap.add_argument("--concurrency", type=int, default=120)
    args = ap.parse_args()

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
    monitor = MilestoneMonitor(budget.used_rmb)

    if args.calib_aegis:
        build_aegis_validation_manifest(args.calib_aegis)
        rows = read_jsonl(AEGIS_VALIDATION_MANIFEST)
        out_path = AEGIS_VALIDATION_PRED
        existing = load_existing(out_path)
        asyncio.run(run_benchmark("aegis2", rows, teacher, existing, out_path, {}, limit=0))
        save_budget(budget, tag="calib_aegis")
        log_budget_history(budget, "calib_aegis")
        print(f"[cost] calibration used_rmb={budget.used_rmb:.4f}")
        return

    manifest = read_jsonl(FULL_MANIFEST) if FULL_MANIFEST.exists() else read_jsonl(MANIFEST_DIR / "test_manifest.jsonl")
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for r in manifest:
        by_benchmark[r["source"]].append(r)

    reused = build_reused_aegis()
    print(f"[teacher] commit={git_commit()} prompt_digest={prompt_digest()}")

    for b, rows in by_benchmark.items():
        if args.benchmark and b != args.benchmark:
            continue
        if b == "aegis2":
            resp_rows = [r for r in rows if r["evaluation_mode"] == "response"]
            prompt_rows = [r for r in rows if r["evaluation_mode"] == "prompt_only"]
            if args.mode in (None, "response") and resp_rows:
                asyncio.run(run_benchmark(b, resp_rows, teacher, load_existing(TEACHER_T6_DIR / f"{b}_t6_predictions.jsonl"),
                                          TEACHER_T6_DIR / f"{b}_t6_predictions.jsonl", reused,
                                          limit=PILOT_PER_SOURCE if args.pilot else 0, prompt_only=False))
            if args.mode in (None, "prompt") and prompt_rows:
                asyncio.run(run_benchmark(b, prompt_rows, teacher, load_existing(PROMPT_OUT_FILE),
                                          PROMPT_OUT_FILE, {},
                                          limit=PILOT_PER_SOURCE if args.pilot else 0, prompt_only=True))
        else:
            out_path = TEACHER_T6_DIR / f"{b}_t6_predictions.jsonl"
            asyncio.run(run_benchmark(b, rows, teacher, load_existing(out_path), out_path, reused,
                                      limit=PILOT_PER_SOURCE if args.pilot else 0))
        monitor.check(budget.used_rmb)
        if not budget.can_spend(0.05):
            print(f"[budget] stopping: used={budget.used_rmb:.2f} cap={budget.effective_cap:.2f}")
            break

    save_budget(budget, tag="pilot" if args.pilot else "full")
    log_budget_history(budget, "pilot" if args.pilot else "full")
    ledger = client.ledger.snapshot(client.prices)
    cost_dir = EXPERIMENT_DIR / "metrics"
    cost_dir.mkdir(parents=True, exist_ok=True)
    tag = "pilot" if args.pilot else "full"
    cost_file = cost_dir / f"cost_teacher_t6_{tag}.json"
    cost_file.write_text(
        json.dumps(
            {**ledger, "used_rmb": round(budget.used_rmb, 6),
             "prev_round_rmb": PREV_ROUND_RMB,
             "cumulative_rmb": round(budget.used_rmb + PREV_ROUND_RMB, 6),
             "cap_rmb": budget.effective_cap,
             "cache": cache.stats(), "commit": git_commit(), "prompt_digest": prompt_digest(), "tag": tag},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[cost] used_rmb={budget.used_rmb:.4f} cumulative={budget.used_rmb + PREV_ROUND_RMB:.4f} -> {cost_file}")


if __name__ == "__main__":
    main()
