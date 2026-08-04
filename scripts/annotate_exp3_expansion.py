# -*- coding: utf-8 -*-
"""Annotate the 4,000-row expansion pool with the frozen T6 teacher (guide 7.2/8.3).

- frozen config threshold (0.85); no correction (guide 8.3 cost-cut);
- dedicated 30-yuan-round budget: hard cap 27, reserved 3 (stop at 24);
- cache-aware, id-based resume, checkpoint every 25 rows;
- per-row output matches agent_predictions/*.jsonl schema plus confidence tier.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher
from frauddistill.providers.deepseek_client import DeepSeekClient
from frauddistill.runtime.budget import BudgetState
from frauddistill.runtime.cache import RequestCache
from frauddistill.teacher.conflict_detector import detect_conflicts
from frauddistill.teacher.evidence_table import build_evidence_table

POOL = REPO / "data/prepared/exp3_neural_student/expansion_pool.jsonl"
OUT = REPO / "data/prepared/exp3_neural_student/expansion_annotated.jsonl"
CONFIG = REPO / "configs/experiments/exp3_agent_distillation_ablation.yaml"
FROZEN = REPO / "experiments/exp3_agent_distillation_ablation/outputs/frozen_config.json"
CACHE_DIR = REPO / "outputs/api_cache_exp3"
BUDGET_FILE = CACHE_DIR / "budget_state_expansion.json"
COST_FILE = REPO / "experiments/exp3_agent_distillation_ablation/outputs/metrics/cost_expansion.json"
HARD_CAP_RMB = 27.0
RESERVED_RMB = 3.0
ROW_CONCURRENCY = 40


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_existing(path: Path) -> dict[str, dict]:
    out = {}
    for r in load_jsonl(path):
        out[str(r["id"])] = r
    return out


def tier_of(sig: dict, flags: list[str]) -> str:
    """Guide 7.4 tiers; mirrors scripts/audit_student_training_data.py."""
    conf = float(sig.get("teacher_confidence", sig.get("confidence", 0.0)))
    agree = float(sig.get("agent_agreement", 0.0))
    spans = bool((sig.get("unsafe_evidence_spans") or []) or (sig.get("safe_evidence_spans") or []))
    score = float(sig.get("teacher_score", 0.5))
    label = str(sig.get("teacher_label", "safe"))
    valid = not (label == "safe" and score > 0.8) and not (label == "unsafe" and score < 0.2) and bool(sig.get("decision_basis"))
    if valid and agree >= 0.75 and conf >= 0.80 and spans and not flags:
        return "high"
    if conf >= 0.60:
        return "medium"
    return "low"


def build_teacher(client, config: dict, threshold: float, token_caps: str = "cheap") -> MultiAgentTeacher:
    mt = config.get("agents", {})
    if token_caps == "cheap":
        # guide 8.3 cost-cut: agent outputs 120-180 tokens (prompts stay frozen)
        max_tokens = {"fraud": 180, "refusal": 180, "context": 150, "arbiter": 220,
                      "advocate": 160, "correction_arbiter": 200, "single_judge": 150}
    else:
        max_tokens = {
            "fraud": int(mt.get("fraud", {}).get("max_tokens", 400)),
            "refusal": int(mt.get("refusal", {}).get("max_tokens", 400)),
            "context": int(mt.get("context", {}).get("max_tokens", 340)),
            "arbiter": int(mt.get("arbiter", {}).get("max_tokens", 360)),
            "advocate": int(mt.get("advocate", {}).get("max_tokens", 260)),
            "correction_arbiter": int(mt.get("correction_arbiter", {}).get("max_tokens", 360)),
            "single_judge": int(mt.get("single_judge", {}).get("max_tokens", 200)),
        }
    return MultiAgentTeacher(
        client,
        use_fraud_agent=True,
        use_refusal_agent=True,
        use_context_agent=True,
        arbiter_mode="evidence",
        use_correction=False,          # guide 8.3: no correction for expansion
        threshold=threshold,
        max_tokens=max_tokens,
    )


async def process_row(teacher: MultiAgentTeacher, sample: dict) -> dict:
    sig = await teacher.run_async(sample)
    flags = list((sig.get("conflict_flags") or []) + (sig.get("contradiction_flags") or []))
    return {
        "id": str(sample["id"]),
        "group_id": str(sample.get("group_id", "")),
        "split": str(sample.get("split", "train")),
        "sample": {
            "user_query": sample["user_query"],
            "target_model_answer": sample["target_model_answer"],
            "gold_label": sample.get("gold_label", "safe"),
            "gold_type": sample.get("gold_type", sample.get("gold_label", "safe")),
            "subtype": sample.get("subtype", "general"),
            "language": sample.get("language", ""),
            "target_model": sample.get("target_model", ""),
            "pair_id": sample.get("pair_id"),
            "block": sample.get("block", ""),
        },
        "signal": sig,
        "evidence_table": sig.get("raw_agent_outputs", {}).get("evidence_table"),
        "conflict_flags": flags,
        "confidence_tier": tier_of(sig, flags),
        "latency_ms": round(float(sig.get("latency_ms", 0)), 1),
    }


def write_results(path: Path, results: dict[str, dict], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rid in order:
            if rid in results:
                f.write(json.dumps(results[rid], ensure_ascii=False) + "\n")


def save_budget(budget) -> None:
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_FILE.write_text(json.dumps({
        "used_rmb": round(budget.used_rmb, 6),
        "cap_rmb": round(budget.effective_cap, 4),
        "reserved_rmb": budget.reserved_rmb,
    }), encoding="utf-8")


async def run_pool(rows, client, config, existing, threshold, token_caps="cheap", checkpoint_every=25, out_path=None):
    teacher = build_teacher(client, config, threshold, token_caps=token_caps)
    todo = [r for r in rows if str(r["id"]) not in existing]
    print(f"[expansion] total={len(rows)} done={len(rows) - len(todo)} todo={len(todo)}")
    results: dict[str, dict] = dict(existing)
    sem = asyncio.Semaphore(ROW_CONCURRENCY)
    failures = 0

    async def worker(sample):
        async with sem:
            try:
                rec = await process_row(teacher, sample)
                return str(sample["id"]), rec, ""
            except Exception as exc:  # noqa: BLE001
                return str(sample["id"]), None, repr(exc)

    done = 0
    for i in range(0, len(todo), ROW_CONCURRENCY):
        batch = todo[i:i + ROW_CONCURRENCY]
        for rid, rec, err in await asyncio.gather(*(worker(r) for r in batch)):
            if rec is None:
                failures += 1
                print(f"  FAIL {rid}: {err[:160]}")
                continue
            results[rid] = rec
            done += 1
        if done and done % checkpoint_every == 0:
            if out_path:
                write_results(out_path, results, [str(r["id"]) for r in rows])
            save_budget(client.budget)
            print(f"[expansion] {done}/{len(todo)} done used_rmb={client.budget.used_rmb:.4f} failures={failures}")
    if out_path:
        write_results(out_path, results, [str(r["id"]) for r in rows])
    save_budget(client.budget)
    print(f"[expansion] finished done={done} failures={failures} used_rmb={client.budget.used_rmb:.4f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--limit", type=int, default=0, help="debug: annotate first N rows only")
    ap.add_argument("--concurrency", type=int, default=120)
    ap.add_argument("--timeout", type=float, default=300.0, help="per-request timeout seconds")
    ap.add_argument("--token-caps", choices=["cheap", "frozen"], default="cheap",
                    help="guide 8.3: cheap = 120-180 token agent outputs; frozen = config values")
    ap.add_argument("--no-frozen", action="store_true", help="use 0.5 threshold instead of frozen 0.85")
    args = ap.parse_args()

    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    threshold = 0.85
    if FROZEN.exists():
        threshold = float(json.loads(FROZEN.read_text(encoding="utf-8")).get("threshold", threshold))
    if args.no_frozen:
        threshold = 0.5
    print(f"[config] threshold={threshold} frozen={FROZEN.exists()}")

    budget = BudgetState(max_rmb=HARD_CAP_RMB, reserved_rmb=RESERVED_RMB)
    if BUDGET_FILE.exists():
        st = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
        budget.used_rmb = float(st.get("used_rmb", 0.0))
        print(f"[budget] resumed used_rmb={budget.used_rmb:.4f} effective_cap={budget.effective_cap:.4f}")

    cache = RequestCache(str(CACHE_DIR))
    client = DeepSeekClient(
        model=config["provider"]["model"],
        json_mode=True,
        thinking_disabled=True,
        timeout_seconds=args.timeout,
        max_retries=int(config["provider"].get("max_retries", 1)),
        cache=cache,
        budget=budget,
        concurrency=args.concurrency,
    )

    rows = load_jsonl(POOL)
    if args.limit:
        rows = rows[: args.limit]
    existing = load_existing(OUT)
    results = asyncio.run(run_pool(rows, client, config, existing, threshold,
                                  token_caps=args.token_caps, out_path=OUT))
    print(f"[expansion] wrote {OUT} rows={len(results)}")

    ledger = client.ledger.snapshot(client.prices)
    COST_FILE.parent.mkdir(parents=True, exist_ok=True)
    COST_FILE.write_text(json.dumps({
        **ledger, "used_rmb": round(budget.used_rmb, 6),
        "token_caps": args.token_caps,
        "threshold": threshold, "rows": len(rows), "rows_done": len(results),
        "cache": cache.stats(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cost] used_rmb={budget.used_rmb:.4f} -> {COST_FILE}")


if __name__ == "__main__":
    main()
