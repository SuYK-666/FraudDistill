# -*- coding: utf-8 -*-
"""Run the enhanced MAT teacher for Exp3 (guide 15/22/23).

Modes:
  pilot  - stratified 400-row subset of dev (pilot belongs to dev, guide 14.4)
  dev    - full dev split (1047 rows; pilot rows resume from checkpoint)
  test   - held-out test split (1262 rows), run once with frozen config
  judge  - T1 Single DeepSeek Judge on dev+test (2304 rows)

Per-row pipeline: 3 specialists in parallel -> evidence table -> evidence
arbiter -> conflict detection -> optional correction -> unified TeacherSignal.
Checkpointing: rows already present in the output jsonl are skipped.
Budget: persisted BudgetState (hard cap 36 RMB effective; correction stops at 34).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher
from frauddistill.providers.deepseek_client import DeepSeekClient
from frauddistill.runtime.budget import BudgetState
from frauddistill.runtime.cache import RequestCache
from frauddistill.teacher.conflict_detector import detect_conflicts
from frauddistill.teacher.evidence_table import build_evidence_table

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data/prepared/exp3_agent_distillation/exp3_dataset.jsonl"
OUT_ROOT = Path(os.environ.get("EXP3_OUT_ROOT") or REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs")
AGENT_DIR = OUT_ROOT / "agent_predictions"
JUDGE_DIR = OUT_ROOT / "judge_predictions"
CACHE_DIR = REPO / "outputs" / "api_cache_exp3"
BUDGET_FILE = CACHE_DIR / "budget_state.json"
FROZEN_FILE = OUT_ROOT / "frozen_config.json"

DEFAULT_THRESHOLD = 0.5
CORRECTION_STOP_RMB = 34.0
HARD_CAP_RMB = 40.0
RESERVED_RMB = 0.0

PILOT_SUBTYPES = [
    "direct_fraud", "trust_facilitation", "clean_refusal_to_fraud",
    "partial_leakage", "over_refusal", "hard_safe", "quotation_analysis",
    "toxic", "context_flip", "regular_safe",
]
PILOT_PER_SUBTYPE = 40


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[str(r["id"])] = r
    return out


def select_pilot(dev_rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_subtype: dict[str, list[dict]] = defaultdict(list)
    for r in dev_rows:
        by_subtype[r["subtype"]].append(r)
    picked: list[dict] = []
    for st in PILOT_SUBTYPES:
        rows = by_subtype.get(st, [])
        rng.shuffle(rows)
        picked.extend(rows[:PILOT_PER_SUBTYPE])
    # top up from any remaining dev rows until 400
    rest = [r for r in dev_rows if r not in picked]
    rng.shuffle(rest)
    picked.extend(rest[: max(0, 400 - len(picked))])
    return picked[:400]


def load_frozen() -> dict:
    if FROZEN_FILE.exists():
        return json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
    return {}


def build_teacher(client, config: dict, threshold: float) -> MultiAgentTeacher:
    mt = config.get("agents", {})
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
        use_correction=True,
        threshold=threshold,
        max_tokens=max_tokens,
    )


async def process_row(teacher: MultiAgentTeacher, sample: dict, budget) -> dict:
    env = {}
    tasks = []
    for name, agent in (
        ("fraud", teacher.fraud_agent),
        ("refusal", teacher.refusal_agent),
        ("context", teacher.context_agent),
    ):
        tasks.append((name, agent.run_async(sample, teacher.client)))
    for name, task in tasks:
        env[name] = await task

    table = build_evidence_table(env.get("fraud"), env.get("refusal"), env.get("context"))
    arb_raw = await teacher.arbiter_agent.run_async(sample, table, threshold=teacher.threshold, client=teacher.client)
    arb_pre = dict(arb_raw)
    conflict_flags = detect_conflicts(table, float(arb_raw.get("teacher_score", 0.5)))

    correction = None
    correction_env = None
    correction_skipped = False
    do_correction = (
        teacher.use_correction
        and conflict_flags
        and teacher.client is not None
        and (budget.used_rmb < CORRECTION_STOP_RMB or os.environ.get("EXP3_FORCE_CORRECTION") == "1")
    )
    if do_correction:
        unsafe_env, safe_env = await asyncio.gather(
            teacher.unsafe_advocate.run_async(sample, table),
            teacher.safe_advocate.run_async(sample, table),
        )
        correction_env = {"unsafe_advocate": unsafe_env, "safe_advocate": safe_env}
        corr = await teacher.correction_arbiter.run_async(sample, table, unsafe_env, safe_env)
        if corr.get("status") == "ok" and corr.get("parsed"):
            correction = corr.get("parsed")
            arb_pre = teacher._merge_correction(arb_pre, corr, conflict_flags)
    elif teacher.use_correction and conflict_flags and teacher.client is not None:
        correction_skipped = True

    # Hard constraints outrank correction: re-apply them after any correction merge.
    if correction is not None:
        constrained, hard_overrides = teacher.arbiter_agent.apply_hard_constraints(arb_pre, table, teacher.threshold)
        for key in ("teacher_label", "teacher_score", "teacher_type"):
            arb_pre[key] = constrained.get(key, arb_pre.get(key))
        if hard_overrides:
            arb_pre["contradiction_flags"] = list(
                dict.fromkeys((arb_pre.get("contradiction_flags") or []) + hard_overrides)
            )

    arb_pre["contradiction_flags"] = list(
        dict.fromkeys((arb_pre.get("contradiction_flags") or []) + conflict_flags)
    )
    arb_pre["correction_used"] = correction is not None
    arb_pre["correction"] = correction
    arb_pre["conflict_flags"] = conflict_flags
    arb_pre["correction_skipped"] = correction_skipped

    return {
        "id": str(sample["id"]),
        "group_id": str(sample.get("group_id", "")),
        "split": str(sample.get("split", "")),
        "sample": {
            "user_query": sample["user_query"],
            "target_model_answer": sample["target_model_answer"],
            "gold_label": sample["gold_label"],
            "gold_type": sample.get("gold_type", sample["gold_label"]),
            "subtype": sample.get("subtype", "general"),
            "block": sample.get("block", ""),
            "language": sample.get("language", ""),
            "target_model": sample.get("target_model", ""),
            "pair_id": sample.get("pair_id"),
        },
        "fraud": env.get("fraud"),
        "refusal": env.get("refusal"),
        "context": env.get("context"),
        "evidence_table": table,
        "arbiter_pre_correction": arb_pre,
        "signal": arb_pre,
        "conflict_flags": conflict_flags,
        "correction": correction_env,
        "latency_ms": round(
            float(arb_pre.get("latency_ms", 0))
            + sum(float(env[k].get("latency_ms", 0)) for k in ("fraud", "refusal", "context") if env.get(k)),
            1,
        ),
    }


ROW_CONCURRENCY = 40  # rows in flight; API calls are bounded by the client semaphore (120)


async def run_split(mode: str, rows: list[dict], client, config: dict, existing: dict, threshold: float, mark_pilot: set[str], checkpoint_every: int = 25, out_path: Path | None = None) -> dict:
    teacher = build_teacher(client, config, threshold)
    if mode == "train" and os.environ.get("EXP3_TRAIN_NO_CORRECTION") == "1":
        teacher.use_correction = False
    todo = [r for r in rows if str(r["id"]) not in existing]
    print(f"[{mode}] total={len(rows)} done={len(rows) - len(todo)} todo={len(todo)}")
    results: dict[str, dict] = dict(existing)
    failures = 0
    sem = asyncio.Semaphore(ROW_CONCURRENCY)

    async def worker(sample: dict) -> tuple[str, dict | None, str]:
        async with sem:
            try:
                rec = await process_row(teacher, sample, client.budget)
                rec["is_pilot"] = str(sample["id"]) in mark_pilot
                return str(sample["id"]), rec, ""
            except Exception as exc:  # noqa: BLE001
                return str(sample["id"]), None, f"{type(exc).__name__}: {exc}"

    started = __import__("time").perf_counter()
    batch: list[asyncio.Task] = []
    done_count = 0
    for sample in todo:
        batch.append(asyncio.create_task(worker(sample)))
        if len(batch) >= ROW_CONCURRENCY:
            for rid, rec, err in await asyncio.gather(*batch):
                done_count += 1
                if rec is not None:
                    results[rid] = rec
                else:
                    failures += 1
                    print(f"[{mode}] FAILED {rid}: {err}")
            batch = []
            if done_count % checkpoint_every == 0:
                el = __import__("time").perf_counter() - started
                rate = done_count / max(el, 1e-9)
                print(f"[{mode}] progress {done_count}/{len(todo)} rows/s={rate:.2f} used_rmb={client.budget.used_rmb:.4f} failures={failures}")
                if out_path is not None:
                    write_rows(out_path, results, [str(r["id"]) for r in rows])
                    save_budget(client.budget)
    if batch:
        for rid, rec, err in await asyncio.gather(*batch):
            done_count += 1
            if rec is not None:
                results[rid] = rec
            else:
                failures += 1
                print(f"[{mode}] FAILED {rid}: {err}")
    print(f"[{mode}] finished done={done_count} failures={failures} used_rmb={client.budget.used_rmb:.4f}")
    return results


async def run_judge(mode: str, rows: list[dict], client, config: dict, existing: dict, out_path: Path | None = None) -> dict:
    mt = config.get("agents", {})
    max_tokens = int(mt.get("single_judge", {}).get("max_tokens", 200))
    teacher = MultiAgentTeacher(client, use_fraud_agent=False, use_refusal_agent=False, use_context_agent=False, use_correction=False)
    judge = teacher.single_judge
    judge.max_tokens = max_tokens
    todo = [r for r in rows if str(r["id"]) not in existing]
    print(f"[judge:{mode}] total={len(rows)} done={len(rows) - len(todo)} todo={len(todo)}")
    results: dict[str, dict] = dict(existing)
    failures = 0
    sem = asyncio.Semaphore(ROW_CONCURRENCY)

    async def worker(sample: dict) -> tuple[str, dict | None, str]:
        async with sem:
            try:
                env = await judge.run_async(sample, client)
                return str(sample["id"]), {
                    "id": str(sample["id"]),
                    "group_id": str(sample.get("group_id", "")),
                    "split": mode,
                    "gold_label": sample["gold_label"],
                    "subtype": sample.get("subtype", "general"),
                    **env,
                }, ""
            except Exception as exc:  # noqa: BLE001
                return str(sample["id"]), None, f"{type(exc).__name__}: {exc}"

    started = __import__("time").perf_counter()
    batch: list[asyncio.Task] = []
    done_count = 0
    for sample in todo:
        batch.append(asyncio.create_task(worker(sample)))
        if len(batch) >= ROW_CONCURRENCY:
            for rid, rec, err in await asyncio.gather(*batch):
                done_count += 1
                if rec is not None:
                    results[rid] = rec
                else:
                    failures += 1
                    print(f"[judge:{mode}] FAILED {rid}: {err}")
            batch = []
            if done_count % 50 == 0:
                el = __import__("time").perf_counter() - started
                print(f"[judge:{mode}] progress {done_count}/{len(todo)} rows/s={done_count / max(el, 1e-9):.2f} used_rmb={client.budget.used_rmb:.4f}")
                if out_path is not None:
                    write_rows(out_path, results, [str(r["id"]) for r in rows])
    if batch:
        for rid, rec, err in await asyncio.gather(*batch):
            done_count += 1
            if rec is not None:
                results[rid] = rec
            else:
                failures += 1
                print(f"[judge:{mode}] FAILED {rid}: {err}")
    print(f"[judge:{mode}] finished done={done_count} failures={failures}")
    return results


def write_rows(path: Path, results: dict[str, dict], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rid in order:
            if rid in results:
                f.write(json.dumps(results[rid], ensure_ascii=False) + "\n")


def save_budget(budget) -> None:
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_FILE.write_text(
        json.dumps({"used_rmb": round(budget.used_rmb, 6), "cap_rmb": budget.effective_cap, "reserved_rmb": RESERVED_RMB}, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "train", "dev", "test", "judge"], required=True)
    ap.add_argument("--config", default=str(REPO / "configs/experiments/exp3_agent_distillation_ablation.yaml"))
    ap.add_argument("--frozen", action="store_true", help="use frozen_config.json threshold")
    ap.add_argument("--no-correction", action="store_true", help="skip correction stage (e.g. train split)")
    ap.add_argument("--concurrency", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="limit rows (debug)")
    args = ap.parse_args()

    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    frozen = load_frozen()
    threshold = float(frozen.get("threshold", DEFAULT_THRESHOLD)) if args.frozen else DEFAULT_THRESHOLD

    budget = BudgetState(max_rmb=HARD_CAP_RMB, reserved_rmb=RESERVED_RMB)
    if BUDGET_FILE.exists():
        st = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
        budget.used_rmb = float(st.get("used_rmb", 0.0))
        print(f"[budget] resumed used_rmb={budget.used_rmb:.4f}")

    cache = RequestCache(str(CACHE_DIR))
    client = DeepSeekClient(
        model=config["provider"]["model"],
        json_mode=True,
        thinking_disabled=True,
        timeout_seconds=float(config["provider"].get("timeout_seconds", 90)),
        max_retries=int(config["provider"].get("max_retries", 1)),
        cache=cache,
        budget=budget,
        concurrency=args.concurrency,
    )

    rows = load_dataset(DATASET)
    if args.limit:
        rows = rows[: args.limit]
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    test_rows = [r for r in rows if r["split"] == "test"]

    if args.mode == "train":
        if args.no_correction:
            os.environ["EXP3_TRAIN_NO_CORRECTION"] = "1"
        out_path = AGENT_DIR / "train.jsonl"
        existing = load_existing(out_path)
        results = asyncio.run(run_split("train", train_rows, client, config, existing, threshold, set(), out_path=out_path))
        write_rows(out_path, results, [str(r["id"]) for r in train_rows])
        print(f"[train] wrote {out_path} rows={len(results)}")
    elif args.mode == "pilot":
        pilot = select_pilot(dev_rows, int(config["experiment"]["seed"]))
        out_path = AGENT_DIR / "dev.jsonl"
        existing = load_existing(out_path)
        mark_pilot = {str(r["id"]) for r in pilot}
        results = asyncio.run(run_split("pilot", pilot, client, config, existing, threshold, mark_pilot, out_path=out_path))
        write_rows(out_path, results, [str(r["id"]) for r in dev_rows])
        print(f"[pilot] wrote {out_path} rows={len(results)}")
    elif args.mode == "dev":
        out_path = AGENT_DIR / "dev.jsonl"
        existing = load_existing(out_path)
        results = asyncio.run(run_split("dev", dev_rows, client, config, existing, threshold, set(), out_path=out_path))
        write_rows(out_path, results, [str(r["id"]) for r in dev_rows])
        print(f"[dev] wrote {out_path} rows={len(results)}")
    elif args.mode == "test":
        out_path = AGENT_DIR / "test.jsonl"
        existing = load_existing(out_path)
        results = asyncio.run(run_split("test", test_rows, client, config, existing, threshold, set(), out_path=out_path))
        write_rows(out_path, results, [str(r["id"]) for r in test_rows])
        print(f"[test] wrote {out_path} rows={len(results)}")
    elif args.mode == "judge":
        for split, rows in (("dev", dev_rows), ("test", test_rows)):
            out_path = JUDGE_DIR / f"{split}.jsonl"
            existing = load_existing(out_path)
            results = asyncio.run(run_judge(split, rows, client, config, existing, out_path=out_path))
            write_rows(out_path, results, [str(r["id"]) for r in rows])
            print(f"[judge] wrote {out_path} rows={len(results)}")

    # persist budget + ledger snapshot
    save_budget(budget)
    ledger = client.ledger.snapshot(client.prices)
    cost_dir = OUT_ROOT / "metrics"
    cost_dir.mkdir(parents=True, exist_ok=True)
    cost_file = cost_dir / f"cost_{args.mode}.json"
    cost_file.write_text(json.dumps({**ledger, "used_rmb": round(budget.used_rmb, 6), "threshold": threshold, "cache": cache.stats()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cost] used_rmb={budget.used_rmb:.4f} -> {cost_file}")


if __name__ == "__main__":
    main()