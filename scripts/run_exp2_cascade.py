"""Run the budgeted cascade teacher for exp2 v2 (pilot / dev / full).

Usage:
  python scripts/run_exp2_cascade.py --benchmark fraudr1 --mode pilot --n 100
  python scripts/run_exp2_cascade.py --benchmark dna --mode full
  python scripts/run_exp2_cascade.py --benchmark aegis2 --mode full

Checkpointing: rows already present in the output jsonl are skipped.
Budget: persisted across runs in outputs/api_cache_exp2_cascade/budget_state.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frauddistill.data.split_groups import stratified_sample
from frauddistill.pipeline import BudgetedCascadePipeline, PipelineConfig
from frauddistill.providers.deepseek_client import DeepSeekClient
from frauddistill.runtime.budget import BudgetState
from frauddistill.runtime.cache import RequestCache

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "exp2_prior_work_comparison"))
# benchmark key -> canonical experiment folder (v2 "dna" lives under do_not_answer/)
OUT_DIR_ALIAS = {"dna": "do_not_answer"}
CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "exp2_budgeted_cascade.yaml"))
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "api_cache_exp2_cascade"))
BUDGET_FILE = os.path.join(CACHE_DIR, "budget_state.json")

UNIFIED = {
    "fraudr1": "fraudr1/unified/fraudr1_eval.jsonl",
    "fraudr1_diag": "fraudr1/diagnostic/fraudr1_balanced_diag.jsonl",
    "orbench": "orbench/unified/orbench_eval.jsonl",
    "dna": "do_not_answer/unified/do_not_answer_eval.jsonl",
    "aegis2": "aegis2/unified/aegis2_eval_valid_qy.jsonl",
}

GOLD_JOIN = {
    "fraudr1": ("fraudr1/human_audit/human_audit_adjudicated.jsonl", "binary"),
    "orbench": ("orbench/human_audit/human_audit_adjudicated.jsonl", "binary"),
}


def derive_gold(benchmark: str, rows: list[dict]) -> None:
    """Fill gold_binary from adjudicated answer-level audits (fraudr1 / orbench).

    Answer-level detection: gold=1 means the ANSWER is unsafe. For OR-Bench the
    official expected_behavior describes the PROMPT (should_refuse), which must
    NOT be used as the answer label (refusals to toxic prompts are safe).
    """
    if benchmark in GOLD_JOIN:
        rel, key = GOLD_JOIN[benchmark]
        jpath = os.path.join(BASE, rel)
        if not os.path.exists(jpath):
            print(f"[gold] missing join file {jpath}")
            return
        join = {}
        with open(jpath, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    join[str(j["id"])] = j.get(key)
                except Exception:
                    pass
        for r in rows:
            if r.get("gold_binary") is None:
                r["gold_binary"] = join.get(str(r["id"]))


def load_rows(benchmark: str, path_override: str = "") -> list[dict]:
    if path_override:
        path = path_override
    else:
        path = os.path.join(BASE, UNIFIED[benchmark])
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def strata_key(benchmark: str, r: dict) -> str:
    if benchmark == "fraudr1_diag":
        return "|".join([str(r.get("category") or "?"), str(r.get("language") or "?")])
    if benchmark == "fraudr1":
        m = r.get("metadata", {})
        return "|".join([
            str(r.get("category") or "?"),
            str(r.get("language") or "?"),
            str(m.get("fraudr1_variant") or "?"),
            str(m.get("fraudr1_scenario") or "?"),
        ])
    if benchmark == "orbench":
        oid = str(r.get("original_id") or "")
        prefix = oid.split("_")[0] if "_" in oid else "unknown"
        return prefix + "|" + str(r.get("category") or "?")
    if benchmark == "dna":
        return "|".join([str(r.get("category") or "?"), str(r.get("target_model") or "?")])
    if benchmark == "aegis2":
        return str(r.get("category") or "?") + "|" + str(r.get("sub_category") or "?")
    return "?"


def _with_strata(rows: list[dict], benchmark: str) -> list[dict]:
    for r in rows:
        r["_strata"] = strata_key(benchmark, r)
    return rows


PREDICTION_FILES = {
    "orbench": ("orbench/baseline_predictions/orbench_response_checker_predictions.jsonl",
                "orbench/teacher_predictions/orbench_teacher_predictions.jsonl"),
    "dna": ("do_not_answer/baseline_predictions/dna_longformer_harmful_predictions.jsonl",
            "do_not_answer/teacher_predictions/do_not_answer_teacher_predictions.jsonl"),
    "aegis2": ("aegis2/baseline_predictions/aegis_nemoguard_predictions.jsonl",
               "aegis2/teacher_predictions/aegis2_teacher_predictions.jsonl"),
    "fraudr1": ("fraudr1/baseline_predictions/fraudr1_official_judge_predictions.jsonl",
                "fraudr1/teacher_predictions/fraudr1_teacher_predictions.jsonl"),
}


def load_predictions(benchmark: str) -> tuple[dict, dict, dict]:
    """Return {id: prediction_binary} for baseline/teacher and {id: risk_score} for teacher."""
    if benchmark not in PREDICTION_FILES:
        return {}, {}, {}
    bl_path, te_path = PREDICTION_FILES[benchmark]

    def one(rel):
        m: dict[str, int] = {}
        path = os.path.join(BASE, rel)
        if not os.path.exists(path):
            return m
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if j.get("prediction_binary") is not None:
                        m[str(j["id"])] = int(j["prediction_binary"])
                except Exception:
                    pass
        return m

    def one_risk(rel):
        m: dict[str, float] = {}
        path = os.path.join(BASE, rel)
        if not os.path.exists(path):
            return m
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if j.get("risk_score") is not None:
                        m[str(j["id"])] = float(j["risk_score"])
                except Exception:
                    pass
        return m

    return one(bl_path), one(te_path), one_risk(te_path)


def sample_pilot(rows: list[dict], n: int, seed: int, benchmark: str) -> list[dict]:
    """Prioritize baseline/teacher disagreement and both-wrong rows, then stratified fill.

    Guide phase 1: prefer baseline-wrong/teacher-right, baseline-right/teacher-wrong,
    both-wrong, decision-boundary rows, then cover major subgroups.
    """
    bl, te, te_risk = load_predictions(benchmark)
    gold_rows = [r for r in rows if r.get("gold_binary") is not None]
    rng = random.Random(seed)

    def status(r):
        g = r.get("gold_binary")
        b = bl.get(str(r["id"]))
        t = te.get(str(r["id"]))
        if b is not None and t is not None and b != t:
            return "disagree"
        if g is not None and b is not None and t is not None and b != g and t != g:
            return "both_wrong"
        if b is not None and t is not None and b == t and g is not None and b != g:
            return "both_wrong"
        return "fill"

    by_status: dict[str, list[dict]] = {"disagree": [], "both_wrong": [], "fill": []}
    for r in rows:
        by_status[status(r)].append(r)

    chosen: list[dict] = []
    chosen_ids: set[str] = set()

    def take(pool, k):
        rng.shuffle(pool)
        picked = [r for r in pool if r["id"] not in chosen_ids][:k]
        for r in picked:
            chosen.append(r)
            chosen_ids.add(r["id"])

    take(by_status["disagree"], int(n * 0.40))
    take(by_status["both_wrong"], int(n * 0.20))
    # boundary: teacher risk_score in [0.35, 0.70] among remaining rows
    boundary = [r for r in rows if r["id"] not in chosen_ids
                and 0.35 <= te_risk.get(str(r["id"]), -1) <= 0.70]
    take(boundary, int(n * 0.10))
    # stratified fill across (category, language, gold) for the rest
    remaining = [r for r in rows if r["id"] not in chosen_ids]
    fill = stratified_sample(_with_strata(remaining, benchmark), n - len(chosen),
                             stratify_key="_strata", seed=seed)
    chosen.extend(fill)
    return chosen[:n]


def sample_dev(rows: list[dict], n: int, seed: int, exclude_ids: set[str], benchmark: str) -> list[dict]:
    pool = [r for r in rows if r["id"] not in exclude_ids]
    return stratified_sample(_with_strata(pool, benchmark), n, stratify_key="_strata", seed=seed)


def load_budget() -> BudgetState:
    used = 0.0
    if os.path.exists(BUDGET_FILE):
        try:
            used = float(json.load(open(BUDGET_FILE, encoding="utf-8")).get("used_rmb", 0.0))
        except Exception:
            used = 0.0
    return BudgetState(max_rmb=27.0, reserved_rmb=3.0, used_rmb=used)


def save_budget(state: BudgetState) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(BUDGET_FILE, "w", encoding="utf-8") as f:
        json.dump({"used_rmb": round(state.used_rmb, 4)}, f)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(UNIFIED))
    ap.add_argument("--mode", required=True, choices=["pilot", "dev", "full"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--exclude", default="", help="comma-separated jsonl paths whose ids are excluded (dev)")
    ap.add_argument("--manifest", default="", help="path to a dev manifest jsonl (rows to run)")
    ap.add_argument("--tag", default="", help="extra tag for output filename")
    ap.add_argument("--max-tokens-triage", type=int, default=180)
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(CONFIG)
    if args.manifest:
        rows = load_rows(args.benchmark, path_override=args.manifest)
    else:
        rows = load_rows(args.benchmark)
    derive_gold(args.benchmark, rows)
    print(f"[{args.benchmark}] loaded {len(rows)} rows (mode={args.mode} manifest={args.manifest})")

    exclude_ids: set[str] = set()
    for p in args.exclude.split(","):
        if not p:
            continue
        with open(p.strip(), encoding="utf-8") as f:
            for line in f:
                try:
                    exclude_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    if args.mode == "pilot":
        todo = sample_pilot(rows, args.n or 100, args.seed, args.benchmark)
    elif args.mode == "dev":
        todo = sample_dev(rows, args.n or 300, args.seed, exclude_ids, args.benchmark)
    else:
        todo = rows
    print(f"[{args.benchmark}] todo={len(todo)} (excluded={len(exclude_ids)})")

    cache = RequestCache(CACHE_DIR)
    budget = load_budget()
    print(f"[budget] used={budget.used_rmb:.4f} cap={budget.effective_cap:.2f} RMB")
    client = DeepSeekClient(
        model=cfg_model(args),
        cache=cache,
        budget=budget,
        concurrency=120,
        max_retries=1,
    )
    pipeline = BudgetedCascadePipeline(client, cfg, prompt_version="v2.0")

    out_bench = OUT_DIR_ALIAS.get(args.benchmark, args.benchmark)
    os.makedirs(os.path.join(BASE, out_bench, "cascade_predictions"), exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(
        BASE, out_bench, "cascade_predictions",
        f"cascade_{args.mode}{tag}_{args.seed}.jsonl",
    )
    summary = await pipeline.process_batch(todo, out_path, checkpoint=True)

    print(f"\n[{args.benchmark}] summary: {summary}")
    snap = client.ledger.snapshot(client.prices)
    print(f"[usage] {json.dumps(snap, ensure_ascii=False)}")
    save_budget(budget)
    print(f"[budget] now used={budget.used_rmb:.4f} RMB")
    # cache stats
    if cache is not None:
        print(f"[cache] {cache.stats()}")


def cfg_model(args) -> str:
    # model fixed by config; keep simple
    return "deepseek-v4-flash"


if __name__ == "__main__":
    asyncio.run(main())
