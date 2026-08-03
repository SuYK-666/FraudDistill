"""Build frozen Phase-2 dev manifests (300 rows per benchmark, group-stratified).

Excludes pilot ids (already used for prompt validation). OR-Bench dev is drawn
from the 600 audited rows so calibration has gold labels. DNA/Aegis/FraudR1-diag
dev rows carry their official gold. Seed fixed for reproducibility.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frauddistill.data.split_groups import stratified_sample

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "exp2_prior_work_comparison"))
OUT_DIR = os.path.join(BASE, "_dev_manifest")
SEED = 20260803
N = 300

UNIFIED = {
    "fraudr1_diag": "fraudr1/diagnostic/fraudr1_balanced_diag.jsonl",
    "orbench": "orbench/unified/orbench_eval.jsonl",
    "dna": "do_not_answer/unified/do_not_answer_eval.jsonl",
    "aegis2": "aegis2/unified/aegis2_eval_valid_qy.jsonl",
}
PILOT_FILES = {
    "fraudr1_diag": "fraudr1_diag/cascade_predictions/cascade_pilot_v2b_20260803.jsonl",
    "orbench": "orbench/cascade_predictions/cascade_pilot_v2b_20260803.jsonl",
    "dna": "dna/cascade_predictions/cascade_pilot_v2b_20260803.jsonl",
    "aegis2": "aegis2/cascade_predictions/cascade_pilot_v2b_20260803.jsonl",
}


def load(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def ids_of(p):
    out = set()
    if not os.path.exists(p):
        return out
    for r in load(p):
        out.add(str(r.get("id", "")))
    return out


def strata_key(benchmark, r, audit=None):
    """Group-stratified key. DNA/Aegis and audited OR-Bench dev strata include
    the gold label so the frozen dev set keeps enough positives to calibrate
    recall/FPR (guide Phase 2: dev set must be usable for threshold tuning)."""
    if benchmark == "fraudr1_diag":
        return "|".join([str(r.get("category") or "?"), str(r.get("language") or "?")])
    if benchmark == "orbench":
        oid = str(r.get("original_id") or r.get("id") or "")
        prefix = oid.split("_")[0] if "_" in oid else "unknown"
        g = "?"
        if audit is not None:
            a = audit.get(str(r.get("id", "")))
            g = str(a.get("binary", "?")) if a else "?"
        return "g" + g + "|" + prefix + "|" + str(r.get("category") or "?")
    if benchmark == "dna":
        return "g" + str(r.get("gold_binary") or "?") + "|" + str(r.get("category") or "?") + "|" + str(r.get("target_model") or "?")
    if benchmark == "aegis2":
        return "g" + str(r.get("gold_binary") or "?") + "|" + str(r.get("category") or "?") + "|" + str(r.get("sub_category") or "?")
    return "?"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}
    for bench, rel in UNIFIED.items():
        rows = load(os.path.join(BASE, rel))
        pilot_ids = ids_of(os.path.join(BASE, PILOT_FILES[bench]))
        pool = [r for r in rows if str(r.get("id", "")) not in pilot_ids]
        audit = None
        if bench == "orbench":
            audited = ids_of(os.path.join(BASE, "orbench/human_audit/human_audit_adjudicated.jsonl"))
            before = len(pool)
            pool = [r for r in pool if str(r.get("id", "")) in audited]
            audit = {str(a.get("id", "")): a for a in load(os.path.join(BASE, "orbench/human_audit/human_audit_adjudicated.jsonl"))}
            print(f"[{bench}] audited pool: {before} -> {len(pool)}")
        for r in pool:
            r["_strata"] = strata_key(bench, r, audit)
        if bench in ("dna", "orbench"):
            # gold-balanced dev (pos/neg halves) so recall/FPR calibration is stable
            pos = [r for r in pool if r.get("gold_binary") == 1]
            neg = [r for r in pool if r.get("gold_binary") == 0]
            if audit is not None:
                pos = [r for r in pool if audit.get(str(r.get("id", ""))) and audit[str(r.get("id", ""))].get("binary") == 1]
                neg = [r for r in pool if audit.get(str(r.get("id", ""))) and audit[str(r.get("id", ""))].get("binary") == 0]
            half = N // 2
            picked = stratified_sample(pos, min(half, len(pos)), stratify_key="_strata", seed=SEED)
            picked += stratified_sample(neg, N - len(picked), stratify_key="_strata", seed=SEED + 1)
        else:
            picked = stratified_sample(pool, N, stratify_key="_strata", seed=SEED)
        out_path = os.path.join(OUT_DIR, f"{bench}_dev{N}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in picked:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gold = Counter(r.get("gold_binary") for r in picked)
        strata = Counter(r["_strata"] for r in picked)
        summary[bench] = {"n": len(picked), "gold": dict(gold), "strata": len(strata), "file": out_path}
        print(f"[{bench}] wrote {len(picked)} rows gold={dict(gold)} strata={len(strata)}")
    with open(os.path.join(OUT_DIR, "dev_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n": N, "note": "Phase-2 dev; pilot ids excluded; orbench restricted to audited rows", **summary}, f, ensure_ascii=False, indent=2)
    print("done")


if __name__ == "__main__":
    main()
