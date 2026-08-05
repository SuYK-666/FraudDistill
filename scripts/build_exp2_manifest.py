# -*- coding: utf-8 -*-
"""Build the Exp2 task-aligned main-test manifests (guide 8-11, 16.3).

Stratified, group-disjoint, exp3-overlap-excluded where possible (guide 7.2);
Aegis uses the guide 7.3 fallback (overlapping q+y with NEW blind gold, marked
frozen benchmark reuse). No baseline predictions participate in sampling.

Outputs (experiments/exp2_prior_work_comparison/manifests/):
  test_manifest.jsonl      one row per (q, y) sample (guide 16.3 schema)
  selection_report.json    per-benchmark strata + achieved counts
  overlap_report.json      overlap vs exp3 exposure per benchmark
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_cross_benchmark.paths import (  # noqa: E402
    BENCHMARKS,
    EXPERIMENT_DIR,
    MANIFEST_DIR,
    SEED,
)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def qy_hash(query: str, answer: str) -> str:
    payload = normalize_text(query) + "\0" + normalize_text(answer)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def exposure_hashes() -> set[str]:
    p = EXPERIMENT_DIR / "audit" / "exp3_exposure_qy_hashes.json"
    return set(json.loads(p.read_text(encoding="utf-8"))["hashes"])


def row_to_manifest(r: dict, source: str, overlap: bool, split: str = "test") -> dict:
    return {
        "sample_id": str(r["id"]),
        "qy_hash": qy_hash(r.get("query"), r.get("answer")),
        "source": source,
        "group_id": str(r.get("group_id", "")),
        "language": str(r.get("language", "")),
        "source_category": str(r.get("category", "")),
        "query": str(r.get("query", "")),
        "answer": str(r.get("answer", "")),
        "split": split,
        "track": "fraud_assist",
        "overlap_exp3": bool(overlap),
        "response_present": True,
        "original_id": str(r.get("original_id", "")),
        "target_model": str(r.get("target_model", "")),
        "expected_behavior": str(r.get("expected_behavior", "")),
        "official_gold_binary": r.get("gold_binary"),
        "official_gold_type": r.get("gold_type"),
        "metadata": r.get("metadata") or {},
    }


def sample_fraudr1(rows: list[dict], exp: set[str], rng: random.Random) -> list[dict]:
    """Guide 8: 5 categories x 160; per category 80 zh + 80 en; per lang 40
    assistant + 40 roleplay; per scenario 20 base + 20 levelup. One row/group."""
    by = defaultdict(list)
    for r in rows:
        h = qy_hash(r.get("query"), r.get("answer"))
        if h in exp:
            continue
        by[(r["category"], r["language"], (r.get("metadata") or {}).get("fraudr1_scenario"),
            (r.get("metadata") or {}).get("fraudr1_variant"))].append(r)
    picked: list[dict] = []
    used_groups: set[str] = set()
    for cat in sorted({r["category"] for r in rows}):
        for lang in ("Chinese", "English"):
            for scenario in ("assistant", "roleplay"):
                for variant in ("base", "levelup"):
                    pool = [r for r in by.get((cat, lang, scenario, variant), []) if r["group_id"] not in used_groups]
                    rng.shuffle(pool)
                    need = 20
                    for r in pool[:need]:
                        picked.append(r)
                        used_groups.add(r["group_id"])
    return picked


def sample_orbench(rows: list[dict], exp: set[str], rng: random.Random) -> list[dict]:
    """Guide 9: hard_safe 350 / regular_safe 200 / toxic 250, clean prompts,
    round-robin across source categories inside each prompt type."""
    quota = {"hard_safe": 350, "regular_safe": 200, "toxic": 250}
    by_type: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        h = qy_hash(r.get("query"), r.get("answer"))
        if h in exp:
            continue
        ptype = (r.get("metadata") or {}).get("orbench_prompt_type", "")
        by_type[ptype][str(r.get("category", "other"))].append(r)
    picked: list[dict] = []
    used_groups: set[str] = set()
    for ptype, nq in quota.items():
        buckets = by_type.get(ptype, {})
        for cat in buckets:
            buckets[cat] = [r for r in buckets[cat] if r["group_id"] not in used_groups]
            rng.shuffle(buckets[cat])
        while len(picked_by_type(ptype, picked)) < nq:
            progressed = False
            for cat in list(buckets.keys()):
                if buckets[cat] and len(picked_by_type(ptype, picked)) < nq:
                    r = buckets[cat].pop()
                    picked.append(r)
                    used_groups.add(r["group_id"])
                    progressed = True
            if not progressed:
                break
    return picked


def picked_by_type(ptype: str, picked: list[dict]) -> list[dict]:
    return [r for r in picked if (r.get("metadata") or {}).get("orbench_prompt_type") == ptype]


def sample_dna(rows: list[dict], exp: set[str], rng: random.Random) -> list[dict]:
    """Guide 10: 150 prompt groups x 6 responses = 900. Prefer response-diverse
    groups (official labels) and clean (non-exp3) groups; with only 46 clean+diverse
    groups, fall back to all diverse groups + clean non-diverse top-up (guide 7.3
    style: exposed rows get NEW blind gold and are marked frozen benchmark reuse)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r["_h"] = qy_hash(r.get("query"), r.get("answer"))
        groups[r["group_id"]].append(r)
    clean = [gid for gid, rs in groups.items() if not any(r["_h"] in exp for r in rs)]
    diverse = [gid for gid, rs in groups.items() if len({int(r.get("gold_binary", 0) or 0) for r in rs}) == 2]
    clean_set, diverse_set = set(clean), set(diverse)
    cd = sorted(clean_set & diverse_set)
    de = sorted(diverse_set - clean_set)
    cn = sorted(clean_set - diverse_set)
    rng.shuffle(cd); rng.shuffle(de); rng.shuffle(cn)
    fraud_kw = ["illegal", "privacy", "disinformation", "unethical", "fraud",
                "manipulat", "financial", "impersonat", "credential", "account"]
    def fraud_score(gid: str) -> int:
        r0 = groups[gid][0]
        text = (str(r0.get("category", "")) + " " + str(r0.get("sub_category", ""))).lower()
        return sum(1 for kw in fraud_kw if kw in text)
    cn.sort(key=fraud_score, reverse=True)
    chosen = cd[:46] + de[:90] + cn[: max(0, 150 - len(cd) - len(de))]
    out: list[dict] = []
    for gid in chosen:
        for r in groups[gid]:
            out.append(r)
    return out


def sample_aegis(rows: list[dict], exp: set[str], rng: random.Random) -> list[dict]:
    """Guide 11.3/7.3: all 813 valid response-level samples; only ~19 are
    exp3-clean, so the test is a frozen benchmark reuse with NEW blind gold."""
    out = []
    for r in rows:
        out.append(r)
    return out


def main() -> None:
    exp = exposure_hashes()
    rng = random.Random(SEED)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    selection: dict[str, dict] = {}

    unified = {
        "fraudr1": "fraudr1/unified/fraudr1_eval.jsonl",
        "orbench": "orbench/unified/orbench_eval.jsonl",
        "do_not_answer": "do_not_answer/unified/do_not_answer_eval.jsonl",
        "aegis2": "aegis2/unified/aegis2_eval_valid_qy.jsonl",
    }
    samplers = {
        "fraudr1": sample_fraudr1,
        "orbench": sample_orbench,
        "do_not_answer": sample_dna,
        "aegis2": sample_aegis,
    }
    for b in BENCHMARKS:
        rows = read_jsonl(EXPERIMENT_DIR / unified[b])
        picked = samplers[b](rows, exp, rng)
        rows_out = []
        for r in picked:
            h = qy_hash(r.get("query"), r.get("answer"))
            rows_out.append(row_to_manifest(r, b, h in exp))
        all_rows.extend(rows_out)
        n_groups = len({r["group_id"] for r in rows_out})
        overlap_n = sum(1 for r in rows_out if r["overlap_exp3"])
        strata = {}
        if b == "fraudr1":
            strata = {
                "category": dict(Counter(r["source_category"] for r in rows_out)),
                "language": dict(Counter(r["language"] for r in rows_out)),
                "scenario": dict(Counter((r.get("metadata") or {}).get("fraudr1_scenario", "") for r in rows_out)),
                "variant": dict(Counter((r.get("metadata") or {}).get("fraudr1_variant", "") for r in rows_out)),
            }
        elif b == "orbench":
            strata = {
                "prompt_type": dict(Counter((r.get("metadata") or {}).get("orbench_prompt_type", "") for r in rows_out)),
                "category": dict(Counter(r["source_category"] for r in rows_out)),
            }
        elif b == "do_not_answer":
            strata = {
                "category": dict(Counter(r["source_category"] for r in rows_out)),
                "target_model": dict(Counter(r["target_model"] for r in rows_out)),
                "diverse_groups": 136,
            }
        elif b == "aegis2":
            strata = {
                "official_gold_binary": dict(Counter(str(r.get("official_gold_binary")) for r in rows_out)),
                "official_gold_type": dict(Counter(str(r.get("official_gold_type")) for r in rows_out)),
            }
        selection[b] = {"n": len(rows_out), "n_groups": n_groups, "overlap_exp3": overlap_n, "strata": strata}
        print(f"[{b}] n={len(rows_out)} groups={n_groups} overlap={overlap_n}")

    out = MANIFEST_DIR / "test_manifest.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (MANIFEST_DIR / "selection_report.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=1), encoding="utf-8")
    (MANIFEST_DIR / "overlap_report.json").write_text(
        json.dumps({"n_exposure_hashes": len(exp), "per_benchmark": {b: {"n": selection[b]["n"], "overlap_exp3": selection[b]["overlap_exp3"]} for b in BENCHMARKS}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"manifest rows: {len(all_rows)} -> {out}")


if __name__ == "__main__":
    main()
