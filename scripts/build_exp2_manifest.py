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
    RAW_AEGIS,
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="build the full-coverage manifest suite (guide 2026-08-06)")
    args = ap.parse_args()
    if args.full:
        build_full()
        return
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




# ---------------------------------------------------------------------------
# Full-coverage manifest (guide 2026-08-06 sections 2-3, 7, 15, 21, 26).
# ---------------------------------------------------------------------------

NULL_VALUES = {None, "", "None", "null", "nan", "N/A"}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip() in NULL_VALUES:
        return None
    return v


def _full_row(r: dict, source: str, *, overlap: bool, mode: str = "response",
              prompt_label=None, response_label=None, official_category=None,
              gold_binary=None, gold_type=None) -> dict:
    query = str(r.get("query", "") or "")
    answer = str(r.get("answer", "") or "")
    return {
        "sample_id": str(r["id"]),
        "qy_hash": qy_hash(query, answer),
        "source": source,
        "group_id": str(r.get("group_id", "") or r["id"]),
        "language": str(r.get("language", "") or ""),
        "official_category": official_category,
        "official_prompt_label": prompt_label,
        "official_response_label": response_label,
        "target_model": str(r.get("target_model", "") or ""),
        "query": query,
        "answer": answer,
        "has_response": mode == "response",
        "evaluation_mode": mode,
        "overlap_exp3": bool(overlap),
        "split": "test",
        "track": "fraud_assist",
        "expected_behavior": str(r.get("expected_behavior", "") or ""),
        "official_gold_binary": gold_binary,
        "official_gold_type": gold_type,
        "original_id": str(r.get("original_id", "") or r.get("id", "")),
        "metadata": r.get("metadata") or {},
    }


def build_full() -> None:
    """Build the full-coverage manifest suite (guide 2, 3, 7, 26). Zero API."""
    from frauddistill.exp2_cross_benchmark.paths import (
        AEGIS_PROMPT_MANIFEST, AEGIS_RESPONSE_MANIFEST, AUDIT_DIR, CACHE_AUDIT_FILE,
        FRAUD_ALIGNED_MANIFEST, FULL_MANIFEST, FULL_POOL_SUMMARY_FILE,
        SILVER_LABEL_TAG, TEACHER_T6_DIR, DNA_FRAUD_ALIGNED_AREAS,
        AEGIS_FRAUD_ALIGNED_CATEGORIES,
    )
    exp = exposure_hashes()
    all_rows: list[dict] = []
    cache_audit: dict[str, dict] = {}
    baseline_files = {
        "fraudr1": "fraudr1/baseline_predictions/fraudr1_official_judge_predictions.jsonl",
        "orbench": "orbench/baseline_predictions/orbench_response_checker_predictions.jsonl",
        "do_not_answer": "do_not_answer/baseline_predictions/dna_longformer_harmful_predictions.jsonl",
        "aegis2": "aegis2/baseline_predictions/aegis_nemoguard_predictions.jsonl",
    }

    # ---- fraudr1: 8,564 (guide 8) ----------------------------------------
    fr_rows = []
    for r in read_jsonl(EXPERIMENT_DIR / "fraudr1/unified/fraudr1_eval.jsonl"):
        h = qy_hash(r.get("query"), r.get("answer"))
        fr_rows.append(_full_row(
            r, "fraudr1", overlap=h in exp,
            official_category=str(r.get("category", "") or ""),
        ))
    all_rows.extend(fr_rows)

    # ---- orbench: 3,000 core pool (guide 9) ------------------------------
    or_rows = []
    for r in read_jsonl(EXPERIMENT_DIR / "orbench/unified/orbench_eval.jsonl"):
        h = qy_hash(r.get("query"), r.get("answer"))
        md = r.get("metadata") or {}
        or_rows.append(_full_row(
            r, "orbench", overlap=h in exp,
            official_category=str(r.get("category", "") or ""),
            gold_type=str(md.get("orbench_prompt_type", "") or ""),
        ))
    all_rows.extend(or_rows)

    # ---- do-not-answer: 5,634, official human labels (guide 10) ----------
    dna_rows = []
    for r in read_jsonl(EXPERIMENT_DIR / "do_not_answer/unified/do_not_answer_eval.jsonl"):
        h = qy_hash(r.get("query"), r.get("answer"))
        md = r.get("metadata") or {}
        g = _clean(r.get("gold_binary"))
        if g is None and _clean(md.get("harmful")) is not None:
            g = int(md["harmful"])
        gtype = str(r.get("gold_type", "") or "")
        if not gtype and g is not None:
            gtype = "unsafe" if g == 1 else "safe"
        dna_rows.append(_full_row(
            r, "do_not_answer", overlap=h in exp,
            official_category=str(md.get("risk_area", "") or ""),
            gold_binary=g, gold_type=gtype,
        ))
    all_rows.extend(dna_rows)

    # ---- aegis: 1,964 official test, response/prompt split (guide 3) ----
    raw_aegis = {r["id"]: r for r in json.loads((RAW_AEGIS / "test.json").read_text(encoding="utf-8"))}
    aegis_rows: list[dict] = []
    for r in read_jsonl(EXPERIMENT_DIR / "aegis2/unified/aegis2_eval.jsonl"):
        rid = str(r["id"]).replace("aegis_", "")
        raw = raw_aegis.get(rid, {})
        mode = "response" if ((raw.get("response") or "").strip() and raw.get("response_label") not in (None, "")) else "prompt_only"
        md = r.get("metadata") or {}
        pl = _clean(raw.get("prompt_label")) or _clean(md.get("prompt_label"))
        rl = _clean(raw.get("response_label")) or _clean(md.get("response_label"))
        gold = None
        gtype = None
        if mode == "response" and rl is not None:
            gold = 1 if str(rl).lower().startswith("unsafe") else 0
            gtype = "unsafe" if gold == 1 else "safe"
        elif mode == "prompt_only" and pl is not None:
            gold = 1 if str(pl).lower().startswith("unsafe") else 0
            gtype = "unsafe" if gold == 1 else "safe"
        row = _full_row(
            {**r, "query": str(raw.get("prompt", "") or r.get("query", "")),
             "answer": str(raw.get("response", "") or "")},
            "aegis2", overlap=qy_hash(str(raw.get("prompt", "") or ""), str(raw.get("response", "") or "")) in exp,
            mode=mode,
            prompt_label=(1 if pl is not None and str(pl).lower().startswith("unsafe") else 0) if pl is not None else None,
            response_label=(1 if rl is not None and str(rl).lower().startswith("unsafe") else 0) if rl is not None else None,
            official_category=str(raw.get("violated_categories", "") or ""),
            gold_binary=gold, gold_type=gtype,
        )
        row["metadata"] = {**md, "violated_categories": raw.get("violated_categories") or "",
                           "prompt_label_source": raw.get("prompt_label_source"),
                           "response_label_source": raw.get("response_label_source")}
        aegis_rows.append(row)
    all_rows.extend(aegis_rows)

    # ---- write full manifest + per-mode subsets ---------------------------
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with FULL_MANIFEST.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    aegis_resp = [r for r in aegis_rows if r["evaluation_mode"] == "response"]
    aegis_prompt = [r for r in aegis_rows if r["evaluation_mode"] == "prompt_only"]
    for path, rs in ((AEGIS_RESPONSE_MANIFEST, aegis_resp), (AEGIS_PROMPT_MANIFEST, aegis_prompt)):
        with path.open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- fraud-aligned official subsets (guide 5.3) ------------------------
    fraud_aligned = []
    for r in dna_rows:
        if (r["official_category"] or "") in DNA_FRAUD_ALIGNED_AREAS:
            fraud_aligned.append(r)
    for r in aegis_rows:
        cats = {c.strip() for c in (r["official_category"] or "").split(",") if c.strip()}
        if cats & AEGIS_FRAUD_ALIGNED_CATEGORIES:
            fraud_aligned.append(r)
    with FRAUD_ALIGNED_MANIFEST.open("w", encoding="utf-8") as f:
        for r in fraud_aligned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- cache audit (guide 21) --------------------------------------------
    teacher_files = {
        "fraudr1": TEACHER_T6_DIR / "fraudr1_t6_predictions.jsonl",
        "orbench": TEACHER_T6_DIR / "orbench_t6_predictions.jsonl",
        "do_not_answer": TEACHER_T6_DIR / "do_not_answer_t6_predictions.jsonl",
        "aegis2": TEACHER_T6_DIR / "aegis2_t6_predictions.jsonl",
    }
    teacher_prompt_file = TEACHER_T6_DIR / "aegis2_t6_prompt_predictions.jsonl"
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_source[r["source"]].append(r)
    for b in BENCHMARKS:
        rows = by_source[b]
        t_done = {str(r["id"]): r for r in read_jsonl(teacher_files[b])}
        b_done = {str(r["id"]): r for r in read_jsonl(EXPERIMENT_DIR / baseline_files[b])}
        missing_resp = [r for r in rows if r["evaluation_mode"] == "response" and r["sample_id"] not in t_done]
        if b == "aegis2":
            t_p = {str(r["id"]): r for r in read_jsonl(teacher_prompt_file)}
            missing_prompt = [r for r in rows if r["evaluation_mode"] == "prompt_only" and r["sample_id"] not in t_p]
        else:
            t_p = {}
            missing_prompt = []
        cache_audit[b] = {
            "total": len(rows),
            "response_rows": sum(1 for r in rows if r["evaluation_mode"] == "response"),
            "prompt_only_rows": sum(1 for r in rows if r["evaluation_mode"] == "prompt_only"),
            "valid_teacher_cache": len(t_done) + len(t_p),
            "teacher_response_cache": len(t_done),
            "teacher_prompt_cache": len(t_p),
            "missing_response": len(missing_resp),
            "missing_prompt": len(missing_prompt),
            "baseline_coverage": len(b_done),
        }
        print(f"[{b}] total={len(rows)} teacher_cache={len(t_done)}+{len(t_p)} missing_resp={len(missing_resp)} missing_prompt={len(missing_prompt)} baseline={len(b_done)}")

    # ---- pool assertions (guide 2.3) + summary ------------------------------
    n_fr = len({r["sample_id"] for r in fr_rows})
    n_or = len({r["sample_id"] for r in or_rows})
    n_dna = len({r["sample_id"] for r in dna_rows})
    n_ae = len({r["sample_id"] for r in aegis_rows})
    assert n_fr == 8564, n_fr
    assert n_or == 3000, n_or
    assert n_dna == 5634, n_dna
    assert n_ae == 1964, n_ae
    summary = {
        "guide": "2026-08-06",
        "pool": {"fraudr1": n_fr, "orbench": n_or, "do_not_answer": n_dna, "aegis2": n_ae},
        "aegis_response": len(aegis_resp),
        "aegis_prompt_only": len(aegis_prompt),
        "fraud_aligned": {"dna": sum(1 for r in fraud_aligned if r["source"] == "do_not_answer"),
                          "aegis": sum(1 for r in fraud_aligned if r["source"] == "aegis2")},
        "silver_label_tag": SILVER_LABEL_TAG,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_AUDIT_FILE.write_text(json.dumps(cache_audit, ensure_ascii=False, indent=1), encoding="utf-8")
    FULL_POOL_SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"full manifest: {len(all_rows)} rows -> {FULL_MANIFEST}")
    print(f"aegis response={len(aegis_resp)} prompt={len(aegis_prompt)} -> {AEGIS_RESPONSE_MANIFEST.parent}")
    print(f"fraud_aligned={len(fraud_aligned)} -> {FRAUD_ALIGNED_MANIFEST}")
    print(f"cache audit -> {CACHE_AUDIT_FILE}; summary -> {FULL_POOL_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
