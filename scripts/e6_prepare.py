# -*- coding: utf-8 -*-
"""E6 manifest build + leakage audit (offline, no API)."""
from __future__ import annotations
import json, random, re, sys, csv, io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from e6_common import (DATA_DIR, PROTOCOL_DIR, SEED, read_jsonl, write_jsonl, write_json, read_json,
                       norm_query, sha256_text, manifest_sha256, TRAIN_MANIFEST, E6_DIR)

CURATED = ROOT / "scripts" / "e6_curated_questions.json"
CJK = re.compile(r"[\u4e00-\u9fff]")

CATS = ["impersonation", "fraudulent service", "phishing", "fake job posting", "network friendship"]
STRATUM_ORDER = ["direct_unsafe", "roleplay_unsafe", "unseen_unsafe", "hard_safe", "anti_fraud", "matched_safe", "benign"]

def lang_of(q: str) -> str:
    return "zh" if CJK.search(q or "") else "en"

def load_pools():
    return {
        "direct": read_jsonl(ROOT / "data/prepared/e4e5_v2/fraudr1_unseen_prompts.jsonl"),
        "roleplay": [r for r in read_jsonl(ROOT / "experiments/exp2_prior_work_comparison/balanced_design/manifests/balanced_test_manifest.jsonl")
                     if r.get("source") == "fraudr1"],
        "u1_unsafe": [r for r in read_jsonl(ROOT / "experiments/exp4_unseen/manifests/frozen_test.jsonl")
                      if r.get("primary_shift") == "U1_category" and r.get("gold_label") == "unsafe"],
        "u1_safe": [r for r in read_jsonl(ROOT / "experiments/exp4_unseen/manifests/frozen_test.jsonl")
                    if r.get("primary_shift") == "U1_category" and r.get("gold_label") == "safe"],
        "or": read_jsonl(ROOT / "experiments/exp2_prior_work_comparison/balanced_design/prompts/or_prompt_pool.jsonl"),
        "benign": [r for r in read_jsonl(ROOT / "data/prepared/e1_final_triad_v4/E1_V4_PANEL_ANCHOR.jsonl")
                   if "BENIGN" in r.get("response_id", "")],
    }

def dedup_keep(rows, key_fn, rng):
    seen, out = set(), []
    for r in rows:
        k = key_fn(r)
        if k in seen or not k:
            continue
        seen.add(k); out.append(r)
    return out

def build_manifest():
    rng = random.Random(SEED)
    curated = read_json(CURATED, {})
    pools = load_pools()
    train_norm = {norm_query(r.get("user_query", "")) for r in read_jsonl(TRAIN_MANIFEST)}
    def fresh(cands, k=1):
        out = []
        for r in cands:
            if norm_query(r.get("user_query", r.get("query", r.get("q_private", "")))).lower() in {t.lower() for t in train_norm}:
                continue
            out.append(r)
            if len(out) >= k:
                break
        return out
    rows = []
    families_seen = set()

    def add(prompt_family, source, source_id, language, stratum, risk, user_query, provenance, extra=None):
        row = {
            "prompt_family": prompt_family,
            "source": source,
            "source_id": source_id,
            "language": language,
            "prompt_stratum": stratum,
            "prompt_risk_label": risk,
            "should_refuse": risk == "unsafe",
            "user_query": user_query,
            "split_provenance": provenance,
            "student_train_overlap": False,
            "q_hash": sha256_text(norm_query(user_query)),
            "prompt_family_key": sha256_text(prompt_family)[:12],
        }
        if extra:
            row.update(extra)
        rows.append(row)

    # ---- stratum 1: direct_unsafe 50 (25 zh / 25 en, 5 categories x 5) ----
    direct = dedup_keep(pools["direct"], lambda r: norm_query(r.get("user_query", "")).lower(), rng)
    for lang in ["Chinese", "English"]:
        for cat in CATS:
            cands = [r for r in direct if r.get("language") == lang and r.get("fraud_category") == cat and norm_query(r.get("user_query", "")) not in train_norm]
            rng.shuffle(cands)
            for r in cands[:5]:
                add(f"fraudr1_unseen_{'zh' if lang=='Chinese' else 'en'}_{cat}", "fraudr1",
                    r.get("id", ""), "zh" if lang == "Chinese" else "en", "direct_unsafe", "unsafe",
                    norm_query(r.get("user_query", "")), "e4_holdout",
                    {"fraud_category": cat, "fraud_subcategory": r.get("fraud_subcategory")})

    # ---- stratum 2: roleplay_unsafe 30 (15 zh / 15 en) ----
    rp = dedup_keep(pools["roleplay"], lambda r: norm_query(r.get("query", "")).lower(), rng)
    for lang in ["zh", "en"]:
        cands = [r for r in rp if lang_of(r.get("query", "")) == lang and norm_query(r.get("query", "")) not in train_norm]
        rng.shuffle(cands)
        for r in cands[:15]:
            add(r.get("group_id") or f"roleplay_{r.get('sample_id')}", "fraudr1_roleplay",
                r.get("sample_id", ""), lang, "roleplay_unsafe", "unsafe",
                norm_query(r.get("query", "")), "e2_reserved", {"fraud_category": r.get("group_id", "")[:60]})

    # ---- stratum 3: unseen_unsafe 20 (10 zh / 10 en; elder 5 + naked 5 per lang) ----
    u1u = dedup_keep(pools["u1_unsafe"], lambda r: norm_query(r.get("user_query", "")).lower(), rng)
    for lang, langkey in [("zh", "zh"), ("en", "en")]:
        for cat in ["elder_health_product", "naked_chat_sextortion"]:
            cands = [r for r in u1u if r.get("language") == langkey and r.get("fraud_category") == cat and norm_query(r.get("user_query", "")) not in train_norm]
            rng.shuffle(cands)
            for r in cands[:5]:
                add(r.get("template_family_id") or r.get("family_id") or f"u1_{cat}",
                    "e4_holdout_u1", r.get("id", ""), lang, "unseen_unsafe", "unsafe",
                    norm_query(r.get("user_query", "")), "e4_holdout", {"fraud_category": cat})

    # ---- stratum 4: hard_safe 40 (20 en OR-Bench + 20 zh translations) ----
    # selection fixed from e6_dump_candidates.json (deterministic, seed 20260810)
    dump = read_json(ROOT / "scripts" / "e6_dump_candidates.json", {})
    chosen = dump.get("hard_safe_en", [])
    assert len(chosen) == 20, len(chosen)
    or_by_q = {norm_query(r.get("query", "")).lower(): r for r in pools["or"]}
    tr = curated.get("hard_safe_zh_translations", {})
    for item in chosen:
        q_en = norm_query(item["query"])
        if q_en in train_norm:
            raise RuntimeError(f"hard_safe en overlaps train: {q_en[:60]}")
        r = or_by_q.get(q_en.lower())
        if r is None:
            raise RuntimeError(f"or row not found: {q_en[:60]}")
        src_id = r.get("prompt_id", "") or q_en[:60]
        fam = src_id  # per-query OR-Bench family (zh translation shares the family)
        add(fam, "or_bench", src_id, "en", "hard_safe", "safe", q_en, "or_bench",
            {"fraud_category": item.get("category")})
        q_zh = (tr.get(q_en) or tr.get(src_id) or "").strip()
        if not q_zh:
            raise RuntimeError(f"missing zh translation for {src_id}")
        add(fam, "or_bench_zh_translation", f"zh_translation_of:{src_id}", "zh", "hard_safe", "safe", q_zh, "or_bench",
            {"fraud_category": item.get("category")})

    # ---- stratum 5: anti_fraud 30 (15 zh / 15 en authored) ----
    for i, q in enumerate(curated.get("anti_fraud_en", [])):
        nq = norm_query(q)
        if nq in train_norm:
            raise RuntimeError(f"anti_fraud en overlaps train: {nq[:60]}")
        add("e6_author_anti_fraud", "anti_fraud_education", f"e6_author_af_en_{i+1:02d}", "en", "anti_fraud", "safe",
            nq, "other_author", {"authored": True})
    for i, q in enumerate(curated.get("anti_fraud_zh", [])):
        nq = norm_query(q)
        if nq in train_norm:
            raise RuntimeError(f"anti_fraud zh overlaps train: {nq[:60]}")
        add("e6_author_anti_fraud", "anti_fraud_education", f"e6_author_af_zh_{i+1:02d}", "zh", "anti_fraud", "safe",
            nq, "other_author", {"authored": True})

    # ---- stratum 6: matched_safe 20 (10 zh / 10 en curated from U1 safe) ----
    u1s = {norm_query(r.get("user_query", "")).lower(): r for r in pools["u1_safe"]}
    for q in curated.get("matched_safe_en", []):
        r = u1s.get(norm_query(q).lower())
        if not r: raise RuntimeError(f"matched_safe en row not found: {q[:60]}")
        if norm_query(q) in train_norm: raise RuntimeError(f"matched_safe en overlaps train: {q[:60]}")
        add(r.get("template_family_id") or f"u1_{r.get('fraud_category')}", "e4_holdout_u1", r.get("id", ""), "en",
            "matched_safe", "safe", norm_query(q), "e4_holdout", {"fraud_category": r.get("fraud_category")})
    for q in curated.get("matched_safe_zh", []):
        r = u1s.get(norm_query(q).lower())
        if not r: raise RuntimeError(f"matched_safe zh row not found: {q[:60]}")
        if norm_query(q) in train_norm: raise RuntimeError(f"matched_safe zh overlaps train: {q[:60]}")
        add(r.get("template_family_id") or f"u1_{r.get('fraud_category')}", "e4_holdout_u1", r.get("id", ""), "zh",
            "matched_safe", "safe", norm_query(q), "e4_holdout", {"fraud_category": r.get("fraud_category")})

    # ---- stratum 7: benign 10 (5 zh / 5 en from E1 anchor BENIGN) ----
    ben = dedup_keep(pools["benign"], lambda r: norm_query(r.get("q_private", "")).lower(), rng)
    for lang in ["zh", "en"]:
        cands = [r for r in ben if lang_of(r.get("q_private", "")) == lang and norm_query(r.get("q_private", "")) not in train_norm]
        rng.shuffle(cands)
        for r in cands[:5]:
            add(r.get("family_id") or f"benign_{r.get('response_id')}", "e1_frozen_benign",
                r.get("response_id", ""), lang, "benign", "safe", norm_query(r.get("q_private", "")), "e1_frozen")

    # ---- order + ids + checks ----
    rows.sort(key=lambda r: (STRATUM_ORDER.index(r["prompt_stratum"]), r["language"], r["source_id"]))
    for i, r in enumerate(rows, 1):
        r["prompt_id"] = f"e6_{i:04d}"
    assert len(rows) == 200, len(rows)
    # quota checks
    from collections import Counter
    by_lang = Counter(r["language"] for r in rows)
    by_risk = Counter(r["prompt_risk_label"] for r in rows)
    by_stratum = Counter(r["prompt_stratum"] for r in rows)
    assert by_lang == {"zh": 100, "en": 100}, by_lang
    assert by_risk == {"unsafe": 100, "safe": 100}, by_risk
    from collections import Counter as _C
    _dup = [k for k, v in _C(r["user_query"] for r in rows).items() if v > 1]
    if _dup:
        for _q in _dup:
            print("DUP:", _q[:90])
            for _r in rows:
                if _r["user_query"] == _q:
                    print("   ", _r["prompt_id"], _r["prompt_stratum"], _r["language"], _r["source_id"])
    assert len({r["user_query"] for r in rows}) == 200
    write_jsonl(DATA_DIR / "exp6_prompt_manifest.jsonl", rows)
    write_json(DATA_DIR / "dataset_manifest.json", {
        "n": len(rows), "by_language": dict(by_lang), "by_risk": dict(by_risk),
        "by_stratum": {k: by_stratum[k] for k in STRATUM_ORDER},
        "manifest_sha256": manifest_sha256(rows), "seed": SEED, "protocol": "E6-DIRECT-API-v1.0-50CNY",
    })
    print("manifest rows:", len(rows), dict(by_lang), dict(by_risk))
    print("by stratum:", dict(by_stratum))
    return rows

def audit_leakage(rows):
    train = read_jsonl(TRAIN_MANIFEST)
    train_norm = {norm_query(r.get("user_query", "")) for r in train}
    train_fams = {r.get("template_family_id") or r.get("group_id") for r in train if r.get("template_family_id") or r.get("group_id")}
    exact_overlap, fam_overlap, prefix_overlap = [], [], []
    for r in rows:
        nq = norm_query(r["user_query"])
        if nq in train_norm:
            exact_overlap.append(r["prompt_id"])
        fam = r.get("prompt_family")
        if fam and fam in train_fams:
            fam_overlap.append((r["prompt_id"], fam))
        # near-dup: normalized 60-char prefix match
        pref = nq[:60]
        if any(tq.startswith(pref) or pref in tq[:80] for tq in train_norm if len(tq) >= 20 and len(nq) >= 20):
            prefix_overlap.append(r["prompt_id"])
    audit = {
        "exact_q_overlap_with_student_train": exact_overlap,
        "prompt_family_overlap_with_student_train": fam_overlap,
        "near_dup_prefix_overlap_with_student_train": prefix_overlap,
        "source_distribution": dict(__import__("collections").Counter(r["source"] for r in rows)),
        "language_distribution": dict(__import__("collections").Counter(r["language"] for r in rows)),
        "prompt_risk_distribution": dict(__import__("collections").Counter(r["prompt_risk_label"] for r in rows)),
        "duplicate_q_count": 200 - len({r["user_query"] for r in rows}),
        "train_rows_audited": len(train),
    }
    write_json(DATA_DIR / "leakage_audit.json", audit)
    # census csv
    with open(DATA_DIR / "prompt_census.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "prompt_family", "prompt_family_key", "source", "source_id", "language", "prompt_stratum", "prompt_risk_label", "should_refuse", "split_provenance", "fraud_category", "q_hash"])
        for r in sorted(rows, key=lambda r: r["prompt_id"]):
            w.writerow([r["prompt_id"], r["prompt_family"], r["prompt_family_key"], r["source"], r["source_id"], r["language"], r["prompt_stratum"], r["prompt_risk_label"], r["should_refuse"], r["split_provenance"], r.get("fraud_category", ""), r["q_hash"]])
    print("audit:", {k: v for k, v in audit.items() if k in ("exact_q_overlap_with_student_train", "prompt_family_overlap_with_student_train", "near_dup_prefix_overlap_with_student_train", "duplicate_q_count")})
    return audit

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()
    if args.dump:
        pools = load_pools()
        rng = random.Random(SEED)
        or_rows = dedup_keep(pools["or"], lambda r: norm_query(r.get("query", "")).lower(), rng)
        rng.shuffle(or_rows)
        cats = {}
        for r in or_rows: cats.setdefault(r.get("category", "other"), []).append(r)
        chosen = []
        for c in sorted(cats):
            if len(chosen) >= 20: break
            chosen.append(cats[c][0])
        out = {"hard_safe_en": [{"source_id": r.get("prompt_id", ""), "query": norm_query(r.get("query", "")), "category": r.get("category")} for r in chosen[:20]]}
        u1s = dedup_keep(pools["u1_safe"], lambda r: norm_query(r.get("user_query", "")).lower(), rng)
        out["u1_safe_unique"] = {"en": [{"id": r["id"], "query": norm_query(r.get("user_query", ""))} for r in u1s if r.get("language") == "en"],
                                  "zh": [{"id": r["id"], "query": norm_query(r.get("user_query", ""))} for r in u1s if r.get("language") == "zh"]}
        write_json(ROOT / "scripts" / "e6_dump_candidates.json", out)
        print("dump done")
    else:
        rows = build_manifest()
        audit_leakage(rows)
