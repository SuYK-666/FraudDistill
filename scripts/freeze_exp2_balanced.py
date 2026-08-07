# -*- coding: utf-8 -*-
"""Exp2 balanced diagnostics Phases 5-7: balanced selection, artifact audit, freeze.

Guide: FraudDistill_?????????????????.md sections 5-8, 11-14.
Selection uses ONLY: adjudicated gold, quotas, source/style balance, difficulty
and quality. It NEVER reads evaluator predictions (selection log records this).

Quotas (main table):
  Fraud 1,500 pos + 1,500 neg (dev 100 cases = 200 rows)
  OR    1,000 pos + 1,000 neg (dev 100 prompts = 200 rows)
  DNA   2,500 pos + 2,500 neg (833 prompts x 6 + 2 extra rows; dev 50 x 6)
  Aegis official 813 rows (394 unsafe / 419 safe), unchanged

Usage:
  python scripts/freeze_exp2_balanced.py plan      # selection + shortfall plan
  python scripts/freeze_exp2_balanced.py spares    # build spare pools + topup request
  python scripts/freeze_exp2_balanced.py freeze    # select + write manifests + digests
  python scripts/freeze_exp2_balanced.py audit     # artifact audit on frozen sets
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BASE = REPO / "experiments" / "exp2_prior_work_comparison" / "balanced_design"
GOLD_FILE = BASE / "audits" / "gold_candidates.jsonl"
PROMPTS = BASE / "prompts"
ARCHIVE = REPO / "experiments" / "exp2_prior_work_comparison" / "archive" / "natural_prevalence_20260806"
OUT = BASE
SEED = 20260806
RNG = random.Random(SEED)

FAMILIES = ["Fraudulent Services", "Impersonation", "Phishing Scams", "Fake Job Postings", "Online Relationships"]
OR_QUOTA = {"hard_safe": 400, "regular_safe": 300, "toxic": 300}
OR_DEV = {"hard_safe": 40, "regular_safe": 30, "toxic": 30}
DNA_TEST_PROMPTS = 833
DNA_DEV_PROMPTS = 50


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eligible(row: dict) -> bool:
    gb = row.get("gold_binary")
    if gb not in (0, 1):
        return False
    if row.get("ambiguous") or row.get("judgeA_ambiguous"):
        return False
    try:
        conf = float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    if conf < 0.70:
        return False
    if not (row.get("evidence_spans") or []):
        return False
    # conflict must be adjudicated
    if (row.get("judgeB_binary") is not None and row.get("judgeA_binary") is not None
            and int(row["judgeA_binary"]) != int(row["judgeB_binary"])
            and row.get("label_source") != "adjudicated"):
        return False
    # low-conf / ambiguous Judge A rows need adjudication
    if (row.get("judgeA_ambiguous")
            or (row.get("judgeA_conf") is not None and float(row["judgeA_conf"]) < 0.70)):
        if row.get("label_source") != "adjudicated":
            return False
    return True


def gold_type_ok(row: dict) -> bool:
    t = str(row.get("gold_type") or "")
    if row["gold_binary"] == 1:
        return t in ("fraud_assistance", "behavior_error", "harmful_response")
    return t == "safe"


# ---------------------------------------------------------------- plan
def plan() -> None:
    rows = read_jsonl(GOLD_FILE)
    if not rows:
        print("[plan] no gold_candidates.jsonl yet - run annotate merge first")
        return
    by_bench = defaultdict(list)
    for r in rows:
        by_bench[r["benchmark"]].append(r)
    summary = {}
    for b, grp in by_bench.items():
        elig = [r for r in grp if eligible(r) and gold_type_ok(r)]
        summary[b] = {
            "total_candidates": len(grp), "eligible": len(elig),
            "eligible_by_gold": Counter(r["gold_binary"] for r in elig),
        }
    for b in ("fraudr1", "orbench"):
        elig = [r for r in rows if r["benchmark"] == b and eligible(r) and gold_type_ok(r)]
        by_grp = defaultdict(list)
        for r in elig:
            by_grp[r["group_id"]].append(r)
        n_pair = sum(1 for g in by_grp.values() if any(x["gold_binary"] == 1 for x in g) and any(x["gold_binary"] == 0 for x in g))
        summary[b]["groups"] = {"pair": n_pair, "total": len(by_grp),
                                "needed": 1600 if b == "fraudr1" else 1100}
    b = "dna"
    elig = [r for r in rows if r["benchmark"] == b and eligible(r) and gold_type_ok(r)]
    by_grp = defaultdict(list)
    for r in elig:
        by_grp[r["group_id"]].append(r)
    n_full = sum(1 for g in by_grp.values() if sum(x["gold_binary"] == 1 for x in g) >= 3 and sum(x["gold_binary"] == 0 for x in g) >= 3)
    summary[b]["groups"] = {"full_3_3": n_full, "total": len(by_grp), "needed": 883}
    (BASE / "audits" / "selection_plan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- spares
def build_spares() -> dict:
    """Spare prompt pools for top-up generation (guide 11.2: ?????????)."""
    spares: dict[str, list[dict]] = {"fraudr1": [], "orbench": [], "dna": []}
    used_fraud = {r["case_id"] for r in read_jsonl(PROMPTS / "fraud_prompt_pool.jsonl")}
    full = read_jsonl(ARCHIVE / "manifests" / "full_manifest.jsonl")
    by_grp: dict[str, dict] = {}
    for r in full:
        if r["source"] != "fraudr1":
            continue
        g = r["group_id"]
        meta = r.get("metadata") or {}
        by_grp.setdefault(g, {"family": r["official_category"], "lang": r["language"], "queries": {}})
        v = (meta.get("fraudr1_variant"), meta.get("fraudr1_scenario"))
        by_grp[g]["queries"][v] = r["query"]
    for g, info in by_grp.items():
        if g in used_fraud:
            continue
        for (va, sc), q in info["queries"].items():
            if not q:
                continue
            spares["fraudr1"].append({"case_id": g, "family": info["family"], "language": info["lang"],
                                      "variant": va, "scenario": sc, "query": q})
    # extend from raw prompts.jsonl (Online Relationships etc. are scarce in the
    # old full manifest; the raw dataset provides 4,282 additional cases)
    fam_map = {"network friendship": "Online Relationships",
               "fake job posting": "Fake Job Postings",
               "phishing": "Phishing Scams",
               "fraudulent service": "Fraudulent Services",
               "impersonation": "Impersonation"}
    raw = read_jsonl(REPO / "data" / "raw" / "fraudr1" / "prompts.jsonl")
    raw_used = {s["case_id"] for s in spares["fraudr1"]}
    for r in raw:
        fam = fam_map.get(str(r.get("fraud_category", "")).lower())
        if not fam or not r.get("user_query"):
            continue
        cid = f"raw_fraudr1_{r['id']}"
        if cid in raw_used or cid in used_fraud:
            continue
        spares["fraudr1"].append({"case_id": cid, "family": fam, "language": r.get("language", "English"),
                                  "variant": "raw", "scenario": "raw", "query": r["user_query"]})
        raw_used.add(cid)
    print(f"[spares] fraudr1 cases={len(by_grp) - len(used_fraud)} rows={len(spares['fraudr1'])}")

    used_or = {r["prompt_id"] for r in read_jsonl(PROMPTS / "or_prompt_pool.jsonl")}
    spec = [("hard_safe", REPO / "data" / "raw" / "or_bench" / "or-bench-hard-1k.csv", "hard"),
            ("regular_safe", REPO / "data" / "raw" / "or_bench" / "or-bench-80k.csv", "regular"),
            ("toxic", REPO / "data" / "raw" / "or_bench" / "or-bench-toxic.csv", "toxic")]
    for stratum, path, prefix in spec:
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = row.get("prompt") or row.get("question") or ""
                pid = f"orbench_{prefix}_{q}"
                if pid in used_or or not q:
                    continue
                spares["orbench"].append({"prompt_id": pid, "stratum": stratum, "query": q,
                                          "category": row.get("category", "")})
    print(f"[spares] orbench rows={len(spares['orbench'])}")

    used_dna = {r["prompt_id"] for r in read_jsonl(PROMPTS / "dna_prompt_pool.jsonl")}
    with (REPO / "data" / "raw" / "do_not_answer" / "data_en.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = f"dna_prompt_{row['id']}"
            if pid in used_dna or not row.get("question"):
                continue
            spares["dna"].append({"prompt_id": pid, "risk_area": row.get("risk_area", ""),
                                  "types_of_harm": row.get("types_of_harm", ""),
                                  "specific_harms": row.get("specific_harms", ""),
                                  "query": row["question"]})
    print(f"[spares] dna rows={len(spares['dna'])}")
    (BASE / "audits" / "spare_pools.json").write_text(json.dumps(spares, ensure_ascii=False, indent=1), encoding="utf-8")
    return spares


# ---------------------------------------------------------------- selection
def select_fraud(rows: list[dict], pools: dict) -> dict:
    elig = [r for r in rows if r.get("benchmark") == "fraudr1" and eligible(r) and gold_type_ok(r)]
    by_grp = defaultdict(list)
    for r in elig:
        by_grp[r["group_id"]].append(r)
    pool_meta = {r["case_id"]: r for r in pools["fraud"]}
    def pair_ok(vs: list[dict]) -> bool:
        pos = [x for x in vs if x["gold_binary"] == 1]
        neg = [x for x in vs if x["gold_binary"] == 0]
        for p in pos:
            for nn in neg:
                lp, ln = len(p["answer"]), len(nn["answer"])
                if lp and ln and min(lp, ln) / max(lp, ln) >= 0.67:
                    return True
        return False
    good = {g: vs for g, vs in by_grp.items() if pair_ok(vs)}
    REFUSAL_RE = re.compile(r"(cannot|can't|unable|sorry|apolog|not able|I can't|不能|无法|抱歉|对不起|帮不了|拒绝|不可以|不允许|无能为力)", re.I)
    DANGER_RE = re.compile(r"(scam|fraud|phishing|malware|hack|exploit|illegal|诈骗|欺诈|钓鱼|黑客|漏洞|非法|色情|武器|赌博|毒品|暴力|sex|porn|weapon)", re.I)
    def row_meta(g: str) -> tuple[str, str]:
        r0 = by_grp[g][0]
        fam = r0.get("family") or pool_meta.get(g, {}).get("family", "")
        lang = r0.get("language") or pool_meta.get(g, {}).get("language", "")
        return fam, lang
    def grp_style(g: str) -> tuple:
        # best length-matched pair's style score (topup pool makes style reachable)
        vs = by_grp[g]
        best = 9.0
        pos_c = [x for x in vs if x["gold_binary"] == 1]
        neg_c = [x for x in vs if x["gold_binary"] == 0]
        for p0 in pos_c:
            for nn in neg_c:
                if nn["answer"] == p0["answer"]:
                    continue
                lp, ln = len(p0["answer"]), len(nn["answer"])
                if not lp or not ln:
                    continue
                ratio = min(lp, ln) / max(lp, ln)
                if ratio < 0.67:
                    continue
                sc = (ratio - 1.0) ** 2
                if REFUSAL_RE.search(p0["answer"]):
                    sc -= 0.04
                if DANGER_RE.search(p0["answer"]):
                    sc -= 0.02
                if not DANGER_RE.search(nn["answer"]):
                    sc -= 0.04
                if sc < best:
                    best = sc
        return (best, RNG.random())
    # test first: style-prioritized order, hard caps 300/family, 750/language
    order = sorted(good, key=grp_style)
    test_cases: list[str] = []
    fam_cnt = Counter(); lang_cnt = Counter()
    for g in order:
        if len(test_cases) >= 1500:
            break
        fam, lang = row_meta(g)
        if fam_cnt[fam] >= 300 or lang_cnt[lang] >= 800:
            continue
        test_cases.append(g); fam_cnt[fam] += 1; lang_cnt[lang] += 1
    # dev: 100 cases from the remainder, soft caps 20/family + 50/language,
    # then fall back to any remaining case so dev is always full
    rest = [g for g in order if g not in test_cases]
    dev_cases: list[str] = []
    dev_fam = Counter(); dev_lang = Counter()
    for g in rest:
        if len(dev_cases) >= 100:
            break
        fam, lang = row_meta(g)
        if dev_fam[fam] < 20 and dev_lang[lang] < 50:
            dev_cases.append(g); dev_fam[fam] += 1; dev_lang[lang] += 1
    for g in rest:
        if len(dev_cases) >= 100:
            break
        if g not in dev_cases:
            dev_cases.append(g)
    return {"dev_cases": dev_cases, "test_cases": test_cases, "by_grp": good}


def select_or(rows: list[dict], pools: dict) -> dict:
    elig = [r for r in rows if r.get("benchmark") == "orbench" and eligible(r) and gold_type_ok(r)]
    by_grp = defaultdict(list)
    for r in elig:
        by_grp[r["group_id"]].append(r)
    pool_meta = {r["prompt_id"]: r for r in pools["or"]}
    def pair_ok(vs: list[dict]) -> bool:
        pos = [x for x in vs if x["gold_binary"] == 1]
        neg = [x for x in vs if x["gold_binary"] == 0]
        for p in pos:
            for nn in neg:
                lp, ln = len(p["answer"]), len(nn["answer"])
                if lp and ln and min(lp, ln) / max(lp, ln) >= 0.67:
                    return True
        return False
    good = {g: vs for g, vs in by_grp.items() if pair_ok(vs)}
    strata = defaultdict(list)
    for g in good:
        r0 = by_grp[g][0]
        stratum = r0.get("stratum") or pool_meta.get(g, {}).get("stratum", "hard_safe")
        strata[stratum].append(g)
    EXPLICIT_REFUSAL = re.compile(
        r"(cannot|can't|unable to|sorry|i'm sorry|i am unable|我不能|我无法|无法提供|拒绝回答|对不起|抱歉)", re.I)
    def no_kw_rank(g: str) -> int:
        if k in ("hard_safe", "regular_safe"):
            vs = good[g]
            for x in vs:
                if x["gold_binary"] == 1 and not EXPLICIT_REFUSAL.search(x["answer"]):
                    return 0
        return 1
    dev: dict[str, list[str]] = {k: [] for k in OR_DEV}
    test: dict[str, list[str]] = {k: [] for k in OR_QUOTA}
    for k in OR_DEV:
        pool = list(strata.get(k, []))
        RNG.shuffle(pool)
        dev[k] = pool[:OR_DEV[k]]
        rest = [g for g in pool if g not in set(dev[k])]
        rest.sort(key=lambda g: (no_kw_rank(g), RNG.random()))
        test[k] = rest[:OR_QUOTA[k]]
    return {"dev": dev, "test": test, "by_grp": good}


def select_dna(rows: list[dict], pools: dict) -> dict:
    elig = [r for r in rows if r.get("benchmark") == "dna" and eligible(r) and gold_type_ok(r)]
    by_grp = defaultdict(list)
    for r in elig:
        by_grp[r["group_id"]].append(r)
    pool_meta = {r["prompt_id"]: r for r in pools["dna"]}

    def counts(vs: list[dict]) -> tuple[int, int]:
        # distinct answer texts per side, so duplicate generator strings never
        # count toward the 3/3 per-prompt quota
        return (len({x["answer"] for x in vs if x["gold_binary"] == 1}),
                len({x["answer"] for x in vs if x["gold_binary"] == 0}))

    full = {g: vs for g, vs in by_grp.items() if min(counts(vs)) >= 3}
    # dev: prefer pool-designated dev prompts; fall back to any full prompt
    dev_pool = [g for g in full if pool_meta.get(g, {}).get("split") == "dev"]
    RNG.shuffle(dev_pool)
    dev = dev_pool[:DNA_DEV_PROMPTS]
    if len(dev) < DNA_DEV_PROMPTS:
        extra = [g for g in full if g not in dev]
        RNG.shuffle(extra)
        dev += extra[:DNA_DEV_PROMPTS - len(dev)]
    test_full = [g for g in full if g not in dev]
    RNG.shuffle(test_full)
    partial = [g for g, vs in by_grp.items() if g not in full and g not in dev
               and counts(vs)[0] >= 1 and counts(vs)[1] >= 1]
    partial.sort(key=lambda g: min(counts(by_grp[g])), reverse=True)
    test = test_full[:DNA_TEST_PROMPTS]
    if len(test) < DNA_TEST_PROMPTS:
        test += partial[:DNA_TEST_PROMPTS - len(test)]
    # pair plan: 3 per full group; adjust total to 833*3 + 2 (5,000 rows)
    plan: dict[str, int] = {}
    for g in test:
        p, n = counts(by_grp[g])
        plan[g] = min(p, n, 3)
    total = sum(plan.values())
    need = DNA_TEST_PROMPTS * 3 + 1  # 2499 pairs + 1 extra pair = 2500 pos + 2500 neg = 5000 rows
    if total < need:
        for g in sorted(test, key=lambda g: min(counts(by_grp[g])), reverse=True):
            if total >= need:
                break
            p, n = counts(by_grp[g])
            room = min(p, n, 6) - plan[g]
            add = min(room, need - total)
            plan[g] += add
            total += add
    elif total > need:
        for g in sorted(test, key=lambda g: plan[g], reverse=True):
            if total <= need:
                break
            if plan[g] > 1:
                plan[g] -= 1
                total -= 1
    return {"dev": dev, "test": test, "by_grp": by_grp, "plan": plan,
            "counts": {g: counts(by_grp[g]) for g in test}}


# ---------------------------------------------------------------- freeze outputs
def write_frozen(plan_out: dict, pools: dict, rows_by_sid: dict) -> None:
    test_rows: list[dict] = []
    dev_rows: list[dict] = []
    gold_test: list[dict] = []
    gold_dev: list[dict] = []
    excluded: list[dict] = []
    sel_log: list[dict] = []

    # global q+y dedup + within-group text dedup
    used_qy: set[str] = set()
    src_rank = {"adjudicated": 0, "double_agreed": 1, "single_A": 2, "missing": 3}
    def rank_key(r: dict) -> tuple:
        return (src_rank.get(str(r.get("label_source")), 3), -float(r.get("confidence") or 0.0))
    def pick_unique(vs: list[dict], want: int, k: int, group_texts: set[str],
                    split: str, prefer_score=None) -> list[dict]:
        out = []
        cand = [r for r in vs if r.get("gold_binary") == want]
        if prefer_score is None:
            cand.sort(key=rank_key)
        else:
            cand.sort(key=lambda r: (-prefer_score(r), *rank_key(r)))
        for r in cand:
            if len(out) >= k:
                break
            if r["qy_hash"] in used_qy:
                continue
            if r["answer"] in group_texts:
                continue
            out.append(r)
            group_texts.add(r["answer"])
        return out
    def take(m, gold_entry, split) -> None:
        (gold_test if split == "test" else gold_dev).append(gold_entry)
        (test_rows if split == "test" else dev_rows).append(m)
        used_qy.add(m["qy_hash"])
        sel_log.append({"phase": "freeze", "split": split, "sample_id": m["sample_id"], "group_id": m["group_id"],
                        "benchmark": m["source"], "gold_binary": gold_entry["gold_binary"],
                        "label_source": gold_entry.get("label_source")})
    def gold_entry(r: dict, g: str, benchmark: str, **extra) -> dict:
        e = {"sample_id": r["sample_id"], "group_id": g, "benchmark": benchmark, "source": benchmark,
             "gold_binary": r["gold_binary"], "gold_type": r["gold_type"],
             "behavior_subtype": r.get("judgeA_subtype") or r.get("behavior_subtype"),
             "confidence": r.get("confidence"), "evidence_spans": r.get("evidence_spans"),
             "label_source": r.get("label_source"), "intended_label": r.get("intended_label"),
             "qy_hash": r["qy_hash"]}
        e.update(extra)
        return e

    # Fraud (exactly 1 positive + 1 negative per case)
    REFUSAL_RE = re.compile(r"(cannot|can't|unable|sorry|apolog|not able|I can't|不能|无法|抱歉|对不起|帮不了|拒绝|不可以|不允许|无能为力)", re.I)
    DANGER_RE = re.compile(r"(scam|fraud|phishing|malware|hack|exploit|illegal|诈骗|欺诈|钓鱼|黑客|漏洞|非法|色情|武器|赌博|毒品|暴力|sex|porn|weapon)", re.I)
    EXPLICIT_REFUSAL_RE = re.compile(
        r"(cannot|can't|unable to|sorry|i'm sorry|i am unable|我不能|我无法|无法提供|拒绝回答|对不起|抱歉)", re.I)
    style = {"fraud_pos_refusal": 0, "fraud_pos_danger": 0, "fraud_neg_clean": 0,
             "dna_neg_danger": 0, "dna_neg_norefusal": 0, "dna_pos_refusal": 0,
             "dna_pos_danger": 0, "or_no_kw": 0}
    def fraud_case_rows(g: str, split: str) -> list[dict]:
        """Pick the (positive, negative) pair with the best length balance, then style."""
        vs = plan_out["fraud"]["by_grp"][g]
        group_texts: set[str] = set()
        pos_c = [r for r in vs if r["gold_binary"] == 1 and r["qy_hash"] not in used_qy]
        neg_c = [r for r in vs if r["gold_binary"] == 0 and r["qy_hash"] not in used_qy]
        best = None
        for p in pos_c:
            for nn in neg_c:
                if nn["answer"] == p["answer"]:
                    continue
                lp, ln = len(p["answer"]), len(nn["answer"])
                if not lp or not ln:
                    continue
                ratio = min(lp, ln) / max(lp, ln)
                if ratio < 0.67:
                    continue
                s_score = (ratio - 1.0) ** 2
                if split == "test":
                    if REFUSAL_RE.search(p["answer"]) and style["fraud_pos_refusal"] < 460:
                        s_score -= 0.04
                    if DANGER_RE.search(p["answer"]) and style["fraud_pos_danger"] < 520:
                        s_score -= 0.02
                    if not DANGER_RE.search(nn["answer"]) and style["fraud_neg_clean"] < 640:
                        s_score -= 0.04
                key = (s_score, *rank_key(p), *rank_key(nn))
                if best is None or key < best[0]:
                    best = (key, p, nn)
        if best is None:
            return []
        group_texts.update((best[1]["answer"], best[2]["answer"]))
        return [best[1], best[2]]

    # repair: fill fraud test up to 1,500 cases (dedup may have dropped cases).
    # iterate the style-ordered selection first so style quotas are honored.
    fraud_test_cases = {m["group_id"] for m in test_rows if m["source"] == "fraudr1"}
    fraud_dev_pre = set(plan_out["fraud"]["dev_cases"])
    if len(fraud_test_cases) < 1500:
        order = [g for g in plan_out["fraud"]["test_cases"] if g not in fraud_test_cases]
        order += [g for g in plan_out["fraud"]["by_grp"]
                  if g not in fraud_test_cases and g not in fraud_dev_pre and g not in order]
        fam_cnt = Counter()
        lang_cnt = Counter()
        for g in fraud_test_cases:
            vs = plan_out["fraud"]["by_grp"][g]
            fam_cnt[vs[0].get("family")] += 1
            lang_cnt[vs[0].get("language")] += 1
        for g in order:
            if len(fraud_test_cases) >= 1500:
                break
            if g in fraud_test_cases or g in fraud_dev_pre:
                continue
            vs = plan_out["fraud"]["by_grp"][g]
            fam = vs[0].get("family") or ""
            lang = vs[0].get("language") or ""
            if fam_cnt[fam] >= 300 or lang_cnt[lang] >= 800:
                continue
            chosen = fraud_case_rows(g, "test")
            if len(chosen) < 2:
                continue
            for r in chosen:
                m = {"sample_id": r["sample_id"], "source": "fraudr1", "group_id": g,
                     "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
                     "language": r.get("language"), "family": r.get("family"),
                     "stratum": r.get("stratum"), "source_type": r.get("source"), "split": "test"}
                take(m, gold_entry(r, g, "fraudr1"), "test")
            fam_cnt[fam] += 1; lang_cnt[lang] += 1
            fraud_test_cases.add(g)
    # dev repair: fill fraud dev up to 100 cases (dedup vs test may drop cases)
    fraud_dev_cases = {m["group_id"] for m in dev_rows if m["source"] == "fraudr1"}
    if len(fraud_dev_cases) < 100:
        order = sorted(plan_out["fraud"]["by_grp"], key=lambda g: RNG.random())
        for g in order:
            if len(fraud_dev_cases) >= 100:
                break
            if g in fraud_test_cases or g in fraud_dev_cases:
                continue
            chosen = fraud_case_rows(g, "dev")
            if len(chosen) < 2:
                continue
            for r in chosen:
                m = {"sample_id": r["sample_id"], "source": "fraudr1", "group_id": g,
                     "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
                     "language": r.get("language"), "family": r.get("family"),
                     "stratum": r.get("stratum"), "source_type": r.get("source"), "split": "dev"}
                take(m, gold_entry(r, g, "fraudr1"), "dev")
            fraud_dev_cases.add(g)

    # OR (exactly 1 behavior-error + 1 correct-behavior per prompt)
    def or_case_rows(g: str, split: str) -> list[dict]:
        """Pick the (behavior-error, correct-behavior) pair with best length balance."""
        vs = plan_out["or"]["by_grp"][g]
        group_texts: set[str] = set()
        pos_c = [r for r in vs if r["gold_binary"] == 1 and r["qy_hash"] not in used_qy]
        neg_c = [r for r in vs if r["gold_binary"] == 0 and r["qy_hash"] not in used_qy]
        best = None
        for p in pos_c:
            for nn in neg_c:
                if nn["answer"] == p["answer"]:
                    continue
                lp, ln = len(p["answer"]), len(nn["answer"])
                if not lp or not ln:
                    continue
                ratio = min(lp, ln) / max(lp, ln)
                if ratio < 0.67:
                    continue
                s_score = (ratio - 1.0) ** 2
                if split == "test":
                    if not EXPLICIT_REFUSAL_RE.search(p["answer"]) and style["or_no_kw"] < 190:
                        s_score -= 0.04
                key = (s_score, *rank_key(p), *rank_key(nn))
                if best is None or key < best[0]:
                    best = (key, p, nn)
        if best is None:
            return []
        group_texts.update((best[1]["answer"], best[2]["answer"]))
        return [best[1], best[2]]

    for split in ("test", "dev"):
        sel = plan_out["or"]["test"] if split == "test" else plan_out["or"]["dev"]
        for stratum, gs in sel.items():
            for g in gs:
                vs = plan_out["or"]["by_grp"][g]
                chosen = or_case_rows(g, split)
                if split == "test" and len(chosen) == 2 and not EXPLICIT_REFUSAL_RE.search(chosen[0]["answer"]):
                    style["or_no_kw"] += 1
                for r in chosen:
                    m = {"sample_id": r["sample_id"], "source": "orbench", "group_id": g,
                         "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
                         "language": r.get("language"), "stratum": r.get("stratum") or stratum,
                         "request_gold_type": stratum, "source_type": r.get("source"), "split": split}
                    take(m, gold_entry(r, g, "orbench", request_gold_type=stratum), split)
    # repair: fill OR test strata up to quota (dedup may have dropped prompts)
    for stratum, quota in OR_QUOTA.items():
        cur = len({m["group_id"] for m in test_rows if m["source"] == "orbench"
                   and (m.get("request_gold_type") == stratum or m.get("stratum") == stratum)})
        if cur >= quota:
            continue
        order = [g for g, gs in plan_out["or"]["by_grp"].items()
                 if (gs[0].get("stratum") or "") == stratum]
        RNG.shuffle(order)
        picked = {m["group_id"] for m in test_rows if m["source"] == "orbench"}
        dev_pre = {g for gs in plan_out["or"]["dev"].values() for g in gs}
        for g in order:
            if cur >= quota:
                break
            if g in picked or g in dev_pre:
                continue
            chosen = or_case_rows(g, "test")
            if len(chosen) < 2:
                continue
            for r in chosen:
                m = {"sample_id": r["sample_id"], "source": "orbench", "group_id": g,
                     "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
                     "language": r.get("language"), "stratum": r.get("stratum") or stratum,
                     "request_gold_type": stratum, "source_type": r.get("source"), "split": "test"}
                take(m, gold_entry(r, g, "orbench", request_gold_type=stratum), "test")
            cur += 1
    # dev repair: fill OR dev strata up to quota (dedup vs test may drop prompts)
    or_test_groups = {m["group_id"] for m in test_rows if m["source"] == "orbench"}
    or_dev_groups = {m["group_id"] for m in dev_rows if m["source"] == "orbench"}
    for stratum, quota in OR_DEV.items():
        cur = len({m["group_id"] for m in dev_rows if m["source"] == "orbench"
                   and m.get("request_gold_type") == stratum})
        if cur >= quota:
            continue
        order = [g for g, gs in plan_out["or"]["by_grp"].items()
                 if (gs[0].get("stratum") or "") == stratum]
        RNG.shuffle(order)
        for g in order:
            if cur >= quota:
                break
            if g in or_test_groups or g in or_dev_groups:
                continue
            chosen = or_case_rows(g, "dev")
            if len(chosen) < 2:
                continue
            for r in chosen:
                m = {"sample_id": r["sample_id"], "source": "orbench", "group_id": g,
                     "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
                     "language": r.get("language"), "stratum": r.get("stratum") or stratum,
                     "request_gold_type": stratum, "source_type": r.get("source"), "split": "dev"}
                take(m, gold_entry(r, g, "orbench", request_gold_type=stratum), "dev")
            cur += 1
            or_dev_groups.add(g)
    # DNA (multi-answer plan: k pairs per prompt, k>=3 where capacity allows)
    def dna_choose(vs: list[dict], k: int, group_texts: set[str], split: str) -> list[dict]:
        """Pick k positive + k negative answers with the best length-matched subset."""
        pos_c = [r for r in vs if r["gold_binary"] == 1 and r["qy_hash"] not in used_qy
                 and r["answer"] not in group_texts]
        neg_c = [r for r in vs if r["gold_binary"] == 0 and r["qy_hash"] not in used_qy
                 and r["answer"] not in group_texts]
        def pos_sort(r: dict) -> tuple:
            sc = 0.0
            if split == "test":
                if REFUSAL_RE.search(r["answer"]) and style["dna_pos_refusal"] < 780:
                    sc += 0.03
                if DANGER_RE.search(r["answer"]) and style["dna_pos_danger"] < 700:
                    sc += 0.02
            return (-sc, *rank_key(r))
        def neg_sort(r: dict) -> tuple:
            sc = 0.0
            if split == "test":
                if DANGER_RE.search(r["answer"]) and style["dna_neg_danger"] < 920:
                    sc += 0.05
                if not REFUSAL_RE.search(r["answer"]) and style["dna_neg_norefusal"] < 1680:
                    sc += 0.03
            return (-sc, *rank_key(r))
        pos_c.sort(key=pos_sort)
        neg_c.sort(key=neg_sort)
        # dedup by answer TEXT across both sides (generator sometimes puts the
        # same string in both the positive and negative slots)
        seen_text: set[str] = set()
        pos_dedup: list[dict] = []
        for r in pos_c:
            if r["answer"] not in seen_text:
                seen_text.add(r["answer"])
                pos_dedup.append(r)
        neg_dedup: list[dict] = []
        for r in neg_c:
            if r["answer"] not in seen_text:
                seen_text.add(r["answer"])
                neg_dedup.append(r)
        for limit in (8, 14):
            pc = pos_dedup[:limit]
            nc = neg_dedup[:limit]
            if len(pc) < k or len(nc) < k:
                continue
            best = None
            for psub in itertools.combinations(pc, k):
                pl = sorted(len(r["answer"]) for r in psub)
                for nsub in itertools.combinations(nc, k):
                    nl = sorted(len(r["answer"]) for r in nsub)
                    ratios = [min(a, b) / max(a, b) for a, b in zip(pl, nl)]
                    if min(ratios) < 0.67:
                        continue
                    sc = sum((x - 1.0) ** 2 for x in ratios)
                    key = (sc, tuple(rank_key(r) for r in psub), tuple(rank_key(r) for r in nsub))
                    if best is None or key < best[0]:
                        best = (key, list(psub), list(nsub))
            if best is not None:
                return best[1] + best[2]
        return []

    for split in ("test", "dev"):
        if split == "test":
            sel = plan_out["dna"]["test"]
            plan = plan_out["dna"].get("plan", {g: 3 for g in sel})
        else:
            sel = plan_out["dna"]["dev"]
            plan = {g: 3 for g in sel}
        for g in sel:
            vs = plan_out["dna"]["by_grp"][g]
            k = plan.get(g, 3)
            group_texts: set[str] = set()
            chosen = dna_choose(vs, k, group_texts, split)
            if split == "test":
                for r in chosen:
                    if r["gold_binary"] == 1:
                        if REFUSAL_RE.search(r["answer"]):
                            style["dna_pos_refusal"] += 1
                        if DANGER_RE.search(r["answer"]):
                            style["dna_pos_danger"] += 1
                    else:
                        if DANGER_RE.search(r["answer"]):
                            style["dna_neg_danger"] += 1
                        if not REFUSAL_RE.search(r["answer"]):
                            style["dna_neg_norefusal"] += 1
            for r in chosen:
                m = {"sample_id": r["sample_id"], "source": "do_not_answer", "group_id": g,
                     "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
                     "risk_area": r.get("risk_area"), "types_of_harm": r.get("types_of_harm"),
                     "specific_harms": r.get("specific_harms"), "source_type": r.get("source"),
                     "split": split}
                take(m, gold_entry(r, g, "do_not_answer"), split)
    # repair: raise DNA test pairs to 2,500 and prompts to 833
    def dna_counts(g: str) -> tuple[int, int]:
        vs = plan_out["dna"]["by_grp"].get(g, [])
        return (sum(1 for x in vs if x["gold_binary"] == 1),
                sum(1 for x in vs if x["gold_binary"] == 0))
    dna_test_groups = {m["group_id"] for m in test_rows if m["source"] == "do_not_answer"}
    dna_dev_groups = {m["group_id"] for m in dev_rows if m["source"] == "do_not_answer"}
    if len(dna_test_groups) < 833:
        order = sorted(plan_out["dna"]["by_grp"], key=lambda g: RNG.random())
        for g in order:
            if len(dna_test_groups) >= 833:
                break
            if g in dna_test_groups or g in dna_dev_groups:
                continue
            p, n = dna_counts(g)
            if p < 1 or n < 1:
                continue
            vs = plan_out["dna"]["by_grp"][g]
            plan_out["dna"]["test"].append(g)
            plan_out["dna"].setdefault("plan", {})[g] = min(p, n, 3)
            group_texts: set[str] = set()
            chosen = dna_choose(vs, min(p, n, 3), group_texts, "test")
            for r in chosen:
                m = {"sample_id": r["sample_id"], "source": "do_not_answer", "group_id": g,
                     "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
                     "risk_area": r.get("risk_area"), "types_of_harm": r.get("types_of_harm"),
                     "specific_harms": r.get("specific_harms"), "source_type": r.get("source"),
                     "split": "test"}
                take(m, gold_entry(r, g, "do_not_answer"), "test")
            dna_test_groups.add(g)
    # top-up pairs within selected groups up to the 2,500 target.
    # room is computed from DISTINCT answer texts available per side, so groups
    # whose candidates contain duplicate strings are never counted as having room.
    def dna_pos_neg() -> tuple[int, int]:
        pos = sum(1 for e in gold_test if e["benchmark"] == "do_not_answer" and e["gold_binary"] == 1)
        neg = sum(1 for e in gold_test if e["benchmark"] == "do_not_answer" and e["gold_binary"] == 0)
        return pos, neg
    def distinct_counts(g: str) -> tuple[int, int]:
        vs = plan_out["dna"]["by_grp"].get(g, [])
        return (len({x["answer"] for x in vs if x["gold_binary"] == 1}),
                len({x["answer"] for x in vs if x["gold_binary"] == 0}))
    stalled = set()
    while True:
        pos, neg = dna_pos_neg()
        if min(pos, neg) >= 2500:
            break
        extra_group = None
        if min(pos, neg) >= 2499:
            # one prompt supplies the extra pair (guide 7.1: 833*3 + 1 = 2,500)
            cand4 = [(g, min(distinct_counts(g))) for g in dna_test_groups
                     if g not in stalled and min(distinct_counts(g)) >= 4]
            if cand4:
                extra_group = max(cand4, key=lambda x: x[1])[0]
        best = None
        for g in dna_test_groups:
            if g in stalled:
                continue
            dp, dn = distinct_counts(g)
            have_p = sum(1 for e in gold_test if e["benchmark"] == "do_not_answer" and e["group_id"] == g and e["gold_binary"] == 1)
            have_n = sum(1 for e in gold_test if e["benchmark"] == "do_not_answer" and e["group_id"] == g and e["gold_binary"] == 0)
            cap = 4 if g == extra_group else 3
            room_p = min(cap, dp) - have_p
            room_n = min(cap, dn) - have_n
            score = min(room_p, room_n)
            if score > 0 and (best is None or score > best[0]):
                best = (score, g, room_p, room_n)
        if best is None:
            break
        _, g, room_p, room_n = best
        vs = plan_out["dna"]["by_grp"][g]
        group_texts = {m["answer"] for m in test_rows if m["source"] == "do_not_answer" and m["group_id"] == g}
        chosen = dna_choose(vs, min(room_p, room_n), group_texts, "test")
        if not chosen:
            stalled.add(g)
            continue
        for r in chosen:
            m = {"sample_id": r["sample_id"], "source": "do_not_answer", "group_id": g,
                 "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
                 "risk_area": r.get("risk_area"), "types_of_harm": r.get("types_of_harm"),
                 "specific_harms": r.get("specific_harms"), "source_type": r.get("source"),
                 "split": "test"}
            take(m, gold_entry(r, g, "do_not_answer"), "test")
            group_texts.add(r["answer"])
    # dev repair: fill DNA dev up to 50 groups x 6 rows (dedup vs test may drop)
    dna_dev_groups = {m["group_id"] for m in dev_rows if m["source"] == "do_not_answer"}
    if len(dna_dev_groups) < 50:
        order = sorted(plan_out["dna"]["by_grp"], key=lambda g: RNG.random())
        for g in order:
            if len(dna_dev_groups) >= 50:
                break
            if g in dna_test_groups or g in dna_dev_groups:
                continue
            dp, dn = distinct_counts(g)
            if dp < 3 or dn < 3:
                continue
            vs = plan_out["dna"]["by_grp"][g]
            group_texts: set[str] = set()
            chosen = dna_choose(vs, 3, group_texts, "dev")
            if len(chosen) < 6:
                continue
            for r in chosen:
                m = {"sample_id": r["sample_id"], "source": "do_not_answer", "group_id": g,
                     "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
                     "risk_area": r.get("risk_area"), "types_of_harm": r.get("types_of_harm"),
                     "specific_harms": r.get("specific_harms"), "source_type": r.get("source"),
                     "split": "dev"}
                take(m, gold_entry(r, g, "do_not_answer"), "dev")
            dna_dev_groups.add(g)
    # Aegis official (test only)
    aegis = read_jsonl(ARCHIVE / "manifests" / "aegis_response_test_813.jsonl")
    for r in aegis:
        m = {"sample_id": r["sample_id"], "source": "aegis2", "group_id": r["group_id"],
             "query": r["query"], "answer": r["answer"], "qy_hash": r["qy_hash"],
             "official_response_label": r.get("official_response_label"), "split": "test"}
        test_rows.append(m)
        gold_test.append({"sample_id": r["sample_id"], "group_id": r["group_id"], "benchmark": "aegis2",
                          "source": "aegis2",
                          "gold_binary": int(r.get("official_response_label") in (1, "unsafe")),
                          "gold_type": "harmful_response" if r.get("official_response_label") in (1, "unsafe") else "safe",
                          "label_source": "aegis_official", "qy_hash": r["qy_hash"]})

    # write manifests + gold
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    (OUT / "gold").mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "manifests" / "balanced_test_manifest.jsonl", test_rows)
    write_jsonl(OUT / "manifests" / "balanced_dev_manifest.jsonl", dev_rows)
    write_jsonl(OUT / "gold" / "balanced_test_gold.jsonl", gold_test)
    write_jsonl(OUT / "gold" / "balanced_dev_gold.jsonl", gold_dev)
    write_jsonl(OUT / "audits" / "selection_log.jsonl", sel_log)

    # digests + data card
    digests = {}
    for f in ("balanced_test_manifest.jsonl", "balanced_dev_manifest.jsonl",
              "balanced_test_gold.jsonl", "balanced_dev_gold.jsonl"):
        digests[f] = sha256_file(OUT / "manifests" / f if "manifest" in f else OUT / "gold" / f)
    counts = Counter((r["source"], r["gold_binary"]) for r in gold_test)
    card = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "design": "balanced response-level diagnostics (guide 2026-08-06)",
        "test_counts": {f"{k[0]}:{k[1]}": v for k, v in counts.items()},
        "test_rows": len(test_rows), "dev_rows": len(dev_rows),
        "gold_test_rows": len(gold_test), "gold_dev_rows": len(gold_dev),
        "digests": digests,
        "selection_rule": "adjudicated gold only; no evaluator predictions used",
    }
    (OUT / "manifests" / "data_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[freeze] test_rows={len(test_rows)} dev_rows={len(dev_rows)} gold_test={len(gold_test)} gold_dev={len(gold_dev)}")
    print(f"[freeze] counts={counts}")
    print(f"[freeze] digests={ {k: v[:10] for k, v in digests.items()} }")


# ---------------------------------------------------------------- freeze
def freeze() -> None:
    rows = read_jsonl(GOLD_FILE)
    if not rows:
        print("[freeze] no gold_candidates.jsonl - run annotate merge first")
        return
    pools = {k: read_jsonl(PROMPTS / f) for k, f in [
        ("fraud", "fraud_prompt_pool.jsonl"), ("or", "or_prompt_pool.jsonl"), ("dna", "dna_prompt_pool.jsonl")]}
    plan_out = {"fraud": select_fraud(rows, pools), "or": select_or(rows, pools), "dna": select_dna(rows, pools)}
    print(f"[freeze] fraud test_cases={len(plan_out['fraud']['test_cases'])} dev={len(plan_out['fraud']['dev_cases'])}")
    print(f"[freeze] or test={ {k: len(v) for k, v in plan_out['or']['test'].items()} } dev={ {k: len(v) for k, v in plan_out['or']['dev'].items()} }")
    print(f"[freeze] dna test={len(plan_out['dna']['test'])} dev={len(plan_out['dna']['dev'])}")
    short = {
        "fraudr1": {"test": 1500 - len(plan_out["fraud"]["test_cases"]), "dev": 100 - len(plan_out["fraud"]["dev_cases"])},
        "orbench": {k: OR_QUOTA[k] - len(v) for k, v in plan_out["or"]["test"].items()},
        "dna": {"test": DNA_TEST_PROMPTS - len(plan_out["dna"]["test"]), "dev": DNA_DEV_PROMPTS - len(plan_out["dna"]["dev"])},
    }
    (BASE / "audits" / "freeze_shortfall.json").write_text(json.dumps(short, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[freeze] shortfall:", json.dumps(short, ensure_ascii=False))
    total_short = sum(short["fraudr1"].values()) + sum(short["orbench"].values()) + sum(short["dna"].values())
    partial = "--partial" in sys.argv
    if total_short > 0 and not partial:
        spares = build_spares()
        req = {"fraudr1": [], "orbench": [], "dna": []}
        # fraud shortfall by family
        if short["fraudr1"]["test"] > 0 or short["fraudr1"]["dev"] > 0:
            need = short["fraudr1"]["test"] + short["fraudr1"]["dev"]
            RNG.shuffle(spares["fraudr1"])
            for s in spares["fraudr1"]:
                if len(req["fraudr1"]) >= need:
                    break
                req["fraudr1"].append(s)
        if sum(short["orbench"].values()) > 0:
            for stratum in OR_QUOTA:
                need = short["orbench"][stratum]
                pool = [s for s in spares["orbench"] if s["stratum"] == stratum]
                RNG.shuffle(pool)
                req["orbench"] += pool[:need]
        if short["dna"]["test"] > 0:
            need = short["dna"]["test"] + short["dna"]["dev"]
            RNG.shuffle(spares["dna"])
            req["dna"] = spares["dna"][:need]
        (BASE / "audits" / "topup_request.json").write_text(json.dumps(req, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[freeze] topup request written: fraudr1={len(req['fraudr1'])} orbench={len(req['orbench'])} dna={len(req['dna'])}")
        print("[freeze] run topup generation + judgeA/judgeB/adjudicate/merge, then freeze again")
        return
    write_frozen(plan_out, pools, {r["sample_id"]: r for r in rows})


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if action == "plan":
        plan()
    elif action == "spares":
        build_spares()
    elif action == "freeze":
        freeze()
    elif action == "audit":
        print("audit not implemented yet")
    else:
        raise SystemExit(f"unknown action {action}")


if __name__ == "__main__":
    main()
