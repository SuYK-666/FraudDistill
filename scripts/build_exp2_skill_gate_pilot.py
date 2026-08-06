# -*- coding: utf-8 -*-
"""Build the skills-gate pilot manifest (guide sections 21-24, 33.4).

400-row fresh frozen pilot (no DNA, no overlap with prior rounds):
  Aegis official validation   180  (90 unsafe / 90 safe, content-type stratified)
  Fraud-R1 boundary holdout   140  (protocol positive roleplay 60 /
                                    protocol safe roleplay 40 /
                                    content positive 20 / content safe 20)
  OR-Bench holdout regression  40  (hard-safe disagreement 15 /
                                    over-refusal candidates 10 /
                                    toxic clean refusal 10 / rare 5)

Exclusions (guide 22.4): round-1 pilot ids, round-2 pilot ids,
boundary_repair_dev groups, paper_holdout groups. Aegis rows come from the
OFFICIAL validation split (data/raw/aegis/validation.json) only.

Outputs under experiments/exp2_prior_work_comparison/pilot/:
  skill_gate_manifest.jsonl   (400 rows, guide 22.5 fields)
  skill_gate_smoke.jsonl      (40: aegis 15 + fraud 15 + or 10)
  skill_gate_diagnostic.jsonl (80: aegis 40 + fraud 30 + or 10, disjoint smoke)
  skill_gate_main.jsonl       (320: manifest minus diagnostic)

Usage:
  python scripts/build_exp2_skill_gate_pilot.py --dry-run
  python scripts/build_exp2_skill_gate_pilot.py --seed 20260806
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_cross_benchmark.paths import EXPERIMENT_DIR, FULL_MANIFEST, MANIFEST_DIR, RAW_AEGIS, TEACHER_T6_DIR
from frauddistill.exp2_static_repair.pilot_split import assign_split

PILOT_DIR = EXPERIMENT_DIR / "pilot"
MANIFEST = PILOT_DIR / "skill_gate_manifest.jsonl"
SMOKE = PILOT_DIR / "skill_gate_smoke.jsonl"
DIAGNOSTIC = PILOT_DIR / "skill_gate_diagnostic.jsonl"
MAIN = PILOT_DIR / "skill_gate_main.jsonl"

ROUND1_FILE = PILOT_DIR / "archive" / "repair_pilot_predictions_v1_20260806.jsonl"
ROUND2_FILE = PILOT_DIR / "repair_pilot_predictions.jsonl"
PAPER_HOLDOUT_FILE = MANIFEST_DIR / "paper_holdout_ids.json"
BOUNDARY_DEV_FILE = MANIFEST_DIR / "boundary_repair_dev_ids.json"

T6_FILES = {
    "fraudr1": TEACHER_T6_DIR / "fraudr1_t6_predictions.jsonl",
    "orbench": TEACHER_T6_DIR / "orbench_t6_predictions.jsonl",
}
BASELINE_FILES = {
    "fraudr1": EXPERIMENT_DIR / "fraudr1" / "baseline_predictions" / "fraudr1_official_judge_predictions.jsonl",
    "orbench": EXPERIMENT_DIR / "orbench" / "baseline_predictions" / "orbench_response_checker_predictions.jsonl",
}

GENERIC_MARKERS = [
    "is defined as", "refers to", "is a type of", "is a form of", "in general",
    "generally", "typically", "is used to", "is the practice of", "definition",
    "overview", "background", "means", "??", "??", "??", "???", "??",
    "??", "??", "??", "??", "concept", "meaning",
]
NEWS_MARKERS = [
    "reported", "according to", "in 20", "news", "article", "history",
    "historical", "in the past", "??", "??", "???", "??", "??",
    "?", "???", "incident", "case study",
]
ACTION_MARKERS = [
    "step", "steps", "first", "then", "next", "finally", "you can", "you should",
    "you need to", "go to", "click", "call", "send", "submit", "enter", "create",
    "make sure", "install", "set up", "setup", "download", "use", "provide",
    "??", "??", "??", "??", "??", "???", "???", "???",
    "??", "??", "??", "??", "??", "??", "??", "??", "??", "??",
]
WARNING_MARKERS = [
    "warning", "risk", "danger", "should not", "do not", "avoid", "be careful",
    "important to note", "safety", "??", "??", "??", "??", "??",
    "??", "??", "??",
]
REFUSAL_MARKERS = [
    "i can't", "i cannot", "can't help", "cannot help", "i'm sorry", "unable to",
    "cannot assist", "won't", "refuse", "??", "??", "????", "??", "??",
]
FRAUD_CATEGORY_MARKERS = ["fraud", "deception", "scam", "phishing", "impersonation",
                          "??", "??", "??", "??", "??"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_ids(path: Path) -> set[str]:
    return {str(r["id"]) for r in read_jsonl(path)}


def sample_quota(pool: list[str], n: int, rng: random.Random, key=None) -> list[str]:
    if n <= 0 or not pool:
        return []
    if key is not None:
        buckets: dict[str, list[str]] = defaultdict(list)
        for i in pool:
            buckets[key(i)].append(i)
        out: list[str] = []
        names = sorted(buckets)
        rng.shuffle(names)
        per = {nm: (n * len(buckets[nm])) // len(pool) for nm in names}
        rem = n - sum(per.values())
        for nm in names:
            if rem <= 0:
                break
            per[nm] += 1
            rem -= 1
        for nm in names:
            rng.shuffle(buckets[nm])
            out.extend(buckets[nm][: per[nm]])
        if len(out) < n:
            rest = [i for i in pool if i not in out]
            rng.shuffle(rest)
            out.extend(rest[: n - len(out)])
        return out[:n]
    pool = list(pool)
    rng.shuffle(pool)
    return pool[:n]


def parse_fraudr1_meta(sample_id: str) -> dict:
    m = re.match(r"^fraudr1_(zh|en)_(base|levelup)_(\d+)(_roleplay)?$", sample_id)
    if not m:
        return {"language": "unknown", "level": "unknown", "roleplay": "unknown"}
    return {"language": "Chinese" if m.group(1) == "zh" else "English",
            "level": m.group(2), "roleplay": "roleplay" if m.group(4) else "assistant"}


def aegis_content_bucket(row: dict) -> str:
    ans = str(row.get("answer", "") or "").lower()
    gold = row.get("gold")
    cat = str(row.get("official_category", "") or "").lower()
    a = sum(1 for m in ACTION_MARKERS if m in ans)
    g = sum(1 for m in GENERIC_MARKERS if m in ans)
    n = sum(1 for m in NEWS_MARKERS if m in ans)
    w = sum(1 for m in WARNING_MARKERS if m in ans)
    r = sum(1 for m in REFUSAL_MARKERS if m in ans)
    fraud_cat = any(m in cat for m in FRAUD_CATEGORY_MARKERS)
    if gold == 1:
        if fraud_cat:
            return "unsafe_fraud_deception"
        if r >= 1 and a < 2:
            return "unsafe_refusal_leakage"
        if w >= 1 and a >= 1:
            return "unsafe_mixed_warning_harmful"
        if a >= 2:
            return "unsafe_actionable_instruction"
        return "unsafe_targeted_harmful"
    if n >= 1 and n >= g:
        return "safe_news_history"
    if r >= 1 and a == 0:
        return "safe_clean_refusal"
    if g >= 1:
        return "safe_generic_educational"
    if w >= 1 and a == 0:
        return "safe_warning_only"
    return "safe_other"


def pilot_row(mani: dict, sid: str, stratum: str, *, gold=None, notes="",
              baseline=None, old=None) -> dict:
    m = mani.get(sid, {})
    qy = str(m.get("query", "")) + "\n" + str(m.get("answer", ""))
    return {
        "sample_id": sid,
        "group_id": m.get("group_id", sid),
        "source": m.get("source", "?"),
        "track": m.get("evaluation_mode", "response"),
        "split": "skill_gate_pilot",
        "stratum": stratum,
        "query": m.get("query", ""),
        "answer": m.get("answer", ""),
        "gold": gold if gold is not None else m.get("gold", m.get("official_gold_binary")),
        "qy_hash": hashlib.sha256(qy.encode("utf-8")).hexdigest()[:16],
        "previously_used": False,
        "notes": notes,
        "official_gold_type": m.get("official_gold_type"),
        "official_category": m.get("official_category", ""),
        "official_response_label": m.get("official_response_label"),
        "evaluation_mode": m.get("evaluation_mode", "response"),
        "target_model": m.get("target_model"),
        "language": m.get("language"),
        "metadata": m.get("metadata") or {},
        "overlap_exp3": bool(m.get("overlap_exp3")),
        "baseline_pred": None if baseline is None else baseline.get("prediction_binary"),
        "baseline_checker_class": None if baseline is None else baseline.get("checker_class"),
        "old_teacher_pred": None if old is None else old.get("prediction_binary"),
    }


def build_aegis(raw: list[dict], rng: random.Random, excluded_ids: set[str]) -> list[dict]:
    rows = []
    for r in raw:
        resp = (r.get("response") or "").strip()
        rl = r.get("response_label")
        if not resp or rl in (None, ""):
            continue
        sid = f"aegis_{r['id']}"
        if sid in excluded_ids:
            continue
        rows.append({
            "sample_id": sid,
            "group_id": sid,
            "source": "aegis2",
            "evaluation_mode": "response",
            "query": str(r.get("prompt", "") or ""),
            "answer": resp,
            "gold": 1 if str(rl).lower().startswith("unsafe") else 0,
            "official_category": str(r.get("violated_categories", "") or ""),
            "official_gold_type": None,
            "official_response_label": 1 if str(rl).lower().startswith("unsafe") else 0,
            "language": "English",
            "target_model": "unknown",
            "overlap_exp3": False,
        })
    by_id = {r["sample_id"]: r for r in rows}
    unsafe = [r["sample_id"] for r in rows if r["gold"] == 1]
    safe = [r["sample_id"] for r in rows if r["gold"] == 0]

    def quota_select(gold_pool: list[str], n: int) -> list[str]:
        """Proportional allocation across content buckets (largest remainder),
        then top-up from the same gold class."""
        buckets: dict[str, list[str]] = defaultdict(list)
        for sid in gold_pool:
            buckets[aegis_content_bucket(by_id[sid])].append(sid)
        for lst in buckets.values():
            rng.shuffle(lst)
        shares = {b: n * len(lst) / max(len(gold_pool), 1) for b, lst in buckets.items()}
        alloc = {b: int(s) for b, s in shares.items()}
        rem = n - sum(alloc.values())
        order = sorted(shares, key=lambda b: (shares[b] - alloc[b], b), reverse=True)
        for b in order[:rem]:
            alloc[b] += 1
        out: list[str] = []
        for b in sorted(buckets):
            out.extend(buckets[b][: alloc[b]])
        if len(out) < n:
            rest = [s for s in gold_pool if s not in out]
            rng.shuffle(rest)
            out.extend(rest[: n - len(out)])
        return out

    sel_unsafe = quota_select(unsafe, 90)
    sel_safe = quota_select(safe, 90)
    selected = sel_unsafe + sel_safe
    rng.shuffle(selected)
    out = []
    for sid in selected:
        m = by_id[sid]
        out.append(pilot_row({sid: m}, sid, f"aegis_{aegis_content_bucket(m)}",
                             gold=m["gold"]))
    return out


def build_fraudr1(mani: dict, old: dict, base: dict, rng: random.Random,
                  hold: set[str], dev: set[str], paper: set[str], r12: set[str]) -> list[dict]:
    def legal(sid):
        m = mani.get(sid, {})
        return (m.get("source") == "fraudr1"
                and m.get("group_id", sid) in hold
                and m.get("group_id", sid) not in dev
                and m.get("group_id", sid) not in paper
                and sid not in r12)

    def fam(sid):
        return str(mani.get(sid, {}).get("official_category", "?"))

    def lang(sid):
        return parse_fraudr1_meta(sid)["language"]

    def rp(sid):
        return parse_fraudr1_meta(sid)["roleplay"] == "roleplay"

    jo_rp = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 1
             and old[sid].get("prediction_binary") == 0 and rp(sid)]
    cp = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 1
          and old[sid].get("prediction_binary") == 1]
    safe_rp = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 0
               and old[sid].get("prediction_binary") == 0 and rp(sid)]
    safe_as = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 0
               and old[sid].get("prediction_binary") == 0 and not rp(sid)]

    fam_quotas = {"Phishing Scams": 15, "Fake Job Postings": 12, "Impersonation": 12,
                  "Fraudulent Services": 11, "Online Relationships": 10}
    by_fam: dict[str, list[str]] = defaultdict(list)
    for sid in jo_rp:
        by_fam[fam(sid)].append(sid)
    sel_jo = []
    for f, n in fam_quotas.items():
        sel_jo += sample_quota(by_fam.get(f, []), n, rng, lang)
    if len(sel_jo) < 60:
        rest = [s for s in jo_rp if s not in sel_jo]
        rng.shuffle(rest)
        sel_jo += rest[: 60 - len(sel_jo)]
    sel_sr = sample_quota(safe_rp, 40, rng, fam)
    sel_cp = sample_quota(cp, 20, rng, fam)
    sel_sa = sample_quota(safe_as, 20, rng, fam)

    def strata_of(sid):
        if sid in sel_jo:
            return "fraudr1_protocol_positive_roleplay"
        if sid in sel_sr:
            return "fraudr1_protocol_safe_roleplay"
        if sid in sel_cp:
            return "fraudr1_content_positive"
        return "fraudr1_content_safe"

    selected = sel_jo + sel_sr + sel_cp + sel_sa
    return [pilot_row(mani, sid, strata_of(sid), baseline=base.get(sid), old=old.get(sid))
            for sid in selected]


def build_orbench(mani: dict, old: dict, base: dict, rng: random.Random,
                  hold: set[str], dev: set[str], paper: set[str], r12: set[str]) -> list[dict]:
    def legal(sid):
        m = mani.get(sid, {})
        return (m.get("source") == "orbench"
                and m.get("group_id", sid) in hold
                and m.get("group_id", sid) not in dev
                and m.get("group_id", sid) not in paper
                and sid not in r12)

    hard = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_type") == "hard_safe"]
    regular = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_type") == "regular_safe"]
    toxic = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_type") == "toxic"]
    hard_dis = [sid for sid in hard if old[sid].get("prediction_binary") == 1]
    over_ref = [sid for sid in hard + regular if old[sid].get("prediction_binary") == 1]

    def chk_refused(sid):
        cls = base.get(sid, {}).get("checker_class")
        if cls:
            return cls in ("direct_refusal", "indirect_refusal")
        return bool(base.get(sid, {}).get("prediction_binary") == 1)

    toxic_clean = [sid for sid in toxic if chk_refused(sid)]
    rare = ([sid for sid in toxic if not chk_refused(sid)] +
            [sid for sid in toxic if base.get(sid, {}).get("checker_class") == "direct_refusal"])
    sel_hard = sample_quota(hard_dis, 15, rng)
    sel_over = sample_quota([s for s in over_ref if s not in sel_hard], 10, rng)
    sel_toxic = sample_quota(toxic_clean, 10, rng)
    sel_rare = list(dict.fromkeys(sample_quota(rare, 5, rng)))

    def strata_of(sid):
        if sid in sel_hard:
            return "or_hard_safe_disagreement"
        if sid in sel_over:
            return "or_over_refusal_candidate"
        if sid in sel_toxic:
            return "or_toxic_clean_refusal"
        return "or_toxic_rare_direct_partial"

    selected = sel_hard + sel_over + sel_toxic + sel_rare
    return [pilot_row(mani, sid, strata_of(sid), baseline=base.get(sid), old=old.get(sid))
            for sid in selected]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    r1 = load_ids(ROUND1_FILE) if ROUND1_FILE.exists() else set()
    r2 = load_ids(ROUND2_FILE) if ROUND2_FILE.exists() else set()
    paper = set(json.loads(PAPER_HOLDOUT_FILE.read_text(encoding="utf-8")))
    dev = set(json.loads(BOUNDARY_DEV_FILE.read_text(encoding="utf-8")))
    r12 = r1 | r2

    full = [json.loads(l) for l in FULL_MANIFEST.open(encoding="utf-8") if l.strip()]
    mani = {r["sample_id"]: r for r in full}
    hold = set(json.loads((MANIFEST_DIR / "boundary_repair_holdout_ids.json").read_text(encoding="utf-8")))

    old: dict[str, dict] = {}
    base: dict[str, dict] = {}
    for b, f in T6_FILES.items():
        old[b] = {str(r["id"]): r for r in read_jsonl(f)}
    for b, f in BASELINE_FILES.items():
        base[b] = {str(r["id"]): r for r in read_jsonl(f)}

    rng = random.Random(args.seed)
    raw_validation = json.loads((RAW_AEGIS / "validation.json").read_text(encoding="utf-8"))
    aegis = build_aegis(raw_validation, rng, r12)
    fraudr1 = build_fraudr1(mani, old["fraudr1"], base["fraudr1"], rng, hold, dev, paper, r12)
    orbench = build_orbench(mani, old["orbench"], base["orbench"], rng, hold, dev, paper, r12)
    all_rows = aegis + fraudr1 + orbench

    seen: set[str] = set()
    deduped = []
    for r in all_rows:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        deduped.append(r)
    all_rows = deduped

    print(f"[skill-gate] total={len(all_rows)} (target 400)")
    print("  by source:", dict(Counter(r["source"] for r in all_rows)))
    print("  strata:")
    for k, v in sorted(Counter(r["stratum"] for r in all_rows).items()):
        print(f"    {k}: {v}")
    assert len(aegis) == 180 and len(fraudr1) == 140 and len(orbench) == 40, "quota mismatch"
    assert not (seen & r12), "overlap with round1/round2"
    assert not (seen & paper), "overlap with paper_holdout"
    assert not (seen & dev), "overlap with boundary_repair_dev"
    av_sids = {r["sample_id"] for r in aegis}
    assert not (av_sids & {r["sample_id"] for r in full}), "aegis rows leaked from test split"
    if args.dry_run:
        return

    write_jsonl(MANIFEST, all_rows)
    # diagnostic: 80 (aegis 40 + fraud 30 + or 10); smoke: 40 subset of main
    per_src = {"aegis2": 40, "fraudr1": 30, "orbench": 10}
    diagnostic = []
    counters = Counter()
    for r in all_rows:
        if counters[r["source"]] < per_src[r["source"]]:
            diagnostic.append(r)
            counters[r["source"]] += 1
    diag_ids = {r["sample_id"] for r in diagnostic}
    main_rows = [r for r in all_rows if r["sample_id"] not in diag_ids]
    smoke = []
    per_smoke = {"aegis2": 15, "fraudr1": 15, "orbench": 10}
    counters = Counter()
    for r in main_rows:
        if counters[r["source"]] < per_smoke[r["source"]]:
            smoke.append(r)
            counters[r["source"]] += 1
    write_jsonl(SMOKE, smoke)
    write_jsonl(DIAGNOSTIC, diagnostic)
    write_jsonl(MAIN, main_rows)
    print(f"[skill-gate] wrote manifest={len(all_rows)} smoke={len(smoke)} "
          f"diagnostic={len(diagnostic)} main={len(main_rows)}")


if __name__ == "__main__":
    main()
