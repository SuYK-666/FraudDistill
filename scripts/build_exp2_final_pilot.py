# -*- coding: utf-8 -*-
"""Build the FINAL exp2 pilot manifest (final-pilot guide sections 4-5).

300 unique rows, all fresh (no overlap with round-1/round-2 pilots, boundary
repair dev, paper holdout, or the skills-gate pilot):
  Aegis official validation  160  (80 unsafe / 80 safe response-level,
                                   content-category stratified)
  Fraud-R1 holdout           120  (protocol positive roleplay 60 [Fake Job >=16,
                                   all five families], protocol safe roleplay 30,
                                   content positive 15, content safe 15)
  OR-Bench holdout            20  (hard-safe 8, regular-safe 4,
                                   toxic clean refusal 6, rare disagreement 2)

Smoke: 30 rows drawn from the 300 (aegis 16, fraudr1 12, or 2).

Outputs under experiments/exp2_prior_work_comparison/pilot/:
  final_pilot_manifest.jsonl   (300 rows)
  final_pilot_smoke.jsonl      (30 rows, subset of the manifest)
  final_pilot.jsonl            (300 rows, same as manifest)

Usage:
  python scripts/build_exp2_final_pilot.py --dry-run
  python scripts/build_exp2_final_pilot.py --aegis-validation 160 --fraudr1 120 --orbench 20 --seed 20260806
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

PILOT_DIR = EXPERIMENT_DIR / "pilot"
MANIFEST = PILOT_DIR / "final_pilot_manifest.jsonl"
SMOKE = PILOT_DIR / "final_pilot_smoke.jsonl"
ALL = PILOT_DIR / "final_pilot.jsonl"

ROUND1_FILE = PILOT_DIR / "archive" / "repair_pilot_predictions_v1_20260806.jsonl"
ROUND2_FILE = PILOT_DIR / "repair_pilot_predictions.jsonl"
SKILL_GATE_MANIFEST = PILOT_DIR / "skill_gate_manifest.jsonl"
PAPER_HOLDOUT_FILE = MANIFEST_DIR / "paper_holdout_ids.json"
BOUNDARY_DEV_FILE = MANIFEST_DIR / "boundary_repair_dev_ids.json"
BOUNDARY_HOLDOUT_FILE = MANIFEST_DIR / "boundary_repair_holdout_ids.json"

T6_FILES = {
    "fraudr1": TEACHER_T6_DIR / "fraudr1_t6_predictions.jsonl",
    "orbench": TEACHER_T6_DIR / "orbench_t6_predictions.jsonl",
}
BASELINE_FILES = {
    "fraudr1": EXPERIMENT_DIR / "fraudr1" / "baseline_predictions" / "fraudr1_official_judge_predictions.jsonl",
    "orbench": EXPERIMENT_DIR / "orbench" / "baseline_predictions" / "orbench_response_checker_predictions.jsonl",
}

AEGIS_UNSAFE_REQUIRED = [
    "PII/Privacy", "Profanity", "Violence", "Sexual", "Controlled/Regulated Substances",
    "Guns and Illegal Weapons", "Hate/Identity Hate", "Harassment", "Criminal Planning/Confessions",
]
FRAUD_FAMILIES = ["Phishing Scams", "Fake Job Postings", "Impersonation", "Fraudulent Services", "Online Relationships"]


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
        return {"language": "unknown", "level": "unknown", "roleplay": False}
    return {"language": "Chinese" if m.group(1) == "zh" else "English",
            "level": m.group(2), "roleplay": bool(m.group(4))}


# ---------------------------------------------------------------------------
# Aegis content buckets (safe side). English + Chinese markers; the function is
# used only for stratification, never for gold labels.
# ---------------------------------------------------------------------------
GENERIC_MARKERS = ["is defined as", "refers to", "is a type of", "is a form of", "in general",
                   "generally", "typically", "is used to", "is the practice of", "definition",
                   "overview", "background", "means", "concept", "meaning",
                   "是指", "定义为", "是一种", "通常", "一般", "概念", "含义", "百科", "是什么"]
NEWS_MARKERS = ["reported", "according to", "in 20", "news", "article", "history",
                "historical", "in the past", "incident", "case study", "报道", "新闻", "历史", "事件"]
ACTION_MARKERS = ["step", "steps", "first", "then", "next", "finally", "you can", "you should",
                  "you need to", "go to", "click", "call", "send", "submit", "enter", "create",
                  "make sure", "install", "set up", "setup", "download", "use", "provide",
                  "步骤", "首先", "然后", "最后", "可以", "需要", "点击", "发送", "提交", "下载"]
WARNING_MARKERS = ["warning", "risk", "danger", "should not", "do not", "avoid", "be careful",
                   "important to note", "safety", "警告", "风险", "危险", "不要", "避免", "注意", "安全"]
REFUSAL_MARKERS = ["i can't", "i cannot", "can't help", "cannot help", "i'm sorry", "unable to",
                   "cannot assist", "won't", "refuse", "不能", "无法", "无法协助", "不能帮助", "拒绝", "抱歉"]


def aegis_content_bucket(row: dict) -> str:
    ans = str(row.get("answer", "") or "").lower()
    a = sum(1 for m in ACTION_MARKERS if m in ans)
    g = sum(1 for m in GENERIC_MARKERS if m in ans)
    n = sum(1 for m in NEWS_MARKERS if m in ans)
    w = sum(1 for m in WARNING_MARKERS if m in ans)
    r = sum(1 for m in REFUSAL_MARKERS if m in ans)
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
        "split": "final_pilot",
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
    n_u, n_s = 80, 80

    def cat(sid: str) -> str:
        vc = str(by_id[sid]["official_category"])
        for c in AEGIS_UNSAFE_REQUIRED:
            if c in vc:
                return c
        return "other_unsafe"

    def quota_select(gold_pool: list[str], n: int, key) -> list[str]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for sid in gold_pool:
            buckets[key(sid)].append(sid)
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

    sel_unsafe = quota_select(unsafe, n_u, cat)
    # guarantee every required unsafe category appears at least once
    present = {cat(s) for s in sel_unsafe}
    for c in AEGIS_UNSAFE_REQUIRED:
        if c not in present:
            for sid in unsafe:
                if sid not in sel_unsafe and cat(sid) == c:
                    sel_unsafe[-1] = sid
                    present.add(c)
                    break
    sel_safe = quota_select(safe, n_s, lambda sid: aegis_content_bucket(by_id[sid]))
    selected = sel_unsafe + sel_safe
    rng.shuffle(selected)
    out = []
    for sid in selected:
        m = by_id[sid]
        out.append(pilot_row({sid: m}, sid, f"aegis_{aegis_content_bucket(m) if m['gold'] == 0 else cat(sid)}",
                             gold=m["gold"]))
    return out


def build_fraudr1(mani: dict, old: dict, base: dict, rng: random.Random,
                  hold: set[str], dev: set[str], paper: set[str], excl: set[str]) -> list[dict]:
    def legal(sid):
        m = mani.get(sid, {})
        return (m.get("source") == "fraudr1"
                and m.get("group_id", sid) in hold
                and m.get("group_id", sid) not in dev
                and m.get("group_id", sid) not in paper
                and sid not in excl)

    def fam(sid):
        return str(mani.get(sid, {}).get("official_category", "?"))

    def is_rp(sid):
        return parse_fraudr1_meta(sid)["roleplay"]

    jo = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 1 and is_rp(sid)]
    safe_rp = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 0 and is_rp(sid)]
    pos_as = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 1 and not is_rp(sid)]
    safe_as = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 0 and not is_rp(sid)]

    # protocol positive roleplay 60, Fake Job >= 16, all five families
    by_fam: dict[str, list[str]] = defaultdict(list)
    for sid in jo:
        by_fam[fam(sid)].append(sid)
    quota = {"Phishing Scams": 13, "Fake Job Postings": 16, "Impersonation": 11,
             "Fraudulent Services": 11, "Online Relationships": 9}
    sel_jo: list[str] = []
    for f, n in quota.items():
        sel_jo += sample_quota(by_fam.get(f, []), n, rng)
    if len(sel_jo) < 60:
        rest = [s for s in jo if s not in sel_jo]
        rng.shuffle(rest)
        sel_jo += rest[: 60 - len(sel_jo)]
    sel_jo = sel_jo[:60]
    assert len(sel_jo) == 60, f"not enough protocol positive roleplay rows: {len(sel_jo)}"
    fakes = [s for s in sel_jo if fam(s) == "Fake Job Postings"]
    assert len(fakes) >= 16, f"Fake Job quota not met: {len(fakes)}"

    sel_sr = sample_quota(safe_rp, 30, rng, fam)
    sel_cp = sample_quota(pos_as, 15, rng, fam)
    sel_ca = sample_quota(safe_as, 15, rng, fam)

    def strata_of(sid):
        if sid in sel_jo:
            return "fraudr1_protocol_positive_roleplay"
        if sid in sel_sr:
            return "fraudr1_protocol_safe_roleplay"
        if sid in sel_cp:
            return "fraudr1_content_positive"
        return "fraudr1_content_safe"

    # Ordering (guide 7 half-stop checks): first 60 fraud rows = 30 protocol
    # positive (family-spread, includes Fake Job) + 30 protocol safe, so a
    # mid-run check can see recall, specificity AND Fake Job coverage.
    # deterministic mix (no extra rng draws, keeps later selection stable)
    sel_jo.sort(key=lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    first_jo, rest_jo = sel_jo[:30], sel_jo[30:]
    selected = first_jo + sel_sr + rest_jo + sel_cp + sel_ca
    return [pilot_row(mani, sid, strata_of(sid), baseline=base.get(sid), old=old.get(sid))
            for sid in selected]


def build_orbench(mani: dict, old: dict, base: dict, rng: random.Random,
                  hold: set[str], dev: set[str], paper: set[str], excl: set[str]) -> list[dict]:
    def legal(sid):
        m = mani.get(sid, {})
        return (m.get("source") == "orbench"
                and m.get("group_id", sid) in hold
                and m.get("group_id", sid) not in dev
                and m.get("group_id", sid) not in paper
                and sid not in excl)

    hard = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_type") == "hard_safe"]
    regular = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_type") == "regular_safe"]
    toxic = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_type") == "toxic"]
    def chk_refused(sid):
        cls = base.get(sid, {}).get("checker_class")
        if cls:
            return cls in ("direct_refusal", "indirect_refusal")
        return bool(base.get(sid, {}).get("prediction_binary") == 1)

    # Final-pilot guide 4.3: strata are gold-type based.
    # hard-safe 8 / regular-safe 4 / toxic clean refusal 6 / rare disagreement 2.
    # Prefer rows where the frozen teacher disagreed (refused) so the pilot
    # still probes over-refusal boundaries; fall back to the general pool.
    hard_dis = [sid for sid in hard if old.get(sid, {}).get("prediction_binary") == 1]
    over_ref = [sid for sid in hard + regular if old.get(sid, {}).get("prediction_binary") == 1]
    toxic_clean = [sid for sid in toxic if chk_refused(sid)]
    rare = ([sid for sid in toxic if not chk_refused(sid)] +
            [sid for sid in toxic if base.get(sid, {}).get("checker_class") == "direct_refusal"])

    sel_hard = sample_quota(hard_dis, min(8, len(hard_dis)), rng)
    if len(sel_hard) < 8:
        sel_hard += sample_quota([s for s in hard if s not in sel_hard], 8 - len(sel_hard), rng)
    sel_reg = sample_quota([s for s in over_ref if s not in sel_hard], min(4, len(over_ref)), rng)
    if len(sel_reg) < 4:
        sel_reg += sample_quota([s for s in regular if s not in sel_hard and s not in sel_reg],
                                4 - len(sel_reg), rng)
    sel_toxic = sample_quota(toxic_clean, 6, rng)
    rare_excl = [s for s in rare if s not in sel_toxic]
    sel_rare = list(dict.fromkeys(sample_quota(rare_excl, 2, rng)))
    if len(sel_rare) < 2:
        sel_rare += [s for s in rare if s not in sel_toxic and s not in sel_rare][: 2 - len(sel_rare)]

    def strata_of(sid):
        if sid in sel_hard:
            return "or_hard_safe"
        if sid in sel_reg:
            return "or_regular_safe"
        if sid in sel_toxic:
            return "or_toxic_clean_refusal"
        return "or_toxic_rare_direct_partial"

    selected = sel_hard + sel_reg + sel_toxic + sel_rare
    return [pilot_row(mani, sid, strata_of(sid), baseline=base.get(sid), old=old.get(sid))
            for sid in selected]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aegis-validation", type=int, default=160)
    ap.add_argument("--fraudr1", type=int, default=120)
    ap.add_argument("--orbench", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    r1 = load_ids(ROUND1_FILE) if ROUND1_FILE.exists() else set()
    r2 = load_ids(ROUND2_FILE) if ROUND2_FILE.exists() else set()
    paper = set(json.loads(PAPER_HOLDOUT_FILE.read_text(encoding="utf-8")))
    dev = set(json.loads(BOUNDARY_DEV_FILE.read_text(encoding="utf-8")))
    hold = set(json.loads(BOUNDARY_HOLDOUT_FILE.read_text(encoding="utf-8")))
    sg = {r["sample_id"] for r in read_jsonl(SKILL_GATE_MANIFEST)}
    r12 = r1 | r2

    full = [json.loads(l) for l in FULL_MANIFEST.open(encoding="utf-8") if l.strip()]
    mani = {r["sample_id"]: r for r in full}

    old: dict[str, dict] = {}
    base: dict[str, dict] = {}
    for b, f in T6_FILES.items():
        old[b] = {str(r["id"]): r for r in read_jsonl(f)}
    for b, f in BASELINE_FILES.items():
        base[b] = {str(r["id"]): r for r in read_jsonl(f)}

    rng = random.Random(args.seed)
    raw_validation = json.loads((RAW_AEGIS / "validation.json").read_text(encoding="utf-8"))
    aegis = build_aegis(raw_validation, rng, r12 | sg)
    fraudr1 = build_fraudr1(mani, old["fraudr1"], base["fraudr1"], rng, hold, dev, paper, r12 | sg)
    orbench = build_orbench(mani, old["orbench"], base["orbench"], rng, hold, dev, paper, r12 | sg)

    all_rows = aegis + fraudr1 + orbench
    seen: set[str] = set()
    deduped = []
    for r in all_rows:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        deduped.append(r)
    all_rows = deduped

    print(f"[final-pilot] total={len(all_rows)} (target {args.aegis_validation + args.fraudr1 + args.orbench})")
    print("  by source:", dict(Counter(r["source"] for r in all_rows)))
    print("  strata:")
    for k, v in sorted(Counter(r["stratum"] for r in all_rows).items()):
        print(f"    {k}: {v}")

    assert len(aegis) == args.aegis_validation, "aegis quota mismatch"
    assert len(fraudr1) == args.fraudr1, "fraudr1 quota mismatch"
    assert len(orbench) == args.orbench, "orbench quota mismatch"
    assert not (seen & r12), "overlap with round1/round2"
    assert not (seen & sg), "overlap with skills-gate pilot"
    assert not (seen & paper), "overlap with paper_holdout"
    assert not (seen & dev), "overlap with boundary_repair_dev"
    av_sids = {r["sample_id"] for r in aegis}
    assert not (av_sids & {r["sample_id"] for r in full}), "aegis rows leaked from test split"
    # doc 4.4: every aegis row must come from the official validation split
    for r in aegis:
        assert str(r["sample_id"]).startswith("aegis_")
        assert r["split"] == "final_pilot"
    # Fake Job family coverage
    jo = [r for r in fraudr1 if r["stratum"] == "fraudr1_protocol_positive_roleplay"]
    fams = Counter(r["official_category"] for r in jo)
    print("  protocol families:", dict(fams))
    assert all(fams.get(f, 0) > 0 for f in FRAUD_FAMILIES), "five-family coverage missing"
    assert fams.get("Fake Job Postings", 0) >= 16, "Fake Job < 16"

    if args.dry_run:
        return

    write_jsonl(MANIFEST, all_rows)
    write_jsonl(ALL, all_rows)

    # smoke: 30 rows drawn from the 300, stratified across tracks:
    # aegis 16 (8 unsafe + 8 safe), fraudr1 12 (6 protocol pos + 3 protocol
    # safe + 2 content pos + 1 content safe), or 2 (1 hard + 1 toxic).
    def pick_from(pool: list[dict], n: int) -> list[dict]:
        out: list[dict] = []
        for r in pool:
            if len(out) >= n:
                break
            out.append(r)
        return out

    aegis_rows = [r for r in all_rows if r["source"] == "aegis2"]
    a_unsafe = pick_from([r for r in aegis_rows if r["gold"] == 1], 8)
    a_safe = pick_from([r for r in aegis_rows if r["gold"] == 0], 8)
    fr_rows = [r for r in all_rows if r["source"] == "fraudr1"]
    f_jo = pick_from([r for r in fr_rows if r["stratum"] == "fraudr1_protocol_positive_roleplay"], 6)
    f_sr = pick_from([r for r in fr_rows if r["stratum"] == "fraudr1_protocol_safe_roleplay"], 3)
    f_cp = pick_from([r for r in fr_rows if r["stratum"] == "fraudr1_content_positive"], 2)
    f_ca = pick_from([r for r in fr_rows if r["stratum"] == "fraudr1_content_safe"], 1)
    or_rows = [r for r in all_rows if r["source"] == "orbench"]
    o_hard = pick_from([r for r in or_rows if r["stratum"] == "or_hard_safe"], 1)
    o_tox = pick_from([r for r in or_rows if r["stratum"].startswith("or_toxic")], 1)
    smoke = a_unsafe + a_safe + f_jo + f_sr + f_cp + f_ca + o_hard + o_tox
    assert len(smoke) == 30, len(smoke)
    write_jsonl(SMOKE, smoke)
    print(f"[final-pilot] wrote manifest={len(all_rows)} smoke={len(smoke)}")


if __name__ == "__main__":
    main()