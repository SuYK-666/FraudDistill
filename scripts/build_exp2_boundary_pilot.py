# -*- coding: utf-8 -*-
"""Build the boundary-repair pilot manifest (guide sections 10-11).

New 620-row pilot that does NOT overlap the round-1/round-2 pilots:
  Aegis response 280 (generic FP 70 / news-risk FP 30 / actionable FN 80 /
                      partial-leakage FN 30 / TP 30 / TN 40)
  Fraud-R1         180 (judge-only roleplay 100 / common positive 20 /
                      common-safe roleplay 30 / assistant-safe 20 / t6-only 10)
  DNA              120 (FN 50 / FP 20 / TP 20 / TN 20 / pairs 10 -> capped by
                      data availability, topped up with TN controls)
  OR-Bench          40 (hard-safe disagreement 15 / over-refusal 10 /
                      toxic clean refusal 10 / rare direct/partial 5)

Sampling pool: boundary_repair_dev groups only (guide 10.2, stable hash
"exp2-boundary-"), excluding round-1 IDs, round-2 IDs, paper_holdout groups
and prompt-example IDs (the worked example in the agent prompts is sanitized,
so no sample can be matched; noted in the manifest).

Usage:
  python scripts/build_exp2_boundary_pilot.py --dry-run
  python scripts/build_exp2_boundary_pilot.py --seed 20260806
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

from frauddistill.exp2_cross_benchmark.paths import EXPERIMENT_DIR, FULL_MANIFEST, MANIFEST_DIR, TEACHER_T6_DIR
from frauddistill.exp2_static_repair.pilot_split import assign_split, is_dev, write_boundary_splits

PILOT_DIR = EXPERIMENT_DIR / "pilot"
PILOT_MANIFEST = PILOT_DIR / "boundary_pilot.jsonl"
SMOKE_MANIFEST = PILOT_DIR / "boundary_smoke.jsonl"

OLD_PRED_FILES = {
    "fraudr1": TEACHER_T6_DIR / "fraudr1_t6_predictions.jsonl",
    "orbench": TEACHER_T6_DIR / "orbench_t6_predictions.jsonl",
    "do_not_answer": TEACHER_T6_DIR / "do_not_answer_t6_predictions.jsonl",
    "aegis2": TEACHER_T6_DIR / "aegis2_t6_predictions.jsonl",
}
BASELINE_FILES = {
    "fraudr1": EXPERIMENT_DIR / "fraudr1" / "baseline_predictions" / "fraudr1_official_judge_predictions.jsonl",
    "orbench": EXPERIMENT_DIR / "orbench" / "baseline_predictions" / "orbench_response_checker_predictions.jsonl",
    "do_not_answer": EXPERIMENT_DIR / "do_not_answer" / "baseline_predictions" / "dna_longformer_harmful_predictions.jsonl",
    "aegis2": EXPERIMENT_DIR / "aegis2" / "baseline_predictions" / "aegis_nemoguard_predictions.jsonl",
}
ROUND1_FILE = PILOT_DIR / "archive" / "repair_pilot_predictions_v1_20260806.jsonl"
ROUND2_FILE = PILOT_DIR / "repair_pilot_predictions.jsonl"
PAPER_HOLDOUT_FILE = MANIFEST_DIR / "paper_holdout_ids.json"

# --- text heuristics for Aegis content-type strata (guide 11.1) -------------
GENERIC_MARKERS = [
    "is defined as", "refers to", "is a type of", "is a form of", "in general",
    "generally", "typically", "is used to", "is the practice of", "definition",
    "overview", "background", "means", "百科", "定义", "是指", "是一种", "通常",
    "一般", "概述", "背景", "科普", "concept", "meaning",
]
NEWS_MARKERS = [
    "reported", "according to", "in 20", "news", "article", "history",
    "historical", "in the past", "报道", "新闻", "历史上", "过去", "记者",
    "据", "报道称", "incident", "case study",
]
RISK_WARNING_MARKERS = [
    "warning", "risk", "danger", "should not", "do not", "avoid", "be careful",
    "important to note", "safety", "风险", "警告", "不要", "避免", "提醒",
    "谨防", "安全", "请勿",
]
ACTION_MARKERS = [
    "step", "steps", "first", "then", "next", "finally", "you can", "you should",
    "you need to", "go to", "click", "call", "send", "submit", "enter", "create",
    "make sure", "install", "set up", "setup", "download", "use", "provide",
    "步骤", "首先", "然后", "接着", "最后", "你可以", "你应该", "你需要",
    "点击", "拨打", "发送", "提交", "输入", "创建", "制作", "安装", "下载", "使用",
]
REFUSAL_MARKERS = [
    "i can't", "i cannot", "can't help", "cannot help", "i'm sorry", "unable to",
    "cannot assist", "won't", "refuse", "不能", "无法", "无法协助", "拒绝", "抱歉",
]

# --- fraud-r1 id parsing ------------------------------------------------------
def parse_fraudr1_meta(sample_id: str) -> dict:
    m = re.match(r"^fraudr1_(zh|en)_(base|levelup)_(\d+)(_roleplay)?$", sample_id)
    if not m:
        return {"language": "unknown", "level": "unknown", "roleplay": "unknown"}
    return {"language": "Chinese" if m.group(1) == "zh" else "English",
            "level": m.group(2), "roleplay": "roleplay" if m.group(4) else "assistant"}


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


def sample_quota(pool: list[str], n: int, rng: random.Random, key: str = "") -> list[str]:
    """Deterministic quota sampling with an optional stratification key."""
    if n <= 0 or not pool:
        return []
    if key:
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


def aegis_family_key(sid: str, mani: dict) -> str:
    cat = str(mani.get(sid, {}).get("official_category", "") or "")
    return cat.split(",")[0].strip() or "Uncategorized"


def aegis_label_source_key(sid: str, mani: dict) -> str:
    meta = mani.get(sid, {}).get("metadata") or {}
    return str(meta.get("response_label_source", "unknown"))


def aegis_len_key(sid: str, mani: dict, median: float) -> str:
    n = len(str(mani.get(sid, {}).get("answer", "") or ""))
    return "long" if n >= median else "short"


def classify_aegis(sid: str, mani: dict) -> str:
    """Content-type classification for the Aegis strata (guide 11.1).

    gold=1 rows -> actionable_unsafe / partial_leakage
    gold=0 rows -> generic_encyclopedic / news_history_risk
    """
    m = mani.get(sid, {})
    ans = str(m.get("answer", "") or "").lower()
    gold = m.get("official_gold_binary")
    a_hits = sum(1 for mk in ACTION_MARKERS if mk in ans)
    g_hits = sum(1 for mk in GENERIC_MARKERS if mk in ans)
    n_hits = sum(1 for mk in NEWS_MARKERS if mk in ans)
    r_hits = sum(1 for mk in RISK_WARNING_MARKERS if mk in ans)
    ref_hits = sum(1 for mk in REFUSAL_MARKERS if mk in ans)
    if gold == 1:
        if ref_hits >= 1 and a_hits < 3:
            return "partial_leakage"
        return "actionable_unsafe"
    # gold safe
    if n_hits >= 1 and n_hits >= g_hits:
        return "news_history_risk"
    if g_hits >= 1:
        return "generic_encyclopedic"
    if r_hits >= 1 and a_hits == 0:
        return "news_history_risk"  # risk-explanation bucket
    return "generic_encyclopedic"


def build_aegis(mani: dict, old: dict, base: dict, rng: random.Random) -> list[dict]:
    pool = []
    for sid, m in mani.items():
        if m.get("source") != "aegis2" or m.get("evaluation_mode") != "response":
            continue
        if not is_dev(m.get("group_id", sid)):
            continue
        if sid in _EXCLUDED or m.get("group_id", sid) in _EXCLUDED_GROUPS:
            continue
        pool.append(sid)
    by_type: dict[str, list[str]] = defaultdict(list)
    for sid in pool:
        by_type[classify_aegis(sid, mani)].append(sid)
    gold1 = [sid for sid in pool if mani[sid].get("official_gold_binary") == 1]
    gold0 = [sid for sid in pool if mani[sid].get("official_gold_binary") == 0]
    ans_lens = [len(str(mani[sid].get("answer", "") or "")) for sid in pool]
    median = sorted(ans_lens)[len(ans_lens) // 2] if ans_lens else 100

    def strat_key(sid):
        return f"{aegis_family_key(sid, mani)}|{aegis_label_source_key(sid, mani)}|{aegis_len_key(sid, mani, median)}"

    quotas = {
        "generic_encyclopedic": 70, "news_history_risk": 30,
        "actionable_unsafe": 80, "partial_leakage": 30,
        "tp_control": 30, "tn_control": 40,
    }
    selected: dict[str, list[str]] = {}
    for typ, n in quotas.items():
        if typ in ("generic_encyclopedic", "news_history_risk"):
            src = by_type.get(typ, [])
            selected[typ] = sample_quota(src, n, rng, strat_key)
        elif typ in ("actionable_unsafe", "partial_leakage"):
            src = by_type.get(typ, [])
            selected[typ] = sample_quota(src, n, rng, strat_key)
    used = set().union(*selected.values()) if selected else set()
    selected["tp_control"] = sample_quota([s for s in gold1 if s not in used], 30, rng, strat_key)
    used.update(selected["tp_control"])
    selected["tn_control"] = sample_quota([s for s in gold0 if s not in used], 40, rng, strat_key)
    used.update(selected["tn_control"])
    # top-up any short strata from the same gold class (documented in notes)
    notes: dict[str, str] = {}
    for typ in ("generic_encyclopedic", "news_history_risk"):
        if len(selected[typ]) < quotas[typ]:
            src = [s for s in gold0 if s not in used]
            add = sample_quota(src, quotas[typ] - len(selected[typ]), rng, strat_key)
            selected[typ] += add
            used.update(add)
            notes[typ] = f"topup {len(add)} from gold-safe pool"
    for typ in ("actionable_unsafe", "partial_leakage"):
        if len(selected[typ]) < quotas[typ]:
            src = [s for s in gold1 if s not in used]
            add = sample_quota(src, quotas[typ] - len(selected[typ]), rng, strat_key)
            selected[typ] += add
            used.update(add)
            notes[typ] = f"topup {len(add)} from gold-unsafe pool"
    rows = []
    for typ, sids in selected.items():
        for sid in sids:
            rows.append(pilot_row(mani, sid, f"aegis_{typ}", old.get(sid), base.get(sid),
                                  notes.get(typ, "")))
    return rows


def build_fraudr1(mani: dict, old: dict, base: dict, rng: random.Random) -> list[dict]:
    def legal(sid):
        m = mani.get(sid, {})
        return (m.get("source") == "fraudr1" and is_dev(m.get("group_id", sid))
                and sid not in _EXCLUDED and m.get("group_id", sid) not in _EXCLUDED_GROUPS)

    def fam(sid):
        return str(mani.get(sid, {}).get("official_category", "?"))

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
    t6_only = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 0
               and old[sid].get("prediction_binary") == 1]
    jo_as = [sid for sid in old if legal(sid) and base.get(sid, {}).get("prediction_binary") == 1
             and old[sid].get("prediction_binary") == 0 and not rp(sid)]

    # 100 judge-only roleplay rows spread across the five families
    by_fam: dict[str, list[str]] = defaultdict(list)
    for sid in jo_rp:
        by_fam[fam(sid)].append(sid)
    fam_quotas = {"Phishing Scams": 30, "Fake Job Postings": 20, "Impersonation": 20,
                  "Fraudulent Services": 16, "Online Relationships": 14}
    sel_jo = []
    for f, n in fam_quotas.items():
        sel_jo += sample_quota(by_fam.get(f, []), n, rng, lambda s: parse_fraudr1_meta(s)["language"])
    # top-up short families round-robin from remaining judge-only roleplay
    if len(sel_jo) < 100:
        rest = [s for s in jo_rp if s not in sel_jo]
        rng.shuffle(rest)
        sel_jo += rest[: 100 - len(sel_jo)]

    sel_cp = sample_quota(cp, 20, rng, fam)
    sel_sr = sample_quota(safe_rp, 30, rng, fam)
    sel_sa = sample_quota(safe_as, 20, rng, fam)
    # t6-only (only 4 exist in the dev pool) + assistant judge-only top-up
    sel_t6 = list(sample_quota(t6_only, 10, rng, fam))
    if len(sel_t6) < 10:
        rest = [s for s in jo_as if s not in sel_t6]
        rng.shuffle(rest)
        sel_t6 += rest[: 10 - len(sel_t6)]

    def strata_of(sid):
        if sid in sel_jo:
            return f"fraudr1_judge_only_roleplay_{fam(sid).replace(' ', '_')}"
        if sid in sel_cp:
            return "fraudr1_common_positive"
        if sid in sel_sr:
            return "fraudr1_common_safe_roleplay"
        if sid in sel_sa:
            return "fraudr1_common_safe_assistant"
        return "fraudr1_t6_only_or_disagreement"

    selected = sel_jo + sel_cp + sel_sr + sel_sa + sel_t6
    notes = {}
    if len(sel_t6) < 10:
        notes["fraudr1_t6_only_or_disagreement"] = (
            f"only {len([s for s in t6_only if s in sel_t6])} t6-only rows exist in dev pool; "
            f"topped up with assistant judge-only disagreement rows")
    return [pilot_row(mani, sid, strata_of(sid), old.get(sid), base.get(sid), notes.get(strata_of(sid), ""))
            for sid in selected]


def build_dna(mani: dict, old: dict, base: dict, rng: random.Random) -> list[dict]:
    def legal(sid):
        m = mani.get(sid, {})
        return (m.get("source") == "do_not_answer" and is_dev(m.get("group_id", sid))
                and sid not in _EXCLUDED and m.get("group_id", sid) not in _EXCLUDED_GROUPS)

    def model_key(sid):
        return str(mani.get(sid, {}).get("target_model", "?"))

    fn = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 1
          and old[sid].get("prediction_binary") == 0]
    fp = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 0
          and old[sid].get("prediction_binary") == 1]
    tp = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 1
          and old[sid].get("prediction_binary") == 1]
    tn = [sid for sid in old if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 0
          and old[sid].get("prediction_binary") == 0]
    # same-prompt contrasting pairs (need both gold classes in one group)
    by_group: dict[str, list[str]] = defaultdict(list)
    for sid in fn + tp + fp + tn:
        by_group[mani.get(sid, {}).get("group_id", sid)].append(sid)
    pair_groups = [g for g, v in by_group.items()
                   if any(mani.get(s, {}).get("official_gold_binary") == 1 for s in v)
                   and any(mani.get(s, {}).get("official_gold_binary") == 0 for s in v)]
    rng.shuffle(pair_groups)
    contrast: list[str] = []
    for g in pair_groups:
        if len(contrast) >= 20:
            break
        members = by_group[g]
        uns = sorted([s for s in members if mani.get(s, {}).get("official_gold_binary") == 1],
                     key=lambda s: 0 if s in fn else 1)
        saf = [s for s in members if mani.get(s, {}).get("official_gold_binary") == 0]
        if uns and saf:
            contrast.append(uns[0])
            if len(contrast) < 20:
                contrast.append(saf[0])
    contrast = list(dict.fromkeys(contrast))[:20]

    sel_fn = sample_quota(fn, 50, rng, model_key)
    sel_fp = sample_quota(fp, 20, rng, model_key)
    sel_tp = sample_quota(tp, 20, rng, model_key)
    sel_tn = sample_quota(tn, 20, rng, model_key)
    used = set(sel_fn) | set(sel_fp) | set(sel_tp) | set(sel_tn) | set(contrast)
    topup = sample_quota([s for s in tn if s not in used], 120 - len(used), rng, model_key)

    def strata_of(sid):
        if sid in sel_fn:
            return "dna_false_negative"
        if sid in sel_fp:
            return "dna_false_positive"
        if sid in sel_tp:
            return "dna_tp_control"
        if sid in sel_tn:
            return "dna_tn_control"
        if sid in contrast:
            return "dna_same_prompt_pair"
        return "dna_tn_fpr_control"

    notes = {}
    if not fn:
        notes["dna_false_negative"] = (
            "0 gold-unsafe (FN) rows remain in the non-overlapping dev pool: "
            "round-2 consumed every legal DNA positive; boundary FN stratum empty")
    if not tp:
        notes["dna_tp_control"] = "0 TP controls remain (round-2 consumed all DNA positives)"
    if not contrast:
        notes["dna_same_prompt_pair"] = "0 contrasting pairs remain (no gold-unsafe rows left)"
    selected = sel_fn + sel_fp + sel_tp + sel_tn + contrast + topup
    return [pilot_row(mani, sid, strata_of(sid), old.get(sid), base.get(sid), notes.get(strata_of(sid), ""))
            for sid in selected]


def build_orbench(mani: dict, old: dict, base: dict, rng: random.Random) -> list[dict]:
    def legal(sid):
        m = mani.get(sid, {})
        return (m.get("source") == "orbench" and is_dev(m.get("group_id", sid))
                and sid not in _EXCLUDED and m.get("group_id", sid) not in _EXCLUDED_GROUPS)

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
    rare = [sid for sid in toxic if not chk_refused(sid)] + \
           [sid for sid in toxic if base.get(sid, {}).get("checker_class") == "direct_refusal"]

    sel_hard = sample_quota(hard_dis, 15, rng)
    sel_over = sample_quota([s for s in over_ref if s not in sel_hard], 10, rng)
    sel_toxic = sample_quota(toxic_clean, 10, rng)
    sel_rare = list(dict.fromkeys(sample_quota(rare, 5, rng)))
    if len(sel_rare) < 5:
        rest = [s for s in toxic if s not in sel_rare and s not in sel_toxic]
        rng.shuffle(rest)
        sel_rare += rest[: 5 - len(sel_rare)]

    def strata_of(sid):
        if sid in sel_hard:
            return "or_hard_safe_disagreement"
        if sid in sel_over:
            return "or_over_refusal_candidate"
        if sid in sel_toxic:
            return "or_toxic_clean_refusal"
        return "or_toxic_rare_direct_partial"

    selected = sel_hard + sel_over + sel_toxic + sel_rare
    notes = {}
    direct_count = sum(1 for s in sel_rare if base.get(s, {}).get("checker_class") == "direct_answer")
    if direct_count < 2:
        notes["or_toxic_rare_direct_partial"] = (
            f"only {direct_count} true direct_answer rows exist in dev pool; topped up with direct_refusal rows")
    return [pilot_row(mani, sid, strata_of(sid), old.get(sid), base.get(sid), notes.get(strata_of(sid), ""))
            for sid in selected]


def pilot_row(mani: dict, sid: str, stratum: str, old_pred: dict | None, base_pred: dict | None,
              notes: str = "") -> dict:
    m = mani.get(sid, {})
    gold = m.get("official_gold_binary", m.get("official_response_label"))
    mode = m.get("evaluation_mode", "response")
    base_class = None
    if base_pred is not None:
        base_class = base_pred.get("checker_class")
    return {
        "sample_id": sid,
        "source": m.get("source", "?"),
        "group_id": m.get("group_id", sid),
        "query": m.get("query", ""),
        "answer": m.get("answer", ""),
        "has_response": m.get("has_response", True),
        "evaluation_mode": mode,
        "split": assign_split(m.get("group_id", sid)),
        "stratum": stratum,
        "notes": notes,
        "gold_binary": gold,
        "official_gold_type": m.get("official_gold_type"),
        "official_category": m.get("official_category", ""),
        "target_model": m.get("target_model"),
        "language": m.get("language"),
        "metadata": m.get("metadata") or {},
        "baseline_pred": None if base_pred is None else base_pred.get("prediction_binary"),
        "baseline_checker_class": base_class,
        "query_redacted": "REDACTED" in str(m.get("query", "")),
        "old_teacher_pred": None if old_pred is None else old_pred.get("prediction_binary"),
        "old_risk_score": None if old_pred is None else old_pred.get("risk_score"),
        "old_prediction_type": None if old_pred is None else old_pred.get("prediction_type"),
        "boundary_pilot": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global _EXCLUDED, _EXCLUDED_GROUPS
    mani = {r["sample_id"]: r for r in read_jsonl(FULL_MANIFEST)}
    write_boundary_splits(list(mani.values()), MANIFEST_DIR)
    r1 = load_ids(ROUND1_FILE) if ROUND1_FILE.exists() else set()
    r2 = load_ids(ROUND2_FILE) if ROUND2_FILE.exists() else set()
    holdout = set(json.loads(PAPER_HOLDOUT_FILE.read_text(encoding="utf-8")))
    _EXCLUDED = r1 | r2
    _EXCLUDED_GROUPS = holdout

    old: dict[str, dict[str, dict]] = {}
    base: dict[str, dict[str, dict]] = {}
    for b, f in OLD_PRED_FILES.items():
        old[b] = {str(r["id"]): r for r in read_jsonl(f)}
    for b, f in BASELINE_FILES.items():
        base[b] = {str(r["id"]): r for r in read_jsonl(f)}

    rng = random.Random(args.seed)
    aegis = build_aegis(mani, old["aegis2"], base["aegis2"], rng)
    fraudr1 = build_fraudr1(mani, old["fraudr1"], base["fraudr1"], rng)
    dna = build_dna(mani, old["do_not_answer"], base["do_not_answer"], rng)
    orbench = build_orbench(mani, old["orbench"], base["orbench"], rng)

    all_rows = aegis + fraudr1 + dna + orbench
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in all_rows:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        deduped.append(r)
    all_rows = deduped
    print(f"[boundary] total={len(all_rows)} (target 620)")
    print("  by source:", dict(Counter(r["source"] for r in all_rows)))
    print("  strata:")
    for k, v in sorted(Counter(r["stratum"] for r in all_rows).items()):
        print(f"    {k}: {v}")
    print(f"  overlaps round1/round2: {len(seen & (r1 | r2))}")
    print(f"  non-dev rows: {sum(1 for r in all_rows if r['split'] != 'boundary_repair_dev')}")
    if args.dry_run:
        return
    write_jsonl(PILOT_MANIFEST, all_rows)
    # smoke: first 10 rows per source (guide 16.1: 40 total)
    smoke: list[dict] = []
    per_src: dict[str, int] = defaultdict(int)
    for r in all_rows:
        if per_src[r["source"]] < 10:
            smoke.append(r)
            per_src[r["source"]] += 1
    write_jsonl(SMOKE_MANIFEST, smoke)
    print(f"[boundary] wrote {len(all_rows)} rows -> {PILOT_MANIFEST}")
    print(f"[boundary] wrote {len(smoke)} smoke rows -> {SMOKE_MANIFEST}")


if __name__ == "__main__":
    main()
