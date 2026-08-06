# -*- coding: utf-8 -*-
"""Build the targeted-capability-repair pilot manifest (guide 13) and the
frozen test splits (guide 12.6).

Usage:
  python scripts/build_exp2_repair_pilot.py            # full build
  python scripts/build_exp2_repair_pilot.py --dry-run  # print stratum sizes only

Outputs:
  manifests/repair_dev_ids.json        group ids assigned to repair_dev
  manifests/paper_holdout_ids.json     group ids assigned to paper_holdout
  manifests/split_digest.json          hash + counts for the frozen split
  pilot/repair_pilot.jsonl             ~1400 stratified pilot rows (guide 13)

Pilot legality (guide 13.6): samples come only from repair_dev / official
validation / existing error pools, never from paper_holdout.
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

from frauddistill.exp2_cross_benchmark.paths import (  # noqa: E402
    EXPERIMENT_DIR,
    FULL_MANIFEST,
    MANIFEST_DIR,
    TEACHER_T6_DIR,
    RAW_AEGIS,
)

PILOT_DIR = EXPERIMENT_DIR / "pilot"
PILOT_MANIFEST = PILOT_DIR / "repair_pilot.jsonl"

OLD_PRED_FILES = {
    "fraudr1": TEACHER_T6_DIR / "fraudr1_t6_predictions.jsonl",
    "orbench": TEACHER_T6_DIR / "orbench_t6_predictions.jsonl",
    "do_not_answer": TEACHER_T6_DIR / "do_not_answer_t6_predictions.jsonl",
    "aegis2": TEACHER_T6_DIR / "aegis2_t6_predictions.jsonl",
    "aegis2_prompt": TEACHER_T6_DIR / "aegis2_t6_prompt_predictions.jsonl",
}
BASELINE_FILES = {
    "fraudr1": EXPERIMENT_DIR / "fraudr1" / "baseline_predictions" / "fraudr1_official_judge_predictions.jsonl",
    "orbench": EXPERIMENT_DIR / "orbench" / "baseline_predictions" / "orbench_response_checker_predictions.jsonl",
    "do_not_answer": EXPERIMENT_DIR / "do_not_answer" / "baseline_predictions" / "dna_longformer_harmful_predictions.jsonl",
    "aegis2": EXPERIMENT_DIR / "aegis2" / "baseline_predictions" / "aegis_nemoguard_predictions.jsonl",
}
AEGIS_VALIDATION_MANIFEST = MANIFEST_DIR / "aegis_validation_manifest.jsonl"
AEGIS_VALIDATION_PRED = TEACHER_T6_DIR / "aegis_validation_t6_predictions.jsonl"


def stable_bucket(group_id: str, modulo: int = 100) -> int:
    digest = hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def assign_split(group_id: str) -> str:
    bucket = stable_bucket(group_id)
    if bucket < 20:
        return "paper_holdout"
    if bucket < 40:
        return "repair_dev"
    return "descriptive_only"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_manifest() -> dict[str, dict]:
    return {r["sample_id"]: r for r in read_jsonl(FULL_MANIFEST)}


def sample_quota(pool: list[str], n: int, rng: random.Random, key: str = "") -> list[str]:
    """Deterministic quota sampling with an optional stratification key."""
    if n <= 0:
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
        for nm in names:  # distribute remainder round-robin
            if rem <= 0:
                break
            per[nm] += 1
            rem -= 1
        for nm in names:
            rng.shuffle(buckets[nm])
            out.extend(buckets[nm][: per[nm]])
        if len(out) < n:  # top-up from remaining pool
            rest = [i for i in pool if i not in out]
            rng.shuffle(rest)
            out.extend(rest[: n - len(out)])
        return out[:n]
    pool = list(pool)
    rng.shuffle(pool)
    return pool[:n]


def parse_fraudr1_meta(sample_id: str) -> dict:
    """fraudr1_<lang>_<level>_<num>[_roleplay] -> metadata dict."""
    m = re.match(r"^fraudr1_(zh|en)_(base|levelup)_(\d+)(_roleplay)?$", sample_id)
    if not m:
        return {"language": "unknown", "level": "unknown", "roleplay": "unknown"}
    return {"language": "Chinese" if m.group(1) == "zh" else "English",
            "level": m.group(2), "roleplay": "roleplay" if m.group(4) else "assistant"}


def fraudr1_balance_key(sample_id: str) -> str:
    meta = parse_fraudr1_meta(sample_id)
    return f"{meta['language']}|{meta['level']}|{meta['roleplay']}"


def balanced_quota(pool: list[str], n: int, rng: random.Random, key: str) -> list[str]:
    """Sample n rows while balancing the two values of `key` as evenly as the
    pool allows (e.g. Chinese/English 50/50 for fraudr1 judge-only strata)."""
    if n <= 0 or not pool:
        return []
    buckets: dict[str, list[str]] = defaultdict(list)
    for i in pool:
        buckets[key(i)].append(i)
    names = sorted(buckets)
    if len(names) < 2:
        return sample_quota(pool, n, rng)
    per = {nm: n // len(names) for nm in names}
    rem = n - sum(per.values())
    for nm in names:  # give the remainder to the larger bucket
        if rem <= 0:
            break
        if len(buckets[nm]) > per[nm]:
            per[nm] += 1
            rem -= 1
    out: list[str] = []
    for nm in names:
        rng.shuffle(buckets[nm])
        out.extend(buckets[nm][: per[nm]])
    if len(out) < n:
        rest = [i for i in pool if i not in out]
        rng.shuffle(rest)
        out.extend(rest[: n - len(out)])
    return out[:n]


def build_fraudr1(mani: dict, old: dict, base: dict, rng: random.Random) -> list[dict]:
    """Guide 13.1: 320 rows."""
    pool_jo: dict[str, list[str]] = defaultdict(list)
    for sid, r in old.items():
        if base.get(sid, {}).get("prediction_binary") != 1:
            continue
        if r.get("prediction_binary") != 0:
            continue
        m = mani.get(sid, {})
        if m.get("source") != "fraudr1" or assign_split(m.get("group_id", sid)) == "paper_holdout":
            continue
        pool_jo[m.get("official_category", "?")].append(sid)
    # judge-only quotas per family (guide 13.1)
    quotas = {"Fraudulent Services": 50, "Phishing Scams": 50, "Impersonation": 40,
              "Online Relationships": 40, "Fake Job Postings": 20}
    selected: list[str] = []
    for fam, n in quotas.items():
        # balance Chinese/English and base/levelup within each family; when one
        # language pool is short, fill the remainder from the other language.
        family_pool = pool_jo.get(fam, [])
        zh_pool = [i for i in family_pool if parse_fraudr1_meta(i)["language"] == "Chinese"]
        en_pool = [i for i in family_pool if parse_fraudr1_meta(i)["language"] == "English"]
        half = min(n // 2, len(zh_pool), len(en_pool))
        sel = balanced_quota(zh_pool, half, rng, lambda i: parse_fraudr1_meta(i)["level"])
        sel += balanced_quota(en_pool, half, rng, lambda i: parse_fraudr1_meta(i)["level"])
        rest = [i for i in family_pool if i not in sel]
        if len(sel) < n:
            rng.shuffle(rest)
            sel += rest[: n - len(sel)]
        selected += sel
    # common-positive / common-safe controls
    cp = [sid for sid, r in old.items()
          if base.get(sid, {}).get("prediction_binary") == 1 and r.get("prediction_binary") == 1
          and mani.get(sid, {}).get("source") == "fraudr1"
          and assign_split(mani.get(sid, {}).get("group_id", sid)) != "paper_holdout"]
    cs = [sid for sid, r in old.items()
          if base.get(sid, {}).get("prediction_binary") == 0 and r.get("prediction_binary") == 0
          and mani.get(sid, {}).get("source") == "fraudr1"
          and assign_split(mani.get(sid, {}).get("group_id", sid)) != "paper_holdout"]
    t6_only = [sid for sid, r in old.items()
               if base.get(sid, {}).get("prediction_binary") == 0 and r.get("prediction_binary") == 1
               and mani.get(sid, {}).get("source") == "fraudr1"
               and assign_split(mani.get(sid, {}).get("group_id", sid)) != "paper_holdout"]
    selected += sample_quota(cp, 50, rng, fraudr1_balance_key)
    selected += sample_quota(cs, 50, rng, fraudr1_balance_key)
    dis = sample_quota(t6_only, 20, rng, fraudr1_balance_key)
    # top-up disagreement from Phishing judge-only (largest family pool)
    if len(dis) < 20:
        used = set(selected) | set(dis)
        rest = [i for i in pool_jo.get("Phishing Scams", []) if i not in used]
        rng.shuffle(rest)
        dis += rest[: 20 - len(dis)]
    selected += dis
    def strata_of(sid):
        if sid in t6_only:
            return "fraudr1_t6_only"
        if sid in cp:
            return "fraudr1_common_positive"
        if sid in cs:
            return "fraudr1_common_safe"
        return f"fraudr1_judge_only_{mani.get(sid, {}).get('official_category', '?').replace(' ', '_')}"
    return [pilot_row(mani, sid, strata_of(sid), old.get(sid), base.get(sid)) for sid in selected]


def build_orbench(mani: dict, old: dict, base: dict, rng: random.Random) -> list[dict]:
    """Guide 13.2: 320 rows."""
    def gold(sid):
        return mani.get(sid, {}).get("official_gold_type")

    def did_refuse(r):
        a = r.get("arbiter_json") or {}
        v = a.get("did_refuse_score")
        if v is None:
            rr = r.get("agent_refusal_json") or {}
            v = rr.get("explicit_refusal", rr.get("refusal_strength", 0))
        try:
            return float(v) >= 0.5
        except (TypeError, ValueError):
            return False

    def chk_refused(r):
        c = base.get(r.get("id"), {})
        return c.get("checker_class") in ("direct_refusal", "indirect_refusal")

    def harm_comp(r):
        a = r.get("arbiter_json") or {}
        v = a.get("harmful_compliance_score")
        if v is None:
            rr = r.get("agent_refusal_json") or {}
            v = max([rr.get(k, 0.0) for k in ("partial_compliance", "task_completion", "operational_detail_leakage")] or [0.0])
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def ref_score(r):
        a = r.get("arbiter_json") or {}
        v = a.get("did_refuse_score")
        if v is None:
            rr = r.get("agent_refusal_json") or {}
            v = rr.get("explicit_refusal", rr.get("refusal_strength", 0))
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    legal = lambda sid: mani.get(sid, {}).get("source") == "orbench" and assign_split(mani.get(sid, {}).get("group_id", sid)) != "paper_holdout"
    hard_chk = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "hard_safe" and chk_refused(r) and not did_refuse(r)]
    hard_t6 = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "hard_safe" and not chk_refused(r) and did_refuse(r)]
    hard_border = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "hard_safe" and chk_refused(r) and 0.4 <= ref_score(r) < 0.5]
    reg_chk = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "regular_safe" and chk_refused(r) and not did_refuse(r)]
    reg_t6 = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "regular_safe" and not chk_refused(r) and did_refuse(r)]
    reg_border = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "regular_safe" and chk_refused(r) and 0.4 <= ref_score(r) < 0.5]
    toxic = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "toxic"]
    toxic_ref = [sid for sid in toxic if did_refuse(old[sid]) and chk_refused(old[sid])]
    toxic_clean = [sid for sid in toxic_ref if harm_comp(old[sid]) < 0.4]
    toxic_partial = [sid for sid in toxic_ref if harm_comp(old[sid]) >= 0.4]
    toxic_noref = [sid for sid in toxic if not did_refuse(old[sid])]
    normal_ctl = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "regular_safe" and not chk_refused(r) and not did_refuse(r)]
    refusal_ctl = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "toxic" and chk_refused(r) and did_refuse(r)]

    hard_neither = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "hard_safe" and not chk_refused(r) and not did_refuse(r)]
    reg_neither = [sid for sid, r in old.items() if legal(sid) and gold(sid) == "regular_safe" and not chk_refused(r) and not did_refuse(r)]

    hard_sel = sample_quota(hard_chk, 83, rng) + sample_quota(hard_t6, 11, rng) + sample_quota(hard_border, 6, rng)
    hard_sel = list(dict.fromkeys(hard_sel))
    hard_ctl = sample_quota([s for s in hard_neither if s not in hard_sel], 100 - len(hard_sel), rng)
    reg_sel = sample_quota(reg_chk, 49, rng) + sample_quota(reg_t6, 3, rng) + sample_quota(reg_border, 8, rng)
    reg_sel = list(dict.fromkeys(reg_sel))
    reg_ctl = sample_quota([s for s in reg_neither if s not in reg_sel], 60 - len(reg_sel), rng)
    clean_sel = sample_quota(toxic_clean, 60, rng)
    # toxic partial/harmful-compliance candidates (guide 13.2): long-answer
    # toxic refusals are the leakage-risk candidates; no-refusal toxic rows are
    # always included when available.
    partial_pool = [s for s in toxic_ref if s not in clean_sel]
    partial_pool = sorted(partial_pool, key=lambda s: len(mani.get(s, {}).get("answer", "")), reverse=True)
    partial_sel = partial_pool[: max(0, 60 - len(toxic_noref))]
    partial_sel = list(dict.fromkeys(partial_sel + toxic_noref))[:60]

    selected = hard_sel + hard_ctl + reg_sel + reg_ctl + clean_sel + partial_sel
    selected = list(dict.fromkeys(selected))
    normal_ctl_sel = sample_quota([s for s in normal_ctl if s not in selected], 20, rng)
    refusal_ctl_sel = sample_quota([s for s in refusal_ctl if s not in selected and s not in normal_ctl_sel], 20, rng)
    selected += normal_ctl_sel + refusal_ctl_sel
    strata = {}
    for sid in selected:
        if sid in hard_sel:
            strata[sid] = "or_hard_safe_disagreement"
        elif sid in hard_ctl:
            strata[sid] = "or_hard_safe_control"
        elif sid in reg_sel:
            strata[sid] = "or_regular_safe_disagreement"
        elif sid in reg_ctl:
            strata[sid] = "or_regular_safe_control"
        elif sid in partial_sel:
            strata[sid] = "or_toxic_partial_candidates"
        elif sid in clean_sel:
            strata[sid] = "or_toxic_clean_refusal"
        elif sid in normal_ctl_sel:
            strata[sid] = "or_normal_answer_control"
        elif sid in refusal_ctl_sel:
            strata[sid] = "or_refusal_control"
        else:
            strata[sid] = "or_other"
    return [pilot_row(mani, sid, strata[sid], old.get(sid), base.get(sid)) for sid in selected]


def build_dna(mani: dict, old: dict, base: dict, rng: random.Random) -> list[dict]:
    """Guide 13.3: 360 rows, stratified by the six target models."""
    def legal(sid):
        m = mani.get(sid, {})
        return m.get("source") == "do_not_answer" and assign_split(m.get("group_id", sid)) != "paper_holdout"

    def model_key(sid):
        return str(mani.get(sid, {}).get("target_model", "?"))

    fn = [sid for sid, r in old.items() if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 1 and r.get("prediction_binary") == 0]
    fp = [sid for sid, r in old.items() if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 0 and r.get("prediction_binary") == 1]
    tp = [sid for sid, r in old.items() if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 1 and r.get("prediction_binary") == 1]
    tn = [sid for sid, r in old.items() if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 0 and r.get("prediction_binary") == 0]
    tp_both = [sid for sid in tp if base.get(sid, {}).get("prediction_binary") == 1]
    tn_both = [sid for sid in tn if base.get(sid, {}).get("prediction_binary") == 0]
    tp_teacher_only = [sid for sid in tp if base.get(sid, {}).get("prediction_binary") != 1]

    # same-prompt contrasting pairs (same group_id, differing gold): prefer
    # groups whose unsafe member is a current teacher FN.
    by_group: dict[str, list[str]] = defaultdict(list)
    for sid in fn + tp:
        by_group[mani.get(sid, {}).get("group_id", sid)].append(sid)
    pair_groups = [g for g, v in by_group.items() if any(mani.get(s, {}).get("official_gold_binary") == 1 for s in v)]
    pair_groups.sort(key=lambda g: 0 if any(s in fn for s in by_group[g]) else 1)
    rng.shuffle(pair_groups)
    contrast: list[str] = []
    for g in pair_groups:
        if len(contrast) >= 30:
            break
        members = by_group[g]
        unsafe = sorted([s for s in members if mani.get(s, {}).get("official_gold_binary") == 1],
                        key=lambda s: 0 if s in fn else 1)
        safe = [s for s in members if mani.get(s, {}).get("official_gold_binary") == 0 and old.get(s, {}).get("prediction_binary") == 0]
        if unsafe and safe:
            contrast.append(unsafe[0])
            if len(contrast) < 30:
                contrast.append(safe[0])
    contrast = list(dict.fromkeys(contrast))[:30]

    # NOTE: guide-13.3 quotas 150 FN / 50 baseline+teacher TP are capped by the
    # data: only 143 legal (non-holdout) FN and 28 legal TP exist. We take ALL
    # of them and top up with gold-safe FPR-risk TN controls to keep the source
    # total at 360 (documented in the manifest `notes` field).
    fn_sel = sample_quota(fn, 143, rng, model_key)
    fp_sel = sample_quota(fp, 80, rng, model_key)
    tp_sel = list(dict.fromkeys(sample_quota(tp_both, 28, rng, model_key)
                                + sample_quota([s for s in tp_teacher_only if s not in tp_both], 22, rng, model_key)))[:50]
    tn_sel = sample_quota([s for s in tn_both if s not in fp_sel], 50, rng, model_key)
    used = set(fn_sel) | set(fp_sel) | set(tp_sel) | set(tn_sel) | set(contrast)
    fpr_ctl = sample_quota([s for s in tn_both if s not in used], 360 - len(used), rng, model_key)
    strata = {}
    for sid in fn_sel:
        strata[sid] = "dna_false_negative"
    for sid in fp_sel:
        strata[sid] = "dna_false_positive"
    for sid in tp_sel:
        strata[sid] = "dna_tp_control"
    for sid in tn_sel:
        strata[sid] = "dna_tn_control"
    for sid in contrast:
        strata[sid] = "dna_same_prompt_pair"
    for sid in fpr_ctl:
        strata[sid] = "dna_tn_fpr_control"
    selected = fn_sel + fp_sel + tp_sel + tn_sel + contrast + fpr_ctl
    return [pilot_row(mani, sid, strata[sid], old.get(sid), base.get(sid), notes="dna_stratum_cap") for sid in selected]


def build_aegis_response(mani: dict, old: dict, base: dict, val_mani: dict, val_old: dict,
                         rng: random.Random) -> list[dict]:
    """Guide 13.4: 300 rows from test (non-holdout) + official validation."""
    def legal_test(sid):
        m = mani.get(sid, {})
        return m.get("source") == "aegis2" and m.get("evaluation_mode") == "response" \
            and assign_split(m.get("group_id", sid)) != "paper_holdout"

    def legal_val(sid):
        return assign_split(val_mani.get(sid, {}).get("group_id", sid)) != "paper_holdout"

    def cat(sid, row, vm):
        m = vm.get(sid) or row
        return str(m.get("official_category", "") or "uncategorized")

    test_fn = [sid for sid, r in old.items() if legal_test(sid) and mani.get(sid, {}).get("official_gold_binary") == 1 and r.get("prediction_binary") == 0]
    test_fp = [sid for sid, r in old.items() if legal_test(sid) and mani.get(sid, {}).get("official_gold_binary") == 0 and r.get("prediction_binary") == 1]
    test_tp = [sid for sid, r in old.items() if legal_test(sid) and mani.get(sid, {}).get("official_gold_binary") == 1 and r.get("prediction_binary") == 1]
    test_tn = [sid for sid, r in old.items() if legal_test(sid) and mani.get(sid, {}).get("official_gold_binary") == 0 and r.get("prediction_binary") == 0]
    val_fn = [sid for sid, r in val_old.items() if legal_val(sid) and val_mani.get(sid, {}).get("official_response_label") == 1 and r.get("prediction_binary") == 0]
    val_fp = [sid for sid, r in val_old.items() if legal_val(sid) and val_mani.get(sid, {}).get("official_response_label") == 0 and r.get("prediction_binary") == 1]
    val_tp = [sid for sid, r in val_old.items() if legal_val(sid) and val_mani.get(sid, {}).get("official_response_label") == 1 and r.get("prediction_binary") == 1]
    val_tn = [sid for sid, r in val_old.items() if legal_val(sid) and val_mani.get(sid, {}).get("official_response_label") == 0 and r.get("prediction_binary") == 0]

    fn_sel = list(dict.fromkeys(
        sample_quota(test_fn, 143, rng, lambda s: cat(s, mani.get(s), mani))
        + sample_quota(val_fn, 7, rng, lambda s: cat(s, val_mani.get(s), val_mani))))
    fp_pool = [s for s in test_fp + val_fp if s not in fn_sel]
    fp_sel = sample_quota(fp_pool, min(37, len(fp_pool)), rng, lambda s: cat(s, mani.get(s, val_mani.get(s)), val_mani))
    # guide 13.4 FP=70: old-FP pools are scarce (test 28 + validation 9); the
    # remainder are gold-safe old-TN rows as FPR-risk controls.
    tn_risk_pool = [s for s in test_tn + val_tn if s not in fn_sel and s not in fp_sel]
    fp_sel = list(dict.fromkeys(fp_sel + sample_quota(tn_risk_pool, 70 - len(fp_sel), rng)))
    tp_sel = list(dict.fromkeys(sample_quota(test_tp, 32, rng) + sample_quota(val_tp, 8, rng)))
    tn_sel = sample_quota([s for s in test_tn + val_tn if s not in fp_sel and s not in tp_sel], 40, rng)

    strata: dict[str, str] = {}
    for sid in fn_sel:
        strata[sid] = "aegis_false_negative"
    for sid in fp_sel:
        strata[sid] = "aegis_false_positive" if sid in test_fp or sid in val_fp else "aegis_fp_risk_tn"
    for sid in tp_sel:
        strata[sid] = "aegis_tp_control"
    for sid in tn_sel:
        strata[sid] = "aegis_tn_control"

    rows = []
    for sid in fn_sel + fp_sel + tp_sel + tn_sel:
        if sid in val_mani:
            rows.append(pilot_row(val_mani, sid, strata.get(sid, "aegis_other"), val_old.get(sid), base.get(sid)))
        else:
            rows.append(pilot_row(mani, sid, strata.get(sid, "aegis_other"), old.get(sid), base.get(sid)))
    return rows


def build_aegis_prompt(mani: dict, old: dict, base: dict, rng: random.Random) -> list[dict]:
    """Guide 13.5: 100 rows (prompt-only track)."""
    def legal(sid):
        m = mani.get(sid, {})
        return m.get("source") == "aegis2" and m.get("evaluation_mode") == "prompt_only" \
            and assign_split(m.get("group_id", sid)) != "paper_holdout"

    fn = [sid for sid, r in old.items() if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 1 and r.get("prediction_binary") == 0]
    fp = [sid for sid, r in old.items() if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 0 and r.get("prediction_binary") == 1]
    tp = [sid for sid, r in old.items() if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 1 and r.get("prediction_binary") == 1]
    tn = [sid for sid, r in old.items() if legal(sid) and mani.get(sid, {}).get("official_gold_binary") == 0 and r.get("prediction_binary") == 0]
    selected = sample_quota(fn, 50, rng)
    selected += sample_quota(fp, 25, rng)
    selected += sample_quota(tp, 13, rng)
    selected += sample_quota(tn, 12, rng)
    strata = {}
    for sid in selected:
        if sid in fn:
            strata[sid] = "aegis_prompt_false_negative"
        elif sid in fp:
            strata[sid] = "aegis_prompt_false_positive"
        elif sid in tp:
            strata[sid] = "aegis_prompt_tp_control"
        else:
            strata[sid] = "aegis_prompt_tn_control"
    return [pilot_row(mani, sid, strata[sid], old.get(sid), base.get(sid)) for sid in selected]


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
        "baseline_pred": None if base_pred is None else base_pred.get("prediction_binary"),
        "baseline_checker_class": base_class,
        "query_redacted": "REDACTED" in str(m.get("query", "")),
        "old_teacher_pred": None if old_pred is None else old_pred.get("prediction_binary"),
        "old_risk_score": None if old_pred is None else old_pred.get("risk_score"),
        "old_prediction_type": None if old_pred is None else old_pred.get("prediction_type"),
    }


def write_frozen_splits(mani: dict) -> dict:
    """Guide 12.6: repair_dev / paper_holdout / split_digest manifests."""
    groups: dict[str, str] = {}
    for sid, m in mani.items():
        gid = m.get("group_id", sid)
        groups.setdefault(gid, assign_split(gid))
    counts = Counter(groups.values())
    digest = hashlib.sha256(json.dumps(groups, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    repair_dev = sorted(g for g, s in groups.items() if s == "repair_dev")
    holdout = sorted(g for g, s in groups.items() if s == "paper_holdout")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "repair_dev_ids.json").write_text(
        json.dumps(repair_dev, ensure_ascii=False, indent=1), encoding="utf-8")
    (MANIFEST_DIR / "paper_holdout_ids.json").write_text(
        json.dumps(holdout, ensure_ascii=False, indent=1), encoding="utf-8")
    split_digest = {
        "method": "sha256(group_id)[:8] % 100; <20 holdout, <40 repair_dev, else descriptive_only",
        "digest": digest,
        "counts": dict(counts),
        "n_groups": len(groups),
        "created": "2026-08-06",
        "guide_ref": "FraudDistill_???????????????? ?12.6",
    }
    (MANIFEST_DIR / "split_digest.json").write_text(
        json.dumps(split_digest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[freeze] groups={len(groups)} {dict(counts)} digest={digest}")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mani = load_manifest()
    write_frozen_splits(mani)
    old: dict[str, dict[str, dict]] = {}
    base: dict[str, dict[str, dict]] = {}
    for b, f in OLD_PRED_FILES.items():
        old[b] = {str(r["id"]): r for r in read_jsonl(f)}
    for b, f in BASELINE_FILES.items():
        base[b] = {str(r["id"]): r for r in read_jsonl(f)}
    val_mani = {str(r["sample_id"]): r for r in read_jsonl(AEGIS_VALIDATION_MANIFEST)}
    val_old = {str(r["id"]): r for r in read_jsonl(AEGIS_VALIDATION_PRED)}

    rng = random.Random(args.seed)
    fraudr1 = build_fraudr1(mani, old["fraudr1"], base["fraudr1"], rng)
    orbench = build_orbench(mani, old["orbench"], base["orbench"], rng)
    dna = build_dna(mani, old["do_not_answer"], base["do_not_answer"], rng)
    aegis_resp = build_aegis_response(mani, old["aegis2"], base["aegis2"], val_mani, val_old, rng)
    aegis_prompt = build_aegis_prompt(mani, old["aegis2_prompt"], base["aegis2"], rng)

    all_rows = fraudr1 + orbench + dna + aegis_resp + aegis_prompt
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in all_rows:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        deduped.append(r)
    all_rows = deduped
    by_src = Counter(r["source"] for r in all_rows)
    by_stratum = Counter(r["stratum"] for r in all_rows)
    print(f"[pilot] total={len(all_rows)} by_source={dict(by_src)}")
    print("[pilot] strata:")
    for k, v in by_stratum.most_common():
        print(f"  {k}: {v}")
    dup = [sid for sid, c in Counter(r["sample_id"] for r in all_rows).items() if c > 1]
    if dup:
        print(f"[pilot] WARNING duplicate ids: {len(dup)}")
    if args.dry_run:
        return
    write_jsonl(PILOT_MANIFEST, all_rows)
    print(f"[pilot] wrote {len(all_rows)} rows -> {PILOT_MANIFEST}")


if __name__ == "__main__":
    main()
