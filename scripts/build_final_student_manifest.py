# -*- coding: utf-8 -*-
"""Build the final student training manifest (guide 3-8, final 1.5B student).

Sources (guide 3.3):
  A  exp3_dataset.jsonl split==train, minus Exp2 balanced test/dev overlap
     and Exp3 dev/test leakage (id / group / qy_hash / template family).
  B  Exp2 balanced dev manifest + adjudicated gold + FraudDistill teacher
     predictions (700 rows, all have gold + teacher signal).
  C  4,000-row expansion, keep <=1200 teacher-only auxiliary rows,
     preferring partial_leakage / direct_fraud / context_flip / over_refusal.

Outputs:
  data/prepared/exp3_neural_student/final_train_manifest.jsonl
  data/prepared/exp3_neural_student/final_data_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from frauddistill.student.dataset import map_gold_type  # noqa: E402

LABEL_TO_ID = {"safe": 0, "fraud_assistance": 1, "refusal_failure": 2, "over_refusal": 3}

SOURCE_TARGETS = {"benchmark": 0.48, "paired_dev": 0.17, "synthetic_core": 0.23, "hard_expansion": 0.12}
EN_TARGET = 0.63


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text)).lower()


def qy_hash(q: str, a: str) -> str:
    return hashlib.sha256(f"{norm(q)}\n{norm(a)}".encode("utf-8")).hexdigest()


def detect_lang(text: str) -> str:
    t = str(text or "")
    if not t:
        return "en"
    cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk / max(len(t), 1) > 0.05 else "en"


def template_family_of(row: dict, dataset_row: dict | None = None) -> str:
    """Unified template-family key used for leakage checks.

    - synthetic / benchmark / paired_dev rows: group_id is the exact template
      instance (coarse labels like "Fake Job Postings" are benchmark families
      shared by dev/test by design and must NOT be used as leakage keys);
    - expansion rows: the exp3x_* family prefix is Exp3-specific, so the
      template_family_id prefix is a valid template-level key.
    """
    g = str(row.get("group_id") or row.get("group") or "")
    src = str(row.get("source") or "")
    fam = str(row.get("template_family_id") or row.get("family") or "")
    if "exp3x" in fam:
        return fam.split("_pair")[0].split("_p")[0] if False else fam
    return g


def build_exclusion_sets(exp3_dev_test: list[dict], balanced_test: list[dict],
                         balanced_dev: list[dict]) -> dict:
    def coll(row):
        q = row.get("user_query") or row.get("query") or ""
        a = row.get("target_model_answer") or row.get("answer") or ""
        return {
            "ids": {row.get("id") or row.get("sample_id")},
            "qy": {qy_hash(q, a)} if q and a else set(),
            "groups": {str(row.get("group_id") or row.get("group") or "")} - {""},
            "fams": {template_family_of(row)} - {""},
        }

    def merge(rows):
        out = {"ids": set(), "qy": set(), "groups": set(), "fams": set()}
        for r in rows:
            c = coll(r)
            for k in out:
                out[k] |= c[k]
        return out

    return {
        "exp3_dev_test": merge(exp3_dev_test),
        "balanced_test": merge(balanced_test),
        "balanced_dev": merge(balanced_dev),
    }


def source_a_rows(dataset_rows, excl, teacher_map) -> tuple[list[dict], dict]:
    rows, dropped = [], Counter()
    for r in dataset_rows:
        if r["split"] != "train":
            continue
        c = {
            "ids": {r["id"]},
            "qy": {qy_hash(r["user_query"], r["target_model_answer"])},
            "groups": {str(r.get("group_id") or "")} - {""},
            "fams": {template_family_of(r)} - {""},
        }
        leak = False
        for name in ("balanced_test", "balanced_dev", "exp3_dev_test"):
            ex = excl[name]
            if (c["ids"] & ex["ids"]) or (c["qy"] & ex["qy"]) or (c["groups"] & ex["groups"]) or (c["fams"] & ex["fams"]):
                dropped[f"{name}:{r['source']}"] += 1
                leak = True
                break
        if leak:
            continue
        tr = teacher_map.get(r["id"]) or {}
        sig = tr.get("signal") or {}
        gold_type = map_gold_type(
            str(r["gold_label"]), str(r.get("gold_type", "")), str(r.get("subtype", "")),
            str(sig.get("teacher_type", "safe")),
            teacher_unsafe=(str(sig.get("teacher_label", "safe")) == "unsafe"))
        bucket = "synthetic_core" if str(r.get("source")) == "synthetic" else "benchmark"
        rows.append({
            "id": r["id"], "sample_id": r["id"], "group_id": r.get("group_id", ""),
            "template_family_id": template_family_of(r), "pair_id": r.get("pair_id"),
            "user_query": r["user_query"], "target_model_answer": r["target_model_answer"],
            "gold_label": r["gold_label"], "gold_type": gold_type,
            "gold_binary": 1 if r["gold_label"] == "unsafe" else 0,
            "gold_source": "procedural_weak" if r["source"] == "synthetic"
                           else ("audit" if r["source"] in ("e1_context_r2", "fraudr1_all") else "official"),
            "source": r["source"], "subtype": r.get("subtype", ""),
            "language": detect_lang(r["user_query"] + " " + r["target_model_answer"])
                        if not str(r.get("language", "")).strip() else str(r["language"]).lower(),
            "target_model": r.get("target_model", ""),
            "teacher_label": str(sig.get("teacher_label", "safe")),
            "teacher_score": float(sig.get("teacher_score", 0.5)),
            "teacher_type": str(sig.get("teacher_type", "safe")),
            "teacher_confidence": float(sig.get("teacher_confidence", sig.get("confidence", 0.5) or 0.5)),
            "agent_agreement": float(sig.get("agent_agreement", 0.5)),
            "conflict_flags": list((sig.get("conflict_flags") or []) + (tr.get("conflict_flags") or [])),
            "teacher_gold_agree": str(sig.get("teacher_label", "safe")) == r["gold_label"],
            "teacher_only": False,
            "source_bucket": bucket,
            "qy_hash": qy_hash(r["user_query"], r["target_model_answer"]),
        })
    return rows, dropped


def source_b_rows(balanced_dev, gold_map, pred_map) -> tuple[list[dict], dict]:
    rows, stats = [], Counter()
    for r in balanced_dev:
        sid = r["sample_id"]
        g = gold_map.get(sid) or {}
        p = pred_map.get(sid) or {}
        if not g or not p:
            stats["missing_gold_or_pred"] += 1
            continue
        gt = str(g.get("gold_type", ""))
        subtype = str(g.get("behavior_subtype", ""))
        if gt == "safe":
            gold_type = "safe"
        elif gt == "fraud_assistance":
            gold_type = "fraud_assistance"
        elif gt == "harmful_response":
            gold_type = "fraud_assistance" if "fraud" in subtype.lower() else "refusal_failure"
        elif gt == "behavior_error":
            gold_type = "refusal_failure" if "harmful compliance" in subtype.lower() else "over_refusal"
        else:
            gold_type = map_gold_type(str(g.get("gold_binary", 0)), "", "", "", teacher_unsafe=(g.get("gold_binary") == 1))
        gold_bin = int(g.get("gold_binary", 0))
        lang = str(r.get("language") or "")
        if lang in ("", "None", "null"):
            lang = detect_lang(r["query"] + " " + r["answer"])
        else:
            lang = lang.lower()
            lang = {"english": "en", "chinese": "zh"}.get(lang, lang)
        rows.append({
            "id": sid, "sample_id": sid, "group_id": str(r.get("group_id") or ""),
            "template_family_id": str(r.get("family") or r.get("group_id") or sid),
            "pair_id": r.get("pair_id"),
            "user_query": r["query"], "target_model_answer": r["answer"],
            "gold_label": "unsafe" if gold_bin == 1 else "safe", "gold_type": gold_type,
            "gold_binary": gold_bin,
            "gold_source": "official" if str(g.get("label_source", "")) == "official" else "audit",
            "source": str(r.get("source") or ""), "subtype": subtype,
            "language": lang, "target_model": "",
            "teacher_label": "unsafe" if int(p.get("prediction_binary", 0)) == 1 else "safe",
            "teacher_score": float(p.get("risk_score", 0.5) or 0.5),
            "teacher_type": str(p.get("prediction_type", "safe") or "safe"),
            "teacher_confidence": float(p.get("confidence", 0.5) or 0.5),
            "agent_agreement": float(p.get("agent_agreement", 0.5) or 0.5),
            "conflict_flags": list(p.get("contradiction_flags") or p.get("conflict_flags") or []),
            "teacher_gold_agree": int(p.get("prediction_binary", 0)) == gold_bin,
            "teacher_only": False,
            "source_bucket": "paired_dev",
            "qy_hash": qy_hash(r["query"], r["answer"]),
        })
    return rows, stats


def source_c_rows(expansion, excl, max_expansion: int) -> tuple[list[dict], dict]:
    """Select teacher-only expansion rows (guide 1.2, 32)."""
    prio = {"partial_leakage": 1, "direct_fraud": 2, "hard_safe": 3,
            "context_flip": 4, "over_refusal": 5, "clean_refusal": 6,
            "general_safety": 7, "trust_facilitation": 8,
            "anti_fraud_education": 9, "quotation_analysis": 10}
    picked, dropped = [], Counter()
    # pair-aware: keep whole context_flip pairs when selected
    cf_pairs = {}
    for e in expansion:
        s = e["sample"]
        if s.get("subtype") == "context_flip" and s.get("pair_id"):
            cf_pairs.setdefault(s["pair_id"], []).append(e)
    # non-context-flip rows ordered by priority
    others = [e for e in expansion if e["sample"].get("subtype") != "context_flip"]
    others.sort(key=lambda e: (prio.get(e["sample"].get("subtype", ""), 99), e["id"]))
    used = 0
    for e in others:
        s = e["sample"]
        q, a = s["user_query"], s["target_model_answer"]
        c = {"qy": {qy_hash(q, a)}, "groups": {str(e.get("group_id") or "")} - {""},
             "fams": {template_family_of(s)} - {""}}
        leak = any((c["qy"] & ex["qy"]) or (c["groups"] & ex["groups"]) or (c["fams"] & ex["fams"])
                   for ex in (excl["balanced_test"], excl["balanced_dev"], excl["exp3_dev_test"]))
        if leak:
            dropped[f"leak:{s.get('subtype','')}"] += 1
            continue
        if used >= max_expansion:
            dropped[f"budget:{s.get('subtype','')}"] += 1
            continue
        picked.append(e)
        used += 1
        # context_flip budget handled separately below
    # context-flip pairs: keep up to 100 rows (50 pairs) if budget allows
    cf_budget = max_expansion - len(picked)
    if cf_budget > 0:
        cf_list = sorted(cf_pairs.items(), key=lambda kv: kv[0])
        cf_used = 0
        for pid, members in cf_list:
            if cf_used + len(members) > cf_budget:
                continue
            ok = True
            for e in members:
                s = e["sample"]
                c = {"qy": {qy_hash(s["user_query"], s["target_model_answer"])},
                     "groups": {str(e.get("group_id") or "")} - {""},
                     "fams": {template_family_of(s)} - {""}}
                if any((c["qy"] & ex["qy"]) or (c["groups"] & ex["groups"]) or (c["fams"] & ex["fams"])
                       for ex in (excl["balanced_test"], excl["balanced_dev"], excl["exp3_dev_test"])):
                    ok = False
                    break
            if not ok:
                continue
            picked.extend(members)
            cf_used += len(members)
            if cf_used + len(members) > cf_budget:
                pass
    rows = []
    for e in picked:
        s = e["sample"]
        sig = e.get("signal") or {}
        q, a = s["user_query"], s["target_model_answer"]
        gold_label = str(s.get("gold_label", "safe"))
        gold_type = str(s.get("gold_type", "safe"))
        if gold_type not in LABEL_TO_ID:
            gold_type = map_gold_type(gold_label, "", str(s.get("subtype", "")),
                                      str(sig.get("teacher_type", "safe")),
                                      teacher_unsafe=(str(sig.get("teacher_label", "safe")) == "unsafe"))
        rows.append({
            "id": e["id"], "sample_id": e["id"], "group_id": e.get("group_id", ""),
            "template_family_id": template_family_of(s), "pair_id": s.get("pair_id"),
            "user_query": q, "target_model_answer": a,
            "gold_label": gold_label, "gold_type": gold_type,
            "gold_binary": 1 if gold_label == "unsafe" else 0,
            "gold_source": "procedural_weak",
            "source": "exp3_expansion", "subtype": s.get("subtype", ""),
            "language": detect_lang(q + " " + a) if not str(s.get("language", "")).strip() else str(s["language"]).lower(),
            "target_model": s.get("target_model", ""),
            "teacher_label": str(sig.get("teacher_label", "safe")),
            "teacher_score": float(sig.get("teacher_score", 0.5) or 0.5),
            "teacher_type": str(sig.get("teacher_type", "safe") or "safe"),
            "teacher_confidence": float(sig.get("teacher_confidence", sig.get("confidence", 0.5) or 0.5)),
            "agent_agreement": float(sig.get("agent_agreement", 0.5) or 0.5),
            "conflict_flags": list(sig.get("conflict_flags") or e.get("conflict_flags") or []),
            "teacher_gold_agree": bool(sig.get("teacher_gold_agree", False)),
            "teacher_only": True,
            "source_bucket": "hard_expansion",
            "qy_hash": qy_hash(q, a),
        })
    return rows, dropped


def compute_sampler_weights(rows: list[dict]) -> list[dict]:
    """Source-aware + class-aware + type-aware + language-aware weights.

    Iterative marginal matching (guide 8, 31):
      - bucket effective share: benchmark .48 / paired_dev .17 /
        synthetic_core .23 / hard_expansion .12 (within +/-5pp);
      - safe/unsafe 50/50 inside every bucket;
      - unsafe risk types: every type keeps >=10% of unsafe mass;
      - global English effective share ~63% (guide 8.4, 60-65%);
      - final clip: no row weight > 4x median (guide 31).
    """
    n = len(rows)
    bucket = [r["source_bucket"] for r in rows]
    cls = ["safe" if r["gold_binary"] == 0 else "unsafe" for r in rows]
    typ = [r["gold_type"] for r in rows]
    lang = [r["language"] for r in rows]
    w = [1.0] * n

    def normalize():
        tot = sum(w)
        for i in range(n):
            w[i] /= tot

    def bucket_share(b):
        return sum(w[i] for i in range(n) if bucket[i] == b)

    for _ in range(40):
        normalize()
        # 1) bucket marginal
        for b in SOURCE_TARGETS:
            t = SOURCE_TARGETS[b]
            s_b = bucket_share(b)
            if s_b > 0:
                for i in range(n):
                    if bucket[i] == b:
                        w[i] *= t / s_b
        # 2) class marginal within bucket
        for b in SOURCE_TARGETS:
            rows_b = [i for i in range(n) if bucket[i] == b]
            if not rows_b:
                continue
            for c in ("safe", "unsafe"):
                idx = [i for i in rows_b if cls[i] == c]
                s_c = sum(w[i] for i in idx) / max(sum(w[i] for i in rows_b), 1e-12)
                if s_c > 0:
                    for i in idx:
                        w[i] *= 0.5 / s_c
        # 3) unsafe type coverage: floor 10% per type of unsafe mass
        uns_idx = [i for i in range(n) if cls[i] == "unsafe"]
        if uns_idx:
            uns_mass = sum(w[i] for i in uns_idx)
            for t in ("fraud_assistance", "refusal_failure", "over_refusal"):
                idx = [i for i in uns_idx if typ[i] == t]
                s_t = sum(w[i] for i in idx) / max(uns_mass, 1e-12)
                if s_t < 0.10 and s_t > 0:
                    for i in idx:
                        w[i] *= (0.10 / s_t)
        # 4) language marginal (global, EN ~0.63)
        en_idx = [i for i in range(n) if lang[i] == "en"]
        zh_idx = [i for i in range(n) if lang[i] != "en"]
        if en_idx and zh_idx:
            en_mass = sum(w[i] for i in en_idx)
            zh_mass = sum(w[i] for i in zh_idx)
            tot = en_mass + zh_mass
            for i in en_idx:
                w[i] *= (0.63 / max(en_mass / tot, 1e-9))
            for i in zh_idx:
                w[i] *= (0.37 / max(zh_mass / tot, 1e-9))

    normalize()
    med = sorted(w)[n // 2] if n else 1.0
    cap = med * 4.0
    for i in range(n):
        w[i] = min(w[i], cap)
    for i, r in enumerate(rows):
        r["sample_weight"] = round(w[i], 8)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp3-train", default=str(REPO / "data/prepared/exp3_agent_distillation/exp3_dataset.jsonl"))
    ap.add_argument("--exp2-balanced-dev", default=str(REPO / "experiments/exp2_prior_work_comparison/balanced_design/manifests/balanced_dev_manifest.jsonl"))
    ap.add_argument("--exp2-balanced-test", default=str(REPO / "experiments/exp2_prior_work_comparison/balanced_design/manifests/balanced_test_manifest.jsonl"))
    ap.add_argument("--expansion", default=str(REPO / "data/prepared/exp3_neural_student/expansion_annotated.jsonl"))
    ap.add_argument("--max-expansion", type=int, default=1100)
    ap.add_argument("--out", default=str(REPO / "data/prepared/exp3_neural_student/final_train_manifest.jsonl"))
    args = ap.parse_args()

    dataset = read_jsonl(Path(args.exp3_train))
    balanced_test = read_jsonl(Path(args.exp2_balanced_test))
    balanced_dev = read_jsonl(Path(args.exp2_balanced_dev))
    expansion = read_jsonl(Path(args.expansion))

    exp3_dev_test = [r for r in dataset if r["split"] in ("dev", "test")]
    excl = build_exclusion_sets(exp3_dev_test, balanced_test, balanced_dev)

    teacher_map = {}
    for r in read_jsonl(REPO / "experiments/exp3_agent_distillation_ablation/outputs/agent_predictions/train.jsonl"):
        teacher_map[r["id"]] = r

    rows_a, drop_a = source_a_rows([r for r in dataset if r["split"] == "train"], excl, teacher_map)

    gold_map = {g["sample_id"]: g for g in read_jsonl(REPO / "experiments/exp2_prior_work_comparison/balanced_design/gold/balanced_dev_gold.jsonl")}
    pred_map = {}
    for pfile in (REPO / "experiments/exp2_prior_work_comparison/balanced_design/predictions/dev").glob("*.jsonl"):
        for p in read_jsonl(pfile):
            pred_map[p.get("id") or p.get("sample_id")] = p
    rows_b, drop_b = source_b_rows(balanced_dev, gold_map, pred_map)
    # Source B must stay disjoint from Exp2 balanced test and Exp3 dev/test
    # at the group / qy level (guide 3.3, 5.1): drop rows whose canonical
    # group or qy appears in either holdout.
    b_keep = []
    for r in rows_b:
        g = str(r.get("group_id") or "")
        hit = False
        for ex in (excl["balanced_test"], excl["exp3_dev_test"]):
            if (r["qy_hash"] in ex["qy"]) or (g and g in ex["groups"]):
                hit = True
                break
        if hit:
            drop_b[f"source_b_leak:{r['source']}"] += 1
        else:
            b_keep.append(r)
    rows_b = b_keep

    rows_c, drop_c = source_c_rows(expansion, excl, args.max_expansion)

    # exact q+y dedup across all sources
    seen_qy = {}
    for r in rows_a + rows_b + rows_c:
        if r["qy_hash"] in seen_qy:
            print(f"WARN duplicate qy: {r['id']} vs {seen_qy[r['qy_hash']]}")
        else:
            seen_qy[r["qy_hash"]] = r["id"]

    all_rows = compute_sampler_weights(rows_a + rows_b + rows_c)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n", encoding="utf-8")

    audit = {
        "n_train": len(all_rows),
        "n_source_a": len(rows_a), "n_source_b": len(rows_b), "n_source_c": len(rows_c),
        "by_bucket": dict(Counter(r["source_bucket"] for r in all_rows)),
        "by_source": dict(Counter(r["source"] for r in all_rows)),
        "by_language": dict(Counter(r["language"] for r in all_rows)),
        "by_gold": dict(Counter(r["gold_label"] for r in all_rows)),
        "by_gold_type": dict(Counter(r["gold_type"] for r in all_rows)),
        "by_subtype": dict(Counter(r["subtype"] for r in all_rows).most_common(30)),
        "by_template_family": len({r["template_family_id"] for r in all_rows}),
        "teacher_only": sum(1 for r in all_rows if r["teacher_only"]),
        "pair_rows": sum(1 for r in all_rows if r.get("pair_id")),
        "dropped": dict(drop_a + drop_b + drop_c),
        "max_expansion": args.max_expansion,
        "leakage_checks": {
            "exp3_dev_test_ids": len(excl["exp3_dev_test"]["ids"]),
            "exp3_dev_test_groups": len(excl["exp3_dev_test"]["groups"]),
            "exp3_dev_test_fams": len(excl["exp3_dev_test"]["fams"]),
            "balanced_test_ids": len(excl["balanced_test"]["ids"]),
            "balanced_test_groups": len(excl["balanced_test"]["groups"]),
            "balanced_test_fams": len(excl["balanced_test"]["fams"]),
        },
    }
    audit_path = out.with_name("final_data_audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
