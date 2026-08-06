# -*- coding: utf-8 -*-
"""Exp2 FULL-coverage offline evaluation (guide 2026-08-06).

Three-layer reporting (guide 5):
  Layer 1 - Full Native Protocol : per-benchmark native metrics on the full
            project pools (Fraud-R1 distribution, OR dual-axis, DNA official
            human labels, Aegis official labels) + audited evaluator analysis
            on the LLM-panel silver subsets.
  Layer 2 - Unified Evaluator Comparison : only rows with independent
            response-level Gold (DNA official human, Aegis official response
            label; Fraud-R1/OR silver subsets marked as silver, guide 15).
  Layer 3 - Fraud-Aligned Official Subsets (guide 5.3).
Plus: Aegis prompt-only appendix (guide 3.2), operating points (guide 14),
cluster bootstrap + exact McNemar + Holm (guide 17) and consistency asserts.

LLM-panel 2026-08-05 labels are reported as "LLM-panel silver labels", never
as human Gold. Zero API. Outputs under metrics/ (guide 26).

Usage: python scripts/evaluate_exp2.py
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_cross_benchmark.metrics import (  # noqa: E402
    _fpr_at_recall,
    _recall_at_fpr,
    auprc_auroc,
    binary_metrics,
    exact_mcnemar,
    four_class_f1,
    group_bootstrap_delta,
)
from frauddistill.exp2_cross_benchmark.paths import (  # noqa: E402
    AEGIS_FRAUD_ALIGNED_CATEGORIES,
    BENCHMARKS,
    EXPERIMENT_DIR,
    FRAUD_ALIGNED_MANIFEST,
    FULL_MANIFEST,
    METRICS_DIR,
    SILVER_LABEL_TAG,
    TEACHER_T6_DIR,
)
from frauddistill.exp2_cross_benchmark.paths import DNA_FRAUD_ALIGNED_AREAS  # noqa: E402

SILVER_LABEL_NOTE = ("LLM-panel silver labels (2026-08-05 blind panel, adjudicated); "
                     "NOT human Gold (guide 15)")

BASELINE_FILES = {
    "fraudr1": "fraudr1/baseline_predictions/fraudr1_official_judge_predictions.jsonl",
    "orbench": "orbench/baseline_predictions/orbench_response_checker_predictions.jsonl",
    "do_not_answer": "do_not_answer/baseline_predictions/dna_longformer_harmful_predictions.jsonl",
    "aegis2": "aegis2/baseline_predictions/aegis_nemoguard_predictions.jsonl",
}
METHOD_NAMES = {
    "fraudr1": ("Fraud-R1 GPTCheck protocol", "FraudDistill Evidence MAT (T6)"),
    "orbench": ("OR-Bench official response checker", "FraudDistill Evidence MAT (T6)"),
    "do_not_answer": ("LibrAI/longformer-harmful-ro", "FraudDistill Evidence MAT (T6)"),
    "aegis2": ("nvidia/llama-3.1-nemoguard-8b-content-safety", "FraudDistill Evidence MAT (T6)"),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, max((m - rank) * pvals[idx], prev))
        adjusted[idx] = val
        prev = val
    return adjusted


def cohen_kappa(y1: np.ndarray, y2: np.ndarray):
    n = len(y1)
    if n == 0:
        return None
    classes = sorted(set(int(v) for v in np.unique(np.concatenate([y1, y2]))))
    if len(classes) < 2:
        return None
    po = float(np.mean(y1 == y2))
    pe = 0.0
    for c in classes:
        p1 = float(np.sum(y1 == c)) / n
        p2 = float(np.sum(y2 == c)) / n
        pe += p1 * p2
    if pe == 1.0:
        return None
    return round((po - pe) / (1.0 - pe), 4)


def _round4(v):
    return round(float(v), 4) if v is not None else None


def load_all() -> dict:
    manifest = read_jsonl(FULL_MANIFEST)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in manifest:
        by_source[r["source"]].append(r)
    data = {}
    for b in BENCHMARKS:
        teacher = {str(r["id"]): r for r in read_jsonl(TEACHER_T6_DIR / f"{b}_t6_predictions.jsonl")}
        baseline = {str(r["id"]): r for r in read_jsonl(EXPERIMENT_DIR / BASELINE_FILES[b])}
        silver = {str(r["id"]): r for r in read_jsonl(
            EXPERIMENT_DIR / b / "human_audit" / f"human_audit_adjudicated_{SILVER_LABEL_TAG}.jsonl")}
        data[b] = {"rows": by_source.get(b, []), "teacher": teacher, "baseline": baseline, "silver": silver}
    data["aegis2"]["teacher_prompt"] = {str(r["id"]): r for r in read_jsonl(TEACHER_T6_DIR / "aegis2_t6_prompt_predictions.jsonl")}
    return data


def fraudr1_native(data: dict) -> dict:
    rows, teacher, baseline, silver = (data["fraudr1"]["rows"], data["fraudr1"]["teacher"],
                                       data["fraudr1"]["baseline"], data["fraudr1"]["silver"])
    recs = []
    for r in rows:
        t = teacher.get(r["sample_id"]); b = baseline.get(r["sample_id"])
        if t is None or b is None:
            continue
        recs.append({
            "id": r["sample_id"], "group_id": r["group_id"],
            "category": r.get("official_category", ""), "language": r.get("language", ""),
            "variant": (r.get("metadata") or {}).get("fraudr1_variant", ""),
            "scenario": (r.get("metadata") or {}).get("fraudr1_scenario", ""),
            "data_type": (r.get("metadata") or {}).get("data_type", ""),
            "b_pred": int(b.get("prediction_binary", 0) or 0),
            "t_pred": int(t.get("prediction_binary", 0) or 0),
            "t_type": t.get("prediction_type", "safe"),
            "t_abstain": bool(t.get("abstain")),
            "t_score": float(t.get("risk_score", 0.0) or 0.0),
        })
    n = len(recs)
    out = {
        "n": n, "coverage": _round4(1.0 - sum(r["t_abstain"] for r in recs) / max(n, 1)),
        "judge_unsafe_rate": _round4(sum(r["b_pred"] for r in recs) / max(n, 1)),
        "t6_unsafe_rate": _round4(sum(r["t_pred"] for r in recs) / max(n, 1)),
        "pairwise_agreement": _round4(sum(1 for r in recs if r["b_pred"] == r["t_pred"]) / max(n, 1)),
        "kappa": cohen_kappa(np.array([r["b_pred"] for r in recs]), np.array([r["t_pred"] for r in recs])),
        "t6_only_positive": sum(1 for r in recs if r["t_pred"] == 1 and r["b_pred"] == 0),
        "baseline_only_positive": sum(1 for r in recs if r["b_pred"] == 1 and r["t_pred"] == 0),
        "both_positive": sum(1 for r in recs if r["b_pred"] == 1 and r["t_pred"] == 1),
        "category_rates": {}, "language_rates": {}, "variant_rates": {}, "scenario_rates": {}, "data_type_rates": {},
    }
    for dim in ("category", "language", "variant", "scenario", "data_type"):
        key = f"{dim}_rates"
        for val in sorted({r[dim] for r in recs}):
            rs = [r for r in recs if r[dim] == val]
            out[key][val] = {
                "n": len(rs),
                "judge_rate": _round4(sum(r["b_pred"] for r in rs) / len(rs)),
                "t6_rate": _round4(sum(r["t_pred"] for r in rs) / len(rs)),
            }
    out["silver_subset"] = silver_subset_metrics(recs, silver)
    return out

def silver_subset_metrics(recs: list[dict], silver: dict) -> dict:
    y, bp, tp, tscore, groups, gold_type, tf, pl, cr = [], [], [], [], [], [], [], [], []
    ttypes = []
    for r in recs:
        s = silver.get(r["id"])
        if s is None or s.get("binary") is None:
            continue
        y.append(int(s["binary"] or 0)); bp.append(r["b_pred"]); tp.append(r["t_pred"])
        tscore.append(r["t_score"]); groups.append(r["group_id"])
        gold_type.append(str(s.get("four_type", "safe")))
        tf.append(bool(s.get("trust_facilitation"))); pl.append(bool(s.get("partial_leakage"))); cr.append(bool(s.get("clean_refusal")))
        ttypes.append(str(r["t_type"]))
    out: dict = {"n": len(y), "label_note": SILVER_LABEL_NOTE}
    if not y:
        return out
    y = np.array(y); bp = np.array(bp); tp = np.array(tp); tscore = np.array(tscore)
    out["gold_positive_rate"] = _round4(float(y.mean()))
    out["baseline"] = {k: _round4(v) for k, v in binary_metrics(y, bp).items() if k != "n"}
    out["teacher"] = {k: _round4(v) for k, v in binary_metrics(y, tp).items() if k != "n"}
    out["teacher"]["four_class_macro_f1"] = _round4(four_class_f1(gold_type, ttypes))
    out["teacher"]["auprc"], out["teacher"]["auroc"] = auprc_auroc(y, tscore)
    out["mcnemar"] = exact_mcnemar(bp, tp)
    boot = group_bootstrap_delta(list(zip(groups, y.tolist(), bp.tolist(), tp.tolist(), tscore.tolist())), reps=10000)
    out["bootstrap_delta_macro_f1"] = boot
    def mech_recall(flag_list, preds):
        idx = [i for i, g in enumerate(flag_list) if g]
        if not idx:
            return None
        return _round4(sum(1 for i in idx if preds[i] == 1) / len(idx))
    out["mechanism"] = {
        "trust_facilitation_recall": mech_recall(tf, tp),
        "partial_leakage_recall": mech_recall(pl, tp),
        "clean_refusal_fpr": _round4(sum(1 for i, g in enumerate(cr) if g and tp[i] == 1) / max(sum(cr), 1)) if any(cr) else None,
    }
    return out


def orbench_native(data: dict) -> dict:
    rows, teacher, baseline, silver = (data["orbench"]["rows"], data["orbench"]["teacher"],
                                       data["orbench"]["baseline"], data["orbench"]["silver"])
    recs = []
    for r in rows:
        t = teacher.get(r["sample_id"]); b = baseline.get(r["sample_id"])
        if t is None or b is None:
            continue
        recs.append({
            "id": r["sample_id"], "group_id": r["group_id"],
            "prompt_type": (r.get("metadata") or {}).get("orbench_prompt_type", ""),
            "category": r.get("official_category", ""),
            "t_type": t.get("prediction_type", "safe"), "t_pred": int(t.get("prediction_binary", 0) or 0),
            "t_score": float(t.get("risk_score", 0.0) or 0.0), "t_abstain": bool(t.get("abstain")),
            "b_pred": int(b.get("prediction_binary", 0) or 0),
            "b_type": b.get("prediction_type", "safe"),
        })
    n = len(recs)
    def rate(ptype, cond):
        rs = [r for r in recs if r["prompt_type"] == ptype]
        if not rs:
            return None, 0
        return _round4(sum(1 for r in rs if cond(r)) / len(rs)), len(rs)
    hs_fpr, hs_n = rate("hard_safe", lambda r: r["t_type"] == "over_refusal")
    rs_fpr, rs_n = rate("regular_safe", lambda r: r["t_type"] == "over_refusal")
    tox_comp, tox_n = rate("toxic", lambda r: r["t_type"] in {"fraud_assistance", "refusal_failure"})
    tox_ref, _ = rate("toxic", lambda r: r["t_type"] == "over_refusal")
    b_hs_fpr, _ = rate("hard_safe", lambda r: r["b_type"] == "over_refusal")
    b_rs_fpr, _ = rate("regular_safe", lambda r: r["b_type"] == "over_refusal")
    b_tox_comp, _ = rate("toxic", lambda r: r["b_type"] == "refusal_failure")
    toxic_safety = _round4(1.0 - tox_comp) if tox_comp is not None else None
    safety_utility = _round4(((1.0 - hs_fpr) + toxic_safety) / 2) if hs_fpr is not None and toxic_safety is not None else None
    b_toxic_safety = _round4(1.0 - b_tox_comp) if b_tox_comp is not None else None
    b_safety_utility = _round4(((1.0 - b_hs_fpr) + b_toxic_safety) / 2) if b_hs_fpr is not None and b_toxic_safety is not None else None
    out = {
        "n": n, "n_hard_safe": hs_n, "n_regular_safe": rs_n, "n_toxic": tox_n,
        "teacher": {
            "hard_safe_fpr": hs_fpr, "regular_safe_fpr": rs_fpr,
            "toxic_over_refusal_rate": tox_ref, "toxic_harmful_compliance_rate": tox_comp,
            "toxic_safety_rate": toxic_safety, "safety_utility_score": safety_utility,
        },
        "baseline": {
            "hard_safe_fpr": b_hs_fpr, "regular_safe_fpr": b_rs_fpr,
            "toxic_harmful_compliance_rate": b_tox_comp,
            "toxic_safety_rate": b_toxic_safety, "safety_utility_score": b_safety_utility,
        },
        "prompt_type_dist": dict(Counter(r["prompt_type"] for r in recs)),
        "silver_subset": silver_subset_metrics(recs, silver),
    }
    return out


def dna_native(data: dict) -> dict:
    rows, teacher, baseline, silver = (data["do_not_answer"]["rows"], data["do_not_answer"]["teacher"],
                                       data["do_not_answer"]["baseline"], data["do_not_answer"]["silver"])
    recs = []
    for r in rows:
        if r.get("official_gold_binary") is None:
            continue
        t = teacher.get(r["sample_id"]); b = baseline.get(r["sample_id"])
        if t is None or b is None:
            continue
        recs.append({
            "id": r["sample_id"], "group_id": r["group_id"],
            "target_model": r.get("target_model", ""), "risk_area": r.get("official_category", ""),
            "gold": int(r["official_gold_binary"]), "gold_type": r.get("official_gold_type", ""),
            "b_pred": int(b.get("prediction_binary", 0) or 0),
            "b_score": float(b.get("prob_harmful", 0.0) or 0.0),
            "t_pred": int(t.get("prediction_binary", 0) or 0),
            "t_type": t.get("prediction_type", "safe"),
            "t_score": float(t.get("risk_score", 0.0) or 0.0),
            "t_abstain": bool(t.get("abstain")),
        })
    y = np.array([r["gold"] for r in recs]); bp = np.array([r["b_pred"] for r in recs])
    tp = np.array([r["t_pred"] for r in recs]); ts = np.array([r["t_score"] for r in recs])
    bs = np.array([r["b_score"] for r in recs])
    t_met = binary_metrics(y, tp); b_met = binary_metrics(y, bp)
    t_met["auprc"], t_met["auroc"] = auprc_auroc(y, ts)
    b_met["auprc"], b_met["auroc"] = auprc_auroc(y, bs)
    out = {
        "n": len(recs), "n_groups": len({r["group_id"] for r in recs}),
        "gold_positive_rate": _round4(float(y.mean())),
        "teacher": {k: _round4(v) for k, v in t_met.items()},
        "baseline": {k: _round4(v) for k, v in b_met.items()},
        "coverage": _round4(1.0 - sum(r["t_abstain"] for r in recs) / max(len(recs), 1)),
        "mcnemar": exact_mcnemar(bp, tp),
        "bootstrap": group_bootstrap_delta([(r["group_id"], r["gold"], r["b_pred"], r["t_pred"], r["t_score"]) for r in recs], reps=10000),
        "matched": {
            "matched_fpr_recall": _recall_at_fpr(y, ts, b_met["fpr"]),
            "matched_recall_fpr": _fpr_at_recall(y, ts, b_met["unsafe_recall"]),
            "auprc_delta": _round4(t_met["auprc"] - b_met["auprc"]),
        },
        "per_target_model": {},
        "within_prompt": within_prompt_metrics(recs),
    }
    for tm in sorted({r["target_model"] for r in recs}):
        rs = [r for r in recs if r["target_model"] == tm]
        yy = np.array([r["gold"] for r in rs]); tt = np.array([r["t_pred"] for r in rs]); bb = np.array([r["b_pred"] for r in rs])
        ttm = binary_metrics(yy, tt); btm = binary_metrics(yy, bb)
        out["per_target_model"][tm] = {
            "n": len(rs), "gold_rate": _round4(float(yy.mean())),
            "baseline_macro_f1": _round4(btm["true_macro_f1"]), "teacher_macro_f1": _round4(ttm["true_macro_f1"]),
            "baseline_unsafe_recall": _round4(btm["unsafe_recall"]), "teacher_unsafe_recall": _round4(ttm["unsafe_recall"]),
            "baseline_fpr": _round4(btm["fpr"]), "teacher_fpr": _round4(ttm["fpr"]),
        }
    out["silver_subset"] = silver_subset_metrics(
        [{"id": r["id"], "group_id": r["group_id"], "b_pred": r["b_pred"], "t_pred": r["t_pred"],
          "t_score": r["t_score"], "t_type": r["t_type"]} for r in recs], silver)
    if out["silver_subset"].get("mechanism"):
        out["silver_subset_mechanism"] = out["silver_subset"]["mechanism"]
    return out


def within_prompt_metrics(recs: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        groups[r["group_id"]].append(r)
    pair_acc_sum = 0.0
    pair_n = 0
    for gid, rs in groups.items():
        unsafe = [r for r in rs if r["gold"] == 1]
        safe = [r for r in rs if r["gold"] == 0]
        if not unsafe or not safe:
            continue
        for u in unsafe:
            for s in safe:
                pair_n += 1
                pair_acc_sum += 1.0 if u["t_score"] > s["t_score"] else 0.0
    return {"within_prompt_pair_accuracy": _round4(pair_acc_sum / pair_n) if pair_n else None, "pairs": pair_n}

def aegis_response_native(data: dict) -> dict:
    rows, teacher, baseline, silver = (data["aegis2"]["rows"], data["aegis2"]["teacher"],
                                       data["aegis2"]["baseline"], data["aegis2"]["silver"])
    resp_rows = [r for r in rows if r["evaluation_mode"] == "response"]
    recs = []
    for r in resp_rows:
        if r.get("official_gold_binary") is None:
            continue
        t = teacher.get(r["sample_id"]); b = baseline.get(r["sample_id"])
        if t is None or b is None:
            continue
        recs.append({
            "id": r["sample_id"], "group_id": r["group_id"],
            "gold": int(r["official_gold_binary"]),
            "b_pred": int(b.get("prediction_binary", 0) or 0),
            "t_pred": int(t.get("prediction_binary", 0) or 0),
            "t_score": float(t.get("risk_score", 0.0) or 0.0),
            "t_type": t.get("prediction_type", "safe"), "t_abstain": bool(t.get("abstain")),
            "categories": r.get("official_category", ""),
        })
    y = np.array([r["gold"] for r in recs]); bp = np.array([r["b_pred"] for r in recs]); tp = np.array([r["t_pred"] for r in recs])
    ts = np.array([r["t_score"] for r in recs])
    t_met = binary_metrics(y, tp); b_met = binary_metrics(y, bp)
    t_met["auprc"], t_met["auroc"] = auprc_auroc(y, ts)
    out = {
        "n": len(recs), "n_groups": len({r["group_id"] for r in recs}),
        "gold_positive_rate": _round4(float(y.mean())),
        "teacher": {k: _round4(v) for k, v in t_met.items()},
        "baseline": {k: _round4(v) for k, v in b_met.items()},
        "coverage": _round4(1.0 - sum(r["t_abstain"] for r in recs) / max(len(recs), 1)),
        "mcnemar": exact_mcnemar(bp, tp),
        "bootstrap": group_bootstrap_delta([(r["group_id"], r["gold"], r["b_pred"], r["t_pred"], r["t_score"]) for r in recs], reps=10000),
        "matched": {
            "matched_fpr_recall": _recall_at_fpr(y, ts, b_met["fpr"]),
            "matched_recall_fpr": _fpr_at_recall(y, ts, b_met["unsafe_recall"]),
        },
        "per_category_recall": {},
        "silver_subset": silver_subset_metrics(
            [{"id": r["id"], "group_id": r["group_id"], "b_pred": r["b_pred"], "t_pred": r["t_pred"],
              "t_score": r["t_score"], "t_type": r["t_type"]} for r in recs], silver),
    }
    for cat in sorted({c.strip() for r in recs for c in (r["categories"] or "").split(",") if c.strip()}):
        rs = [r for r in recs if cat in (r["categories"] or "")]
        if len(rs) < 10:
            continue
        yy = np.array([r["gold"] for r in rs]); tt = np.array([r["t_pred"] for r in rs]); bb = np.array([r["b_pred"] for r in rs])
        out["per_category_recall"][cat] = {
            "n": len(rs), "gold_rate": _round4(float(yy.mean())),
            "teacher_recall": _round4(binary_metrics(yy, tt)["unsafe_recall"]),
            "baseline_recall": _round4(binary_metrics(yy, bb)["unsafe_recall"]),
            "teacher_fpr": _round4(binary_metrics(yy, tt)["fpr"]),
            "baseline_fpr": _round4(binary_metrics(yy, bb)["fpr"]),
        }
    return out


def aegis_prompt_appendix(data: dict) -> dict:
    rows = [r for r in data["aegis2"]["rows"] if r["evaluation_mode"] == "prompt_only"]
    teacher_p = data["aegis2"]["teacher_prompt"]
    baseline = data["aegis2"]["baseline"]
    recs = []
    for r in rows:
        if r.get("official_gold_binary") is None:
            continue
        t = teacher_p.get(r["sample_id"])
        b = baseline.get(r["sample_id"])
        if t is None or b is None:
            continue
        m = re.search(r'"User Safety"\s*:\s*"(safe|unsafe)"', b.get("raw_output") or "")
        b_prompt_pred = 1 if (m and m.group(1) == "unsafe") else (0 if m else None)
        if b_prompt_pred is None:
            continue
        recs.append({
            "id": r["sample_id"], "group_id": r["group_id"],
            "gold": int(r["official_gold_binary"]),
            "b_pred": b_prompt_pred,
            "t_pred": int(t.get("prediction_binary", 0) or 0),
            "t_score": float(t.get("risk_score", 0.0) or 0.0),
            "t_abstain": bool(t.get("abstain")),
        })
    out = {"n": len(recs), "gold_positive_rate": _round4(float(np.mean([r["gold"] for r in recs]))) if recs else None}
    if recs:
        y = np.array([r["gold"] for r in recs]); bp = np.array([r["b_pred"] for r in recs]); tp = np.array([r["t_pred"] for r in recs])
        ts = np.array([r["t_score"] for r in recs])
        t_met = binary_metrics(y, tp); b_met = binary_metrics(y, bp)
        t_met["auprc"], t_met["auroc"] = auprc_auroc(y, ts)
        out["teacher"] = {k: _round4(v) for k, v in t_met.items()}
        out["baseline"] = {k: _round4(v) for k, v in b_met.items()}
        out["mcnemar"] = exact_mcnemar(bp, tp)
        out["bootstrap"] = group_bootstrap_delta([(r["group_id"], r["gold"], r["b_pred"], r["t_pred"], r["t_score"]) for r in recs], reps=10000)
        out["note"] = "input-risk transfer only; no answer-unsafe claim; not mixed with response-level Macro-F1 (guide 3)"
    return out


def unified_rows(l1: dict, dna: dict, aegis_resp: dict) -> list[dict]:
    rows = []
    rows.append({
        "benchmark": "Do-Not-Answer (official human labels)",
        "gold_note": "official human harmfulness (full pool)", "n": dna["n"],
        "n_pos": int(round(dna["gold_positive_rate"] * dna["n"])),
        "baseline": dna["baseline"], "teacher": dna["teacher"], "mcnemar": dna["mcnemar"], "bootstrap": dna["bootstrap"],
        "key": "dna_official",
    })
    rows.append({
        "benchmark": "Aegis 2.0 response (official labels)",
        "gold_note": "official response_label (full test)", "n": aegis_resp["n"],
        "n_pos": int(round(aegis_resp["gold_positive_rate"] * aegis_resp["n"])),
        "baseline": aegis_resp["baseline"], "teacher": aegis_resp["teacher"],
        "mcnemar": aegis_resp["mcnemar"], "bootstrap": aegis_resp["bootstrap"],
        "key": "aegis_response_official",
    })
    for b, label in (("fraudr1", "Fraud-R1"), ("orbench", "OR-Bench")):
        sr = l1[b]["silver_subset"]
        rows.append({
            "benchmark": f"{label} (audited subset)",
            "gold_note": SILVER_LABEL_NOTE, "n": sr.get("n", 0),
            "n_pos": int(round((sr.get("gold_positive_rate") or 0) * sr.get("n", 0))),
            "baseline": sr.get("baseline", {}), "teacher": sr.get("teacher", {}),
            "mcnemar": sr.get("mcnemar", {}), "bootstrap": sr.get("bootstrap_delta_macro_f1", {}),
            "key": f"{b}_silver",
        })
    return rows


def fraud_aligned(data: dict) -> dict:
    out = {}
    rows = [r for r in data["do_not_answer"]["rows"]
            if (r.get("official_category") or "") in DNA_FRAUD_ALIGNED_AREAS and r.get("official_gold_binary") is not None]
    teacher, baseline = data["do_not_answer"]["teacher"], data["do_not_answer"]["baseline"]
    recs = []
    for r in rows:
        t = teacher.get(r["sample_id"]); b = baseline.get(r["sample_id"])
        if t is None or b is None:
            continue
        recs.append({"id": r["sample_id"], "group_id": r["group_id"], "gold": int(r["official_gold_binary"]),
                     "b_pred": int(b.get("prediction_binary", 0) or 0), "b_score": float(b.get("prob_harmful", 0.0) or 0.0),
                     "t_pred": int(t.get("prediction_binary", 0) or 0), "t_score": float(t.get("risk_score", 0.0) or 0.0)})
    if recs:
        y = np.array([r["gold"] for r in recs]); bp = np.array([r["b_pred"] for r in recs]); tp = np.array([r["t_pred"] for r in recs])
        ts = np.array([r["t_score"] for r in recs]); bs = np.array([r["b_score"] for r in recs])
        t_met = binary_metrics(y, tp); b_met = binary_metrics(y, bp)
        t_met["auprc"], _ = auprc_auroc(y, ts); b_met["auprc"], _ = auprc_auroc(y, bs)
        out["dna_malicious_uses"] = {
            "n": len(recs), "gold_positive_rate": _round4(float(y.mean())),
            "baseline": {k: _round4(v) for k, v in b_met.items()}, "teacher": {k: _round4(v) for k, v in t_met.items()},
            "mcnemar": exact_mcnemar(bp, tp),
            "bootstrap": group_bootstrap_delta([(r["group_id"], r["gold"], r["b_pred"], r["t_pred"], r["t_score"]) for r in recs], reps=10000),
            "matched_fpr_recall": _recall_at_fpr(y, ts, b_met["fpr"]),
        }
    rows = [r for r in data["aegis2"]["rows"] if r["evaluation_mode"] == "response" and r.get("official_gold_binary") is not None]
    teacher, baseline = data["aegis2"]["teacher"], data["aegis2"]["baseline"]
    recs = []
    for r in rows:
        cats = {c.strip() for c in (r.get("official_category") or "").split(",") if c.strip()}
        if not (cats & AEGIS_FRAUD_ALIGNED_CATEGORIES):
            continue
        t = teacher.get(r["sample_id"]); b = baseline.get(r["sample_id"])
        if t is None or b is None:
            continue
        recs.append({"id": r["sample_id"], "group_id": r["group_id"], "gold": int(r["official_gold_binary"]),
                     "b_pred": int(b.get("prediction_binary", 0) or 0),
                     "t_pred": int(t.get("prediction_binary", 0) or 0), "t_score": float(t.get("risk_score", 0.0) or 0.0)})
    if recs:
        y = np.array([r["gold"] for r in recs]); bp = np.array([r["b_pred"] for r in recs]); tp = np.array([r["t_pred"] for r in recs])
        ts = np.array([r["t_score"] for r in recs])
        t_met = binary_metrics(y, tp); b_met = binary_metrics(y, bp)
        t_met["auprc"], _ = auprc_auroc(y, ts)
        out["aegis_fraud_categories"] = {
            "n": len(recs), "gold_positive_rate": _round4(float(y.mean())),
            "categories": sorted(AEGIS_FRAUD_ALIGNED_CATEGORIES),
            "baseline": {k: _round4(v) for k, v in b_met.items()}, "teacher": {k: _round4(v) for k, v in t_met.items()},
            "mcnemar": exact_mcnemar(bp, tp),
            "bootstrap": group_bootstrap_delta([(r["group_id"], r["gold"], r["b_pred"], r["t_pred"], r["t_score"]) for r in recs], reps=10000),
        }
    return out


def operating_points(l1: dict, dna: dict, aegis_resp: dict) -> dict:
    calib = {}
    calib_path = METRICS_DIR / "calibration.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
    out: dict = {"rule": "thresholds only from pre-registered non-test sources (guide 14)"}
    out["dna"] = {
        "categorical_0_5_macro_f1": dna["teacher"]["true_macro_f1"],
        "auprc": dna["teacher"]["auprc"],
        "matched_fpr_recall": dna["matched"]["matched_fpr_recall"],
        "matched_recall_fpr": dna["matched"]["matched_recall_fpr"],
        "auprc_delta_vs_longformer": dna["matched"]["auprc_delta"],
    }
    ac = calib.get("aegis_response_official_validation", {})
    out["aegis_response"] = {
        "categorical_0_5_macro_f1": aegis_resp["teacher"]["true_macro_f1"],
        "validation_best_mcc_point": ac.get("best_mcc"),
        "validation_fpr_le_0_08_point": ac.get("best_fpr_le_0_08"),
        "matched_fpr_recall": aegis_resp["matched"]["matched_fpr_recall"],
        "matched_recall_fpr": aegis_resp["matched"]["matched_recall_fpr"],
    }
    fc = calib.get("fraudr1_exp3_dev", {})
    out["fraudr1"] = {
        "exp3_dev_recall_first_point": fc.get("recall_first_fpr_le_0_12"),
        "silver_subset_teacher": l1["fraudr1"]["silver_subset"].get("teacher"),
    }
    return out

def main() -> None:
    data = load_all()
    l1: dict = {}
    l1["fraudr1"] = fraudr1_native(data)
    l1["orbench"] = orbench_native(data)
    dna = dna_native(data)
    aegis_resp = aegis_response_native(data)
    aegis_prompt = aegis_prompt_appendix(data)

    unified = unified_rows(l1, dna, aegis_resp)
    faligned = fraud_aligned(data)
    ops = operating_points(l1, dna, aegis_resp)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    prim = []
    for u in unified:
        if u["key"] in ("dna_official", "aegis_response_official"):
            prim.append((u["key"], u["mcnemar"], u["bootstrap"]))
    for key in ("dna_malicious_uses", "aegis_fraud_categories"):
        if key in faligned:
            prim.append((key, faligned[key]["mcnemar"], faligned[key]["bootstrap"]))
    pvals = [p["p"] for _, p, _ in prim]
    adj = holm_adjust(pvals) if pvals else []
    paired = {}
    for i, (key, mcnemar, boot) in enumerate(prim):
        paired[key] = {"mcnemar": mcnemar, "mcnemar_p_holm": _round4(adj[i]) if adj else None, "bootstrap": boot}
    (METRICS_DIR / "paired_significance.json").write_text(json.dumps(paired, ensure_ascii=False, indent=1), encoding="utf-8")

    import csv
    rows_out = []
    for u in unified:
        if "Fraud-R1" in u["benchmark"]:
            bname = METHOD_NAMES["fraudr1"][0]
        elif "OR-Bench" in u["benchmark"]:
            bname = METHOD_NAMES["orbench"][0]
        elif "Do-Not-Answer" in u["benchmark"]:
            bname = METHOD_NAMES["do_not_answer"][0]
        else:
            bname = METHOD_NAMES["aegis2"][0]
        rows_out.append({
            "benchmark": u["benchmark"], "gold_note": u["gold_note"], "n": u["n"], "n_pos": u["n_pos"],
            "baseline_method": bname,
            "baseline_macro_f1": (u["baseline"] or {}).get("true_macro_f1"),
            "teacher_macro_f1": (u["teacher"] or {}).get("true_macro_f1"),
            "baseline_recall": (u["baseline"] or {}).get("unsafe_recall"),
            "teacher_recall": (u["teacher"] or {}).get("unsafe_recall"),
            "baseline_fpr": (u["baseline"] or {}).get("fpr"),
            "teacher_fpr": (u["teacher"] or {}).get("fpr"),
            "baseline_auprc": (u["baseline"] or {}).get("auprc"),
            "teacher_auprc": (u["teacher"] or {}).get("auprc"),
            "baseline_mcc": (u["baseline"] or {}).get("mcc"),
            "teacher_mcc": (u["teacher"] or {}).get("mcc"),
            "delta_macro_f1": _round4(((u["teacher"] or {}).get("true_macro_f1") or 0) - ((u["baseline"] or {}).get("true_macro_f1") or 0)),
            "mcnemar_p": (u["mcnemar"] or {}).get("p"),
            "bootstrap_delta": (u["bootstrap"] or {}).get("observed_delta"),
            "bootstrap_ci95_low": (u["bootstrap"] or {}).get("ci95_low"),
            "bootstrap_ci95_high": (u["bootstrap"] or {}).get("ci95_high"),
        })
    with (METRICS_DIR / "unified_audited.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)

    native_rows = []
    fr = l1["fraudr1"]
    native_rows.append({"benchmark": "Fraud-R1", "n": fr["n"], "metric": "judge_unsafe_rate", "baseline": fr["judge_unsafe_rate"], "teacher": fr["t6_unsafe_rate"]})
    native_rows.append({"benchmark": "Fraud-R1", "n": fr["n"], "metric": "pairwise_agreement", "baseline": None, "teacher": fr["pairwise_agreement"]})
    ob = l1["orbench"]
    for k in ("hard_safe_fpr", "regular_safe_fpr", "toxic_harmful_compliance_rate", "safety_utility_score"):
        native_rows.append({"benchmark": "OR-Bench", "n": ob["n"], "metric": k,
                            "baseline": (ob["baseline"] or {}).get(k), "teacher": (ob["teacher"] or {}).get(k)})
    for k in ("true_macro_f1", "unsafe_recall", "fpr", "auprc", "mcc"):
        native_rows.append({"benchmark": "Do-Not-Answer", "n": dna["n"], "metric": k,
                            "baseline": (dna["baseline"] or {}).get(k), "teacher": (dna["teacher"] or {}).get(k)})
        native_rows.append({"benchmark": "Aegis response", "n": aegis_resp["n"], "metric": k,
                            "baseline": (aegis_resp["baseline"] or {}).get(k), "teacher": (aegis_resp["teacher"] or {}).get(k)})
    with (METRICS_DIR / "full_native.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "n", "metric", "baseline", "teacher"])
        w.writeheader(); w.writerows(native_rows)

    fa_rows = []
    for key, entry in faligned.items():
        fa_rows.append({"subset": key, "n": entry["n"], "gold_rate": entry["gold_positive_rate"],
                        "baseline_macro_f1": entry["baseline"]["true_macro_f1"], "teacher_macro_f1": entry["teacher"]["true_macro_f1"],
                        "baseline_recall": entry["baseline"]["unsafe_recall"], "teacher_recall": entry["teacher"]["unsafe_recall"],
                        "baseline_fpr": entry["baseline"]["fpr"], "teacher_fpr": entry["teacher"]["fpr"],
                        "mcnemar_p": entry["mcnemar"]["p"]})
    with (METRICS_DIR / "fraud_aligned.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fa_rows[0].keys()))
        w.writeheader(); w.writerows(fa_rows)

    (METRICS_DIR / "operating_points.json").write_text(json.dumps(ops, ensure_ascii=False, indent=1), encoding="utf-8")
    (METRICS_DIR / "aegis_prompt_appendix.json").write_text(json.dumps(aegis_prompt, ensure_ascii=False, indent=1), encoding="utf-8")
    (METRICS_DIR / "mechanism_metrics.json").write_text(json.dumps(
        {"fraudr1": l1["fraudr1"]["silver_subset"].get("mechanism", {}),
         "orbench": l1["orbench"]["silver_subset"].get("mechanism", {}),
         "do_not_answer": dna.get("silver_subset_mechanism", {}),
         "aegis2": aegis_resp["silver_subset"].get("mechanism", {})},
        ensure_ascii=False, indent=1), encoding="utf-8")

    canonical = {
        "guide": "2026-08-06 full-coverage",
        "silver_label_note": SILVER_LABEL_NOTE,
        "layer1_full_native": {
            "fraudr1": {k: v for k, v in l1["fraudr1"].items() if k != "silver_subset"},
            "orbench": {k: v for k, v in l1["orbench"].items() if k != "silver_subset"},
            "do_not_answer": {k: v for k, v in dna.items() if k != "silver_subset"},
            "aegis2_response": {k: v for k, v in aegis_resp.items() if k != "silver_subset"},
            "aegis2_prompt_appendix": aegis_prompt,
        },
        "layer2_unified": unified,
        "layer3_fraud_aligned": faligned,
        "operating_points": ops,
        "paired_significance": paired,
        "silver_subsets": {
            "fraudr1": l1["fraudr1"].get("silver_subset", {}),
            "orbench": l1["orbench"].get("silver_subset", {}),
            "do_not_answer": dna.get("silver_subset", {}),
            "aegis2": aegis_resp.get("silver_subset", {}),
        },
    }
    (METRICS_DIR / "canonical_metrics.json").write_text(json.dumps(canonical, ensure_ascii=False, indent=1), encoding="utf-8")
    print("===== LAYER 2: UNIFIED EVALUATOR COMPARISON =====")
    for u in unified:
        bd = u["bootstrap"] or {}
        print(f"{u['benchmark']:42s} N={u['n']:5d} | T MF1={(u['teacher'] or {}).get('true_macro_f1')} vs B {(u['baseline'] or {}).get('true_macro_f1')} | dMF1 CI=[{bd.get('ci95_low')},{bd.get('ci95_high')}] | p={(u['mcnemar'] or {}).get('p')}")
    print("===== LAYER 3: FRAUD-ALIGNED =====")
    for key, entry in faligned.items():
        bd = entry.get("bootstrap") or {}
        print(f"{key:30s} N={entry['n']:5d} | T MF1={entry['teacher']['true_macro_f1']} vs B {entry['baseline']['true_macro_f1']} | CI=[{bd.get('ci95_low')},{bd.get('ci95_high')}]")
    print("===== OR NATIVE =====")
    print(json.dumps(l1["orbench"]["teacher"], ensure_ascii=False))
    print("===== AEGIS PROMPT APPENDIX =====")
    print(json.dumps({k: v for k, v in aegis_prompt.items() if k != "note"}, ensure_ascii=False)[:600])
    print("outputs ->", METRICS_DIR)


if __name__ == "__main__":
    main()
