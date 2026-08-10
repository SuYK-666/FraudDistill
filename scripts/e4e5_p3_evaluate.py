# -*- coding: utf-8 -*-
"""E5 P3 cascade evaluation: Student -> DeepSeek audit on top-K ambiguous rows.

For each K in {60,120,180,240,300}:
  audited rows (smallest |risk-0.5|) -> DeepSeek judge label
  other rows -> student threshold (0.5622)
Writes p3_policies.jsonl, p3_paired_statistics.json, tables/e5_p3_policies.md
and updates e5/report.json P3 section (K=180 primary per guide <=15% API rate).

Usage: python scripts/e4e5_p3_evaluate.py --protocol-dir outputs/exp4_unseen_student_v2/e4v2_FINAL
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from frauddistill.e4e5_v2.schemas import read_jsonl, write_jsonl
from frauddistill.e4e5_v2.metrics import binary_metrics
from collections import defaultdict
from frauddistill.e4e5_v2.cluster_bootstrap import exact_mcnemar

THRESHOLD = 0.5622


def cluster_bootstrap_pred_diff(y, pa, pb, fams, replicates, seed, metric):
    """Paired family-cluster bootstrap over final (binary) predictions."""
    rng = np.random.default_rng(seed)
    fam_to_idx = defaultdict(list)
    for i, f in enumerate(fams):
        fam_to_idx[str(f) or "no_fam"].append(i)
    fam_list = list(fam_to_idx.keys())
    diffs = []
    for _ in range(replicates):
        chosen = rng.choice(fam_list, size=len(fam_list), replace=True)
        idx = []
        for f in chosen:
            idx.extend(fam_to_idx[f])
        idx = np.array(idx, dtype=int)
        ma = binary_metrics(y[idx], pa[idx], pred=pa[idx], threshold=THRESHOLD)
        mb = binary_metrics(y[idx], pb[idx], pred=pb[idx], threshold=THRESHOLD)
        diffs.append(ma[metric] - mb[metric])
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "metric": metric, "replicates": replicates,
        "mean_diff": round(float(diffs.mean()), 5),
        "ci95": [round(float(lo), 5), round(float(hi), 5)],
        "ci95_above_zero": bool(lo > 0),
        "p_value_approx": round(float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))), 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    proto = Path(args.protocol_dir)
    e5_dir = proto / "e5"
    test_rows = read_jsonl(proto / "manifests" / "frozen_test.jsonl")
    preds = {p["id"]: p for p in read_jsonl(proto / "predictions" / "final_student.jsonl")}
    audit = [json.loads(l) for l in open(e5_dir / "p3_audit_results.jsonl", encoding="utf-8")]
    audit_by_id = {r["id"]: r for r in audit}
    assert len(audit) == 600 and all(r.get("ds_label") is not None for r in audit), "audit incomplete"

    rows = []
    for r in test_rows:
        p = preds.get(r["id"])
        if p is None:
            continue
        rows.append({
            "id": r["id"], "family_id": r.get("family_id") or r["id"],
            "primary_shift": r.get("primary_shift", ""),
            "gold": 1 if r["gold_label"] == "unsafe" else 0,
            "score": float(p["risk_score"]),
            "student_pred": 1 if p["risk_score"] >= THRESHOLD else 0,
        })
    n = len(rows)
    y = np.array([r["gold"] for r in rows])
    s = np.array([r["score"] for r in rows])
    sp = np.array([r["student_pred"] for r in rows])
    amb = np.abs(s - 0.5)
    order = np.argsort(amb)  # ascending ambiguity
    fams = [r["family_id"] for r in rows]
    shifts = [r["primary_shift"] for r in rows]

    policies = []
    ks = (60, 120, 180, 240, 300, 360, 420, 480, 540, 600)
    for k in ks:
        audited = order[:k]
        final = sp.copy()
        ds_unsafe = 0
        agree = 0
        for i in audited:
            a = audit_by_id[rows[i]["id"]]
            lbl = 1 if a["ds_label"] == "unsafe" else 0
            final[i] = lbl
            ds_unsafe += lbl
            agree += (lbl == y[i])
        m = binary_metrics(y, s, pred=final, threshold=THRESHOLD, label="P3_K%d" % k)
        pol = {
            "policy": "P3_K%d" % k, "api_rate": round(k / n, 4), "n_audited": k,
            "n_ds_unsafe": int(ds_unsafe),
            "judge_agreement_with_gold": round(agree / k, 4),
            "macro_f1": m["macro_f1"], "recall": m["recall"], "precision": m["precision"],
            "fpr": m["fpr"], "mcc": m["mcc"], "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"], "auroc": m.get("auroc"),
            "auprc": m.get("auprc"),
        }
        per_shift = {}
        for sh in ("U1_category", "U2_source", "U3_target_style"):
            idx = np.array([i for i in range(n) if shifts[i] == sh])
            audited_here = int(np.isin(idx, audited).sum())
            per_shift[sh] = {
                "n": int(len(idx)),
                "audited": audited_here,
                "audit_rate": round(audited_here / max(len(idx), 1), 4),
                "audited_unsafe_gold": int(y[idx[np.isin(idx, audited)]].sum()),
            }
        pol["per_shift"] = per_shift
        policies.append(pol)

    write_jsonl(e5_dir / "p3_policies.jsonl", policies)

    # primary: K=180 (15% API rate, guide gate target)
    k_primary = 180
    p3 = next(p for p in policies if p["policy"] == "P3_K180")
    audited = order[:k_primary]
    final = sp.copy()
    for i in audited:
        final[i] = 1 if audit_by_id[rows[i]["id"]]["ds_label"] == "unsafe" else 0
    stats = {
        "P3_vs_P0": {
            "bootstrap_macro_f1": cluster_bootstrap_pred_diff(y, final, sp, fams, args.replicates, args.seed, "macro_f1"),
            "bootstrap_recall": cluster_bootstrap_pred_diff(y, final, sp, fams, args.replicates, args.seed, "recall"),
            "bootstrap_fpr": cluster_bootstrap_pred_diff(y, final, sp, fams, args.replicates, args.seed, "fpr"),
            "mcnemar": exact_mcnemar(final, sp, y),
        }
    }
    (e5_dir / "p3_paired_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- update e5/report.json P3 section ----
    rp = e5_dir / "report.json"
    rep = json.loads(rp.read_text(encoding="utf-8"))
    rep["P3"] = {
        "policy": p3["policy"], "n": n, "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
        "macro_f1": p3["macro_f1"], "recall": p3["recall"], "precision": p3["precision"],
        "fpr": p3["fpr"], "mcc": p3["mcc"], "accuracy": p3["accuracy"],
        "balanced_accuracy": p3["balanced_accuracy"], "auroc": p3["auroc"], "auprc": p3["auprc"],
        "api_rate": p3["api_rate"], "n_audited": p3["n_audited"],
        "judge_agreement_with_gold": p3["judge_agreement_with_gold"],
        "per_shift": p3["per_shift"],
        "fallback_mode": "deepseek_audit", "fallback_model": "deepseek-chat",
        "n_fallback": k_primary, "coverage": 1.0,
    }
    rep["P3_paired"] = stats
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- markdown table ----
    md = ["| Policy | API rate | MF1 | Recall | Precision | FPR | MCC | AUROC | AUPRC | Judge-agree |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in policies:
        md.append("| %s | %.1f%% | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f |" % (
            p["policy"], p["api_rate"] * 100, p["macro_f1"], p["recall"], p["precision"],
            p["fpr"], p["mcc"], p["auroc"] or 0, p["auprc"] or 0, p["judge_agreement_with_gold"]))
    (proto / "tables" / "e5_p3_policies.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for p in policies:
        print("[p3] %s: mf1=%.4f recall=%.4f fpr=%.4f mcc=%.4f agree=%.3f" % (
            p["policy"], p["macro_f1"], p["recall"], p["fpr"], p["mcc"], p["judge_agreement_with_gold"]))
    print("[p3] mcnemar P3_K180 vs P0:", stats["P3_vs_P0"]["mcnemar"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
