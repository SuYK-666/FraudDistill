# -*- coding: utf-8 -*-
"""Paired cluster bootstrap (10k), exact McNemar, Holm correction and canonical metrics.

Guide 3.2-3.4, 3.8, 25.1:
- observed_delta must equal metric(A, full) - metric(B, full) for A->B pairs;
- bootstrap_mean_delta and ci95 are reported separately;
- exact McNemar uses b = A wrong & B right, c = A right & B wrong;
- Holm correction uses statsmodels.multipletests (raw p saved too);
- one canonical outputs/metrics/final_metrics.json feeds report/tables/figures.

Usage: python scripts/evaluate_exp3.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

import build_exp3_agent_ablations as ablations

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("EXP3_OUT_ROOT") or REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs")
METRICS = OUT_ROOT / "metrics"
N_REPS = 10000
SEED = 20260804


# ------------------------------------------------------------------ helpers
def ece(p, y, bins=10):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(y), 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi) if hi == 1.0 else (p >= lo) & (p < hi)
        if not mask.any():
            continue
        out += (mask.sum() / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(out)


def _macro_f1(y, pred):
    """True macro-F1 (guide 3.1): mean of per-class F1, zero_division=0."""
    out = []
    for cls in (0, 1):
        tp = int(((pred == cls) & (y == cls)).sum())
        fp = int(((pred == cls) & (y != cls)).sum())
        fn = int(((pred != cls) & (y == cls)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9) if (prec + rec) > 0 else 0.0
        out.append(f1)
    return float(np.mean(out))


def _recall(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp / max(tp + fn, 1)


def _fpr(y, pred):
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return fp / max(tn + fp, 1)


def overall_metrics(y, pred, scores=None):
    """Guide 23.1 metric set for one method on one (pooled) sample."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray([1 if str(x) == "unsafe" else 0 for x in pred], dtype=int)
    n = len(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    m = {
        "n": n,
        "acc": round((tp + tn) / max(n, 1), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "unsafe_f1": round(float(f1_score(y, pred, pos_label=1, zero_division=0)), 4),
        "safe_f1": round(float(f1_score(y, pred, pos_label=0, zero_division=0)), 4),
        "macro_f1": round(_macro_f1(y, pred), 4),
        "fpr": round(_fpr(y, pred), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if scores is not None:
        sc = np.asarray(scores, dtype=float)
        if len(np.unique(sc)) > 1 and 0 < y.sum() < n:
            m["auprc"] = round(float(average_precision_score(y, sc)), 4)
            m["auroc"] = round(float(roc_auc_score(y, sc)), 4)
        p = np.clip(sc, 0.0, 1.0)
        m["brier"] = round(float(np.mean((p - y) ** 2)), 4)
        m["ece"] = round(ece(p, y, bins=10), 4)
    return m


def bootstrap_ci(recs, sa, sb, n_reps=N_REPS, seed=SEED):
    """Cluster bootstrap (guide 25.1/25.3): resample groups, recompute pooled metric.

    Vectorized: per-record arrays are precomputed; each repetition concatenates
    the rows of the resampled groups, so 10k reps stay fast.
    """
    rng = np.random.default_rng(seed)
    y = np.array([1 if r["sample"]["gold_label"] == "unsafe" else 0 for r in recs], dtype=int)
    pa = np.array([1 if sa["label"](r) == "unsafe" else 0 for r in recs], dtype=int)
    pb = np.array([1 if sb["label"](r) == "unsafe" else 0 for r in recs], dtype=int)
    sca = np.array([float(sa["score"](r)) for r in recs], dtype=float)
    scb = np.array([float(sb["score"](r)) for r in recs], dtype=float)
    gid = np.array([str(r.get("group_id", r.get("id", ""))) for r in recs])
    _, inverse = np.unique(gid, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    group_rows = [np.where(inverse == i)[0] for i in range(n_groups)]
    idx = rng.integers(0, n_groups, size=(n_reps, n_groups))
    acc = {"macro_f1": np.empty(n_reps), "recall": np.empty(n_reps),
           "fpr": np.empty(n_reps), "auprc": np.empty(n_reps)}
    for rep in range(n_reps):
        rows = np.concatenate([group_rows[i] for i in idx[rep]])
        yy, ppa, ppb = y[rows], pa[rows], pb[rows]
        acc["macro_f1"][rep] = _macro_f1(yy, ppb) - _macro_f1(yy, ppa)
        acc["recall"][rep] = _recall(yy, ppb) - _recall(yy, ppa)
        acc["fpr"][rep] = _fpr(yy, ppb) - _fpr(yy, ppa)
        ssa, ssb = sca[rows], scb[rows]
        if len(np.unique(ssb)) > 1 and 0 < yy.sum() < len(yy):
            acc["auprc"][rep] = average_precision_score(yy, ssb) - average_precision_score(yy, ssa)
        else:
            acc["auprc"][rep] = float("nan")
    out = {}
    for k in ("macro_f1", "recall", "fpr", "auprc"):
        v = acc[k]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out[k] = {"observed_delta": float("nan"), "bootstrap_mean_delta": float("nan"),
                      "ci95": [float("nan"), float("nan")], "ci_excludes_zero": False}
            continue
        lo, hi = np.percentile(v, [2.5, 97.5])
        obs_a = _macro_f1(y, pa) if k == "macro_f1" else (_recall(y, pa) if k == "recall"
                else (_fpr(y, pa) if k == "fpr" else
                      (average_precision_score(y, sca) if len(np.unique(sca)) > 1 else float("nan"))))
        obs_b = _macro_f1(y, pb) if k == "macro_f1" else (_recall(y, pb) if k == "recall"
                else (_fpr(y, pb) if k == "fpr" else
                      (average_precision_score(y, scb) if len(np.unique(scb)) > 1 else float("nan"))))
        out[k] = {
            "observed_delta": round(float(obs_b - obs_a), 4),
            "bootstrap_mean_delta": round(float(np.mean(v)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "ci_excludes_zero": bool((lo > 0) or (hi < 0)),
        }
    return out


def exact_mcnemar(recs, get_label_a, get_label_b):
    """b = A wrong & B right; c = A right & B wrong (guide 3.3)."""
    b = c = 0
    for r in recs:
        gold = r["sample"]["gold_label"]
        a_ok = get_label_a(r) == gold
        b_ok = get_label_b(r) == gold
        if (not a_ok) and b_ok:
            b += 1
        elif a_ok and (not b_ok):
            c += 1
    p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if (b + c) > 0 else 1.0
    return {"b_only_a_wrong_b_right": b, "c_only_a_right_b_wrong": c, "p_exact": round(p, 6)}


def holm(pvals):
    """Holm correction via statsmodels; keeps raw p (guide 3.4)."""
    from statsmodels.stats.multitest import multipletests
    reject, p_adj, _, _ = multipletests(np.asarray(pvals, dtype=float), alpha=0.05, method="holm")
    return [float(x) for x in p_adj], [bool(x) for x in reject]


def main() -> None:
    data = ablations.load_records()
    dev, test = data["dev"], data["test"]
    frozen = json.loads((OUT_ROOT / "frozen_config.json").read_text(encoding="utf-8")) if (OUT_ROOT / "frozen_config.json").exists() else {}
    frozen_threshold = float(frozen.get("threshold", 0.5))

    order = ["T0_rule", "T1_single_judge", "T2_fraud_only", "T3_fraud_refusal",
             "T4_fraud_refusal_context", "T5_rule_arbiter", "T6_evidence_arbiter", "T7_full_correction"]
    dev_settings = ablations.settings_with_judge(ablations.settings_from_records(dev, frozen_threshold), data["judge_dev"], frozen_threshold)
    test_settings = ablations.settings_with_judge(ablations.settings_from_records(test, frozen_threshold), data["judge_test"], frozen_threshold)
    thr_file = METRICS / "thresholds_table_a.json"
    thresholds = json.loads(thr_file.read_text(encoding="utf-8")) if thr_file.exists() else {n: frozen_threshold for n in order}
    for name in order:
        dev_settings[name]["label"] = lambda r, _n=name: ablations.label_from_score(dev_settings[_n]["score"](r), thresholds[_n])
        test_settings[name]["label"] = lambda r, _n=name: ablations.label_from_score(test_settings[_n]["score"](r), thresholds[_n])

    final = {
        "n_test": len(test),
        "n_dev": len(dev),
        "frozen_config": {"threshold": frozen_threshold, "calibration": frozen.get("calibration")},
        "protocol": {
            "table_a_thresholds": {k: round(float(v), 4) for k, v in thresholds.items()},
            "note": "Table A: per-method dev threshold (objective true macro-F1); "
                    "Tables B/C in nested_ablation_matched_fpr/recall.csv (guide 3.5)",
        },
        "methods": {},
        "comparisons": {},
        "generated_at": "2026-08-04",
    }
    gold = [1 if r["sample"]["gold_label"] == "unsafe" else 0 for r in test]
    for name in order:
        labels = [test_settings[name]["label"](r) for r in test]
        scores = [test_settings[name]["score"](r) for r in test]
        final["methods"][name] = overall_metrics(gold, labels, scores)

    primary_pairs = [
        ("T1_single_judge", "T6_evidence_arbiter"),   # main: MAT vs single judge
        ("T5_rule_arbiter", "T6_evidence_arbiter"),   # evidence arbiter value
        ("T6_evidence_arbiter", "T7_full_correction"),  # correction value
        ("T2_fraud_only", "T3_fraud_refusal"),        # refusal agent value
        ("T3_fraud_refusal", "T4_fraud_refusal_context"),  # context agent value
    ]
    pvals = []
    for name_a, name_b in primary_pairs:
        a, b = test_settings[name_a], test_settings[name_b]
        boot = bootstrap_ci(test, a, b)
        mcn = exact_mcnemar(test, a["label"], b["label"])
        final["comparisons"][f"{name_a}->{name_b}"] = {
            "delta_direction": f"metric({name_b}) - metric({name_a}); positive = {name_b} better",
            "bootstrap": boot,
            "mcnemar": mcn,
        }
        pvals.append(mcn["p_exact"])
        print(f"{name_a} -> {name_b}: macro_f1_delta={boot['macro_f1']['observed_delta']} "
              f"ci={boot['macro_f1']['ci95']} mcnemar_p={mcn['p_exact']}")

    p_adj, reject = holm(pvals)
    for (name_a, name_b), p_raw, p_holm, rej in zip(primary_pairs, pvals, p_adj, reject):
        final["comparisons"][f"{name_a}->{name_b}"]["holm"] = {
            "raw_p": round(p_raw, 6), "holm_p": round(p_holm, 6), "reject_h0": bool(rej)}

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "final_metrics.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    # backward-compatible view for legacy consumers (report currently cites it)
    legacy = {"n_test": final["n_test"], "threshold": frozen_threshold, "comparisons": final["comparisons"]}
    (METRICS / "paired_significance.json").write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", METRICS / "final_metrics.json")


if __name__ == "__main__":
    main()
