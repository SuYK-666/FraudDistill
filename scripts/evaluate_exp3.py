# -*- coding: utf-8 -*-
"""Paired cluster bootstrap (10k), exact McNemar and Holm correction (guide 20).

Usage: python scripts/evaluate_exp3.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score

import build_exp3_agent_ablations as ablations

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("EXP3_OUT_ROOT") or REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs")
METRICS = OUT_ROOT / "metrics"
N_REPS = 10000


def per_group(recs, get_label, get_score):
    groups: dict[str, list] = defaultdict(list)
    for r in recs:
        groups[str(r["group_id"])].append(r)
    out = []
    for gid, gr in groups.items():
        y = np.array([1 if r["sample"]["gold_label"] == "unsafe" else 0 for r in gr])
        pred = np.array([1 if get_label(r) == "unsafe" else 0 for r in gr])
        scores = np.array([float(get_score(r)) for r in gr])
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        fpr = fp / max(tn + fp, 1)
        out.append({"y_sum": int(y.sum()), "n": len(gr), "f1": f1, "rec": rec, "fpr": fpr,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return out


def weighted_metric(groups, key, weight_key="n"):
    usable = [g for g in groups if not (isinstance(g[key], float) and g[key] != g[key])]
    num = sum(g[key] * g[weight_key] for g in usable)
    den = sum(g[weight_key] for g in usable)
    return num / max(den, 1e-9)


def bootstrap_ci(recs, get_label_a, get_score_a, get_label_b, get_score_b, n_reps=10000, seed=20260804):
    rng = np.random.default_rng(seed)
    ga = per_group(recs, get_label_a, get_score_a)
    gb = per_group(recs, get_label_b, get_score_b)
    n = len(ga)
    idx = rng.integers(0, n, size=(n_reps, n))
    res = {"macro_f1": [], "recall": [], "fpr": []}
    for row in idx:
        sa = [ga[i] for i in row]
        sb = [gb[i] for i in row]
        res["macro_f1"].append(weighted_metric(sb, "f1") - weighted_metric(sa, "f1"))
        res["recall"].append(weighted_metric(sb, "rec") - weighted_metric(sa, "rec"))
        res["fpr"].append(weighted_metric(sb, "fpr") - weighted_metric(sa, "fpr"))
    out = {}
    for k, v in res.items():
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = {"delta": round(float(np.mean(v)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "ci_excludes_zero": bool((lo > 0) or (hi < 0))}
    return out


def exact_mcnemar(recs, get_label_a, get_label_b):
    b = c = 0
    for r in recs:
        pa = get_label_a(r)
        pb = get_label_b(r)
        gold = r["sample"]["gold_label"]
        a_ok = pa == gold
        b_ok = pb == gold
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
    p = float(binomtest(b, b + c).pvalue) if (b + c) > 0 else 1.0
    return {"b_only": b, "c_only": c, "p_exact": round(p, 6)}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(order, 1):
        adj = min(1.0, max(pvals[i] * (m - rank + 1), prev))
        adjusted[i] = adj
        prev = adj
    return adjusted


def main() -> None:
    data = ablations.load_records()
    test = data["test"]
    frozen = json.loads((OUT_ROOT / "frozen_config.json").read_text(encoding="utf-8")) if (OUT_ROOT / "frozen_config.json").exists() else {}
    threshold = float(frozen.get("threshold", 0.5))
    settings = ablations.settings_with_judge(ablations.settings_from_records(test, threshold), data["judge_test"], threshold)

    primary_pairs = [
        ("T1_single_judge", "T7_full_correction"),   # main: MAT vs single judge
        ("T5_rule_arbiter", "T6_evidence_arbiter"),  # evidence arbiter value
        ("T6_evidence_arbiter", "T7_full_correction"),  # correction value
        ("T2_fraud_only", "T3_fraud_refusal"),       # refusal agent value
        ("T3_fraud_refusal", "T4_fraud_refusal_context"),  # context agent value
    ]
    def pooled_auprc(recs, get_score):
        y = np.array([1 if r["sample"]["gold_label"] == "unsafe" else 0 for r in recs])
        scores = np.array([float(get_score(r)) for r in recs])
        if len(set(scores)) > 1 and 0 < y.sum() < len(y):
            return round(float(average_precision_score(y, scores)), 4)
        return None

    auprc_points = {name: pooled_auprc(test, s["score"]) for name, s in settings.items()}
    results = {"n_test": len(test), "threshold": threshold, "auprc_pooled": auprc_points, "comparisons": {}}
    pvals = []
    for name_a, name_b in primary_pairs:
        a, b = settings[name_a], settings[name_b]
        boot = bootstrap_ci(test, a["label"], a["score"], b["label"], b["score"], n_reps=N_REPS)
        mcn = exact_mcnemar(test, a["label"], b["label"])
        results["comparisons"][f"{name_a}->{name_b}"] = {"bootstrap": boot, "mcnemar": mcn}
        pvals.append(mcn["p_exact"])
        print(f"{name_a} -> {name_b}: {json.dumps(boot, ensure_ascii=False)} mcnemar_p={mcn['p_exact']}")

    adj = holm(pvals)
    for (name_a, name_b), p_adj in zip(primary_pairs, adj):
        results["comparisons"][f"{name_a}->{name_b}"]["mcnemar_p_holm"] = round(p_adj, 6)

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "paired_significance.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", METRICS / "paired_significance.json")


if __name__ == "__main__":
    main()