# -*- coding: utf-8 -*-
"""Family-cluster paired bootstrap + exact McNemar + Holm correction."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from .metrics import binary_metrics_raw


def family_cluster_indices(family_ids: list[str], rng: np.random.Generator, n_families: int | None = None) -> np.ndarray:
    """Sample families with replacement, return row indices."""
    fam_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(family_ids):
        fam_to_idx[f or f"__no_fam_{i}"].append(i)
    fams = list(fam_to_idx.keys())
    n = n_families or len(fams)
    chosen = rng.choice(fams, size=n, replace=True)
    idx = []
    for f in chosen:
        idx.extend(fam_to_idx[f])
    return np.array(idx, dtype=int)


def _fast_family_sampler(family_codes: np.ndarray, n_families: int):
    """Precompute per-family row index arrays for fast cluster resampling."""
    fam_rows: list[np.ndarray] = []
    for f in range(n_families):
        fam_rows.append(np.flatnonzero(family_codes == f))
    return fam_rows


def _metrics_from_confusion(tp, fp, fn, tn, n) -> dict:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(tn + fp, 1)
    spec = tn / max(tn + fp, 1)
    f1u = 2 * tp / max(2 * tp + fp + fn, 1)
    f1s = 2 * tn / max(2 * tn + fp + fn, 1)
    mf1 = (f1u + f1s) / 2.0
    mcc = 0.0
    if n > 1:
        denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
        mcc = (tp * tn - fp * fn) / denom
    return {"f1_unsafe": f1u, "macro_f1": mf1, "recall": rec, "fpr": fpr, "mcc": mcc, "precision": prec}


def paired_cluster_bootstrap(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                             thresholds: tuple[float, float], family_ids: list[str],
                             replicates: int = 10000, seed: int = 20260808,
                             metric: str | None = None, metrics: list[str] | None = None,
                             pred_a: np.ndarray | None = None, pred_b: np.ndarray | None = None,
                             include_auroc_auprc: bool = False) -> dict:
    """Paired family-cluster bootstrap (vectorized family sampling).

    All requested metrics are computed from the SAME resampled rows, so a single
    10,000-replicate pass covers every metric. Metric diffs are new - baseline
    (scores_a vs scores_b).
    """
    if metric is not None:
        metrics = [metric]
    if metrics is None:
        metrics = ["macro_f1", "recall", "fpr", "mcc"]
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    th_a, th_b = thresholds
    pa = (sa >= th_a).astype(int) if pred_a is None else np.asarray(pred_a, dtype=int)
    pb = (sb >= th_b).astype(int) if pred_b is None else np.asarray(pred_b, dtype=int)
    fams = [str(f) for f in family_ids]
    fam_to_code = {}
    codes = np.empty(len(fams), dtype=int)
    for i, f in enumerate(fams):
        c = fam_to_code.setdefault(f, len(fam_to_code))
        codes[i] = c
    n_fam = len(fam_to_code)
    fam_rows = _fast_family_sampler(codes, n_fam)
    need_roc = include_auroc_auprc or "auroc" in metrics or "auprc" in metrics

    acc = {m: [] for m in metrics}
    if need_roc:
        acc.setdefault("auroc", [])
        acc.setdefault("auprc", [])
    for _ in range(replicates):
        chosen = rng.integers(0, n_fam, size=n_fam)
        parts = []
        for c in chosen:
            parts.append(fam_rows[c])
        idx = np.concatenate(parts)
        ya = y[idx]
        p_a = pa[idx]
        p_b = pb[idx]
        n = int(len(idx))
        tp = int(((p_a == 1) & (ya == 1)).sum()); fp = int(((p_a == 1) & (ya == 0)).sum())
        fn = int(((p_a == 0) & (ya == 1)).sum()); tn = int(((p_a == 0) & (ya == 0)).sum())
        ma = _metrics_from_confusion(tp, fp, fn, tn, n)
        tp = int(((p_b == 1) & (ya == 1)).sum()); fp = int(((p_b == 1) & (ya == 0)).sum())
        fn = int(((p_b == 0) & (ya == 1)).sum()); tn = int(((p_b == 0) & (ya == 0)).sum())
        mb = _metrics_from_confusion(tp, fp, fn, tn, n)
        for m in metrics:
            acc[m].append(ma[m] - mb[m])
        if need_roc:
            npos = int(ya.sum()); nneg = n - npos
            if npos > 0 and nneg > 0:
                acc["auroc"].append(roc_auc_score(ya, sa[idx]) - roc_auc_score(ya, sb[idx]))
                acc["auprc"].append(average_precision_score(ya, sa[idx]) - average_precision_score(ya, sb[idx]))
            else:
                acc["auroc"].append(0.0); acc["auprc"].append(0.0)

    out = {"replicates": replicates, "n": len(y)}
    for m, vals in acc.items():
        diffs = np.asarray(vals)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pval = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
        out[m] = {
            "metric": m,
            "mean_diff": float(diffs.mean()),
            "ci95": [float(lo), float(hi)],
            "ci95_above_zero": bool(lo > 0),
            "p_value_approx": pval,
        }
    if metric is not None:
        return out[metric]
    return out


def exact_mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray | None = None) -> dict:
    """Exact McNemar on the disagreement cells (b = A wrong/B right; c = A right/B wrong)."""
    pa = np.asarray(pred_a, dtype=int)
    pb = np.asarray(pred_b, dtype=int)
    b = int(((pa == 0) & (pb == 1)).sum()) if y is None else int(((pa != y) & (pb == y)).sum())
    c = int(((pa == 1) & (pb == 0)).sum()) if y is None else int(((pa == y) & (pb != y)).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_exact": 1.0, "direction": "tie"}
    from math import comb
    k = min(b, c)
    p = 0.0
    for i in range(k + 1):
        p += comb(n, i) * 0.5 ** n
    p = min(1.0, 2 * p)  # two-sided
    return {"b": b, "c": c, "n_disagreements": n, "p_exact": float(p),
            "direction": "A_better" if b < c else ("B_better" if c < b else "tie")}


def holm_correct(pvalues: list[float], names: list[str]) -> dict:
    """Holm-Bonferroni correction (step-down, cumulative MAX); returns per-name adjusted p."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = pvalues[i] * (m - rank)
        running = max(running, adj)
        adjusted[i] = min(running, 1.0)
    return {names[i]: {"p_raw": pvalues[i], "p_holm": adjusted[i]} for i in range(m)}


def run_paired_statistics(rows: list[dict], pred_maps: dict[str, dict], thresholds: dict[str, float],
                          comparisons: list[tuple[str, str]], family_key: str = "family_id",
                          replicates: int = 10000, seed: int = 20260808,
                          metrics: list[str] | None = None) -> dict:
    y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows], dtype=int)
    fams = [str(r.get(family_key) or r["id"]) for r in rows]
    out = {}
    for name_a, name_b in comparisons:
        sa = np.array([pred_maps[name_a][r["id"]]["risk_score"] for r in rows], dtype=float)
        sb = np.array([pred_maps[name_b][r["id"]]["risk_score"] for r in rows], dtype=float)
        pa = (sa >= thresholds[name_a]).astype(int)
        pb = (sb >= thresholds[name_b]).astype(int)
        boot = paired_cluster_bootstrap(y, sa, sb, (thresholds[name_a], thresholds[name_b]), fams,
                                        replicates, seed, metrics=metrics,
                                        include_auroc_auprc=True)
        mc = exact_mcnemar(pa, pb, y)
        out[f"{name_a}_vs_{name_b}"] = {
            "bootstrap": boot,
            "mcnemar": mc,
            "n": len(rows),
        }
    # Holm across main comparisons (McNemar p values)
    pvals, names = [], []
    for k, v in out.items():
        pvals.append(v["mcnemar"]["p_exact"])
        names.append(k)
    out["_holm"] = holm_correct(pvals, names)
    return out


def write_paired_statistics(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

