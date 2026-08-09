# -*- coding: utf-8 -*-
"""Family-cluster paired bootstrap + exact McNemar + Holm correction."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .metrics import binary_metrics


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


def paired_cluster_bootstrap(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                             thresholds: tuple[float, float], family_ids: list[str],
                             replicates: int = 10000, seed: int = 20260808,
                             metric: str = "macro_f1") -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    th_a, th_b = thresholds
    fams = [str(f) for f in family_ids]
    diffs = []
    for _ in range(replicates):
        idx = family_cluster_indices(fams, rng)
        pa = (sa[idx] >= th_a).astype(int)
        pb = (sb[idx] >= th_b).astype(int)
        ma = binary_metrics(y[idx], sa[idx], pred=pa, threshold=th_a)
        mb = binary_metrics(y[idx], sb[idx], pred=pb, threshold=th_b)
        diffs.append(ma[metric] - mb[metric])
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "metric": metric,
        "replicates": replicates,
        "mean_diff": round(float(diffs.mean()), 5),
        "ci95": [round(float(lo), 5), round(float(hi), 5)],
        "ci95_above_zero": lo > 0,
        "p_value_approx": round(float((diffs <= 0).mean()), 5),
    }


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
    p = 0.0
    for k in range(min(b, c), n + 1):
        p += comb(n, k) * 0.5 ** n
    p = min(1.0, 2 * p)  # two-sided
    return {"b": b, "c": c, "n_disagreements": n, "p_exact": round(p, 6),
            "direction": "A_better" if b < c else ("B_better" if c < b else "tie")}


def holm_correct(pvalues: list[float], names: list[str]) -> dict:
    """Holm-Bonferroni correction; returns per-name adjusted p."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 1.0
    for rank, i in enumerate(order):
        adj = pvalues[i] * (m - rank)
        running = min(running, adj)
        adjusted[i] = min(running, 1.0)
    return {names[i]: {"p_raw": round(pvalues[i], 6), "p_holm": round(adjusted[i], 6)} for i in range(m)}


def run_paired_statistics(rows: list[dict], pred_maps: dict[str, dict], thresholds: dict[str, float],
                          comparisons: list[tuple[str, str]], family_key: str = "family_id",
                          replicates: int = 10000, seed: int = 20260808) -> dict:
    y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows], dtype=int)
    fams = [str(r.get(family_key) or r["id"]) for r in rows]
    out = {}
    for name_a, name_b in comparisons:
        sa = np.array([pred_maps[name_a][r["id"]]["risk_score"] for r in rows], dtype=float)
        sb = np.array([pred_maps[name_b][r["id"]]["risk_score"] for r in rows], dtype=float)
        pa = (sa >= thresholds[name_a]).astype(int)
        pb = (sb >= thresholds[name_b]).astype(int)
        ci_mf1 = paired_cluster_bootstrap(y, sa, sb, (thresholds[name_a], thresholds[name_b]), fams, replicates, seed, "macro_f1")
        ci_rec = paired_cluster_bootstrap(y, sa, sb, (thresholds[name_a], thresholds[name_b]), fams, replicates, seed, "recall")
        ci_fpr = paired_cluster_bootstrap(y, sa, sb, (thresholds[name_a], thresholds[name_b]), fams, replicates, seed, "fpr")
        mc = exact_mcnemar(pa, pb, y)
        out[f"{name_a}_vs_{name_b}"] = {
            "bootstrap_macro_f1": ci_mf1,
            "bootstrap_recall": ci_rec,
            "bootstrap_fpr": ci_fpr,
            "mcnemar": mc,
            "n": len(rows),
        }
    # Holm across main comparisons
    pvals, names = [], []
    for k, v in out.items():
        pvals.append(v["mcnemar"]["p_exact"])
        names.append(k)
    out["_holm"] = holm_correct(pvals, names)
    return out


def write_paired_statistics(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
