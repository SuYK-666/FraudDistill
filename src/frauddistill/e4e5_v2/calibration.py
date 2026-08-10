# -*- coding: utf-8 -*-
"""E5 calibration: temperature scaling, risk-constrained threshold (P1),
calibration metrics (ECE/Brier/NLL), low-label curves."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def clopper_pearson_ucb(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided upper confidence bound for Binomial proportion (Clopper-Pearson)."""
    if n == 0:
        return 1.0
    if k == 0:
        return 1 - alpha ** (1.0 / n)
    if k == n:
        return 1.0  # all negative examples flagged -> UCB saturates at 1
    # solve via beta quantile: UCB = Beta(1-alpha; k+1, n-k) quantile
    from scipy.stats import beta
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def fit_temperature(probs_safe: np.ndarray, y: np.ndarray, n_grid: int = 2001,
                    lo: float = 0.1, hi: float = 5.0) -> float:
    """Grid-search temperature minimizing NLL on calibration labels.
    probs_safe: P(safe) raw probabilities."""
    p = np.clip(np.asarray(probs_safe, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=int)  # 1 = unsafe
    logit = np.log(p / (1 - p))
    best_t, best_nll = 1.0, None
    for t in np.linspace(lo, hi, n_grid):
        pt = np.clip(1.0 / (1.0 + np.exp(-logit / t)), 1e-9, 1 - 1e-9)
        nll = -np.mean(y * np.log(1 - pt) + (1 - y) * np.log(pt))
        if not np.isfinite(nll):
            continue
        if best_nll is None or nll < best_nll:
            best_t, best_nll = t, float(nll)
    return float(best_t)


def risk_threshold(score: np.ndarray, y: np.ndarray, fpr_target: float = 0.05,
                   alpha: float = 0.05) -> dict:
    """Max recall subject to one-sided 95% Clopper-Pearson UCB(FPR)<=target.
    score: risk score (higher = more unsafe). y: 1=unsafe."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    cands = np.unique(np.concatenate([s, [0.0, 1.0]]))
    best = None
    for th in cands:
        pred = (s >= th).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fpr = fp / max(tn + fp, 1)
        ucb = clopper_pearson_ucb(fp, tn + fp, alpha)
        if ucb > fpr_target:
            continue
        recall = int(((pred == 1) & (y == 1)).sum()) / max(int((y == 1).sum()), 1)
        cand = {"threshold": float(th), "fpr": float(fpr), "fpr_ucb95": float(ucb),
                "recall": float(recall), "fp": fp, "n_negative": tn + fp}
        if best is None or cand["recall"] > best["recall"]:
            best = cand
    return best


def ece_score(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi) if hi == 1.0 else (p >= lo) & (p < hi)
        if not mask.any():
            continue
        out += (mask.sum() / len(y)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(out)


def mce_score(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi) if hi == 1.0 else (p >= lo) & (p < hi)
        if not mask.any():
            continue
        out = max(out, abs(float(p[mask].mean()) - float(y[mask].mean())))
    return float(out)


def calibration_metrics(probs_safe: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(np.asarray(probs_safe, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    brier = float(np.mean((p - (1 - y)) ** 2))
    nll = float(-np.mean(y * np.log(1 - p) + (1 - y) * np.log(p)))
    return {"brier": brier, "nll": nll, "ece": ece_score(p, y), "mce": mce_score(p, y),
            "n": int(len(y))}


def low_label_curve(cal_rows: list[dict], scores: np.ndarray, y: np.ndarray,
                    sizes=(50, 100, 200, 600), seeds=30, family_ids=None,
                    rng_seed: int = 20260809) -> dict:
    """Family-level stratified resampling of calibration labels; fit temperature
    + risk threshold; report mean/sd of Brier/ECE/FPR/Recall on the frozen test
    scores (evaluation happens outside; here only returns policy params)."""
    import random
    from collections import defaultdict
    fam_idx: dict = defaultdict(list)
    for i, f in enumerate(family_ids or [f"f{i}" for i in range(len(y))]):
        fam_idx[f].append(i)
    fams = list(fam_idx.keys())
    rng = random.Random(rng_seed)
    out = {}
    for n in sizes:
        rows_out = []
        for seed_i in range(seeds):
            rng2 = random.Random(rng_seed + seed_i * 7919)
            chosen = rng2.sample(fams, min(n, len(fams)))
            idx = [i for f in chosen for i in fam_idx[f]][:n]
            p_s = np.asarray([scores[i] for i in idx])
            # risk score is 1-P(safe); convert back to P(safe)
            p_safe = 1.0 - p_s
            t = fit_temperature(p_safe, y[idx])
            th = risk_threshold(p_s, y[idx], 0.05)
            rows_out.append({"seed": seed_i, "n_actual": len(idx), "temperature": t,
                             "threshold": th["threshold"] if th else None,
                             "cal_recall": th["recall"] if th else None,
                             "cal_fpr_ucb": th["fpr_ucb95"] if th else None})
        out[str(n)] = rows_out
    return out
