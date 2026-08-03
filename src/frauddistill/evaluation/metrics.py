"""Evaluation metrics for exp2 budgeted cascade (guide section 18).

Computes Accuracy / Precision / Recall / Macro-F1 / FPR / AUPRC / N+,
clustered (group) bootstrap CIs, confusion matrix and per-domain metrics.
"""
from __future__ import annotations

import random
from typing import Any, Iterable

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


def binary_metrics(golds: list[int], preds: list[int], scores: list[float] | None = None) -> dict[str, Any]:
    g = np.asarray(golds, dtype=int)
    p = np.asarray(preds, dtype=int)
    n = len(g)
    pos = int(g.sum())
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    tn = int(((p == 0) & (g == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    safe_prec = tn / max(tn + fn, 1)
    safe_rec = tn / max(tn + fp, 1)
    safe_f1 = 2 * safe_prec * safe_rec / max(safe_prec + safe_rec, 1e-12)
    macro_f1 = (f1 + safe_f1) / 2
    fpr = fp / max(fp + tn, 1)
    acc = (tp + tn) / max(n, 1)
    auprc = None
    if scores is not None and len(scores) == n and average_precision_score is not None:
        try:
            auprc = float(average_precision_score(g, scores))
        except Exception:
            auprc = None
    return {
        "n": n,
        "n_positive": pos,
        "n_negative": n - pos,
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "macro_f1": round(macro_f1, 6),
        "fpr": round(fpr, 6),
        "auprc": round(auprc, 6) if auprc is not None else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def group_bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    gold_key: str = "gold_binary",
    pred_key: str = "prediction_binary",
    group_key: str = "group_id",
    reps: int = 2000,
    seed: int = 20260803,
    metrics: tuple[str, ...] = ("macro_f1", "fpr", "recall"),
) -> dict[str, dict]:
    """Clustered bootstrap over groups; returns {metric: {ci_low, ci_high}}."""
    groups: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        gid = str(r.get(group_key) or r.get("id"))
        g = r.get(gold_key)
        p = r.get(pred_key)
        if g is None or p is None:
            continue
        groups.setdefault(gid, []).append((int(g), int(p)))
    gids = list(groups.keys())
    rng = random.Random(seed)
    out: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(reps):
        picked = [rng.choice(gids) for _ in range(len(gids))]
        golds, preds = [], []
        for gid in picked:
            for g, p in groups[gid]:
                golds.append(g)
                preds.append(p)
        m = binary_metrics(golds, preds)
        for name in metrics:
            out[name].append(m[name])
    result = {}
    for name, vals in out.items():
        vals = sorted(vals)
        result[name] = {
            "ci_low": round(vals[int(0.025 * len(vals))], 6),
            "ci_high": round(vals[int(0.975 * len(vals))], 6),
            "mean": round(float(np.mean(vals)), 6),
        }
    return result


def evaluate_predictions(rows: Iterable[dict], *, gold_key="gold_binary", pred_key="prediction_binary", score_key="risk_score") -> dict[str, Any]:
    rows = [r for r in rows if r.get(gold_key) is not None and r.get(pred_key) is not None]
    golds = [int(r[gold_key]) for r in rows]
    preds = [int(r[pred_key]) for r in rows]
    scores = [float(r.get(score_key) or 0.0) for r in rows] if score_key else None
    return binary_metrics(golds, preds, scores)


def subgroup_metrics(rows: list[dict], *, dims: tuple[str, ...] = ("category", "language"), gold_key="gold_binary", pred_key="prediction_binary") -> dict[str, list[dict]]:
    """Per-subgroup metrics for a set of prediction rows."""
    out: dict[str, list[dict]] = {}
    for dim in dims:
        buckets: dict[str, list[dict]] = {}
        for r in rows:
            if r.get(gold_key) is None or r.get(pred_key) is None:
                continue
            key = str(r.get(dim) or "unknown")
            buckets.setdefault(key, []).append(r)
        out[dim] = []
        for key, items in buckets.items():
            m = binary_metrics([int(r[gold_key]) for r in items], [int(r[pred_key]) for r in items])
            m[dim] = key
            m["n"] = len(items)
            out[dim].append(m)
        out[dim].sort(key=lambda x: -x["n"])
    return out
