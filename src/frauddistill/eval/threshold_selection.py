from __future__ import annotations

import numpy as np


def select_qy_threshold_with_ablation_constraints(
    labels: list[str],
    qy_scores: list[float],
    y_only_scores: list[float],
    y_only_threshold: float,
    *,
    min_macro_gain: float = 0.020,
    min_recall_gain: float = 0.030,
    max_fpr_increase: float = 0.020,
) -> dict[str, float | bool]:
    """Choose a q+y operating point on dev only, relative to y-only dev metrics."""
    y = np.asarray([label == "unsafe" for label in labels], dtype=bool)
    qy = np.asarray(qy_scores, dtype=float)
    y_only = np.asarray(y_only_scores, dtype=float)
    baseline = _binary_summary(y, y_only >= y_only_threshold)
    candidates = []
    for threshold in np.unique(qy):
        summary = _binary_summary(y, qy >= threshold)
        feasible = (
            summary["macro_f1"] >= baseline["macro_f1"] + min_macro_gain
            and summary["recall"] >= baseline["recall"] + min_recall_gain
            and summary["fpr"] <= baseline["fpr"] + max_fpr_increase
        )
        candidates.append((feasible, summary["recall"], summary["macro_f1"], -summary["fpr"], float(threshold), summary))
    feasible = [item for item in candidates if item[0]]
    if feasible:
        chosen = max(feasible, key=lambda item: (item[1], item[2], item[3]))
    else:
        chosen = max(candidates, key=lambda item: (item[2], item[1], item[3]))
    _, _, _, _, threshold, summary = chosen
    return {"threshold": threshold, "constraint_satisfied_on_dev": bool(feasible), **{f"dev_qy_{key}": value for key, value in summary.items()}, **{f"dev_y_only_{key}": value for key, value in baseline.items()}}


def _binary_summary(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1_unsafe = 2 * precision * recall / max(precision + recall, 1e-12)
    safe_precision = tn / max(tn + fn, 1)
    safe_recall = tn / max(tn + fp, 1)
    f1_safe = 2 * safe_precision * safe_recall / max(safe_precision + safe_recall, 1e-12)
    return {"recall": recall, "fpr": fp / max(fp + tn, 1), "macro_f1": (f1_unsafe + f1_safe) / 2}


def select_fpr_constrained_threshold(labels: list[str], scores: list[float], max_fpr: float = 0.05) -> dict[str, float]:
    y = np.asarray([label == "unsafe" for label in labels], dtype=bool)
    s = np.asarray(scores, dtype=float)
    candidates = np.unique(np.r_[0.0, s, 1.0])
    choices = []
    for threshold in candidates:
        pred = s >= threshold
        fpr = float((pred & ~y).sum() / max((~y).sum(), 1))
        recall = float((pred & y).sum() / max(y.sum(), 1))
        if fpr <= max_fpr:
            choices.append((recall, -fpr, float(threshold)))
    if not choices:
        return {"threshold": 1.0, "dev_fpr": 0.0, "dev_recall": 0.0, "max_fpr": max_fpr}
    recall, neg_fpr, threshold = max(choices)
    return {"threshold": threshold, "dev_fpr": -neg_fpr, "dev_recall": recall, "max_fpr": max_fpr}
