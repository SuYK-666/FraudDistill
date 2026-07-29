from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: list[str], y_pred: list[str], y_score: list[float] | None = None) -> dict[str, float]:
    true = np.array([1 if y == "unsafe" else 0 for y in y_true])
    pred = np.array([1 if y == "unsafe" else 0 for y in y_pred])
    result = {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(0.5 * (
            ((pred == 1) & (true == 1)).sum() / max((true == 1).sum(), 1)
            + ((pred == 0) & (true == 0)).sum() / max((true == 0).sum(), 1)
        )),
        "specificity": float(((pred == 0) & (true == 0)).sum() / max((true == 0).sum(), 1)),
        "mcc": float(matthews_corrcoef(true, pred)) if len(set(true.tolist())) == 2 else 0.0,
        "fpr": float(((pred == 1) & (true == 0)).sum() / max((true == 0).sum(), 1)),
        "fnr": float(((pred == 0) & (true == 1)).sum() / max((true == 1).sum(), 1)),
    }
    if y_score is not None:
        score = np.asarray(y_score, dtype=float)
        score = np.clip(score, 1e-6, 1.0 - 1e-6)
        result["brier"] = float(brier_score_loss(true, score))
        result["ece"] = float(expected_calibration_error(true, score))
        result["nll"] = float(log_loss(true, score, labels=[0, 1]))
        if len(set(true.tolist())) == 2:
            result["auroc"] = float(roc_auc_score(true, score))
            result["auprc"] = float(average_precision_score(true, score))
    return result


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    if total == 0:
        return 0.0
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        if not mask.any():
            continue
        confidence = float(np.mean(y_score[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += (float(mask.sum()) / total) * abs(accuracy - confidence)
    return ece
