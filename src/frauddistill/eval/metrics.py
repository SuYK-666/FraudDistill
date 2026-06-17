from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y_true: list[str], y_pred: list[str], y_score: list[float] | None = None) -> dict[str, float]:
    true = np.array([1 if y == "unsafe" else 0 for y in y_true])
    pred = np.array([1 if y == "unsafe" else 0 for y in y_pred])
    result = {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "fpr": float(((pred == 1) & (true == 0)).sum() / max((true == 0).sum(), 1)),
        "fnr": float(((pred == 0) & (true == 1)).sum() / max((true == 1).sum(), 1)),
    }
    if y_score is not None and len(set(true.tolist())) == 2:
        result["auroc"] = float(roc_auc_score(true, y_score))
        result["auprc"] = float(average_precision_score(true, y_score))
    return result
