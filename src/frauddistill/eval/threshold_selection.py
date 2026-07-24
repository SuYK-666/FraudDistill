from __future__ import annotations

import numpy as np


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
