from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np


MetricFn = Callable[[list[str], list[str]], float]


def holm_adjust(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool | int]]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted_values: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p_value) in enumerate(ordered, start=1):
        adjusted = min(1.0, (m - rank + 1) * float(p_value))
        running_max = max(running_max, adjusted)
        adjusted_values[name] = running_max
    return {
        name: {
            "raw_p": float(p_values[name]),
            "holm_p": float(adjusted_values[name]),
            "rank": next(rank for rank, (ordered_name, _) in enumerate(ordered, start=1) if ordered_name == name),
            "reject": bool(adjusted_values[name] < alpha),
        }
        for name in p_values
    }


def paired_cluster_permutation_delta(
    y_true: list[str],
    pred_a: list[str],
    pred_b: list[str],
    clusters: list[str],
    metric_fn: MetricFn,
    iterations: int = 10000,
    seed: int = 0,
) -> dict[str, float | int]:
    if not (len(y_true) == len(pred_a) == len(pred_b) == len(clusters)):
        raise ValueError("all inputs must have the same length")
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, cluster in enumerate(clusters):
        grouped[str(cluster)].append(idx)
    keys = sorted(grouped)
    observed = metric_fn(y_true, pred_b) - metric_fn(y_true, pred_a)
    rng = np.random.default_rng(seed)
    null = np.empty(iterations, dtype=float)
    for i in range(iterations):
        swapped_a = list(pred_a)
        swapped_b = list(pred_b)
        for key in keys:
            if rng.random() < 0.5:
                for idx in grouped[key]:
                    swapped_a[idx], swapped_b[idx] = swapped_b[idx], swapped_a[idx]
        null[i] = metric_fn(y_true, swapped_b) - metric_fn(y_true, swapped_a)
    if observed >= 0:
        p_value = (float(np.sum(null >= observed)) + 1.0) / (iterations + 1.0)
    else:
        p_value = (float(np.sum(null <= observed)) + 1.0) / (iterations + 1.0)
    return {
        "iterations": int(iterations),
        "cluster_count": int(len(keys)),
        "observed_delta": float(observed),
        "one_sided_p": float(p_value),
    }

