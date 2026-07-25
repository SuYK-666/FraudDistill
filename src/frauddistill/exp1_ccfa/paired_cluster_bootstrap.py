from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np


MetricFn = Callable[[list[str], list[str]], float]


def paired_cluster_bootstrap_delta(
    y_true: list[str],
    pred_a: list[str],
    pred_b: list[str],
    clusters: list[str],
    metric_fn: MetricFn,
    iterations: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    if not (len(y_true) == len(pred_a) == len(pred_b) == len(clusters)):
        raise ValueError("all inputs must have the same length")
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, cluster in enumerate(clusters):
        grouped[str(cluster)].append(idx)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        indices = [idx for key in sampled_keys for idx in grouped[str(key)]]
        gold = [y_true[idx] for idx in indices]
        a = [pred_a[idx] for idx in indices]
        b = [pred_b[idx] for idx in indices]
        deltas[i] = metric_fn(gold, b) - metric_fn(gold, a)
    return {
        "iterations": int(iterations),
        "cluster_count": int(len(keys)),
        "delta_mean": float(np.mean(deltas)),
        "ci_lower": float(np.quantile(deltas, 0.025)),
        "ci_upper": float(np.quantile(deltas, 0.975)),
    }

