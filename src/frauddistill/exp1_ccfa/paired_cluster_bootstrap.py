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
    cluster_counts = np.asarray([_cluster_counts([y_true[idx] for idx in grouped[key]], [pred_a[idx] for idx in grouped[key]], [pred_b[idx] for idx in grouped[key]]) for key in keys], dtype=np.int64)
    deltas = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        sampled = rng.integers(0, len(keys), size=len(keys))
        counts = cluster_counts[sampled].sum(axis=0)
        deltas[i] = _macro_f1_from_counts(counts[4:8]) - _macro_f1_from_counts(counts[0:4])
    return {
        "iterations": int(iterations),
        "cluster_count": int(len(keys)),
        "delta_mean": float(np.mean(deltas)),
        "ci_lower": float(np.quantile(deltas, 0.025)),
        "ci_upper": float(np.quantile(deltas, 0.975)),
    }


def _cluster_counts(gold: list[str], pred_a: list[str], pred_b: list[str]) -> list[int]:
    return [*_confusion_counts(gold, pred_a), *_confusion_counts(gold, pred_b)]


def _confusion_counts(gold: list[str], pred: list[str]) -> list[int]:
    tp = tn = fp = fn = 0
    for y, p in zip(gold, pred):
        y_pos = y == "unsafe" or y == 1
        p_pos = p == "unsafe" or p == 1
        if y_pos and p_pos:
            tp += 1
        elif (not y_pos) and (not p_pos):
            tn += 1
        elif (not y_pos) and p_pos:
            fp += 1
        else:
            fn += 1
    return [tp, tn, fp, fn]


def _macro_f1_from_counts(counts: np.ndarray) -> float:
    tp, tn, fp, fn = [float(x) for x in counts]
    pos_den = (2.0 * tp) + fp + fn
    neg_den = (2.0 * tn) + fp + fn
    pos_f1 = 0.0 if pos_den <= 0 else (2.0 * tp) / pos_den
    neg_f1 = 0.0 if neg_den <= 0 else (2.0 * tn) / neg_den
    return (pos_f1 + neg_f1) / 2.0
