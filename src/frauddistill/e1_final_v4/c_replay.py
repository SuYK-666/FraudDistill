# -*- coding: utf-8 -*-
"""E1-C: independent natural low-prevalence replay on A7500 (zero API)."""
from __future__ import annotations

import collections
from typing import Any

import numpy as np

from frauddistill.e1_v10.metrics import auprc, auroc, binary_metrics, ece
from frauddistill.e1_final_v3.io import read_jsonl, write_json


def _recall_at_fpr(y: list[int], s: list[float], target_fpr: float) -> float:
    order = sorted(zip(s, y), key=lambda x: -x[0])
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 0.0
    tp = fp = 0
    best = 0.0
    for score, yy in order:
        if yy == 1:
            tp += 1
        else:
            fp += 1
        if fp / max(1, n_neg) <= target_fpr:
            best = max(best, tp / n_pos)
        else:
            break
    return best


def _precision_at_k(y: list[int], s: list[float], k: int) -> float:
    order = sorted(zip(s, y), key=lambda x: -x[0])
    head = order[:k]
    if not head:
        return 0.0
    return sum(1 for _, yy in head if yy) / len(head)


def c_replay(a_rows: list[dict[str, Any]], predict_fn, threshold: float, mode: str) -> dict[str, Any]:
    scores = predict_fn(a_rows)
    y = [int(r["gold_central"]) for r in a_rows]
    s = [float(x) for x in scores]
    preds = [1 if x >= threshold else 0 for x in s]
    evals = [{**r, "gold": g, "pred": p, "score": sc} for r, g, p, sc in zip(a_rows, y, preds, s)]
    m = binary_metrics(evals)
    m.update({
        "recall_at_fpr_1pct": _recall_at_fpr(y, s, 0.01),
        "recall_at_fpr_5pct": _recall_at_fpr(y, s, 0.05),
        "precision_at_10": _precision_at_k(y, s, 10),
        "precision_at_25": _precision_at_k(y, s, 25),
        "precision_at_50": _precision_at_k(y, s, 50),
        "precision_at_100": _precision_at_k(y, s, 100),
        "prevalence": round(sum(y) / len(y), 6),
        "auprc_lift": round(auprc(y, s) / max(1e-9, sum(y) / len(y)), 3),
    })
    return m


def c_report(cfg, out_dir, a_rows, m1_predictors) -> dict[str, Any]:
    """m1_predictors: dict mode -> list of (seed, predict_fn, threshold)."""
    results = {}
    for mode, preds in m1_predictors.items():
        per_seed = []
        for seed, fn, thr in preds:
            per_seed.append({"seed": seed, "threshold": thr, "metrics": c_replay(a_rows, fn, thr, mode)})
        results[mode] = per_seed
    # aggregate mean/sd over seeds for headline metrics
    agg = {}
    for mode, per_seed in results.items():
        keys = ["macro_f1", "recall", "fpr", "auroc", "auprc", "precision", "recall_at_fpr_1pct", "recall_at_fpr_5pct", "precision_at_10", "precision_at_25", "precision_at_50", "precision_at_100"]
        agg[mode] = {k: {"mean": round(float(np.mean([s["metrics"][k] for s in per_seed])), 6), "sd": round(float(np.std([s["metrics"][k] for s in per_seed])), 6)} for k in keys}
    write_json(out_dir / "E1_V4_C_RESULT.json", {"per_seed": results, "aggregate": agg})
    return {"per_seed": results, "aggregate": agg}
