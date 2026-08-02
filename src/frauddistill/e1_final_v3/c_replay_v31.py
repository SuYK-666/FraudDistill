from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from frauddistill.e1_v10.metrics import auprc, binary_metrics, ece, groupby


def recall_at_fpr(labels: list[int], scores: np.ndarray, targets: list[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(set(y.tolist())) < 2 or len(y) == 0:
        return {f"recall_at_fpr_{int(t * 100)}": 0.0 for t in targets}
    fpr, tpr, _ = roc_curve(y, s)
    out = {}
    for target in targets:
        idx = np.argmax(fpr >= target) if np.any(fpr >= target) else len(fpr) - 1
        out[f"recall_at_fpr_{int(round(target * 100))}pct"] = float(tpr[idx])
    return out


def precision_at_budget(labels: list[int], scores: np.ndarray, budgets: list[int]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    out = {}
    for b in budgets:
        top = order[:b]
        out[f"precision_at_budget_{b}"] = float(np.mean(y[top])) if len(top) else 0.0
    return out


def c_block(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    valid = [r for r in rows if int(r.get("gold_central", -1)) >= 0]
    labels = [int(r["gold_central"]) for r in valid]
    scores = np.asarray([scores[i] for i in range(len(rows)) if int(rows[i].get("gold_central", -1)) >= 0], dtype=float)
    preds = (scores >= threshold).astype(int)
    evals = [{**r, "gold": int(r["gold_central"]), "pred": int(p), "score": float(s)} for r, p, s in zip(valid, preds, scores)]
    m = binary_metrics(evals)
    m.update(
        {
            "auprc": auprc(labels, scores),
            "auroc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0,
            "brier": float(np.mean((np.asarray(labels, dtype=float) - scores) ** 2)),
            "ece": ece(labels, scores.tolist()),
            **recall_at_fpr(labels, scores, [0.01, 0.05, 0.10]),
            **precision_at_budget(labels, scores, [10, 25, 50, 100, 200]),
            "threshold": threshold,
        }
    )
    return m


def directional(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for r, s in zip(rows, scores):
        if int(r.get("gold_central", -1)) < 0:
            continue
        by_key[(str(r.get("target_provider", "")), str(r.get("scenario", "")), str(r.get("language", "")), str(r.get("fraud_category", "")))].append((r, float(s)))
    out = []
    for key, group in sorted(by_key.items()):
        sub_rows = [r for r, _ in group]
        sub_scores = np.asarray([s for _, s in group])
        block = c_block(sub_rows, sub_scores, threshold)
        out.append({"target_model": key[0], "setting": key[1], "language": key[2], "fraud_category": key[3], **block})
    return out


def paired_bootstrap_gain(
    rows: list[dict[str, Any]],
    scores_qy: np.ndarray,
    scores_y: np.ndarray,
    cluster_key: str = "canonical_case_id",
    iterations: int = 2000,
    seed: int = 20260802,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    clusters = list(groupby(rows, cluster_key).values())
    if not clusters:
        return {"gain_point": 0.0, "low": 0.0, "high": 0.0}

    def auprc_of(sub_rows: list[dict[str, Any]], scores: np.ndarray) -> float:
        labels = [int(r["gold_central"]) for r in sub_rows]
        return auprc(labels, scores)

    index = {id(r): i for i, r in enumerate(rows)}
    gains = []
    for _ in range(max(1, iterations)):
        ids = [index[id(r)] for c in (rng.choice(clusters) for _ in clusters) for r in c]
        gains.append(auprc_of([rows[i] for i in ids], scores_qy[ids]) - auprc_of([rows[i] for i in ids], scores_y[ids]))
    gains.sort()
    point = auprc_of(rows, scores_qy) - auprc_of(rows, scores_y)
    return {"gain_point": point, "low": gains[int(0.025 * (len(gains) - 1))], "high": gains[int(0.975 * (len(gains) - 1))]}
