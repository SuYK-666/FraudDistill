from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from frauddistill.eval.metrics import binary_metrics
from frauddistill.eval.paired_compare import mcnemar_exact


def holm(pairs: list[dict]) -> list[dict]:
    ordered = sorted(enumerate(pairs), key=lambda item: item[1]["p_value"])
    adjusted = [None] * len(pairs)
    m = len(pairs)
    running = 0.0
    for rank, (idx, row) in enumerate(ordered):
        value = min(1.0, (m - rank) * float(row["p_value"]))
        running = max(running, value)
        adjusted[idx] = {**row, "holm_p": running}
    return adjusted


def paired_stats(rows: list[dict], predictions: dict[str, list[dict]], iterations: int = 1000, seed: int = 1) -> dict:
    y_true = [row["gold_label"] for row in rows]
    by_mode = {mode: {row["id"]: row for row in pred_rows} for mode, pred_rows in predictions.items()}
    metrics = {}
    for mode in predictions:
        preds = [by_mode[mode][row["id"]]["pred_label"] for row in rows]
        scores = [float(by_mode[mode][row["id"]]["pred_score"]) for row in rows]
        metrics[mode] = binary_metrics(y_true, preds, scores)
    comparisons = []
    for left, right in (("q_only", "y_only"), ("y_only", "q_y"), ("q_only", "q_y")):
        lpred = [by_mode[left][row["id"]]["pred_label"] for row in rows]
        rpred = [by_mode[right][row["id"]]["pred_label"] for row in rows]
        mc = mcnemar_exact(y_true, lpred, rpred)
        delta = metrics[right]["macro_f1"] - metrics[left]["macro_f1"]
        comparisons.append({"comparison": f"{right}-{left}", "delta_macro_f1": delta, **mc})
    return {"metrics": metrics, "comparisons": holm(comparisons), "bootstrap": cluster_bootstrap(rows, predictions, iterations, seed)}


def cluster_bootstrap(rows: list[dict], predictions: dict[str, list[dict]], iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    clusters = defaultdict(list)
    for row in rows:
        clusters[row.get("canonical_prompt_cluster", row["id"])].append(row)
    keys = list(clusters)
    by_mode = {mode: {row["id"]: row for row in pred_rows} for mode, pred_rows in predictions.items()}
    values = defaultdict(list)
    for _ in range(iterations):
        sampled = [item for key in rng.choice(keys, size=len(keys), replace=True) for item in clusters[key]]
        y_true = [row["gold_label"] for row in sampled]
        for mode in predictions:
            pred = [by_mode[mode][row["id"]]["pred_label"] for row in sampled]
            score = [float(by_mode[mode][row["id"]]["pred_score"]) for row in sampled]
            values[f"{mode}.macro_f1"].append(binary_metrics(y_true, pred, score)["macro_f1"])
        values["delta.q_y-y_only"].append(values["q_y.macro_f1"][-1] - values["y_only.macro_f1"][-1])
        values["delta.y_only-q_only"].append(values["y_only.macro_f1"][-1] - values["q_only.macro_f1"][-1])
    return {key: ci(vals) for key, vals in values.items()}


def ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "low": float(np.percentile(arr, 2.5)), "high": float(np.percentile(arr, 97.5))}


def decision_from_stats(stats: dict, gates: dict) -> dict:
    m = stats["metrics"]
    delta_qy_y = m["q_y"]["macro_f1"] - m["y_only"]["macro_f1"]
    delta_y_q = m["y_only"]["macro_f1"] - m["q_only"]["macro_f1"]
    checks = {
        "q_y_macro_f1": m["q_y"]["macro_f1"] >= float(gates["q_y_macro_f1_min"]),
        "y_only_macro_f1": m["y_only"]["macro_f1"] >= float(gates["y_only_macro_f1_min"]),
        "q_only_macro_f1": m["q_only"]["macro_f1"] <= float(gates["q_only_macro_f1_max"]),
        "q_y_minus_y": delta_qy_y >= float(gates["q_y_minus_y_min"]),
        "y_minus_q": delta_y_q >= float(gates["y_minus_q_min"]),
        "q_y_minus_y_ci_lower": stats["bootstrap"]["delta.q_y-y_only"]["low"] > float(gates["q_y_minus_y_ci_lower_min"]),
        "q_y_recall": m["q_y"]["recall"] >= float(gates["q_y_recall_min"]),
        "q_y_precision": m["q_y"]["precision"] >= float(gates["q_y_precision_min"]),
        "q_y_fpr": m["q_y"]["fpr"] <= float(gates["q_y_fpr_max"]),
        "q_y_auprc": m["q_y"].get("auprc", 0.0) >= float(gates["q_y_auprc_min"]),
    }
    if all(checks.values()):
        decision = "E1_FULL_PASS"
    elif m["q_only"]["macro_f1"] < m["y_only"]["macro_f1"] < m["q_y"]["macro_f1"] and delta_qy_y >= 0.015:
        decision = "E1_WEAK_PASS"
    else:
        decision = "E1_STOP"
    return {"decision": decision, "checks": checks, "delta_q_y_minus_y": delta_qy_y, "delta_y_minus_q": delta_y_q}
