"""Canonical metrics & statistics (guide sections 9-14).

Single sklearn-backed binary evaluation, exact McNemar via binomtest,
Holm via statsmodels multipletests, and a paired group bootstrap whose
metric_fn is the exact same function used for the observed point estimate.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from frauddistill.exp2_static_repair.frames import EvaluationFrame

BINARY_CLASSES = ["safe", "unsafe"]
TYPE_CLASSES = ["safe", "fraud_assistance", "refusal_failure", "over_refusal"]


def evaluate_binary(frame: EvaluationFrame) -> dict:
    """Single implementation of binary metrics from one frame (guide 9.1)."""
    y_true = frame.y_true_binary
    y_pred = frame.y_pred_binary
    y_score = frame.y_score

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    unsafe_f1 = float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
    safe_f1 = float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    unsafe_recall = float(recall_score(y_true, y_pred, zero_division=0))

    metrics = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "unsafe_recall": unsafe_recall,
        "unsafe_f1": unsafe_f1,
        "safe_f1": safe_f1,
        "macro_f1": macro_f1,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))

    # Reconstructibility asserts (guide 9.2 / 29.2)
    assert metrics["n_positive"] == tp + fn
    assert metrics["n"] == tn + fp + fn + tp
    assert abs(macro_f1 - (unsafe_f1 + safe_f1) / 2) < 1e-12
    return metrics


def evaluate_four_class(frame: EvaluationFrame) -> dict:
    """Four-class macro-F1 (guide 10); never merged with binary macro-F1."""
    y_true = frame.y_true_type
    y_pred = frame.y_pred_type
    if y_true is None or y_pred is None:
        raise ValueError("four-class metrics need y_true_type/y_pred_type")
    per = {}
    for cls in TYPE_CLASSES:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per[cls] = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "four_class_macro_f1": float(np.mean(list(per.values()))),
        "per_class_f1": per,
        "n": int(len(y_true)),
    }


def choose_score_direction(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    allow_flip: bool = False,
) -> tuple[np.ndarray, dict]:
    """Detect reversed risk scores (guide 11.2); never silently flips."""
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return s, {"ap_forward": None, "ap_reverse": None, "flipped": False}
    ap_forward = float(average_precision_score(y, s))
    ap_reverse = float(average_precision_score(y, 1.0 - s))
    info = {"ap_forward": ap_forward, "ap_reverse": ap_reverse}
    if ap_reverse > ap_forward + 0.02:
        if not allow_flip:
            raise ValueError(
                "Risk score appears reversed (guide 11.2); fix the score "
                "definition instead of flipping silently."
            )
        return 1.0 - s, {**info, "flipped": True}
    return s, {**info, "flipped": False}


def exact_mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact two-sided McNemar via binomial test (guide 12)."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    a_wrong_b_right = int(np.sum(~a_correct & b_correct))
    a_right_b_wrong = int(np.sum(a_correct & ~b_correct))
    discordant = a_wrong_b_right + a_right_b_wrong
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(
            k=min(a_wrong_b_right, a_right_b_wrong),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue)
    accuracy_delta = float(np.mean(pred_b == y_true) - np.mean(pred_a == y_true))
    discordant_delta = float((a_wrong_b_right - a_right_b_wrong) / len(y_true))
    assert abs(accuracy_delta - discordant_delta) < 1e-12
    return {
        "baseline_wrong_teacher_right": a_wrong_b_right,
        "baseline_right_teacher_wrong": a_right_b_wrong,
        "raw_p": p_value,
        "accuracy_delta": accuracy_delta,
    }


def apply_holm(rows: list[dict]) -> list[dict]:
    """Holm-Bonferroni adjustment for the pre-registered primary family."""
    from statsmodels.stats.multitest import multipletests

    raw = np.array([row["raw_p"] for row in rows])
    reject, adjusted, _, _ = multipletests(raw, alpha=0.05, method="holm")
    output = []
    for row, p_adj, rejected in zip(rows, adjusted, reject, strict=True):
        output.append({**row, "holm_p": float(p_adj), "reject_h0": bool(rejected)})
    return output


def macro_f1_fn(y_true, y_pred) -> float:
    """The single metric_fn used by the main table AND the bootstrap."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def paired_group_bootstrap(
    frame: EvaluationFrame,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn=macro_f1_fn,
    reps: int = 10_000,
    seed: int = 20260806,
) -> dict:
    """Group-level paired bootstrap; groups are the resampling unit (14.3)."""
    rng = np.random.default_rng(seed)
    y_true = frame.y_true_binary
    group_ids = frame.group_ids
    groups = np.unique(group_ids)

    observed = metric_fn(y_true, pred_b) - metric_fn(y_true, pred_a)
    group_to_indices = {g: np.flatnonzero(group_ids == g) for g in groups}
    deltas = np.empty(reps, dtype=float)
    for i in range(reps):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_to_indices[g] for g in sampled])
        deltas[i] = metric_fn(y_true[indices], pred_b[indices]) - metric_fn(
            y_true[indices], pred_a[indices]
        )
    return {
        "observed_delta": float(observed),
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "reps": reps,
        "seed": seed,
    }
