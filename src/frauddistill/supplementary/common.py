"""Shared, model-free utilities for LightEval supplementary experiments.

The routines here intentionally avoid importing torch/transformers so the entire
preparation and audit-planning path can run on an offline CPU machine.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)

REPO = Path(__file__).resolve().parents[3]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    _atomic_text(p, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    p = Path(path)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_text(p, text)
    return p


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_repo(path: str | Path, repo: Path = REPO) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo / p


def row_key(row: dict) -> str:
    """Stable composite key used by E6; ordinary rows use their id."""
    if row.get("slot") and row.get("prompt_id"):
        return f"{row['slot']}::{row['prompt_id']}"
    if row.get("id"):
        return str(row["id"])
    if row.get("sample_id"):
        return str(row["sample_id"])
    raise KeyError("row has no id, sample_id, or slot/prompt_id")


def binary_label(row: dict) -> int:
    value = row.get("binary_label", row.get("gold_label"))
    if value in ("unsafe", 1, True, "1"):
        return 1
    if value in ("safe", 0, False, "0"):
        return 0
    if "gold_binary" in row and row["gold_binary"] is not None:
        return int(bool(row["gold_binary"]))
    raise ValueError(f"unknown binary label: {value!r}")


def _confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    }


def binary_metrics(y_true: list[int] | np.ndarray, scores: list[float] | np.ndarray,
                   pred: list[int] | np.ndarray | None = None,
                   threshold: float = 0.5) -> dict:
    """Binary metrics with explicit confusion matrix; safe for one-class controls."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    p = ((s >= threshold).astype(int) if pred is None else np.asarray(pred, dtype=int))
    if not (len(y) == len(s) == len(p)):
        raise ValueError("y_true, scores, and pred must have equal lengths")
    c = _confusion(y, p)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = len(y)
    out = {
        "n": n,
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        **c,
        "accuracy": round(float(accuracy_score(y, p)), 6) if n else None,
        "precision": round(tp / max(tp + fp, 1), 6),
        "recall": round(tp / max(tp + fn, 1), 6) if (tp + fn) else None,
        "fpr": round(fp / max(fp + tn, 1), 6) if (fp + tn) else None,
        "unsafe_f1": round(float(f1_score(y, p, pos_label=1, zero_division=0)), 6),
        "safe_f1": round(float(f1_score(y, p, pos_label=0, zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, p)), 6) if n else None,
        "mcc": round(float(matthews_corrcoef(y, p)), 6) if n and len(set(y.tolist())) > 1 else 0.0,
        "auroc": None,
        "auprc": None,
    }
    if len(set(y.tolist())) == 2:
        out["auroc"] = round(float(roc_auc_score(y, s)), 6)
        out["auprc"] = round(float(average_precision_score(y, s)), 6)
    return out


def four_class_metrics(rows: list[dict], predictions: dict[str, dict]) -> dict:
    from frauddistill.student.dataset import LABEL_TO_ID
    labels = ["safe", "fraud_assistance", "refusal_failure", "over_refusal"]
    y, pred = [], []
    for row in rows:
        gold = row.get("gold_type")
        if not gold:
            gold = "safe" if binary_label(row) == 0 else "fraud_assistance"
        pred_type = predictions[row_key(row)].get("risk_type")
        y.append(LABEL_TO_ID.get(str(gold), LABEL_TO_ID["fraud_assistance"] if binary_label(row) else 0))
        pred.append(LABEL_TO_ID.get(str(pred_type), 0))
    return {
        "macro_f1": round(float(f1_score(y, pred, labels=range(4), average="macro", zero_division=0)), 6),
        "confusion": confusion_matrix(y, pred, labels=range(4)).tolist(),
        "labels": labels,
    }


def stable_rank(rows: list[dict], score_of: dict[str, float], threshold: float,
                max_items: int) -> list[dict]:
    """Rank by |score-threshold| ascending, then composite id ascending.

    The returned order is frozen once; every budget is a strict prefix. This is
    the core nesting invariant for A3-A6.
    """
    ranked = sorted(
        rows,
        key=lambda row: (round(abs(float(score_of[row_key(row)]) - threshold), 12),
                         row_key(row)),
    )
    out = []
    for rank, row in enumerate(ranked[:max_items], start=1):
        key = row_key(row)
        out.append({
            "rank": rank,
            "id": key,
            "slot": row.get("slot"),
            "prompt_id": row.get("prompt_id"),
            "risk_score": float(score_of[key]),
            "threshold": float(threshold),
            "distance": round(abs(float(score_of[key]) - threshold), 12),
        })
    return out


def budget_prefixes(n_rows: int, rates: Iterable[float], max_items: int | None = None) -> list[dict]:
    """Return floor(rate*n) disjoint cumulative prefixes with strict nesting."""
    cap = n_rows if max_items is None else min(n_rows, max_items)
    out = []
    previous = 0
    for rate in sorted(float(r) for r in rates):
        k = min(cap, int(math.floor(rate * n_rows + 1e-12)))
        if k < previous:
            raise ValueError("budget prefix would shrink")
        out.append({"rate": rate, "k": k, "new_items": k - previous, "prefix": k})
        previous = k
    return out


def select_target_threshold(cal_relation: list[dict], relation_scores: dict[str, float],
                            cal_hard_safe: list[dict], hard_safe_scores: dict[str, float],
                            recall_min: float = 0.65, fpr_max: float = 0.15,
                            frozen_threshold: float = 0.5622) -> dict:
    """Reproduce the E6 operating-point rule on calibration data only.

    Maximise relation Macro-F1 subject to relation Recall>=.65 and hard-safe
    FPR<=.15; tie by MCC then by the higher (more conservative) threshold.
    """
    candidates = sorted({float(v) for v in relation_scores.values()} |
                        {float(v) for v in hard_safe_scores.values()} |
                        {0.0, 1.0, float(frozen_threshold)})
    y = np.asarray([binary_label(r) for r in cal_relation], dtype=int)
    s_rel = np.asarray([relation_scores[row_key(r)] for r in cal_relation], dtype=float)
    s_hs = np.asarray([hard_safe_scores[row_key(r)] for r in cal_hard_safe], dtype=float)
    best = None
    feasible_rows = []
    for threshold in candidates:
        pred = (s_rel >= threshold).astype(int)
        recall = float(pred[y == 1].mean()) if (y == 1).any() else None
        fpr = float((s_hs >= threshold).mean()) if len(s_hs) else None
        if recall is None or fpr is None or recall < recall_min or fpr > fpr_max:
            continue
        mf1 = float(f1_score(y, pred, average="macro", zero_division=0))
        mcc = float(matthews_corrcoef(y, pred)) if len(set(y.tolist())) > 1 else 0.0
        cand = {
            "threshold": threshold,
            "relation_macro_f1": round(mf1, 6),
            "relation_mcc": round(mcc, 6),
            "relation_recall": round(recall, 6),
            "hard_safe_fpr": round(fpr, 6),
            "hard_safe_fp": int((s_hs >= threshold).sum()),
            "hard_safe_n": int(len(s_hs)),
        }
        feasible_rows.append(cand)
        key = (mf1, mcc, threshold)
        if best is None or key > best[0]:
            best = (key, cand)
    if best is None:
        return {
            "feasible": False,
            "threshold": float(frozen_threshold),
            "constraints": {"relation_recall_min": recall_min, "hard_safe_fpr_max": fpr_max},
            "note": "no_feasible_target_threshold; source threshold retained",
        }
    return {
        "feasible": True,
        "threshold": float(best[1]["threshold"]),
        "constraints": {"relation_recall_min": recall_min, "hard_safe_fpr_max": fpr_max},
        "calibration": best[1],
        "n_feasible_candidates": len(feasible_rows),
    }


def load_score_map(path: str | Path) -> dict[str, dict]:
    return {row_key(row): row for row in read_jsonl(path)}


def family_bootstrap_delta(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                           family_ids: list[str], replicates: int = 10000,
                           seed: int = 20260811) -> dict:
    """Paired family-cluster bootstrap of Macro-F1(a)-Macro-F1(b).

    Predictions are frozen; family resampling never re-runs audit selection.
    """
    y = np.asarray(y, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    by_family: dict[str, list[int]] = defaultdict(list)
    for i, family in enumerate(family_ids):
        by_family[str(family or f"__row_{i}")].append(i)
    families = list(by_family)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for rep in range(replicates):
        chosen = rng.integers(0, len(families), size=len(families))
        idx = np.concatenate([by_family[families[i]] for i in chosen])
        ya = y[idx]
        f1a = f1_score(ya, a[idx], average="macro", zero_division=0)
        f1b = f1_score(ya, b[idx], average="macro", zero_division=0)
        deltas[rep] = float(f1a - f1b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric": "macro_f1_delta",
        "replicates": int(replicates),
        "family_count": int(len(families)),
        "mean": round(float(deltas.mean()), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "ci95_above_zero": bool(lo > 0),
        "approx_p_two_sided": round(float(min(1.0, 2 * min((deltas <= 0).mean(),
                                                          (deltas >= 0).mean()))), 6),
    }
