# -*- coding: utf-8 -*-
"""Calibration backends: raw / Platt / isotonic / temperature (guide 3.6, 10.4).

Selection protocol (guide 3.6):
1. compare dev Brier; 2. tie-break by ECE; 3. must not change AUPRC ranking;
4. if no post-processor beats raw, keep raw.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _to_binary(labels) -> np.ndarray:
    return np.asarray([1 if str(label) == "unsafe" else 0 for label in labels], dtype=float)


def _clip(scores) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return 0.0
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def true_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """True macro F1 = mean of per-class F1 (guide 3.1)."""
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro", zero_division=0))


class ScoreCalibrator:
    """Calibration backend fitted on dev only (guide 10.4)."""

    METHODS = ("raw", "platt", "isotonic", "temperature")

    def __init__(self, method: str = "platt"):
        if method not in self.METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {self.METHODS}")
        self.method = method
        self.model: Any = None
        self.threshold: float = 0.5
        self.temperature: float = 1.0

    # ---------------------------------------------------------------- fit
    def fit(self, scores: list[float], labels: list[str]) -> "ScoreCalibrator":
        y = _to_binary(labels)
        x = _clip(scores).reshape(-1, 1)
        if self.method == "raw":
            self.model = None
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(x.ravel(), y)
        elif self.method == "temperature":
            # temperature scaling: p = sigmoid(logit / T); T fitted by 1D search on dev NLL
            logits = np.log(_clip(scores) / (1.0 - _clip(scores)))
            best_t, best_nll = 1.0, float("inf")
            for t in np.linspace(0.2, 5.0, 97):
                p = 1.0 / (1.0 + np.exp(-logits / t))
                p = np.clip(p, 1e-6, 1.0 - 1e-6)
                nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                if nll < best_nll:
                    best_nll, best_t = nll, t
            self.temperature = float(best_t)
            self.model = None
        else:  # platt
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(x, y)
        return self

    def calibrate(self, scores: list[float]) -> list[float]:
        x = _clip(scores).reshape(-1, 1)
        if self.method == "raw":
            out = x.ravel()
        elif self.method == "isotonic":
            out = self.model.predict(x.ravel())
        elif self.method == "temperature":
            logits = np.log(x.ravel() / (1.0 - x.ravel()))
            out = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        else:  # platt
            out = self.model.predict_proba(x)[:, 1]
        return [float(max(0.0, min(1.0, v))) for v in out]

    # ------------------------------------------------------------ metrics
    def evaluate(self, scores: list[float], labels: list[str]) -> dict[str, float]:
        y = _to_binary(labels)
        p = np.asarray(self.calibrate(scores), dtype=float)
        return {
            "brier": round(brier_score(y, p), 6),
            "ece": round(ece_score(y, p), 6),
            "auprc": round(float(_auprc(y, p)), 6),
            "auroc": round(float(_auroc(y, p)), 6),
        }

    def select_threshold(
        self,
        scores: list[float],
        labels: list[str],
        max_fpr: float = 0.08,
    ) -> float:
        """Pick threshold maximizing true macro-F1 under a hard FPR cap."""
        y = _to_binary(labels).astype(int)
        s = np.asarray(self.calibrate(scores), dtype=float)
        candidates = np.unique(np.round(np.clip(s, 0.0, 1.0), 4))
        if len(candidates) < 2:
            candidates = np.linspace(0.05, 0.95, 19)
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            pred = (s >= t).astype(int)
            tn = float(((pred == 0) & (y == 0)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            fpr = fp / max(tn + fp, 1.0)
            if fpr > max_fpr:
                continue
            f1 = true_macro_f1(y, pred)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = float(best_t)
        return self.threshold

    # ---------------------------------------------------------------- io
    def save(self, path: str) -> None:
        payload = {"method": self.method, "threshold": self.threshold, "temperature": self.temperature}
        if self.method == "platt" and self.model is not None:
            payload["coef"] = float(self.model.coef_[0][0])
            payload["intercept"] = float(self.model.intercept_[0])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ScoreCalibrator":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        cal = cls(payload.get("method", "platt"))
        cal.threshold = float(payload.get("threshold", 0.5))
        cal.temperature = float(payload.get("temperature", 1.0))
        if payload.get("coef") is not None:
            cal.model = LogisticRegression(max_iter=1000)
            cal.model.coef_ = np.array([[payload["coef"]]])
            cal.model.intercept_ = np.array([payload["intercept"]])
            cal.model.classes_ = np.array([0, 1])
        return cal


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(set(y.tolist())) == 2 and len(set(np.round(p, 6).tolist())) > 1:
        return float(average_precision_score(y, p))
    return float("nan")


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(set(y.tolist())) == 2 and len(set(np.round(p, 6).tolist())) > 1:
        return float(roc_auc_score(y, p))
    return float("nan")
