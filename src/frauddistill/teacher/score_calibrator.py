from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    """Platt / isotonic calibration fitted on dev only (guide 10.4)."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self.model: Any = None
        self.threshold: float = 0.5

    def fit(self, scores: list[float], labels: list[str]) -> "ScoreCalibrator":
        y = np.asarray([1 if str(label) == "unsafe" else 0 for label in labels], dtype=float)
        x = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6).reshape(-1, 1)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(x.ravel(), y)
        else:
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(x, y)
        return self

    def calibrate(self, scores: list[float]) -> list[float]:
        if self.model is None:
            return [float(max(0.0, min(1.0, s))) for s in scores]
        x = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6).reshape(-1, 1)
        if self.method == "isotonic":
            return [float(max(0.0, min(1.0, v))) for v in self.model.predict(x.ravel())]
        return [float(max(0.0, min(1.0, v))) for v in self.model.predict_proba(x)[:, 1]]

    def select_threshold(
        self,
        scores: list[float],
        labels: list[str],
        max_fpr: float = 0.08,
    ) -> float:
        """Pick threshold maximizing macro-F1 under a hard FPR cap (guide 24)."""
        y = np.asarray([1 if str(label) == "unsafe" else 0 for label in labels], dtype=int)
        s = np.asarray(scores, dtype=float)
        candidates = np.unique(np.round(np.clip(s, 0.0, 1.0), 4))
        if len(candidates) < 2:
            candidates = np.linspace(0.05, 0.95, 19)
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            pred = (s >= t).astype(int)
            tn = float(((pred == 0) & (y == 0)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            fn = float(((pred == 0) & (y == 1)).sum())
            tp = float(((pred == 1) & (y == 1)).sum())
            fpr = fp / max(tn + fp, 1.0)
            if fpr > max_fpr:
                continue
            prec = tp / max(tp + fp, 1.0)
            rec = tp / max(tp + fn, 1.0)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = float(best_t)
        return self.threshold

    def save(self, path: str) -> None:
        payload = {"method": self.method, "threshold": self.threshold}
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
        if payload.get("coef") is not None:
            cal.model = LogisticRegression(max_iter=1000)
            cal.model.coef_ = np.array([[payload["coef"]]])
            cal.model.intercept_ = np.array([payload["intercept"]])
            cal.model.classes_ = np.array([0, 1])
        return cal