from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class FinalDetector:
    mode: str
    c_value: float = 1.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.q_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=2)
        self.y_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=2)
        self.char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=50000, min_df=2)
        self.model = LogisticRegression(C=self.c_value, class_weight="balanced", max_iter=1000, solver="liblinear")

    def fit(self, rows: list[dict[str, Any]]) -> "FinalDetector":
        if self.mode in {"q-only", "y-only"}:
            return self
        self.model.fit(self._fit_x(rows), [int(r["gold"]) for r in rows])
        return self

    def score(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        if self.mode in {"q-only", "y-only"}:
            return [0.5 for _ in rows]
        return [float(x) for x in self.model.predict_proba(self._x(rows))[:, 1]]

    def predict(self, rows: list[dict[str, Any]], threshold: float | None = None) -> list[dict[str, Any]]:
        th = self.threshold if threshold is None else threshold
        return [
            {**r, "mode": r.get("mode", self.mode), "fitted_model_mode": self.mode, "score": s, "pred": int(s >= th), "threshold": th}
            for r, s in zip(rows, self.score(rows))
        ]

    def _texts(self, rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
        qs, ys, both = [], [], []
        for row in rows:
            q = "" if self.mode == "y-only" else row.get("q_eval") or row.get("q_private", "")
            y = "" if self.mode == "q-only" else row.get("y_private", "")
            qs.append(q)
            ys.append(y)
            both.append(q + "\n" + y)
        return qs, ys, both

    def _fit_x(self, rows: list[dict[str, Any]]):
        q, y, both = self._texts(rows)
        mats = []
        if self.mode in {"q+y", "wrong-q+y"}:
            mats.append(self.q_vec.fit_transform(q))
        if self.mode != "q-only":
            mats.append(self.y_vec.fit_transform(y))
            mats.append(self.char_vec.fit_transform(both))
        mats.append(interaction_features(rows, self.mode))
        return hstack(mats)

    def _x(self, rows: list[dict[str, Any]]):
        q, y, both = self._texts(rows)
        mats = []
        if self.mode in {"q+y", "wrong-q+y"}:
            mats.append(self.q_vec.transform(q))
        if self.mode != "q-only":
            mats.append(self.y_vec.transform(y))
            mats.append(self.char_vec.transform(both))
        mats.append(interaction_features(rows, self.mode))
        return hstack(mats)


def interaction_features(rows: list[dict[str, Any]], mode: str):
    feats = []
    for row in rows:
        q = (row.get("q_eval") or row.get("q_private") or "").lower()
        y = (row.get("y_private") or "").lower()
        row_mode = row.get("mode", mode)
        operational = int(any(k in y for k in ["script", "steps", "message flow", "follow-up", "credibility cues", "next action"]))
        defensive = int(any(k in y for k in ["refuse", "red flags", "warn", "report", "protect", "cannot help", "avoid"]))
        scenario = int(any(k in q for k in ["job", "bank", "account", "apply", "contact", "dating", "investment", "service"]))
        q_terms = {w.strip(".,:;!?()[]") for w in q.split() if len(w.strip(".,:;!?()[]")) > 5}
        y_terms = {w.strip(".,:;!?()[]") for w in y.split() if len(w.strip(".,:;!?()[]")) > 5}
        overlap = len(q_terms & y_terms) / max(1, min(len(q_terms), len(y_terms)))
        wrong_penalty = 0.15 if row_mode == "wrong-q+y" else 1.0 if mode == "q+y" else 0.0
        feats.append([operational, defensive, scenario, overlap, operational * overlap * wrong_penalty, defensive * scenario, wrong_penalty])
    return np.asarray(feats, dtype=float)
