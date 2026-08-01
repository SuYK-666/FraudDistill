from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class Detector:
    mode: str
    c_value: float = 1.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.q_char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=30000, min_df=2)
        self.y_char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=30000, min_df=2)
        self.q_word = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
        self.y_word = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
        self.model = LogisticRegression(C=self.c_value, class_weight="balanced", max_iter=1000, solver="liblinear")

    def fit(self, rows: list[dict[str, Any]]) -> "Detector":
        self.model.fit(self._fit_x(rows), [int(r["gold"]) for r in rows])
        return self

    def score(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        return [float(x) for x in self.model.predict_proba(self._x(rows))[:, 1]]

    def predict(self, rows: list[dict[str, Any]], threshold: float | None = None) -> list[dict[str, Any]]:
        th = self.threshold if threshold is None else threshold
        return [
            {
                **r,
                "mode": r.get("mode", self.mode),
                "fitted_model_mode": self.mode,
                "score": s,
                "pred": int(s >= th),
                "threshold": th,
            }
            for r, s in zip(rows, self.score(rows))
        ]

    def _texts(self, rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        q, y = [], []
        for r in rows:
            q.append("" if self.mode == "y-only" else r.get("q_eval") or r.get("q_private", ""))
            y.append("" if self.mode == "q-only" else r.get("y_private", ""))
        return q, y

    def _fit_x(self, rows: list[dict[str, Any]]):
        q, y = self._texts(rows)
        mats = []
        if self.mode != "y-only":
            mats.extend([self.q_char.fit_transform(q), self.q_word.fit_transform(q)])
        if self.mode != "q-only":
            mats.extend([self.y_char.fit_transform(y), self.y_word.fit_transform(y)])
        return hstack(mats)

    def _x(self, rows: list[dict[str, Any]]):
        q, y = self._texts(rows)
        mats = []
        if self.mode != "y-only":
            mats.extend([self.q_char.transform(q), self.q_word.transform(q)])
        if self.mode != "q-only":
            mats.extend([self.y_char.transform(y), self.y_word.transform(y)])
        return hstack(mats)
