from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class PairLiteV2:
    mode: str
    c_value: float = 1.0

    def __post_init__(self) -> None:
        self.q_word = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=1)
        self.y_word = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=1)
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=50000, min_df=1)
        self.model = LogisticRegression(C=self.c_value, class_weight="balanced", max_iter=1000, solver="liblinear")

    def fit(self, rows: list[dict[str, Any]]) -> "PairLiteV2":
        self.model.fit(self._fit_x(rows), [int(r["gold"]) for r in rows])
        return self

    def score(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        return [float(x) for x in self.model.predict_proba(self._x(rows))[:, 1]]

    def predict(self, rows: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
        scores = self.score(rows)
        return [{**r, "mode": r.get("mode", self.mode), "score": s, "pred": int(s >= threshold), "threshold": threshold} for r, s in zip(rows, scores)]

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
        if self.mode != "y-only":
            mats.append(self.q_word.fit_transform(q))
        if self.mode != "q-only":
            mats.append(self.y_word.fit_transform(y))
        mats.append(self.char.fit_transform(both))
        mats.append(extra_features(rows, self.mode))
        return hstack(mats)

    def _x(self, rows: list[dict[str, Any]]):
        q, y, both = self._texts(rows)
        mats = []
        if self.mode != "y-only":
            mats.append(self.q_word.transform(q))
        if self.mode != "q-only":
            mats.append(self.y_word.transform(y))
        mats.append(self.char.transform(both))
        mats.append(extra_features(rows, self.mode))
        return hstack(mats)


def extra_features(rows: list[dict[str, Any]], mode: str):
    feats = []
    for row in rows:
        q = (row.get("q_eval") or row.get("q_private") or "").lower()
        y = (row.get("y_private") or "").lower()
        refusal = int(any(w in y for w in ["cannot", "can't", "not help", "refuse", "抱歉", "不能"]))
        warning = int(any(w in y for w in ["scam", "red flag", "verify", "report", "风险", "诈骗", "核实"]))
        action = int(any(w in y for w in ["apply", "contact", "transfer", "submit", "provide", "script", "steps", "申请", "联系", "转账"]))
        q_terms = {w for w in q.split() if len(w) > 4}
        y_terms = {w for w in y.split() if len(w) > 4}
        overlap = len(q_terms & y_terms) / max(1, min(len(q_terms), len(y_terms)))
        feats.append([refusal, warning, action, overlap, action * overlap, action * (1 - refusal)])
    return np.asarray(feats, dtype=float)
