from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .relation_features import cross_qy_features, unary_text_features


class PairTfidfDetector:
    """Shared-vocabulary dual channel detector with explicit q/y masks and interaction features."""

    def __init__(self, max_features: int = 60000, C: float = 1.0):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.classifier = LogisticRegression(max_iter=3000, class_weight="balanced", C=C, solver="liblinear")

    def fit(self, rows: list[dict], labels: list[str], mode: str = "q_y") -> "PairTfidfDetector":
        corpus = [str(row.get("user_query", "")) for row in rows] + [str(row.get("target_model_answer", "")) for row in rows]
        self.vectorizer.fit(corpus)
        self.classifier.fit(self.features(rows, mode), np.asarray([label == "unsafe" for label in labels], dtype=int))
        return self

    def features(self, rows: list[dict], mode: str = "q_y"):
        mode = "q_y" if mode == "q+y" else mode
        if mode not in {"q_only", "y_only", "q_y"}:
            raise ValueError(f"unknown input mode: {mode}")
        queries = [str(row.get("user_query", "")) for row in rows]
        answers = [str(row.get("target_model_answer", "")) for row in rows]
        q = self.unary_q_features(queries)
        y = self.unary_y_features(answers)
        xy = self.cross_qy_features(queries, answers)
        zero_q = csr_matrix(q.shape, dtype=q.dtype)
        zero_y = csr_matrix(y.shape, dtype=y.dtype)
        zero_xy = csr_matrix(xy.shape, dtype=xy.dtype)
        if mode == "q_only":
            return hstack([q, zero_y, zero_xy], format="csr")
        if mode == "y_only":
            return hstack([zero_q, y, zero_xy], format="csr")
        return hstack([q, y, xy], format="csr")

    def unary_q_features(self, queries: list[str]):
        q = self.vectorizer.transform(queries)
        return hstack([q, csr_matrix(unary_text_features(queries))], format="csr")

    def unary_y_features(self, answers: list[str]):
        y = self.vectorizer.transform(answers)
        return hstack([y, csr_matrix(unary_text_features(answers))], format="csr")

    def cross_qy_features(self, queries: list[str], answers: list[str]):
        q = self.vectorizer.transform(queries)
        y = self.vectorizer.transform(answers)
        return hstack([abs(q - y), q.multiply(y), csr_matrix(cross_qy_features(queries, answers))], format="csr")

    def predict_proba(self, rows: list[dict], mode: str = "q_y") -> np.ndarray:
        return self.classifier.predict_proba(self.features(rows, mode))[:, 1]
