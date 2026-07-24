from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .relation_features import relation_feature_matrix


class PairTfidfDetector:
    """Shared-vocabulary dual channel detector with explicit q/y masks and interaction features."""

    def __init__(self, max_features: int = 60000, C: float = 1.0):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.classifier = LogisticRegression(max_iter=3000, class_weight="balanced", C=C, solver="liblinear")

    def fit(self, rows: list[dict], labels: list[str]) -> "PairTfidfDetector":
        corpus = [str(row.get("user_query", "")) for row in rows] + [str(row.get("target_model_answer", "")) for row in rows]
        self.vectorizer.fit(corpus)
        self.classifier.fit(self.features(rows, "q_y"), np.asarray([label == "unsafe" for label in labels], dtype=int))
        return self

    def features(self, rows: list[dict], mode: str = "q_y"):
        mode = "q_y" if mode == "q+y" else mode
        if mode not in {"q_only", "y_only", "q_y"}:
            raise ValueError(f"unknown input mode: {mode}")
        queries = [str(row.get("user_query", "")) for row in rows]
        answers = [str(row.get("target_model_answer", "")) for row in rows]
        q = self.vectorizer.transform(queries)
        y = self.vectorizer.transform(answers)
        zero = csr_matrix(q.shape, dtype=q.dtype)
        if mode == "q_only":
            return hstack([q, zero, zero, zero, csr_matrix((len(rows), 10))], format="csr")
        if mode == "y_only":
            return hstack([zero, y, zero, zero, csr_matrix((len(rows), 10))], format="csr")
        return hstack([q, y, abs(q - y), q.multiply(y), csr_matrix(relation_feature_matrix(queries, answers))], format="csr")

    def predict_proba(self, rows: list[dict], mode: str = "q_y") -> np.ndarray:
        return self.classifier.predict_proba(self.features(rows, mode))[:, 1]
