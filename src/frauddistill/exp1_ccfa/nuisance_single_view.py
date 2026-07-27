from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion


@dataclass
class NuisanceScores:
    q_prob: np.ndarray
    y_prob: np.ndarray
    q_logit: np.ndarray
    y_logit: np.ndarray


class SingleViewNuisanceSelectors:
    def __init__(self, c: float = 1.0, seed: int = 20260727):
        self.c = c
        self.seed = seed
        self.q_vectorizer = _vectorizer()
        self.y_vectorizer = _vectorizer()
        self.q_model = LogisticRegression(C=c, max_iter=300, class_weight="balanced", solver="liblinear", random_state=seed)
        self.y_model = LogisticRegression(C=c, max_iter=300, class_weight="balanced", solver="liblinear", random_state=seed)

    def fit(self, rows: list[dict]) -> "SingleViewNuisanceSelectors":
        labels = np.asarray([row["exp1_label"] == "unsafe" for row in rows], dtype=np.int8)
        q_texts = [str(row.get("user_query", "")) for row in rows]
        y_texts = [str(row.get("target_model_answer", "")) for row in rows]
        self.q_model.fit(self.q_vectorizer.fit_transform(q_texts), labels)
        self.y_model.fit(self.y_vectorizer.fit_transform(y_texts), labels)
        return self

    def score(self, rows: list[dict]) -> NuisanceScores:
        q_prob = self.q_model.predict_proba(self.q_vectorizer.transform([str(row.get("user_query", "")) for row in rows]))[:, 1]
        y_prob = self.y_model.predict_proba(self.y_vectorizer.transform([str(row.get("target_model_answer", "")) for row in rows]))[:, 1]
        return NuisanceScores(q_prob=q_prob, y_prob=y_prob, q_logit=logit(q_prob), y_logit=logit(y_prob))

    def auc(self, rows: list[dict], view: str) -> float:
        labels = [row["exp1_label"] == "unsafe" for row in rows]
        scores = self.score(rows).q_prob if view == "q" else self.score(rows).y_prob
        if len(set(labels)) < 2:
            return 0.5
        return float(roc_auc_score(labels, scores))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SingleViewNuisanceSelectors":
        return joblib.load(path)


def independent_single_view_probe_auc(train: list[dict], test: list[dict], view: str, seed: int = 20260727) -> float:
    selector = SingleViewNuisanceSelectors(c=0.3, seed=seed).fit(train)
    return selector.auc(test, view)


def logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def _vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=80000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000, sublinear_tf=True)),
        ]
    )
