from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from scipy.special import expit, logit
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from frauddistill.exp1_ccfa.pairlite_cpu import PairLiteCPUDetector
from frauddistill.student.relation_features import cross_qy_features, unary_text_features


@dataclass
class ResidualRelationProfile:
    train_rows: int
    relation_dim: int
    selected_lambda: float


class ResidualRelationCPUDetector:
    """CPU-only residual relation model.

    The full score is y-only logit plus a bounded relation correction.  With
    lambda_value=0 it is exactly y-only, which keeps the v6r1 comparison
    interpretable and prevents q features from replacing the answer signal.
    """

    def __init__(
        self,
        seed: int = 20260724,
        pairlite_config: dict | None = None,
        lambda_value: float = 0.0,
        correction_clip: float = 1.5,
        svd_components: int = 64,
        mlp_hidden: int = 64,
        mlp_alpha: float = 1e-2,
        mlp_max_iter: int = 200,
    ):
        self.seed = seed
        self.pairlite_config = dict(pairlite_config or {})
        self.lambda_value = float(lambda_value)
        self.correction_clip = float(correction_clip)
        self.svd_components = int(svd_components)
        self.q_model = self._pairlite("B1")
        self.y_model = self._pairlite("B1")
        self.add_model = self._pairlite("B1")
        self.tfidf = _tfidf_union(max_features=int(self.pairlite_config.get("word_features", 80000)))
        self.svd = TruncatedSVD(n_components=self.svd_components, random_state=seed)
        self.scaler = StandardScaler()
        self.relation_head = MLPClassifier(
            hidden_layer_sizes=(int(mlp_hidden),),
            alpha=float(mlp_alpha),
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=int(mlp_max_iter),
            random_state=seed,
        )
        self.profile: ResidualRelationProfile | None = None

    def fit(self, rows: list[dict], labels: list[str], sample_weight: np.ndarray | None = None) -> "ResidualRelationCPUDetector":
        self.q_model.fit(rows, labels, "q_only")
        self.y_model.fit(rows, labels, "y_only")
        self.add_model.fit(rows, labels, "q_y")
        q_texts = [str(row.get("user_query", "")) for row in rows]
        y_texts = [str(row.get("target_model_answer", "")) for row in rows]
        self.tfidf.fit([*q_texts, *y_texts])
        self.svd.fit(self.tfidf.transform([*q_texts, *y_texts]))
        features = self._relation_matrix(rows)
        scaled = self.scaler.fit_transform(features)
        y = np.asarray([label == "unsafe" for label in labels], dtype=np.int8)
        try:
            self.relation_head.fit(scaled, y, sample_weight=sample_weight)
        except TypeError:
            self.relation_head.fit(scaled, y)
        self.profile = ResidualRelationProfile(train_rows=len(rows), relation_dim=int(features.shape[1]), selected_lambda=self.lambda_value)
        return self

    def predict_proba(self, rows: list[dict], mode: str = "q_y") -> np.ndarray:
        return expit(self.decision_logit(rows))

    def decision_logit(self, rows: list[dict]) -> np.ndarray:
        y_prob = np.clip(self.y_model.predict_proba(rows, "y_only"), 1e-5, 1 - 1e-5)
        y_logit = logit(y_prob)
        correction = self.relation_correction(rows)
        return y_logit + self.lambda_value * np.clip(correction, -self.correction_clip, self.correction_clip)

    def relation_correction(self, rows: list[dict]) -> np.ndarray:
        features = self.scaler.transform(self._relation_matrix(rows))
        prob = np.clip(self.relation_head.predict_proba(features)[:, 1], 1e-5, 1 - 1e-5)
        y_prob = np.clip(self.y_model.predict_proba(rows, "y_only"), 1e-5, 1 - 1e-5)
        return logit(prob) - logit(y_prob)

    def with_lambda(self, value: float) -> "ResidualRelationCPUDetector":
        self.lambda_value = float(value)
        if self.profile is not None:
            self.profile.selected_lambda = self.lambda_value
        return self

    def _relation_matrix(self, rows: list[dict]) -> np.ndarray:
        q_texts = [str(row.get("user_query", "")) for row in rows]
        y_texts = [str(row.get("target_model_answer", "")) for row in rows]
        q_sparse = self.tfidf.transform(q_texts)
        y_sparse = self.tfidf.transform(y_texts)
        q = self.svd.transform(q_sparse).astype(np.float32)
        y = self.svd.transform(y_sparse).astype(np.float32)
        denom = np.maximum(np.linalg.norm(q, axis=1) * np.linalg.norm(y, axis=1), 1e-9)
        cosine = np.sum(q * y, axis=1, keepdims=True) / denom[:, None]
        q_prob = self.q_model.predict_proba(rows, "q_only")[:, None]
        y_prob = self.y_model.predict_proba(rows, "y_only")[:, None]
        add_prob = self.add_model.predict_proba(rows, "q_y")[:, None]
        q_logit = logit(np.clip(q_prob, 1e-5, 1 - 1e-5))
        y_logit = logit(np.clip(y_prob, 1e-5, 1 - 1e-5))
        add_logit = logit(np.clip(add_prob, 1e-5, 1 - 1e-5))
        cross = cross_qy_features(q_texts, y_texts)
        unary_gap = unary_text_features(y_texts) - unary_text_features(q_texts)
        return np.hstack([y_logit, q_logit, add_logit, q, y, np.abs(q - y), q * y, cosine, cross, unary_gap]).astype(np.float32)

    def _pairlite(self, level: str) -> PairLiteCPUDetector:
        cfg = self.pairlite_config
        return PairLiteCPUDetector(
            level=level,
            alpha=float(cfg.get("alpha", 0.0003)),
            l1_ratio=float(cfg.get("l1_ratio", 0.0)),
            max_iter=int(cfg.get("max_iter", 45)),
            seed=self.seed,
            word_features=int(cfg.get("word_features", 80000)),
            char_features=int(cfg.get("char_features", 80000)),
            hash_features=int(cfg.get("hash_features", 262144)),
            top_k_cross=int(cfg.get("top_k_cross", 12)),
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ResidualRelationCPUDetector":
        return joblib.load(path)


def _tfidf_union(max_features: int) -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=max_features, sublinear_tf=True)),
        ]
    )
