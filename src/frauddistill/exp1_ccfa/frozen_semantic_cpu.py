from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from frauddistill.exp1_ccfa.embedding_cache import FrozenEmbeddingCache


SEMANTIC_LEVELS = ("S0", "S1", "S2")
SEMANTIC_INPUT_MODES = ("q_only", "y_only", "q_y")


@dataclass
class SemanticCPUProfile:
    train_rows: int
    feature_dim: int


class FrozenSemanticCPUDetector:
    def __init__(
        self,
        level: str,
        encoder_config: dict,
        cache_dir: str,
        c: float = 1.0,
        relation_weight: float = 1.0,
        max_iter: int = 200,
        seed: int = 20260724,
    ):
        if level not in SEMANTIC_LEVELS:
            raise ValueError(f"unknown semantic level: {level}")
        self.level = level
        self.cache = FrozenEmbeddingCache(cache_dir, encoder_config)
        self.c = c
        self.relation_weight = relation_weight
        self.max_iter = max_iter
        self.seed = seed
        self.classifier = LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced", random_state=seed, solver="liblinear")
        self.feature_rms_: np.ndarray | None = None
        self.profile: SemanticCPUProfile | None = None

    def fit(self, rows: list[dict], labels: list[str], mode: str = "q_y") -> "FrozenSemanticCPUDetector":
        features = self.features(rows, mode, fit_scaler=True)
        y = np.asarray([label == "unsafe" for label in labels], dtype=np.int8)
        self.classifier.fit(features, y)
        self.profile = SemanticCPUProfile(train_rows=len(rows), feature_dim=int(features.shape[1]))
        return self

    def predict_proba(self, rows: list[dict], mode: str = "q_y") -> np.ndarray:
        return self.classifier.predict_proba(self.features(rows, mode))[:, 1]

    def features(self, rows: list[dict], mode: str = "q_y", fit_scaler: bool = False) -> np.ndarray:
        if mode not in SEMANTIC_INPUT_MODES:
            raise ValueError(f"unknown input mode: {mode}")
        query_prefix = str(self.cache.config.get("query_prefix", self.cache.config.get("prefix", "")))
        passage_prefix = str(self.cache.config.get("passage_prefix", self.cache.config.get("prefix", "")))
        q = self.cache.encode([str(row.get("user_query", "")) for row in rows], prefix=query_prefix)
        y = self.cache.encode([str(row.get("target_model_answer", "")) for row in rows], prefix=passage_prefix)
        zeros = np.zeros_like(q)
        if mode == "q_only":
            features = np.hstack([q, zeros, self._zero_relation(q)])
            return self._scale_features(features, fit_scaler)
        if mode == "y_only":
            features = np.hstack([zeros, y, self._zero_relation(q)])
            return self._scale_features(features, fit_scaler)
        features = np.hstack([q, y, self.relation_features(q, y, rows)])
        return self._scale_features(features, fit_scaler)

    def relation_features(self, q: np.ndarray, y: np.ndarray, rows: list[dict]) -> np.ndarray:
        if self.level == "S0":
            return self._zero_relation(q)
        if self.level == "S2":
            pair_prefix = str(self.cache.config.get("pair_prefix", self.cache.config.get("query_prefix", self.cache.config.get("prefix", ""))))
            pair_texts = [f"{row.get('user_query', '')} [SEP] {row.get('target_model_answer', '')}" for row in rows]
            pair_config_prefix = pair_prefix
            pair = self.cache.encode(pair_texts, prefix=pair_config_prefix)
        else:
            pair = np.zeros((q.shape[0], 0), dtype=np.float32)
        product = q * y
        abs_diff = np.abs(q - y)
        cosine = np.sum(q * y, axis=1, keepdims=True)
        return (np.hstack([pair, abs_diff, product, cosine]) * float(getattr(self, "relation_weight", 1.0))).astype(np.float32)

    def _zero_relation(self, q: np.ndarray) -> np.ndarray:
        if self.level == "S0":
            return np.zeros((q.shape[0], 0), dtype=np.float32)
        multiplier = 3 if self.level == "S2" else 2
        return np.zeros((q.shape[0], q.shape[1] * multiplier + 1), dtype=np.float32)

    def _scale_features(self, features: np.ndarray, fit_scaler: bool) -> np.ndarray:
        features = features.astype(np.float32)
        if fit_scaler or self.feature_rms_ is None:
            rms = np.sqrt(np.mean(np.square(features), axis=0, keepdims=True))
            self.feature_rms_ = np.maximum(rms, 1e-6).astype(np.float32)
        return features / self.feature_rms_

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FrozenSemanticCPUDetector":
        return joblib.load(path)
