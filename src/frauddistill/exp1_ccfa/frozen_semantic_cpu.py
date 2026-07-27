from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from frauddistill.exp1_ccfa.embedding_cache import FrozenEmbeddingCache


SEMANTIC_LEVELS = ("S0", "S1")
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
        max_iter: int = 200,
        seed: int = 20260724,
    ):
        if level not in SEMANTIC_LEVELS:
            raise ValueError(f"unknown semantic level: {level}")
        self.level = level
        self.cache = FrozenEmbeddingCache(cache_dir, encoder_config)
        self.c = c
        self.max_iter = max_iter
        self.seed = seed
        self.classifier = LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced", random_state=seed, solver="liblinear")
        self.profile: SemanticCPUProfile | None = None

    def fit(self, rows: list[dict], labels: list[str], mode: str = "q_y") -> "FrozenSemanticCPUDetector":
        features = self.features(rows, mode)
        y = np.asarray([label == "unsafe" for label in labels], dtype=np.int8)
        self.classifier.fit(features, y)
        self.profile = SemanticCPUProfile(train_rows=len(rows), feature_dim=int(features.shape[1]))
        return self

    def predict_proba(self, rows: list[dict], mode: str = "q_y") -> np.ndarray:
        return self.classifier.predict_proba(self.features(rows, mode))[:, 1]

    def features(self, rows: list[dict], mode: str = "q_y") -> np.ndarray:
        if mode not in SEMANTIC_INPUT_MODES:
            raise ValueError(f"unknown input mode: {mode}")
        q = self.cache.encode([str(row.get("user_query", "")) for row in rows])
        y = self.cache.encode([str(row.get("target_model_answer", "")) for row in rows])
        zeros = np.zeros_like(q)
        if mode == "q_only":
            return np.hstack([q, zeros, self._zero_relation(q)])
        if mode == "y_only":
            return np.hstack([zeros, y, self._zero_relation(q)])
        return np.hstack([q, y, self.relation_features(q, y, rows)])

    def relation_features(self, q: np.ndarray, y: np.ndarray, rows: list[dict]) -> np.ndarray:
        if self.level == "S0":
            return self._zero_relation(q)
        product = q * y
        diff = q - y
        cosine = np.sum(q * y, axis=1, keepdims=True)
        scalars = np.asarray([_scalar_features(row) for row in rows], dtype=np.float32)
        return np.hstack([product, diff, cosine, scalars]).astype(np.float32)

    def _zero_relation(self, q: np.ndarray) -> np.ndarray:
        if self.level == "S0":
            return np.zeros((q.shape[0], 0), dtype=np.float32)
        return np.zeros((q.shape[0], q.shape[1] * 2 + 4), dtype=np.float32)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FrozenSemanticCPUDetector":
        return joblib.load(path)


def _scalar_features(row: dict) -> list[float]:
    q = str(row.get("user_query", "")).lower()
    y = str(row.get("target_model_answer", "")).lower()
    fraud_prompt = 1.0 if row.get("prompt_risk_domain") == "fraud_core" else 0.0
    refusal = 1.0 if any(token in y for token in ("can't", "cannot", "sorry", "not able", "illegal", "won't")) else 0.0
    action = 1.0 if any(token in y for token in ("step", "first", "then", "use", "send", "click", "password", "credential")) else 0.0
    ratio = len(y) / max(len(q), 1)
    return [ratio, refusal * fraud_prompt, action * fraud_prompt, fraud_prompt]
