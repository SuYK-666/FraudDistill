from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from frauddistill.student.relation_features import cross_qy_features, unary_text_features


PAIRLITE_LEVELS = ("L0", "L1", "L2")
PAIRLITE_INPUT_MODES = ("q_only", "y_only", "q_y")
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.I)


@dataclass
class PairLiteProfile:
    fit_seconds: float
    train_rows: int
    feature_dim: int
    train_nnz: int
    peak_ram_mb: float | None = None


class PairLiteCPUDetector:
    """CPU-only sparse detector for E1-CPU-v5.

    Feature contract:
    q-only = [phi_q(q), zeros_like(phi_y), zeros_like(phi_xy)]
    y-only = [zeros_like(phi_q), phi_y(y), zeros_like(phi_xy)]
    q+y    = [phi_q(q), phi_y(y), phi_xy(q,y)]
    """

    def __init__(
        self,
        level: str = "L2",
        alpha: float = 3e-5,
        l1_ratio: float = 0.05,
        max_iter: int = 40,
        seed: int = 20260726,
        word_features: int = 100000,
        char_features: int = 100000,
        hash_features: int = 262144,
        top_k_cross: int = 12,
    ):
        if level not in PAIRLITE_LEVELS:
            raise ValueError(f"unknown PairLite level: {level}")
        self.level = level
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.seed = seed
        self.hash_features = hash_features
        self.top_k_cross = top_k_cross
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=word_features,
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=char_features,
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.hasher = FeatureHasher(n_features=hash_features, input_type="string", alternate_sign=False, dtype=np.float32)
        self.classifier = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            class_weight="balanced",
            average=True,
            max_iter=max_iter,
            tol=1e-4,
            random_state=seed,
        )
        self.profile: PairLiteProfile | None = None

    def fit(self, rows: list[dict], labels: list[str], mode: str = "q_y") -> "PairLiteCPUDetector":
        corpus = [str(row.get("user_query", "")) for row in rows] + [str(row.get("target_model_answer", "")) for row in rows]
        self.word_vectorizer.fit(corpus)
        if self.level in {"L1", "L2"}:
            self.char_vectorizer.fit(corpus)
        started = time.perf_counter()
        features = self.features(rows, mode)
        self.classifier.fit(features, np.asarray([label == "unsafe" for label in labels], dtype=np.int8))
        self.profile = PairLiteProfile(
            fit_seconds=time.perf_counter() - started,
            train_rows=len(rows),
            feature_dim=features.shape[1],
            train_nnz=int(features.nnz),
        )
        return self

    def predict_proba(self, rows: list[dict], mode: str = "q_y") -> np.ndarray:
        return self.classifier.predict_proba(self.features(rows, mode))[:, 1]

    def features(self, rows: list[dict], mode: str = "q_y"):
        mode = "q_y" if mode == "q+y" else mode
        if mode not in PAIRLITE_INPUT_MODES:
            raise ValueError(f"unknown input mode: {mode}")
        queries = [str(row.get("user_query", "")) for row in rows]
        answers = [str(row.get("target_model_answer", "")) for row in rows]
        q = self.unary_q_features(queries)
        y = self.unary_y_features(answers)
        xy = self.cross_qy_features(queries, answers)
        zero_q = csr_matrix(q.shape, dtype=np.float32)
        zero_y = csr_matrix(y.shape, dtype=np.float32)
        zero_xy = csr_matrix(xy.shape, dtype=np.float32)
        if mode == "q_only":
            return hstack([q, zero_y, zero_xy], format="csr", dtype=np.float32)
        if mode == "y_only":
            return hstack([zero_q, y, zero_xy], format="csr", dtype=np.float32)
        return hstack([q, y, xy], format="csr", dtype=np.float32)

    def unary_q_features(self, queries: list[str]):
        return self._unary_features(queries)

    def unary_y_features(self, answers: list[str]):
        return self._unary_features(answers)

    def cross_qy_features(self, queries: list[str], answers: list[str]):
        q_word = self.word_vectorizer.transform(queries)
        y_word = self.word_vectorizer.transform(answers)
        blocks = [abs(q_word - y_word), q_word.multiply(y_word), csr_matrix(cross_qy_features(queries, answers), dtype=np.float32)]
        if self.level == "L2":
            blocks.append(self.hasher.transform(self._hashed_cross_tokens(q, y) for q, y in zip(queries, answers)))
        return hstack(blocks, format="csr", dtype=np.float32)

    def _unary_features(self, texts: list[str]):
        blocks = [self.word_vectorizer.transform(texts), csr_matrix(unary_text_features(texts), dtype=np.float32)]
        if self.level in {"L1", "L2"}:
            blocks.insert(1, self.char_vectorizer.transform(texts))
        return hstack(blocks, format="csr", dtype=np.float32)

    def _hashed_cross_tokens(self, query: str, answer: str) -> list[str]:
        q_tokens = _top_tokens(query, self.top_k_cross)
        y_tokens = _top_tokens(answer, self.top_k_cross)
        return [f"q={q}|y={y}" for q in q_tokens for y in y_tokens]


def _top_tokens(text: str, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for token in _TOKEN_RE.findall(text.lower()):
        counts[token] = counts.get(token, 0) + 1
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]
