"""Single EvaluationFrame object (guide section 8).

Every metric/statistic accepts only an EvaluationFrame so that N+, Recall,
FPR and Macro-F1 are always recomputable from one confusion matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvaluationFrame:
    benchmark: str
    track: str
    sample_ids: np.ndarray
    group_ids: np.ndarray
    y_true_binary: np.ndarray
    y_pred_binary: np.ndarray
    y_score: np.ndarray | None = None

    y_true_type: np.ndarray | None = None
    y_pred_type: np.ndarray | None = None

    prediction_digest: str = ""
    gold_digest: str = ""
    manifest_digest: str = ""

    score_head: str = ""


def validate_frame(frame: EvaluationFrame) -> None:
    """Structural validation; raises AssertionError on any inconsistency."""
    n = len(frame.sample_ids)
    assert n > 0, "empty frame"
    assert len(np.unique(frame.sample_ids)) == n, "duplicate sample ids"
    assert len(frame.group_ids) == n
    assert len(frame.y_true_binary) == n
    assert len(frame.y_pred_binary) == n
    assert set(np.unique(frame.y_true_binary)) <= {0, 1}
    assert set(np.unique(frame.y_pred_binary)) <= {0, 1}
    if frame.y_score is not None:
        assert len(frame.y_score) == n
        assert np.isfinite(frame.y_score).all(), "non-finite scores"
        assert ((0.0 <= frame.y_score) & (frame.y_score <= 1.0)).all(), "scores out of [0,1]"
    if frame.y_true_type is not None:
        assert len(frame.y_true_type) == n
    if frame.y_pred_type is not None:
        assert len(frame.y_pred_type) == n


def build_frame(
    *,
    benchmark: str,
    track: str,
    sample_ids: list[str],
    group_ids: list[str],
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float] | None = None,
    y_true_type: list[str] | None = None,
    y_pred_type: list[str] | None = None,
    prediction_digest: str = "",
    gold_digest: str = "",
    manifest_digest: str = "",
    score_head: str = "",
) -> EvaluationFrame:
    frame = EvaluationFrame(
        benchmark=benchmark,
        track=track,
        sample_ids=np.asarray(sample_ids),
        group_ids=np.asarray(group_ids),
        y_true_binary=np.asarray(y_true, dtype=int),
        y_pred_binary=np.asarray(y_pred, dtype=int),
        y_score=np.asarray(y_score, dtype=float) if y_score is not None else None,
        y_true_type=np.asarray(y_true_type) if y_true_type is not None else None,
        y_pred_type=np.asarray(y_pred_type) if y_pred_type is not None else None,
        prediction_digest=prediction_digest,
        gold_digest=gold_digest,
        manifest_digest=manifest_digest,
        score_head=score_head,
    )
    validate_frame(frame)
    return frame
