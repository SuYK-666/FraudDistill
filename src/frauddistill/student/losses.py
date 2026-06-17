from __future__ import annotations


def distillation_weights(conflict: bool, agree_weight: float = 0.3, conflict_weight: float = 0.1) -> float:
    return conflict_weight if conflict else agree_weight


def total_loss_value(gold_loss: float, score_loss: float = 0.0, type_loss: float = 0.0, span_loss: float = 0.0, weights: dict | None = None) -> float:
    weights = weights or {"gold": 1.0, "teacher_score": 0.3, "teacher_type": 0.3, "teacher_span": 0.2}
    return (
        weights.get("gold", 1.0) * gold_loss
        + weights.get("teacher_score", 0.0) * score_loss
        + weights.get("teacher_type", 0.0) * type_loss
        + weights.get("teacher_span", 0.0) * span_loss
    )
