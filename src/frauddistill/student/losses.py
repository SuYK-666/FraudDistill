from __future__ import annotations


def distillation_weights(conflict: bool, agree_weight: float = 0.3, conflict_weight: float = 0.1) -> float:
    return conflict_weight if conflict else agree_weight


def multi_head_distillation_loss(outputs, batch, weights: dict | None = None):
    """Tensor loss; gold labels are always retained even when the teacher disagrees."""
    import torch
    import torch.nn.functional as F

    weights = {"soft": 0.3, "type": 0.2, "rank": 0.1, "relevance": 0.1, **(weights or {})}
    gold_loss = F.binary_cross_entropy_with_logits(outputs["binary_logits"], batch["gold_label"].float())
    teacher_weight = batch.get("teacher_weight", torch.ones_like(batch["gold_label"].float()))
    soft_raw = F.mse_loss(outputs["teacher_score_pred"], batch["teacher_score"].float(), reduction="none")
    soft_loss = (soft_raw * teacher_weight).sum() / teacher_weight.sum().clamp_min(1.0)
    type_raw = F.cross_entropy(outputs["type_logits"], batch["teacher_type"].long(), reduction="none")
    type_loss = (type_raw * teacher_weight).sum() / teacher_weight.sum().clamp_min(1.0)
    relevance_loss = F.mse_loss(outputs["relevance_pred"], batch.get("relevance", batch["gold_label"]).float())
    unsafe = batch["gold_label"].bool()
    safe = ~unsafe
    if unsafe.any() and safe.any():
        rank_loss = torch.relu(0.10 - outputs["unsafe_probability"][unsafe].mean() + outputs["unsafe_probability"][safe].mean())
    else:
        rank_loss = torch.zeros((), device=gold_loss.device)
    total = gold_loss + weights["soft"] * soft_loss + weights["type"] * type_loss + weights["rank"] * rank_loss + weights["relevance"] * relevance_loss
    return total, {"gold": gold_loss.detach(), "soft": soft_loss.detach(), "type": type_loss.detach(), "rank": rank_loss.detach(), "relevance": relevance_loss.detach()}


def total_loss_value(gold_loss: float, score_loss: float = 0.0, type_loss: float = 0.0, span_loss: float = 0.0, weights: dict | None = None) -> float:
    weights = weights or {"gold": 1.0, "teacher_score": 0.3, "teacher_type": 0.3, "teacher_span": 0.2}
    return weights.get("gold", 1.0) * gold_loss + weights.get("teacher_score", 0.0) * score_loss + weights.get("teacher_type", 0.0) * type_loss + weights.get("teacher_span", 0.0) * span_loss
