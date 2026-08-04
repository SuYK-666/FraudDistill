# -*- coding: utf-8 -*-
"""Multi-task distillation losses (guide 15) + legacy linear losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_weights(conflict: bool, agree_weight: float = 0.3, conflict_weight: float = 0.1) -> float:
    return conflict_weight if conflict else agree_weight


def multi_head_distillation_loss(outputs, batch, weights: dict | None = None):
    """Tensor loss; gold labels are always retained even when the teacher disagrees."""
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


# ---------------------------------------------------------------------------
# Neural student loss (guide 15)
# ---------------------------------------------------------------------------
class FraudDistillLoss:
    """L = lambda_gold * L_gold + lambda_soft * L_soft + lambda_pair * L_pair.

    Guide 15.1-15.5: unified 4-class softmax; gold CE (class-weighted),
    temperature-scaled KL against the teacher distribution, and a pairwise
    margin loss over context-flip / matched hard pairs (p_unsafe ordering).
    """

    def __init__(self, lambda_gold: float = 0.65, lambda_soft: float = 0.25, lambda_pair: float = 0.10,
                 temperature: float = 1.5, pair_margin: float = 0.20,
                 class_weights: list[float] | None = None, slice_weights: dict | None = None):
        self.lambda_gold = lambda_gold
        self.lambda_soft = lambda_soft
        self.lambda_pair = lambda_pair
        self.temperature = temperature
        self.pair_margin = pair_margin
        self.class_weights = torch.tensor(class_weights or [1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.slice_weights = slice_weights or {}

    def __call__(self, logits, gold_labels, teacher_distribution, sample_weight,
                 pair_metadata=None):
        device = logits.device
        cw = self.class_weights.to(device)
        sw = sample_weight.to(device).detach()

        # gold CE with sample weights
        gold = F.cross_entropy(logits, gold_labels, weight=cw, reduction="none")
        loss_gold = (gold * sw).sum() / sw.sum().clamp_min(1.0)

        loss_soft = torch.zeros((), device=device)
        if self.lambda_soft > 0:
            student_log_prob = F.log_softmax(logits / self.temperature, dim=-1)
            teacher_prob = teacher_distribution.to(device).detach()
            kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
            loss_soft = (kl * sw).sum() / sw.sum().clamp_min(1.0) * (self.temperature ** 2)

        loss_pair = torch.zeros((), device=device)
        if self.lambda_pair > 0 and pair_metadata is not None and len(pair_metadata) > 0:
            p_unsafe = 1.0 - F.softmax(logits, dim=-1)[:, 0]
            pairs = [pm for pm in pair_metadata if pm is not None]
            if pairs:
                unsafe_idx = torch.tensor([p["unsafe_idx"] for p in pairs], dtype=torch.long, device=device)
                safe_idx = torch.tensor([p["safe_idx"] for p in pairs], dtype=torch.long, device=device)
                loss_pair = torch.relu(self.pair_margin - p_unsafe[unsafe_idx] + p_unsafe[safe_idx]).mean()

        loss_total = self.lambda_gold * loss_gold + self.lambda_soft * loss_soft + self.lambda_pair * loss_pair
        return loss_total, {
            "loss_gold": loss_gold.detach(),
            "loss_soft": loss_soft.detach(),
            "loss_pair": loss_pair.detach(),
            "loss_total": loss_total.detach(),
        }
