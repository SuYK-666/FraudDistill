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


# ---------------------------------------------------------------------------
# Final student loss (guide 12, final 1.5B student retrain)
# ---------------------------------------------------------------------------
class FinalDistillLoss:
    """L = CE4 + 0.30*binary + 0.30*w_t*KL + 0.05*pair (gold rows)
        L = 0.40*w_t*KL + 0.05*pair                 (teacher-only rows)

    Guide 12: Gold CE dominates; binary auxiliary BCE on p_safe; MAT soft KL
    at T=2.0 (KL x T^2); very light pair margin. Teacher reliability weight
    w_t = clip(0.5 + 0.5*teacher_confidence, 0.5, 1.0), halved when teacher
    disagrees with gold. Teacher-only rows (hard expansion) never get hard CE.
    """

    def __init__(self, lambda_binary: float = 0.30, lambda_kl: float = 0.30,
                 lambda_pair: float = 0.05, temperature: float = 2.0,
                 pair_margin: float = 0.20, class_weights: list[float] | None = None):
        self.lambda_binary = lambda_binary
        self.lambda_kl = lambda_kl
        self.lambda_pair = lambda_pair
        self.temperature = temperature
        self.pair_margin = pair_margin
        self.class_weights = torch.tensor(class_weights or [1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.uses_batch = True

    def __call__(self, logits, batch):
        device = logits.device
        cw = self.class_weights.to(device)
        gold_labels = batch["gold_type_id"]
        teacher_distribution = batch["teacher_distribution"]
        sw = batch["sample_weight"].to(device).detach()

        gold = F.cross_entropy(logits, gold_labels, weight=cw, reduction="none")
        # teacher-only rows: no hard CE and no binary term (guide 12.2)
        teacher_only = batch.get("teacher_only")
        if teacher_only is not None:
            tmask = teacher_only.to(device).float()
            gold = gold * (1.0 - tmask)
        # plain batch mean (guide 31: sample_weight is sampler-only; using it
        # again in the loss would double-count and, with sum<1 weights, shrink
        # the loss by orders of magnitude through the clamp)
        gold_denom = (1.0 - tmask).sum().clamp_min(1.0) if teacher_only is not None else float(gold.numel())
        loss_gold = gold.sum() / gold_denom

        # binary auxiliary: p_safe vs gold binary (guide 12.1)
        loss_binary = torch.zeros((), device=device)
        if self.lambda_binary > 0:
            gold_bin = batch["gold_binary"].to(device).float()
            p_safe = F.softmax(logits, dim=-1)[:, 0]
            bce = F.binary_cross_entropy(p_safe.clamp(1e-7, 1 - 1e-7), gold_bin, reduction="none")
            if teacher_only is not None:
                bce = bce * (1.0 - tmask)
            loss_binary = bce.sum() / gold_denom

        # MAT soft KL with teacher reliability weight (guide 12.3/12.4)
        loss_kl = torch.zeros((), device=device)
        if self.lambda_kl > 0:
            teacher_prob = teacher_distribution.to(device).detach()
            conf = batch.get("teacher_confidence")
            w_t = torch.ones_like(sw)
            if conf is not None:
                w_t = (0.5 + 0.5 * conf.to(device).float()).clamp(0.5, 1.0)
            disagree = batch.get("teacher_gold_agree")
            if disagree is not None:
                w_t = torch.where(disagree.to(device).bool(), w_t * 0.5, w_t)
            student_log_prob = F.log_softmax(logits / self.temperature, dim=-1)
            kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
            loss_kl = (kl * w_t).sum() / w_t.sum().clamp_min(1.0) * (self.temperature ** 2)

        # very light pair loss (guide 12.5)
        loss_pair = torch.zeros((), device=device)
        if self.lambda_pair > 0 and batch.get("pair_metadata"):
            p_unsafe = 1.0 - F.softmax(logits, dim=-1)[:, 0]
            pairs = [pm for pm in batch["pair_metadata"] if pm is not None]
            if pairs:
                unsafe_idx = torch.tensor([p["unsafe_idx"] for p in pairs], dtype=torch.long, device=device)
                safe_idx = torch.tensor([p["safe_idx"] for p in pairs], dtype=torch.long, device=device)
                loss_pair = torch.relu(self.pair_margin - p_unsafe[unsafe_idx] + p_unsafe[safe_idx]).mean()

        loss_total = (loss_gold
                      + self.lambda_binary * loss_binary
                      + self.lambda_kl * loss_kl
                      + self.lambda_pair * loss_pair)
        return loss_total, {
            "loss_gold": loss_gold.detach(),
            "loss_binary": loss_binary.detach(),
            "loss_kl": loss_kl.detach(),
            "loss_pair": loss_pair.detach(),
            "loss_total": loss_total.detach(),
        }
