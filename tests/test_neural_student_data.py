# -*- coding: utf-8 -*-
"""Unit tests for neural-student data and losses (guide 4.2, 11, 15)."""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pytest
import torch

from frauddistill.student.dataset import (LABEL_TO_ID, build_neural_examples, map_gold_type,
                                          neural_input_text, sample_weight, teacher_distribution,
                                          blend_target_distribution)
from frauddistill.student.losses import FraudDistillLoss


def test_label_mapping():
    assert map_gold_type("unsafe", "refusal_failure", "direct_fraud", "fraud_assistance") == "refusal_failure"
    assert map_gold_type("unsafe", "unsafe", "direct_fraud", "fraud_assistance") == "fraud_assistance"
    assert map_gold_type("unsafe", "unsafe", "partial_leakage", "refusal_failure") == "refusal_failure"
    assert map_gold_type("safe", "unsafe", "direct_fraud", "fraud_assistance") == "safe"
    assert map_gold_type("unsafe", "unsafe", "unknown_sub", "over_refusal") == "over_refusal"


def test_input_template_has_no_metadata():
    text = neural_input_text("q", "a", context="ctx")
    assert "[QUERY]" in text and "[ANSWER]" in text and "[CONTEXT]" in text
    for forbidden in ("benchmark", "gold", "teacher", "dataset", "target model"):
        assert forbidden not in text.lower()


def test_teacher_distribution_mass_and_type():
    d = teacher_distribution(0.82, "refusal_failure")
    assert abs(sum(d) - 1.0) < 1e-9
    assert d[0] == pytest.approx(0.18)
    assert d[LABEL_TO_ID["refusal_failure"]] == pytest.approx(0.82)
    d_safe = teacher_distribution(0.3, "safe")
    assert d_safe[0] == pytest.approx(1.0)


def test_blend_gold_priority():
    d = blend_target_distribution("unsafe", LABEL_TO_ID["fraud_assistance"], 0.9, "fraud_assistance", "official", "high")
    assert abs(sum(d) - 1.0) < 1e-9
    assert d[LABEL_TO_ID["fraud_assistance"]] > 0.8   # gold dominates


def test_sample_weight_rules():
    w_high = sample_weight("official", 0.98, 1.0, [], "direct_fraud")
    w_low = sample_weight("procedural_weak", 0.5, 0.5, ["conflict"], "direct_fraud")
    assert w_high > w_low
    w_slice = sample_weight("official", 0.98, 1.0, [], "partial_leakage")
    assert w_slice > sample_weight("official", 0.98, 1.0, [], "direct_fraud")


def test_build_neural_examples_fields():
    rows = [{"id": "x1", "group_id": "g1", "user_query": "q", "target_model_answer": "a",
             "gold_label": "unsafe", "gold_type": "unsafe", "subtype": "direct_fraud",
             "teacher_label": "unsafe", "teacher_score": 0.9, "teacher_type": "fraud_assistance",
             "teacher_confidence": 0.9, "agent_agreement": 1.0, "confidence_tier": "high",
             "conflict_flags": [], "gold_source": "official", "pair_id": "p1", "language": "en",
             "source": "fraudr1_all", "target_model": "m"}]
    ex = build_neural_examples(rows)[0]
    assert ex["gold_type_id"] == 1
    assert ex["pair_id"] == "p1"
    assert len(ex["teacher_distribution"]) == 4
    assert ex["sample_weight"] > 0


def test_fraud_distill_loss_components():
    loss_fn = FraudDistillLoss(lambda_gold=0.65, lambda_soft=0.25, lambda_pair=0.10, temperature=1.5)
    torch.manual_seed(0)
    logits = torch.randn(4, 4)
    gold = torch.tensor([0, 1, 2, 3])
    tdist = torch.tensor([[1.0, 0, 0, 0], [0, 0.8, 0.2, 0], [0.1, 0, 0.9, 0], [0, 0, 0, 1.0]])
    sw = torch.ones(4)
    pairs = [{"unsafe_idx": 1, "safe_idx": 0}, {"unsafe_idx": 2, "safe_idx": 3}]
    total, comps = loss_fn(logits, gold, tdist, sw, pairs)
    assert set(comps) == {"loss_gold", "loss_soft", "loss_pair", "loss_total"}
    assert comps["loss_total"] == pytest.approx(total.item())
    assert comps["loss_gold"] > 0 and comps["loss_soft"] > 0
    assert comps["loss_pair"] >= 0


def test_gold_only_has_no_soft_pair():
    loss_fn = FraudDistillLoss(lambda_gold=1.0, lambda_soft=0.0, lambda_pair=0.0)
    logits = torch.randn(2, 4)
    gold = torch.tensor([0, 1])
    tdist = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    sw = torch.ones(2)
    total, comps = loss_fn(logits, gold, tdist, sw, None)
    assert comps["loss_soft"] == 0.0 and comps["loss_pair"] == 0.0
