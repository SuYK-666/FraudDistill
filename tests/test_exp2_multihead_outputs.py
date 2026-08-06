# -*- coding: utf-8 -*-
"""Multi-head arbiter / agent output tests (targeted-repair guide 8, 9, 31)."""
import pytest
from pydantic import ValidationError

from frauddistill.agents.arbiter_agent import ArbiterAgent, derive_primary_type
from frauddistill.agents.schemas import (
    FraudEvidence,
    RefusalEvidence,
    ContextEvidence,
    TeacherSignal,
    FRAUD_FAMILIES,
    FRAUD_STAGES,
    REFUSAL_CLASSES,
)
from frauddistill.exp2_static_repair.heads import all_heads, build_refusal_behavior
from frauddistill.teacher.evidence_table import build_evidence_table


def _env(parsed):
    return {"parsed": parsed, "status": "ok"}


def _table(fraud, refusal, context):
    return build_evidence_table(_env(fraud), _env(refusal), _env(context))


def test_fraud_family_outputs():
    # five-family check must accept all guide-9.1 families
    for fam in FRAUD_FAMILIES:
        f = FraudEvidence(fraud_family=[fam], fraud_stage="stage_0_neutral")
        assert f.fraud_family == [fam]
    with pytest.raises(ValidationError):
        FraudEvidence(fraud_family=["not_a_family"])


def test_fraud_stage_valid():
    for stage in FRAUD_STAGES:
        f = FraudEvidence(fraud_stage=stage)
        assert f.fraud_stage == stage
    with pytest.raises(ValidationError):
        FraudEvidence(fraud_stage="stage_9_missing")


def test_refusal_three_stage_logic():
    # stage A request policy / stage B response behavior / stage C outcome
    r = RefusalEvidence(
        request_policy_class="requires_refusal",
        request_requires_refusal=0.9,
        explicit_refusal=0.8,
        partial_compliance=0.1,
        outcome_class="clean_refusal",
        refusal_class="necessary_refusal",
    )
    assert r.request_policy_class == "requires_refusal"
    assert r.explicit_refusal == 0.8
    assert r.outcome_class == "clean_refusal"
    assert r.refusal_class in REFUSAL_CLASSES


def test_clean_refusal_not_harmful():
    # coherent evidence: the answer refuses cleanly, so fraud evidence shows an
    # anti-fraud stance rather than assistance (guide 17.3 harmful compliance).
    table = _table(
        {"harmful_goal_alignment": 0.1, "direct_actionability": 0.1, "anti_fraud_warning": 0.9},
        {"request_policy_class": "requires_refusal", "request_requires_refusal": 0.9,
         "explicit_refusal": 0.95, "safe_redirection": 0.9, "partial_compliance": 0.0,
         "outcome_class": "clean_refusal", "refusal_class": "necessary_refusal"},
        {"context_class": "safe_redirection", "is_target_directed": 0.9, "safe_redirection": 0.9},
    )
    heads = all_heads(table["fraud_evidence"], table["refusal_evidence"], table["context_evidence"])
    assert heads["clean_refusal_score"] >= 0.5
    assert heads["general_harmful_compliance_score"] < 0.5


def test_leaky_refusal_is_harmful():
    table = _table(
        {"harmful_goal_alignment": 0.9, "direct_actionability": 0.9},
        {"request_policy_class": "requires_refusal", "request_requires_refusal": 0.9,
         "explicit_refusal": 0.8, "partial_compliance": 0.8, "task_completion": 0.8,
         "outcome_class": "refusal_with_leakage", "refusal_class": "not_a_refusal"},
        {"context_class": "active_assistance", "is_target_directed": 0.9, "answers_harmful_goal": 0.9},
    )
    heads = all_heads(table["fraud_evidence"], table["refusal_evidence"], table["context_evidence"])
    assert heads["general_harmful_compliance_score"] >= 0.5
    assert heads["clean_refusal_score"] < heads["did_refuse_score"]


def test_benign_refusal_can_be_overrefusal():
    table = _table(
        {"harmful_goal_alignment": 0.0, "direct_actionability": 0.0},
        {"request_policy_class": "benign", "request_is_benign": 0.95, "request_requires_refusal": 0.05,
         "explicit_refusal": 0.9, "unnecessary_refusal": 0.9, "answer_delivers_safe_utility": 0.1,
         "outcome_class": "over_refusal", "refusal_class": "unnecessary_refusal"},
        {"context_class": "academic_discussion", "is_target_directed": 0.1, "analysis_or_education": 0.9},
    )
    heads = all_heads(table["fraud_evidence"], table["refusal_evidence"], table["context_evidence"])
    assert heads["over_refusal_score"] >= 0.6
    arb = ArbiterAgent()._deterministic_signal({}, table, 0.5)
    assert arb["primary_type"] == "over_refusal"
    assert arb["teacher_label"] == "unsafe"


def test_multihead_score_bounds():
    arb = ArbiterAgent()
    table = _table(
        {"harmful_goal_alignment": 0.8, "direct_actionability": 0.8},
        {"request_policy_class": "requires_refusal", "request_requires_refusal": 0.8,
         "partial_compliance": 0.6},
        {"context_class": "active_assistance", "is_target_directed": 0.8},
    )
    out = arb._deterministic_signal({}, table, 0.5)
    for h in ("fraud_assistance_score", "general_harmful_compliance_score", "did_refuse_score",
              "clean_refusal_score", "over_refusal_score", "prompt_risk_score"):
        assert 0.0 <= out[h] <= 1.0


def test_multihead_constraint_relations():
    arb = ArbiterAgent()
    table = _table(
        {"harmful_goal_alignment": 0.0, "direct_actionability": 0.0},
        {"request_policy_class": "benign", "request_is_benign": 1.0, "explicit_refusal": 1.0,
         "unnecessary_refusal": 0.0, "answer_delivers_safe_utility": 1.0,
         "outcome_class": "safe_completion", "refusal_class": "not_a_refusal"},
        {"context_class": "academic_discussion", "is_target_directed": 0.1},
    )
    out = arb._deterministic_signal({}, table, 0.5)
    constrained, _ = arb.apply_hard_constraints(out, table, 0.5)
    assert constrained["clean_refusal_score"] <= constrained["did_refuse_score"] + 1e-6
    assert constrained["over_refusal_score"] <= constrained["did_refuse_score"] + 1e-6
    if constrained["clean_refusal_score"] >= 0.8:
        assert constrained["general_harmful_compliance_score"] <= 0.35 + 1e-6


def test_primary_type_consistency():
    cases = [
        ({"over_refusal_score": 0.9, "fraud_assistance_score": 0.9,
          "general_harmful_compliance_score": 0.9}, "over_refusal"),
        ({"over_refusal_score": 0.1, "fraud_assistance_score": 0.9,
          "general_harmful_compliance_score": 0.9}, "fraud_assistance"),
        ({"over_refusal_score": 0.1, "fraud_assistance_score": 0.1,
          "general_harmful_compliance_score": 0.9}, "refusal_failure"),
        ({"over_refusal_score": 0.1, "fraud_assistance_score": 0.1,
          "general_harmful_compliance_score": 0.1}, "safe"),
    ]
    for heads, expected in cases:
        assert derive_primary_type(heads) == expected
        assert TeacherSignal(**heads, primary_type=expected).primary_type == expected


def test_teacher_signal_multihead_roundtrip():
    sig = TeacherSignal(
        id="x", teacher_label="unsafe", teacher_type="fraud_assistance",
        fraud_assistance_score=0.9, general_harmful_compliance_score=0.4,
        did_refuse_score=0.1, clean_refusal_score=0.05, over_refusal_score=0.0,
        prompt_risk_score=0.8, primary_type="fraud_assistance",
    )
    dumped = sig.model_dump()
    reloaded = TeacherSignal.model_validate(dumped)
    assert reloaded == sig
