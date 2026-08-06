# -*- coding: utf-8 -*-
"""Strict schema tests (guide 28.1, section 6)."""
import pytest
from pydantic import ValidationError

from frauddistill.exp2_static_repair.schemas import (
    StrictContextEvidence,
    StrictFraudEvidence,
    StrictRefusalEvidence,
    finish_reason_status,
    reject_suspicious_empty_evidence,
    require_fields,
)

FRAUD_OK = {
    "fraud_domain": "phishing",
    "direct_actionability": 0.1,
    "harmful_goal_alignment": 0.2,
    "trust_facilitation": 0.0,
    "risk_downplaying": 0.0,
    "fraud_cost_reduction": 0.0,
    "fraud_probability": 0.3,
    "evidence_spans": ["x"],
    "counter_evidence_spans": [],
    "concise_reason": "some reason",
}

REFUSAL_OK = {
    "should_refuse_probability": 0.5,
    "explicit_refusal": 0.8,
    "refusal_strength": 0.7,
    "partial_compliance": 0.0,
    "task_completion": 0.0,
    "operational_detail_leakage": 0.0,
    "leakage_before_refusal": 0.0,
    "leakage_after_refusal": 0.0,
    "safe_redirection": 0.9,
    "risk_warning": 0.6,
    "prompt_is_benign": 0.0,
    "unnecessary_refusal": 0.0,
    "answer_delivers_safe_utility": 1.0,
    "refusal_failure_probability": 0.1,
    "over_refusal_probability": 0.0,
    "refusal_state": "S1_clear_refusal",
    "evidence_spans": ["x"],
    "concise_reason": "clear refusal",
}

CONTEXT_OK = {
    "query_answer_alignment": 1.0,
    "harmful_goal_alignment": 0.0,
    "task_completion": 0.0,
    "use_mode": "safe_redirection",
    "quoted_content_only": 0.0,
    "analysis_or_education": 0.0,
    "safe_redirection": 0.9,
    "active_endorsement": 0.0,
    "evidence_spans": [],
    "concise_reason": "redirected safely",
}


def test_empty_dict_rejected():
    with pytest.raises(ValidationError):
        StrictFraudEvidence.model_validate({})
    with pytest.raises(ValidationError):
        StrictRefusalEvidence.model_validate({})
    with pytest.raises(ValidationError):
        StrictContextEvidence.model_validate({})


def test_missing_required_field_rejected():
    bad = dict(FRAUD_OK)
    bad.pop("evidence_spans")
    with pytest.raises(ValidationError):
        StrictFraudEvidence.model_validate(bad)


def test_extra_field_rejected():
    bad = dict(FRAUD_OK)
    bad["unexpected"] = 1.0
    with pytest.raises(ValidationError):
        StrictFraudEvidence.model_validate(bad)


def test_require_fields_helper():
    require_fields(FRAUD_OK, {"fraud_probability", "concise_reason"})
    with pytest.raises(ValueError):
        require_fields({}, {"fraud_probability"})


def test_all_zero_without_reason_rejected():
    empty = dict(FRAUD_OK)
    empty.update({
        "fraud_probability": 0.0, "direct_actionability": 0.0,
        "harmful_goal_alignment": 0.0, "trust_facilitation": 0.0,
        "risk_downplaying": 0.0, "fraud_cost_reduction": 0.0,
        "evidence_spans": [], "counter_evidence_spans": [], "concise_reason": "ok",
    })
    model = StrictFraudEvidence.model_validate(empty)
    # valid structurally, but suspicious-empty check must fire
    with pytest.raises(ValueError):
        reject_suspicious_empty_evidence(model)


def test_finish_reason_length_not_accepted():
    status, retry = finish_reason_status("length")
    assert status == "truncated"
    assert retry is True
    status, retry = finish_reason_status("stop")
    assert status == "ok"
    assert retry is False
    status, retry = finish_reason_status("insufficient_system_resource")
    assert status == "provider_interrupted"
    assert retry is True
