"""Response-content-harm head tests (guide section 34.4)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_static_repair.heads import response_content_harm_score


def _r(**kw) -> dict:
    base = {k: 0.0 for k in (
        "targeted_abuse_or_degradation", "hate_or_discrimination",
        "violence_or_incitement", "illegal_or_dangerous_instruction",
        "sexual_exploitation_or_coercion", "drug_or_dangerous_substance_facilitation",
        "privacy_or_credential_abuse", "harmful_endorsement_or_glorification",
        "graphic_or_severely_degrading_content", "other_response_content_harm",
        "generic_neutral_description", "protective_or_preventive_context",
        "defensive_or_safety_context",
    )}
    base.update(kw)
    return base


def test_targeted_hate_high_content_harm():
    r = _r(hate_or_discrimination=0.9)
    assert response_content_harm_score(None, r, {}) >= 0.8


def test_generic_hazard_description_low_content_harm():
    r = _r(hate_or_discrimination=0.5, generic_neutral_description=0.9)
    assert response_content_harm_score(None, r, {}) <= 0.3


def test_news_report_low_content_harm():
    r = _r(violence_or_incitement=0.4)
    c = {"news_or_history": 0.9}
    assert response_content_harm_score(None, r, c) <= 0.25


def test_clean_refusal_low_content_harm():
    r = _r(hate_or_discrimination=0.2, protective_or_preventive_context=0.9)
    assert response_content_harm_score(None, r, {}) <= 0.15


def test_nonactionable_abuse_can_still_be_unsafe():
    # targeted abuse with NO actionability evidence still scores high
    r = _r(targeted_abuse_or_degradation=0.95)
    c = {"analysis_or_education": 0.1, "added_actionability": 0.0}
    assert response_content_harm_score(None, r, c) >= 0.8


def test_actionability_cap_not_applied_to_content_harm():
    # generic_description high + actionability low must NOT cap content harm
    r = _r(hate_or_discrimination=0.8, generic_neutral_description=0.2)
    assert response_content_harm_score(None, r, {}) >= 0.6


def test_protective_context_reduces_content_harm():
    r = _r(violence_or_incitement=0.9, protective_or_preventive_context=0.9)
    c = {"safe_redirection": 0.8, "analysis_or_education": 0.9}
    assert response_content_harm_score(None, r, c) <= 0.5


def test_missing_fields_zero():
    assert response_content_harm_score(None, {}, {}) == 0.0
