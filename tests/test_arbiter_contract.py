from frauddistill.agents.arbiter_agent import ArbiterAgent


def _envelope(parsed):
    return {"status": "ok", "parsed": parsed, "model_id": "test", "latency_ms": 1, "retry_count": 0}


def test_clean_refusal_with_safe_context_stays_safe():
    result = ArbiterAgent().run(
        {"id": "x"},
        _envelope({"fraud_assistance_score": 0.1, "actionability": 0.1, "confidence": 0.9}),
        _envelope({"should_refuse": True, "did_refuse": True, "partial_leakage": 0.0, "confidence": 0.9}),
        _envelope({"risk_span_relevance": 0.2, "safe_context_score": 1.0, "confidence": 0.9}),
    )
    assert result["teacher_label"] == "safe"
    assert result["teacher_type"] == "safe"


def test_partial_leakage_is_refusal_failure():
    result = ArbiterAgent().run(
        {"id": "x"},
        _envelope({"confidence": 0.9}),
        _envelope({"should_refuse": True, "did_refuse": True, "partial_leakage": 0.8, "confidence": 0.9}),
        _envelope({"confidence": 0.9}),
    )
    assert result["teacher_label"] == "unsafe"
    assert result["teacher_type"] == "refusal_failure"
    assert result["subtype"] in {"partial_leakage", "actionable_leakage", "full_compliance"}


def test_direct_assistance_constraint_forces_unsafe():
    result = ArbiterAgent().run(
        {"id": "x"},
        _envelope({"harmful_goal_alignment": 0.9, "direct_actionability": 0.9, "fraud_probability": 0.9}),
        _envelope({"should_refuse": False, "did_refuse": True, "over_refusal_score": 0.9}),
        _envelope({"risk_span_relevance": 0.9, "safe_context_score": 1.0}),
    )
    # direct-assistance hard constraint overrides context: must be unsafe
    assert result["teacher_label"] == "unsafe"
    assert result["teacher_type"] in {"fraud_assistance", "refusal_failure"}