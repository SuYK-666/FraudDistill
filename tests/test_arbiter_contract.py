from frauddistill.agents.arbiter_agent import ArbiterAgent


def _envelope(parsed):
    return {"status": "ok", "parsed": parsed, "model_id": "test", "latency_ms": 1, "retry_count": 0}


def test_safe_context_and_over_refusal_are_not_primary_unsafe():
    result = ArbiterAgent().run({"id": "x"}, _envelope({"fraud_assistance_score": .9, "actionability": .9, "confidence": .9}), _envelope({"should_refuse": False, "did_refuse": True, "over_refusal_score": .9, "confidence": .9}), _envelope({"risk_span_relevance": .9, "safe_context_score": 1.0, "confidence": .9}))
    assert result["teacher_label"] == "safe"


def test_partial_leakage_is_unsafe():
    result = ArbiterAgent().run({"id": "x"}, _envelope({"confidence": .9}), _envelope({"should_refuse": True, "did_refuse": True, "partial_leakage": .8, "confidence": .9}), _envelope({"confidence": .9}))
    assert result["teacher_label"] == "unsafe"
    assert result["teacher_type"] == "partial_leakage"
