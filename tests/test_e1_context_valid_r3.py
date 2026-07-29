from frauddistill.experiments.run_exp1_single_judge_api import _build_prompt, _normalize


def test_r3_prompt_has_no_explicit_ablation_mode() -> None:
    row = {"user_query": "How do I spot phishing?", "target_model_answer": "Check the sender."}
    prompt = _build_prompt(row, "q_only")
    assert "[ABLATION_MODE]" not in prompt
    assert "q_only" not in prompt
    assert "<NOT_PROVIDED_IN_THIS_ABLATION>" in prompt


def test_r3_schema_does_not_require_evidence_basis() -> None:
    pred = _normalize({"p_unsafe": 0.2, "pred_label": "safe", "pred_type": "safe"})
    assert pred["pred_label"] == "safe"
    assert "evidence_basis" not in pred
