# -*- coding: utf-8 -*-
"""Final-pilot guide section 2: forced response-content-harm loading in
general_response_safety task mode + Fake Job / hard-exit C2 prompt anchors."""
from __future__ import annotations

from pathlib import Path

import pytest

from frauddistill.skills.c2_prompts import build_c2_prompts, patch_fraud
from frauddistill.skills.registry import SkillRegistry, check_expected_skills
from frauddistill.skills.runtime import build_skill_runtime
from frauddistill.skills.router import SkillRouter

REPO = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO / "skills"


def test_response_content_harm_skill_registered():
    reg = SkillRegistry(SKILLS_ROOT).discover()
    assert "response-content-harm" in reg
    assert "response-content-harm" not in check_expected_skills(reg)


def test_general_response_safety_forces_content_harm():
    rt = build_skill_runtime(SKILLS_ROOT, content_harm=True, strict=True)
    # benign query+answer with no hazard signal: content-harm must still load
    sample = {
        "user_query": "How do I reset my router?",
        "target_model_answer": "Unplug it, wait ten seconds, plug it back in.",
        "turn_count": 1,
    }
    sel = rt.selection("refusal", sample, task_mode="general_response_safety")
    assert "response-content-harm" in sel.selected
    assert "request-policy-risk" in sel.selected
    assert "safe-context-disambiguation" in sel.selected


def test_general_response_safety_only_via_task_mode():
    rt = build_skill_runtime(SKILLS_ROOT, content_harm=True, strict=True)
    sample = {
        "user_query": "How do I reset my router?",
        "target_model_answer": "Unplug it, wait ten seconds, plug it back in.",
        "turn_count": 1,
    }
    sel = rt.selection("refusal", sample)
    # no hazard signal -> content-harm is not added without the task mode
    assert "response-content-harm" not in sel.selected


def test_router_skips_unknown_benchmark_name():
    """task_mode is abstract; benchmark names never enter the router."""
    reg = SkillRegistry(SKILLS_ROOT).discover()
    router = SkillRouter(reg)
    sample = {"user_query": "x", "target_model_answer": "y", "turn_count": 1}
    sel = router.select(agent_name="refusal", sample=sample, task_mode="general_response_safety")
    joined = " ".join(sel.reasons.values()).lower()
    assert "aegis" not in joined


def test_c2_fraud_fake_job_checks_present():
    prompts = build_c2_prompts()
    fraud = prompts["fraud"]
    assert "FAKE JOB POSTINGS" in fraud
    assert "off-platform recruitment flow" in fraud
    assert "hard_safety_exit" in fraud
    assert "in_role_cautious_continuation" in fraud
    assert "soft_caution_or_verification" in fraud


def test_c2_prompts_build_stable():
    prompts = build_c2_prompts()
    for agent in ("fraud", "refusal", "context", "arbiter"):
        assert prompts[agent]
    for agent in ("refusal", "arbiter"):
        assert "response_content_harm_score" in prompts[agent]