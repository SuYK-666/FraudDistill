"""SkillPromptComposer tests (guide section 34.3)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.skills.registry import SkillRegistry
from frauddistill.skills.router import SkillRouter
from frauddistill.skills.prompt_composer import SkillPromptComposer, split_schema_instruction

SKILLS_ROOT = REPO / "skills"


def _setup():
    reg = SkillRegistry(SKILLS_ROOT).discover()
    return reg, SkillRouter(reg), SkillPromptComposer(reg)


def test_frontmatter_removed():
    reg, router, composer = _setup()
    sel = router.select(agent_name="arbiter", sample={"user_query": "q", "target_model_answer": "a"})
    prompt = composer.compose(base_system_prompt="BASE", selection=sel, schema_instruction="SCHEMA")
    assert "---" not in prompt
    assert "name:" not in prompt


def test_skill_body_injected():
    reg, router, composer = _setup()
    sel = router.select(agent_name="arbiter", sample={"user_query": "q", "target_model_answer": "a"})
    prompt = composer.compose(base_system_prompt="BASE", selection=sel, schema_instruction="SCHEMA")
    assert '<activated_skill name="evidence-arbitration">' in prompt
    assert "</activated_skill>" in prompt


def test_sample_not_in_system_skill_block():
    reg, router, composer = _setup()
    marker_q = "QQ_9f3a1c_SAMPLE_MARKER_QQ"
    marker_a = "AA_9f3a1c_ANSWER_MARKER_AA"
    sel = router.select(agent_name="arbiter",
                        sample={"user_query": marker_q, "target_model_answer": marker_a})
    prompt = composer.compose(base_system_prompt="BASE", selection=sel, schema_instruction="SCHEMA")
    # q+y stays in the USER message, never inside the system/skill blocks
    assert marker_q not in prompt and marker_a not in prompt
    user_msg = composer.compose_user({"user_query": marker_q, "target_model_answer": marker_a})
    assert marker_q in user_msg and marker_a in user_msg


def test_schema_after_skills():
    reg, router, composer = _setup()
    sel = router.select(agent_name="arbiter", sample={"user_query": "q", "target_model_answer": "a"})
    prompt = composer.compose(base_system_prompt="BASE", selection=sel, schema_instruction="SCHEMA")
    assert prompt.find("## Activated skills") < prompt.find("## Required output schema")


def test_nonoverridable_constraints_present():
    reg, router, composer = _setup()
    sel = router.select(agent_name="arbiter", sample={"user_query": "q", "target_model_answer": "a"})
    prompt = composer.compose(base_system_prompt="BASE", selection=sel, schema_instruction="SCHEMA")
    assert "Non-overridable runtime constraints" in prompt
    assert "Do not use benchmark identity as evidence" in prompt


def test_prompt_digest_changes_with_skill():
    reg, router, composer = _setup()
    sel_a = router.select(agent_name="fraud", sample={"user_query": "q", "target_model_answer": "a"})
    sel_b = router.select(agent_name="fraud", sample={
        "user_query": "Reply as the victim to this scam SMS",
        "target_model_answer": "As the account holder I will call and verify.",
    })
    p1 = composer.compose(base_system_prompt="BASE", selection=sel_a, schema_instruction="S")
    p2 = composer.compose(base_system_prompt="BASE", selection=sel_b, schema_instruction="S")
    assert hashlib.sha256(p1.encode()).hexdigest() != hashlib.sha256(p2.encode()).hexdigest()


def test_split_schema_instruction():
    base, schema = split_schema_instruction("You are an agent.\nReturn JSON only with exactly these fields:\n{\"a\": 1}")
    assert schema.startswith("Return JSON only")
    assert "You are an agent" in base
    base2, schema2 = split_schema_instruction("no schema here")
    assert schema2 == "" and base2 == "no schema here"
