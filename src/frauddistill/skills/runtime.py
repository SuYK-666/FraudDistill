"""Skill runtime facade (guide sections 12-13): wires registry + router +
composer and provides per-agent composed prompts + traces.

The runtime is instruction-only: it never executes skill content, never loads
scripts, never touches the network, and never modifies files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from frauddistill.skills.registry import SkillRegistry, check_expected_skills, registry_digest
from frauddistill.skills.router import SkillRouter, MAX_SKILL_CHARS
from frauddistill.skills.prompt_composer import SkillPromptComposer, split_schema_instruction
from frauddistill.skills.schemas import AgentSkillTrace, SkillSelection
from frauddistill.skills.trace import build_skill_trace

# Agent system prompts are imported lazily to avoid import cycles.
_AGENT_PROMPTS: dict[str, str] | None = None


def _default_agent_prompts() -> dict[str, str]:
    global _AGENT_PROMPTS
    if _AGENT_PROMPTS is None:
        from frauddistill.agents import (  # noqa: F401
            arbiter_agent,
            fraud_assistance_agent,
            refusal_quality_agent,
            relevance_agent,
        )

        _AGENT_PROMPTS = {
            "fraud": fraud_assistance_agent.SYSTEM_PROMPT,
            "refusal": refusal_quality_agent.SYSTEM_PROMPT,
            "context": relevance_agent.SYSTEM_PROMPT,
            "arbiter": arbiter_agent.SYSTEM_PROMPT,
        }
    return _AGENT_PROMPTS


def _agent_base_and_schema(
    agent_name: str, base_prompts: dict[str, str] | None = None
) -> tuple[str, str]:
    prompts = base_prompts or _default_agent_prompts()
    base, schema = split_schema_instruction(prompts[agent_name])
    return base, schema


class SkillRuntime:
    def __init__(
        self,
        registry: SkillRegistry,
        router: SkillRouter,
        composer: SkillPromptComposer,
        *,
        content_harm: bool = True,
        base_prompts: dict[str, str] | None = None,
    ):
        self.registry = registry
        self.router = router
        self.composer = composer
        self.content_harm = content_harm
        # base_prompts override: C2 uses task-aligned prompt variants
        # (response-content-harm head + hard-exit/soft-caution split).
        self.base_prompts = base_prompts

    def selection(self, agent_name: str, sample: dict[str, Any], *, task_mode: str | None = None) -> SkillSelection:
        return self.router.select(
            agent_name=agent_name,
            sample=sample,
            content_harm=self.content_harm,
            task_mode=task_mode,
        )

    def compose_system(self, agent_name: str, selection: SkillSelection) -> str:
        base, schema = _agent_base_and_schema(agent_name, self.base_prompts)
        return self.composer.compose(
            base_system_prompt=base,
            selection=selection,
            schema_instruction=schema,
        )

    def user_message(self, sample: dict[str, Any]) -> str:
        return self.composer.compose_user(sample)

    def trace(self, selections: dict[str, SkillSelection]) -> dict:
        agents: dict[str, AgentSkillTrace] = {}
        for name, sel in selections.items():
            digests = {
                skill: self.registry.get(skill).digest for skill in sel.selected
            }
            agents[name] = AgentSkillTrace(
                selected=sel.selected,
                digests=digests,
                reasons=sel.reasons,
                total_chars=sel.total_chars,
            )
        return build_skill_trace(
            self.registry, self.router.version(), agents
        )


def build_skill_runtime(
    skills_root: Path | None = None,
    *,
    content_harm: bool = True,
    strict: bool = True,
    base_prompts: dict[str, str] | None = None,
) -> SkillRuntime:
    if skills_root is None:
        skills_root = Path(__file__).resolve().parents[3] / "skills"
    registry = SkillRegistry(skills_root)
    registry.discover()
    missing = check_expected_skills(registry)
    if strict and missing:
        raise RuntimeError(f"Missing skills: {missing}")
    router = SkillRouter(registry, max_chars_by_agent=dict(MAX_SKILL_CHARS))
    composer = SkillPromptComposer(registry)
    return SkillRuntime(
        registry, router, composer,
        content_harm=content_harm,
        base_prompts=base_prompts,
    )
