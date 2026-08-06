"""Skill trace helpers (guide section 11): per-agent traces + registry digest."""
from __future__ import annotations

from frauddistill.skills.registry import SkillRegistry, registry_digest
from frauddistill.skills.schemas import AgentSkillTrace


def build_skill_trace(
    registry: SkillRegistry,
    router_version: str,
    agents: dict[str, AgentSkillTrace],
) -> dict:
    return {
        "registry_version": "sha256:" + registry_digest(registry),
        "router_version": router_version,
        "agents": {name: trace.to_dict() for name, trace in agents.items()},
    }
