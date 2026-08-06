"""Skills runtime for FraudDistill (guide: Skills integration).

Instruction-only, progressively-disclosed domain skills selected by a
deterministic local router and composed into agent system prompts.
"""
from frauddistill.skills.registry import SkillRecord, SkillRegistry, registry_digest
from frauddistill.skills.router import SkillRouter, SkillSelection, extract_signals
from frauddistill.skills.prompt_composer import SkillPromptComposer, split_schema_instruction
from frauddistill.skills.runtime import SkillRuntime, build_skill_runtime

__all__ = [
    "SkillRecord",
    "SkillRegistry",
    "SkillRouter",
    "SkillSelection",
    "SkillPromptComposer",
    "SkillRuntime",
    "build_skill_runtime",
    "extract_signals",
    "registry_digest",
    "split_schema_instruction",
]
