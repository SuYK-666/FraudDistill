"""SkillRegistry: scan skills/*/SKILL.md, validate frontmatter, keep digests
(guide section 8). Skills are instruction-only: no scripts, no network, no
file writes; only Markdown bodies are read.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from frauddistill.skills.schemas import SkillRecord


class SkillRegistry:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._index: dict[str, SkillRecord] = {}

    def discover(self) -> "SkillRegistry":
        records: dict[str, SkillRecord] = {}
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skill_dir = skill_file.parent.resolve()
            if self.root not in skill_dir.parents:
                raise ValueError(f"Skill escapes root: {skill_dir}")
            raw = skill_file.read_text(encoding="utf-8")
            frontmatter, body = self._split_frontmatter(raw)
            name = str(frontmatter["name"])
            description = str(frontmatter["description"])
            compatibility = str(frontmatter.get("compatibility", ""))
            if name != skill_file.parent.name:
                raise ValueError(
                    f"Skill name mismatch: {name} != {skill_file.parent.name}"
                )
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            records[name] = SkillRecord(
                name=name,
                description=description,
                compatibility=compatibility,
                body=body.strip(),
                path=skill_file,
                digest=digest,
                char_count=len(body),
            )
        self._index = records
        return self

    def get(self, name: str) -> SkillRecord:
        try:
            return self._index[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._index)

    def descriptions(self) -> dict[str, str]:
        return {name: record.description for name, record in self._index.items()}

    def records(self) -> dict[str, SkillRecord]:
        return dict(self._index)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __len__(self) -> int:
        return len(self._index)

    @staticmethod
    def _split_frontmatter(raw: str) -> tuple[dict, str]:
        if not raw.startswith("---\n"):
            raise ValueError("SKILL.md missing YAML frontmatter")
        parts = raw.split("---", 2)
        if len(parts) < 3:
            raise ValueError("SKILL.md frontmatter not closed")
        _, yaml_text, body = parts
        metadata = yaml.safe_load(yaml_text) or {}
        if "name" not in metadata:
            raise ValueError("Skill missing name")
        if "description" not in metadata:
            raise ValueError("Skill missing description")
        if not str(metadata["description"]).strip():
            raise ValueError(f"Skill {metadata.get('name')} has empty description")
        return metadata, body


# The 21 skills that already exist in the repo skills/ tree (user-confirmed
# final; no new skill files are added or modified). response-content-harm is
# implemented at the code level (refusal schema + head + adapter) and the
# router simply skips it when the skill is absent.
EXPECTED_SKILLS = {
    "adversarial-language-normalization",
    "agent-output-quality-gate",
    "benchmark-output-adapter",
    "bilingual-fraud-analysis",
    "evidence-arbitration",
    "evidence-consistency-check",
    "evidence-span-grounding",
    "fraud-assistance-core",
    "fraud-harmful-engagement",
    "fraud-taxonomy-routing",
    "multiturn-context-reconstruction",
    "overrefusal-diagnosis",
    "partial-leakage-detection",
    "refusal-outcome",
    "request-policy-risk",
    "response-actionability",
    "response-content-harm",
    "roleplay-safety-boundary",
    "runtime-cost-controller",
    "safe-context-disambiguation",
    "skill-router",
    "uncertainty-calibration-abstention",
}


def registry_digest(registry: SkillRegistry) -> str:
    payload = "\n".join(
        f"{name}:{registry.get(name).digest}" for name in sorted(registry.descriptions())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_expected_skills(registry: SkillRegistry) -> list[str]:
    """Return missing expected skills (raises nothing; caller decides)."""
    return sorted(EXPECTED_SKILLS - set(registry.descriptions()))
