"""Skill runtime schemas (guide sections 8-9)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    compatibility: str
    body: str
    path: Path
    digest: str
    char_count: int


@dataclass(frozen=True)
class SkillSelection:
    agent_name: str
    selected: tuple[str, ...]
    reasons: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    total_chars: int = 0


@dataclass(frozen=True)
class AgentSkillTrace:
    selected: tuple[str, ...] = ()
    digests: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    total_chars: int = 0

    def to_dict(self) -> dict:
        return {
            "selected": list(self.selected),
            "digests": dict(self.digests),
            "reasons": dict(self.reasons),
            "total_chars": self.total_chars,
        }
