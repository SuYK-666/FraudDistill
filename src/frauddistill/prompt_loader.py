"""Load prompt text files from configs/prompts (cached, UTF-8)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "configs" / "prompts"

_cache: dict[str, str] = {}


def load_prompt(name: str) -> str:
    if name not in _cache:
        _cache[name] = (PROMPT_DIR / name).read_text(encoding="utf-8").strip()
    return _cache[name]
