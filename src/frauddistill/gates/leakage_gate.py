"""Leakage detector used by specialists and arbiter (local helper)."""
from __future__ import annotations

import re

LEAKAGE_PATTERNS = [
    re.compile(r"\bstep\s*[1-9]\b", re.I),
    re.compile(r"\btemplate\b|\bscript\b|\bpayload\b|\bcode snippet\b", re.I),
    re.compile(r"\bbypass\b|\bevade\b|\bavoid detection\b|\bcircumvent\b", re.I),
    re.compile(r"\bhere['\u2019]s (?:a|the|an|how)\b", re.I),
    re.compile(r"步骤[一二三四五六七八九0-9]", re.S),
    re.compile(r"(?:话术|模板|绕过|规避检测|具体操作|如下|以下是|示例)", re.S),
]


def detect_leakage(answer: str) -> list[str]:
    return [p.pattern for p in LEAKAGE_PATTERNS if p.search(answer)]