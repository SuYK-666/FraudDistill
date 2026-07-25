from __future__ import annotations

import re


SAFE_WORDS = ("safe", "benign", "allowed", "harmless", "refusal")
UNSAFE_WORDS = ("unsafe", "harmful", "fraud", "phishing", "scam", "disallowed")


def parse_guard_label(text: str) -> dict:
    lowered = text.strip().lower()
    if not lowered:
        return {"label": "abstain", "parse_success": False}
    label_match = re.search(r"\b(label|decision|risk)\s*[:=]\s*(safe|unsafe|abstain)\b", lowered)
    if label_match:
        return {"label": label_match.group(2), "parse_success": True}
    unsafe = any(word in lowered for word in UNSAFE_WORDS)
    safe = any(word in lowered for word in SAFE_WORDS)
    if unsafe and not safe:
        return {"label": "unsafe", "parse_success": True}
    if safe and not unsafe:
        return {"label": "safe", "parse_success": True}
    return {"label": "abstain", "parse_success": False}

