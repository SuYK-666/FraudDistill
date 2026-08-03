"""Dataset audit helpers for exp2 v2 (guide section 4)."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from frauddistill.data.input_normalizer import normalize_jsonl


def audit_unified(path: str) -> dict[str, Any]:
    samples, counts = normalize_jsonl(path)
    gold = Counter()
    for s in samples:
        g = s.metadata.get("gold_binary")
        if g is None:
            gold["missing"] += 1
        else:
            gold["positive" if int(g) == 1 else "negative"] += 1
    return {
        "n_total": len(samples),
        "status_counts": dict(counts),
        "gold_counts": dict(gold),
        "n_valid_qy": counts.get("valid", 0),
    }


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
