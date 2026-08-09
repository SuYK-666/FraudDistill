# -*- coding: utf-8 -*-
"""Exact / near-duplicate overlap audit for E4/E5 candidate pools."""
from __future__ import annotations

import itertools
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .exposure_registry import ExposureRegistry
from .schemas import norm_text, write_jsonl


def ngram_sig(text: str, n: int = 8) -> set[str]:
    t = norm_text(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def exact_overlap_gate(registry: ExposureRegistry, rows: list[dict]) -> dict:
    bad = []
    for r in rows:
        audit = registry.audit_candidate(r)
        if not audit["passed"]:
            bad.append({**audit, "user_query": r.get("user_query"), "target_model_answer": r.get("target_model_answer")})
    return {
        "n_candidates": len(rows),
        "n_failed": len(bad),
        "passed": len(bad) == 0,
        "failures": bad[:200],
    }


def near_duplicate_pairs(rows: list[dict], threshold: float = 0.85, max_pairs: int = 5000) -> list[dict]:
    """Normalized 8-gram Jaccard near-duplicate scan (sampled when huge)."""
    sigs = [(r, ngram_sig(str(r.get("user_query") or "") + " " + str(r.get("target_model_answer") or ""))) for r in rows]
    if len(sigs) > 4000:
        step = max(1, len(sigs) // 4000)
        sigs = sigs[::step]
    out = []
    for (r1, s1), (r2, s2) in itertools.combinations(sigs, 2):
        j = jaccard(s1, s2)
        if j >= threshold:
            out.append({
                "id_a": r1.get("id"), "id_b": r2.get("id"),
                "jaccard": round(j, 4),
                "qa": str(r1.get("user_query") or "")[:120],
                "qb": str(r2.get("user_query") or "")[:120],
            })
            if len(out) >= max_pairs:
                break
    return out


def run_exact_and_near(registry: ExposureRegistry, rows: list[dict], out_dir: Path,
                       jaccard_threshold: float = 0.85) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = exact_overlap_gate(registry, rows)
    write_jsonl(out_dir / "exact_overlap.json", [gate] if isinstance(gate, dict) else gate)
    pairs = near_duplicate_pairs(rows, threshold=jaccard_threshold)
    write_jsonl(out_dir / "near_duplicate_pairs.jsonl", pairs)
    return {"exact_gate": gate, "near_pairs": len(pairs), "threshold": jaccard_threshold}
