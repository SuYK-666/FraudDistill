"""Boundary-repair pilot split (guide section 10).

New stable hash with the guide-10.2 prefix "exp2-boundary-" so the split is
independent of the round-1/round-2 splits:

    stable_bucket(group_id) = sha256("exp2-boundary-" + group_id)[:8] % 100
    0-49  -> boundary_repair_dev
    50-99 -> boundary_repair_holdout

The boundary pilot samples only from boundary_repair_dev and excludes
round-1 pilot IDs, round-2 pilot IDs, the old paper_holdout groups and the
prompt-example sample (guide 10.3).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PREFIX = "exp2-boundary-"


def stable_bucket(group_id: str, modulo: int = 100) -> int:
    digest = hashlib.sha256(f"{PREFIX}{group_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def assign_split(group_id: str) -> str:
    bucket = stable_bucket(group_id)
    if bucket < 50:
        return "boundary_repair_dev"
    return "boundary_repair_holdout"


def is_dev(group_id: str) -> bool:
    return assign_split(group_id) == "boundary_repair_dev"


def write_boundary_splits(rows: list[dict], manifest_dir: Path) -> dict:
    """Write boundary_repair_dev_ids / boundary_repair_holdout_ids + digest."""
    groups: dict[str, str] = {}
    for r in rows:
        gid = str(r.get("group_id") or r.get("sample_id") or "")
        groups.setdefault(gid, assign_split(gid))
    from collections import Counter

    counts = Counter(groups.values())
    digest = hashlib.sha256(json.dumps(groups, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    dev = sorted(g for g, s in groups.items() if s == "boundary_repair_dev")
    holdout = sorted(g for g, s in groups.items() if s == "boundary_repair_holdout")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "boundary_repair_dev_ids.json").write_text(
        json.dumps(dev, ensure_ascii=False, indent=1), encoding="utf-8")
    (manifest_dir / "boundary_repair_holdout_ids.json").write_text(
        json.dumps(holdout, ensure_ascii=False, indent=1), encoding="utf-8")
    digest_file = {
        "method": 'sha256("exp2-boundary-"+group_id)[:8] % 100; <50 boundary_repair_dev, else boundary_repair_holdout',
        "digest": digest,
        "counts": dict(counts),
        "n_groups": len(groups),
        "created": "2026-08-06",
        "guide_ref": "FraudDistill_实验二量化修复与小规模验证实施指南 §10.2",
    }
    (manifest_dir / "boundary_split_digest.json").write_text(
        json.dumps(digest_file, ensure_ascii=False, indent=1), encoding="utf-8")
    return digest_file
