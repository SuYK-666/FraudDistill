"""Group-aware splits for exp2 v2 (guide section 4.4)."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def group_split(
    rows: list[dict[str, Any]],
    *,
    group_key: str = "group_id",
    stratify_key: str | None = None,
    test_frac: float = 0.2,
    seed: int = 20260803,
) -> tuple[list[dict], list[dict]]:
    """Split rows by group, optionally stratified by stratify_key value of the first row in group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(group_key) or r.get("id"))].append(r)
    rng = random.Random(seed)
    strat_buckets: dict[str, list[str]] = defaultdict(list)
    for gid, members in groups.items():
        key = str(members[0].get(stratify_key) or "unknown") if stratify_key else "all"
        strat_buckets[key].append(gid)
    train_groups: set[str] = set()
    test_groups: set[str] = set()
    for bucket, gids in strat_buckets.items():
        rng.shuffle(gids)
        n_test = max(1, int(round(len(gids) * test_frac)))
        test_groups.update(gids[:n_test])
        train_groups.update(gids[n_test:])
    train_rows = [r for gid in train_groups for r in groups[gid]]
    test_rows = [r for gid in test_groups for r in groups[gid]]
    return train_rows, test_rows


def stratified_sample(rows: list[dict[str, Any]], n: int, *, stratify_key: str, seed: int = 20260803) -> list[dict]:
    """Sample exactly n rows, preserving relative stratum proportions (largest-remainder)."""
    if n <= 0 or not rows:
        return []
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(stratify_key) or "unknown")].append(r)
    rng = random.Random(seed)
    n = min(n, len(rows))
    base = n / len(rows)
    alloc = {key: len(members) * base for key, members in buckets.items()}
    floors = {key: int(v) for key, v in alloc.items()}
    rem = n - sum(floors.values())
    for key in sorted(buckets, key=lambda k: (alloc[k] - floors[k], -len(buckets[k])), reverse=True)[:rem]:
        floors[key] += 1
    out: list[dict] = []
    for key, members in buckets.items():
        rng.shuffle(members)
        out.extend(members[: floors[key]])
    rng.shuffle(out)
    return out[:n]
