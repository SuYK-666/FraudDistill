from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def assign_b_splits(rows: list[dict[str, Any]], *, seed: int, model_dev: int, calibration: int, anchor: int, reserve: int) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["canonical_q_id"]].append(row)
    valid = [g for g in groups.values() if len(g) == 2 and {r["source_material_label"] for r in g} == {0, 1}]
    rng = random.Random(seed)
    rng.shuffle(valid)
    roles = [("model_dev", model_dev), ("calibration", calibration), ("anchor", anchor), ("reserve", reserve)]
    out: list[dict[str, Any]] = []
    pos = 0
    for role, n in roles:
        for group in valid[pos : pos + n]:
            for row in group:
                out.append({**row, "split_role": role})
        pos += n
    return out


def build_wrong_q_map(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qs = {}
    for row in rows:
        qs.setdefault(row["canonical_q_id"], row)
    ordered = list(qs.values())
    out = []
    for i, row in enumerate(ordered):
        for j in range(1, len(ordered)):
            cand = ordered[(i + j) % len(ordered)]
            if cand["canonical_q_id"] != row["canonical_q_id"] and cand["language"] == row["language"] and cand["behavior_cluster_id"] != row["behavior_cluster_id"]:
                out.append(
                    {
                        "canonical_q_id": row["canonical_q_id"],
                        "wrong_canonical_q_id": cand["canonical_q_id"],
                        "wrong_q_private_sha256": cand.get("q_private_sha256"),
                        "language": row["language"],
                        "length_delta": abs(len(row["q_private"]) - len(cand["q_private"])),
                    }
                )
                break
    return out


def panel_rows(rows: list[dict[str, Any]], split_role: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("split_role") == split_role]


def context_subset(rows: list[dict[str, Any]], minimum: int) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["canonical_q_id"]].append(row)
    selected = []
    for group in groups.values():
        if any(r.get("source_material_label") == 1 for r in group) and any(r.get("source_material_label") == 0 for r in group):
            selected.extend(group)
        if len({r["canonical_q_id"] for r in selected}) >= minimum:
            break
    return selected


def split_disjoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_role = defaultdict(set)
    for row in rows:
        by_role[row.get("split_role")].add(row["canonical_q_id"])
    conflicts = []
    roles = list(by_role)
    for i, a in enumerate(roles):
        for b in roles[i + 1 :]:
            overlap = by_role[a] & by_role[b]
            if overlap:
                conflicts.append({"left": a, "right": b, "overlap": len(overlap)})
    return {"passed": not conflicts, "conflicts": conflicts}
