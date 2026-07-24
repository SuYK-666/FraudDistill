from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import GroupShuffleSplit

from .deduplicate import build_group_id, enrich_dedup_fields


def grouped_train_dev_test_split(rows: list[dict[str, Any]], seed: int = 42, test_size: float = 0.15, dev_size: float = 0.15) -> dict[str, list[dict[str, Any]]]:
    if not 0 < test_size < 1 or not 0 < dev_size < 1 or test_size + dev_size >= 1:
        raise ValueError("invalid split ratios")
    enriched = enrich_dedup_fields(rows)
    # A row can share a case ID with one sample and an exact/near-duplicate
    # query with another. Build transitive components, not a priority key.
    groups = _connected_group_ids(enriched)
    indices = list(range(len(enriched)))
    train_dev, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(indices, groups=groups))
    remaining_groups = [groups[i] for i in train_dev]
    relative_dev = dev_size / (1 - test_size)
    train_rel, dev_rel = next(GroupShuffleSplit(n_splits=1, test_size=relative_dev, random_state=seed + 1).split(train_dev, groups=remaining_groups))
    split_indices = {"train": [train_dev[i] for i in train_rel], "dev": [train_dev[i] for i in dev_rel], "test": list(test)}
    result = {name: [dict(enriched[i], split=name, group_id=groups[i]) for i in values] for name, values in split_indices.items()}
    assert_no_leakage(result)
    return result


def assert_no_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    names = list(splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_groups = {str(row.get("group_id") or build_group_id(row)) for row in splits[left]}
            right_groups = {str(row.get("group_id") or build_group_id(row)) for row in splits[right]}
            if left_groups & right_groups:
                raise AssertionError(f"group leakage between {left} and {right}")
            left_queries = {str(row.get("normalized_query", "")) for row in splits[left]}
            right_queries = {str(row.get("normalized_query", "")) for row in splits[right]}
            if left_queries & right_queries:
                raise AssertionError(f"exact query leakage between {left} and {right}")


def write_split_audit(splits: dict[str, list[dict[str, Any]]], path: str | Path) -> dict[str, Any]:
    assert_no_leakage(splits)
    audit = {name: {"rows": len(rows), "groups": len({str(row.get("group_id") or build_group_id(row)) for row in rows}), "labels": dict(Counter(str(row.get("gold_label")) for row in rows))} for name, rows in splits.items()}
    audit["leakage"] = {"exact_query_across_split": 0, "group_across_split": 0}
    Path(path).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _connected_group_ids(rows: list[dict[str, Any]]) -> list[str]:
    parent = list(range(len(rows)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        keys = [
            "query:" + str(row.get("query_hash", "")),
            "cluster:" + str(row.get("near_duplicate_cluster_id", "")),
            "case:" + str(row.get("fraud_case_id") or metadata.get("fraud_case_id") or metadata.get("fraudr1_raw_id") or ""),
            "prompt:" + str(row.get("original_prompt_id") or metadata.get("base_prompt_id") or ""),
        ]
        for key in keys:
            if key.endswith(":"):
                continue
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    component_members: dict[int, list[int]] = {}
    for index in range(len(rows)):
        component_members.setdefault(find(index), []).append(index)
    labels = {}
    for root, members in component_members.items():
        signature = min(str(rows[index].get("query_hash", rows[index].get("id"))) for index in members)
        labels[root] = "component:" + signature[:16]
    return [labels[find(index)] for index in range(len(rows))]
