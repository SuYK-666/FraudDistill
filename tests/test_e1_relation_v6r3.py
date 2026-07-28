from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frauddistill.exp1_ccfa.relation_manifest_v6r3 import (
    max_component_matching_v6r3,
    row_uid_v6r3,
    split_quota_audit,
    v6r3_census,
)


def test_v6r3_row_uid_includes_revision_and_split() -> None:
    base = {"source": "s", "user_query": "q", "target_model_answer": "y", "metadata": {"source_dataset": "d", "dataset_revision": "r1", "official_split": "train", "original_id": "1"}}
    changed = {"source": "s", "user_query": "q", "target_model_answer": "y", "metadata": {"source_dataset": "d", "dataset_revision": "r2", "official_split": "train", "original_id": "1"}}
    assert row_uid_v6r3(base) != row_uid_v6r3(changed)


def test_v6r3_component_matching_uses_each_component_once() -> None:
    unsafe = {"leakage_supercomponent_id": "u", "exp1_label": "unsafe"}
    safe_a = {"leakage_supercomponent_id": "s1", "exp1_label": "safe"}
    safe_b = {"leakage_supercomponent_id": "s2", "exp1_label": "safe"}
    payload = {
        tuple(sorted(("u", "s1"))): (0.1, unsafe, safe_a),
        tuple(sorted(("u", "s2"))): (0.2, unsafe, safe_b),
    }
    matched = max_component_matching_v6r3([("u", "s1", 0.1), ("u", "s2", 0.2)], payload, 2)
    components = [component for match in matched for component in match["components"]]
    assert len(components) == len(set(components))


def test_v6r3_split_false_check_blocks_top_level_pass() -> None:
    gates = {
        "g0r3": {
            "require_clean_git": False,
            "required_source_failures_max": 0,
            "prompt_label_fallback_max": 0,
            "project_manual_test_labels_max": 0,
            "r1_groups_min": 1,
            "r2_true_max_matching_min": 1,
            "r2_selected_groups": 1,
            "r3_balanced_capacity_min": 2,
            "r3_selected_rows": 2,
            "r3_source_count_min": 1,
            "r3_largest_source_max": 1.0,
            "r3_fraud_families_min": 1,
        }
    }
    row = {"relation_group_id": "g", "leakage_supercomponent_id": "a", "exp1_label": "safe", "source": "s", "fraud_family": "f"}
    census = v6r3_census(
        {"s": [row]},
        {"failures": []},
        [row, {**row, "leakage_supercomponent_id": "b", "exp1_label": "unsafe"}],
        [row, {**row, "leakage_supercomponent_id": "c", "exp1_label": "unsafe"}],
        [row, {**row, "leakage_supercomponent_id": "d", "exp1_label": "unsafe"}],
        {"master_train": []},
        {"passed": False, "max_matching_groups": 1, "selected_groups": 1},
        {"max_balanced_r3_rows": 2},
        {"passed": True, "checks": {}},
        {"passed": True},
        {"loaded": True, "rows_admitted": 1},
        {"experiment": {"protocol": "E1_CPU_CCF-A_v6r3"}, "gates": gates},
        require_clean_git=False,
    )
    assert census["decision"] == "E1_V6R3_G0_STOP"


def test_v6r3_split_quota_audit_requires_exact_balance() -> None:
    config = {"data": {"quotas": {"x": {"R1": 2, "R2": 0, "R3": 0}}}}
    rows = [{"exp1_label": "safe", "semantic_component_id": "a"}, {"exp1_label": "safe", "semantic_component_id": "b"}]
    audit = split_quota_audit({"x": rows}, config)
    assert audit["checks"]["x_balance"] is False
