from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frauddistill.exp1_ccfa.relation_manifest import attach_leakage_supercomponents, max_cardinality_min_cost, subset_super_overlap


def test_v6r2_exact_query_builds_same_supercomponent() -> None:
    rows = [
        {"id": "a", "semantic_component_id": "a", "user_query": "same q", "target_model_answer": "answer one"},
        {"id": "b", "semantic_component_id": "b", "user_query": "same q", "target_model_answer": "answer two"},
    ]
    out = attach_leakage_supercomponents(rows)
    assert out[0]["leakage_supercomponent_id"] == out[1]["leakage_supercomponent_id"]


def test_v6r2_exact_answer_builds_same_supercomponent() -> None:
    rows = [
        {"id": "a", "semantic_component_id": "a", "user_query": "q1", "target_model_answer": "same answer"},
        {"id": "b", "semantic_component_id": "b", "user_query": "q2", "target_model_answer": "same answer"},
    ]
    out = attach_leakage_supercomponents(rows)
    assert out[0]["leakage_supercomponent_id"] == out[1]["leakage_supercomponent_id"]


def test_v6r2_subset_super_overlap_detects_overlap() -> None:
    assert subset_super_overlap({"R1": [{"leakage_supercomponent_id": "x"}], "R2": [{"leakage_supercomponent_id": "x"}]}) == 1


def test_v6r2_max_matching_prefers_cardinality() -> None:
    edges = [(0, 0, 100.0), (0, 1, 0.1), (1, 0, 0.1)]
    matched = max_cardinality_min_cost(edges, 2)
    assert len(matched) == 2
