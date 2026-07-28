from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frauddistill.exp1_ccfa.embedding_cache import text_cache_key
from frauddistill.exp1_ccfa.public_gold import aegis_test_rows
from frauddistill.exp1_ccfa.relation_manifest import take_balanced_from_pool, take_grouped


def test_v6_aegis_has_no_prompt_label_fallback_or_empty_response() -> None:
    rows = aegis_test_rows("data/raw/aegis/test.json")
    assert rows
    assert all(row["metadata"]["p3_label_source"] == "response_label" for row in rows)
    assert all(row["target_model_answer"].strip() for row in rows)


def test_v6_same_q_group_sampling_keeps_pair_together() -> None:
    blocked: set[str] = set()
    rows = [
        {"id": "a_safe", "relation_group_id": "g1", "semantic_component_id": "c1", "exp1_label": "safe"},
        {"id": "a_unsafe", "relation_group_id": "g1", "semantic_component_id": "c1", "exp1_label": "unsafe"},
        {"id": "b_safe", "relation_group_id": "g2", "semantic_component_id": "c2", "exp1_label": "safe"},
        {"id": "b_unsafe", "relation_group_id": "g2", "semantic_component_id": "c2", "exp1_label": "unsafe"},
    ]
    selected = take_grouped(rows, 2, blocked, 7, "pilot:R1", "relation_group_id")
    assert len(selected) == 2
    assert len({row["relation_group_id"] for row in selected}) == 1
    assert {row["exp1_label"] for row in selected} == {"safe", "unsafe"}


def test_v6_split_can_keep_same_component_inside_one_split_only() -> None:
    blocked: set[str] = set()
    rows = [
        {"id": "safe", "semantic_component_id": "c1", "exp1_label": "safe"},
        {"id": "unsafe", "semantic_component_id": "c1", "exp1_label": "unsafe"},
    ]
    selected = take_balanced_from_pool(rows, 2, blocked, 11, "train")
    assert len(selected) == 2
    assert blocked == {"c1"}
    second = take_balanced_from_pool(rows, 2, blocked, 12, "model_dev")
    assert second == []


def test_v6_per_text_cache_key_is_order_independent() -> None:
    config = {"model_id": "m", "revision": "r", "prefix": "query: ", "max_length": 128, "pooling": "mean", "normalize": True, "backend": "transformers"}
    assert text_cache_key("same text", config) == text_cache_key("same text", dict(config))


def test_v6_per_text_cache_key_changes_with_prefix() -> None:
    config = {"model_id": "m", "revision": "r", "prefix": "query: ", "max_length": 128, "pooling": "mean", "normalize": True, "backend": "transformers"}
    changed = {**config, "prefix": "passage: "}
    assert text_cache_key("same text", config) != text_cache_key("same text", changed)
