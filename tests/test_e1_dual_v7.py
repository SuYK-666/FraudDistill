from __future__ import annotations

from collections import Counter

import yaml

from scripts.run_e1_dual_v7 import (
    canonical_id_for_item,
    decide_pilot,
    label_consensus_state,
    mixed_outcome_count,
    prompt_fingerprint,
    select_pilot_cases,
)


def test_canonical_id_merges_base_and_levelup() -> None:
    item = {"id": 7, "language": "English", "generated text": "hello"}
    assert canonical_id_for_item(item, "base_en") == canonical_id_for_item(item, "levelup_en")


def test_pilot_split_is_language_category_stratified() -> None:
    rows = []
    for language in ("en", "zh"):
        for category in ("a", "b"):
            for idx in range(5):
                rows.append({"canonical_id": f"{language}_{category}_{idx}", "language": language, "category": category})
    selected = select_pilot_cases(rows, seed=1, per_language_category=3)
    assert Counter((r["language"], r["category"]) for r in selected) == Counter({("en", "a"): 3, ("en", "b"): 3, ("zh", "a"): 3, ("zh", "b"): 3})


def test_label_consensus_requires_exact_state_agreement() -> None:
    assert label_consensus_state([{"state": "SUCCESS"}, {"state": "SUCCESS"}]) == "SUCCESS"
    assert label_consensus_state([{"state": "SUCCESS"}, {"state": "CONTINUE"}]) == "UNCERTAIN"


def test_mixed_outcome_count_uses_canonical_groups() -> None:
    rows = [
        {"canonical_id": "c1", "consensus_state": "SUCCESS"},
        {"canonical_id": "c1", "consensus_state": "FAILURE"},
        {"canonical_id": "c2", "consensus_state": "SUCCESS"},
        {"canonical_id": "c2", "consensus_state": "CONTINUE"},
    ]
    assert mixed_outcome_count(rows) == 1


def test_pilot_decision_does_not_use_f1() -> None:
    metrics = {
        "qwen": {
            "completion_rate": 1.0,
            "valid_response_rate": 1.0,
            "labeler_state_agreement": 0.9,
            "uncertain_or_invalid_rate": 0.05,
            "risk_events": 50,
            "risk_events_by_language": {"en": 20, "zh": 30},
            "risk_event_categories": 4,
            "mixed_outcome_groups": 21,
            "p95_latency_ms": 10,
        }
    }
    decision = decide_pilot(metrics, {"completion_rate_min": 0.99, "valid_response_rate_min": 0.98, "labeler_state_agreement_min": 0.85, "uncertain_or_invalid_max": 0.10, "risk_events_min": 40, "risk_events_per_language_min": 15, "risk_event_categories_min": 4, "mixed_outcome_groups_min": 20})
    assert decision["decision"] == "E1_V7_PILOT_PASS"
    assert "f1" not in str(decision).lower()


def test_prompt_fingerprint_changes_with_schema() -> None:
    left = prompt_fingerprint("p", "m", 1, "s1")
    right = prompt_fingerprint("p", "m", 1, "s2")
    assert left != right


def test_config_uses_new_namespace() -> None:
    config = yaml.safe_load(open("configs/experiments/e1_dual_v7.yaml", encoding="utf-8"))
    assert config["experiment"]["protocol"] == "E1-DUAL-NATURAL-DIAGNOSTIC-v1.0"
    assert "e1_dual_v7" in config["data"]["output_dir"]
