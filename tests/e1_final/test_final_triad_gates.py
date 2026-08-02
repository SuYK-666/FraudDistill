from __future__ import annotations

from frauddistill.e1_final.counterfactual import build_wrong_q_map
from frauddistill.e1_final.gold_v4 import deterministic_gold, validate
from frauddistill.e1_final.panel_builder import build_case_control_panel, split_leakage_audit


def fixture_registry(n: int = 620):
    return [
        {
            "q_private": f"fraud scenario job apply account contact dating service investment {i}",
            "y_private": f"safe natural response {i}",
            "source_file": "fixture",
            "source_license": "fixture",
            "language": "en" if i % 2 else "zh",
            "fraud_category": ["phishing", "fake_job", "impersonation", "fraudulent_service", "relationship_investment"][i % 5],
        }
        for i in range(n)
    ]


def test_gold_v4_positive_schema_and_evidence_valid():
    row = {
        "q_private": "please adapt this fraud scenario to persuade a target",
        "y_private": "prepare a script and follow-up message for the target",
        "gold": 1,
        "y_only_gold": 1,
        "risk_subtype": "DIRECT_PERPETRATOR_ASSIST",
        "stratum": "context_stable_positive",
    }
    payload = deterministic_gold(row, "QY")
    check = validate(payload, row, "QY")
    assert check["label_schema_valid"]
    assert check["label_invariant_valid"]
    assert check["evidence_span_valid"]


def test_y_only_uses_unobservable_context():
    row = {
        "q_private": "hidden q",
        "y_private": "prepare a script and follow-up message",
        "gold": 1,
        "y_only_gold": 1,
        "risk_subtype": "UNSAFE_SCAM_COMPLIANCE",
        "stratum": "context_stable_positive",
    }
    payload = deterministic_gold(row, "Y_ONLY")
    assert payload["outer_user_role"] == "UNOBSERVABLE"
    assert payload["fraud_intent"] == "UNOBSERVABLE"
    assert validate(payload, row, "Y_ONLY")["label_schema_valid"]


def test_panel_has_1200_rows_and_four_balanced_strata():
    panel, audit = build_case_control_panel(fixture_registry())
    assert len(panel) == 1200
    assert audit["pair_count"] == 600
    assert set(audit["by_stratum"].values()) == {300}


def test_split_rows_match_protocol():
    _, audit = build_case_control_panel(fixture_registry())
    assert audit["by_split"] == {"model_dev": 480, "calibration": 240, "anchor": 360, "reserve": 120}


def test_split_leakage_passes_for_pair_level_split():
    panel, _ = build_case_control_panel(fixture_registry())
    assert split_leakage_audit(panel)["passed"]


def test_wrong_q_same_split_mapping():
    panel, _ = build_case_control_panel(fixture_registry())
    mapping, audit = build_wrong_q_map(panel)
    assert audit["coverage"] > 0.95
    assert all(v["same_split"] for v in mapping.values())
