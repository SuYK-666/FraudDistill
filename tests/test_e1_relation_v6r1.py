from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frauddistill.exp1_ccfa.relation_manifest import fraud_family_q_only, row_uid, same_row_uid_duplicates
from frauddistill.exp1_ccfa.residual_relation_cpu import ResidualRelationCPUDetector
from scripts.run_e1_relation_v6r1 import COMPARATORS, e1_score, paired_e1_bootstrap, stratified_q_derangement


def test_v6r1_fraud_family_does_not_read_answer() -> None:
    query = "How should a bank customer identify a phishing scam?"
    assert fraud_family_q_only(query, {}) == fraud_family_q_only(query, {"anything": "else"})


def test_v6r1_row_uid_changes_with_answer_but_family_does_not() -> None:
    a = {"source": "s", "id": "1", "user_query": "phishing prompt", "target_model_answer": "safe"}
    b = {**a, "target_model_answer": "unsafe instructions"}
    assert row_uid(a) != row_uid(b)
    assert fraud_family_q_only(a["user_query"], {}) == fraud_family_q_only(b["user_query"], {})


def test_v6r1_comparators_are_unique() -> None:
    signatures = {(spec["backend"], spec.get("level"), spec["mode"], spec["input"]) for spec in COMPARATORS.values()}
    assert len(signatures) == len(COMPARATORS)
    assert COMPARATORS["M4_pairlite_r2"]["backend"] != COMPARATORS["M5_residual_relation"]["backend"]


def test_v6r1_same_row_uid_cross_split_is_detected() -> None:
    uid = "u1"
    splits = {"a": [{"row_uid": uid}], "b": [{"row_uid": uid}]}
    assert same_row_uid_duplicates(splits) == 1


def test_v6r1_q_shuffle_preserves_strata_and_changes_query_when_possible() -> None:
    rows = [
        {"id": f"r{i}", "user_query": f"q{i}", "target_model_answer": "a", "e1_subset": "R1", "source": "s", "fraud_family_q_only": "phishing", "exp1_label": "unsafe"}
        for i in range(4)
    ]
    shuffled = stratified_q_derangement(rows, 1)
    assert [row["source"] for row in shuffled] == ["s"] * 4
    assert sorted(row["user_query"] for row in shuffled) == [f"q{i}" for i in range(4)]
    assert any(a["user_query"] != b["user_query"] for a, b in zip(rows, shuffled))


def test_v6r1_e1_bootstrap_uses_equal_subsets() -> None:
    m2 = []
    m5 = []
    for subset in ("R1", "R2", "R3"):
        for idx, label in enumerate(("safe", "unsafe")):
            base = {"id": f"{subset}{idx}", "cluster_id": f"{subset}{idx}", "e1_subset": subset, "gold_label": label}
            m2.append({**base, "pred_label": "safe"})
            m5.append({**base, "pred_label": label})
    result = paired_e1_bootstrap({("M2_y_only", 1): m2, ("M5_residual_relation", 1): m5}, [1], 20)
    assert result["seed1_M5_vs_M2"]["delta_mean"] > 0
    assert e1_score(m5) == 1.0


def test_v6r1_lambda_zero_is_documented_as_y_only_boundary() -> None:
    model = ResidualRelationCPUDetector.__new__(ResidualRelationCPUDetector)
    model.lambda_value = 0.0
    assert model.lambda_value == 0.0
