from __future__ import annotations

import pytest
from sklearn.metrics import f1_score

from frauddistill.exp1_ccfa.public_gold import aegis_test_rows, polyguard_language_component_audit
from frauddistill.exp1_ccfa.exact_mcnemar import exact_mcnemar
from frauddistill.exp1_ccfa.pair_cross_encoder import format_pair_text
from frauddistill.exp1_ccfa.pairlite_cpu import PairLiteCPUDetector
from frauddistill.exp1_ccfa.paired_cluster_bootstrap import paired_cluster_bootstrap_delta
from frauddistill.exp1_ccfa.stat_tests import holm_adjust
from frauddistill.exp1_ccfa.semantic_components import (
    attach_semantic_components,
    explicit_label_token_audit,
    leakage_audit,
    normalize_label,
    split_by_component,
)
from frauddistill.guards.parsing import parse_guard_label
from frauddistill.student.pair_tfidf import PairTfidfDetector


def test_missing_label_raises() -> None:
    with pytest.raises(ValueError):
        normalize_label({"id": "r1", "user_query": "q", "target_model_answer": "a"})


def test_input_mode_projection_text_boundaries() -> None:
    row = {"user_query": "question text", "target_model_answer": "answer text"}
    assert "answer text" not in format_pair_text(row, "q_only")
    assert "question text" not in format_pair_text(row, "y_only")
    assert "question text" in format_pair_text(row, "q_y")
    assert "answer text" in format_pair_text(row, "q_y")


def test_semantic_component_split_has_no_leakage() -> None:
    rows = [
        {"id": f"r{i}", "source_prompt_id": f"p{i}", "user_query": "q", "target_model_answer": "a", "pair_fraud_label": "safe" if i % 2 else "unsafe"}
        for i in range(20)
    ]
    attached = attach_semantic_components(rows)
    train, model_dev, threshold_dev, audit = split_by_component(attached, seed=7)
    assert train and model_dev and threshold_dev
    assert audit["passed"]
    assert leakage_audit({"train": train, "model_dev": model_dev, "threshold_dev": threshold_dev})["passed"]


def test_explicit_label_tokens_are_blocked() -> None:
    audit = explicit_label_token_audit(
        [{"id": "bad", "user_query": "SAFE CONTEXT: transfer money", "target_model_answer": "", "pair_fraud_label": "safe"}]
    )
    assert not audit["passed"]
    assert audit["hit_count"] == 1


def test_guard_parser_abstains_on_ambiguous_output() -> None:
    assert parse_guard_label("maybe, not enough information") == {"label": "abstain", "parse_success": False}
    assert parse_guard_label("Decision: unsafe") == {"label": "unsafe", "parse_success": True}


def test_exact_mcnemar_known_counts() -> None:
    result = exact_mcnemar(
        ["safe", "safe", "unsafe", "unsafe"],
        ["safe", "unsafe", "unsafe", "safe"],
        ["safe", "safe", "safe", "unsafe"],
    )
    assert result["b"] == 1
    assert result["c"] == 2
    assert 0 <= result["p_value"] <= 1


def test_cluster_bootstrap_reports_cluster_count_not_row_count() -> None:
    y_true = ["safe", "unsafe", "safe", "unsafe"]
    pred_a = ["safe", "safe", "safe", "safe"]
    pred_b = ["safe", "unsafe", "safe", "unsafe"]
    clusters = ["c1", "c1", "c2", "c2"]
    metric_fn = lambda y, p: float(f1_score(y, p, average="macro", zero_division=0))
    result = paired_cluster_bootstrap_delta(y_true, pred_a, pred_b, clusters, metric_fn, iterations=20, seed=1)
    assert result["cluster_count"] == 2
    assert result["ci_lower"] <= result["delta_mean"] + 1e-12
    assert result["delta_mean"] <= result["ci_upper"] + 1e-12


def test_holm_adjustment_is_monotonic() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"]["holm_p"] <= adjusted["c"]["holm_p"] <= adjusted["b"]["holm_p"]
    assert all(0 <= row["holm_p"] <= 1 for row in adjusted.values())


def test_tfidf_mode_specific_fit_changes_feature_contract() -> None:
    rows = [
        {"user_query": "fraud transfer risk", "target_model_answer": "refuse risk"},
        {"user_query": "benign report help", "target_model_answer": "helpful help"},
        {"user_query": "fraud phishing risk", "target_model_answer": "steps risk"},
        {"user_query": "benign victim help", "target_model_answer": "protect help"},
    ]
    labels = ["unsafe", "safe", "unsafe", "safe"]
    detector = PairTfidfDetector()
    detector.fit(rows, labels, mode="q_only")
    assert detector.features(rows, "q_only").shape[0] == len(rows)


def test_pairlite_q_only_does_not_read_answer() -> None:
    rows = [
        {"user_query": "phishing transfer risk", "target_model_answer": "send code now"},
        {"user_query": "benign report help", "target_model_answer": "report safely"},
        {"user_query": "credential scam", "target_model_answer": "use this script"},
        {"user_query": "victim prevention", "target_model_answer": "do not share otp"},
    ]
    model = PairLiteCPUDetector(level="L2", word_features=100, char_features=100, hash_features=64, seed=1).fit(
        rows, ["unsafe", "safe", "unsafe", "safe"], mode="q_only"
    )
    changed = [dict(row, target_model_answer=f"changed answer {idx}") for idx, row in enumerate(rows)]
    diff = model.features(rows, "q_only") - model.features(changed, "q_only")
    assert diff.nnz == 0


def test_pairlite_y_only_does_not_read_query() -> None:
    rows = [
        {"user_query": "phishing transfer risk", "target_model_answer": "send code now"},
        {"user_query": "benign report help", "target_model_answer": "report safely"},
        {"user_query": "credential scam", "target_model_answer": "use this script"},
        {"user_query": "victim prevention", "target_model_answer": "do not share otp"},
    ]
    model = PairLiteCPUDetector(level="L2", word_features=100, char_features=100, hash_features=64, seed=1).fit(
        rows, ["unsafe", "safe", "unsafe", "safe"], mode="y_only"
    )
    changed = [dict(row, user_query=f"changed query {idx}") for idx, row in enumerate(rows)]
    diff = model.features(rows, "y_only") - model.features(changed, "y_only")
    assert diff.nnz == 0


def test_pairlite_cross_block_changes_when_either_side_changes() -> None:
    rows = [
        {"user_query": "phishing transfer risk", "target_model_answer": "send code now"},
        {"user_query": "benign report help", "target_model_answer": "report safely"},
        {"user_query": "credential scam", "target_model_answer": "use this script"},
        {"user_query": "victim prevention", "target_model_answer": "do not share otp"},
    ]
    model = PairLiteCPUDetector(level="L2", word_features=100, char_features=100, hash_features=64, seed=1).fit(
        rows, ["unsafe", "safe", "unsafe", "safe"], mode="q_y"
    )
    baseline = model.cross_qy_features([row["user_query"] for row in rows], [row["target_model_answer"] for row in rows])
    changed_q = model.cross_qy_features([f"changed query {idx}" for idx, _ in enumerate(rows)], [row["target_model_answer"] for row in rows])
    changed_y = model.cross_qy_features([row["user_query"] for row in rows], [f"changed answer {idx}" for idx, _ in enumerate(rows)])
    assert (baseline - changed_q).nnz > 0
    assert (baseline - changed_y).nnz > 0


def test_polyguard_same_base_id_across_languages_share_component() -> None:
    rows = attach_semantic_components(
        [
            {
                "id": "p3_polyguard_7_english",
                "source_prompt_id": "polyguard_7",
                "language": "English",
                "user_query": "q en",
                "target_model_answer": "a en",
                "pair_fraud_label": "safe",
                "metadata": {"source_base_id": "7"},
            },
            {
                "id": "p3_polyguard_7_chinese",
                "source_prompt_id": "polyguard_7",
                "language": "Chinese",
                "user_query": "q zh",
                "target_model_answer": "a zh",
                "pair_fraud_label": "safe",
                "metadata": {"source_base_id": "7"},
            },
        ]
    )
    assert rows[0]["semantic_component_id"] == rows[1]["semantic_component_id"]
    assert polyguard_language_component_audit(rows)["passed"]


def test_aegis_p3_uses_official_test_split_only() -> None:
    rows = aegis_test_rows("data/raw/aegis/test.json")
    assert len(rows) == 1964
    assert {row["metadata"]["official_split"] for row in rows} == {"test"}
    assert all("train" not in row["source_prompt_id"] and "validation" not in row["source_prompt_id"] for row in rows)


def test_frozen_panel_predict_path_does_not_fit() -> None:
    import inspect
    from scripts.run_exp1_ccfa import predict_frozen_panel

    source = inspect.getsource(predict_frozen_panel)
    assert ".fit(" not in source
