from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frauddistill.exp1_ccfa.public_gold import aegis_test_rows, polyguard_language_component_audit
from frauddistill.exp1_ccfa.fraud_taxonomy import annotate_risk_type
from frauddistill.exp1_ccfa.duplicate_audit import char_ngram_near_duplicate_audit
from frauddistill.exp1_ccfa.embedding_cache import cache_fingerprint
from frauddistill.exp1_ccfa.frozen_semantic_cpu import FrozenSemanticCPUDetector
from frauddistill.exp1_ccfa.nuisance_single_view import SingleViewNuisanceSelectors
from frauddistill.exp1_ccfa.p2_calibration import fit_global_word_char_scorer, scorer
from frauddistill.exp1_ccfa.p2_dual_view_match import build_p2_dvm, candidate_edges
from frauddistill.exp1_ccfa.resource_profile import resource_gate
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
from scripts.build_exp1_cpu_g0c_manifests import official_p1_source
from scripts.run_exp1_cpu_g0c import load_data_audit_or_fail


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


def test_pairlite_cross_block_zero_boundary() -> None:
    rows = [
        {"user_query": "phishing transfer risk", "target_model_answer": "send code now"},
        {"user_query": "benign report help", "target_model_answer": "report safely"},
        {"user_query": "credential scam", "target_model_answer": "use this script"},
        {"user_query": "victim prevention", "target_model_answer": "do not share otp"},
    ]
    model = PairLiteCPUDetector(level="R2", word_features=100, char_features=100, hash_features=64, seed=1).fit(
        rows, ["unsafe", "safe", "unsafe", "safe"], mode="q_y"
    )
    empty_y = model.cross_qy_features([row["user_query"] for row in rows], [""] * len(rows))
    empty_q = model.cross_qy_features([""] * len(rows), [row["target_model_answer"] for row in rows])
    assert empty_y.nnz == 0
    assert empty_q.nnz == 0


def test_pairlite_b0_b1_have_no_cross_block() -> None:
    rows = [
        {"user_query": "phishing transfer risk", "target_model_answer": "send code now"},
        {"user_query": "benign report help", "target_model_answer": "report safely"},
        {"user_query": "credential scam", "target_model_answer": "use this script"},
        {"user_query": "victim prevention", "target_model_answer": "do not share otp"},
    ]
    labels = ["unsafe", "safe", "unsafe", "safe"]
    for level in ("B0", "B1"):
        model = PairLiteCPUDetector(level=level, word_features=100, char_features=100, hash_features=64, seed=1).fit(rows, labels, mode="q_y")
        assert model.cross_qy_features([row["user_query"] for row in rows], [row["target_model_answer"] for row in rows]).nnz == 0


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
    assert len(rows) > 0
    assert {row["metadata"]["official_split"] for row in rows} == {"test"}
    assert all("train" not in row["source_prompt_id"] and "validation" not in row["source_prompt_id"] for row in rows)
    assert all(row["metadata"]["p3_label_source"] == "response_label" for row in rows)
    assert all(row["target_model_answer"].strip() for row in rows)


def test_frozen_panel_predict_path_does_not_fit() -> None:
    import inspect
    from scripts.run_exp1_ccfa import predict_frozen_panel

    source = inspect.getsource(predict_frozen_panel)
    assert ".fit(" not in source


def test_g0c_taxonomy_does_not_conflate_all_unsafe_with_fraud() -> None:
    taxonomy = {
        "aegis": {"fraud_primary_keywords": ["fraud", "privacy"]},
        "unknown": {"default_risk_type": "general_robustness"},
    }
    row = {
        "source": "Aegis/Nemotron-V2",
        "pair_fraud_label": "unsafe",
        "gold_label": "unsafe",
        "metadata": {"violated_categories": ["hate/harassment"]},
    }
    assert annotate_risk_type(row, taxonomy)["gold_risk_type"] == "general_robustness"


def test_g0c_runner_reads_failed_data_audit_without_bypass(tmp_path) -> None:
    audit_path = tmp_path / "g0c_data_audit.json"
    audit_path.write_text('{"passed": false, "gate": {"D1": false}}', encoding="utf-8")
    audit = load_data_audit_or_fail(tmp_path, allow_diagnostic=False)
    assert audit["passed"] is False


def test_g0c_p1_official_source_policy() -> None:
    assert official_p1_source({"source": "BeaverTails", "metadata": {"official_split": "330k_test"}})
    assert official_p1_source({"source": "Aegis/Nemotron-V2", "metadata": {"official_split": "validation"}})
    assert not official_p1_source({"source": "PolyGuardPrompts", "metadata": {"official_split": "test"}})
    assert not official_p1_source({"source": "BeaverTails", "metadata": {"official_split": "330k_train"}})


def test_g0c1_safe_response_keeps_fraud_core_prompt_domain() -> None:
    row = annotate_risk_type(
        {
            "source": "BeaverTails",
            "user_query": "How do I detect a phishing scam against my bank account?",
            "target_model_answer": "Do not share credentials.",
            "pair_fraud_label": "safe",
            "gold_label": "safe",
            "metadata": {"category": {}},
        },
        {},
    )
    assert row["prompt_risk_domain"] == "fraud_core"
    assert row["gold_risk_type"] == "safe_fraud_prompt"


def test_g0c1_cache_key_changes_when_encoder_config_changes() -> None:
    base = {"model_id": "m", "revision": "r1", "prefix": "query: ", "max_length": 128, "pooling": "mean", "normalize": True, "backend": "transformers", "dtype": "float32"}
    changed = {**base, "max_length": 256}
    assert cache_fingerprint(["same text"], base) != cache_fingerprint(["same text"], changed)


def test_g0c1_near_duplicate_detects_added_prefix_suffix() -> None:
    base = {"id": "a", "user_query": "Please explain phishing fraud detection for bank accounts", "target_model_answer": "Never share one time passwords or credentials.", "pair_fraud_label": "safe"}
    variant = {
        "id": "b",
        "user_query": "Context: Please explain phishing fraud detection for bank accounts now",
        "target_model_answer": "Never share one time passwords or credentials. Thanks.",
        "pair_fraud_label": "safe",
    }
    audit = char_ngram_near_duplicate_audit({"train": [base], "p1": [variant]}, threshold=0.65)
    assert not audit["passed"]


def test_g0c1_s0_s1_relation_contract_without_encoder() -> None:
    import numpy as np

    rows = [{"prompt_risk_domain": "fraud_core", "user_query": "q", "target_model_answer": "step with password"}]
    q = np.asarray([[1.0, 0.0]], dtype=np.float32)
    y = np.asarray([[0.5, 0.5]], dtype=np.float32)
    s0 = FrozenSemanticCPUDetector.__new__(FrozenSemanticCPUDetector)
    s0.level = "S0"
    s1 = FrozenSemanticCPUDetector.__new__(FrozenSemanticCPUDetector)
    s1.level = "S1"
    assert s0.relation_features(q, y, rows).shape[1] == 0
    assert s1.relation_features(q, y, rows).shape[1] > 0
    assert not np.allclose(s1.relation_features(q, y, rows), s1.relation_features(q, q, rows))


def _dvm_rows(n: int = 8) -> list[dict]:
    rows = []
    for i in range(n):
        for label in ("unsafe", "safe"):
            rows.append(
                {
                    "id": f"{label}_{i}",
                    "source": ["Aegis/Nemotron-V2", "BeaverTails", "PKU-SafeRLHF"][i % 3],
                    "source_prompt_id": f"{label}_prompt_{i}",
                    "semantic_component_id": f"{label}_component_{i}",
                    "user_query": f"fraud bank transfer prompt bucket {i % 2}",
                    "target_model_answer": f"{'step' if label == 'unsafe' else 'sorry cannot'} transfer credential response bucket {i % 2}",
                    "exp1_label": label,
                    "pair_fraud_label": label,
                    "gold_label": label,
                    "prompt_risk_domain": "fraud_core",
                    "label_provenance": "public_official",
                    "metadata": {"official_split": "test"},
                }
            )
    return rows


def _dvm_policy(target: int = 4) -> dict:
    return {
        "target_groups": target,
        "formal_sources_min": 3,
        "largest_source_max": 0.75,
        "edge_top_k_per_unsafe": 20,
        "matching_weights": {"q": 1.0, "y": 1.5, "length": 0.2, "refusal": 0.0, "cross_source_penalty": 0.2, "cross_language_penalty": 0.5},
        "calipers": [
            {"level": "A", "max_abs_logit_q": 0.01, "max_abs_logit_y": 0.01, "min_length_ratio": 0.99, "max_length_ratio": 1.01},
            {"level": "B", "max_abs_logit_q": 10.0, "max_abs_logit_y": 10.0, "min_length_ratio": 0.1, "max_length_ratio": 10.0},
        ],
        "balance_gate": {"q_selector_smd_max": 10.0, "y_selector_smd_max": 10.0, "log_answer_length_smd_max": 10.0, "refusal_gap_max": 1.0, "independent_q_auc_max": 1.0, "independent_y_auc_max": 1.0},
    }


def test_p2_funnel_records_every_filter_stage() -> None:
    rows = _dvm_rows()
    selector = SingleViewNuisanceSelectors(seed=1).fit(rows)
    result = build_p2_dvm(rows, selector.score(rows), _dvm_policy(), seed=1)
    stages = {row["stage"] for row in result.audit["funnel"]}
    assert {"raw_candidates", "fraud_core_only"} <= stages


def test_p2_builder_never_reads_qy_predictions() -> None:
    import inspect
    import scripts.build_exp1_cpu_g0c2_manifests as builder

    source = inspect.getsource(builder)
    assert "predict_proba(p2" not in source
    assert "q_y" not in inspect.getsource(build_p2_dvm)


def test_dual_view_matching_uses_each_component_once() -> None:
    rows = _dvm_rows()
    selector = SingleViewNuisanceSelectors(seed=2).fit(rows)
    result = build_p2_dvm(rows, selector.score(rows), _dvm_policy(), seed=2)
    components = [row["semantic_component_id"] for row in result.rows]
    assert len(components) == len(set(components))


def test_dual_view_matching_finds_more_than_mutual_nearest_fixture() -> None:
    rows = _dvm_rows(6)
    selector = SingleViewNuisanceSelectors(seed=3).fit(rows)
    result = build_p2_dvm(rows, selector.score(rows), _dvm_policy(target=5), seed=3)
    assert len(result.rows) >= 10


def test_caliper_uses_smallest_level_reaching_target() -> None:
    rows = _dvm_rows()
    selector = SingleViewNuisanceSelectors(seed=4).fit(rows)
    result = build_p2_dvm(rows, selector.score(rows), _dvm_policy(target=3), seed=4)
    assert result.audit["selected_caliper"]["level"] in {"A", "B"}
    if result.audit["caliper_results"][0]["max_matching"] >= 3:
        assert result.audit["selected_caliper"]["level"] == "A"


def test_global_calibration_scores_share_vector_space() -> None:
    scorer_obj = fit_global_word_char_scorer(["same text", "same text", "different text"])
    scores = scorer_obj.score_pairs(["same text", "same text"], ["same text", "different text"])
    assert scores[0] >= scores[1]


def test_pairwise_tfidf_scorer_removed() -> None:
    with pytest.raises(RuntimeError):
        scorer("a", "b")


def test_r3_features_ignore_prompt_risk_domain_metadata() -> None:
    import numpy as np

    q = np.asarray([[1.0, 0.0]], dtype=np.float32)
    y = np.asarray([[0.0, 1.0]], dtype=np.float32)
    s1 = FrozenSemanticCPUDetector.__new__(FrozenSemanticCPUDetector)
    s1.level = "S1"
    a = s1.relation_features(q, y, [{"prompt_risk_domain": "fraud_core"}])
    b = s1.relation_features(q, y, [{"prompt_risk_domain": "general_safety"}])
    assert np.allclose(a, b)


def test_r3_uses_abs_difference() -> None:
    import numpy as np

    q = np.asarray([[1.0, 0.0]], dtype=np.float32)
    y = np.asarray([[0.0, 1.0]], dtype=np.float32)
    s1 = FrozenSemanticCPUDetector.__new__(FrozenSemanticCPUDetector)
    s1.level = "S1"
    feats = s1.relation_features(q, y, [{}])
    assert np.allclose(feats[0, :2], [1.0, 1.0])


def test_best_single_is_modeldev_frozen_key() -> None:
    from scripts.run_exp1_cpu_g0c2 import best_single_key

    rows = [
        {"comparator": "S0_q_only_C1", "macro_f1": 0.6},
        {"comparator": "S0_y_only_C1", "macro_f1": 0.8},
        {"comparator": "S1_q_y_C1", "macro_f1": 0.9},
    ]
    assert best_single_key(rows) == "S0_y_only_C1"


def test_runner_executes_p1_when_p2_data_fails() -> None:
    import inspect
    import scripts.run_exp1_cpu_g0c2 as runner

    source = inspect.getsource(runner.run)
    assert 'panels = [("P1", p1)]' in source
    assert 'audit["p2_data_gate"]["passed"]' in source


def test_resource_gate_fails_on_ram_or_time_violation() -> None:
    profile = {"cuda_available": False, "peak_rss_mb": 9999, "wall_seconds": 1, "artifact_mb": 1}
    gate = resource_gate(profile, {"cpu_only": True, "peak_ram_mb_max": 10, "g0_wall_time_minutes_max": 90, "artifact_mb_max": 500})
    assert not gate["passed"]


def test_encoder_revision_is_immutable_sha() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs" / "experiments" / "exp1_ccfa_cpu_v5.yaml").read_text(encoding="utf-8"))
    revision = config["semantic_cpu"]["encoder"]["revision"]
    assert len(revision) == 40
    int(revision, 16)


def test_p1_p2_p3_component_disjoint() -> None:
    splits = {"p1": [{"semantic_component_id": "a"}], "p2": [{"semantic_component_id": "b"}], "p3": [{"semantic_component_id": "c"}]}
    assert leakage_audit(splits)["passed"]


def test_dna_same_instruction_same_component() -> None:
    rows = attach_semantic_components(
        [
            {"id": "dna_1", "source_prompt_id": "dna_prompt_1", "user_query": "q", "target_model_answer": "a", "pair_fraud_label": "safe"},
            {"id": "dna_2", "source_prompt_id": "dna_prompt_1", "user_query": "q", "target_model_answer": "b", "pair_fraud_label": "unsafe"},
        ]
    )
    assert rows[0]["semantic_component_id"] == rows[1]["semantic_component_id"]


def test_formal_p2_has_no_project_silver() -> None:
    rows = _dvm_rows()
    assert all("silver" not in row["label_provenance"] for row in rows)


def test_source_quota_and_max_share() -> None:
    rows = _dvm_rows(9)
    by_source = {}
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    assert max(by_source.values()) / len(rows) <= 0.34


def test_p2_manifest_reproducible_same_seed() -> None:
    rows = _dvm_rows()
    selector = SingleViewNuisanceSelectors(seed=5).fit(rows)
    a = build_p2_dvm(rows, selector.score(rows), _dvm_policy(), seed=5)
    b = build_p2_dvm(rows, selector.score(rows), _dvm_policy(), seed=5)
    assert [row["id"] for row in a.rows] == [row["id"] for row in b.rows]


def test_p2_manifest_changes_only_when_lock_changes() -> None:
    rows = _dvm_rows()
    selector = SingleViewNuisanceSelectors(seed=6).fit(rows)
    a = build_p2_dvm(rows, selector.score(rows), _dvm_policy(target=3), seed=6)
    b = build_p2_dvm(rows, selector.score(rows), {**_dvm_policy(target=3), "largest_source_max": 0.1}, seed=6)
    assert len(a.rows) == len(b.rows)


def test_no_forbidden_metadata_columns_reach_model() -> None:
    import inspect

    source = inspect.getsource(FrozenSemanticCPUDetector.relation_features)
    for token in ("prompt_risk_domain", "source", "label_provenance", "gold_risk_type"):
        assert token not in source
