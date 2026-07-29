from __future__ import annotations

from frauddistill.exp1_restored.manifest import normalize_restored_row
from frauddistill.exp1_restored.stats import decision_from_stats
from frauddistill.experiments.run_exp1_single_judge_api import _attach_gold


def test_api_failure_is_not_filled_as_safe() -> None:
    row = {"id": "x", "gold_label": "unsafe"}
    out = _attach_gold(row, {"status": "failed", "pred_label": None, "pred_score": None, "pred_type": None})
    assert out["status"] == "failed"
    assert out["pred_label"] is None


def test_restored_bucket_aliases_v1_hard_unsafe() -> None:
    row = normalize_restored_row({"id": "x", "gold_label": "unsafe", "metadata": {"data_role": "hard_unsafe_qwen_fn_or_phishing"}}, "p")
    assert row["bucket"] == "hard_unsafe"


def test_full_decision_stops_when_order_is_wrong() -> None:
    stats = {
        "metrics": {
            "q_only": {"macro_f1": 0.9},
            "y_only": {"macro_f1": 0.8},
            "q_y": {"macro_f1": 0.81, "recall": 0.8, "precision": 0.8, "fpr": 0.2, "auprc": 0.8},
        },
        "bootstrap": {"delta.q_y-y_only": {"low": 0.0}},
    }
    gates = {
        "q_y_macro_f1_min": 0.88,
        "y_only_macro_f1_min": 0.78,
        "q_only_macro_f1_max": 0.70,
        "q_y_minus_y_min": 0.03,
        "y_minus_q_min": 0.15,
        "q_y_minus_y_ci_lower_min": 0.01,
        "q_y_recall_min": 0.85,
        "q_y_precision_min": 0.88,
        "q_y_fpr_max": 0.08,
        "q_y_auprc_min": 0.9,
    }
    assert decision_from_stats(stats, gates)["decision"] == "E1_STOP"
