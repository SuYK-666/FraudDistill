import importlib.util
import sys
from pathlib import Path

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_high_standard_rerun.py"
    spec = importlib.util.spec_from_file_location("run_high_standard_rerun", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def protocol_rows(n_groups: int = 24) -> list[dict]:
    rows = []
    for group in range(n_groups):
        for idx in range(2):
            label = "unsafe" if group % 2 else "safe"
            rows.append(
                {
                    "id": f"g{group}_{idx}",
                    "split_group": f"g{group}",
                    "prompt_hash": f"p{group}",
                    "answer_hash": f"a{group}_{idx}",
                    "source": "synthetic",
                    "language": "zh" if group % 3 else "en",
                    "fraud_category": "phishing_scams",
                    "user_query": f"query {group}",
                    "target_model_answer": "send verification code" if label == "unsafe" else "report it safely",
                    "label": label,
                    "gold_label": label,
                    "reference_type": "fixture",
                }
            )
    return rows


def test_split_grouped_preserves_all_rows_and_groups():
    runner = load_runner()
    rows = protocol_rows()
    train, dev, test = runner.split_grouped(rows)
    assert len(train) + len(dev) + len(test) == len(rows)
    runner.assert_split_integrity({"train": train, "dev": dev, "test": test}, len(rows))


def test_calibration_split_is_group_disjoint():
    runner = load_runner()
    rows = protocol_rows(40)
    train, calib, threshold, test = runner.split_for_calibration(rows)
    assert len(train) + len(calib) + len(threshold) + len(test) == len(rows)
    runner.assert_split_integrity({"train": train, "calibration": calib, "threshold": threshold, "test": test}, len(rows))


def test_distilled_student_inference_does_not_call_teacher(monkeypatch):
    runner = load_runner()
    rows = protocol_rows(16)
    model = runner.train_distill_model(rows, "full")

    def fail_teacher(_row):
        raise AssertionError("teacher must not run at inference time")

    monkeypatch.setattr(runner, "weak_teacher", fail_teacher)
    scores = runner.score_model(model, rows[:4], "q+y")
    assert len(scores) == 4


def test_exp6_pair_label_recomputed_from_response():
    runner = load_runner()
    dangerous_refusal = {
        "gold_label": "unsafe",
        "target_model_answer": "Sorry, I cannot help with phishing or credential theft.",
    }
    safe_but_bad_response = {
        "gold_label": "safe",
        "target_model_answer": "First ask for the password and verification code, then send a login link.",
    }
    assert runner.behavior_from_response(dangerous_refusal, "unsafe", 0.9)["pair_fraud_label"] == "safe"
    assert runner.behavior_from_response(safe_but_bad_response, "safe", 0.1)["response_harm_label"] == "unsafe"
