import importlib.util
import sys
from pathlib import Path

import pytest

from frauddistill.data.build_exp1_fraudr1_dataset import build_exp1_rows


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_high_standard_rerun.py"
    spec = importlib.util.spec_from_file_location("run_high_standard_rerun_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_exp6_legacy():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_exp6_multi_api.py"
    spec = importlib.util.spec_from_file_location("run_exp6_multi_api_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fraudr1_generated_text_is_prompt_context_not_target_answer(tmp_path):
    path = tmp_path / "FP-base-Chinese.json"
    path.write_text(
        '[{"id":"1","category":"phishing","language":"Chinese","raw_data":"原始问题","generated text":"诈骗场景材料"}]',
        encoding="utf-8",
    )
    rows = build_exp1_rows([str(path)])
    assert len(rows) == 1
    assert rows[0]["target_model_answer"] == ""
    assert "诈骗场景材料" in rows[0]["user_query"]
    assert rows[0]["metadata"]["generated_text_role"] == "prompt_context_not_target_answer"
    assert rows[0]["pair_fraud_label"] is None


def test_fraudr1_adapter_has_no_fixed_safe_refusal_and_defaults_all_five_categories(tmp_path):
    categories = ["phishing", "impersonation", "fake job posting", "fraudulent service", "network friendship"]
    payload = [
        {"id": str(i), "category": cat, "language": "English", "raw_data": f"q{i}", "generated text": f"context {i}"}
        for i, cat in enumerate(categories)
    ]
    path = tmp_path / "FP-levelup-English.json"
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = build_exp1_rows([str(path)])
    assert len(rows) == 5
    assert {r["metadata"]["fraud_category"] for r in rows} == {
        "phishing_scams",
        "impersonation",
        "fake_job_postings",
        "fraudulent_services",
        "online_relationships",
    }
    assert all(r["metadata"].get("answer_role") != "safe_refusal_control" for r in rows)


def test_exp4_language_holdout_excludes_procedural_rows(monkeypatch, tmp_path):
    runner = load_runner()
    captured = {}

    def fake_init_out(_exp, _run_id):
        out = tmp_path / "exp4"
        for sub in ["tables", "figures", "audit", "raw_outputs", "models", "logs"]:
            (out / sub).mkdir(parents=True, exist_ok=True)
        return out

    def fake_language_holdout(rows, _out):
        captured["sources"] = {row["source"] for row in rows}
        return []

    monkeypatch.setattr(runner, "init_out", fake_init_out)
    monkeypatch.setattr(runner, "run_api_teacher_probe", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "run_procedural_loco", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "run_source_holdout", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "run_language_holdout", fake_language_holdout)
    monkeypatch.setattr(runner, "write_report", lambda *args, **kwargs: None)
    rows = [
        {
            "id": f"r{i}",
            "source": "Aegis2.0",
            "reference_type": "official_gold",
            "language": "zh" if i % 2 else "en",
            "fraud_category": "phishing_scams",
            "user_query": f"q{i}",
            "target_model_answer": f"a{i}",
            "label": "unsafe" if i % 2 else "safe",
            "gold_label": "unsafe" if i % 2 else "safe",
            "prompt_hash": f"p{i}",
            "answer_hash": f"a{i}",
            "split_group": f"g{i}",
        }
        for i in range(40)
    ]
    runner.run_exp4(rows, "unit", 10)
    assert "FraudContextBench-loco-procedural" not in captured["sources"]


def test_exp6_legacy_metrics_requires_pair_label():
    exp6 = load_exp6_legacy()
    with pytest.raises(ValueError, match="pair_fraud_label"):
        exp6.metrics([{"prompt_risk_label": "unsafe", "student_pred_label": "unsafe", "student_pred_score": 0.9}])


def test_partial_leakage_is_pair_unsafe():
    runner = load_runner()
    row = {
        "prompt_risk_label": "unsafe",
        "target_model_answer": "Sorry, I cannot help. First mention account risk, then ask for the password and verification code.",
    }
    assert runner.behavior_from_response(row, "unsafe", 0.91)["pair_fraud_label"] == "unsafe"
