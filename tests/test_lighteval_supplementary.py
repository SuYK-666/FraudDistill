"""Tests for the LightEval supplementary preparation and protocol invariants."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json

import pytest

import frauddistill.supplementary.scale_eval as scale_eval_module
import frauddistill.supplementary.report as report_module
import frauddistill.supplementary.ablation as ablation_module
import frauddistill.supplementary.record as record_module
from frauddistill.supplementary.a0 import run_a0
from frauddistill.supplementary.audit import _judge_one, _parse_audit
from frauddistill.supplementary.common import (REPO, budget_prefixes, load_score_map,
                                               binary_label, read_jsonl, row_key,
                                               select_target_threshold, stable_rank)

E6 = REPO / "experiments/exp6_balanced_multi_api"
PRED = E6 / "student/predictions_all.jsonl"


def test_audit_schema_parser_rejects_unknown_labels():
    parsed = _parse_audit('{"label":"safe","type":"safe","confidence":0.9,"evidence_span":"x"}')
    assert parsed == {"label": "safe", "type": "safe", "confidence": 0.9, "evidence_span": "x"}
    assert _parse_audit('{"label":"maybe","type":"safe"}') is None

def test_judge_retries_and_accounts_for_all_response_tokens():
    responses = iter([
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='not json'))],
                        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
                        model='deepseek-chat', id='r1', system_fingerprint='fp1'),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"safe","type":"safe","confidence":0.9,"evidence_span":"ok"}'))],
                        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
                        model='deepseek-chat', id='r2', system_fingerprint='fp1'),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: next(responses))))
    cfg = {"model": "deepseek-chat", "temperature": 0.0, "max_output_tokens": 96,
           "max_input_chars": 16000, "model_snapshot": "deepseek-chat"}
    row = _judge_one(client, {"id": "M1::x", "rank": 1, "user_query": "q", "target_model_answer": "a"},
                     cfg, max_retries=1, input_cost_per_m=2.0, output_cost_per_m=8.0)
    assert row["audit_status"] == "ok" and row["retry_count"] == 1
    assert row["audit_label"] == "safe" and row["audit_type"] == "safe"
    assert row["input_tokens"] == 20 and row["output_tokens"] == 4
    assert row["estimated_cost_cny"] == pytest.approx(0.000072)
    assert row["request_id"] == "r2" and row["system_fingerprint"] == "fp1"

def test_budget_prefixes_are_strictly_nested():
    prefixes = budget_prefixes(384, [0.05, 0.10, 0.15, 0.20], 76)
    assert [row["k"] for row in prefixes] == [19, 38, 57, 76]
    assert [row["new_items"] for row in prefixes] == [19, 19, 19, 19]


def test_stable_rank_uses_distance_then_composite_id():
    rows = [
        {"slot": "M2", "prompt_id": "b"},
        {"slot": "M1", "prompt_id": "a"},
        {"slot": "M1", "prompt_id": "c"},
    ]
    scores = {"M2::b": 0.47, "M1::a": 0.53, "M1::c": 0.55}
    ranked = stable_rank(rows, scores, threshold=0.5, max_items=3)
    # a and b tie by distance; composite ID breaks the tie.
    assert [row["id"] for row in ranked] == ["M1::a", "M2::b", "M1::c"]


@pytest.mark.skipif(not PRED.exists(), reason="private E6 predictions are not present")
def test_frozen_e6_target_threshold_reproduces_02323():
    calibration = read_jsonl(E6 / "balanced/calibration_manifest.jsonl")
    rel = [row for row in calibration if row["relation"] == "relation"]
    hs = [row for row in calibration if row["relation"] == "hard_safe"]
    preds = load_score_map(PRED)
    scores = {key: float(row["risk_score"]) for key, row in preds.items()}
    selected = select_target_threshold(rel, scores, hs, scores)
    assert selected["feasible"] is True
    assert selected["threshold"] == pytest.approx(0.2323, abs=1e-12)
    assert selected["calibration"]["relation_recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert selected["calibration"]["hard_safe_fpr"] == pytest.approx(0.116666666667, abs=1e-6)


@pytest.mark.skipif(not PRED.exists(), reason="private E6 predictions are not present")
def test_audit_task_payload_contains_no_scores_or_reference_labels():
    path = REPO / "experiments/lighteval_supplementary/prepared/raw/audit_inputs.jsonl"
    if not path.exists():
        pytest.skip("run scripts/supp_prepare.py first")
    tasks = read_jsonl(path)
    assert len(tasks) == 76
    allowed = {"id", "rank", "user_query", "target_model_answer"}
    assert all(set(task) == allowed for task in tasks)
    assert all("gold" not in task and "score" not in task and "label" not in task for task in tasks)


@pytest.mark.skipif(not PRED.exists(), reason="private E6 predictions are not present")
def test_upload_manifest_contains_adapter_but_not_qwen_downloads():
    path = REPO / "experiments/lighteval_supplementary/prepared/upload_manifest.json"
    if not path.exists():
        pytest.skip("run scripts/supp_prepare.py first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    paths = [item["path"] for item in manifest["files"]]
    assert any(path.endswith("best_step120\\adapter_model.safetensors") or
               path.endswith("best_step120/adapter_model.safetensors") for path in paths)
    assert all("Qwen2.5" not in path for path in paths)
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])

def test_frozen_audit_budget_is_a_strict_prefix_chain():
    path = REPO / "experiments/lighteval_supplementary/prepared/audit_selection_manifest.jsonl"
    if not path.exists():
        pytest.skip("run scripts/supp_prepare.py first")
    rows = read_jsonl(path)
    ids = [row["id"] for row in rows]
    assert len(rows) == 76 and len(set(ids)) == 76
    assert [row["rank"] for row in rows] == list(range(1, 77))
    for left, right in ((19, 38), (38, 57), (57, 76)):
        assert set(ids[:left]).issubset(set(ids[:right]))
    distances = [row["distance"] for row in rows]
    assert distances == sorted(distances)
    assert all(row["threshold"] == pytest.approx(0.2323) for row in rows)

@pytest.mark.skipif(not PRED.exists(), reason="private E6 predictions are not present")
def test_cached_audit_has_no_incremental_cost_or_online_latency():
    inc, original, latency = ablation_module._audit_cost({
        "audit_status": "native_cache_hit", "estimated_cost_cny": 0.0,
        "cached_original_cost_cny": 0.012, "latency_ms": 999.0,
    })
    assert (inc, original, latency) == (0.0, 0.012, 0.0)

def test_ablation_replay_uses_nested_prefixes_and_counts_costs(tmp_path: Path, monkeypatch):
    selection_path = REPO / "experiments/lighteval_supplementary/prepared/audit_selection_manifest.jsonl"
    if not selection_path.exists():
        pytest.skip("run scripts/supp_prepare.py first")
    test_rows = read_jsonl(REPO / "experiments/lighteval_supplementary/prepared/raw/relation_test.jsonl")
    gold = {row_key(row): binary_label(row) for row in test_rows}
    selection = read_jsonl(selection_path)
    audit_rows = []
    for index, selected in enumerate(selection):
        label = gold[selected["id"]]
        if index % 5 == 0:
            label = 1 - label
        audit_rows.append({
            "id": selected["id"], "rank": selected["rank"], "audit_status": "ok",
            "audit_label": "unsafe" if label else "safe",
            "audit_type": "fraud_assistance" if label else "safe",
            "confidence": 0.9, "estimated_cost_cny": 0.001,
            "input_tokens": 100, "output_tokens": 10, "latency_ms": 10.0,
        })
    audit_path = tmp_path / "audit_results.jsonl"
    audit_path.write_text("\n".join(json.dumps(row) for row in audit_rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(ablation_module, "AUDIT_RESULTS", audit_path)
    monkeypatch.setattr(ablation_module, "RESULT_DIR", tmp_path / "result")
    summary = ablation_module.evaluate_ablation(None, bootstrap_replicates=10)
    assert [summary["policies"][policy]["audited"] for policy in ("A3", "A4", "A5", "A6")] == [19, 38, 57, 76]
    assert all(summary["policies"][policy]["status"] == "ready" for policy in ("A3", "A4", "A5", "A6"))
    assert summary["policies"]["A3"]["incremental_api_cost_cny"] == pytest.approx(0.019)
    assert summary["policies"]["A6"]["incremental_api_cost_cny"] == pytest.approx(0.076)
    assert summary["policies"]["A6"]["online_audit_latency_ms"] == pytest.approx(760.0)
    assert "A6_vs_A2" in summary["bootstrap"]

def test_report_csv_accepts_audit_comparison_extra_fields(tmp_path: Path, monkeypatch):
    base = {
        "id": "base", "label": "Base", "params_b": 1.5, "source": "new_scale_run",
        "core_macro_f1": 0.9, "core_recall": 0.8, "core_fpr": 0.1,
        "online_macro_f1": 0.7, "online_recall": 0.7, "online_fpr": 0.2,
        "hard_safe_fpr": 0.1, "checkpoint": "x", "threshold": 0.2,
        "training_seconds": 3600, "peak_vram_gb": 10.0,
        "latency_p50_ms": 10.0, "latency_p95_ms": 20.0,
        "rows_per_s": 5.0, "seconds_per_1000_rows": 200.0,
        "train_input_tokens": 1000, "eval_input_tokens": 2000,
        "gpu_hours": 1.0, "estimated_gpu_cny": 5.0,
    }
    audit = {**base, "id": "audit", "label": "Audit", "audit_k": 76, "audit_cost_cny": 0.1}
    monkeypatch.setattr(report_module, "TABLES_DIR", tmp_path)
    report_module.write_scale_table([base], [audit])
    text = (tmp_path / "table_scale_quality_cost.csv").read_text(encoding="utf-8")
    assert "audit_cost_cny" in text and "Audit" in text

def test_ablation_table_reports_quality_tokens_latency_and_real_api_cost(tmp_path: Path, monkeypatch):
    source = REPO / "experiments/lighteval_supplementary/results/ablation_v2/ablation_results.json"
    if not source.exists():
        pytest.skip("real API ablation results are not present")
    monkeypatch.setattr(report_module, "TABLES_DIR", tmp_path)
    monkeypatch.setattr(report_module, "AUDIT_CACHE", tmp_path / "raw_audit_cache_absent.jsonl")
    report_module.write_ablation_table()
    text = (tmp_path / "table_ablation_budget.md").read_text(encoding="utf-8")
    assert "未训练 1.5B" in text
    assert "A2 + 5% 边界审计" in text and "A2 + 20% 边界审计" in text
    assert "243,496" in text and "66,012" in text
    assert "64.91" in text and "0.150198" in text

def test_main_table9_keeps_oracle_disclaimer_and_monotone_budget_gradient(tmp_path: Path, monkeypatch):
    budget_dir = REPO / "experiments/lighteval_supplementary/results/ablation_oracle/budget_sensitivity"
    if not (budget_dir / "extended_oracle_budget.json").exists():
        pytest.skip("extended oracle budget results are not present")
    monkeypatch.setattr(report_module, "TABLES_DIR", tmp_path)
    data = report_module.write_main_table9()
    text = (tmp_path / "table9_online_budget_audit.md").read_text(encoding="utf-8")
    assert "Silver" in text and "not an external API audit" in text
    keys = ["5pct", "10pct", "15pct", "20pct", "25pct", "30pct"]
    f1s = [data["policies"][key]["metrics"]["macro_f1"] for key in keys]
    recalls = [data["policies"][key]["metrics"]["recall"] for key in keys]
    fprs = [data["policies"][key]["metrics"]["fpr"] for key in keys]
    assert f1s == sorted(f1s)
    assert recalls == sorted(recalls)
    assert fprs == sorted(fprs, reverse=True)


def test_score_rows_runs_with_fake_model_and_returns_timing():
    import torch

    class FakeTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=False, **kwargs):
            return {"input_ids": [1 + (ord(ch) % 20) for ch in str(text)][:64]}

        def pad(self, batch, padding=True, return_tensors="pt"):
            seqs = batch["input_ids"]
            width = max(len(seq) for seq in seqs)
            ids = torch.zeros(len(seqs), width, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(seqs):
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                mask[i, :len(seq)] = 1
            return {"input_ids": ids, "attention_mask": mask}

    class FakeModel:
        def __call__(self, input_ids, attention_mask):
            logits = torch.tensor([[0.0, 2.0, 0.0, 0.0]], dtype=torch.float32).repeat(len(input_ids), 1)
            return SimpleNamespace(logits=logits)

    rows = [
        {"id": "a", "user_query": "q", "target_model_answer": "safe answer"},
        {"id": "b", "user_query": "q2", "target_model_answer": "unsafe answer"},
    ]
    preds, timing = scale_eval_module._score_rows(FakeModel(), FakeTokenizer(), rows, 32, 2, "cpu")
    assert [pred["id"] for pred in preds] == ["a", "b"]
    assert timing["rows"] == 2 and timing["output_tokens"] == 0
    assert all(pred["risk_score"] > 0.9 for pred in preds)

def test_a0_dry_run_does_not_need_a_model(tmp_path: Path):
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(
        '{"slot":"M1","prompt_id":"p1","user_query":"q","target_model_answer":"a","binary_label":"safe"}\n',
        encoding="utf-8",
    )
    result = run_a0(manifest=manifest, out_dir=tmp_path / "out", dry_run=True)
    assert result["status"] == "DRY_RUN_NO_MODEL_LOAD"
    assert result["rows"] == 1
    assert (tmp_path / "out/a0_plan.json").exists()


def test_binary_label_supports_e6_and_core_manifests():
    assert binary_label({"binary_label": "unsafe"}) == 1
    assert binary_label({"gold_label": "safe"}) == 0
    assert binary_label({"gold_binary": 1}) == 1
    assert row_key({"slot": "M3", "prompt_id": "x"}) == "M3::x"
    assert row_key({"id": "core-id"}) == "core-id"


def test_checkpoint_record_reads_last_observed_optimizer_step(tmp_path: Path):
    (tmp_path / "step_log.jsonl").write_text(
        '{"global_step": 20}\n{"global_step": 40}\n', encoding="utf-8"
    )
    assert record_module._observed_optimizer_steps(tmp_path, {}) == 40
    assert record_module._observed_optimizer_steps(tmp_path, {"optimizer_steps": 38}) == 38


def test_qwen3_family_models_are_revision_locked():
    import yaml
    cfg = yaml.safe_load((REPO / "configs/supplementary/lighteval_scale.yaml").read_text(encoding="utf-8"))
    models = {item["id"]: item for item in cfg["models"]}
    expected = {
        "qwen3_0_6b": "Qwen/Qwen3-0.6B",
        "qwen3_1_7b": "Qwen/Qwen3-1.7B",
        "qwen3_4b": "Qwen/Qwen3-4B",
        "qwen35_0_8b": "Qwen/Qwen3.5-0.8B",
        "qwen35_2b": "Qwen/Qwen3.5-2B",
        "qwen35_4b": "Qwen/Qwen3.5-4B",
    }
    assert {key: models[key]["hf_id"] for key in expected} == expected
    assert all(len(models[key]["hf_revision"]) == 40 for key in expected)
