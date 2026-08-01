from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

from frauddistill.e1_v10.metrics import auprc, ece, gwet_ac1, positive_agreement

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_e1_v10_trilayer", ROOT / "scripts" / "run_e1_v10_trilayer.py")
assert spec and spec.loader
v10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v10)


def test_threshold_is_really_applied() -> None:
    panel = [{"probe_id": "a", "pair_id": "p1", "gold": 1}, {"probe_id": "b", "pair_id": "p1", "gold": 0}]
    preds = [
        {"status": "ok", "probe_id": "a", "pair_id": "p1", "mode": "q+y", "content_json": {"risk_probability": 0.7}},
        {"status": "ok", "probe_id": "b", "pair_id": "p1", "mode": "q+y", "content_json": {"risk_probability": 0.4}},
    ]
    high = v10.analyze_panel({"stats": {"bootstrap_iterations": 10, "bootstrap_seed": 1}}, panel, preds, thresholds={"modes": {"q+y": {"threshold": 0.8}}})
    low = v10.analyze_panel({"stats": {"bootstrap_iterations": 10, "bootstrap_seed": 1}}, panel, preds, thresholds={"modes": {"q+y": {"threshold": 0.5}}})
    assert v10.metric_by_mode(high, "q+y", "tp") == 0
    assert v10.metric_by_mode(low, "q+y", "tp") == 1


def test_q_only_expands_to_pair_accuracy_half() -> None:
    panel = [
        {"probe_id": "p_pos", "pair_id": "p", "gold": 1},
        {"probe_id": "p_neg", "pair_id": "p", "gold": 0},
    ]
    preds = [{"status": "ok", "evaluator_key": "eval", "pair_id": "p", "mode": "q-only", "content_json": {"risk_probability": 0.9}}]
    expanded = v10.expand_q_only(panel, preds)
    rows = []
    for row in expanded:
        rows.append({"gold": row["gold"], "pred": int(row["content_json"]["risk_probability"] >= 0.5), "score": row["content_json"]["risk_probability"]})
    from frauddistill.e1_v10.metrics import binary_metrics

    assert binary_metrics(rows)["accuracy"] == 0.5


def test_wrong_q_uses_different_canonical() -> None:
    panel = [
        {"probe_id": "a", "canonical_id": "c1", "q_sha256": "q1", "language": "en", "category": "x", "q": "q1"},
        {"probe_id": "b", "canonical_id": "c2", "q_sha256": "q2", "language": "en", "category": "x", "q": "q2"},
    ]
    mapping = v10.wrong_q_map(panel)
    assert mapping
    assert all(row["wrong_q_sha256"] != next(p["q_sha256"] for p in panel if p["probe_id"] == row["probe_id"]) for row in mapping)


def test_gold_reliability_metrics_low_prevalence() -> None:
    pairs = [(0, 0)] * 98 + [(1, 1)] * 1 + [(1, 0)] * 1
    assert positive_agreement(pairs) < 1.0
    assert gwet_ac1(pairs) > 0.9


def test_auprc_and_ece_fixtures() -> None:
    assert abs(auprc([1, 0, 1], [0.5, 0.5, 0.1]) - (7 / 12)) < 1e-12
    assert abs(ece([1, 0], [0.9, 0.9], bins=1) - 0.4) < 1e-12


def test_budget_jsonl_threaded_writes_are_valid(tmp_path: Path) -> None:
    config = {
        "budget": {
            "pricing_usd_per_million": {"m": {"input": 1, "output": 1}},
            "usd_to_cny": 1,
            "qwen_hard_stop_cny": 1,
            "deepseek_hard_stop_cny": 1,
        }
    }
    v10.init_budget(tmp_path)

    def write_one(i: int) -> None:
        v10.append_budget(tmp_path, config, {"phase": "x", "provider": "qwen", "requested_model": "m", "response_model": "m", "request_id": str(i), "status": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "task_id": str(i)})

    threads = [threading.Thread(target=write_one, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    audit = v10.audit_jsonl_lines(tmp_path / "E1_V10_BUDGET_LEDGER.jsonl")
    assert audit["lines"] == 100
    assert audit["invalid_json"] == 0
    for line in (tmp_path / "E1_V10_BUDGET_LEDGER.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_pair_consensus_exports_gold_field() -> None:
    target = [{"response_id": "r1", "q": "q", "text": "y"}]
    label = {
        "schema_version": "e1_v10_contextual_gold_v1",
        "defense_state": "FAILURE",
        "assistance_severity": 2,
        "actionable_assistance": True,
        "trust_or_normalization": False,
        "fraud_explicitly_recognized": False,
        "precaution_present": False,
        "context_necessity": "HIGH",
        "evidence_q": ["q"],
        "evidence_y": ["y"],
        "reason_codes": ["material"],
        "confidence": 0.9,
    }
    labels = [
        {"status": "ok", "response_id": "r1", "labeler_key": "a", "content_json": label},
        {"status": "ok", "response_id": "r1", "labeler_key": "b", "content_json": {**label, "assistance_severity": 3}},
    ]
    rows = v10.build_pair_source_consensus(target, labels)
    assert rows[0]["material_assist"] == 1
    assert rows[0]["gold"] == 1
