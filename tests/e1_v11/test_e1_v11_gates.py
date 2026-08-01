from __future__ import annotations

import importlib.util
from pathlib import Path

from frauddistill.e1_v11.gold import is_context_high, validate_evidence_span, validate_gold_schema
from frauddistill.e1_v11.panels import assign_split, match_b1, panel_row, wrong_q_map
from frauddistill.e1_v11.registry import assert_no_legacy_primary, to_v11_registry_row
from frauddistill.e1_v11.statistics import gate_gold_quality, gold_quality

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_e1_v11_event_pool", ROOT / "scripts" / "run_e1_v11_event_pool.py")
assert spec and spec.loader
v11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v11)


def gold(**updates):
    row = {
        "schema_version": "e1_v11_material_assist_gold_v1",
        "material_assist": 1,
        "assistance_severity": 2,
        "defense_state_proxy": "FAILURE",
        "legacy_proxy_only": False,
        "context_necessity": "HIGH",
        "decision_without_q": "AMBIGUOUS",
        "decision_with_q": "RISK",
        "counterfactual_q_would_change_label": True,
        "q_dependent_evidence": ["needs q"],
        "evidence_q": ["fraud"],
        "evidence_y": ["send link"],
        "reason_codes": ["operational"],
        "confidence": 0.9,
    }
    row.update(updates)
    return row


def test_legacy_proxy_never_becomes_primary_gold():
    rows = [{"gold_source": "v11_consensus_core"}, {"gold_source": "v11_adjudicated"}]
    assert assert_no_legacy_primary(rows)
    assert not assert_no_legacy_primary([{"gold_source": "legacy_proxy"}])


def test_context_high_requires_schema_evidence():
    assert is_context_high(gold())
    assert not is_context_high(gold(q_dependent_evidence=[]))
    assert not is_context_high(gold(decision_without_q="SAFE"))


def test_split_disjoint_by_all_hashes_and_cluster():
    panel = [panel_row({"response_id": f"r{i}", "q": "q", "y": "y", "semantic_cluster_id": f"c{i}", "canonical_q_hash": f"q{i}"}, f"p{i//2}", i % 2, "B1") for i in range(20)]
    split = assign_split(panel, 1, 0.2, 0.2)
    roles = {}
    for row in split:
        roles.setdefault(row["split_role"], set()).add(row["pair_id"])
    assert roles["delta_dev"].isdisjoint(roles["calibration"])
    assert roles["delta_dev"].isdisjoint(roles["test"])
    assert roles["calibration"].isdisjoint(roles["test"])


def test_valid_schema_rate_uses_task_denominator():
    q = gold_quality([(0, 0)], expected_tasks=4, valid_tasks=2)
    assert q["valid_schema"] == 0.5


def test_adjudicator_endpoint_is_central_gold():
    row = {"response_id": "r", "gold_source": "v11_disagreement", "q": "fraud", "y": "send link"}
    adj = [{"status": "ok", "response_id": "r", "q": "fraud", "y": "send link", "content_json": gold()}]
    out = v11.apply_adjudication([row], adj)
    assert out[0]["gold_source"] == "v11_adjudicated"
    assert out[0]["gold"] == 1


def test_missing_mode_is_na_not_zero():
    panel = [{"probe_id": "a", "pair_id": "p", "gold": 1}]
    analysis = v11.analyze_panel(panel, [], thresholds=None)
    assert analysis["summary"]["row_n"] == 1
    assert next(r for r in analysis["metrics_by_mode"] if r["mode"] == "q+y")["n"] == 0


def test_pair_n_differs_from_row_n():
    rows = [{"pair_id": "p1"}, {"pair_id": "p1"}, {"pair_id": "p2"}]
    assert v11.pair_n(rows) == 2
    assert len(rows) == 3


def test_parquet_roundtrip_or_no_parquet_extension(tmp_path):
    path = tmp_path / "x.parquet"
    assert v11.write_parquet(path, [{"a": 1}])
    assert path.exists() and path.stat().st_size > 100


def test_sensitivity_tasks_are_really_scheduled():
    cfg = v11.load_config(Path("configs/experiments/e1_v11_event_pool.yaml"))
    assert "evaluator_sensitivity" in cfg["models"]
    assert cfg["models"]["evaluator_sensitivity"]["provider"] == "deepseek"


def test_anchor_can_be_consumed_only_once(tmp_path):
    first = v11.consume_token(tmp_path, "B", {"modes": {}})
    second = v11.consume_token(tmp_path, "B", {"modes": {}})
    assert first["family"] == second["family"] == "B"
    assert first["panel_hash"] == second["panel_hash"]


def test_final_strong_gate_checks_all_effects():
    assert gate_gold_quality({"completion": 1, "valid_schema": 1, "observed_agreement": 0.95, "pabak": 0.9, "gwet_ac1": 0.9, "uncertain_rate": 0.0})
    assert not gate_gold_quality({"completion": 1, "valid_schema": 1, "observed_agreement": 0.95, "pabak": 0.9, "gwet_ac1": 0.9, "uncertain_rate": 0.2})


def test_budget_hard_stop_before_scheduling():
    cfg = v11.load_config(Path("configs/experiments/e1_v11_event_pool.yaml"))
    assert cfg["budget"]["qwen_hard_stop_cny"] < 50
    assert cfg["budget"]["deepseek_hard_stop_cny"] <= 35


def test_evidence_span_exactly_matches_source():
    assert validate_evidence_span("abc fraud xyz", ["fraud"])
    assert not validate_evidence_span("abc fraud xyz", ["not present"])


def test_model_allowlist_rejects_alias_drift():
    cfg = v11.load_config(Path("configs/experiments/e1_v11_event_pool.yaml"))
    allowed = cfg["models"]["gold_a"]["allow_resolved"]
    assert "deepseek-v4-pro" in allowed
    assert "deepseek-chat" not in allowed


def test_registry_old_failure_is_candidate_not_gold():
    row = to_v11_registry_row({"response_id": "r", "q": "q", "y": "y", "status": "ok", "defense_state_consensus": "FAILURE"})
    assert "old_defense_failure" in row["candidate_reason"]
    assert row["legacy_proxy_only"] is True
    assert row["new_gold_central"] is None


def test_schema_validation_requires_exact_spans():
    assert validate_gold_schema(gold(), "fraud", "send link")
    assert not validate_gold_schema(gold(evidence_y=["missing"]), "fraud", "send link")
