# -*- coding: utf-8 -*-
"""Delta planner tests (targeted-repair guide section 17, 31)."""
import pytest

from frauddistill.exp2_static_repair.delta_planner import (
    AGENT_NAMES,
    agent_versions,
    agents_to_rerun,
    merge_agent_outputs,
    prediction_digest,
    qy_hash,
)


def test_agent_versions_has_all_four_agents():
    v = agent_versions()
    assert set(v) == set(AGENT_NAMES)
    for agent in AGENT_NAMES:
        assert v[agent].startswith("sha256:")


def test_agent_versions_stable_within_process():
    assert agent_versions() == agent_versions()


def test_delta_planner_only_invalidates_changed_agents():
    new = agent_versions()
    old = dict(new)
    old["fraud"] = "sha256:old_fraud_digest"
    changed = agents_to_rerun(old, new)
    assert "fraud" in changed
    assert "arbiter" in changed          # arbiter re-runs when a specialist changed
    assert "refusal" not in changed
    assert "context" not in changed


def test_delta_planner_arbiter_only_change():
    new = agent_versions()
    old = dict(new)
    old["arbiter"] = "sha256:old_arbiter_digest"
    changed = agents_to_rerun(old, new)
    assert changed == ["arbiter"]


def test_delta_planner_no_changes():
    assert agents_to_rerun(agent_versions(), agent_versions()) == []


def test_delta_planner_empty_old_reruns_all():
    assert agents_to_rerun(None, agent_versions()) == list(AGENT_NAMES)


def test_merge_agent_outputs_replaces_only_changed():
    old_row = {
        "id": "x1",
        "agent_outputs": {"fraud": {"parsed": {"a": 1}}, "refusal": {"parsed": {"b": 1}},
                          "context": {"parsed": {"c": 1}}},
        "arbiter_output": {"parsed": {"old": 1}},
        "risk_score": 0.5,
    }
    partial = {
        "agent_outputs": {"fraud": {"parsed": {"a": 2}}},
        "arbiter_output": {"parsed": {"new": 1}},
    }
    merged = merge_agent_outputs(old_row, partial)
    assert merged["agent_outputs"]["fraud"]["parsed"]["a"] == 2
    assert merged["agent_outputs"]["refusal"]["parsed"]["b"] == 1
    assert merged["agent_outputs"]["context"]["parsed"]["c"] == 1
    assert merged["arbiter_output"]["parsed"]["new"] == 1
    assert merged["risk_score"] == 0.5


def test_qy_hash_deterministic():
    assert qy_hash("q1", "a1") == qy_hash("q1", "a1")
    assert qy_hash("q1", "a1") != qy_hash("q2", "a1")


def test_prediction_digest_stable():
    row = {"id": "x", "prediction_binary": 1, "risk_score": 0.7, "prediction_type": "fraud_assistance"}
    assert prediction_digest(row) == prediction_digest({**row, "created_at": "now"})
