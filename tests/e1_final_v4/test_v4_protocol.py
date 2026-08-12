# -*- coding: utf-8 -*-
"""E1-FINAL-TRIAD-v4 protocol gate tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v4.judge_views import _visible
from frauddistill.e1_final_v4.detectors import _text
from frauddistill.e1_final_v4.gold import consensus_from_votes, needs_adjudication
from frauddistill.e1_final_v3.budget import hard_stop_decision


def test_q_only_payload_has_no_y():
    t = _text("q_only", "the query", "the answer")
    assert "the answer" not in t


def test_y_only_payload_has_no_q():
    t = _text("y_only", "the query", "the answer")
    assert "the query" not in t


def test_wrong_q_y_uses_wrong_q():
    t = _text("wrong_q_y", "correct q", "answer", wrong_q="wrong q")
    assert "wrong q" in t and "correct q" not in t


def test_judge_view_visible_rules():
    v, rule = _visible("q_only", "qtext", "ytext")
    assert "ytext" not in v and "qtext" in v
    v, rule = _visible("y_only", "qtext", "ytext")
    assert "qtext" not in v and "ytext" in v
    v, rule = _visible("wrong_q_y", "qtext", "ytext", wrong_q="wq")
    assert "wq" in v and "qtext" not in v


def test_gold_missing_not_negative():
    c = consensus_from_votes(None, None)
    assert c["label"] is None and c["needs_adjudication"] is True


def test_needs_adjudication_on_disagree():
    va = {"material_central": 1, "material_lower": 1, "material_upper": 1, "uncertain": False}
    vb = {"material_central": 0, "material_lower": 0, "material_upper": 0, "uncertain": False}
    assert needs_adjudication(va, vb) is True
    assert consensus_from_votes(va, vb)["needs_adjudication"] is True


def test_hard_stop_budget_qwen():
    limits = {"hard_stop_total_cny": 100, "qwen_hard_stop_cny": 60, "deepseek_hard_stop_cny": 60}
    ledger = [{"provider": "qwen", "cost_cny": 61}]
    assert hard_stop_decision(ledger, limits)["hard_stop"] is True
    ledger2 = [{"provider": "deepseek", "cost_cny": 59}, {"provider": "qwen", "cost_cny": 10}]
    assert hard_stop_decision(ledger2, limits)["hard_stop"] is False


def test_c_family_independence_on_split():
    # Final protocol (v4.5 amendment): B reuses A canonical cases by design; E1-C must
    # only use A canonical cases that never entered the B panel (guide §10.2, case-level).
    # Frozen C reserve = 624 rows / 6 positives (see E1_FINAL_REPORT §8.2).
    v32_dir = ROOT / "data" / "prepared" / "e1_final_triad_v32"
    if not (v32_dir / "E1_V32_REAL_POOL.jsonl").exists():
        pytest.skip("data not present")
    out_dir = ROOT / "data" / "prepared" / "e1_final_triad_v4"
    if not (out_dir / "E1_V4_PANEL_ALL.jsonl").exists():
        pytest.skip("panel not yet built")
    import json as _json
    panel = [ _json.loads(l) for l in (out_dir / "E1_V4_PANEL_ALL.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    a_rows = [ _json.loads(l) for l in (v32_dir / "E1_V32_REAL_POOL.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    b_cases = {r.get("canonical_case_id") for r in panel if r.get("canonical_case_id")}
    c_rows = [r for r in a_rows if r.get("canonical_case_id") not in b_cases]
    c_cases = {r.get("canonical_case_id") for r in c_rows if r.get("canonical_case_id")}
    overlap = c_cases & b_cases
    assert len(overlap) == 0, f"C reuses B families: {list(overlap)[:10]}"
    assert len(c_rows) == 624, f"C reserve rows = {len(c_rows)}, expected 624"
    pos = sum(1 for r in c_rows if int(r.get("gold_central", 0) or 0) == 1)
    assert pos == 6, f"C reserve positives = {pos}, expected 6"
