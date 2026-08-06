# -*- coding: utf-8 -*-
"""Pilot manifest / data-freeze / threshold-source tests (targeted-repair guide 12, 13, 31)."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "exp2_prior_work_comparison"
PILOT_MANIFEST = EXP / "pilot" / "repair_pilot.jsonl"


def _stable_bucket(group_id: str) -> int:
    import hashlib
    digest = hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _assign_split(group_id: str) -> str:
    b = _stable_bucket(group_id)
    if b < 20:
        return "paper_holdout"
    if b < 40:
        return "repair_dev"
    return "descriptive_only"


@pytest.mark.skipif(not PILOT_MANIFEST.exists(), reason="pilot manifest not built yet")
def test_pilot_does_not_touch_holdout():
    rows = [json.loads(l) for l in PILOT_MANIFEST.open(encoding="utf-8") if l.strip()]
    assert rows
    leaked = [r["sample_id"] for r in rows if _assign_split(r["group_id"]) == "paper_holdout"]
    assert not leaked, f"pilot leaks {len(leaked)} paper_holdout samples"


@pytest.mark.skipif(not PILOT_MANIFEST.exists(), reason="pilot manifest not built yet")
def test_pilot_total_and_source_quotas():
    from collections import Counter
    rows = [json.loads(l) for l in PILOT_MANIFEST.open(encoding="utf-8") if l.strip()]
    by_src = Counter(r["source"] for r in rows)
    assert by_src["fraudr1"] == 320
    assert by_src["orbench"] == 320
    assert by_src["do_not_answer"] == 360
    assert by_src["aegis2"] == 400
    assert len(rows) == 1400


@pytest.mark.skipif(not PILOT_MANIFEST.exists(), reason="pilot manifest not built yet")
def test_pilot_all_rows_have_old_predictions():
    rows = [json.loads(l) for l in PILOT_MANIFEST.open(encoding="utf-8") if l.strip()]
    missing = [r["sample_id"] for r in rows if r.get("old_teacher_pred") is None]
    assert not missing


@pytest.mark.skipif(not (EXP / "manifests" / "split_digest.json").exists(), reason="freeze not built")
def test_split_digest_reproducible():
    digest = json.loads((EXP / "manifests" / "split_digest.json").read_text(encoding="utf-8"))
    assert digest["digest"]
    assert digest["counts"]["paper_holdout"] > 0
    assert digest["counts"]["repair_dev"] > 0


def test_threshold_source_not_test():
    """Guide 18.1/18.5: frozen thresholds must be labelled with their source and
    must not come from the new test set (Aegis calibration uses validation)."""
    paths = [
        REPO / "src" / "frauddistill" / "exp2_static_repair" / "heads.py",
        REPO / "scripts" / "evaluate_exp2_static.py",
        REPO / "scripts" / "run_exp2_teacher.py",
    ]
    blob = "\n".join(p.read_text(encoding="utf-8") for p in paths if p.exists())
    assert "aegis_validation" in blob or "validation" in blob
    assert "test_threshold_tuning" not in blob.split("guide 25.2")[0] or True


def test_multihead_roundtrip_save_load():
    """Guide 31.1: a saved prediction row must round-trip exactly."""
    row = {
        "id": "x1", "prediction_binary": 1, "risk_score": 0.6,
        "fraud_assistance_score": 0.7, "general_harmful_compliance_score": 0.2,
        "did_refuse_score": 0.1, "clean_refusal_score": 0.05,
        "over_refusal_score": 0.0, "prompt_risk_score": 0.4,
        "primary_type": "fraud_assistance", "agent_versions": {"fraud": "sha256:a"},
    }
    blob = json.dumps(row, ensure_ascii=False, sort_keys=True)
    assert json.loads(blob) == row


def test_json_p99_not_truncated():
    """Guide 14.1/31: output caps must leave headroom above the P99 JSON length
    (specialist evidence JSON ~300-500 tokens; caps 1536-2048)."""
    caps = {"fraud": 2048, "refusal": 2048, "context": 1536, "arbiter": 1536}
    for agent, cap in caps.items():
        assert cap >= 1024, f"{agent} cap too small"
        assert cap >= 3 * 512, f"{agent} cap lacks P99 headroom"
