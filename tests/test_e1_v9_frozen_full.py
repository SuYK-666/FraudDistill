from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_e1_v9_frozen_full", ROOT / "scripts" / "run_e1_v9_frozen_full.py")
assert spec and spec.loader
v9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9)


def config() -> dict:
    return yaml.safe_load(v9.CONFIG_PATH.read_text(encoding="utf-8"))


def test_runner_has_no_duplicate_top_level_functions() -> None:
    audit = v9.duplicate_functions([ROOT / "scripts" / "run_e1_v9_frozen_full.py"])
    assert audit["duplicates"] == {}


def test_prompt_parity_hides_only_q_and_y() -> None:
    row = {"q": "same q", "y": "answer y"}
    audit = v9.prompt_parity_audit(row)
    assert audit["passed"]


def test_rich_schema_validation_and_evidence_spans() -> None:
    js = {
        "defense_state": "FAILURE",
        "assistance_severity": 2,
        "actionable_assistance": True,
        "trust_or_normalization": True,
        "fraud_explicitly_recognized": False,
        "precaution_present": False,
        "context_necessity": "HIGH",
        "evidence_q": ["fraud request"],
        "evidence_y": ["send the money"],
        "confidence": 0.9,
        "reason_codes": ["trust"],
        "schema_version": "e1_v9_contextual_gold_v1",
    }
    assert v9.validate_rich_schema(js)
    assert v9.evidence_valid("fraud request here", "please send the money", js)
    assert not v9.evidence_valid("other", "please send the money", js)


def test_latest_ok_by_task_rejects_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "a", "status": "ok", "content_json": {"parse_error": True}}),
                json.dumps({"task_id": "b", "status": "ok", "content_json": {"prediction": "SAFE"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    latest = v9.latest_ok_by_task(path)
    assert "a" not in latest
    assert "b" in latest


def test_split_overlap_zero_on_reuse_data() -> None:
    cfg = config()
    old = ROOT / cfg["data"]["old_v8_dir"]
    if not (old / "FROZEN_ANCHOR_CASES.jsonl").exists():
        return
    audit = v9.split_audit(list(v9.read_jsonl(old / "PILOT_CASES.jsonl")), list(v9.read_jsonl(old / "MODEL_DEV_CASES.jsonl")), list(v9.read_jsonl(old / "FROZEN_ANCHOR_CASES.jsonl")))
    assert audit["overlap_total"] == 0


def test_g1_exact_q_pairs_from_v81() -> None:
    cfg = config()
    old81 = ROOT / cfg["data"]["old_v81_dir"]
    if not (old81 / "P1_CORE_PANEL.jsonl").exists():
        return
    rows = list(v9.read_jsonl(old81 / "P1_CORE_PANEL.jsonl"))
    assert len({r["pair_id"] for r in rows}) >= 56
    assert v9.all_exact_q(rows)


def test_binary_metrics_fixture() -> None:
    rows = [
        {"gold": 1, "pred": 1, "score": 0.9},
        {"gold": 1, "pred": 0, "score": 0.4},
        {"gold": 0, "pred": 1, "score": 0.7},
        {"gold": 0, "pred": 0, "score": 0.1},
    ]
    m = v9.binary_metrics(rows)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert m["fn"] == 1
    assert m["accuracy"] == 0.5
