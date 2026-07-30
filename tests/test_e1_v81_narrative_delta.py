from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_e1_v81_narrative_delta", ROOT / "scripts" / "run_e1_v81_narrative_delta.py")
assert spec and spec.loader
v81 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v81)


def _config() -> dict:
    return yaml.safe_load(v81.CONFIG_PATH.read_text(encoding="utf-8"))


def test_latest_ok_by_task_keeps_prior_success_after_error(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    rows = [
        {"task_id": "a", "status": "ok", "text": "first"},
        {"task_id": "a", "status": "error", "error": "later failure"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    latest = v81.latest_ok_by_task(path)

    assert latest["a"]["text"] == "first"


def test_latest_ok_by_task_rejects_parse_error_json(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    rows = [
        {"task_id": "a", "status": "ok", "content_json": {"parse_error": True}},
        {"task_id": "b", "status": "ok", "content_json": {"contextual_defense_state": "SUCCESS"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    latest = v81.latest_ok_by_task(path)

    assert "a" not in latest
    assert "b" in latest


def test_budget_ledger_has_header_and_uses_90_percent_stop_line(tmp_path: Path) -> None:
    config = {
        "budget": {
            "qwen_hard_cap_cny": 10.0,
            "deepseek_hard_cap_cny": 10.0,
            "total_hard_cap_cny": 20.0,
            "stop_new_tasks_fraction": 0.9,
            "pricing_cny_per_million": {"qwen-test": {"input": 1.0, "output": 0.0}},
        }
    }
    v81.init_budget_ledger(tmp_path)
    v81.append_budget_row(
        config,
        tmp_path,
        {
            "phase": "unit",
            "provider": "qwen",
            "requested_model": "qwen-test",
            "response_model": "qwen-test",
            "request_id": "req",
            "usage": {"prompt_tokens": 9_000_000, "completion_tokens": 0},
            "status": "ok",
        },
    )

    with (tmp_path / "E1_V81_BUDGET_LEDGER.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["phase"] == "unit"
    assert v81.budget_would_stop(config, tmp_path, "qwen")


def test_p3_target_model_mapping_supports_qwen_and_deepseek() -> None:
    assert v81.target_model_config_key("deepseek") == "deepseek_target"
    assert v81.target_model_config_key("qwen") == "qwen_target"
    with pytest.raises(ValueError):
        v81.target_model_config_key("unknown")


def test_p1_builder_preserves_exact_q_pairs_on_reused_v8_data(tmp_path: Path) -> None:
    config = _config()
    old_dir = v81.ROOT / config["data"]["old_dir"]
    if not (old_dir / "C_ISO_CONSENSUS.jsonl").exists():
        pytest.skip("V8 reuse data is not available in this checkout")

    pairs, audit = v81.build_p1_pair_candidates(config, tmp_path)

    assert audit["mixed_group_count"] >= 50
    assert audit["q_oracle_accuracy"] == 0.5
    assert all(row["q_byte_identical_pair"] for row in pairs)
    assert len(pairs) == audit["mixed_group_count"] * 2


def test_p2_sample_is_frozen_and_split_disjoint_when_reuse_data_exists() -> None:
    config = _config()
    old_dir = v81.ROOT / config["data"]["old_dir"]
    if not (old_dir / "MODEL_DEV_CASES.jsonl").exists():
        pytest.skip("V8 reuse data is not available in this checkout")

    cases, audit = v81.select_p2_cases(config)

    assert len(cases) == 50
    assert audit["overlap_with_pilot"] == 0
    assert audit["overlap_with_frozen"] == 0
