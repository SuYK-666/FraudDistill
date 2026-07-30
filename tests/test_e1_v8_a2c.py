from __future__ import annotations

import yaml

from frauddistill.e1_v8.consensus import cohen_kappa, normalize_contextual, normalize_official, strict_consensus
from frauddistill.e1_v8.diagnostic_probe import q_group_majority_oracle_accuracy
from frauddistill.e1_v8.fraudr1_stage_loader import build_split_manifest, canonical_id_for_item, load_canonical_cases
from frauddistill.e1_v8.official_prompt_renderer import isolated_stage_prompt, official_roleplay_prompt, v7_parity_prompt


def test_v8_config_namespace() -> None:
    cfg = yaml.safe_load(open("configs/experiments/e1_v8_a2c.yaml", encoding="utf-8"))
    assert cfg["experiment"]["protocol"] == "E1-V8-A2C-OFFICIAL-DELTA-STAGE-PRESSURE-v1.0"
    assert "e1_v8_a2c" in cfg["data"]["output_dir"]


def test_canonical_id_merges_base_and_levelup() -> None:
    item = {"id": 17, "language": "Chinese"}
    assert canonical_id_for_item(item, "base_zh") == canonical_id_for_item(item, "levelup_zh")


def test_stage_loader_and_split_manifest() -> None:
    cfg = yaml.safe_load(open("configs/experiments/e1_v8_a2c.yaml", encoding="utf-8"))
    cases, audit = load_canonical_cases(cfg)
    assert audit["checks"]["canonical_count_2141"]
    assert audit["checks"]["each_case_four_stages"]
    split = build_split_manifest(cases, cfg["data"]["seed"], 20, 40)
    assert len(split["pilot_cases"]) == 200
    assert len(split["model_dev_cases"]) == 400
    assert len(split["frozen_anchor_cases"]) == 1541
    assert split["audit"]["canonical_cross_split_overlap"] == 0


def test_official_renderer_uses_background_and_no_system() -> None:
    cfg = yaml.safe_load(open("configs/experiments/e1_v8_a2c.yaml", encoding="utf-8"))
    cases, _ = load_canonical_cases(cfg)
    case = next(c for c in cases if c["language"] == "en" and c["data_type"] == "message")
    prompt = official_roleplay_prompt(case, 0)
    assert case["role_background"]["receiver"] in prompt
    assert any(occ in prompt for occ in case["role_background"]["receiver_occupation"])
    assert isolated_stage_prompt(case, 2) != prompt
    assert v7_parity_prompt(case) != prompt


def test_dual_view_consensus_is_strict() -> None:
    assert normalize_official({"official_decision": "YES"}) == "SUCCESS"
    assert normalize_official({"official_decision": "NO"}) == "FAILURE"
    assert normalize_contextual({"contextual_defense_state": "FAILURE"}) == "FAILURE"
    assert strict_consensus([{"official_decision": "YES"}, {"official_decision": "NEXT ROUND"}], "O")["consensus_state"] == "UNCERTAIN"
    assert strict_consensus([{"contextual_defense_state": "FAILURE", "assistance_severity": 2}, {"contextual_defense_state": "FAILURE", "assistance_severity": 3}], "X")["strict_assist"]


def test_kappa_and_q_oracle() -> None:
    assert cohen_kappa([("SUCCESS", "SUCCESS"), ("FAILURE", "FAILURE")]) == 1.0
    panel = [
        {"q_stage_group_id": "g1", "gold": 1},
        {"q_stage_group_id": "g1", "gold": 0},
        {"q_stage_group_id": "g2", "gold": 1},
        {"q_stage_group_id": "g2", "gold": 1},
    ]
    assert q_group_majority_oracle_accuracy(panel) == 0.75
