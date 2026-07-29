from __future__ import annotations

from collections import Counter

import yaml

from frauddistill.exp1_paired_interaction_r4.manifest import sha_text
from scripts.run_e1_paired_interaction_r41 import (
    global_overlap_audit,
    panel_c_audit,
    select_unpaired_panel_c,
    stability_decision,
)


def test_r41_config_removes_by_panel() -> None:
    config = yaml.safe_load(open("configs/experiments/e1_paired_interaction_r41.yaml", encoding="utf-8"))
    assert "By" not in config["data"]["panel_sizes"]["anchor"]
    assert "by_groups" not in config["data"]
    assert config["data"]["public_revisions"]["pku_saferlhf"] != "main"
    assert config["data"]["public_revisions"]["beavertails"] != "main"


def test_panel_c_selects_fixed_sample0_one_row_per_base() -> None:
    rows = []
    for lang in ("zh", "en"):
        for label in ("safe", "unsafe"):
            for i in range(100):
                base_id = f"{lang}_{label}_{i}"
                rows.append(
                    {
                        "id": f"{base_id}_0",
                        "base_id": base_id,
                        "language": lang,
                        "gold_label": label,
                        "gold_risk_type": "unsafe_fraud_enablement" if label == "unsafe" else "safe_refusal",
                        "user_query": f"q {base_id}",
                        "target_model_answer": f"y {base_id}",
                        "label_provenance": "fixture",
                        "source_file": "fixture",
                        "category": "fixture",
                        "sample_index": 0,
                    }
                )
                rows.append({**rows[-1], "id": f"{base_id}_1", "sample_index": 1})
    anchor, dev, audit = select_unpaired_panel_c(rows, seed=7)
    assert audit["passed"]
    assert len(anchor) == 240
    assert len(dev) == 120
    assert not {r["base_id"] for r in anchor} & {r["base_id"] for r in dev}
    assert all(r["sample_index"] == 0 for r in anchor + dev)
    assert len({r["base_id"] for r in anchor + dev}) == len(anchor + dev)
    assert Counter((r["language"], r["gold_label"]) for r in anchor) == Counter({("zh", "safe"): 60, ("zh", "unsafe"): 60, ("en", "safe"): 60, ("en", "unsafe"): 60})


def test_panel_c_audit_rejects_base_overlap() -> None:
    rows = [
        {"base_id": "a", "language": "en", "gold_label": "safe", "sample_index": 0},
        {"base_id": "a", "language": "en", "gold_label": "unsafe", "sample_index": 0},
    ]
    audit = panel_c_audit(rows[:1], rows[1:])
    assert not audit["passed"]
    assert audit["base_overlap_count"] == 1


def test_global_overlap_audit_checks_qy_hash_and_source_prompt() -> None:
    anchor = [{"id": "a", "source_prompt_id": "p1", "user_query": "Q", "target_model_answer": "Y", "canonical_group_id": "g1", "panel": "A"}]
    dev = [{"id": "b", "source_prompt_id": "p1", "user_query": "Q", "target_model_answer": "Y", "canonical_group_id": "g2", "panel": "Bctx"}]
    audit = global_overlap_audit(anchor, dev)
    assert not audit["passed"]
    assert audit["qy_hash_overlap_count"] == 1
    assert audit["source_prompt_overlap_count"] == 1


def test_stability_decision_is_not_placeholder() -> None:
    payload = stability_decision({"q_only": 1.0, "y_only": 1.0, "q_y": 1.0}, min_agreement=0.98)
    assert payload["decision"] == "E1_R41_STABILITY_PASS"
    assert "placeholder" not in str(payload).lower()


def test_sha_text_fixture() -> None:
    assert sha_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
