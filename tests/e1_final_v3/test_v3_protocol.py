from __future__ import annotations

from pathlib import Path

import numpy as np

from frauddistill.e1_final_v3.budget import hard_stop_decision
from frauddistill.e1_final_v3.io import norm, sha_text, write_jsonl
from frauddistill.e1_final_v3.panel_builder import audit_b_capacity
from frauddistill.e1_final_v3.registry import join_gold, load_fraudr1_q_manifest, load_response_rows
from frauddistill.e1_real_v2.pairlite_cpu_v2 import extra_features


def test_a_quota_manifest_normalizes_ten_cells(tmp_path: Path) -> None:
    src = tmp_path / "fraudr1.jsonl"
    rows = []
    for lang in ["English", "Chinese"]:
        for cat in ["fake job posting", "fraudulent service", "impersonation", "network friendship", "phishing_scams"]:
            rows.append({"id": f"{lang}-{cat}", "language": lang, "fraud_category": cat, "user_query": f"{lang} {cat} suspicious prompt"})
    write_jsonl(src, rows)
    loaded, audit = load_fraudr1_q_manifest(src)
    cells = {(r["language"], r["fraud_category"]) for r in loaded}
    assert len(cells) == 10
    assert audit["accepted"] == 10


def test_q_hash_is_recomputed_from_text_not_trusted_field(tmp_path: Path) -> None:
    src = tmp_path / "responses.jsonl"
    q = "same normalized question"
    write_jsonl(
        src,
        [
            {
                "response_id": "r1",
                "provider": "qwen",
                "q": q,
                "text": "target answer",
                "normalized_q_sha256": "bad-old-hash",
            }
        ],
    )
    rows, _ = load_response_rows([src])
    assert rows[0]["q_hash_recomputed"] == sha_text(norm(q))
    assert rows[0]["q_hash_recomputed"] != "bad-old-hash"


def test_duplicate_target_responses_are_deduped_across_sources(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    row = {"response_id": "r1", "provider": "deepseek", "request_id": "req", "q": "q text", "text": "same y"}
    write_jsonl(a, [row])
    write_jsonl(b, [{**row, "response_id": "r2"}])
    rows, audit = load_response_rows([a, b])
    assert len(rows) == 1
    assert audit["rejects"]["duplicate_response"] == 1


def test_missing_gold_remains_unknown_not_zero_negative(tmp_path: Path) -> None:
    src = tmp_path / "responses.jsonl"
    gold = tmp_path / "gold.jsonl"
    write_jsonl(src, [{"response_id": "r1", "provider": "qwen", "q": "q text", "text": "answer"}])
    write_jsonl(gold, [])
    rows, _ = load_response_rows([src])
    joined = join_gold(rows, gold)
    assert joined[0]["gold_status"] == "UNKNOWN"
    audit = audit_b_capacity(joined, {"context_stable_negative": 1})
    assert audit["by_stratum"]["unknown_gold"] == 1


def test_pairlite_extra_features_are_view_isolated() -> None:
    rows = [{"q_private": "fraud transfer job", "y_private": "safe verify report scam"}]
    q_only = extra_features(rows, "q-only")
    y_only = extra_features(rows, "y-only")
    qy = extra_features(rows, "q+y")
    assert q_only.shape == y_only.shape == qy.shape
    assert q_only[0, 2] == 0
    assert y_only[0, 5] == 0
    assert not np.allclose(q_only, y_only)


def test_budget_hard_stop() -> None:
    ledger = [{"provider": "qwen", "cost_cny": 121}, {"provider": "deepseek", "cost_cny": 10}]
    decision = hard_stop_decision(ledger, {"hard_stop_total_cny": 215, "qwen_hard_stop_cny": 120, "deepseek_hard_stop_cny": 90})
    assert decision["hard_stop"] is True
