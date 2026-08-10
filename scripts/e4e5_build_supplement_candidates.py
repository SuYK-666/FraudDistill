# -*- coding: utf-8 -*-
"""E4 v2: build supplemental candidate manifests for G2 blind judging.

Pools:
- u1_leftover: candidate_pool rows (U1) not in old panel (rule_g1 labels)
- u3_leftover: candidate_pool rows (U3) not in old panel
- u2_pku_leftover: candidate_pool pku rows not in old panel (excl. aegis)
- u3_refuse_remaining: refuse_chat rows not in old panel nor u3_safe_candidates
- retries: api_fail rows from g2_u2_candidates / g2_u3_safe
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {p}")

def main():
    panel_ids = set()
    for m in ("frozen_test", "calibration"):
        for r in load(BASE / "manifests" / f"{m}.jsonl"):
            panel_ids.add(r["id"])
    pool = load(BASE / "candidate_pool.jsonl")
    left = [r for r in pool if r["id"] not in panel_ids]
    u1_left = [r for r in left if r["primary_shift"] == "U1_category"]
    u3_left = [r for r in left if r["primary_shift"] == "U3_target_style"]
    u2_pku_left = [r for r in left if r["primary_shift"] == "U2_source" and r["source"] == "pku_saferlhf"]
    print(f"u1_left={len(u1_left)} u3_left={len(u3_left)} u2_pku_left={len(u2_pku_left)}")
    write(BASE / "manifests" / "u1_leftover_candidates_v2.jsonl", u1_left)
    write(BASE / "manifests" / "u3_leftover_candidates_v2.jsonl", u3_left)
    write(BASE / "manifests" / "u2_pku_leftover_candidates_v2.jsonl", u2_pku_left)

    # refuse remaining (from generation files, not in panel nor u3_safe_candidates)
    cand_used = {r["id"] for r in load(BASE / "manifests" / "u3_safe_candidates_v2.jsonl")}
    refuse_remain = []
    for model, fname in (("smollm2_1_7b", "smollm2_1_7b_refuse_chat.jsonl"),
                         ("phi3_5_mini", "phi3_5_mini_refuse_chat.jsonl")):
        for l in open(REPO / "data/prepared/e4e5_v2/generated_u3" / fname, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in panel_ids or r["id"] in cand_used:
                continue
            q = str(r.get("user_query") or "")
            y = str(r.get("target_model_answer") or "")
            if not q or not y:
                continue
            import hashlib
            refuse_remain.append({
                "id": r["id"], "user_query": q, "target_model_answer": y,
                "gold_label": None, "gold_type": None, "gold_binary": None,
                "gold_source": "blind_judge_deepseek",
                "source": "u3_target_style_gen", "source_version": model,
                "language": r.get("language", "en"), "target_model": model,
                "primary_shift": "U3_target_style", "fraud_category": r.get("fraud_category", ""),
                "family_id": r.get("family_id", ""), "pair_id": None,
                "template_id": r.get("prompt_id", ""), "license": "cc-by-4.0",
                "exposure_level": "L3",
                "qy_hash": hashlib.sha256((q + "\x01" + y).encode()).hexdigest(),
                "q_hash": hashlib.sha256(q.encode()).hexdigest(),
                "y_hash": hashlib.sha256(y.encode()).hexdigest(),
                "metadata": {"mode": "refuse_chat", "seed": r.get("seed")},
            })
    print("refuse_remaining:", len(refuse_remain))
    write(BASE / "manifests" / "u3_refuse_remaining_v2.jsonl", refuse_remain)

    # retry api_fail rows from previous judgments
    retries = []
    for fname, tag in (("g2_u2_candidates.jsonl", "G2_U2"), ("g2_u3_safe.jsonl", "G2_U3safe")):
        src = {r["id"]: r for r in load(BASE / "audits" / fname)}
        for r in load(BASE / "audits" / fname):
            if r.get("status") == "api_fail":
                retries.append({"id": r["id"], "tag": tag, "raw": r.get("raw", "")[:120]})
    print("api_fail retries:", len(retries))
    write(BASE / "manifests" / "g2_retry_list.jsonl", retries)

if __name__ == "__main__":
    main()
