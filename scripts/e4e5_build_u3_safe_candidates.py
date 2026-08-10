# -*- coding: utf-8 -*-
"""Build U3 safe-replacement candidates from refuse_chat generations."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "data/prepared/e4e5_v2/generated_u3"
OUT = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL/manifests/u3_safe_candidates_v2.jsonl"


def main() -> None:
    used = set()
    for m in ("frozen_test", "calibration"):
        p = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL/manifests" / f"{m}.jsonl"
        for l in open(p, encoding="utf-8"):
            used.add(json.loads(l)["id"])
    rng = random.Random(20260810)
    out = []
    for model, fname in (("smollm2_1_7b", "smollm2_1_7b_refuse_chat.jsonl"),
                         ("phi3_5_mini", "phi3_5_mini_refuse_chat.jsonl")):
        rows = [json.loads(l) for l in open(GEN / fname, encoding="utf-8")]
        cand = [r for r in rows if r.get("id") not in used and str(r.get("target_model_answer") or "").strip()]
        rng.shuffle(cand)
        for r in cand[:120]:
            q = str(r.get("user_query") or "")
            y = str(r.get("target_model_answer") or "")
            out.append({
                "id": r["id"], "user_query": q, "target_model_answer": y,
                "gold_label": None, "gold_type": None, "gold_binary": None,
                "gold_source": "blind_judge_deepseek",
                "source": "u3_target_style_gen", "source_version": model,
                "language": r.get("language", "en"),
                "target_model": model, "primary_shift": "U3_target_style",
                "fraud_category": r.get("fraud_category", ""),
                "family_id": r.get("family_id", ""), "pair_id": None,
                "template_id": r.get("prompt_id", ""),
                "license": "cc-by-4.0", "exposure_level": "L3",
                "qy_hash": hashlib.sha256((q + "\x01" + y).encode()).hexdigest(),
                "q_hash": hashlib.sha256(q.encode()).hexdigest(),
                "y_hash": hashlib.sha256(y.encode()).hexdigest(),
                "metadata": {"mode": "refuse_chat", "seed": r.get("seed")},
            })
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"U3 safe candidates: {len(out)} -> {OUT}")


if __name__ == "__main__":
    main()
