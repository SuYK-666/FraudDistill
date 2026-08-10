# -*- coding: utf-8 -*-
"""Build candidate manifest from u3v2 refuse_v2 generation files."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"
GEN = REPO / "data/prepared/e4e5_v2/generated_u3"
OUT = BASE / "manifests/u3v2_candidates_v2.jsonl"

def main():
    out = []
    seen = set()
    for fp in sorted(GEN.glob("*_refuse_v2.jsonl")):
        for l in open(fp, encoding="utf-8"):
            r = json.loads(l)
            if r.get("id") in seen: continue
            seen.add(r["id"])
            q = str(r.get("user_query") or "")
            y = str(r.get("target_model_answer") or "")
            if not q or not y: continue
            model = str(r.get("target_model") or "")
            out.append({
                "id": r["id"], "user_query": q, "target_model_answer": y,
                "gold_label": None, "gold_type": None, "gold_binary": None,
                "gold_source": "blind_judge_deepseek",
                "source": "u3_target_style_gen", "source_version": model,
                "language": r.get("language", "en"), "target_model": model,
                "primary_shift": "U3_target_style",
                "fraud_category": r.get("fraud_category", ""),
                "family_id": r.get("family_id") or r.get("prompt_id", ""),
                "pair_id": None, "template_id": r.get("prompt_id", ""),
                "license": "cc-by-4.0", "exposure_level": "L3",
                "qy_hash": hashlib.sha256((q + "\x01" + y).encode()).hexdigest(),
                "q_hash": hashlib.sha256(q.encode()).hexdigest(),
                "y_hash": hashlib.sha256(y.encode()).hexdigest(),
                "metadata": {"mode": r.get("mode", "refuse_chat_v2"), "seed": r.get("seed")},
            })
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"u3v2 candidates: {len(out)} -> {OUT}")

if __name__ == "__main__":
    main()
