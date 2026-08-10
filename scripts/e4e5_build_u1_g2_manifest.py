# -*- coding: utf-8 -*-
"""Build G2 judging manifest from generated U1 strong files (jsonl for g2_judge_file)."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/prepared/e4e5_v2/generated_u1_safe"
OUT = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL/manifests/g2_u1_safe_todo.jsonl"
seen = set()
with open(OUT, "w", encoding="utf-8") as f:
    n = 0
    for fp in sorted(SRC.glob("*_strong.jsonl")):
        for l in open(fp, encoding="utf-8"):
            r = json.loads(l)
            q = str(r.get("user_query") or ""); y = str(r.get("target_model_answer") or "")
            if not q or not y:
                continue
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            f.write(json.dumps({"id": r["id"], "user_query": q, "target_model_answer": y,
                                "qy_hash": r.get("qy_hash", "")}, ensure_ascii=False) + "\n")
            n += 1
print(f"wrote {n} -> {OUT}")
