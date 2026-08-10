# -*- coding: utf-8 -*-
"""Assemble manifests/u1_generated_safe_v2.jsonl from generated natural-mode files."""
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"
SRC = REPO / "data/prepared/e4e5_v2/generated_u1_safe"

def main():
    rows = []
    seen_key = set()
    for f in sorted(SRC.glob("*_strong.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            q = str(r.get("user_query") or ""); y = str(r.get("target_model_answer") or "")
            if not q or not y:
                continue
            k = (q, y)
            if k in seen_key:
                continue
            seen_key.add(k)
            r["qy_hash"] = hashlib.sha256((q + "\x01" + y).encode()).hexdigest()
            rows.append(r)
    out = BASE / "manifests/u1_generated_safe_v2.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {out}")

if __name__ == "__main__":
    main()
