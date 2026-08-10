# -*- coding: utf-8 -*-
"""Reconcile prediction files: ensure every test/cal row has a prediction
(borrowing from the sibling file when the same row was inferred there)."""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def main():
    test_rows = load(BASE / "manifests/frozen_test.jsonl")
    cal_rows = load(BASE / "manifests/calibration.jsonl")
    for key in ("final_student", "neural_gold", "neural_softdistill"):
        f_test = BASE / "predictions" / f"{key}.jsonl"
        f_cal = BASE / "predictions" / f"{key}_calibration.jsonl" if key == "final_student" else None
        preds = {r["id"]: r for r in load(f_test)}
        if f_cal and f_cal.exists():
            for r in load(f_cal):
                preds.setdefault(r["id"], r)
        missing_test = [r["id"] for r in test_rows if r["id"] not in preds]
        missing_cal = [r["id"] for r in cal_rows if r["id"] not in preds]
        print(f"{key}: test_missing={len(missing_test)} cal_missing={len(missing_cal)}")
        # write test file with all test rows
        with open(f_test, "w", encoding="utf-8") as f:
            for r in test_rows:
                p = preds.get(r["id"])
                if p:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        if f_cal:
            with open(f_cal, "w", encoding="utf-8") as f:
                for r in cal_rows:
                    p = preds.get(r["id"])
                    if p:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{key}: test file rows={sum(1 for r in test_rows if r['id'] in preds)}")
        if f_cal:
            print(f"{key}: cal file rows={sum(1 for r in cal_rows if r['id'] in preds)}")
    print("[reconcile] DONE")

if __name__ == "__main__":
    main()
