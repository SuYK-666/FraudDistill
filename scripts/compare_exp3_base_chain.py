# -*- coding: utf-8 -*-
"""Exp3 Student capability chain on the fixed Base-1.5B 500-row subset.

Compares Base-1.5B Zero-shot vs Random Head vs Neural-Gold / SoftDistill /
FullDistill on the SAME 500 test rows (fixed seed), so the capability chain
Base Model -> Trained Student is fully fair.

Usage:
  python scripts/compare_exp3_base_chain.py [--out-dir ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef)

from evaluate_neural_student import ece_score
from frauddistill.student.dataset import LABEL_TO_ID

OUT_ROOT = REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs" / "neural_student"
TEST_JSONL = REPO / "data" / "prepared" / "exp3_agent_distillation" / "test.jsonl"

MODELS = [
    ("Base-1.5B-ZeroShot", "base_zeroshot/predictions_test.jsonl"),
    ("Neural-ZeroShot (random head)", "eval_zero_shot/predictions_test.jsonl"),
    ("Neural-Gold", "eval_gold/predictions_test.jsonl"),
    ("Neural-SoftDistill", "eval_soft/predictions_test.jsonl"),
    ("Neural-FullDistill", "eval_full/predictions_test.jsonl"),
]


def metrics_on(rows, pmap):
    y = [1 if r["gold_label"] == "unsafe" else 0 for r in rows]
    pred = [1 if pmap[r["id"]]["label"] == "unsafe" else 0 for r in rows]
    scores = np.array([pmap[r["id"]]["risk_score"] for r in rows], dtype=float)
    y_arr = np.array(y, dtype=int)
    p_arr = np.array(pred, dtype=int)
    tn = int(((p_arr == 0) & (y_arr == 0)).sum()); fp = int(((p_arr == 1) & (y_arr == 0)).sum())
    fn = int(((p_arr == 0) & (y_arr == 1)).sum()); tp = int(((p_arr == 1) & (y_arr == 1)).sum())
    gold_types = [LABEL_TO_ID.get(r.get("gold_type", ""), (0 if r["gold_label"] == "safe" else 1)) for r in rows]
    pred_types = [LABEL_TO_ID.get(pmap[r["id"]]["risk_type"], 0) for r in rows]
    return {
        "n": len(rows),
        "acc": round(float(accuracy_score(y_arr, p_arr)), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "macro_f1": round(float(f1_score(y_arr, p_arr, average="macro", zero_division=0)), 4),
        "fpr": round(fp / max(tn + fp, 1), 4),
        "auprc": round(float(average_precision_score(y_arr, scores)), 4) if len(set(scores)) > 1 else None,
        "mcc": round(float(matthews_corrcoef(y_arr, p_arr)), 4),
        "4class_macro_f1": round(float(f1_score(gold_types, pred_types, average="macro", zero_division=0)), 4),
        "ece": round(ece_score(scores, y_arr), 4),
        "brier": round(float(np.mean((scores - y_arr) ** 2)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_ids = json.loads((out_dir / "base_zeroshot" / "subset_ids.json").read_text(encoding="utf-8"))["ids"]
    all_test = [json.loads(l) for l in TEST_JSONL.open(encoding="utf-8") if l.strip()]
    rows = [r for r in all_test if r["id"] in subset_ids]
    rows = [r for r in all_test if r["id"] in set(subset_ids)]
    print(f"subset rows: {len(rows)} (seed ids {len(subset_ids)})")

    results = {}
    for name, rel in MODELS:
        p = out_dir / rel
        if not p.exists():
            print(f"WARN missing {rel}; skipped")
            continue
        pmap = {json.loads(l)["id"]: json.loads(l) for l in p.open(encoding="utf-8") if l.strip()}
        missing = [r["id"] for r in rows if r["id"] not in pmap]
        if missing:
            print(f"WARN {name}: {len(missing)} ids missing from predictions")
            rows_ok = [r for r in rows if r["id"] in pmap]
        else:
            rows_ok = rows
        results[name] = metrics_on(rows_ok, pmap)
        print(name, results[name])

    (out_dir / "base_chain_500.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    header = ["model", "n", "acc", "precision", "recall", "macro_f1", "fpr", "auprc", "mcc", "4class_macro_f1", "ece", "brier"]
    with (out_dir / "base_chain_500.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for name, m in results.items():
            f.write(",".join([name] + [str(m.get(h, "")) for h in header[1:]]) + "\n")
    print("saved base_chain_500.csv / .json")


if __name__ == "__main__":
    main()