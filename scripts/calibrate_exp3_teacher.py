# -*- coding: utf-8 -*-
"""Calibrate teacher risk scores on dev and freeze the config (guide 3.6 / 24).

Protocol (guide 3.6): compare raw / Platt / isotonic / temperature on dev by
1) Brier, 2) ECE, 3) AUPRC ranking must not change; keep raw if nothing beats it.

Usage: python scripts/calibrate_exp3_teacher.py [--max-fpr 0.08]
Outputs: outputs/metrics/calibration.json + outputs/metrics/calibrator_comparison.json
         + outputs/frozen_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frauddistill.teacher.score_calibrator import ScoreCalibrator, _to_binary, brier_score, ece_score, true_macro_f1

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("EXP3_OUT_ROOT") or REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs")
DEV_FILE = OUT_ROOT / "agent_predictions" / "dev.jsonl"
METRICS = OUT_ROOT / "metrics"


def read_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def quick(ss: list[float], y: list[int], t: float) -> dict:
    pred = [1 if s >= t else 0 for s in ss]
    tp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    y_arr = __import__("numpy").asarray(y, dtype=int)
    pred_arr = __import__("numpy").asarray(pred, dtype=int)
    return {
        "acc": round((tp + tn) / max(len(y), 1), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "unsafe_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "macro_f1": round(true_macro_f1(y_arr, pred_arr), 4),
        "fpr": round(fp / max(tn + fp, 1), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-fpr", type=float, default=0.08)
    args = ap.parse_args()

    recs = read_records(DEV_FILE)
    print(f"dev records: {len(recs)}")
    if len(recs) < 100:
        print("dev not ready yet")
        sys.exit(2)

    raw_scores = [float((r.get("signal") or {}).get("teacher_score", 0.5)) for r in recs]
    labels = [r["sample"]["gold_label"] for r in recs]
    y = [1 if l == "unsafe" else 0 for l in labels]

    # ---- compare all calibration methods on dev (guide 3.6)
    comparison = {}
    for method in ScoreCalibrator.METHODS:
        cal = ScoreCalibrator(method=method)
        cal.fit(raw_scores, labels)
        cal_scores = cal.calibrate(raw_scores)
        metric = cal.evaluate(raw_scores, labels)
        threshold = cal.select_threshold(cal_scores, labels, max_fpr=args.max_fpr)
        comparison[method] = {
            "method": method,
            "dev_brier": metric["brier"],
            "dev_ece": metric["ece"],
            "dev_auprc": metric["auprc"],
            "dev_auroc": metric["auroc"],
            "threshold": threshold,
            "dev_at_threshold": quick(cal_scores, y, threshold),
            "temperature": getattr(cal, "temperature", None),
        }
        print(f"{method:12s} brier={metric['brier']:.5f} ece={metric['ece']:.5f} "
              f"auprc={metric['auprc']:.5f} thr={threshold:.4f}")

    # selection rule: Brier first, ECE second; AUPRC ranking must not change vs raw
    def sort_key(name):
        m = comparison[name]
        return (m["dev_brier"], m["dev_ece"], -m["dev_auprc"])

    raw_rank = sorted(ScoreCalibrator.METHODS, key=sort_key).index("raw")
    best = sorted(ScoreCalibrator.METHODS, key=sort_key)[0]
    # if best is not raw, verify it does not flip AUPRC ranking relative to raw
    if best != "raw":
        if comparison[best]["dev_auprc"] < comparison["raw"]["dev_auprc"] - 1e-6:
            best = "raw"
            print("AUPRC ranking guard: falling back to raw")
    comparison["_selection"] = {
        "rule": "brier -> ece -> auprc-rank-guard",
        "raw_auprc_rank": raw_rank,
        "selected": best,
    }

    selected = ScoreCalibrator(method=best)
    selected.fit(raw_scores, labels)
    cal_scores = selected.calibrate(raw_scores)
    threshold = selected.select_threshold(cal_scores, labels, max_fpr=args.max_fpr)

    report = {
        "method": best,
        "max_fpr": args.max_fpr,
        "threshold": threshold,
        "n_dev": len(recs),
        "score_mean": round(float(sum(raw_scores) / len(raw_scores)), 4),
        "label_unsafe_share": round(sum(y) / len(y), 4),
        "dev_raw_at_0.5": quick(raw_scores, y, 0.5),
        "dev_calibrated_at_0.5": quick(cal_scores, y, 0.5),
        "dev_calibrated_at_best": quick(cal_scores, y, threshold),
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (METRICS / "calibrator_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    selected.save(str(METRICS / "calibrator.json"))

    frozen = {
        "experiment": "exp3_agent_distillation_ablation",
        "threshold": threshold,
        "calibration": {"method": best, "max_fpr": args.max_fpr},
        "frozen_at": "2026-08-04",
        "note": "fitted on dev only; test untouched",
    }
    (OUT_ROOT / "frozen_config.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("frozen config ->", OUT_ROOT / "frozen_config.json")


if __name__ == "__main__":
    main()
