# -*- coding: utf-8 -*-
"""Calibrate teacher risk scores on dev and freeze the config (guide 10.4 / 24).

Usage: python scripts/calibrate_exp3_teacher.py [--method platt|isotonic] [--max-fpr 0.08]
Outputs: outputs/metrics/calibration.json + outputs/frozen_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frauddistill.teacher.score_calibrator import ScoreCalibrator

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("EXP3_OUT_ROOT") or REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs")
DEV_FILE = OUT_ROOT / "agent_predictions" / "dev.jsonl"
METRICS = OUT_ROOT / "metrics"


def read_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def metrics_at(recs: list[dict], key: str) -> dict:
    y = [1 if r["sample"]["gold_label"] == "unsafe" else 0 for r in recs]
    pred = [1 if (r.get("signal") or {}).get("teacher_label", "safe") == "unsafe" else 0 for r in recs]
    tp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(pred, y) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(pred, y) if a == 0 and b == 0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "acc": (tp + tn) / max(len(y), 1),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "fpr": round(fp / max(tn + fp, 1), 4),
        "n": len(y),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="platt", choices=["platt", "isotonic"])
    ap.add_argument("--max-fpr", type=float, default=0.08)
    args = ap.parse_args()

    recs = read_records(DEV_FILE)
    print(f"dev records: {len(recs)}")
    if len(recs) < 100:
        print("dev not ready yet")
        sys.exit(2)

    scores = [float((r.get("signal") or {}).get("teacher_score", 0.5)) for r in recs]
    labels = [r["sample"]["gold_label"] for r in recs]

    cal = ScoreCalibrator(method=args.method)
    cal.fit(scores, labels)
    cal_scores = cal.calibrate(scores)
    threshold = cal.select_threshold(cal_scores, labels, max_fpr=args.max_fpr)

    # metrics before/after calibration at 0.5 and at chosen threshold
    def apply(ss, t):
        return [1 if s >= t else 0 for s in ss]

    y = [1 if l == "unsafe" else 0 for l in labels]
    def quick(ss, t):
        p = apply(ss, t)
        tp = sum(1 for a, b in zip(p, y) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(p, y) if a == 1 and b == 0)
        fn = sum(1 for a, b in zip(p, y) if a == 0 and b == 1)
        tn = sum(1 for a, b in zip(p, y) if a == 0 and b == 0)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return {"acc": round((tp + tn) / len(y), 4), "precision": round(prec, 4), "recall": round(rec, 4), "macro_f1": round(f1, 4), "fpr": round(fp / max(tn + fp, 1), 4)}

    report = {
        "method": args.method,
        "max_fpr": args.max_fpr,
        "threshold": threshold,
        "dev_raw_at_0.5": quick(scores, 0.5),
        "dev_calibrated_at_0.5": quick(cal_scores, 0.5),
        "dev_calibrated_at_best": quick(cal_scores, threshold),
        "n_dev": len(recs),
        "score_mean": round(float(sum(scores) / len(scores)), 4),
        "label_unsafe_share": round(sum(y) / len(y), 4),
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cal.save(str(METRICS / "calibrator.json"))

    frozen = {
        "experiment": "exp3_agent_distillation_ablation",
        "threshold": threshold,
        "calibration": {"method": args.method, "max_fpr": args.max_fpr},
        "frozen_at": "2026-08-04",
        "note": "fitted on dev only; test untouched",
    }
    (OUT_ROOT / "frozen_config.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("frozen config ->", OUT_ROOT / "frozen_config.json")


if __name__ == "__main__":
    main()