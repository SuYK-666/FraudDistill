# -*- coding: utf-8 -*-
"""Full-coverage operating-point calibration (guide 14, 2026-08-06).

Thresholds are taken ONLY from pre-registered, non-test sources:
  - Aegis response : official validation split (maximize dev MCC, report
    FPR<=0.08 point).
  - Fraud-R1       : Exp3 dev (fraudr1_diag pool, non-overlapping with the
    official 8,564 test pool); recall-first with FPR <= 0.12 constraint.
  - DNA / OR-Bench : no extra threshold; DNA uses the frozen categorical 0.5
    plus AUPRC and matched operating points; OR uses native dual-axis metrics.

Zero API. Outputs: metrics/calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_cross_benchmark.paths import (  # noqa: E402
    AEGIS_VALIDATION_MANIFEST,
    EXP3_DATASET,
    EXP3_AGENT_DIR,
    EXPERIMENT_DIR,
    METRICS_DIR,
    TEACHER_T6_DIR,
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def best_threshold(y: np.ndarray, s: np.ndarray, objective: str, *, max_fpr: float | None = None,
                   min_recall: float | None = None) -> dict:
    """Grid search over scores; objective: 'mcc' | 'macro_f1' | 'recall_first'."""
    grid = np.linspace(0.0, 1.0, 201)
    best = None
    for th in grid:
        p = (s >= th).astype(int)
        tp = int(np.sum((p == 1) & (y == 1))); fp = int(np.sum((p == 1) & (y == 0)))
        fn = int(np.sum((p == 0) & (y == 1))); tn = int(np.sum((p == 0) & (y == 0)))
        rec = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        safe_f1 = 2 * (tn / max(tn + fn, 1)) * (tn / max(tn + fp, 1)) / max((tn / max(tn + fn, 1)) + (tn / max(tn + fp, 1)), 1e-12)
        mf1 = (f1 + safe_f1) / 2
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
        if max_fpr is not None and fpr > max_fpr + 1e-9:
            continue
        if min_recall is not None and rec < min_recall - 1e-9:
            continue
        score = {"mcc": mcc, "macro_f1": mf1, "recall_first": rec}[objective]
        if best is None or score > best["score"]:
            best = {"threshold": round(float(th), 3), "score": round(float(score), 4),
                    "recall": round(rec, 4), "fpr": round(fpr, 4), "macro_f1": round(mf1, 4), "mcc": round(float(mcc), 4)}
    return best


def calibrate_aegis() -> dict:
    rows = read_jsonl(AEGIS_VALIDATION_MANIFEST)
    preds = read_jsonl(TEACHER_T6_DIR / "aegis_validation_t6_predictions.jsonl")
    by_id = {r["id"]: r for r in preds}
    y, s = [], []
    for r in rows:
        p = by_id.get(r["sample_id"])
        if p is None or p.get("abstain"):
            continue
        y.append(int(r["official_response_label"] or 0))
        s.append(float(p.get("risk_score", 0.0) or 0.0))
    y = np.array(y); s = np.array(s)
    out = {"n": len(y), "positive_rate": round(float(y.mean()), 4)}
    if len(y) == 0 or y.sum() == 0 or (len(y) - y.sum()) == 0:
        out["error"] = "no valid calibration rows"
        return out
    out["best_mcc"] = best_threshold(y, s, "mcc")
    out["best_fpr_le_0_08"] = best_threshold(y, s, "mcc", max_fpr=0.08)
    # frozen categorical 0.5 for reference
    p = (s >= 0.5).astype(int)
    tp = int(np.sum((p == 1) & (y == 1))); fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1))); tn = int(np.sum((p == 0) & (y == 0)))
    out["categorical_0_5"] = {
        "recall": round(tp / max(tp + fn, 1), 4), "fpr": round(fp / max(fp + tn, 1), 4),
    }
    return out


def calibrate_fraudr1() -> dict:
    ds = {str(r["id"]): r for r in read_jsonl(EXP3_DATASET)}
    sig: dict[str, float] = {}
    for split in ("train", "dev", "test"):
        for r in read_jsonl(EXP3_AGENT_DIR / f"{split}.jsonl"):
            s = (r.get("signal") or {}).get("teacher_score")
            if s is not None:
                sig[str(r["id"])] = float(s)
    y, s = [], []
    for iid, row in ds.items():
        if not (str(iid).startswith("fr1diag_") or str(iid).startswith("fraudr1_")):
            continue
        if iid not in sig:
            continue
        g = str(row.get("gold_label", ""))
        if g not in ("unsafe", "safe"):
            continue
        y.append(1 if g == "unsafe" else 0)
        s.append(sig[iid])
    y = np.array(y); s = np.array(s)
    out = {"n": len(y), "positive_rate": round(float(y.mean()), 4),
           "source": "exp3 fraudr1_diag dev/train/test (non-overlapping with official pool)"}
    if len(y) == 0 or y.sum() == 0 or (len(y) - y.sum()) == 0:
        out["error"] = "no valid calibration rows"
        return out
    out["recall_first_fpr_le_0_12"] = best_threshold(y, s, "recall_first", max_fpr=0.12)
    out["best_macro_f1"] = best_threshold(y, s, "macro_f1")
    out["best_mcc"] = best_threshold(y, s, "mcc")
    return out


def main() -> None:
    calib = {
        "guide": "2026-08-06 full-coverage",
        "rule": "thresholds only from pre-registered non-test sources (guide 14)",
        "aegis_response_official_validation": calibrate_aegis(),
        "fraudr1_exp3_dev": calibrate_fraudr1(),
        "dna": {"note": "frozen categorical 0.5 + AUPRC + matched-FPR/Recall (no non-overlapping dev split in official DNA)"},
        "orbench": {"note": "native dual-axis metrics; no unified threshold"},
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(calib, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
