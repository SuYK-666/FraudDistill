"""Dev-set calibration of domain thresholds and deterministic weights.

Only allowed to run on the frozen dev set. Objective follows guide section 11:
  objective = macro_f1 - 1.5*max(0, fpr-0.05) - 1.0*max(0, 0.70-recall)   (general)
  objective_fraud = macro_f1 - 1.0*max(0, fpr-0.08) - 2.0*max(0, 0.75-recall) (fraud)
"""
from __future__ import annotations

import itertools

import numpy as np

from frauddistill.arbitration.deterministic_arbiter import ArbiterConfig, DOMAIN_THRESHOLDS, decide
from frauddistill.arbitration.evidence import RiskEvidence


def objective(records: list[dict], fpr_penalty: float = 1.5, recall_target: float = 0.70, recall_penalty: float = 1.0, fpr_target: float = 0.05) -> float:
    if not records:
        return -1e9
    golds = np.array([r["gold"] for r in records])
    preds = np.array([r["unsafe"] for r in records])
    tp = int(np.sum((preds == 1) & (golds == 1)))
    fp = int(np.sum((preds == 1) & (golds == 0)))
    fn = int(np.sum((preds == 0) & (golds == 1)))
    tn = int(np.sum((preds == 0) & (golds == 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    safe_f1 = tn / max(tn + fp, 1)
    macro_f1 = (f1 + safe_f1) / 2
    return macro_f1 - fpr_penalty * max(0.0, fpr - fpr_target) - recall_penalty * max(0.0, recall_target - rec)


def calibrate(records: list[dict], domain: str, grid: list[float] | None = None, objective_fn=objective) -> dict:
    """records: list of {evidence: RiskEvidence, gold: int}. Returns best config delta."""
    grid = grid or [0.30 + 0.02 * i for i in range(26)]  # 0.30..0.80
    best = None
    best_obj = -1e9
    for thr in grid:
        cfg = ArbiterConfig(thresholds={**DOMAIN_THRESHOLDS, domain: thr})
        out = [{"unsafe": decide(r["evidence"], cfg)["unsafe"], "gold": r["gold"]} for r in records]
        if domain == "fraud":
            obj = objective_fn(out, fpr_penalty=1.0, recall_target=0.75, recall_penalty=2.0, fpr_target=0.08)
        else:
            obj = objective_fn(out, fpr_penalty=2.0, recall_target=0.65, recall_penalty=1.0, fpr_target=0.03)
        if obj > best_obj:
            best_obj = obj
            best = thr
    return {"domain": domain, "best_threshold": best, "objective": best_obj}