# -*- coding: utf-8 -*-
"""Final Student vs Neural-SoftDistill paired significance (offline, 0 API).

- paired bootstrap (n_boot=10000, seed=20260809) on test n=1262:
  each iteration resamples the SAME (final_score, soft_score, gold) pairs,
  applies each model's own official frozen threshold
  (final 0.5622, soft 0.5) and computes delta = final - soft for
  Macro-F1 / Recall / FPR / AUPRC / MCC / Acc / 4-class-MF1.
- McNemar exact test on the two binary decision vectors.
4-class gold mapping matches the official evaluator: safe -> 0,
unsafe -> 1 (dataset gold_type is binary; see evaluate_final_student.py).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
DSET = REPO / "data/prepared/exp3_agent_distillation/exp3_dataset.jsonl"
SOFT_PREDS = REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/eval_soft/predictions_test.jsonl"
FINAL_PREDS = REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/test_eval/predictions_test.jsonl"
CAL = REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/calibration.json"
OUT = REPO / "experiments/exp3_agent_distillation_ablation/outputs/metrics/final_vs_soft_significance.json"

FINAL_THR = float(json.loads(CAL.read_text(encoding="utf-8-sig"))["threshold"])
SOFT_THR = 0.5
TYPE_IDS = {"safe": 0, "fraud_assistance": 1, "refusal_failure": 2, "over_refusal": 3}
def gold_type_id(r):
    # official evaluator mapping (evaluate_final_student.full_metrics):
    # fine-grained 4-class gold when present, otherwise unsafe -> 1
    v = TYPE_IDS.get(r.get("gold_type") or "", None)
    return v if v is not None else (0 if r["gold_label"] == "safe" else 1)
N_BOOT = 10000
SEED = 20260809

rows = [json.loads(l) for l in DSET.open(encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("split") == "test" and r.get("gold_label") in ("safe", "unsafe")]
soft = {r["id"]: r for r in (json.loads(l) for l in SOFT_PREDS.open(encoding="utf-8") if l.strip())}
final = {r["id"]: r for r in (json.loads(l) for l in FINAL_PREDS.open(encoding="utf-8") if l.strip())}
assert all(r["id"] in soft and r["id"] in final for r in rows), "id mismatch"

y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows], dtype=int)
gt = np.array([gold_type_id(r) for r in rows], dtype=int)
fs = np.array([final[r["id"]]["risk_score"] for r in rows], dtype=float)
ss = np.array([soft[r["id"]]["risk_score"] for r in rows], dtype=float)
ft = np.array([TYPE_IDS.get(final[r["id"]]["risk_type"], 0) for r in rows], dtype=int)
st = np.array([TYPE_IDS.get(soft[r["id"]]["risk_type"], 0) for r in rows], dtype=int)
n = len(rows)

def metrics_vec(yv, s, thr, tv, gtv):
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (yv == 1)).sum()); fn = int(((p == 0) & (yv == 1)).sum())
    fp = int(((p == 1) & (yv == 0)).sum()); tn = int(((p == 0) & (yv == 0)).sum())
    mf1 = float(f1_score(yv, p, average="macro", zero_division=0))
    rec = tp / max(tp + fn, 1); fpr = fp / max(tn + fp, 1)
    auprc = float(average_precision_score(yv, s)) if 0 < yv.sum() < len(yv) else None
    mcc = float(matthews_corrcoef(yv, p))
    acc = (tp + tn) / len(yv)
    c4 = float(f1_score(gtv, tv, average="macro", zero_division=0))
    return {"macro_f1": mf1, "recall": rec, "fpr": fpr, "auprc": auprc, "mcc": mcc, "acc": acc, "4class_mf1": c4}

base_final = metrics_vec(y, fs, FINAL_THR, ft, gt)
base_soft = metrics_vec(y, ss, SOFT_THR, st, gt)
obs = {k: base_final[k] - base_soft[k] for k in base_final}

rng = np.random.default_rng(SEED)
deltas = {k: np.empty(N_BOOT) for k in base_final}
for b in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    mf = metrics_vec(y[idx], fs[idx], FINAL_THR, ft[idx], gt[idx])
    ms = metrics_vec(y[idx], ss[idx], SOFT_THR, st[idx], gt[idx])
    for k in deltas:
        vf, vs = mf[k], ms[k]
        deltas[k][b] = (vf - vs) if (vf is not None and vs is not None) else np.nan

boot = {}
for k, arr in deltas.items():
    valid = arr[~np.isnan(arr)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    boot[k] = {"observed_delta": round(float(obs[k]), 4),
               "bootstrap_mean_delta": round(float(valid.mean()), 4),
               "ci95": [round(float(lo), 4), round(float(hi), 4)],
               "ci_excludes_zero": bool((lo > 0) or (hi < 0)),
               "n_valid": int(len(valid))}

pf = (fs >= FINAL_THR).astype(int); ps = (ss >= SOFT_THR).astype(int)
b = int(((pf == 0) & (ps == 1)).sum())
c = int(((pf == 1) & (ps == 0)).sum())
mcnemar = {"b_only_soft_right": b, "c_only_final_right": c,
           "p_exact": float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue) if b + c else 1.0}

result = {
    "n_test": n,
    "final_threshold": FINAL_THR, "soft_threshold": SOFT_THR,
    "metrics_final": {k: round(v, 4) for k, v in base_final.items()},
    "metrics_soft": {k: round(v, 4) for k, v in base_soft.items()},
    "observed_delta": {k: round(v, 4) for k, v in obs.items()},
    "bootstrap_n": N_BOOT, "seed": SEED,
    "bootstrap_ci95": boot,
    "mcnemar": mcnemar,
    "note": "paired: same samples, each model uses its own official frozen threshold (final 0.5622 from dev calibration; soft 0.5 fixed protocol). 4class gold: safe->0 unsafe->1, pred argmax type.",
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=1))