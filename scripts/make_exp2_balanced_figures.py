# -*- coding: utf-8 -*-
"""Exp2 balanced final figures: PR curves + metric comparison bars."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("C:/Users/18201/Desktop/FraudDistill")
sys.path.insert(0, str(REPO / "src"))
BASE = REPO / "experiments" / "exp2_prior_work_comparison" / "balanced_design"

def read_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

def score_for(bench, row):
    if bench == "fraudr1": return row.get("fraud_assistance_score")
    if bench == "orbench":
        rgt = str(row.get("request_gold_type") or row.get("stratum") or "")
        if rgt in ("hard_safe", "regular_safe"): return row.get("over_refusal_score")
        return row.get("behavior_error_score") or max(float(row.get("general_harmful_compliance_score") or 0), float(row.get("partial_leakage_score") or 0))
    if bench == "dna": return row.get("general_harmful_compliance_score")
    if bench == "aegis":
        s = row.get("response_content_harm_score")
        return s if s is not None else row.get("risk_score")
    return None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
except Exception as e:
    print("matplotlib/sklearn unavailable:", e)
    raise SystemExit(0)

order = ["fraudr1", "orbench", "dna", "aegis"]
manifest = read_jsonl(BASE / "manifests" / "balanced_test_manifest.jsonl")
by_src = {"fraudr1": "fraudr1", "orbench": "orbench", "dna": "do_not_answer", "aegis": "aegis2"}
gold = {b: {r["sample_id"]: int(r["gold_binary"]) for r in read_jsonl(BASE / "gold" / f"{b}_gold.jsonl")} for b in order}
preds = {b: {r["id"]: r for r in read_jsonl(BASE / "predictions" / f"{b}_fd_predictions.jsonl")} for b in order}
bases = {b: {r["id"]: r for r in read_jsonl(BASE / "baselines" / f"{b}_baseline_predictions.jsonl")} for b in order}

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, b in zip(axes.flat, order):
    y, s, bs = [], [], []
    for m in manifest:
        if m["source"] != by_src[b]: continue
        sid = m["sample_id"]; g = gold[b].get(sid)
        if g is None: continue
        p = preds[b].get(sid)
        if p is None: continue
        sc = score_for(b, p)
        if sc is None: continue
        y.append(g); s.append(float(sc))
        bb = bases[b].get(sid)
        if bb is not None and bb.get("prediction_binary") in (0, 1):
            bs.append(int(bb["prediction_binary"]))
    y = np.array(y); s = np.array(s)
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    ax.plot(rec, prec, label=f"FraudDistill (AP={ap:.3f})", lw=2)
    if len(bs) == len(y):
        b_arr = np.array(bs)
        tp = int(((b_arr==1)&(y==1)).sum()); fp = int(((b_arr==1)&(y==0)).sum()); fn = int(((b_arr==0)&(y==1)).sum())
        bp = tp/max(tp+fp,1); br = tp/max(tp+fn,1)
        ax.scatter([br], [bp], marker="X", s=140, color="crimson", zorder=5, label=f"Original work (P={bp:.3f}, R={br:.3f})")
    ax.set_title(f"{b} (n={len(y)})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
(BASE / "figures").mkdir(exist_ok=True)
fig.savefig(BASE / "figures" / "pr_curves_final_balanced.png", dpi=150)
print("saved pr_curves_final_balanced.png")

d = json.loads((BASE / "metrics" / "canonical_metrics_balanced.json").read_text(encoding="utf-8"))
labels = []; fd = []; bl = []
for b in order:
    e = d["results"].get(b)
    if not e: continue
    base = e.get("baseline")
    for m in ["accuracy", "recall", "macro_f1", "mcc"]:
        labels.append(f"{b}\n{m}"); fd.append(e[m]); bl.append(base[m] if base else 0)
x = np.arange(len(labels)); w = 0.38
fig2, ax2 = plt.subplots(figsize=(13, 5.5))
ax2.bar(x - w/2, fd, w, label="FraudDistill", color="#4C72B0")
ax2.bar(x + w/2, bl, w, label="Original work", color="#DD8452")
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylim(0, 1.05); ax2.legend(); ax2.grid(axis="y", alpha=0.3)
ax2.set_title("FraudDistill vs original work on the final balanced diagnostics set")
fig2.tight_layout()
fig2.savefig(BASE / "figures" / "metric_comparison_final_balanced.png", dpi=150)
print("saved metric_comparison_final_balanced.png")
