"""Generate v2 figures for the budgeted cascade report."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "experiments/exp2_prior_work_comparison"
FIG = f"{BASE}/_figures"
os.makedirs(FIG, exist_ok=True)

main = json.load(open(f"{BASE}/_metrics/main_table_cascade.json", encoding="utf-8"))

# ---------- 1. recall vs FPR scatter (one panel per benchmark) ----------
bench_groups = ["Fraud-R1 (balanced diag)", "OR-Bench", "Do-Not-Answer", "Aegis 2.0 (full 1964)"]
styles = {"Official Judge (GPTCheck)": ("s", "#888888"), "Official Response Checker": ("s", "#888888"),
          "Longformer-Harmful": ("s", "#888888"), "NemoGuard-8B": ("s", "#888888"),
          "FraudDistill-MAT (4-agent)": ("o", "#d95f02"), "Budgeted Cascade (ours)": ("D", "#1b9e77")}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, bg in zip(axes, bench_groups):
    for r in main:
        if r["benchmark"] != bg: continue
        key = r["method"]
        if key.startswith("NemoGuard"): key = "NemoGuard-8B"
        shape, color = styles.get(key, ("o", "#333333"))
        label = "NemoGuard-8B (full)" if key == "NemoGuard-8B" else key
        ax.scatter(r["fpr"], r["rec"], marker=shape, color=color, s=70, zorder=3,
                   label=label if bg == bench_groups[0] else None)
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("FPR"); ax.set_ylabel("Recall")
    ax.set_title(bg, fontsize=9)
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("Exp2 v2: Budgeted Cascade vs prior methods (recall / FPR)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG}/figure_v2_recall_fpr.png", dpi=150)
plt.close(fig)

# ---------- 2. cost per 1k samples ----------
teacher_cost = {"fraudr1": (60.4561, 8564), "orbench": (12.5889, 3000), "do_not_answer": (15.3666, 5634), "aegis2": (4.9177, 1964)}
cascade_cost = json.load(open(f"{BASE}/_metrics/cost_report_cascade.json", encoding="utf-8"))
labels = ["Fraud-R1", "OR-Bench", "Do-Not-Answer", "Aegis 2.0"]
tc = [teacher_cost["fraudr1"][0]/teacher_cost["fraudr1"][1]*1000,
      teacher_cost["orbench"][0]/teacher_cost["orbench"][1]*1000,
      teacher_cost["do_not_answer"][0]/teacher_cost["do_not_answer"][1]*1000,
      teacher_cost["aegis2"][0]/teacher_cost["aegis2"][1]*1000]
cc = [cascade_cost["fraudr1_diag"]["cost_rmb"]/cascade_cost["fraudr1_diag"]["n"]*1000,
      cascade_cost["orbench"]["cost_rmb"]/cascade_cost["orbench"]["n"]*1000,
      cascade_cost["do_not_answer"]["cost_rmb"]/cascade_cost["do_not_answer"]["n"]*1000,
      cascade_cost["aegis2"]["cost_rmb"]/cascade_cost["aegis2"]["n"]*1000]
x = np.arange(4); w = 0.36
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x-w/2, tc, w, label="4-agent MAT (full pool)", color="#d95f02", alpha=0.85)
ax.bar(x+w/2, cc, w, label="Budgeted Cascade (this run)", color="#1b9e77", alpha=0.85)
for i in range(4):
    ax.text(i-w/2, tc[i]+0.05, f"{tc[i]:.2f}", ha="center", fontsize=8)
    ax.text(i+w/2, cc[i]+0.05, f"{cc[i]:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("DeepSeek API cost (RMB per 1,000 samples)")
ax.set_title("Exp2 v2: API cost per 1,000 samples (teacher vs budgeted cascade)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIG}/figure_v2_cost.png", dpi=150); plt.close(fig)

# ---------- 3. dev objective landscape (combined) ----------
import sys
sys.path.insert(0, "src")
from frauddistill.arbitration.deterministic_arbiter import ArbiterConfig, DOMAIN_THRESHOLDS, decide
from frauddistill.arbitration.evidence import RiskEvidence
from frauddistill.gates.refusal_gate import run_refusal_gate

def load_map(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]
def ev_to_evidence(r):
    e = r.get("evidence") or {}
    return RiskEvidence(
        prompt_harm_intent=float(e.get("prompt_harm_intent",0) or 0),
        answer_task_alignment=float(e.get("answer_task_alignment",0) or 0),
        explicit_refusal=float(e.get("explicit_refusal",0) or 0),
        safe_redirection=float(e.get("safe_redirection",0) or 0),
        actionable_assistance=float(e.get("actionable_assistance",0) or 0),
        leakage_after_refusal=float(e.get("leakage_after_refusal",0) or 0),
        trust_facilitation=float(e.get("trust_facilitation",0) or 0),
        general_harmful_compliance=float(e.get("general_harmful_compliance",0) or 0),
        anti_fraud_education=float(e.get("anti_fraud_education",0) or 0),
        risk_domain=e.get("risk_domain", r.get("risk_domain","unknown")),
        confidence=float(e.get("confidence", r.get("confidence",0) or 0)),
        provisional_label=e.get("provisional_label","unknown"),
        evidence_spans=e.get("evidence_spans",[]),
        privacy_sensitive_query=bool(e.get("privacy_sensitive_query",False)),
    )
def metrics(g,p):
    n=len(g); pos=sum(g); neg=n-pos
    tp=sum(1 for a,b in zip(g,p) if a==1 and b==1); fp=sum(1 for a,b in zip(g,p) if a==0 and b==1)
    fn=sum(1 for a,b in zip(g,p) if a==1 and b==0); tn=sum(1 for a,b in zip(g,p) if a==0 and b==0)
    f1=2*(tp/(tp+fp) if tp+fp else 0)*(tp/(tp+fn) if tp+fn else 0)/((tp/(tp+fp) if tp+fp else 0)+(tp/(tp+fn) if tp+fn else 0)) if ((tp/(tp+fp) if tp+fp else 0)+(tp/(tp+fn) if tp+fn else 0)) else 0
    fpr=fp/(fp+tn) if fp+tn else 0; rec=tp/pos if pos else 0
    return f1, fpr, rec
def obj_fraud(m): return m[0]-1.0*max(0.0,m[1]-0.08)-2.0*max(0.0,0.75-m[2])
def obj_general(m): return m[0]-2.0*max(0.0,m[1]-0.03)-1.0*max(0.0,0.65-m[2])

files = {
 "fraudr1_diag": (f"{BASE}/fraudr1_diag/cascade_predictions/cascade_dev_20260803.jsonl", obj_fraud),
 "orbench": (f"{BASE}/orbench/cascade_predictions/cascade_dev_20260803.jsonl", obj_general),
 "dna": (f"{BASE}/do_not_answer/cascade_predictions/cascade_dev_20260803.jsonl", obj_general),
 "aegis2": (f"{BASE}/aegis2/cascade_predictions/cascade_dev_20260803.jsonl", obj_general),
}
data = {}
for name, (p, fn) in files.items():
    rows = [r for r in load_map(p) if r.get("gold_binary") is not None]
    g = [r["gold_binary"] for r in rows]
    gate = [run_refusal_gate(r.get("query",""), r.get("answer",""), min_confidence=0.85).decision=="safe_refusal" for r in rows]
    evs = [ev_to_evidence(r) for r in rows]
    data[name] = {"g": g, "rows": rows, "gate": gate, "evs": evs, "fn": fn}
def replay(name, thr):
    d = data[name]; preds = []
    for safe, ev, r in zip(d["gate"], d["evs"], d["rows"]):
        if safe: preds.append(0); continue
        preds.append(1 if decide(ev, ArbiterConfig(thresholds=thr), answer_text=r.get("answer",""), query_text=r.get("query",""))["unsafe"] else 0)
    return preds
fraud_ts = [x/100 for x in range(26, 70, 2)]
gen_ts = [x/100 for x in range(30, 72, 2)]
grid = np.zeros((len(fraud_ts), len(gen_ts)))
for i, ft in enumerate(fraud_ts):
    for j, gt in enumerate(gen_ts):
        thr = dict(DOMAIN_THRESHOLDS); thr["fraud"]=ft; thr["general_safety"]=gt; thr["benign"]=gt; thr["unknown"]=gt
        tot = 0.0
        for name in files:
            m = metrics(data[name]["g"], replay(name, thr))
            tot += data[name]["fn"](m)
        grid[i, j] = tot
fig, ax = plt.subplots(figsize=(7.5, 5))
im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
               extent=[gen_ts[0]-0.01, gen_ts[-1]+0.01, fraud_ts[0]-0.01, fraud_ts[-1]+0.01])
ax.set_xlabel("general_safety threshold"); ax.set_ylabel("fraud threshold")
ax.set_title("Dev combined objective  (fraudr1+orbench+dna+aegis2 dev300)")
cb = fig.colorbar(im, ax=ax); cb.set_label("sum of guide-11.2 objectives")
# mark frozen point
fz = (0.34, 0.46)
ax.plot(fz[0], fz[1], "o", color="red", ms=8, mfc="none", mew=2)
ax.text(fz[0], fz[1], " frozen (0.34, 0.46)", color="red", fontsize=9, va="bottom")
fig.tight_layout(); fig.savefig(f"{FIG}/figure_v2_dev_objective.png", dpi=150); plt.close(fig)
print("dev objective heatmap done")

# ---------- 4. cascade confusion matrices ----------
from matplotlib.patches import Rectangle
for bench, label in [("fraudr1_diag", "Fraud-R1 (balanced diag)"), ("orbench", "OR-Bench (600 audited)"),
                     ("do_not_answer", "Do-Not-Answer (5634)"), ("aegis2", "Aegis 2.0 (1964)")]:
    rows = [r for r in load_map(f"{BASE}/{bench}/cascade_predictions/cascade_full_20260803.jsonl") if r.get("gold_binary") is not None]
    g = [r["gold_binary"] for r in rows]; p = [r.get("prediction_binary") or 0 for r in rows]
    tn = sum(1 for a,b in zip(g,p) if a==0 and b==0); fp = sum(1 for a,b in zip(g,p) if a==0 and b==1)
    fn = sum(1 for a,b in zip(g,p) if a==1 and b==0); tp = sum(1 for a,b in zip(g,p) if a==1 and b==1)
    M = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(M, cmap="Blues")
    for ii in range(2):
        for jj in range(2):
            ax.text(jj, ii, str(M[ii, jj]), ha="center", va="center", fontsize=16,
                    color="white" if M[ii, jj] > M.max()*0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred safe", "pred unsafe"]); ax.set_yticklabels(["gold safe", "gold unsafe"])
    ax.set_title(f"Cascade · {label}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/confusion_cascade_{bench}.png", dpi=150); plt.close(fig)
print("confusion matrices done")
print("ALL FIGURES WRITTEN")
