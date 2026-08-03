"""Generate v2 (budgeted cascade) report artifacts for exp2 (fixed)."""
from __future__ import annotations
import json, math, os, random, collections
import numpy as np

BASE = "experiments/exp2_prior_work_comparison"

def load(p):
    if not os.path.exists(p): return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def load_map(p):
    return {r["id"]: r for r in load(p)}

def metrics(g, p):
    n = len(g); pos = sum(g); neg = n - pos
    tp = sum(1 for a,b in zip(g,p) if a==1 and b==1)
    fp = sum(1 for a,b in zip(g,p) if a==0 and b==1)
    fn = sum(1 for a,b in zip(g,p) if a==1 and b==0)
    tn = sum(1 for a,b in zip(g,p) if a==0 and b==0)
    acc = (tp+tn)/n if n else 0.0
    rec = tp/pos if pos else 0.0
    prec = tp/(tp+fp) if tp+fp else 0.0
    fpr = fp/(fp+tn) if fp+tn else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    neg_f1 = tn/(tn+fp) if tn+fp else 0.0
    macro_f1 = (f1 + neg_f1)/2.0   # v1 convention
    return {"n": n, "n_pos": pos, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "macro_f1": macro_f1,
            "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def pr_auc(y, s):
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    if y.sum() == 0 or y.sum() == len(y): return 0.5
    order = np.argsort(-s, kind="mergesort")
    y_s = y[order]
    prec = np.cumsum(y_s) / np.arange(1, len(y_s)+1)
    rec = np.cumsum(y_s) / y_s.sum()
    return float(np.sum(prec * np.concatenate([[rec[0]], np.diff(rec)])))

def paired_sig(g, p_a, p_b, groups, reps=2000, seed=20260803):
    n = len(g)
    def cf(p):
        tp = sum(1 for t,x in zip(g,p) if t==1 and x==1); fp = sum(1 for t,x in zip(g,p) if t==0 and x==1)
        fn = sum(1 for t,x in zip(g,p) if t==1 and x==0); tn = sum(1 for t,x in zip(g,p) if t==0 and x==0)
        return tp, fp, fn, tn
    ta, fa, fna, tna = cf(p_a); tb, fb, fnb, tnb = cf(p_b)
    def mf(tp, fp, fn, tn):
        f1 = 2*(tp/(tp+fp) if tp+fp else 0)*(tp/(tp+fn) if tp+fn else 0) / ((tp/(tp+fp) if tp+fp else 0)+(tp/(tp+fn) if tp+fn else 0)) if ((tp/(tp+fp) if tp+fp else 0)+(tp/(tp+fn) if tp+fn else 0)) else 0
        return (f1 + tn/(tn+fp) if tn+fp else 0)/2
    d_f1 = mf(*cf(p_a)) - mf(*cf(p_b))
    d_acc = (ta+tna)/n - (tb+tnb)/n
    d_fpr = (fa/(fa+tna) if fa+tna else 0) - (fb/(fb+tnb) if fb+tnb else 0)
    gids = sorted(set(groups)); idx_by = collections.defaultdict(list)
    for i, gr in enumerate(groups): idx_by[gr].append(i)
    rng = random.Random(seed)
    f1s, accs, fprs = [], [], []
    for _ in range(reps):
        sel = []
        for _ in range(len(gids)): sel.extend(idx_by[rng.choice(gids)])
        def calc(p1, p2):
            tp1 = sum(1 for i in sel if g[i]==1 and p1[i]==1); fp1 = sum(1 for i in sel if g[i]==0 and p1[i]==1)
            fn1 = sum(1 for i in sel if g[i]==1 and p1[i]==0); tn1 = sum(1 for i in sel if g[i]==0 and p1[i]==0)
            tp2 = sum(1 for i in sel if g[i]==1 and p2[i]==1); fp2 = sum(1 for i in sel if g[i]==0 and p2[i]==1)
            fn2 = sum(1 for i in sel if g[i]==1 and p2[i]==0); tn2 = sum(1 for i in sel if g[i]==0 and p2[i]==0)
            m = len(sel)
            return (mf(tp1,fp1,fn1,tn1)-mf(tp2,fp2,fn2,tn2), (tp1+tn1)/m-(tp2+tn2)/m,
                    (fp1/(fp1+tn1) if fp1+tn1 else 0)-(fp2/(fp2+tn2) if fp2+tn2 else 0))
        df, da, dfp = calc(p_a, p_b)
        f1s.append(df); accs.append(da); fprs.append(dfp)
    def ci(x):
        x = sorted(x); return [round(x[int(0.025*len(x))],4), round(x[int(0.975*len(x))],4)]
    b_only = sum(1 for a,b in zip(p_a,p_b) if a!=b and b==1)
    a_only = sum(1 for a,b in zip(p_a,p_b) if a!=b and a==1)
    disc = a_only + b_only
    p_mc = 1.0
    if disc:
        k = min(a_only, b_only)
        p_mc = min(1.0, 2.0*sum(math.comb(disc,i)*0.5**disc for i in range(k+1)))
    return {"delta_acc": round(d_acc,4), "delta_macro_f1": round(d_f1,4), "delta_fpr": round(d_fpr,4),
            "ci95_acc": ci(accs), "ci95_macro_f1": ci(f1s), "ci95_fpr": ci(fprs),
            "mcnemar_p": round(p_mc,6), "n_discordant": disc}

# ---------------- load ----------------
P = {}
P["cascade"] = {
    "fraudr1_diag": load_map(f"{BASE}/fraudr1_diag/cascade_predictions/cascade_full_20260803.jsonl"),
    "orbench": load_map(f"{BASE}/orbench/cascade_predictions/cascade_full_20260803.jsonl"),
    "dna": load_map(f"{BASE}/do_not_answer/cascade_predictions/cascade_full_20260803.jsonl"),
    "aegis2": load_map(f"{BASE}/aegis2/cascade_predictions/cascade_full_20260803.jsonl"),
}
P["judge"] = load_map(f"{BASE}/fraudr1_diag/baseline_predictions/fraudr1_official_judge_predictions.jsonl")
P["teacher_diag"] = load_map(f"{BASE}/fraudr1_diag/teacher_predictions/fraudr1_diag_teacher_predictions.jsonl")
P["teacher"] = {
    "fraudr1": load_map(f"{BASE}/fraudr1/teacher_predictions/fraudr1_teacher_predictions.jsonl"),
    "orbench": load_map(f"{BASE}/orbench/teacher_predictions/orbench_teacher_predictions.jsonl"),
    "dna": load_map(f"{BASE}/do_not_answer/teacher_predictions/do_not_answer_teacher_predictions.jsonl"),
    "aegis2": load_map(f"{BASE}/aegis2/teacher_predictions/aegis2_teacher_predictions.jsonl"),
}
P["checker"] = load_map(f"{BASE}/orbench/baseline_predictions/orbench_response_checker_predictions.jsonl")
P["longformer"] = load_map(f"{BASE}/do_not_answer/baseline_predictions/dna_longformer_harmful_predictions.jsonl")
P["nemoguard"] = load_map(f"{BASE}/aegis2/baseline_predictions/aegis_nemoguard_predictions.jsonl")

diag_rows = load_map(f"{BASE}/fraudr1/diagnostic/fraudr1_balanced_diag.jsonl")
orbench_audit = load_map(f"{BASE}/orbench/human_audit/human_audit_adjudicated.jsonl")

def gold_fr1():
    ids = list(diag_rows); return ids, [diag_rows[i]["gold_binary"] for i in ids], [diag_rows[i]["group_id"] for i in ids]
def gold_orbench():
    rows = [r for r in load(f"{BASE}/orbench/unified/orbench_eval.jsonl") if r["id"] in orbench_audit and orbench_audit[r["id"]].get("binary") is not None]
    ids = [r["id"] for r in rows]; return ids, [orbench_audit[i]["binary"] for i in ids], [orbench_audit[i]["group_id"] for i in ids]
def gold_dna():
    rows = load(f"{BASE}/do_not_answer/unified/do_not_answer_eval.jsonl")
    return [r["id"] for r in rows], [r["gold_binary"] for r in rows], [r["group_id"] for r in rows]
def gold_aegis2_full():
    rows = load(f"{BASE}/aegis2/unified/aegis2_eval.jsonl")
    return [r["id"] for r in rows], [r["gold_binary"] for r in rows], [r["group_id"] for r in rows]

def preds(src, ids):
    return [int(src[i]["prediction_binary"]) if i in src and src[i].get("prediction_binary") is not None else 0 for i in ids]
def scores(src, ids, key="risk_score"):
    return [float(src[i].get(key, 0.0)) if i in src and src[i].get(key) is not None else 0.0 for i in ids]

def has_score(src, ids):
    return any(i in src and src[i].get("risk_score") is not None for i in ids[:200])

# ---------------- main table ----------------
MAIN = []
def add_row(bench, method, ids, g, src, pool_n, score_key="risk_score", score_ok=None):
    p = preds(src, ids)
    m = metrics(g, p)
    au = None
    if score_ok is None: score_ok = has_score(src, ids)
    if score_ok:
        au = pr_auc(g, scores(src, ids, score_key))
    MAIN.append({"benchmark": bench, "method": method, "n_pool": pool_n, **m, "auprc": au})

ids, g, grp = gold_fr1()
add_row("Fraud-R1 (balanced diag)", "Official Judge (GPTCheck)", ids, g, P["judge"], 600, score_ok=False)
add_row("Fraud-R1 (balanced diag)", "FraudDistill-MAT (4-agent)", ids, g, P["teacher_diag"], 600)
add_row("Fraud-R1 (balanced diag)", "Budgeted Cascade (ours)", ids, g, P["cascade"]["fraudr1_diag"], 600)
ids, g, grp = gold_orbench()
add_row("OR-Bench", "Official Response Checker", ids, g, P["checker"], 3000, score_ok=False)
add_row("OR-Bench", "FraudDistill-MAT (4-agent)", ids, g, P["teacher"]["orbench"], 3000)
add_row("OR-Bench", "Budgeted Cascade (ours)", ids, g, P["cascade"]["orbench"], 3000)
ids, g, grp = gold_dna()
add_row("Do-Not-Answer", "Longformer-Harmful", ids, g, P["longformer"], 5634, score_key="prob_harmful", score_ok=True)
add_row("Do-Not-Answer", "FraudDistill-MAT (4-agent)", ids, g, P["teacher"]["dna"], 5634)
add_row("Do-Not-Answer", "Budgeted Cascade (ours)", ids, g, P["cascade"]["dna"], 5634)
ids, g, grp = gold_aegis2_full()
nemo_cov = sum(1 for i in ids if i in P["nemoguard"] and P["nemoguard"][i].get("parse_status") == "ok")
nemo_abstain = len(ids) - nemo_cov
nemo_tag = "official, full" if nemo_abstain == 0 else f"full, {nemo_abstain} None abstain"
add_row("Aegis 2.0 (full 1964)", f"NemoGuard-8B ({nemo_tag})", ids, g, P["nemoguard"], 1964, score_ok=False)
add_row("Aegis 2.0 (full 1964)", "FraudDistill-MAT (4-agent)", ids, g, P["teacher"]["aegis2"], 1964)
add_row("Aegis 2.0 (full 1964)", "Budgeted Cascade (ours)", ids, g, P["cascade"]["aegis2"], 1964)

print("=== MAIN TABLE ===")
for r in MAIN:
    au = f"{r['auprc']:.3f}" if r["auprc"] is not None else "-"
    print(f"{r['benchmark']} | {r['method']} | n={r['n']} N+={r['n_pos']} acc={r['acc']:.3f} prec={r['prec']:.3f} rec={r['rec']:.3f} mf1={r['macro_f1']:.3f} fpr={r['fpr']:.3f} auprc={au}")

# ---------------- special tables ----------------
print("\n=== OR-BENCH SPECIAL ===")
rows = [r for r in load(f"{BASE}/orbench/unified/orbench_eval.jsonl") if r["id"] in orbench_audit and orbench_audit[r["id"]].get("binary") is not None]
spec_or = {}
for name, src in [("checker", P["checker"]), ("teacher", P["teacher"]["orbench"]), ("cascade", P["cascade"]["orbench"])]:
    hard = [r for r in rows if r["original_id"].startswith("hard_")]
    reg = [r for r in rows if r["original_id"].startswith("safe_")]
    tox = [r for r in rows if r["original_id"].startswith("toxic_")]
    def sub(rs):
        gg = [orbench_audit[r["id"]]["binary"] for r in rs]
        pp = [int(src[r["id"]]["prediction_binary"]) for r in rs]
        return metrics(gg, pp)
    mh, mr, mt = sub(hard), sub(reg), sub(tox)
    m = metrics([orbench_audit[r["id"]]["binary"] for r in rows], [int(src[r["id"]]["prediction_binary"]) for r in rows])
    spec_or[name] = {"hard_fpr": mh["fpr"], "reg_fpr": mr["fpr"], "toxic_rec": mt["rec"], "macro_f1": m["macro_f1"]}
    print(f"{name}: hard_fpr={mh['fpr']:.3f} reg_fpr={mr['fpr']:.3f} toxic_rec={mt['rec']:.3f} macro_f1={m['macro_f1']:.3f}")

print("\n=== FRAUD-R1 SPECIAL ===")
spec_fr = {}
for name, src in [("judge", P["judge"]), ("teacher", P["teacher_diag"]), ("cascade", P["cascade"]["fraudr1_diag"])]:
    ids = list(diag_rows)
    g = [diag_rows[i]["gold_binary"] for i in ids]
    p = preds(src, ids)
    m = metrics(g, p)
    trust_ids = [i for i in ids if diag_rows[i]["gold_binary"] == 1 and (diag_rows[i].get("metadata") or {}).get("trust_flag")]
    direct_ids = [i for i in ids if diag_rows[i]["gold_binary"] == 1 and not (diag_rows[i].get("metadata") or {}).get("trust_flag")]
    def rec_of(sub):
        return sum(1 for i in sub if src.get(i, {}).get("prediction_binary") == 1)/len(sub) if sub else None
    spec_fr[name] = {"direct_rec": rec_of(direct_ids), "trust_rec": rec_of(trust_ids), "safe_fpr": m["fpr"], "macro_f1": m["macro_f1"]}
    print(f"{name}: direct_rec={rec_of(direct_ids):.3f} trust_rec={rec_of(trust_ids):.3f} safe_fpr={m['fpr']:.3f} macro_f1={m['macro_f1']:.3f}")

print("\n=== COST (cascade) ===")
cost_rows = {}
for bench in ["fraudr1_diag", "orbench", "do_not_answer", "aegis2"]:
    rows = load(f"{BASE}/{bench}/cascade_predictions/cascade_full_20260803.jsonl")
    calls = sum(r.get("usage", {}).get("calls", 0) for r in rows)
    out_t = sum(r.get("usage", {}).get("output", 0) for r in rows)
    miss = sum(r.get("usage", {}).get("input_miss", 0) for r in rows)
    hit = sum(r.get("usage", {}).get("input_hit", 0) for r in rows)
    n = max(len(rows), 1)
    cost = miss/1e6*1.0 + hit/1e6*0.02 + out_t/1e6*2.0
    cost_rows[bench] = {"n": len(rows), "calls": calls, "calls_per_sample": round(calls/n, 3),
                        "in_miss_per_sample": round(miss/n, 1), "in_hit_per_sample": round(hit/n, 1),
                        "out_per_sample": round(out_t/n, 1), "cost_rmb": round(cost, 4)}
    print(bench, cost_rows[bench])

# ---------------- significance ----------------
print("\n=== SIGNIFICANCE (cascade as A) ===")
sig = {}
def run_sig(key, ids, g, grp, src_a, src_b, key_a=None, key_b=None):
    pa = preds(src_a, ids); pb = preds(src_b, ids)
    return paired_sig(g, pa, pb, grp)
ids, g, grp = gold_fr1()
sig["fraudr1_vs_judge"] = run_sig("f", ids, g, grp, P["cascade"]["fraudr1_diag"], P["judge"])
sig["fraudr1_vs_teacher"] = run_sig("f", ids, g, grp, P["cascade"]["fraudr1_diag"], P["teacher_diag"])
ids, g, grp = gold_orbench()
sig["orbench_vs_checker"] = run_sig("o", ids, g, grp, P["cascade"]["orbench"], P["checker"])
sig["orbench_vs_teacher"] = run_sig("o", ids, g, grp, P["cascade"]["orbench"], P["teacher"]["orbench"])
ids, g, grp = gold_dna()
sig["dna_vs_longformer"] = run_sig("d", ids, g, grp, P["cascade"]["dna"], P["longformer"])
sig["dna_vs_teacher"] = run_sig("d", ids, g, grp, P["cascade"]["dna"], P["teacher"]["dna"])
ids, g, grp = gold_aegis2_full()
sig["aegis2_vs_teacher"] = run_sig("a", ids, g, grp, P["cascade"]["aegis2"], P["teacher"]["aegis2"])
for k, v in sig.items():
    print(k, json.dumps(v))

# ---------------- save JSON ----------------
os.makedirs(f"{BASE}/_metrics", exist_ok=True)
os.makedirs(f"{BASE}/_figures", exist_ok=True)
with open(f"{BASE}/_metrics/paired_significance_cascade.json", "w", encoding="utf-8") as f:
    json.dump(sig, f, ensure_ascii=False, indent=2)
with open(f"{BASE}/_metrics/cost_report_cascade.json", "w", encoding="utf-8") as f:
    json.dump(cost_rows, f, ensure_ascii=False, indent=2)
with open(f"{BASE}/_metrics/main_table_cascade.json", "w", encoding="utf-8") as f:
    json.dump(MAIN, f, ensure_ascii=False, indent=2)
with open(f"{BASE}/_metrics/special_tables_cascade.json", "w", encoding="utf-8") as f:
    json.dump({"orbench": spec_or, "fraudr1": spec_fr}, f, ensure_ascii=False, indent=2)

# ---------------- CSV / MD / TEX ----------------
def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"
lines = ["benchmark,method,n_pool,n_gold,n_pos,accuracy,precision,recall,macro_f1,fpr,auprc"]
for r in MAIN:
    au = "" if r["auprc"] is None else f"{r['auprc']:.6f}"
    lines.append(f"{r['benchmark']},{r['method']},{r['n_pool']},{r['n']},{r['n_pos']},{r['acc']:.6f},{r['prec']:.6f},{r['rec']:.6f},{r['macro_f1']:.6f},{r['fpr']:.6f},{au}")
with open(f"{BASE}/_metrics/metrics_8row_table_v2.csv", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")

md = ["| Benchmark | Method | N_pool | N_gold | N+ | Accuracy | Precision | Recall | Macro-F1 | FPR | AUPRC |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in MAIN:
    au = fmt(r["auprc"])
    method = r["method"] if r["method"] != "Budgeted Cascade (ours)" else f"**{r['method']}**"
    md.append(f"| {r['benchmark']} | {method} | {r['n_pool']} | {r['n']} | {r['n_pos']} | {r['acc']:.3f} | {r['prec']:.3f} | {r['rec']:.3f} | {r['macro_f1']:.3f} | {r['fpr']:.3f} | {au} |")
with open(f"{BASE}/_metrics/metrics_8row_table_v2.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(md) + "\n")

tex_rows = []
for r in MAIN:
    au = fmt(r["auprc"])
    b = r["benchmark"].replace("Fraud-R1 (balanced diag)", "Fraud-R1$^*$").replace("Aegis 2.0 (full 1964)", "Aegis 2.0$^\\dagger$").replace(" (valid q+y)", "$^\\dagger$")
    m = r["method"]
    if m == "Budgeted Cascade (ours)":
        m = "\\textbf{Budgeted Cascade}"
    elif m.startswith("NemoGuard"):
        m = "NemoGuard-8B"
    elif m == "FraudDistill-MAT (4-agent)":
        m = "FraudDistill-MAT"
    tex_rows.append(f"{b} & {m} & {r['n_pool']} & {r['n']} & {r['n_pos']} & {r['acc']:.3f} & {r['prec']:.3f} & {r['rec']:.3f} & {r['macro_f1']:.3f} & {r['fpr']:.3f} & {au} \\\\")
tex = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{l l r r r c c c c c c}
\toprule
Dataset & Method & N\textsubscript{pool} & N\textsubscript{gold} & N+ & Acc. & Prec. & Rec. & M-F1 & FPR$\downarrow$ & AUPRC \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\caption{Budgeted-cascade behavior-error detection vs prior baselines and the four-agent MAT on identical (q, y) pairs. $^*$Fraud-R1 uses the new balanced diagnostic gold (300 positives). $\dagger$Aegis now covers all 1,964 official rows: the 1,151 ``None''/empty-answer rows are all safe negatives; NemoGuard-8B abstains on 16 of them (no agent response) and covers the remaining 1,948 rows.}
\label{tab:exp2_v2}
\end{table}"""
with open(f"{BASE}/table_exp2.tex", "w", encoding="utf-8", newline="\n") as f:
    f.write(tex)
print("\nCSV/MD/TEX written")
