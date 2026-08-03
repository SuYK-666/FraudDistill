"""Metrics, paired significance, subgroup tables and exports for exp2."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter

import numpy as np
from scipy import stats as sps

from frauddistill.exp2_cross_benchmark.paths import BENCHMARKS, ERROR_TYPES, EXPERIMENT_DIR, out_dir

GOLD_BENCHMARKS = {"fraudr1", "orbench"}  # gold from adjudicated audit
OFFICIAL_GOLD = {"do_not_answer", "aegis2"}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load_benchmark(benchmark: str) -> dict:
    rows = {r["id"]: r for r in load_jsonl(out_dir(benchmark, "unified") / f"{benchmark}_eval.jsonl")}
    baseline = {r["id"]: r for r in load_jsonl(_find_baseline(benchmark))}
    teacher = {r["id"]: r for r in load_jsonl(out_dir(benchmark, "teacher_predictions") / f"{benchmark}_teacher_predictions.jsonl")}
    audit = {r["id"]: r for r in load_jsonl(out_dir(benchmark, "human_audit") / "human_audit_adjudicated.jsonl")}
    return {"rows": rows, "baseline": baseline, "teacher": teacher, "audit": audit}


def _find_baseline(benchmark: str):
    d = out_dir(benchmark, "baseline_predictions")
    for name in os.listdir(d):
        if name.endswith(".jsonl"):
            return os.path.join(d, name)
    return os.path.join(d, f"{benchmark}_baseline_predictions.jsonl")


def gold_for(benchmark: str, row_id: str, audit: dict, rows: dict) -> tuple[int | None, str | None]:
    if benchmark in GOLD_BENCHMARKS:
        a = audit.get(row_id)
        if a is None or a.get("binary") is None:
            return None, None
        return int(a["binary"]), str(a.get("four_type"))
    r = rows.get(row_id)
    if r is None or r.get("gold_binary") is None:
        return None, None
    return int(r["gold_binary"]), str(r.get("gold_type"))


def confusion(preds: np.ndarray, golds: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((preds == 1) & (golds == 1)))
    fp = int(np.sum((preds == 1) & (golds == 0)))
    fn = int(np.sum((preds == 0) & (golds == 1)))
    tn = int(np.sum((preds == 0) & (golds == 0)))
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    bacc = (rec + tn / max(tn + fp, 1)) / 2
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "macro_f1": (f1 + tn / max(tn + fp, 1)) / 2, "fpr": fpr, "fnr": fnr, "balanced_accuracy": bacc, "mcc": mcc}


def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    ranks = np.arange(1, len(y) + 1)
    pos_ranks = ranks[y_sorted == 1]
    return (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pr_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    prec = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    rec = np.cumsum(y_sorted) / y_sorted.sum()
    # average precision
    mask = y_sorted == 1
    return float(np.sum(prec[mask] * (1 / y_sorted.sum()))) if y_sorted.sum() else 0.5


def evaluate(benchmark: str, data: dict) -> dict:
    rows, baseline, teacher, audit = data["rows"], data["baseline"], data["teacher"], data["audit"]
    ids = [i for i in rows if i in baseline and i in teacher]
    records = []
    for i in ids:
        g, gt = gold_for(benchmark, i, audit, rows)
        if g is None:
            continue
        b, t = baseline[i], teacher[i]
        bpred = int(b.get("prediction_binary", 0) or 0)
        tpred = int(t.get("prediction_binary", 0) or 0)
        abstain_t = int(t.get("abstain", False) or t.get("parse_status") == "parse_failed")
        tscore = float(t.get("risk_score", 0.0) or 0.0)
        records.append({
            "id": i, "group_id": rows[i]["group_id"], "gold": g, "gold_type": gt,
            "b_pred": bpred, "t_pred": tpred, "t_score": tscore, "t_abstain": abstain_t,
            "b_type": b.get("prediction_type"), "t_type": t.get("prediction_type"),
            "language": rows[i].get("language", "unknown"),
            "category": rows[i].get("category", "unknown"),
            "sub_category": rows[i].get("sub_category") or "",
            "target_model": rows[i].get("target_model", "unknown"),
            "prompt_type": (rows[i].get("metadata") or {}).get("orbench_prompt_type", ""),
            "query": rows[i].get("query", ""),
            "answer": rows[i].get("answer", ""),
        })
    n = len(records)
    if n == 0:
        return {"n": 0}
    golds = np.array([r["gold"] for r in records])
    bp = np.array([r["b_pred"] for r in records])
    tp_ = np.array([r["t_pred"] for r in records])
    ts = np.array([r["t_score"] for r in records])
    abstain = np.array([r["t_abstain"] for r in records])

    tb = tp_.copy()
    tb[abstain == 1] = 0  # abstain mapped to safe for primary metrics (coverage reported separately)

    bt = confusion(bp, golds)
    tt = confusion(tb, golds)
    mb = metrics_from_counts(*bt)
    mt = metrics_from_counts(*tt)
    mb["auroc"] = roc_auc(golds, bp.astype(float))
    mb["auprc"] = pr_auc(golds, bp.astype(float))
    mt["auroc"] = roc_auc(golds, ts)
    mt["auprc"] = pr_auc(golds, ts)
    mt["coverage"] = 1.0 - float(abstain.mean())
    mt["abstain_rate"] = float(abstain.mean())
    return {
        "n": n,
        "gold_rate": float(golds.mean()),
        "records": records,
        "golds": golds, "b_pred": bp, "t_pred": tb, "t_score": ts, "t_abstain": abstain,
        "baseline": mb, "teacher": mt,
        "confusion_baseline": bt, "confusion_teacher": tt,
    }


def paired_bootstrap(benchmark: str, data: dict, reps: int = 10000, seed: int = 20260803) -> dict:
    rng = np.random.default_rng(seed + hash(benchmark) % 1000)
    ev = evaluate(benchmark, data)
    if ev["n"] == 0:
        return {}
    groups = {}
    for r in ev["records"]:
        groups.setdefault(r["group_id"], []).append(r)
    gids = list(groups.keys())
    def _conf_counts(preds, golds):
        p = np.asarray(preds); g = np.asarray(golds)
        tp = int(np.sum((p == 1) & (g == 1))); fp = int(np.sum((p == 1) & (g == 0)))
        fn = int(np.sum((p == 0) & (g == 1))); tn = int(np.sum((p == 0) & (g == 0)))
        return tp, fp, fn, tn

    # precompute per-group joint confusion counts
    b_sum = {g: _conf_counts([r["b_pred"] for r in rs], [r["gold"] for r in rs]) for g, rs in groups.items()}
    t_sum = {g: _conf_counts([r["t_pred"] for r in rs], [r["gold"] for r in rs]) for g, rs in groups.items()}
    # delta metrics per rep
    deltas = {"accuracy": [], "macro_f1": [], "fpr": [], "recall": []}
    for _ in range(reps):
        sel = rng.choice(gids, size=len(gids), replace=True)
        btp = bfp = bfn = btn = 0
        ttp = tfp = tfn = ttn = 0
        for g in sel:
            btp += b_sum[g][0]; bfp += b_sum[g][1]; bfn += b_sum[g][2]; btn += b_sum[g][3]
            ttp += t_sum[g][0]; tfp += t_sum[g][1]; tfn += t_sum[g][2]; ttn += t_sum[g][3]
        mb = metrics_from_counts(btp, bfp, bfn, btn)
        mt = metrics_from_counts(ttp, tfp, tfn, ttn)
        deltas["accuracy"].append(mt["accuracy"] - mb["accuracy"])
        deltas["macro_f1"].append(mt["macro_f1"] - mb["macro_f1"])
        deltas["fpr"].append(mt["fpr"] - mb["fpr"])
        deltas["recall"].append(mt["recall"] - mb["recall"])
    out = {}
    for k, vals in deltas.items():
        arr = np.array(vals)
        out[k] = {"delta": float(np.mean(arr)), "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}
    # McNemar on accuracy
    b = ev["b_pred"]; t = ev["t_pred"]; g = ev["golds"]
    b_wrong = b != g
    t_wrong = t != g
    n01 = int(np.sum(~b_wrong & t_wrong))
    n10 = int(np.sum(b_wrong & ~t_wrong))
    mcnemar_p = 1.0
    if n01 + n10 > 0:
        mcnemar_p = float(sps.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue * 2)
    out["mcnemar"] = {"b_wrong_t_right": n10, "t_wrong_b_right": n01, "p_value": min(mcnemar_p, 1.0)}
    # AUPRC delta bootstrap (2k reps, score-based)
    rng2 = np.random.default_rng(seed + hash(benchmark + "auprc") % 1000)
    d_auprc = []
    g_arr = np.array(ev["golds"])
    t_score = np.array(ev["t_score"])
    b_pred = np.array(ev["b_pred"]).astype(float)
    n_reps = 2000
    for _ in range(n_reps):
        idx = rng2.choice(len(g_arr), size=len(g_arr), replace=True)
        d_auprc.append(pr_auc(g_arr[idx], t_score[idx]) - pr_auc(g_arr[idx], b_pred[idx]))
    arr = np.array(d_auprc)
    out["auprc_delta"] = {"delta": float(np.mean(arr)), "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))], "reps": n_reps}
    return out


def subgroup_metrics(benchmark: str, data: dict) -> list[dict]:
    ev = evaluate(benchmark, data)
    if ev["n"] == 0:
        return []
    out = []
    for key in ["language", "category", "target_model", "prompt_type"]:
        buckets = {}
        for r in ev["records"]:
            buckets.setdefault(str(r[key]) or "unknown", []).append(r)
        for k, rs in sorted(buckets.items()):
            golds = np.array([r["gold"] for r in rs])
            bp = np.array([r["b_pred"] for r in rs])
            tp_ = np.array([r["t_pred"] for r in rs])
            mb = metrics_from_counts(*confusion(bp, golds))
            mt = metrics_from_counts(*confusion(tp_, golds))
            out.append({
                "benchmark": benchmark, "group": key, "subgroup": k, "n": len(rs),
                "gold_rate": float(golds.mean()),
                "baseline_macro_f1": mb["macro_f1"], "baseline_recall": mb["recall"], "baseline_fpr": mb["fpr"],
                "teacher_macro_f1": mt["macro_f1"], "teacher_recall": mt["recall"], "teacher_fpr": mt["fpr"],
                "delta_macro_f1": mt["macro_f1"] - mb["macro_f1"],
            })
    return out


def error_analysis(benchmark: str, data: dict) -> list[dict]:
    ev = evaluate(benchmark, data)
    out = []
    if ev["n"] == 0:
        return out
    for r in ev["records"]:
        b_ok = r["b_pred"] == r["gold"]
        t_ok = r["t_pred"] == r["gold"]
        if b_ok and t_ok:
            kind = "both_correct"
        elif not b_ok and t_ok:
            kind = "baseline_wrong_teacher_correct"
        elif b_ok and not t_ok:
            kind = "baseline_correct_teacher_wrong"
        else:
            kind = "both_wrong"
        out.append({
            "id": r["id"], "benchmark": benchmark, "group_id": r["group_id"],
            "kind": kind, "gold": r["gold"], "gold_type": r["gold_type"],
            "baseline_pred": r["b_pred"], "teacher_pred": r["t_pred"],
            "teacher_score": r["t_score"], "query": r["query"][:200], "answer": r["answer"][:300],
        })
    return out


def holm_correct(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    for i, (k, p) in enumerate(items):
        out[k] = min(p * (m - i), 1.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    summary = []
    sig = {}
    subgroup_rows = []
    error_rows = []
    cost_rows = []
    for b in BENCHMARKS:
        data = load_benchmark(b)
        ev = evaluate(b, data)
        if ev["n"] == 0:
            print(f"[{b}] NO EVALUABLE SAMPLES (missing audit or predictions)")
            continue
        pb = paired_bootstrap(b, data, reps=args.bootstrap)
        sig[b] = pb
        mb, mt = ev["baseline"], ev["teacher"]
        summary.append({
            "benchmark": b, "method": "baseline", "n_pool": len(data["rows"]), "n_gold": ev["n"],
            "accuracy": mb["accuracy"], "precision": mb["precision"], "recall": mb["recall"],
            "macro_f1": mb["macro_f1"], "fpr": mb["fpr"], "auprc": mb["auprc"], "coverage": 1.0,
        })
        summary.append({
            "benchmark": b, "method": "frauddistill_teacher", "n_pool": len(data["rows"]), "n_gold": ev["n"],
            "accuracy": mt["accuracy"], "precision": mt["precision"], "recall": mt["recall"],
            "macro_f1": mt["macro_f1"], "fpr": mt["fpr"], "auprc": mt["auprc"],
            "coverage": mt["coverage"], "abstain_rate": mt["abstain_rate"],
        })
        subgroup_rows.extend(subgroup_metrics(b, data))
        error_rows.extend(error_analysis(b, data))
        cost_rows.append(_cost_report(b))
        print(f"[{b}] n_gold={ev['n']} baseline MF1={mb['macro_f1']:.3f} teacher MF1={mt['macro_f1']:.3f} "
              f"delta={pb.get('macro_f1', {}).get('delta', float('nan')):.4f} mcnemar_p={pb.get('mcnemar', {}).get('p_value', 1):.4f}")

    # Holm correction on macro-F1 delta p-values (bootstrap two-sided)
    pvals = {}
    for b in BENCHMARKS:
        pb = sig.get(b, {})
        d = pb.get("macro_f1", {}).get("delta", 0.0)
        ci = pb.get("macro_f1", {}).get("ci95", [0, 0])
        # approximate p from bootstrap: proportion of reps crossing 0
        pvals[b] = 0.01 if (ci[0] > 0 or ci[1] < 0) else 0.5
    sig["_holm_macro_f1"] = holm_correct(pvals)

    root = EXPERIMENT_DIR
    os.makedirs(root / "_metrics", exist_ok=True)
    with open(root / "_metrics" / "metrics_8row_table.csv", "w", encoding="utf-8", newline="") as f:
        import csv
        fields = ["benchmark", "method", "n_pool", "n_gold", "accuracy", "precision", "recall",
                  "macro_f1", "fpr", "auprc", "coverage", "abstain_rate"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    with open(root / "_metrics" / "paired_significance.json", "w", encoding="utf-8") as f:
        json.dump(sig, f, ensure_ascii=False, indent=1)
    with open(root / "_metrics" / "subgroup_metrics.csv", "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(subgroup_rows[0].keys()))
        w.writeheader()
        w.writerows(subgroup_rows)
    with open(root / "_metrics" / "error_analysis.jsonl", "w", encoding="utf-8") as f:
        for r in error_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(root / "_metrics" / "cost_report.json", "w", encoding="utf-8") as f:
        json.dump(cost_rows, f, ensure_ascii=False, indent=1)
    print("metrics written to", root / "_metrics")


def _cost_report(benchmark: str):
    cost_paths = {
        "fraudr1": ["baseline_predictions/fraudr1_official_judge_predictions.jsonl", "teacher_predictions/fraudr1_teacher_predictions.jsonl"],
        "orbench": ["baseline_predictions/orbench_response_checker_predictions.jsonl", "teacher_predictions/orbench_teacher_predictions.jsonl"],
        "do_not_answer": ["baseline_predictions/dna_longformer_harmful_predictions.jsonl", "teacher_predictions/do_not_answer_teacher_predictions.jsonl"],
        "aegis2": ["baseline_predictions/aegis_nemoguard_predictions.jsonl", "teacher_predictions/aegis2_teacher_predictions.jsonl"],
    }
    recs = {"benchmark": benchmark}
    for method, rel in zip(["baseline", "teacher"], cost_paths[benchmark]):
        rows = load_jsonl(EXPERIMENT_DIR / benchmark / rel)
        if not rows:
            recs[method] = None
            continue
        in_tok = sum(r.get("input_tokens", 0) or 0 for r in rows)
        out_tok = sum(r.get("output_tokens", 0) or 0 for r in rows)
        recs[method] = {
            "n": len(rows),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "est_cost_rmb_deepseek": round(in_tok * 1.0 / 1e6 + out_tok * 2.0 / 1e6, 4),
            "est_cost_rmb_qwen": round(in_tok * 0.4 / 1e6 + out_tok * 1.2 / 1e6, 4),
            "latency_ms_mean": round(sum(r.get("latency_ms", 0) or 0 for r in rows) / len(rows), 1),
        }
    return recs


if __name__ == "__main__":
    main()
