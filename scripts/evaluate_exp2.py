# -*- coding: utf-8 -*-
"""Exp2 offline evaluation orchestrator (guide 18-19).

Computes canonical metrics, the 8-row main table, mechanism table, paired
significance (clustered bootstrap + exact McNemar + Holm), subgroup tables,
redacted error analysis and the native-task appendix. Zero API.

Usage: python scripts/evaluate_exp2.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_cross_benchmark.metrics import (  # noqa: E402
    BENCHMARKS,
    EXPERIMENT_DIR,
    METRICS_DIR,
    _mf1,
    binary_metrics,
    evaluate_benchmark,
    load_all,
    subgroup_metrics,
)
from frauddistill.exp2_cross_benchmark.paths import out_dir  # noqa: E402

METHOD_NAMES = {
    "fraudr1": ("Fraud-R1 Official Judge (GPTCheck protocol)", "FraudDistill Evidence MAT (T6)"),
    "orbench": ("OR-Bench Official Response Checker", "FraudDistill Evidence MAT (T6)"),
    "do_not_answer": ("Longformer-Harmful", "FraudDistill Evidence MAT (T6)"),
    "aegis2": ("NemoGuard-8B", "FraudDistill Evidence MAT (T6)"),
}


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, max((m - rank) * pvals[idx], prev))
        adjusted[idx] = val
        prev = val
    return adjusted


def main() -> None:
    data = load_all()
    results = {b: evaluate_benchmark(b, data[b]) for b in BENCHMARKS}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- paired significance with Holm over the 4 primary comparisons ----
    pvals = []
    for b in BENCHMARKS:
        pvals.append(results[b]["mcnemar"]["p"])
    adj = holm_adjust(pvals)
    sig = {}
    for i, b in enumerate(BENCHMARKS):
        res = results[b]
        sig[b] = {
            "mcnemar": res["mcnemar"],
            "mcnemar_p_holm": adj[i],
            "bootstrap_delta_macro_f1": res["bootstrap"],
            "matched": res["matched"],
            "n": res["n"],
            "gold_positive_rate": res["gold_positive_rate"],
        }
    (METRICS_DIR / "paired_significance.json").write_text(json.dumps(sig, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- main 8-row table ----
    rows = []
    for b in BENCHMARKS:
        res = results[b]
        for method, m in (("baseline", res["baseline"]), ("teacher", res["teacher"])):
            rows.append({
                "benchmark": b,
                "method": METHOD_NAMES[b][0 if method == "baseline" else 1],
                "method_key": method,
                "N": res["n"],
                "N+": int(round(res["gold_positive_rate"] * res["n"])),
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "unsafe_recall": round(m["unsafe_recall"], 4),
                "unsafe_f1": round(m["unsafe_f1"], 4),
                "safe_f1": round(m["safe_f1"], 4),
                "true_macro_f1": round(m["true_macro_f1"], 4),
                "fpr": round(m["fpr"], 4),
                "auprc": round(m["auprc"], 4) if m.get("auprc") is not None else "-",
                "mcc": round(m["mcc"], 4),
            })
    import csv
    with (METRICS_DIR / "main_8row.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (METRICS_DIR / "main_8row.md").open("w", encoding="utf-8") as f:
        f.write("| Benchmark | Method | N | N+ | Accuracy | Precision | Unsafe Recall | Unsafe-F1 | Safe-F1 | True Macro-F1 | FPR | AUPRC | MCC |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write("| {} | {} | {} | {} | {} | {} | {} | {} | {} | **{}** | {} | {} | {} |\n".format(
                r["benchmark"], r["method"], r["N"], r["N+"], r["accuracy"], r["precision"],
                r["unsafe_recall"], r["unsafe_f1"], r["safe_f1"], r["true_macro_f1"], r["fpr"], r["auprc"], r["mcc"]))

    # ---- mechanism table ----
    mech_rows = []
    for b in BENCHMARKS:
        mm = results[b]["mechanism"]
        mech_rows.append({
            "benchmark": b,
            "direct_fraud_recall": mm.get("direct_fraud_recall"), "direct_n": mm.get("direct_fraud_n"),
            "trust_facilitation_recall": mm.get("trust_facilitation_recall"), "trust_n": mm.get("trust_facilitation_n"),
            "partial_leakage_recall": mm.get("partial_leakage_recall"), "leakage_n": mm.get("partial_leakage_n"),
            "clean_refusal_fpr": mm.get("clean_refusal_fpr"), "clean_refusal_n": mm.get("clean_refusal_n"),
            "hard_safe_fpr": mm.get("hard_safe_fpr"), "hard_safe_n": mm.get("hard_safe_n"),
            "within_prompt_pair_acc": mm.get("within_prompt_pair_acc_binary"),
            "within_prompt_pairs": mm.get("within_prompt_pairs"),
        })
    with (METRICS_DIR / "mechanism_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mech_rows[0].keys()))
        w.writeheader()
        w.writerows(mech_rows)

    # ---- subgroup ----
    sub = subgroup_metrics(results)
    with (METRICS_DIR / "subgroup_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sub[0].keys()))
        w.writeheader()
        w.writerows(sub)

    # ---- redacted error analysis ----
    with (METRICS_DIR / "error_analysis.jsonl").open("w", encoding="utf-8") as f:
        for b in BENCHMARKS:
            for r in results[b]["records"]:
                if r["b_pred"] != r["t_pred"]:
                    f.write(json.dumps({
                        "id": r["id"], "benchmark": b, "group_id": r["group_id"],
                        "gold": r["gold"], "gold_type": r["gold_type"],
                        "baseline_pred": r["b_pred"], "teacher_pred": r["t_pred"],
                        "teacher_score": r["t_score"], "teacher_type": r["t_type"],
                        "language": r["language"], "category": r["category"],
                        "prompt_type": r["prompt_type"], "target_model": r["target_model"],
                        "error_direction": "baseline_wrong_teacher_correct" if (r["b_pred"] != r["gold"] and r["t_pred"] == r["gold"]) else
                                          "baseline_correct_teacher_wrong" if (r["b_pred"] == r["gold"] and r["t_pred"] != r["gold"]) else "both_wrong",
                    }, ensure_ascii=False) + "\n")

    # ---- native-task appendix (guide 4.2) ----
    native = native_appendix(results)
    (METRICS_DIR / "native_appendix.json").write_text(json.dumps(native, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- canonical metrics (records attached) ----
    canonical = {}
    for b, res in results.items():
        canonical[b] = {k: v for k, v in res.items() if k != "records"}
        canonical[b]["records"] = res["records"]
    (METRICS_DIR / "canonical_metrics.json").write_text(json.dumps(canonical, ensure_ascii=False, indent=1), encoding="utf-8")

    print_summary(results, sig, rows)


def native_appendix(results: dict) -> dict:
    """Native-task track on the same manifest samples (official labels) plus
    full-pool baseline reference for DNA/Aegis (guide 4.2)."""
    import csv
    out = {}
    manifest_rows = {}
    from frauddistill.exp2_cross_benchmark.paths import MANIFEST_DIR
    for r in [json.loads(l) for l in (MANIFEST_DIR / "test_manifest.jsonl").open(encoding="utf-8")]:
        manifest_rows[r["sample_id"]] = r

    baseline_files = {
        "fraudr1": "fraudr1/baseline_predictions/fraudr1_official_judge_predictions.jsonl",
        "orbench": "orbench/baseline_predictions/orbench_response_checker_predictions.jsonl",
        "do_not_answer": "do_not_answer/baseline_predictions/dna_longformer_harmful_predictions.jsonl",
        "aegis2": "aegis2/baseline_predictions/aegis_nemoguard_predictions.jsonl",
    }
    from frauddistill.exp2_cross_benchmark.paths import TEACHER_T6_DIR
    for b in BENCHMARKS:
        bl = {r["id"]: r for r in [json.loads(l) for l in (EXPERIMENT_DIR / baseline_files[b]).open(encoding="utf-8")]}
        teacher = {r["id"]: r for r in [json.loads(l) for l in (TEACHER_T6_DIR / f"{b}_t6_predictions.jsonl").open(encoding="utf-8")]}
        recs = []
        for r in results[b]["records"]:
            row = manifest_rows[r["id"]]
            gold_native = row.get("official_gold_binary")
            if gold_native is None:
                continue
            recs.append({
                "id": r["id"], "group_id": r["group_id"], "native_gold": int(gold_native),
                "native_gold_type": row.get("official_gold_type"),
                "baseline_pred": r["b_pred"], "teacher_pred": r["t_pred"], "teacher_score": r["t_score"],
            })
        y = np.array([r["native_gold"] for r in recs])
        bp = np.array([r["baseline_pred"] for r in recs])
        tp = np.array([r["teacher_pred"] for r in recs])
        entry = {"n": len(recs)}
        if len(y):
            entry["baseline"] = {k: round(v, 4) for k, v in binary_metrics(y, bp).items() if k != "n"}
            entry["teacher"] = {k: round(v, 4) for k, v in binary_metrics(y, tp).items() if k != "n"}
        out[b] = entry

    # full-pool baseline reference on official labels (no new API)
    full = {}
    for b in ["do_not_answer", "aegis2"]:
        rows = [json.loads(l) for l in (EXPERIMENT_DIR / ("do_not_answer/unified/do_not_answer_eval.jsonl" if b == "do_not_answer" else "aegis2/unified/aegis2_eval.jsonl")).open(encoding="utf-8")]
        bl = {r["id"]: r for r in [json.loads(l) for l in (EXPERIMENT_DIR / baseline_files[b]).open(encoding="utf-8")]}
        recs = []
        for r in rows:
            g = r.get("gold_binary")
            p = bl.get(r["id"])
            if g is None or p is None:
                continue
            if b == "aegis2" and not (r.get("answer") or "").strip():
                continue
            recs.append((int(g), int(p.get("prediction_binary", 0) or 0)))
        if recs:
            y = np.array([x[0] for x in recs]); p = np.array([x[1] for x in recs])
            full[b] = {"n": len(recs), **{k: round(v, 4) for k, v in binary_metrics(y, p).items() if k != "n"}}
    out["full_pool_baseline_official_labels"] = full
    return out


def print_summary(results, sig, rows) -> None:
    print("\n===== MAIN 8-ROW TABLE =====")
    for r in rows:
        print(f"{r['benchmark']:>15} | {r['method_key']:>8} | N={r['N']:>4} | MF1={r['true_macro_f1']:.4f} | R={r['unsafe_recall']:.3f} | FPR={r['fpr']:.4f} | AUPRC={r['auprc']}")
    print("\n===== PAIRED SIGNIFICANCE =====")
    for b in BENCHMARKS:
        s = sig[b]
        boot = s["bootstrap_delta_macro_f1"]
        print(f"{b:>15} | dMF1={boot['observed_delta']:+.4f} [{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}] | McNemar p={s['mcnemar']['p']:.5f} (holm {s['mcnemar_p_holm']:.5f})")
    print("\noutputs ->", METRICS_DIR)


if __name__ == "__main__":
    main()
