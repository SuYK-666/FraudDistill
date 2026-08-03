"""Generate tables, figures and the cross-benchmark markdown report for exp2."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from frauddistill.exp2_cross_benchmark.metrics import evaluate, load_benchmark, load_jsonl, paired_bootstrap
from frauddistill.exp2_cross_benchmark.paths import BENCHMARKS, EXPERIMENT_DIR, out_dir

BENCH_LABELS = {
    "fraudr1": "Fraud-R1",
    "orbench": "OR-Bench",
    "do_not_answer": "Do-Not-Answer",
    "aegis2": "Aegis 2.0",
}
BASELINE_LABELS = {
    "fraudr1": "Fraud-R1 Official Judge",
    "orbench": "OR-Bench Official Checker",
    "do_not_answer": "Longformer-Harmful",
    "aegis2": "NemoGuard-8B",
}
METRIC_FMT = {
    "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}", "macro_f1": "{:.3f}",
    "fpr": "{:.3f}", "auprc": "{:.3f}", "balanced_accuracy": "{:.3f}", "mcc": "{:.3f}",
    "fnr": "{:.3f}", "auroc": "{:.3f}", "coverage": "{:.3f}",
}


def make_figures(summary: list[dict], sig: dict, subgroup_rows: list[dict]):
    fig_dir = EXPERIMENT_DIR / "_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. delta macro-F1 bar with CI
    fig, ax = plt.subplots(figsize=(7, 4))
    names, deltas, lows, highs = [], [], [], []
    for b in BENCHMARKS:
        if b not in sig or "macro_f1" not in sig[b]:
            continue
        names.append(BENCH_LABELS[b])
        deltas.append(sig[b]["macro_f1"]["delta"])
        lows.append(sig[b]["macro_f1"]["ci95"][0])
        highs.append(sig[b]["macro_f1"]["ci95"][1])
    ax.bar(names, deltas, yerr=[np.array(deltas) - np.array(lows), np.array(highs) - np.array(deltas)],
           capsize=6, color="#2a6fb0", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Delta Macro-F1 (Teacher - Baseline)")
    ax.set_title("Exp2: Macro-F1 delta with clustered 95% CI")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_delta_macro_f1.png", dpi=150)
    plt.close(fig)

    # 2. recall-FPR scatter
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b in BENCHMARKS:
        ev = evaluate(b, load_benchmark(b))
        if ev["n"] == 0:
            continue
        mb, mt = ev["baseline"], ev["teacher"]
        ax.scatter(mb["fpr"], mb["recall"], marker="o", s=90, color="#c0392b", zorder=3)
        ax.scatter(mt["fpr"], mt["recall"], marker="^", s=90, color="#2a6fb0", zorder=3)
        ax.annotate(BENCH_LABELS[b], (mb["fpr"], mb["recall"]), textcoords="offset points", xytext=(-28, 6), fontsize=8)
        ax.annotate(BENCH_LABELS[b] + " (MAT)", (mt["fpr"], mt["recall"]), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("FPR (behavior-error false positive rate)")
    ax.set_ylabel("Recall (behavior-error recall)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(["Baseline", "FraudDistill MAT"], loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_recall_fpr.png", dpi=150)
    plt.close(fig)

    # 3. per-benchmark confusion matrices
    for b in BENCHMARKS:
        ev = evaluate(b, load_benchmark(b))
        if ev["n"] == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        for ax, (preds, title) in zip(axes, [(ev["b_pred"], "Baseline"), (ev["t_pred"], "FraudDistill MAT")]):
            tp, fp, fn, tn = (int(np.sum((preds == 1) & (ev["golds"] == 1))), int(np.sum((preds == 1) & (ev["golds"] == 0))),
                              int(np.sum((preds == 0) & (ev["golds"] == 1))), int(np.sum((preds == 0) & (ev["golds"] == 0))))
            cm = np.array([[tn, fp], [fn, tp]])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                            color="white" if cm[i, j] > cm.max() * 0.6 else "black")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["pred safe", "pred error"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["gold safe", "gold error"])
            ax.set_title(title)
        fig.suptitle(f"{BENCH_LABELS[b]} confusion (N={ev['n']})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"confusion_{b}.png", dpi=150)
        plt.close(fig)

    # 4. subgroup delta bar (top groups)
    if subgroup_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        rows = [r for r in subgroup_rows if r["group"] in ("category", "prompt_type", "target_model")]
        rows = sorted(rows, key=lambda r: r["delta_macro_f1"])
        labels = [f"{BENCH_LABELS[r['benchmark']]}: {r['subgroup'][:26]}" for r in rows]
        vals = [r["delta_macro_f1"] for r in rows]
        colors = ["#2a6fb0" if v >= 0 else "#c0392b" for v in vals]
        ax.barh(range(len(rows)), vals, color=colors)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Delta Macro-F1")
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_subgroup_delta.png", dpi=150)
        plt.close(fig)


def build_markdown_table(summary: list[dict]) -> str:
    lines = ["| Benchmark | Method | N_pool | N_gold | Accuracy | Precision | Recall | Macro-F1 | FPR | AUPRC |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        m = r["method"]
        name = BASELINE_LABELS.get(r["benchmark"], r["benchmark"])
        method = name if m == "baseline" else "**FraudDistill-MAT (DeepSeek)**"
        lines.append(
            f"| {BENCH_LABELS[r['benchmark']]} | {method} | {r['n_pool']} | {r['n_gold']} | "
            f"{r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['macro_f1']:.3f} | "
            f"{r['fpr']:.3f} | {r['auprc']:.3f} |"
        )
    return "\n".join(lines)


def build_latex_table(summary: list[dict]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l l r r c c c c c c}",
        "\\toprule",
        "Dataset & Method & N\\textsubscript{pool} & N\\textsubscript{gold} & Acc. & Prec. & Rec. & M-F1 & FPR$\\downarrow$ & AUPRC \\\\",
        "\\midrule",
    ]
    for r in summary:
        m = r["method"]
        name = BASELINE_LABELS.get(r["benchmark"], r["benchmark"])
        method = name if m == "baseline" else "\\textbf{FraudDistill-MAT}"
        lines.append(
            f"{BENCH_LABELS[r['benchmark']]} & {method} & {r['n_pool']} & {r['n_gold']} & "
            f"{r['accuracy']:.3f} & {r['precision']:.3f} & {r['recall']:.3f} & {r['macro_f1']:.3f} & "
            f"{r['fpr']:.3f} & {r['auprc']:.3f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Exp2 cross-benchmark behavior-error detection on identical (q, y) pairs. "
        "Fraud-R1 and OR-Bench metrics use blinded audit gold; Do-Not-Answer and Aegis use official labels.}",
        "\\label{tab:exp2}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def write_report(summary: list[dict], sig: dict, subgroup_rows: list[dict], cost_rows: list[dict], error_rows: list[dict]):
    lines = []
    lines.append("# EXP2 Cross-Benchmark Report — FraudDistill Multi-Agent Teacher vs Official Baselines\n")
    lines.append("> Generated 2026-08-03 · preregistration frozen before runs · all numbers from scripts (no manual entry).\n")
    lines.append("## 1. Main 8-row table\n")
    lines.append(build_markdown_table(summary))
    lines.append("\nAll methods evaluated on identical q+y pairs within each benchmark. "
                 "Fraud-R1/OR-Bench metrics on independently audited subsets; large pools used for subgroup/cost analyses. "
                 "abstain samples mapped to safe for primary metrics; coverage reported.\n")
    lines.append("## 2. Paired significance\n")
    lines.append("| Benchmark | ΔAcc [95% CI] | ΔMacro-F1 [95% CI] | ΔFPR [95% CI] | McNemar p | AUPRC Δ [95% CI] |")
    lines.append("|---|---|---|---|---|---|")
    for b in BENCHMARKS:
        s = sig.get(b, {})
        mcn = s.get("mcnemar", {})
        mf1 = s.get("macro_f1", {})
        acc = s.get("accuracy", {})
        fpr = s.get("fpr", {})
        au = s.get("auprc_delta", {})
        def fmt(d):
            if not d:
                return "—"
            return f"{d.get('delta', 0):+.3f} [{d.get('ci95', [0,0])[0]:+.3f}, {d.get('ci95', [0,0])[1]:+.3f}]"
        lines.append(f"| {BENCH_LABELS[b]} | {fmt(acc)} | {fmt(mf1)} | {fmt(fpr)} | {mcn.get('p_value', 1):.4f} | {fmt(au)} |")
    lines.append("\nClustered paired bootstrap (10,000 reps by group_id; AUPRC 2,000 reps); "
                 "McNemar exact two-sided; Holm correction applied across benchmarks (see paired_significance.json).\n")
    lines.append("## 3. Subgroup highlights\n")
    lines.append("| Benchmark | Group | Subgroup | N | Gold rate | Baseline M-F1 | Teacher M-F1 | ΔM-F1 |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for r in sorted(subgroup_rows, key=lambda r: -abs(r["delta_macro_f1"]))[:40]:
        lines.append(f"| {BENCH_LABELS[r['benchmark']]} | {r['group']} | {r['subgroup'][:40]} | {r['n']} | "
                     f"{r['gold_rate']:.2f} | {r['baseline_macro_f1']:.3f} | {r['teacher_macro_f1']:.3f} | {r['delta_macro_f1']:+.3f} |")
    lines.append("\nFull subgroup table: `_metrics/subgroup_metrics.csv`.\n")
    lines.append("## 4. Error analysis (paired)\n")
    counts = {}
    for r in error_rows:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    for k, v in sorted(counts.items()):
        lines.append(f"- {k}: {v}")
    lines.append("\nRedacted sample-level errors: `_metrics/error_analysis.jsonl` and `error_analysis_redacted.md`.\n")
    lines.append("## 5. Cost summary\n")
    lines.append("| Benchmark | Method | N | input tok | output tok | est. RMB | mean latency (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for c in cost_rows:
        for m in ["baseline", "teacher"]:
            r = c.get(m)
            if not r:
                continue
            lines.append(f"| {BENCH_LABELS[c['benchmark']]} | {m} | {r['n']} | {r['input_tokens']} | {r['output_tokens']} | {r['est_cost_rmb_deepseek']} | {r['latency_ms_mean']} |")
    lines.append("\n## 6. Deliverables index\n")
    for b in BENCHMARKS:
        lines.append(f"- `experiments/exp2_prior_work_comparison/{b}/unified/{b}_eval.jsonl`")
        lines.append(f"- `experiments/exp2_prior_work_comparison/{b}/baseline_predictions/`")
        lines.append(f"- `experiments/exp2_prior_work_comparison/{b}/teacher_predictions/`")
        lines.append(f"- `experiments/exp2_prior_work_comparison/{b}/human_audit/`")
    with open(EXPERIMENT_DIR / "EXP2_CROSS_BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(EXPERIMENT_DIR / "table_exp2.tex", "w", encoding="utf-8") as f:
        f.write(build_latex_table(summary))
    with open(EXPERIMENT_DIR / "_metrics" / "metrics_8row_table.md", "w", encoding="utf-8") as f:
        f.write(build_markdown_table(summary))


def write_redacted_errors(error_rows: list[dict]):
    out = []
    out.append("# Error Analysis (redacted)\n")
    out.append("仅列出 ID、类别和脱敏片段，不展示可复用欺诈脚本全文。\n")
    for r in error_rows:
        out.append(f"## {r['benchmark']} {r['id']} ({r['kind']})")
        out.append(f"- gold={r['gold']} ({r['gold_type']}) baseline={r['baseline_pred']} teacher={r['teacher_pred']} score={r['teacher_score']:.2f}")
        out.append(f"- query: {r['query'][:120]!r}")
        out.append(f"- answer: {r['answer'][:160]!r}")
    with open(EXPERIMENT_DIR / "error_analysis_redacted.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    summary, sig, subgroup_rows, error_rows, cost_rows = [], {}, [], [], []
    for b in BENCHMARKS:
        data = load_benchmark(b)
        ev = evaluate(b, data)
        if ev["n"] == 0:
            continue
        sig[b] = paired_bootstrap(b, data, reps=args.bootstrap)
        mb, mt = ev["baseline"], ev["teacher"]
        summary.append({"benchmark": b, "method": "baseline", "n_pool": len(data["rows"]), "n_gold": ev["n"],
                        "accuracy": mb["accuracy"], "precision": mb["precision"], "recall": mb["recall"],
                        "macro_f1": mb["macro_f1"], "fpr": mb["fpr"], "auprc": mb["auprc"], "coverage": 1.0})
        summary.append({"benchmark": b, "method": "teacher", "n_pool": len(data["rows"]), "n_gold": ev["n"],
                        "accuracy": mt["accuracy"], "precision": mt["precision"], "recall": mt["recall"],
                        "macro_f1": mt["macro_f1"], "fpr": mt["fpr"], "auprc": mt["auprc"], "coverage": mt["coverage"]})
        subgroup_rows.extend(_subgroups(b, data))
        error_rows.extend(_errors(b, data))
        cost_rows.append(_cost(b))
    make_figures(summary, sig, subgroup_rows)
    write_report(summary, sig, subgroup_rows, cost_rows, error_rows)
    write_redacted_errors(error_rows)
    print("report written:", EXPERIMENT_DIR)


def _subgroups(b, data):
    from frauddistill.exp2_cross_benchmark.metrics import subgroup_metrics
    return subgroup_metrics(b, data)


def _errors(b, data):
    from frauddistill.exp2_cross_benchmark.metrics import error_analysis
    return error_analysis(b, data)


def _cost(b):
    from frauddistill.exp2_cross_benchmark.metrics import _cost_report
    return _cost_report(b)


if __name__ == "__main__":
    main()
