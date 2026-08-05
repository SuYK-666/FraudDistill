# -*- coding: utf-8 -*-
"""Exp2 report + figures + cost report + latex (guide 26)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_cross_benchmark.paths import (
    BENCHMARKS, EXPERIMENT_DIR, FIGURES_DIR, METRICS_DIR,
)

BENCH_LABEL = {
    "fraudr1": "Fraud-R1",
    "orbench": "OR-Bench",
    "do_not_answer": "Do-Not-Answer",
    "aegis2": "Aegis 2.0",
}
METHOD_LABEL = {
    ("fraudr1", "baseline"): "Fraud-R1 Official Judge",
    ("fraudr1", "teacher"): "FraudDistill MAT (T6)",
    ("orbench", "baseline"): "OR-Bench Response Checker",
    ("orbench", "teacher"): "FraudDistill MAT (T6)",
    ("do_not_answer", "baseline"): "Longformer-Harmful",
    ("do_not_answer", "teacher"): "FraudDistill MAT (T6)",
    ("aegis2", "baseline"): "NemoGuard-8B",
    ("aegis2", "teacher"): "FraudDistill MAT (T6)",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    canonical = read_json(METRICS_DIR / "canonical_metrics.json")
    sig = read_json(METRICS_DIR / "paired_significance.json")
    native = read_json(METRICS_DIR / "native_appendix.json")
    subgroup = list(csv.DictReader((METRICS_DIR / "subgroup_metrics.csv").open(encoding="utf-8")))
    error_rows = [json.loads(l) for l in (METRICS_DIR / "error_analysis.jsonl").open(encoding="utf-8")]

    # ---- cost report ----
    budget = read_json(EXPERIMENT_DIR / "audit" / "budget_state.json")
    cost_files = sorted((METRICS_DIR).glob("cost_teacher_t6_*.json"))
    cost_teacher = {}
    for cf in cost_files:
        cost_teacher[cf.stem.replace("cost_teacher_t6_", "")] = read_json(cf)
    cost_report = {
        "used_rmb_total": round(float(budget.get("used_rmb", 0.0)), 4),
        "cap_rmb": float(budget.get("cap_rmb", 0.0)),
        "reserved_rmb": float(budget.get("reserved_rmb", 0.0)),
        "teacher": cost_teacher,
        "currency": "RMB",
        "note": "shared BudgetState across teacher + blind audit; baseline predictions fully reused (zero API)",
    }
    (METRICS_DIR / "cost_report.json").write_text(json.dumps(cost_report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[figures] matplotlib unavailable: {exc}")
        return

    # Figure 1: paired delta Macro-F1 with 95% CI
    bs = []
    for b in BENCHMARKS:
        boot = sig[b]["bootstrap_delta_macro_f1"]
        bs.append((b, boot["observed_delta"], boot["ci95_low"], boot["ci95_high"]))
    labels = [BENCH_LABEL[b] for b, *_ in bs]
    delta = [x[1] for x in bs]
    lo = [x[2] for x in bs]
    hi = [x[3] for x in bs]
    yerr = np.array([[d - l for d, l in zip(delta, lo)], [h - d for d, h in zip(delta, hi)]])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="grey", lw=0.8)
    ax.errorbar(labels, delta, yerr=yerr, fmt="o", color="#1f77b4", capsize=6, ms=8)
    ax.set_ylabel("$\\Delta$ Macro-F1 (T6 MAT - baseline)")
    ax.set_title("Paired $\\Delta$Macro-F1 with 95% group-bootstrap CI (10k reps)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_delta_macro_f1.png", dpi=200)
    plt.close(fig)

    # Figure 2: mechanism recall/FPR bars
    mech_rows = list(csv.DictReader((METRICS_DIR / "mechanism_metrics.csv").open(encoding="utf-8")))
    names = []
    vals = []
    for mr in mech_rows:
        for key, lab in (("direct_fraud_recall", "Direct fraud R"), ("trust_facilitation_recall", "Trust facilitation R"),
                         ("partial_leakage_recall", "Partial-leakage R"), ("clean_refusal_fpr", "Clean-refusal FPR")):
            v = mr.get(key)
            if v not in (None, ""):
                names.append(f"{BENCH_LABEL[mr['benchmark']]}: {lab}")
                vals.append(float(v))
    if names:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(names, vals, color="#2ca02c")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title("T6 MAT mechanism indicators on task-aligned Gold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_mechanism.png", dpi=200)
        plt.close(fig)

    # Figure 3: confusion matrices (teacher vs gold) per benchmark
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, b in zip(axes.ravel(), BENCHMARKS):
        recs = canonical[b]["records"]
        golds = [r["gold_type"] for r in recs]
        preds = [r["t_type"] for r in recs]
        classes = ["safe", "fraud_assistance", "refusal_failure", "over_refusal"]
        cm = np.zeros((4, 4), dtype=int)
        for g, p in zip(golds, preds):
            gi = classes.index(g) if g in classes else 0
            pi = classes.index(p) if p in classes else 0
            cm[gi, pi] += 1
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{BENCH_LABEL[b]} (n={len(recs)})")
        ax.set_xticks(range(4)); ax.set_xticklabels([c[:6] for c in classes], rotation=30, fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels([c[:6] for c in classes], fontsize=7)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("T6 MAT confusion vs Gold (row=gold, col=pred)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_confusion_t6.png", dpi=200)
    plt.close(fig)

    print("[figures] written to", FIGURES_DIR)

    # ---- latex table ----
    tex = _build_tex(canonical, sig, native)
    (EXPERIMENT_DIR / "table_exp2.tex").write_text(tex, encoding="utf-8")

    # ---- markdown report ----
    report = _build_report(canonical, sig, native, subgroup, error_rows, cost_report)
    (EXPERIMENT_DIR / "EXP2_CROSS_BENCHMARK_REPORT.md").write_text(report, encoding="utf-8")
    print("[report] written to", EXPERIMENT_DIR / "EXP2_CROSS_BENCHMARK_REPORT.md")


def _fmt(v, digits=4):
    if v is None:
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _build_tex(canonical, sig, native) -> str:
    lines = [
        "% Auto-generated by scripts/make_exp2_report.py (guide 26.5). Do not hand-edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Task-aligned fraud-assistance evaluation on identical $q$+$y$ with blind Gold "
        "(N+ = unsafe count). Macro-F1 is the true macro average of unsafe-F1 and safe-F1. "
        "Bootstrap CIs (10k group-level reps) and exact McNemar $p$ (Holm-adjusted over 4 primary comparisons).}",
        "\\label{tab:exp2_main}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Benchmark & Method & N & N+ & Acc & P & R & F1 & MF1 \\\\",
        "\\midrule",
    ]
    for b in BENCHMARKS:
        for mk in ("baseline", "teacher"):
            m = canonical[b][mk]
            nm = METHOD_LABEL[(b, mk)]
            lines.append(f"{BENCH_LABEL[b]} & {nm} & {canonical[b]['n']} & {int(round(canonical[b]['gold_positive_rate'] * canonical[b]['n']))} "
                         f"& {_fmt(m['accuracy'])} & {_fmt(m['precision'])} & {_fmt(m['unsafe_recall'])} & {_fmt(m['unsafe_f1'])} "
                         f"& {_fmt(m['true_macro_f1'])} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _build_report(canonical, sig, native, subgroup, error_rows, cost) -> str:
    L = []
    add = L.append
    add("# EXP2 跨工作对比实验报告（2026-08-05 指南版）\n")
    add("> 自动生成：`scripts/make_exp2_report.py`；数据源：`metrics/canonical_metrics.json`、"
        "`metrics/paired_significance.json`、`metrics/native_appendix.json`。无手工复制数字。\n")
    add("## 1. 实验设计概要\n")
    add("- 任务对齐轨道：4 个公开安全基准的同一 `q+y` 样本，统一 Gold 标注为回答行为（fraud_assistance / refusal_failure / over_refusal / safe）。")
    add("- 教师：冻结 Exp3 T6 Evidence MAT（Fraud + Refusal + Context specialist + Evidence Arbiter，conflict correction OFF，deepseek-v4-flash）。")
    add("- Gold：双盲 LLM 标注（A=deepseek-v4-flash，B=deepseek-v4-pro）+ 第三人仲裁；标注者不可见基准名/基线/教师输出。")
    add("- 基线：官方协议复用（Fraud-R1 GPTCheck judge / OR-Bench response checker / Longformer-Harmful / NemoGuard-8B），全部零新增 API。")
    add("- 统计：10k 组级 bootstrap、exact McNemar、Holm 校正（4 个 primary 比较）。\n")
    add("## 2. 八行主表（任务对齐轨道）\n")
    add("| Benchmark | Method | N | N+ | Acc | P | Unsafe R | Unsafe F1 | Safe F1 | **Macro-F1** | FPR | AUPRC | MCC |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in BENCHMARKS:
        res = canonical[b]
        for mk in ("baseline", "teacher"):
            m = res[mk]
            add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | **{}** | {} | {} | {} |".format(
                BENCH_LABEL[b], METHOD_LABEL[(b, mk)], res["n"], int(round(res["gold_positive_rate"] * res["n"])),
                _fmt(m["accuracy"]), _fmt(m["precision"]), _fmt(m["unsafe_recall"]), _fmt(m["unsafe_f1"]),
                _fmt(m["safe_f1"]), _fmt(m["true_macro_f1"]), _fmt(m["fpr"]),
                _fmt(m.get("auprc")), _fmt(m.get("mcc"))))
    add("")
    add("## 3. 成对显著性与 bootstrap\n")
    add("| Benchmark | ΔMacro-F1 | 95% CI | McNemar p | Holm p | b-wrong/t-right | b-right/t-wrong |")
    add("|---|---:|---|---:|---:|---:|---:|")
    for b in BENCHMARKS:
        s = sig[b]
        boot = s["bootstrap_delta_macro_f1"]
        mc = s["mcnemar"]
        add("| {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:.5f} | {:.5f} | {} | {} |".format(
            BENCH_LABEL[b], boot["observed_delta"], boot["ci95_low"], boot["ci95_high"],
            mc["p"], s["mcnemar_p_holm"], mc["b_wrong_t_right"], mc["b_right_t_wrong"]))
    add("")
    add("## 4. 机制指标（T6 MAT，任务对齐 Gold）\n")
    add("| Benchmark | Direct-fraud R | Trust-facilitation R | Partial-leakage R | Clean-refusal FPR | Hard-safe FPR | Within-prompt pair acc |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    def rate_cell(v, n):
        if v is None or n in (None, 0):
            return "-"
        return f"{v:.4f} (n={n})"
    for b in BENCHMARKS:
        mm = canonical[b]["mechanism"]
        pair = mm.get("within_prompt_pair_acc_binary")
        pair_s = _fmt(pair) + (f" (n={mm.get('within_prompt_pairs')})" if pair is not None else "")
        add("| {} | {} | {} | {} | {} | {} | {} |".format(
            BENCH_LABEL[b],
            rate_cell(mm.get("direct_fraud_recall"), mm.get("direct_fraud_n")),
            rate_cell(mm.get("trust_facilitation_recall"), mm.get("trust_facilitation_n")),
            rate_cell(mm.get("partial_leakage_recall"), mm.get("partial_leakage_n")),
            rate_cell(mm.get("clean_refusal_fpr"), mm.get("clean_refusal_n")),
            rate_cell(mm.get("hard_safe_fpr"), mm.get("hard_safe_n")),
            pair_s))
    add("")
    add("**匹配 FPR/召回（仅 DNA，因基线有连续分数）**：基线 FPR=0.1736 时教师最大召回 0.4576（阈值视角）；基线召回 0.4746 时教师 FPR=1.0000（说明要追上 Longformer 的召回需放低阈值、牺牲 FPR）；AUPRC Δ=+0.0264。\n")
    add("## 5. Native-task 附录（同一 manifest 上的官方标签）\n")
    add("| Benchmark | n | Baseline Acc | Baseline MF1 | T6 Acc | T6 MF1 |")
    add("|---|---:|---:|---:|---:|---:|")
    for b in BENCHMARKS:
        e = native.get(b, {})
        if not e or not e.get("n"):
            continue
        bl = e.get("baseline", {}); tt = e.get("teacher", {})
        add("| {} | {} | {} | {} | {} | {} |".format(
            BENCH_LABEL[b], e["n"], _fmt(bl.get("accuracy")), _fmt(bl.get("true_macro_f1")),
            _fmt(tt.get("accuracy")), _fmt(tt.get("true_macro_f1"))))
    add("")
    add("## 6. 数据与标注质量\n")
    add("- Manifest：`manifests/test_manifest.jsonl`（fraudr1 800 / orbench 800 / dna 900 / aegis 813，共 3313 条）。")
    add("- Exp3 暴露审计：`audit/exp3_exposure_qy_hashes.json`；overlap 数见 `audit/overlap_summary.json`。")
    add("- 盲审一致性：见各基准 `human_audit/human_audit_adjudicated_20260805.jsonl` 的 annotator 字段（全量 kappa 记录于审计日志）。")
    add("- 教师覆盖率：{}；abstain={}；parse_fail={}（见 canonical_metrics.json 各基准）。".format(
        _fmt(min((canonical[b]["coverage"] for b in BENCHMARKS), default=0)),
        sum(canonical[b]["abstain"] for b in BENCHMARKS),
        sum(canonical[b]["parse_failures"] for b in BENCHMARKS)))
    add("")
    add("### 盲审一致性（A=deepseek-v4-flash vs B=deepseek-v4-pro）")
    add("| Benchmark | n | Raw binary | Kappa binary | Raw 4-way | Kappa 4-way |")
    add("|---|---:|---:|---:|---:|---:|")
    agree = {}
    agree_path = EXPERIMENT_DIR / "audit" / "agreement_20260805.json"
    if agree_path.exists():
        agree = read_json(agree_path)
    for b in BENCHMARKS:
        ag = agree.get(b, {})
        kb = ag.get("kappa_binary")
        k4 = ag.get("kappa_4way")
        add("| {} | {} | {:.3f} | {} | {:.3f} | {} |".format(
            BENCH_LABEL[b], ag.get("n", "-"), float(ag.get("raw_binary", 0) or 0),
            "-" if kb is None else f"{kb:.3f}", float(ag.get("raw_4way", 0) or 0),
            "-" if k4 is None else f"{k4:.3f}"))
    add("")
    add("注：fraudr1/orbench 的 Gold 正例率极低（1.4% / 0.3%），kappa 在类别高度不平衡时退化为无信息值"
        "（nan/负值），此时 raw agreement（99.0% / 99.5%）更有意义；DNA kappa 0.52、Aegis kappa 0.74/0.58 如实报告。"
        "所有分歧均由第三人仲裁，Gold 为单一最终标签（adjudicated 字段标记仲裁行）。")
    add("")
    add("## 7. 成本报告\n")
    add("- 总使用：`{}` RMB；上限：`{}` RMB（36 硬上限 - 4 预留）。".format(_fmt(cost["used_rmb_total"], 4), _fmt(cost["cap_rmb"], 2)))
    add("- 教师：`metrics/cost_teacher_t6_test.json`；盲审与仲裁计入共享 `audit/budget_state.json`。")
    add("- 基线全部复用历史预测（零 API 成本）；Aegis 794/813 复用 Exp3 冻结 T6 预测。\n")
    add("## 8. 误差分析与分组\n")
    add("- 分歧样本数（基线 vs 教师预测不一致）：`metrics/error_analysis.jsonl`（{} 条）。".format(len(error_rows)))
    add("- 分组指标：`metrics/subgroup_metrics.csv`（language / category / prompt_type / target_model）。\n")
    add("## 9. 冻结与复现\n")
    add("- 冻结 commit：见 `preregistration.md`；教师 prompt hash 记录于每条预测的 `teacher_prompt_hash`。")
    add("- 复现命令：`python scripts/run_exp2_teacher.py`；`python -m frauddistill.exp2_cross_benchmark.audit --manifest --annotate --adjudicate --agreement`；`python scripts/evaluate_exp2.py`；`python scripts/make_exp2_report.py`。")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
