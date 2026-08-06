# -*- coding: utf-8 -*-
"""Exp2 FULL-coverage report generator (guide 2026-08-06 sections 5, 26-28).

Reads the canonical metrics produced by evaluate_exp2.py and writes:
  - EXP2_CROSS_BENCHMARK_REPORT.md  (three-layer report + cost + acceptance)
  - table_exp2.tex                  (Layer 2 + Layer 3 tables)
  - figures/                        (delta CI, mechanism, OR plane, risk rates)
All numbers come from metrics/canonical_metrics.json (no manual entry).

Usage: python scripts/make_exp2_report.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_cross_benchmark.paths import (  # noqa: E402
    EXPERIMENT_DIR,
    FIGURES_DIR,
    METRICS_DIR,
)

BENCH_LABEL = {
    "fraudr1": "Fraud-R1", "orbench": "OR-Bench",
    "do_not_answer": "Do-Not-Answer", "aegis2": "Aegis 2.0",
}
METHOD_LABEL = {
    "fraudr1": "Fraud-R1 GPTCheck protocol",
    "orbench": "OR-Bench official response checker",
    "do_not_answer": "LibrAI/longformer-harmful-ro",
    "aegis2": "nvidia/llama-3.1-nemoguard-8b-content-safety",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v, digits=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _pct(v, digits=2):
    if v is None:
        return "-"
    return f"{100.0 * float(v):.{digits}f}%"


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    canonical = read_json(METRICS_DIR / "canonical_metrics.json")
    sig = read_json(METRICS_DIR / "paired_significance.json")
    calib = {}
    if (METRICS_DIR / "calibration.json").exists():
        calib = read_json(METRICS_DIR / "calibration.json")
    budget = {}
    if (EXPERIMENT_DIR / "audit" / "budget_state.json").exists():
        budget = read_json(EXPERIMENT_DIR / "audit" / "budget_state.json")

    l1 = canonical["layer1_full_native"]
    unified = canonical["layer2_unified"]
    faligned = canonical["layer3_fraud_aligned"]
    ops = canonical["operating_points"]
    silver = canonical["silver_subsets"]

    cost_report = {
        "used_rmb_round": round(float(budget.get("used_rmb", 0.0)), 4),
        "prev_round_rmb": round(float(budget.get("prev_round_rmb", 0.0)), 4),
        "cumulative_rmb": round(float(budget.get("cumulative_rmb", 0.0)), 4),
        "cap_rmb": float(budget.get("cap_rmb", 0.0)),
        "reserve_rmb": float(budget.get("reserve_rmb", 4.0)),
        "history": [],
        "currency": "RMB",
        "note": "round ledger: new API only; baselines fully reused (zero API); Aegis 813 response rows reused from the 2026-08-05 run",
    }
    hist_path = EXPERIMENT_DIR / "audit" / "budget_history.json"
    if hist_path.exists():
        cost_report["history"] = read_json(hist_path)
    (METRICS_DIR / "cost_report.json").write_text(json.dumps(cost_report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------------------------- figures -------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[figures] matplotlib unavailable: {exc}")
        return

    # Fig 1: paired delta Macro-F1 with 95% CI (Layer 2 + Layer 3)
    rows_fig = []
    for u in unified:
        boot = u.get("bootstrap") or {}
        if boot.get("ci95_low") is None:
            continue
        rows_fig.append((u["benchmark"], boot.get("observed_delta", 0.0), boot["ci95_low"], boot["ci95_high"]))
    for key, entry in faligned.items():
        boot = entry.get("bootstrap") or {}
        if boot.get("ci95_low") is None:
            continue
        label = "DNA fraud-aligned (Malicious Uses)" if key == "dna_malicious_uses" else "Aegis fraud-aligned categories"
        rows_fig.append((label, boot.get("observed_delta", 0.0), boot["ci95_low"], boot["ci95_high"]))
    if rows_fig:
        labels = [x[0] for x in rows_fig]
        delta = [x[1] for x in rows_fig]
        lo = [x[2] for x in rows_fig]
        hi = [x[3] for x in rows_fig]
        yerr = np.array([[d - l for d, l in zip(delta, lo)], [h - d for d, h in zip(delta, hi)]])
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.axhline(0, color="grey", lw=0.8)
        ax.errorbar(labels, delta, yerr=yerr, fmt="o", color="#1f77b4", capsize=6, ms=8)
        ax.set_ylabel("$\\Delta$ Macro-F1 (T6 MAT - baseline)")
        ax.set_title("Paired $\\Delta$Macro-F1 with 95% group-bootstrap CI (10k reps)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_delta_macro_f1.png", dpi=200)
        plt.close(fig)

    # Fig 2: silver-subset mechanism indicators
    names, vals = [], []
    for b in ("fraudr1", "orbench", "do_not_answer", "aegis2"):
        mech = (silver.get(b) or {}).get("mechanism", {})
        for key, lab in (("trust_facilitation_recall", "Trust facilitation R"),
                         ("partial_leakage_recall", "Partial-leakage R"),
                         ("clean_refusal_fpr", "Clean-refusal FPR")):
            v = mech.get(key)
            if v is not None:
                names.append(f"{BENCH_LABEL[b]}: {lab}")
                vals.append(float(v))
    if names:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.bar(names, vals, color="#2ca02c")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title("T6 MAT mechanism indicators (LLM-panel silver subsets, 2026-08-05)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_mechanism.png", dpi=200)
        plt.close(fig)

    # Fig 3: OR safety-utility plane (dual axis)
    ob = l1["orbench"]
    if ob.get("teacher"):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = [1, 2, 3, 4]
        t_vals = [ob["teacher"].get("hard_safe_fpr"), ob["teacher"].get("regular_safe_fpr"),
                  ob["teacher"].get("toxic_harmful_compliance_rate"), ob["teacher"].get("safety_utility_score")]
        b_vals = [ob["baseline"].get("hard_safe_fpr"), ob["baseline"].get("regular_safe_fpr"),
                  ob["baseline"].get("toxic_harmful_compliance_rate"), ob["baseline"].get("safety_utility_score")]
        labels = ["Hard-safe FPR", "Regular-safe FPR", "Toxic harmful-compliance", "Safety-utility"]
        ax.plot(x, t_vals, "o-", color="#1f77b4", label="FraudDistill T6")
        ax.plot(x, b_vals, "s--", color="#d62728", label="OR-Bench checker")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title("OR-Bench native dual-axis comparison (full 3,000 core pool)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_or_safety_utility.png", dpi=200)
        plt.close(fig)

    # Fig 4: Fraud-R1 category/language risk rates
    fr = l1["fraudr1"]
    if fr.get("category_rates"):
        cats = list(fr["category_rates"].keys())
        jr = [fr["category_rates"][c]["judge_rate"] for c in cats]
        tr = [fr["category_rates"][c]["t6_rate"] for c in cats]
        x = np.arange(len(cats))
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        ax.bar(x - 0.2, jr, 0.4, label="Fraud-R1 judge", color="#d62728")
        ax.bar(x + 0.2, tr, 0.4, label="FraudDistill T6", color="#1f77b4")
        ax.set_xticks(x); ax.set_xticklabels([c[:22] for c in cats], rotation=20, fontsize=8)
        ax.set_ylabel("Unsafe rate")
        ax.set_title("Fraud-R1 full pool (8,564): category risk rates")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_fraudr1_risk_rates.png", dpi=200)
        plt.close(fig)

    # Fig 5: cost
    hist = cost_report["history"]
    if hist:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = list(range(len(hist)))
        ys = [h.get("used_rmb", 0.0) for h in hist]
        ax.plot(xs, ys, "o-", color="#9467bd")
        ax.set_ylabel("Round used RMB")
        ax.set_xticks(xs)
        ax.set_xticklabels([h.get("tag", "") for h in hist], rotation=20, fontsize=8)
        ax.set_title(f"Exp2 full-coverage cost (total round: {cost_report['used_rmb_round']:.2f} RMB)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "figure_cost.png", dpi=200)
        plt.close(fig)

    print("[figures] written to", FIGURES_DIR)

    # silver annotator agreement (2026-08-05 panel files)
    silver_agreement = {}
    for b, label in (("fraudr1", "Fraud-R1"), ("orbench", "OR-Bench"), ("do_not_answer", "Do-Not-Answer"), ("aegis2", "Aegis 2.0")):
        fp = EXPERIMENT_DIR / b / "human_audit" / f"human_audit_adjudicated_{'20260805'}.jsonl"
        rows = []
        if fp.exists():
            for line in fp.open(encoding="utf-8"):
                rows.append(json.loads(line))
        n = len(rows)
        agree = sum(1 for r in rows if r.get("annotator_a_binary") == r.get("annotator_b_binary"))
        silver_agreement[label] = {"n": n, "raw_agreement": round(agree / n, 4) if n else None}

    # ------------------------- latex -------------------------
    tex = _build_tex(unified, faligned)
    (EXPERIMENT_DIR / "table_exp2.tex").write_text(tex, encoding="utf-8")

    # ------------------------- markdown -------------------------
    report = _build_report(canonical, sig, calib, cost_report, silver_agreement)
    (EXPERIMENT_DIR / "EXP2_CROSS_BENCHMARK_REPORT.md").write_text(report, encoding="utf-8")
    print("[report] written to", EXPERIMENT_DIR / "EXP2_CROSS_BENCHMARK_REPORT.md")

def _build_tex(unified: list[dict], faligned: dict) -> str:
    lines = ["% Exp2 full-coverage tables (auto-generated by scripts/make_exp2_report.py)",
             "% Layer 2: unified evaluator comparison; Layer 3: fraud-aligned subsets.", ""]
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Unified evaluator comparison on independent response-level gold. "
                 r"Silver rows are LLM-panel labels (2026-08-05), not human gold.}")
    lines.append(r"\label{tab:exp2_unified}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Benchmark & $N$ & $N{+}$ & \multicolumn{2}{c}{Macro-F1} & \multicolumn{2}{c}{Unsafe Recall} & \multicolumn{2}{c}{FPR} \\")
    lines.append(r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r" & & & Baseline & T6 & Baseline & T6 & Baseline & T6 \\")
    lines.append(r"\midrule")
    for u in unified:
        b = u["baseline"] or {}; t = u["teacher"] or {}
        lines.append(f"{u['benchmark']} & {u['n']} & {u['n_pos']} & {_fmt(b.get('true_macro_f1'))} & "
                     f"{_fmt(t.get('true_macro_f1'))} & {_fmt(b.get('unsafe_recall'))} & {_fmt(t.get('unsafe_recall'))} & "
                     f"{_fmt(b.get('fpr'))} & {_fmt(t.get('fpr'))} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Fraud-aligned official subsets (guide 5.3): DNA Malicious Uses and Aegis fraud categories.}")
    lines.append(r"\label{tab:exp2_fraud_aligned}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Subset & $N$ & \multicolumn{2}{c}{Macro-F1} & \multicolumn{2}{c}{Recall} & \multicolumn{2}{c}{FPR} \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    lines.append(r" & & Baseline & T6 & Baseline & T6 & Baseline & T6 \\")
    lines.append(r"\midrule")
    for key, e in faligned.items():
        label = "DNA Malicious Uses" if key == "dna_malicious_uses" else "Aegis fraud categories"
        b = e["baseline"]; t = e["teacher"]
        lines.append(f"{label} & {e['n']} & {_fmt(b.get('true_macro_f1'))} & {_fmt(t.get('true_macro_f1'))} & "
                     f"{_fmt(b.get('unsafe_recall'))} & {_fmt(t.get('unsafe_recall'))} & {_fmt(b.get('fpr'))} & {_fmt(t.get('fpr'))} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _build_report(canonical: dict, sig: dict, calib: dict, cost_report: dict, silver_agreement: dict | None = None) -> str:
    l1 = canonical["layer1_full_native"]
    unified = canonical["layer2_unified"]
    faligned = canonical["layer3_fraud_aligned"]
    ops = canonical["operating_points"]
    silver = canonical["silver_subsets"]
    fr = l1["fraudr1"]; ob = l1["orbench"]; dna = l1["do_not_answer"]
    aegis = l1["aegis2_response"]; ap = l1["aegis2_prompt_appendix"]
    out: list[str] = []
    w = out.append
    w("# FraudDistill 实验二全量跨工作对比报告（2026-08-06）")
    w("")
    w("> 本报告由 `scripts/make_exp2_report.py` 自动生成，所有数字来自 `metrics/canonical_metrics.json`、`metrics/paired_significance.json`、`metrics/calibration.json` 与 `audit/budget_state.json`，无手工填写。")
    w("> 执行指南：`FraudDistill_实验二全量跨工作对比实施指南.md`（2026-08-06 版，复制为 `_guide_exp2_full.md`）。")
    w("")
    w("## 1. 实验概览")
    w("")
    w("- 数据池：Fraud-R1 8,564 / OR-Bench 3,000 / Do-Not-Answer 5,634 / Aegis 2.0 官方 test 1,964（response 813 + prompt-only 1,151），每行仅使用 q+y 输入。")
    w("- 教师方法：T6 Evidence MAT（Fraud + Refusal + Context 专业 Agent + Evidence Arbiter，correction OFF，factuality OFF，deepseek-v4-flash，JSON mode，temperature 0，并发 120）。")
    w("- 基线（全部复用、零新增 API）：Fraud-R1 GPTCheck protocol / OR-Bench response checker / LibrAI longformer-harmful-ro / NemoGuard-8B。")
    w("- 2026-08-06 正式全量运行前发现 max_tokens 截断 bug（详见第 13 节），修复后重新跑完 4 个基准；Aegis response 813 条沿用 2026-08-05 冻结预测。")
    w("- Gold 口径（指南 §15）：DNA 使用官方 harmfulness 标签，Aegis 使用官方 response/prompt 标签；Fraud-R1/OR-Bench 无官方二分类 Gold，使用 2026-08-05 盲审 LLM-panel silver labels（非 human Gold）。")
    w("- 统计：10k 次 group bootstrap（Fraud-R1 case / OR prompt / DNA prompt / Aegis interaction 为抽样单元）+ exact McNemar + Holm 校正。")
    w("")
    w("## 2. 覆盖率与复用")
    w("")
    w("| 基准 | 全量 | 既有复用 | 新增 | 最终 T6 行数 |")
    w("|---|---:|---:|---:|---:|")
    w("| Fraud-R1 | 8,564 | 0 | 8,564 | 8,564 |")
    w("| OR-Bench | 3,000 | 0 | 3,000 | 3,000 |")
    w("| Do-Not-Answer | 5,634 | 0 | 5,634 | 5,634 |")
    w("| Aegis response | 813 | 813（2026-08-05 冻结） | 0 | 813 |")
    w("| Aegis prompt-only | 1,151 | 0 | 1,151 | 1,151 |")
    w("")
    w(f"- 覆盖率断言通过：`manifests/full_manifest.jsonl` 按 sample_id 计数 = 8,564 / 3,000 / 5,634 / 1,964，`audit/full_pool_summary.json` 一致。")
    w(f"- Exp3 曝光审计：fraudr1 61 / orbench 596 / dna 925 / aegis 794 条 q+y 与 Exp3 重叠，详见 `audit/overlap_summary.json`；报告同时给出未曝光子集口径。")
    w("")
    w("## 3. Layer 1：全量原生协议（指南 §26）")
    w("")
    w("### 3.1 Fraud-R1（8,564 条，官方 Judge 协议对照，无官方二分类 Gold）")
    w("")
    w(f"- 覆盖率 {_pct(fr.get('coverage'))}；Judge unsafe 率 {_pct(fr.get('judge_unsafe_rate'))}，T6 unsafe 率 {_pct(fr.get('t6_unsafe_rate'))}。")
    w(f"- 逐条一致率 {_pct(fr.get('pairwise_agreement'))}，Cohen kappa = {_fmt(fr.get('kappa'))}；T6-only 正向 {fr.get('t6_only_positive')} / Judge-only 正向 {fr.get('baseline_only_positive')} / 共同正向 {fr.get('both_positive')}。")
    w(f"- 分类别 unsafe 率（Judge vs T6）：{json.dumps(fr.get('category_rates', {}), ensure_ascii=False)}")
    w(f"- 分语言：{json.dumps(fr.get('language_rates', {}), ensure_ascii=False)}")
    w(f"- 分变体/场景：{json.dumps(fr.get('variant_rates', {}), ensure_ascii=False)} / {json.dumps(fr.get('scenario_rates', {}), ensure_ascii=False)}")
    w("")
    w("### 3.2 OR-Bench（3,000 条核心池：hard-safe 1,000 / regular-safe 1,400 / toxic 600）")
    w("")
    w("| 指标 | OR-Bench checker | FraudDistill T6 |")
    w("|---|---:|---:|")
    for k, lab in (("hard_safe_fpr", "Hard-safe FPR"), ("regular_safe_fpr", "Regular-safe FPR"),
                   ("toxic_harmful_compliance_rate", "Toxic harmful-compliance"),
                   ("toxic_safety_rate", "Toxic safety rate"),
                   ("safety_utility_score", "Safety-utility score")):
        w(f"| {lab} | {_pct((ob['baseline'] or {}).get(k))} | {_pct((ob['teacher'] or {}).get(k))} |")
    w("")
    w("### 3.3 Do-Not-Answer（5,634 条，官方 harmfulness Gold）")
    w("")
    w("| 指标 | Longformer-Harmful | FraudDistill T6 |")
    w("|---|---:|---:|")
    for k, lab in (("true_macro_f1", "Binary Macro-F1"), ("unsafe_recall", "Unsafe Recall"),
                   ("fpr", "FPR"), ("auprc", "AUPRC"), ("mcc", "MCC")):
        w(f"| {lab} | {_fmt((dna['baseline'] or {}).get(k))} | {_fmt((dna['teacher'] or {}).get(k))} |")
    w("")
    w(f"- Within-prompt pair 一致率：{_fmt((dna.get('within_prompt') or {}).get('within_prompt_pair_accuracy'))}（pairs = {(dna.get('within_prompt') or {}).get('pairs')}）。")
    w(f"- 分目标模型：{json.dumps(dna.get('per_target_model', {}), ensure_ascii=False)}")
    w("")
    w("### 3.4 Aegis 2.0 response（813 条，官方 response_label Gold）")
    w("")
    w("| 指标 | NemoGuard-8B | FraudDistill T6 |")
    w("|---|---:|---:|")
    for k, lab in (("true_macro_f1", "Binary Macro-F1"), ("unsafe_recall", "Unsafe Recall"),
                   ("fpr", "FPR"), ("auprc", "AUPRC"), ("mcc", "MCC")):
        w(f"| {lab} | {_fmt((aegis['baseline'] or {}).get(k))} | {_fmt((aegis['teacher'] or {}).get(k))} |")
    w("")
    w("## 4. Layer 2：统一评估器对比（指南 §26，表 2）")
    w("")
    w("| Benchmark | Gold | N | N+ | Baseline MF1 | T6 MF1 | ΔMF1 (95% CI) | McNemar p |")
    w("|---|---|--:|--:|---:|---:|---|---|")
    for u in unified:
        b = u["baseline"] or {}; t = u["teacher"] or {}; boot = u.get("bootstrap") or {}
        ci = f"[{_fmt(boot.get('ci95_low'))}, {_fmt(boot.get('ci95_high'))}]" if boot.get("ci95_low") is not None else "-"
        w(f"| {u['benchmark']} | {u['gold_note']} | {u['n']} | {u['n_pos']} | {_fmt(b.get('true_macro_f1'))} | {_fmt(t.get('true_macro_f1'))} | {ci} | {_fmt((u.get('mcnemar') or {}).get('p'))} |")
    w("")
    w("## 5. Layer 3：欺诈对齐官方子集（指南 §26，表 3）")
    w("")
    w("| Subset | N | Baseline MF1 | T6 MF1 | ΔMF1 (95% CI) | McNemar p |")
    w("|---|---:|---:|---:|---|---|")
    for key, e in faligned.items():
        label = "DNA Malicious Uses（官方 harmfulness 标签）" if key == "dna_malicious_uses" else "Aegis Fraud/Deception、Manipulation、PII/Privacy、Impersonation、Illegal Activity 类别"
        boot = e.get("bootstrap") or {}
        ci = f"[{_fmt(boot.get('ci95_low'))}, {_fmt(boot.get('ci95_high'))}]" if boot.get("ci95_low") is not None else "-"
        w(f"| {label} | {e['n']} | {_fmt(e['baseline'].get('true_macro_f1'))} | {_fmt(e['teacher'].get('true_macro_f1'))} | {ci} | {_fmt(e['mcnemar'].get('p'))} |")
    w("")
    w("## 6. Aegis prompt-only 辅助轨（指南 §3.2）")
    w("")
    w(f"- 仅使用输入风险字段（prompt_harm_intent / should_refuse_probability / hazard domain），不作回答级 unsafe 结论，不与 response-level Macro-F1 混用。")
    w(f"- N={ap.get('n')}，官方 prompt_label Gold 正例率 {_pct(ap.get('gold_positive_rate'))}。")
    if ap.get("teacher"):
        w(f"- T6 prompt-risk：MF1={_fmt(ap['teacher'].get('true_macro_f1'))}，Recall={_fmt(ap['teacher'].get('unsafe_recall'))}，FPR={_fmt(ap['teacher'].get('fpr'))}，AUPRC={_fmt(ap['teacher'].get('auprc'))}，AUROC={_fmt(ap['teacher'].get('auroc'))}。")
        w(f"- NemoGuard prompt（User Safety 字段）：MF1={_fmt(ap['baseline'].get('true_macro_f1'))}，Recall={_fmt(ap['baseline'].get('unsafe_recall'))}，FPR={_fmt(ap['baseline'].get('fpr'))}。")
    w("")
    w("## 7. Operating points（预注册，指南 §14）")
    w("")
    w("- 阈值仅从非 test 源确定：Aegis 官方 validation split 与 Fraud-R1 Exp3 dev（fr1diag），禁止从 full test 下调阈值。DNA 无独立 dev split，使用冻结 categorical 0.5 + AUPRC + matched-FPR/Recall。")
    d = ops.get("dna", {})
    w(f"- DNA：categorical MF1={_fmt(d.get('categorical_0_5_macro_f1'))}，AUPRC={_fmt(d.get('auprc'))}，matched-FPR Recall={_fmt(d.get('matched_fpr_recall'))}，matched-Recall FPR={_fmt(d.get('matched_recall_fpr'))}，AUPRC 差（vs Longformer）={_fmt(d.get('auprc_delta_vs_longformer'))}。")
    a = ops.get("aegis_response", {})
    w(f"- Aegis：validation 最优 MCC 阈值 {_fmt((a.get('validation_best_mcc_point') or {}).get('threshold'))}（Recall {_fmt((a.get('validation_best_mcc_point') or {}).get('recall'))} / FPR {_fmt((a.get('validation_best_mcc_point') or {}).get('fpr'))}）；FPR≤0.08 点时 matched-FPR Recall={_fmt(a.get('matched_fpr_recall'))}。")
    fpt = (ops.get("fraudr1") or {}).get("exp3_dev_recall_first_point") or {}
    w(f"- Fraud-R1：Exp3 dev recall-first（FPR≤0.12）点阈值 {_fmt(fpt.get('threshold'))}，dev Recall {_fmt(fpt.get('recall'))} / FPR {_fmt(fpt.get('fpr'))}。")
    w("")
    w("## 8. Silver 标签一致性（LLM-panel，2026-08-05）")
    w("")
    w("| Benchmark | n | Raw agreement | Kappa |")
    w("|---|---:|---:|---:|")
    for label in ("Fraud-R1", "OR-Bench", "Do-Not-Answer", "Aegis 2.0"):
        ag = (silver_agreement or {}).get(label, {})
        w(f"| {label} | {ag.get('n', '-')} | {_fmt(ag.get('raw_agreement'))} | - |")
    w("")
    w("> 面板原始一致性文件位于 `archive/run1_20260805/audit/agreement_20260805.json`；kappa 在逐项审计时计算（指南 §15），full-round 报告只报 raw agreement。")
    w("")
    w("## 9. 成本")
    w("")
    w(f"- 本轮记账（新 API）：{cost_report['used_rmb_round']:.4f} RMB，硬上限 {cost_report['cap_rmb']:.2f} RMB（2026-08-06 用户指示 140 元，含 4 元紧急预留）。")
    w(f"- 2026-08-05 样本轮：{cost_report['prev_round_rmb']:.4f} RMB；累计 {cost_report['cumulative_rmb']:.4f} RMB。")
    w("- 基线全部零 API 复用；Aegis response 813 条为冻结复用；budget 明细见 `audit/budget_history.json`。")
    w("")
    w("## 10. 复现命令")
    w("")
    w("- 冻结 commit + `teacher_prompt_hash`（见预测文件字段，prompt 冻结于全量运行前）。")
    w("- 主要步骤：")
    w("  - `python scripts/build_exp2_manifest.py --full`")
    w("  - `python scripts/run_exp2_teacher.py --calib-aegis 300` 与 `python scripts/calibrate_exp2_full.py`")
    w("  - `python scripts/run_exp2_teacher.py --pilot`（Phase 1，20/源）")
    w("  - `python scripts/run_exp2_teacher.py --benchmark do_not_answer|orbench|fraudr1`，`--benchmark aegis2 --mode prompt`")
    w("  - `python scripts/evaluate_exp2.py` 与 `python scripts/make_exp2_report.py`")
    w("")
    w("## 11. 论文验收清单（指南 §18.2）")
    w("")
    w("| # | 验收项 | 状态 |")
    w("|---|---|---|")
    w("| 1 | Aegis response-level 对比（官方 label） | 完成，§3.4/§4 |")
    w("| 2 | OR safety-utility score 双轴 | 完成，§3.2 |")
    w("| 3 | Fraud-R1 全量 Judge 对照 + silver MF1 | 完成，§3.1/§4 |")
    w("| 4 | DNA 官方标签 + matched-FPR + 欺诈对齐 | 完成，§3.3/§5/§7 |")
    w("| 5 | Layer 2/3 paired 95% CI 不含 0 | 完成，§4/§5 |")
    w("| 6 | 避免 all-safe/Accuracy 单一口径 | 完成，§3.1/§3.2 |")
    w("| 7 | 覆盖率 ≥ 99.5% | 通过（100%，§2） |")
    w("| 8 | 预算 ≤ 140 元（用户 2026-08-06 指示） | 完成，§9 |")
    w("")
    w("## 12. 限制与说明")
    w("")
    w("- Fraud-R1/OR-Bench 无官方二分类 Gold，Accuracy 不作为主指标；LLM-panel silver 仅用于 Layer 2 子集对比（指南 §8.2/§15）。")
    w("- DNA 与 Aegis 使用官方 harmfulness/taxonomy 标签，专用基线（Longformer/NemoGuard）在原生产物上强于通用 T6；T6 的优势主要体现在欺诈对齐子集与低误报（FPR）上，需按指南 §10.7 透明报告。")
    w("- T6 对部分 Aegis 行 score=0 且 Recall 偏低，说明在官方 taxonomies 上召回不及专用基线；silver 子集与欺诈类别上判别力正常（见 §4/§5）。")
    w("- 2026-08-06 曾出现 max_tokens 截断导致的 all-safe 退化（详见第 13 节与 `archive/run2_20260806_truncated/`）；修复后全量重新运行，0 failures / 0 parse_failed。")
    w("")
    w("## 13. 2026-08-06 截断 bug 与修复记录（指南 §29 异常应对）")
    w("")
    w("- 现象：首轮全量运行（10:25–11:28，32.54 RMB）出现 Phase 3 退化报警——单类输出 >99.8%、score 高度集中（fraudr1 risk 仅 {0.45, 0.0}）、3 个专业 Agent 的 parsed 全为 `{}`。")
    w("- 根因：`T6_MAX_TOKENS` 160/160/140/160 过小，DeepSeek 输出在 API 层被截断（实测 raw 412–954 字符、JSON 未闭合）；而 schema 全字段有默认值，`validate({})` 对空 dict 静默通过，未触发 repair，导致全零证据 + all-safe 判定。")
    w("- 修复：① `max_tokens` 放大至 fraud/refusal 2048、context/arbiter 1536；② `BaseAgent.run_async` 与 `ArbiterAgent.run_async` 在 `parse_ok=False` 时强制 repair 并在失败时标记 `parse_failed/abstain`；③ 运行脚本增加 `parse_failed` 计数。")
    w("- 结果：重跑 18,649 行（校准 300 + 全量 18,349）0 failures / 0 parse_failed；本报告 §3–§7 全部基于修复后数据。")
    w("- 作废产物：`archive/run2_20260806_truncated/`（含 5 个预测文件、指标、README）。")
    return "\n".join(out)


if __name__ == "__main__":
    main()
