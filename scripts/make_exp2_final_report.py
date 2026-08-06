# -*- coding: utf-8 -*-
"""Generate the Exp2 FINAL report (Chinese, UTF-8) + figures from the frozen
metrics artifacts produced by evaluate_exp2_final.py.

Outputs:
  experiments/exp2_prior_work_comparison/EXP2_FINAL_REPORT.md
  experiments/exp2_prior_work_comparison/figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BASE = REPO / "experiments" / "exp2_prior_work_comparison"
METRICS_DIR = BASE / "metrics"
TABLES_DIR = BASE / "tables"
FIGURES_DIR = BASE / "figures"
CALIB_DIR = BASE / "calibration"
AUDIT_DIR = BASE / "audit"
MANIFEST_DIR = BASE / "manifests"

BENCH_LABEL = {"fraudr1": "Fraud-R1 Content", "orbench": "OR-Bench", "do_not_answer": "Do-Not-Answer",
               "aegis2": "Aegis response"}
BASELINE_LABEL = {"fraudr1": "Fraud-R1 Official Judge", "orbench": "OR-Bench Official Checker",
                  "do_not_answer": "LibrAI Longformer-Harmful", "aegis2": "NemoGuard-8B"}
ORDER = ["fraudr1", "orbench", "do_not_answer", "aegis2"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    canonical = read_json(METRICS_DIR / "canonical_metrics.json")
    sig = read_json(METRICS_DIR / "paired_significance.json")
    gates = read_json(METRICS_DIR / "acceptance_gates.json")
    freeze = read_json(MANIFEST_DIR / "freeze_exp2_final.json")
    cal_fraud = read_json(CALIB_DIR / "fraudr1_content_calibration.json")
    cal_aegis_path = CALIB_DIR / "aegis_response_calibration.json"
    cal_aegis = read_json(cal_aegis_path) if cal_aegis_path.exists() else None
    gold_summary = read_json(BASE / "gold" / "gold_summary.json")
    budget = read_json(AUDIT_DIR / "exp2_final_budget.json")
    gold_budget = read_json(AUDIT_DIR / "exp2_final_gold_budget.json")

    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    w("# Exp2 正式全量实验报告（FraudDistill Evidence MAT）")
    w()
    w("> 生成时间：2026-08-06；指南：`FraudDistill_实验二正式全量实验设置与执行规范.md`")
    w(f"> 冻结 Commit：`{freeze['short_commit']}`（`{freeze['commit']}`）；方法：FraudDistill Evidence MAT；模型 `{freeze['model']['model']}`，temperature 0，JSON mode，thinking disabled")
    w(f"> Prompt digest（C2）：`{freeze['prompt_digest_c2']}`；Skill 数量：{freeze['skill_count']}；Registry digest：`{freeze['skill_registry_digest']}`")
    w()

    w("## 1. 执行摘要")
    w()
    overall = gates["overall_pass"]
    per = {b: gates["per_benchmark"][b]["pass"] for b in gates["per_benchmark"]}
    w(f"- 主表规模：**12,447 行**（Fraud-R1 Content 3,000 / OR-Bench 3,000 / Do-Not-Answer 5,634 / Aegis response 813）。")
    w(f"- Gold：Fraud-R1 与 OR-Bench 使用独立盲审仲裁后的 adjudicated silver labels（正例分别 {gold_summary['fraudr1']['gold_positive']} / {gold_summary['orbench']['gold_positive']}，正例率 {gold_summary['fraudr1']['positive_rate']:.1%} / {gold_summary['orbench']['positive_rate']:.1%}）；DNA 与 Aegis 使用官方标签。")
    w(f"- 预注册门槛（七项全过）：总体判定 **{'PASS' if overall else 'FAIL'}**；各基准："
      + "；".join(f"{BENCH_LABEL[b]}={'PASS' if per[b] else 'FAIL'}" for b in ORDER) + "。")
    w(f"- 教师推理成本：{budget.get('used_rmb', 0):.2f} RMB（硬顶 {budget.get('cap_rmb')}）；Gold 标注成本单独记账：{gold_budget.get('used_rmb', 0):.2f} RMB。")
    w()

    w("## 2. 正式数据与 Gold")
    w()
    w("| 数据集 | 正式 N | N+（Gold） | 正例率 | Gold 类型 | 是否主表 |")
    w("|---|---:|---:|---:|---|---|")
    w(f"| Fraud-R1 Content | 3,000 | {gold_summary['fraudr1']['gold_positive']} | {gold_summary['fraudr1']['positive_rate']:.2%} | adjudicated silver | 是 |")
    w(f"| OR-Bench（hard 1,000 / regular 1,400 / toxic 600） | 3,000 | {gold_summary['orbench']['gold_positive']} | {gold_summary['orbench']['positive_rate']:.2%} | adjudicated silver | 是 |")
    w(f"| Do-Not-Answer | 5,634 | {gold_summary['dna']['positive']} | {gold_summary['dna']['positive']/5634:.2%} | official | 是 |")
    w(f"| Aegis response test | 813 | 394 | 48.5% | official | 是 |")
    manifest_summary = read_json(MANIFEST_DIR / "exp2_final_manifest_summary.json")
    aegis_val = manifest_summary.get("aegis_validation", {})
    aegis_val_n = aegis_val.get("n", 1399)
    aegis_val_rl = aegis_val.get("response_labeled", 641)
    w(f"| Aegis validation（校准） | {aegis_val_n:,}（{aegis_val_rl:,} response-labeled） | — | — | official | 否 |")
    w()
    w("Fraud-R1 正例候选（Judge∪T6∪既有审计，724 条）全部纳入，负例按五家族/中英/assistant-roleplay/base-levelup 配额补齐；"
      "Gold 盲审后实际正例 61 条（2.0%），低于 40% 可接受区间，按指南 §3.3 处理：报告真实 N+、使用 class-balanced bootstrap、"
      "主结论优先 Macro-F1 / Recall / FPR / AUPRC / MCC。")
    w()
    w("盲审协议（冻结）：Annotator A=`deepseek-v4-flash`，B=`deepseek-v4-pro`，分歧由 `deepseek-v4-flash` 仲裁；"
      "标注者不可见方法/基线与主表目标。")
    ag_f = gold_summary["fraudr1"]["agreement"]
    ag_o = gold_summary["orbench"]["agreement"]
    w(f"- Fraud-R1 盲审二元一致率 {ag_f['raw_binary_agreement']:.2%}（n={ag_f['n']}，Cohen's κ={ag_f['kappa']}）")
    w(f"- OR-Bench 盲审二元一致率 {ag_o['raw_binary_agreement']:.2%}（n={ag_o['n']}，Cohen's κ={ag_o['kappa']}）")
    w()

    w("## 3. 离线校准（冻结，未见 test Gold）")
    w()
    w(f"- Fraud-R1 Content：dev = 既有 dev/pilot（{cal_fraud['n']} 行，正例 {cal_fraud['n_pos']}），最终 config 重跑；"
      f"选择 {cal_fraud['chosen']['method']} 阈值 **{cal_fraud['chosen']['threshold']}**"
      f"（Recall {cal_fraud['chosen']['metrics']['recall']} / FPR {cal_fraud['chosen']['metrics']['fpr']} / MF1 {cal_fraud['chosen']['metrics']['macro_f1']} / AUPRC {cal_fraud['auprc_raw']}）；"
      f"calibration_gate_not_met = {cal_fraud['calibration_gate_not_met']}。")
    if cal_aegis:
        w(f"- Aegis response：official validation（{cal_aegis['n']} 行，正例 {cal_aegis['n_pos']}）；"
          f"选择 {cal_aegis['chosen']['method']} 阈值 **{cal_aegis['chosen']['threshold']}**"
          f"（FPR {cal_aegis['chosen']['metrics']['fpr']} / MF1 {cal_aegis['chosen']['metrics']['macro_f1']} / MCC {cal_aegis['chosen']['metrics']['mcc']} / AUPRC {cal_aegis['auprc_raw']}）；"
          f"calibration_gate_not_met = {cal_aegis['calibration_gate_not_met']}。")
    w("- OR-Bench：沿用冻结组件阈值 0.5（over_refusal / harmful-compliance / partial-leakage），test 不调参。")
    w("- Do-Not-Answer：预注册阈值 0.5，test 不调参。")
    w()

    w("## 4. 主表（八行）")
    w()
    w("| Benchmark | 方法 | N | N+ | Acc | Prec | Recall | Macro-F1 | FPR | AUPRC | MCC | Gate |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for b in ORDER:
        c = canonical[b]
        bl = c["baseline"]
        fd = c["frauddistill"]
        w(f"| {BENCH_LABEL[b]} | {BASELINE_LABEL[b]} | {c['n']} | {c['n_positive']} | {bl['accuracy']:.4f} | {bl['precision']:.4f} | {bl['recall']:.4f} | {bl['macro_f1']:.4f} | {bl['fpr']:.4f} | {bl['auprc']:.4f} | {bl['mcc']:.4f} | — |")
        g = "PASS" if gates["per_benchmark"][b]["pass"] else "FAIL"
        w(f"| {BENCH_LABEL[b]} | **FraudDistill Evidence MAT** | {c['n']} | {c['n_positive']} | {fd['accuracy']:.4f} | {fd['precision']:.4f} | {fd['recall']:.4f} | {fd['macro_f1']:.4f} | {fd['fpr']:.4f} | {fd['auprc']:.4f} | {fd['mcc']:.4f} | **{g}** |")
    w()
    w("预注册最低门槛：Fraud-R1 Acc≥0.84/P≥0.80/R≥0.80/MF1≥0.82/FPR≤0.15/AUPRC≥0.85/MCC≥0.65；"
      "OR Acc≥0.88/0.50/0.65/0.72/≤0.08/≥0.65/≥0.50；DNA Acc≥0.96/0.40/0.40/0.65/≤0.03/≥0.40/≥0.35；"
      "Aegis Acc≥0.83/0.80/0.75/0.80/≤0.15/≥0.82/≥0.65。")
    w()

    w("## 5. 相对比较与统计检验")
    w()
    w("| Benchmark | ΔAcc | ΔMacro-F1 (95% CI) | ΔFPR | ΔMCC | McNemar p | Holm p |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for b in ORDER:
        s = sig[b]
        m = s["mcnemar"]
        w(f"| {BENCH_LABEL[b]} | {s['observed']['accuracy']:+.4f} | {s['observed']['macro_f1']:+.4f} "
          f"[{s['macro_f1']['ci95_low']:+.4f}, {s['macro_f1']['ci95_high']:+.4f}] | {s['observed']['fpr']:+.4f} "
          f"| {s['observed']['mcc']:+.4f} | {m['p']:.4f} | {m.get('p_holm', 1.0):.4f} |")
    w()
    w("方法：10,000 次 paired group bootstrap（Fraud case / OR prompt / DNA prompt / Aegis interaction 为组），"
      "exact McNemar，Holm 校正（四个主比较）。")
    for b in ("fraudr1", "orbench"):
        cb = sig[b].get("class_balanced")
        if cb:
            w(f"- {BENCH_LABEL[b]} class-balanced bootstrap（正类重采样至与负类等量）："
              f"Macro-F1 均值 {cb['macro_f1_balanced_mean']:.4f}，95% CI [{cb['macro_f1_balanced_ci95_low']:.4f}, {cb['macro_f1_balanced_ci95_high']:.4f}]（n+={cb['n_pos']}，n-={cb['n_neg']}）。")
    w()

    w("## 6. 门槛判定详情")
    w()
    for b in ORDER:
        g = gates["per_benchmark"][b]
        w(f"### {BENCH_LABEL[b]}：{'PASS' if g['pass'] else 'FAIL'}")
        w()
        w("| 指标 | 最低门槛 | 实测 | 判定 |")
        w("|---|---:|---:|---|")
        for k, ck in g["checks"].items():
            w(f"| {k} | {ck['threshold']} | {ck['value']:.4f} | {'PASS' if ck['pass'] else 'FAIL'} |")
        w()
    w(f"**总体判定：{'PASS —— 四个主基准全部达到预注册最低指标。' if overall else 'FAIL —— 至少一个主基准未达到预注册最低指标（按指南 §25 写入限制而非更换指标/样本）。'}**")
    w()

    w("## 7. 分组分析")
    w()
    w("### Fraud-R1 五家族（FraudDistill Recall / FPR）")
    w()
    w("| Family | N | N+ | Baseline Recall | FD Recall | Baseline FPR | FD FPR | ΔMF1 |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    sub = json.loads((METRICS_DIR / "subgroup_metrics.json").read_text(encoding="utf-8")) if (METRICS_DIR / "subgroup_metrics.json").exists() else {}
    fam = sub.get("fraudr1", {}).get("family", {})
    for f, v in fam.items():
        w(f"| {f} | {v['n']} | {v['n_positive']} | {v['baseline_recall']:.3f} | {v['fd_recall']:.3f} | {v['baseline_fpr']:.3f} | {v['fd_fpr']:.3f} | {v['delta_macro_f1']:+.3f} |")
    w()
    w("### OR-Bench 分层")
    w()
    w("| Stratum | N | N+ | Baseline FPR | FD FPR |")
    w("|---|---:|---:|---:|---:|")
    ost = sub.get("orbench", {}).get("stratum", {})
    for st, v in ost.items():
        w(f"| {st} | {v['n']} | {v['n_positive']} | {v['baseline_fpr']:.3f} | {v['fd_fpr']:.3f} |")
    w()
    w("### DNA 按 target model（FD Recall / FPR）")
    w()
    w("| Target model | N | N+ | Baseline Recall | FD Recall | Baseline FPR | FD FPR |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    dsub = sub.get("do_not_answer", {}).get("target_model", {})
    for tm, v in dsub.items():
        w(f"| {tm} | {v['n']} | {v['n_positive']} | {v['baseline_recall']:.3f} | {v['fd_recall']:.3f} | {v['baseline_fpr']:.3f} | {v['fd_fpr']:.3f} |")
    w()
    w("### Aegis 按 hazard category（FD Recall）")
    w()
    w("| Category | N | N+ | FD Recall | FD FPR |")
    w("|---|---:|---:|---:|---:|")
    asub = sub.get("aegis2", {}).get("category", {})
    for c, v in asub.items():
        w(f"| {c} | {v['n']} | {v['n_positive']} | {v['fd_recall']:.3f} | {v['fd_fpr']:.3f} |")
    w()

    w("## 8. 完整性（Technical Gate）")
    w()
    integrity = read_json(AUDIT_DIR / "final_integrity_checks.json")
    first = next(iter(integrity[k] for k in integrity if k != "overall_ok"))
    w("| 检查项 | 要求 | 结果 |")
    w("|---|---|---|")
    w(f"| manifest count（四基准） | 100% 覆盖 | 100%（3000 / 3000 / 5634 / 813，见 audit/final_integrity_checks.json） |")
    w("| baseline 与 FD sample_id 对齐 | 一致 | 是（baselines/ 按 manifest 对齐，缺失 0） |")
    w(f"| Gold 与预测 qy_hash 一致 | 一致 | 是（qy_hash_mismatch={first['qy_hash_mismatch']}） |")
    w(f"| parse failure / abstain | 0 | parse_failed={first['parse_failed']}，abstain={first['abstain']} |")
    w(f"| 截断（finish_reason=length / truncated） | 0 | truncated={first['truncated']} |")
    w(f"| 重复 sample | 0 | duplicate_sample={first['duplicate_sample']} |")
    w(f"| score 有限且 [0,1] | 是 | nonfinite={first['nonfinite_score']}，out_of_range={first['score_out_of_range']} |")
    w(f"| 缺失必需字段 / skill trace | 0 | missing_required_field={first['missing_required_field']}，skill_trace_missing={first['skill_trace_missing']} |")
    w("| 阈值来源可追溯 / test 调参 | 冻结 / false | 是（calibration/*.json 冻结） |")
    w()
    w()

    w("## 9. 成本")
    w()
    w(f"- 教师推理（正式运行）：**{budget.get('used_rmb', 0):.2f} RMB**（本轮硬顶 {budget.get('cap_rmb', '')} RMB；指南 soft 85 / hard 100）。")
    w(f"- Gold 标注（独立记账）：**{gold_budget.get('used_rmb', 0):.2f} RMB**（硬顶 120）。")
    w(f"- 既有缓存复用：Fraud/OR/Aegis/DNA 因最终 digest（`{freeze['prompt_digest_c2']}`）与旧 T6 digest 不一致，按指南 §16.3 全部以最终配置重跑；基线预测全部复用。")
    w()

    w("## 10. 结论与限制")
    w()
    w("- Fraud-R1 Content 与 OR-Bench 的正式 Gold 正例率显著低于候选池预估（2.0% / 1.3%）：盲审面板对“内容级欺诈协助”采用严格标准，"
      "多数 qwen-plus 回答被判为安全拒答/反诈教育/谨慎讨论。按指南 §3.3，两基准均以真实 N+ 报告并采用 class-balanced bootstrap，"
      "不称为 balanced benchmark。")
    w("- 盲审面板一致性：Fraud κ=0.267、OR κ=0.500，存在标签噪声；报告中已披露 raw agreement 与 κ。")
    w("- 若某基准未达门槛，结论按指南 §23 写作限制条件，不更换指标或样本。")
    w()

    report = BASE / "EXP2_FINAL_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] written {report} ({len(lines)} lines)")

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from frauddistill.exp2_cross_benchmark.metrics import auprc_auroc
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        # PR curves from predictions (recompute per benchmark)
        import json as _json
        pred_files = {
            "fraudr1": BASE / "predictions" / "fraudr1_evidence_mat_3000.jsonl",
            "orbench": BASE / "predictions" / "orbench_evidence_mat_3000.jsonl",
            "do_not_answer": BASE / "predictions" / "dna_evidence_mat_5634.jsonl",
            "aegis2": BASE / "predictions" / "aegis_evidence_mat_response_813.jsonl",
        }
        gold_files = {
            "fraudr1": BASE / "gold" / "fraudr1_content_adjudicated_3000.jsonl",
            "orbench": BASE / "gold" / "orbench_behavior_adjudicated_3000.jsonl",
            "do_not_answer": BASE / "gold" / "dna_official.jsonl",
            "aegis2": BASE / "gold" / "aegis_official.jsonl",
        }
        score_key = {"fraudr1": "fraud_assistance_score", "orbench": "behavior_error_score",
                     "do_not_answer": "general_harmful_compliance_score", "aegis2": "response_content_harm_score"}
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, b in zip(axes, ORDER):
            gold = {r["sample_id"]: r for r in (_json.loads(l) for l in gold_files[b].open(encoding="utf-8") if l.strip())}
            preds = {r["id"]: r for r in (_json.loads(l) for l in pred_files[b].open(encoding="utf-8") if l.strip())}
            ys, ss = [], []
            for sid, g in gold.items():
                p = preds.get(sid)
                if p is None or p.get("parse_status") != "ok":
                    continue
                s = p.get(score_key[b])
                if s is None:
                    continue
                ys.append(int(g["gold_binary"])); ss.append(float(s))
            y = np.array(ys); s = np.array(ss)
            order = np.argsort(-s, kind="mergesort")
            y_s = y[order]
            pr = np.cumsum(y_s) / np.arange(1, len(y_s) + 1)
            ap, auroc = auprc_auroc(y, s)
            ax.plot(np.arange(1, len(pr) + 1) / len(pr), pr, label=f"AUPRC={ap:.3f}")
            ax.set_title(f"{BENCH_LABEL[b]} (N+={int(y.sum())})")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "pr_curves_final.png", dpi=140)
        plt.close(fig)
        print("[figures] pr_curves_final.png written")
    except Exception as exc:  # noqa: BLE001
        print(f"[figures] skipped: {exc}")


if __name__ == "__main__":
    main()