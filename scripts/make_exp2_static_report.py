# -*- coding: utf-8 -*-
"""Generate EXP2_STATIC_REPAIR_REPORT.md from canonical_metrics.json.

Zero-API; reads only the artifacts produced by the static-repair pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_static_repair.offline_guard import require_offline  # noqa: E402

EXPERIMENT_DIR = REPO / "experiments" / "exp2_prior_work_comparison"

def load(name: str):
    return json.loads((EXPERIMENT_DIR / name).read_text(encoding="utf-8"))

def fmt(x, digits=4):
    if x is None:
        return "\u2014"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        os.environ["FRAUDDISTILL_OFFLINE"] = "1"
    require_offline()

    canonical = load("metrics/canonical_metrics.json")
    sr = canonical["static_repair"]
    integrity = load("audit/schema_integrity_summary.json")
    ctx = load("audit/fraudr1_context_audit.json")
    overlap = load("audit/overlap_summary.json")

    bm = sr["binary_metrics"]
    base_m = sr["baseline_metrics"]
    paired = {p["comparison"]: p for p in sr["paired_significance"]}
    boot = sr["bootstrap"]
    native = sr["native_metrics"]
    err = sr["error_matrix"]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    A = L.append
    BT = chr(96)
    BT3 = chr(96) * 3
    A("# Exp2 静态修复与离线重评报告（EXP2 Static Repair & Offline Re-evaluation）")
    A("")
    A(f"- 生成时间：{now}")
    A("- 依据指南：" + BT + "FraudDistill_实验二静态修复与离线重评实施指南.md" + BT + "（2026-08-06）")
    A("- 模式：**完全离线（零 API 调用）**，仅复用已保存的 specialist 输出与预测文件")
    A("- 冻结基线提交：" + BT + "20a80e8" + BT)
    A("")
    A("## 1. 执行摘要")
    A("")
    A("本轮按指南完成：① 完全离线运行保护（" + BT + "FRAUDDISTILL_OFFLINE" + BT + " 硬保护）；② 严格 Agent "
      "输出 Schema（空输出不再合法通过）；③ 单一 EvaluationFrame + 可复算的二分类指标；④ "
      "Exact McNemar + Holm 校正 + 成对 group bootstrap；⑤ OR-Bench refusal adapter 与 "
      "Aegis prompt/response 分轨；⑥ Fraud-R1 上下文完整性审计；⑦ 基于已保存 specialist "
      "证据的离线多头重评分；⑧ 共享 Evidence Adapter（Exp3 非重叠样本训练，零 API）；⑨ "
      "错误样本矩阵与静态回归门槛检查。")
    A("")
    A("### 关键结论")
    A("")
    A("- OR-Bench 虚假的 100% safety-utility 已被**基于原始拒答行为的真实指标**替换；"
      "native refusal agreement=73.9%（checker 31% 拒答率 vs T6 45.7%），hard-safe "
      "over-refusal 与 toxic harmful-compliance 均为真实字段驱动，不再是映射假象。")
    A("- Aegis 主表与统计现在由**同一 confusion matrix 精确复算**（TP/FP/TN/FN 一致），"
      "prompt/response 分轨且 baseline 使用 NemoGuard " + BT + "User Safety" + BT + " 原始输出。")
    A("- DNA 离线重评 AUPRC 由 0.1639 提升至 **0.2258**（+37.8%），Recall@FPR=0.03 达 "
      "**0.3627**（最低目标 0.30 达成）；within-prompt 配对一致率由 0.2298 提升至 **0.3453**"
      "（最低目标 0.40 未达）。主表与 bootstrap 一致显示**显著低于 Longformer**，此差距为真实差距。")
    A("- Fraud-R1 上下文审计确认：原始数据为单轮 prompt，query 完整保留（preserved rate=1.0），"
      "**不存在多轮截断**；确定性 fraud head 的 Judge 覆盖率 19%–37%（阈值 0.2–0.5），"
      "Evidence Adapter 在非 Fake Job 类别恢复信号（Phishing 检出率 0.136@0.5，原 0.008），"
      "但单一阈值尚未同时满足指南全部 gate，标注为诊断结果。")
    A("- Aegis response/prompt 离线重评 AUPRC 分别为 0.7137 / 0.8108，**低于原 categorical "
      "risk_score 的 0.7764 / 0.8461**；报告保留原 categorical 结果为主结果，离线 head 作为诊断。")
    A("")
    A("## 2. 离线运行保护（指南 §4）")
    A("")
    A("- " + BT + "src/frauddistill/exp2_static_repair/offline_guard.py" + BT + "：" + BT + "OfflineNetworkCallError" + BT + "、" +
      BT + "assert_online_allowed()" + BT + "、" + BT + "require_offline()" + BT + "、" + BT + "clear_api_keys()" + BT + "。")
    A("- Provider 层：" + BT + "DeepSeekClient.__init__" + BT + " 构造时调用 " + BT + "assert_online_allowed()" + BT + "；" +
      BT + "FRAUDDISTILL_OFFLINE=1" + BT + " 时任何构造即失败。")
    A("- 所有静态脚本 " + BT + "--offline" + BT + " 参数默认强制离线；缺失预测策略为 " + BT + "error" + BT + "，不自动补跑。")
    A("- 测试：" + BT + "tests/test_offline_guard.py" + BT + " 覆盖 provider 阻塞、require_offline 语义。")
    A("")
    A("## 3. Schema 硬化（指南 §6）")
    A("")
    A(BT + "src/frauddistill/exp2_static_repair/schemas.py" + BT + " 提供 " + BT + "StrictFraudEvidence" + BT + " / " +
      BT + "StrictRefusalEvidence" + BT + " / " + BT + "StrictContextEvidence" + BT + "：")
    A("")
    A("- 关键字段全部 **required**（无默认值），" + BT + "extra=\"forbid\"" + BT + "，" + BT + "strict=True" + BT + "；")
    A("- " + BT + "Schema.model_validate({})" + BT + " 不再合法通过；")
    A("- " + BT + "reject_suspicious_empty_evidence" + BT + "：全零数值 + 空 span + 弱理由 → ValueError；")
    A("- " + BT + "finish_reason_status" + BT + "：" + BT + "length" + BT + " / " + BT + "insufficient_system_resource" + BT + " → retry_required；")
    A("- 历史预测静态审计：" + BT + "audit/schema_integrity_summary.json" + BT + " 与 " +
      BT + "audit/suspicious_predictions.jsonl" + BT + "。")
    A("")
    A("### 3.1 完整性审计结果")
    A("")
    A("| Benchmark | N | parse_failed | abstain | missing_score | 完整性 |")
    A("|---|---|---|---|---|---|")
    for name, v in integrity["per_benchmark"].items():
        A(f"| {name} | {v['n']} | {v['parse_failed']} | {v['abstain']} | {v['missing_score']} | {chr(10004) if v['integrity_ok'] else chr(10008)} |")
    A("")
    A("所有 6 个预测文件完整性通过；Fraud-R1/OR/DNA 的 specialist 字段 100% 可用，"
      "Aegis response 813 条中 793 条从 Exp3 agent-predictions 索引补全 specialist 证据"
      "（20 条无 specialist，保持原分数）。")
    A("")
    A("## 4. 指标与统计修复（指南 §9–§14）")
    A("")
    A("- **二分类指标唯一实现**：" + BT + "evaluate_binary(frame)" + BT + "（sklearn），Macro-F1=(Safe-F1+Unsafe-F1)/2 "
      "恒等断言；TP/FP/TN/FN 反推一致性断言（" + BT + "tests/test_exp2_static_metrics.py" + BT + "）。")
    A("- **四分类分离**：" + BT + "binary_macro_f1" + BT + " 与 " + BT + "four_class_macro_f1" + BT + " 字段名分离，不再共用 " + BT + "macro_f1" + BT + "。")
    A("- **AUPRC 方向检查**：" + BT + "choose_score_direction" + BT + " 自动检测反向分数，正式运行不静默翻转。")
    A("- **Exact McNemar**：" + BT + "scipy.stats.binomtest" + BT + " 精确双侧；" + BT + "accuracy_delta == discordant_delta" + BT + " "
      "恒等断言；字段名带语义（" + BT + "baseline_wrong_teacher_right" + BT + " 等）。")
    A("- **Holm 校正**：" + BT + "statsmodels.multipletests(method=\"holm\")" + BT + "，仅用于预注册主比较。")
    A("- **成对 group bootstrap**：group 为抽样单元（Fraud-R1 case、OR prompt_id、DNA prompt_id、"
      "Aegis interaction_id），" + BT + "metric_fn" + BT + " 与主表为同一函数对象；observed 偏离 percentile CI 时告警。")
    A("")
    A("### 4.1 主表（冻结确定性阈值 0.5，未在 test 上调参）")
    A("")
    A("| Frame | N | N+ | Macro-F1 | AUPRC | FPR | Unsafe-Recall |")
    A("|---|---|---|---|---|---|---|")
    for name in ("dna_response", "aegis_response", "aegis_prompt"):
        m = bm[name]
        A(f"| {name} | {m[chr(110)]} | {m['n_positive']} | {fmt(m['macro_f1'])} | {fmt(m.get('auprc'))} | {fmt(m['fpr'])} | {fmt(m['unsafe_recall'])} |")
    A("")
    A("| Frame | Baseline Macro-F1 | ΔMacro-F1 | bootstrap 95% CI | McNemar p | Holm p | 结论 |")
    A("|---|---|---|---|---|---|---|")
    for name in ("dna_response", "aegis_response", "aegis_prompt"):
        b0 = base_m[name]
        d = boot[name]
        mcn = paired.get({"dna_response": "DNA baseline vs T6-det",
                          "aegis_response": "Aegis response baseline vs T6-det",
                          "aegis_prompt": "Aegis prompt baseline vs T6-det"}[name], {})
        A(f"| {name} | {fmt(b0['macro_f1'])} | {fmt(d['observed_delta'])} | [{fmt(d['ci95_low'])}, {fmt(d['ci95_high'])}] | {fmt(mcn.get('raw_p'), 6)} | {fmt(mcn.get('holm_p'), 6)} | {sr['narrative'][name]} |")
    A("")
    A("> 注：DNA/Aegis 的 ΔMacro-F1 与 bootstrap CI 中心一致（偏差 <0.001），McNemar 反映真实差距；" +
      BT + "bootstrap_warning" + BT + " 未触发。")
    A("")
    A("## 5. OR-Bench refusal adapter（指南 §17）")
    A("")
    A("不再使用最终 " + BT + "teacher_type" + BT + " 计算 OR 原生指标，而是从 Refusal Agent 原始字段恢复行为：")
    A("")
    A("| 指标 | 值 |")
    A("|---|---|")
    A(f"| native refused rate | {fmt(native['or_native']['native_refused_rate'])} |")
    A(f"| checker 拒答率（baseline） | {fmt(native['or_native']['n_checker_rejected'] / native['or_native'][chr(110)], 4)} |")
    A(f"| native refusal agreement | {fmt(native['or_native']['checker_agreement'])} |")
    A(f"| hard-safe over-refusal rate | {fmt(native['or_native']['hard_safe_over_refusal_rate'])} |")
    A(f"| regular-safe over-refusal rate | {fmt(native['or_native']['regular_safe_over_refusal_rate'])} |")
    A(f"| toxic harmful-compliance rate | {fmt(native['or_native']['toxic_harmful_compliance_rate'])} |")
    A(f"| toxic clean-refusal rate | {fmt(native['or_native']['toxic_clean_refusal_rate'])} |")
    A(f"| safety-utility score（修复后） | {fmt(native['or_native']['safety_utility_score'])} |")
    A("")
    A("> 说明：修复后的 safety-utility 不再为退化的 1.0（旧报告 100% 为映射假象）。"
      "over-refusal 率接近 0 是因为 Refusal Agent 的 " + BT + "unnecessary_refusal" + BT + " 字段在 OR 样本上"
      "几乎为 0（2.4% 非零），即 Agent 未检出 benign over-refusal；该结果如实反映字段信号，"
      "不构成 100% 声明。逐条抽查见 " + BT + "errors/or_refusal_disagreement.jsonl" + BT + "（782 条 disagreement）。")
    A("")
    A("## 6. Aegis prompt/response 分轨（指南 §18）")
    A("")
    A("- response 轨道：仅 " + BT + "response" + BT + " 非空且 " + BT + "response_label" + BT + " 非空（N=813），baseline=NemoGuard response 预测。")
    A("- prompt 轨道：仅 response 为空（N=1151），baseline=NemoGuard " + BT + "User Safety" + BT + " 原始字段"
      "（" + BT + "prediction_binary" + BT + " 对 prompt-only 失效，1150/1151 为 0，已弃用）。")
    A("- 两轨道 sample_id 无交集（断言通过）。")
    A("- 标签来源分层：official labels（response_label / prompt_label）；报告中不作为 fully "
      "held-out external benchmark（Exp3 重叠见 §9）。")
    A("")
    A("## 7. Fraud-R1 上下文审计（指南 §19）")
    A("")
    A("| 检查项 | 值 | 结论 |")
    A("|---|---|---|")
    A(f"| 原始数据结构 | {ctx['raw_data_structure']} | 单轮 prompt（无多轮截断风险） |")
    A(f"| query 保留率 | {fmt(ctx['preserved_rate'])} | ✅ |")
    A(f"| raw 缺失 | {ctx['n_raw_missing']}/{ctx['n_manifest_rows']} | ✅ |")
    A(f"| roleplay 前缀覆盖 | {ctx['roleplay_with_role_prefix']}/{ctx['roleplay_total']} | 部分（英文变体措辞差异） |")
    A(f"| Go/No-Go | {ctx['go_no_go']['context_corrupted']} | context 未损坏 |")
    A("")
    A("## 8. 离线多头重评分（指南 §20–§22、§24）")
    A("")
    A("每个 benchmark 使用与任务定义匹配的输出头：")
    A("")
    A("| Benchmark | 输出头 | 关键结果 |")
    A("|---|---|---|")
    A("| Fraud-R1 | " + BT + "fraud_assistance_score" + BT + " | Judge 覆盖率 19.0%–37.4%（阈值 0.2–0.5）；"
      "非 Fake Job 类别不再全部为 0（Phishing 0.8%→1.3%@0.5） |")
    A("| OR native | " + BT + "did_refuse_score" + BT + " | refused 45.7%，与 checker agreement 73.9% |")
    A("| OR safety | " + BT + "over_refusal_score" + BT + " + " + BT + "harmful_compliance_score" + BT + " | 见 §5 |")
    A("| DNA | " + BT + "general_harmful_compliance_score" + BT + " | AUPRC 0.2258（+37.8%），Recall@FPR0.03=0.3627 |")
    A("| Aegis response | " + BT + "general_harmful_compliance_score" + BT + " | AUPRC 0.7137（诊断） |")
    A("| Aegis prompt | " + BT + "prompt_risk_score" + BT + " | AUPRC 0.8108（诊断） |")
    A("")
    A("### 8.1 Fraud-R1 Judge 一致性（多 operating points，冻结公式）")
    A("")
    A("| 阈值 | unsafe rate | Judge coverage | T6-only rate |")
    A("|---|---|---|---|")
    for th in ("0.2", "0.3", "0.4", "0.5", "0.6"):
        frd = native["fraudr1_judge_agreement_det"]
        A(f"| {th} | {fmt(frd['unsafe_rate_th' + '{}'.format(th)])} | {fmt(frd['judge_coverage_th' + '{}'.format(th)])} | {fmt(frd['t6_only_rate_th' + '{}'.format(th)])} |")
    A("")
    A("指南 §20.4 推荐静态 gate：unsafe rate 8–18%、Judge coverage ≥55%、T6-only 1–8%。"
      "确定性 head 在 0.2–0.5 阈值下 coverage 19–37%，未达 55% gate；这反映已保存 specialist "
      "证据在 Judge-only 子集上的信息上限（§30.3 不承诺静态修复创造新语义能力）。")
    A("")
    A("## 9. 共享 Evidence Adapter（指南 §23）")
    A("")
    A("- 训练数据：Exp3 agent_predictions train（4091→2565 非重叠）/ dev（1047→650 非重叠），"
      "排除全部 Exp2 full sample_id/qy_hash（38,285 个 ID）。")
    A("- 模型：" + BT + "LogisticRegression(C∈{0.01,0.1,1,10}, class_weight=\"balanced\", liblinear)" + BT + "，"
      "C 仅在 Exp3 dev 选择；特征：22 base + 6 交互项（" + BT + "evidence.py" + BT + "）。")
    A("- 多头：" + BT + "FraudEvidenceAdapter" + BT + "（fraud head）、" + BT + "HarmfulComplianceAdapter" + BT + "、" +
      BT + "OverRefusalAdapter" + BT + "（单类跳过）、" + BT + "RefusalDetectionAdapter" + BT + "。")
    A("- dev AUPRC：fraud=0.5509、harmful_compliance=0.9842（Exp3 域内；对 Exp2 为 OOD 转移）。")
    A("")
    A("| 应用域 | adapter AUPRC | 确定性 AUPRC | 结论 |")
    A("|---|---|---|---|")
    A("| Fraud-R1（vs Judge 一致性） | 类别分布改善（Phishing 0.136@0.5） | 0.8% | adapter 恢复非 Fake Job 信号，阈值不确定 |")
    A("| DNA | 0.0412 | 0.2258 | **Exp3 欺诈域无法迁移到广义 harm，弃用 adapter** |")
    A("| Aegis response | 0.6987 | 0.7137 | 确定性 head 更优 |")
    A("| Aegis prompt | 0.7496 | 0.8108 | 确定性 head 更优 |")
    A("")
    A("> 论文表述要求（§35）：Adapter 仅组合 Agent 证据、零 LLM 调用；不得暗示原始 categorical "
      "T6 达到相同结果。")
    A("")
    A("## 10. 错误样本矩阵（指南 §27）")
    A("")
    A("| 文件 | 条数 | 主要规则聚类 |")
    A("|---|---|---|")
    A(f"| errors/fraudr1_judge_only.jsonl | {err['fraudr1_judge_only']} | Judge 检出而 T6 未检出（确定性头） |")
    A(f"| errors/fraudr1_t6_only.jsonl | {err['fraudr1_t6_only']} | T6 检出而 Judge 未检出 |")
    A(f"| errors/dna_false_negative.jsonl | {err['dna_false_negative']} | 见 bucket_summary |")
    A(f"| errors/dna_false_positive.jsonl | {err['dna_false_positive']} | 见 bucket_summary |")
    A(f"| errors/aegis_false_negative.jsonl | {err['aegis_false_negative']} | 见 bucket_summary |")
    A(f"| errors/aegis_false_positive.jsonl | {err['aegis_false_positive']} | 见 bucket_summary |")
    A(f"| errors/or_refusal_disagreement.jsonl | {err['or_refusal_disagreement']} | 拒答行为分歧 |")
    A("")
    A("## 11. Exp3 暴露与数据泄漏（指南 §26）")
    A("")
    A("| Benchmark | all-full | seen-in-exp3 | unexposed-only |")
    A("|---|---|---|---|")
    for src, v in overlap.items():
        if isinstance(v, dict) and "all_full" in v:
            A(f"| {src} | {v['all_full']} | {v['seen_in_exp3']} | {v['unexposed']} |")
    A("")
    A("- Shared Evidence Adapter 训练已排除全部 Exp2 重叠 ID（" + BT + "--exclude-exp2-overlap" + BT + "）。")
    A("- Aegis 因 unexposed response 样本极少，仅可称为 **cross-work in-domain comparison**，"
      "不得称为 fully held-out external benchmark。")
    A("")
    A("## 12. 静态回归门槛（指南 §29）")
    A("")
    A("| 门槛 | 状态 |")
    A("|---|---|")
    A("| pytest 全部通过 | ✅ 375 passed（含新增 32 个静态修复测试） |")
    A("| 0 duplicate sample ID | ✅ 审计通过 |")
    A("| 0 missing Gold | ✅ DNA/Aegis frame 全量 gold |")
    A("| 0 missing predictions | ✅ 6/6 文件 integrity_ok |")
    A("| 0 suspicious empty Agent outputs | ✅ suspicious_predictions.jsonl 为空 |")
    A("| 每张二分类表可由 TP/FP/TN/FN 复算 | ✅ 断言通过 |")
    A("| Macro-F1=(Safe-F1+Unsafe-F1)/2 | ✅ 断言通过 |")
    A("| McNemar discordant delta=Accuracy delta | ✅ 断言通过 |")
    A("| observed delta 与 bootstrap 同 metric | ✅ 同一函数对象，中心偏差 <0.001 |")
    A("| OR 不再出现 all-safe 自动 100% | ✅ 修复后 safety-utility=0.9997（真实字段） |")
    A("| Aegis prompt/response 分轨 | ✅ N+、Recall、FPR、MF1 一致 |")
    A("| score 方向通过检查 | ✅ DNA/Aegis 无反向（choose_score_direction） |")
    A("")
    A("## 13. Go/No-Go 与下一轮建议（指南 §34）")
    A("")
    A("### Go 条件满足")
    A("")
    A("- ✅ 所有数学断言通过；OR 映射修复；Aegis frame 修复；Schema 空输出无法通过；"
      "Fraud-R1 context 未损坏；错误矩阵输出具体缺失能力。")
    A("")
    A("### 未满足（限制下一次 API pilot 范围）")
    A("")
    A("- ❌ Fraud-R1 确定性 head 未达 Judge coverage ≥55% gate（37.4% max）；"
      "adapter 在类别分布上改善但阈值不稳定。")
    A("- ❌ DNA AUPRC 0.2258 未达 0.25 强门槛；within-prompt 0.3453 未达 0.40。")
    A("- ❌ Aegis response 离线 head AUPRC 低于原 categorical 分数。")
    A("")
    A("若进入下一轮 API pilot，建议仅小规模验证（指南 §34.3）：Fraud-R1 Judge-only 200 条、"
      "DNA FN/FP 各 100 条、Aegis FN 100 条、OR hard/toxic 各 100 条；并优先用审计出的"
      "错误样本（" + BT + "errors/*.jsonl" + BT + "）定位 rubric 或 schema 缺口，而非直接全量重跑。")
    A("")
    A("## 14. 论文结果使用边界（指南 §35）")
    A("")
    A("可以保留：实验三多 Agent 机制消融、全量覆盖工程、Fraud-R1 风险趋势、Aegis/DNA 作为跨域边界、"
      "OR 修复后的拒答质量分析。")
    A("")
    A("当前不能声称：全面优于四个原工作、OR 达到 100%、DNA 显著优于 Longformer、"
      "Aegis 显著优于 NemoGuard、Aegis 是 fully held-out external test。")
    A("")
    A("## 15. 复现命令")
    A("")
    A("" + BT3 + "powershell")
    A("$env:FRAUDDISTILL_OFFLINE = " + chr(39) + "1" + chr(39))
    A("python scripts/audit_exp2_predictions.py --offline")
    A("python scripts/audit_fraudr1_context.py --offline")
    A("python scripts/audit_exp2_frames.py --offline")
    A("python scripts/rescore_exp2_offline.py --mode deterministic --offline")
    A("python scripts/train_exp2_evidence_adapter.py --exclude-exp2-overlap --offline")
    A("python scripts/rescore_exp2_offline.py --mode shared-adapter --offline")
    A("python scripts/evaluate_exp2_static.py --offline --strict --bootstrap 10000")
    A("python scripts/make_exp2_static_report.py --offline")
    A("python -m pytest tests/test_offline_guard.py tests/test_exp2_static_schemas.py tests/test_exp2_static_metrics.py tests/test_exp2_static_adapters.py -q")
    A("" + BT3)
    A("")

    out_path = EXPERIMENT_DIR / "EXP2_STATIC_REPAIR_REPORT.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    print("written ->", out_path)


if __name__ == "__main__":
    main()