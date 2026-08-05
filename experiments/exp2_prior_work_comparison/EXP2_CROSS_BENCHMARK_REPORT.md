# EXP2 跨工作对比实验报告（2026-08-05 指南版）

> 自动生成：`scripts/make_exp2_report.py`；数据源：`metrics/canonical_metrics.json`、`metrics/paired_significance.json`、`metrics/native_appendix.json`。无手工复制数字。

## 1. 实验设计概要

- 任务对齐轨道：4 个公开安全基准的同一 `q+y` 样本，统一 Gold 标注为回答行为（fraud_assistance / refusal_failure / over_refusal / safe）。
- 教师：冻结 Exp3 T6 Evidence MAT（Fraud + Refusal + Context specialist + Evidence Arbiter，conflict correction OFF，deepseek-v4-flash）。
- Gold：双盲 LLM 标注（A=deepseek-v4-flash，B=deepseek-v4-pro）+ 第三人仲裁；标注者不可见基准名/基线/教师输出。
- 基线：官方协议复用（Fraud-R1 GPTCheck judge / OR-Bench response checker / Longformer-Harmful / NemoGuard-8B），全部零新增 API。
- 统计：10k 组级 bootstrap、exact McNemar、Holm 校正（4 个 primary 比较）。

## 2. 八行主表（任务对齐轨道）

| Benchmark | Method | N | N+ | Acc | P | Unsafe R | Unsafe F1 | Safe F1 | **Macro-F1** | FPR | AUPRC | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fraud-R1 | Fraud-R1 Official Judge | 800 | 11 | 0.8425 | 0.0741 | 0.9091 | 0.1370 | 0.9133 | **0.4893** | 0.1584 | - | 0.2334 |
| Fraud-R1 | FraudDistill MAT (T6) | 800 | 11 | 0.9862 | 0.0000 | 0.0000 | 0.0000 | 0.9931 | **0.5000** | 0.0000 | 0.0102 | 0.0000 |
| OR-Bench | OR-Bench Response Checker | 800 | 2 | 0.6425 | 0.0000 | 0.0000 | 0.0000 | 0.7823 | **0.3221** | 0.3559 | - | -0.0371 |
| OR-Bench | FraudDistill MAT (T6) | 800 | 2 | 0.9975 | 0.0000 | 0.0000 | 0.0000 | 0.9987 | **0.5000** | 0.0000 | 0.0205 | 0.0000 |
| Do-Not-Answer | Longformer-Harmful | 900 | 59 | 0.8033 | 0.1609 | 0.4746 | 0.2403 | 0.8870 | **0.5334** | 0.1736 | 0.1733 | 0.1886 |
| Do-Not-Answer | FraudDistill MAT (T6) | 900 | 59 | 0.9389 | 1.0000 | 0.0678 | 0.1270 | 0.9683 | **0.5635** | 0.0000 | 0.1997 | 0.2523 |
| Aegis 2.0 | NemoGuard-8B | 813 | 263 | 0.7798 | 0.6050 | 0.9202 | 0.7300 | 0.8141 | **0.7214** | 0.2873 | - | 0.5922 |
| Aegis 2.0 | FraudDistill MAT (T6) | 813 | 263 | 0.8770 | 0.8075 | 0.8137 | 0.8106 | 0.9089 | **0.8589** | 0.0927 | 0.8497 | 0.7195 |

## 3. 成对显著性与 bootstrap

| Benchmark | ΔMacro-F1 | 95% CI | McNemar p | Holm p | b-wrong/t-right | b-right/t-wrong |
|---|---:|---|---:|---:|---:|---:|
| Fraud-R1 | +0.0719 | [+0.0587, +0.0856] | 0.00000 | 0.00000 | 0 | 135 |
| OR-Bench | +0.1775 | [+0.1606, +0.1944] | 0.00000 | 0.00000 | 0 | 284 |
| Do-Not-Answer | +0.0230 | [-0.0111, +0.0560] | 0.00000 | 0.00000 | 1 | 171 |
| Aegis 2.0 | +0.0486 | [+0.0320, +0.0652] | 0.00000 | 0.00000 | 28 | 163 |

## 4. 机制指标（T6 MAT，任务对齐 Gold）

| Benchmark | Direct-fraud R | Trust-facilitation R | Partial-leakage R | Clean-refusal FPR | Hard-safe FPR | Within-prompt pair acc |
|---|---:|---:|---:|---:|---:|---:|
| Fraud-R1 | 0.0000 (n=9) | 0.0000 (n=10) | - | 0.0000 (n=785) | - | - |
| OR-Bench | - | 0.0000 (n=3) | 0.0000 (n=2) | 0.0000 (n=777) | 0.0000 (n=348) | - |
| Do-Not-Answer | 0.1053 (n=19) | 0.0513 (n=39) | 0.1200 (n=25) | 0.0000 (n=748) | - | 0.0642 (n=265) |
| Aegis 2.0 | 0.9044 (n=136) | 0.6716 (n=67) | 0.8810 (n=126) | 0.0803 (n=473) | - | - |

**匹配 FPR/召回（仅 DNA，因基线有连续分数）**：基线 FPR=0.1736 时教师最大召回 0.4576（阈值视角）；基线召回 0.4746 时教师 FPR=1.0000（说明要追上 Longformer 的召回需放低阈值、牺牲 FPR）；AUPRC Δ=+0.0264。

## 5. Native-task 附录（同一 manifest 上的官方标签）

| Benchmark | n | Baseline Acc | Baseline MF1 | T6 Acc | T6 MF1 |
|---|---:|---:|---:|---:|---:|
| Do-Not-Answer | 900 | 0.9722 | 0.9638 | 0.7878 | 0.5145 |
| Aegis 2.0 | 813 | 0.8819 | 0.8787 | 0.7552 | 0.8072 |

## 6. 数据与标注质量

- Manifest：`manifests/test_manifest.jsonl`（fraudr1 800 / orbench 800 / dna 900 / aegis 813，共 3313 条）。
- Exp3 暴露审计：`audit/exp3_exposure_qy_hashes.json`；overlap 数见 `audit/overlap_summary.json`。
- 盲审一致性：见各基准 `human_audit/human_audit_adjudicated_20260805.jsonl` 的 annotator 字段（全量 kappa 记录于审计日志）。
- 教师覆盖率：1.0000；abstain=0；parse_fail=0（见 canonical_metrics.json 各基准）。

### 盲审一致性（A=deepseek-v4-flash vs B=deepseek-v4-pro）
| Benchmark | n | Raw binary | Kappa binary | Raw 4-way | Kappa 4-way |
|---|---:|---:|---:|---:|---:|
| Fraud-R1 | 795 | 0.990 | - | 0.990 | - |
| OR-Bench | 799 | 0.995 | -0.003 | 0.995 | -0.001 |
| Do-Not-Answer | 898 | 0.929 | 0.521 | 0.922 | 0.489 |
| Aegis 2.0 | 773 | 0.887 | 0.745 | 0.788 | 0.577 |

注：fraudr1/orbench 的 Gold 正例率极低（1.4% / 0.3%），kappa 在类别高度不平衡时退化为无信息值（nan/负值），此时 raw agreement（99.0% / 99.5%）更有意义；DNA kappa 0.52、Aegis kappa 0.74/0.58 如实报告。所有分歧均由第三人仲裁，Gold 为单一最终标签（adjudicated 字段标记仲裁行）。

## 7. 成本报告

- 总使用：`16.8076` RMB；上限：`32.00` RMB（36 硬上限 - 4 预留）。
- 教师：`metrics/cost_teacher_t6_test.json`；盲审与仲裁计入共享 `audit/budget_state.json`。
- 基线全部复用历史预测（零 API 成本）；Aegis 794/813 复用 Exp3 冻结 T6 预测。

## 8. 误差分析与分组

- 分歧样本数（基线 vs 教师预测不一致）：`metrics/error_analysis.jsonl`（782 条）。
- 分组指标：`metrics/subgroup_metrics.csv`（language / category / prompt_type / target_model）。

## 9. 冻结与复现

- 冻结 commit：见 `preregistration.md`；教师 prompt hash 记录于每条预测的 `teacher_prompt_hash`。
- 复现命令：`python scripts/run_exp2_teacher.py`；`python -m frauddistill.exp2_cross_benchmark.audit --manifest --annotate --adjudicate --agreement`；`python scripts/evaluate_exp2.py`；`python scripts/make_exp2_report.py`。
