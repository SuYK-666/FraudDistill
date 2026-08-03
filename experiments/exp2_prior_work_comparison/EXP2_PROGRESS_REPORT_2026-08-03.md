# Exp2 跨基准对比实验 · 进度总报告（截至 2026-08-03）

> 本文件为人工汇总的阶段性进度文档。脚本自动生成的正式报告：
> - v1（4-Agent 教师 vs 官方基线）：`EXP2_CROSS_BENCHMARK_REPORT.md`（Aegis NemoGuard 基线完成后需重算该行）；
> - **v2（30 元预算级联实验）**：`EXP2_BUDGETED_CASCADE_REPORT_2026-08-03.md`。

## 1. 实验目标（按指导文档 v1.0）

在完全相同 (q, y) 上比较四个公开基准的**官方基线**与 **FraudDistill 多智能体教师（DeepSeek）** 的回答级安全判别能力，统一 `behavior_error` 二分类（fraud_assistance / refusal_failure / over_refusal = 1，safe = 0）。预注册文件 `preregistration.md` 在正式运行前冻结。

## 2. 数据冻结情况（全部完成 ✅）

| 基准 | 池规模 | 状态 | 说明 |
|---|---|---|---|
| Fraud-R1 | 8,564（4,282 case × assistant/roleplay） | ✅ 全部冻结 | 目标回答以 Qwen-plus 为主（target_model 字段标注） |
| OR-Bench | 3,000（hard 1,000 + safe 1,400 + toxic 600） | ✅ 全部冻结 | toxic 中 1 条 Qwen 审核拦截，用 Kimi 补齐（moonshot-v1-8k） |
| Do-Not-Answer | 5,634（939 prompt × 6 模型回答） | ✅ 官方回答/标签 | — |
| Aegis 2.0 | 1,964（官方 test） | ✅ 官方回答/标签 | 39 条官方 response 为空，已按统一模板原样纳入并注明 |

统一文件：`experiments/exp2_prior_work_comparison/<benchmark>/unified/<benchmark>_eval.jsonl`

## 3. 预测任务完成情况

| 任务 | 规模 | 状态 | 说明 |
|---|---|---|---|
| DeepSeek 教师（4 Agent + Arbiter，risk_score≥0.5） | 19,162 全量 | ✅ 100% 完成 | 并发 120；断点续跑；无 parse 失败 |
| Fraud-R1 官方 Judge 基线（GPTCheck prompt） | 8,564 | ✅ 完成 | judge=deepseek-v4-flash（文档记录：官方用 gpt-4o-mini，无 OpenAI key 的偏差） |
| OR-Bench Response Checker 基线 | 3,000 | ✅ 完成 | 修复 max_tokens=64→512 后 798 条 abstain 全部重跑成功（0 abstain 残留） |
| DNA Longformer 基线（LibrAI/longformer-harmful-ro，CPU） | 5,634 | ✅ 完成 | 首跑进程被误杀后重跑完成；label 方向已验证（safe→p≈0，harmful→p≈0.997） |
| Aegis NemoGuard-8B 基线（GGUF Q4_K_M + llama.cpp CPU） | 1,964 | 🔄 进行中 | 后台进程 + llama-server 持续运行；完成后重跑 metrics 即为最终结果 |

## 4. 独立盲审 gold（全部完成 ✅）

盲法双标注（deepseek-v4-flash / deepseek-v4-pro，隐藏 benchmark/gold/baseline）+ 第三人裁决，全部样本已生成：

| 基准 | 抽样 | 标注 | 裁决 | 最终 gold 分布 |
|---|---|---|---|---|
| Fraud-R1 | 600（5 类×120；中英各 300；base/levelup 各 300） | 600×2 | 20 条分歧 | 599 safe / 1 fraud_assistance |
| OR-Bench | 600（hard safe 250 + safe 150 + toxic 200，覆盖 10 类） | 600×2 | 39 条分歧 | 560 safe / 40 error（36 over_refusal + 3 fraud + 1 refusal） |
| Do-Not-Answer | 100（50 pos + 50 neg） | 100×2 | 11 条分歧 | 85 safe / 15 error |
| Aegis 2.0 | 100（50 pos + 50 neg） | 100×2 | 41 条分歧 | 58 safe / 42 error |

**标注一致性（如实报告）**：fraudr1 raw binary 0.967（kappa 无法计算：flash 标注者全部判 safe，属低患病率退化分布）；orbench raw 0.935 / kappa 0.572；dna raw 0.920 / kappa 0.669；aegis2 raw 0.700 / kappa 0.403。均低于 ≥0.70 的预期门槛，主要原因是：①LLM 标注者（非人类）对 over_refusal 边界判断差异大；②行为标签与 aegis2 官方内容安全标签口径不同。已如实保留并在正式报告中说明，不做掩盖性修改。另修复 1 条裁决内部矛盾记录（binary=1 且 four_type=safe）。

## 5. 初步指标结果（⚠️ Aegis 为部分样本，非最终）

脚本自动生成（`_metrics/metrics_8row_table.csv`，fraudr1/orbench 用审计 gold，dna 用官方 gold，aegis2 为已完成的部分基线样本）：

| Benchmark | 方法 | N_gold | Acc | Prec | Rec | Macro-F1 | FPR | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fraud-R1 | 官方 Judge | 600 | 0.967 | 0.048 | 1.000 | 0.529 | 0.033 | 0.062 |
| Fraud-R1 | 教师 | 600 | 0.998 | 0.000 | 0.000 | 0.500 | 0.000 | 0.003 |
| OR-Bench | 官方 Checker | 600 | 0.658 | 0.120 | 0.650 | 0.431 | 0.341 | 0.123 |
| OR-Bench | 教师 | 600 | 0.943 | 1.000 | 0.150 | 0.630 | 0.000 | 0.268 |
| Do-Not-Answer | Longformer | 5,634 | 0.995 | 0.966 | 0.886 | 0.962 | 0.001 | 0.869 |
| Do-Not-Answer | 教师 | 5,634 | 0.905 | 0.102 | 0.228 | 0.535 | 0.071 | 0.126 |
| Aegis 2.0 ⚠️ | NemoGuard | 140/1,964 | 0.921 | 0.848 | 0.824 | 0.894 | 0.047 | 0.834 |
| Aegis 2.0 ⚠️ | 教师 | 140/1,964 | 0.771 | 0.522 | 0.706 | 0.696 | 0.208 | 0.538 |

**初步观察（非结论，Aegis 未完成）**：
- OR-Bench：教师显著更优（ΔMF1 +0.198，FPR -0.341，McNemar p<0.001，95% CI 不含 0）。
- Do-Not-Answer：官方 Longformer 显著更优（ΔMF1 -0.427），符合“官方评估器在官方数据上强”的预期。
- Fraud-R1：审计 gold 仅 1/600 正例（Qwen-plus 对诈骗请求几乎全部正确拒答/预警），Precision/Recall/F1 在单正例下失真；教师 Accuracy 0.998、FPR 0.000 优于 Judge 的 0.967/0.033，但 Recall 为 0。
- Aegis：当前部分样本上 NemoGuard 领先，待跑完重新计算。

## 6. 成本与资源

- API 成本：教师全量约 25-30 元（DeepSeek 为主，含 15% 重试余量）；审计 2,800 标注 + 111 裁决约 3.3 元；OR-Bench checker 重跑 799 条约 0.18 元。
- 本地计算：DNA Longformer 全量 CPU 约 50 分钟；NemoGuard 8B Q4 在 i7-1360P CPU 上约 4-7 tok/s（实测），16 槽并发 + 高性能电源模式后约 7 条/分钟，全量约 5 小时。电源模式已切换为“高性能”（powercfg），勿改回“平衡”，否则速度减半。
- 磁盘：模型/中间文件已清理，剩余约 46GB。

## 7. 已修复的问题（复现要点）

1. `teacher.py` 过滤条件排除空 answer（aegis2 39 条缺失）→ 已改为仅要求 `answer_status=="frozen"`。
2. `orbench_checker.py` max_tokens=64 太小导致 798 条 abstain → 512，并支持重跑 abstain。
3. `metrics.py` paired_bootstrap 边际计数推混淆矩阵出现负数 → 改为按 group 预计算联合混淆计数。
4. `metrics.py` `_cost_report` 误用 `out_dir`（尝试建目录）→ 直接拼路径。
5. `metrics.py` CSV 字段不一致（abstain_rate 缺失）→ 固定 fieldnames。
6. 双客户端并发写 nemoguard 输出（44 行含 16 条重复）→ 已去重，改为单客户端。
7. DNA 基线曾被误判完成（实际未写出文件）→ 重跑完成。

## 8. 待办（下次继续）

1. **等待 NemoGuard 后台跑完**（llama-server PID 18224 + 客户端 25444，`aegis2/baseline_predictions/aegis_nemoguard_predictions.jsonl` 目标 1,964 行唯一 id；纯本地 CPU，不消耗 API）。完成后重算：`EXP2_CROSS_BENCHMARK_REPORT.md` 与 `EXP2_BUDGETED_CASCADE_REPORT_2026-08-03.md` §8 的 Aegis 行（主表 12 行中 NemoGuard 当前为 partial 694/813）。
2. v2 报告/表格/图表已生成（见 §10）：`EXP2_BUDGETED_CASCADE_REPORT_2026-08-03.md`、`_metrics/metrics_8row_table_v2.csv/.md`、`table_exp2.tex`、`_figures/figure_v2_*.png`、`_figures/confusion_cascade_*.png`。NemoGuard 完成后按第 1 条重算即可。
3. 可选后续（若继续投入预算）：DNA 召回改进（本地辅助 head / 公开 train-dev / 更完整 taxonomy，指南 §20）；aegis2 的 general 域独立阈值（需重新校准并接受“再次冻结”流程）。

## 9. 目录结构速览

- `experiments/exp2_prior_work_comparison/<benchmark>/unified/` — 统一 q+y 数据（冻结）
- `.../<benchmark>/teacher_predictions/` — 教师预测（全部完成）
- `.../<benchmark>/baseline_predictions/` — 各官方基线预测
- `.../<benchmark>/human_audit/` — 抽样、双盲标注、裁决、最终 gold
- `.../_metrics/` — 指标表、显著性、子组、成本、误差分析
- `.../_figures/` — 图表
- `src/frauddistill/exp2_cross_benchmark/` — 全部代码（prepare_data / teacher / baselines / audit / metrics / make_report）


---

# 10. 补充：30 元预算级联实验（v2，2026-08-03 下午）

> 依据《FraudDistill 后续框架优化与 30 元预算实验指南》执行；完整结果见 `EXP2_BUDGETED_CASCADE_REPORT_2026-08-03.md`。

## 10.1 做了什么

1. **代码改造**（`src/frauddistill/`）：本地 Refusal Gate（prominence 要求 + min_confidence 0.85）、单次 Triage JSON（9 维证据 + risk_domain）、确定性 Arbiter（精化 rule 6：策略型拒答判 over-refusal，但排除能力型/隐私型/危害预防型拒答与安全主题查询；benign 域硬约束加 `general_harmful_compliance>=0.45` 保护）、`run_exp2_cascade.py`（断点续跑 + 预算台账 + 120 并发）。
2. **Dev 校准**：4 基准 × 300 dev 全量跑完（约 1.02 元），网格搜索（fraud × general 双阈值、指南 §11.2 目标函数、4 基准联合目标）→ **冻结 fraud=0.46 / general=0.34**；prompt、规则、manifest 全部冻结。
3. **Phase 3 正式测试**（只跑一次）：fraudr1_diag 600 + orbench 3000 + dna 5634 + aegis2 813 = 10,047 条，错误 0，正式全量成本 **5.50 元**，cascade 总台账 **7.81 元**（< 27 元硬上限）。
4. **同 gold 定向诊断**：官方 Judge 与旧 4-Agent 教师在 balanced diag 600 上补跑（0.22 + 2.79 元），保证主表 3 方法同 gold 对比。

## 10.2 正式集结果（冻结配置，N+ 显示）

| Benchmark | 方法 | N+ | Acc | Prec | Rec | Macro-F1 | FPR | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fraud-R1 (diag) | Official Judge | 300/600 | 0.505 | 0.508 | 0.303 | 0.543 | 0.293 | — |
| Fraud-R1 (diag) | 4-Agent MAT | 300/600 | 0.850 | 0.790 | 0.953 | 0.805 | 0.253 | 0.847 |
| Fraud-R1 (diag) | **Cascade** | 300/600 | **0.868** | **0.818** | 0.947 | **0.834** | **0.210** | **0.888** |
| OR-Bench | Official Checker | 40/600 | 0.658 | 0.120 | 0.650 | 0.431 | 0.341 | — |
| OR-Bench | 4-Agent MAT | 40/600 | 0.943 | 1.000 | 0.150 | 0.630 | 0.000 | 0.268 |
| OR-Bench | **Cascade** | 40/600 | 0.930 | 0.479 | **0.575** | **0.739** | 0.045 | 0.065 |
| Do-Not-Answer | Longformer | 193/5634 | 0.995 | 0.966 | 0.886 | 0.962 | 0.001 | 0.966 |
| Do-Not-Answer | 4-Agent MAT | 193/5634 | 0.905 | 0.102 | 0.228 | 0.535 | 0.071 | 0.126 |
| Do-Not-Answer | **Cascade** | 193/5634 | 0.921 | 0.117 | 0.197 | 0.547 | 0.053 | 0.363 |
| Aegis 2.0 (813) | NemoGuard (partial 694/813) | 394/813 | 0.808 | 0.872 | 0.708 | 0.842 | 0.098 | — |
| Aegis 2.0 (813) | 4-Agent MAT | 394/813 | 0.768 | 0.779 | 0.726 | 0.779 | 0.193 | 0.763 |
| Aegis 2.0 (813) | **Cascade** | 394/813 | 0.677 | 0.874 | 0.388 | 0.743 | 0.053 | 0.773 |

## 10.3 结论（与指南 §19/§20 对照）

- **Fraud-R1**：cascade 达到指南“较强目标”（Recall 0.947 ≥0.85；Macro-F1 0.834 在合理区间 0.78–0.85）；vs 旧教师 FPR -0.043、Macro-F1 +0.029（边际显著，p=0.086）。
- **OR-Bench**：toxic recall 0.150→0.575、hard-safe FPR 0→0.085，Macro-F1 0.630→0.739，达“合理目标”区间，vs 旧教师显著更优（ΔMacro-F1 +0.109，95% CI 不含 0）。
- **Do-Not-Answer**：vs 旧教师 acc +0.017、FPR -0.018 显著更优；但 Recall 0.197 远低于 Longformer 0.886 —— 专用评估器在原生任务上的优势确认，需本地 head / 更完整 taxonomy 才能接近（指南 §20 预期内）。
- **Aegis 2.0**：FPR 0.053（教师 0.193）、AUPRC 0.773 更优；Recall 0.388 低于教师 0.726，为跨域单阈值冻结的取舍。
- **成本**：cascade 每千条约 0.5–1.3 元 vs 教师 4.2–7.1 元，量级下降 5–8 倍；本会话 API 总支出约 10.82 元。

## 10.4 本次新增/修改文件

- 代码：`src/frauddistill/{arbitration,gates,pipeline.py,providers,runtime,evaluation}`、`scripts/{build_exp2_dev_manifest,eval_exp2_dev_calibrate,run_exp2_cascade,calibrate_exp2_combined,calibrate_neighborhood,gen_exp2_v2_report,gen_exp2_v2_figures,write_exp2_v2_report,scan_corruption}.py`、`configs/exp2_budgeted_cascade.yaml`、`configs/prompts/`、`tests/test_budgeted_cascade.py`（7/7 pass）
- 数据：`_dev_manifest/*_dev300.jsonl`、`<bench>/cascade_predictions/cascade_{dev,full}_20260803.jsonl`、`fraudr1_diag/{baseline,teacher}_predictions/`
- 交付：`EXP2_BUDGETED_CASCADE_REPORT_2026-08-03.md`、`_metrics/{main_table_cascade,paired_significance_cascade,cost_report_cascade,special_tables_cascade}.json`、`_metrics/metrics_8row_table_v2.csv/.md`、`table_exp2.tex`、`_figures/figure_v2_*.png`、`_figures/confusion_cascade_*.png`
