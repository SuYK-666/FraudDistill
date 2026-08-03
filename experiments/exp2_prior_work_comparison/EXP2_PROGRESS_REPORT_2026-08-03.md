# Exp2 跨基准对比实验 · 进度总报告（截至 2026-08-03）

> 本文件为人工汇总的阶段性进度文档；脚本自动生成的正式报告见同目录 `EXP2_CROSS_BENCHMARK_REPORT.md`（Aegis 基线完成后需重新生成，当前为部分结果）。

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

1. 等待 NemoGuard 后台跑完（完成后 `aegis2/baseline_predictions/aegis_nemoguard_predictions.jsonl` 应有 1,964 行唯一 id）。
2. 重跑 `python -m frauddistill.exp2_cross_benchmark.metrics` 与 `make_report`，更新 8 行主表、成对显著性、子组表、图表与 LaTeX 表。
3. 补充正式报告中的审计一致性、偏差说明（Judge 模型非 gpt-4o-mini、空 answer 样本、单正例 F1 失真等）。
4. 检查 git 提交是否完整，必要时补充提交。

## 9. 目录结构速览

- `experiments/exp2_prior_work_comparison/<benchmark>/unified/` — 统一 q+y 数据（冻结）
- `.../<benchmark>/teacher_predictions/` — 教师预测（全部完成）
- `.../<benchmark>/baseline_predictions/` — 各官方基线预测
- `.../<benchmark>/human_audit/` — 抽样、双盲标注、裁决、最终 gold
- `.../_metrics/` — 指标表、显著性、子组、成本、误差分析
- `.../_figures/` — 图表
- `src/frauddistill/exp2_cross_benchmark/` — 全部代码（prepare_data / teacher / baselines / audit / metrics / make_report）
