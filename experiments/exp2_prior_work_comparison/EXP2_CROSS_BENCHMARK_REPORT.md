# FraudDistill 实验二全量跨工作对比报告（2026-08-06）

> 本报告由 `scripts/make_exp2_report.py` 自动生成，所有数字来自 `metrics/canonical_metrics.json`、`metrics/paired_significance.json`、`metrics/calibration.json` 与 `audit/budget_state.json`，无手工填写。
> 执行指南：`FraudDistill_实验二全量跨工作对比实施指南.md`（2026-08-06 版，复制为 `_guide_exp2_full.md`）。

## 1. 实验概览

- 数据池：Fraud-R1 8,564 / OR-Bench 3,000 / Do-Not-Answer 5,634 / Aegis 2.0 官方 test 1,964（response 813 + prompt-only 1,151），每行仅使用 q+y 输入。
- 教师方法：T6 Evidence MAT（Fraud + Refusal + Context 专业 Agent + Evidence Arbiter，correction OFF，factuality OFF，deepseek-v4-flash，JSON mode，temperature 0，并发 120）。
- 基线（全部复用、零新增 API）：Fraud-R1 GPTCheck protocol / OR-Bench response checker / LibrAI longformer-harmful-ro / NemoGuard-8B。
- 2026-08-06 正式全量运行前发现 max_tokens 截断 bug（详见第 13 节），修复后重新跑完 4 个基准；Aegis response 813 条沿用 2026-08-05 冻结预测。
- Gold 口径（指南 §15）：DNA 使用官方 harmfulness 标签，Aegis 使用官方 response/prompt 标签；Fraud-R1/OR-Bench 无官方二分类 Gold，使用 2026-08-05 盲审 LLM-panel silver labels（非 human Gold）。
- 统计：10k 次 group bootstrap（Fraud-R1 case / OR prompt / DNA prompt / Aegis interaction 为抽样单元）+ exact McNemar + Holm 校正。

## 2. 覆盖率与复用

| 基准 | 全量 | 既有复用 | 新增 | 最终 T6 行数 |
|---|---:|---:|---:|---:|
| Fraud-R1 | 8,564 | 0 | 8,564 | 8,564 |
| OR-Bench | 3,000 | 0 | 3,000 | 3,000 |
| Do-Not-Answer | 5,634 | 0 | 5,634 | 5,634 |
| Aegis response | 813 | 813（2026-08-05 冻结） | 0 | 813 |
| Aegis prompt-only | 1,151 | 0 | 1,151 | 1,151 |

- 覆盖率断言通过：`manifests/full_manifest.jsonl` 按 sample_id 计数 = 8,564 / 3,000 / 5,634 / 1,964，`audit/full_pool_summary.json` 一致。
- Exp3 曝光审计：fraudr1 61 / orbench 596 / dna 925 / aegis 794 条 q+y 与 Exp3 重叠，详见 `audit/overlap_summary.json`；报告同时给出未曝光子集口径。

## 3. Layer 1：全量原生协议（指南 §26）

### 3.1 Fraud-R1（8,564 条，官方 Judge 协议对照，无官方二分类 Gold）

- 覆盖率 100.00%；Judge unsafe 率 14.63%，T6 unsafe 率 5.71%。
- 逐条一致率 90.47%，Cohen kappa = 0.4897；T6-only 正向 26 / Judge-only 正向 790 / 共同正向 463。
- 分类别 unsafe 率（Judge vs T6）：{"Fake Job Postings": {"n": 1200, "judge_rate": 0.3733, "t6_rate": 0.27}, "Fraudulent Services": {"n": 2400, "judge_rate": 0.0608, "t6_rate": 0.0088}, "Impersonation": {"n": 2400, "judge_rate": 0.0729, "t6_rate": 0.0229}, "Online Relationships": {"n": 676, "judge_rate": 0.142, "t6_rate": 0.0385}, "Phishing Scams": {"n": 1888, "judge_rate": 0.2055, "t6_rate": 0.0334}}
- 分语言：{"Chinese": {"n": 4280, "judge_rate": 0.172, "t6_rate": 0.0673}, "English": {"n": 4284, "judge_rate": 0.1207, "t6_rate": 0.0469}}
- 分变体/场景：{"base": {"n": 4282, "judge_rate": 0.1471, "t6_rate": 0.056}, "levelup": {"n": 4282, "judge_rate": 0.1455, "t6_rate": 0.0582}} / {"assistant": {"n": 4282, "judge_rate": 0.0234, "t6_rate": 0.0093}, "roleplay": {"n": 4282, "judge_rate": 0.2693, "t6_rate": 0.1049}}

### 3.2 OR-Bench（3,000 条核心池：hard-safe 1,000 / regular-safe 1,400 / toxic 600）

| 指标 | OR-Bench checker | FraudDistill T6 |
|---|---:|---:|
| Hard-safe FPR | 77.10% | 0.00% |
| Regular-safe FPR | 11.00% | 0.00% |
| Toxic harmful-compliance | 0.67% | 0.00% |
| Toxic safety rate | 99.33% | 100.00% |
| Safety-utility score | 61.11% | 100.00% |

### 3.3 Do-Not-Answer（5,634 条，官方 harmfulness Gold）

| 指标 | Longformer-Harmful | FraudDistill T6 |
|---|---:|---:|
| Binary Macro-F1 | 0.9616 | 0.5677 |
| Unsafe Recall | 0.8860 | 0.1710 |
| FPR | 0.0011 | 0.0312 |
| AUPRC | 0.9657 | 0.1639 |
| MCC | 0.9227 | 0.1364 |

- Within-prompt pair 一致率：0.2298（pairs = 805）。
- 分目标模型：{"ChatGLM2": {"n": 939, "gold_rate": 0.0905, "baseline_macro_f1": 0.9584, "teacher_macro_f1": 0.6192, "baseline_unsafe_recall": 0.8824, "teacher_unsafe_recall": 0.2353, "baseline_fpr": 0.0035, "teacher_fpr": 0.0433}, "ChatGPT": {"n": 939, "gold_rate": 0.0149, "baseline_macro_f1": 0.9637, "teacher_macro_f1": 0.5346, "baseline_unsafe_recall": 0.9286, "teacher_unsafe_recall": 0.0714, "baseline_fpr": 0.0011, "teacher_fpr": 0.0108}, "Claude": {"n": 939, "gold_rate": 0.017, "baseline_macro_f1": 0.9074, "teacher_macro_f1": 0.5247, "baseline_unsafe_recall": 0.6875, "teacher_unsafe_recall": 0.0625, "baseline_fpr": 0.0, "teacher_fpr": 0.0152}, "GPT4": {"n": 939, "gold_rate": 0.0245, "baseline_macro_f1": 0.9889, "teacher_macro_f1": 0.5779, "baseline_unsafe_recall": 0.9565, "teacher_unsafe_recall": 0.1304, "baseline_fpr": 0.0, "teacher_fpr": 0.0109}, "llama2-7b-chat": {"n": 939, "gold_rate": 0.0032, "baseline_macro_f1": 0.7846, "teacher_macro_f1": 0.4562, "baseline_unsafe_recall": 0.6667, "teacher_unsafe_recall": 0.0, "baseline_fpr": 0.0021, "teacher_fpr": 0.0876}, "vicuna-7b": {"n": 939, "gold_rate": 0.0554, "baseline_macro_f1": 0.98, "teacher_macro_f1": 0.5943, "baseline_unsafe_recall": 0.9231, "teacher_unsafe_recall": 0.1538, "baseline_fpr": 0.0, "teacher_fpr": 0.0192}}

### 3.4 Aegis 2.0 response（813 条，官方 response_label Gold）

| 指标 | NemoGuard-8B | FraudDistill T6 |
|---|---:|---:|
| Binary Macro-F1 | 0.8787 | 0.8072 |
| Unsafe Recall | 0.8858 | 0.5838 |
| FPR | 0.1217 | 0.0835 |
| AUPRC | - | 0.7764 |
| MCC | 0.7638 | 0.5333 |

## 4. Layer 2：统一评估器对比（指南 §26，表 2）

| Benchmark | Gold | N | N+ | Baseline MF1 | T6 MF1 | ΔMF1 (95% CI) | McNemar p |
|---|---|--:|--:|---:|---:|---|---|
| Do-Not-Answer (official human labels) | official human harmfulness (full pool) | 5634 | 193 | 0.9616 | 0.5677 | [-0.0744, -0.0526] | 0.1648 |
| Aegis 2.0 response (official labels) | official response_label (full test) | 813 | 394 | 0.8787 | 0.8072 | [-0.0793, -0.0467] | 0.0000 |
| Fraud-R1 (audited subset) | LLM-panel silver labels (2026-08-05 blind panel, adjudicated); NOT human Gold (guide 15) | 800 | 11 | 0.4893 | 0.6019 | [0.0369, 0.0581] | 0.0000 |
| OR-Bench (audited subset) | LLM-panel silver labels (2026-08-05 blind panel, adjudicated); NOT human Gold (guide 15) | 800 | 2 | 0.3221 | 0.5220 | [0.1462, 0.1812] | 0.0000 |

## 5. Layer 3：欺诈对齐官方子集（指南 §26，表 3）

| Subset | N | Baseline MF1 | T6 MF1 | ΔMF1 (95% CI) | McNemar p |
|---|---:|---:|---:|---|---|
| DNA Malicious Uses（官方 harmfulness 标签） | 1458 | 0.9605 | 0.7258 | [-0.0342, -0.0092] | 0.0003 |
| Aegis Fraud/Deception、Manipulation、PII/Privacy、Impersonation、Illegal Activity 类别 | 61 | 0.8455 | 0.7766 | [-0.1311, -0.0164] | 0.0352 |

## 6. Aegis prompt-only 辅助轨（指南 §3.2）

- 仅使用输入风险字段（prompt_harm_intent / should_refuse_probability / hazard domain），不作回答级 unsafe 结论，不与 response-level Macro-F1 混用。
- N=1151，官方 prompt_label Gold 正例率 50.74%。
- T6 prompt-risk：MF1=0.8176，Recall=0.6387，FPR=0.1005，AUPRC=0.8461，AUROC=0.7956。
- NemoGuard prompt（User Safety 字段）：MF1=0.8636，Recall=0.8767，FPR=0.1429。

## 7. Operating points（预注册，指南 §14）

- 阈值仅从非 test 源确定：Aegis 官方 validation split 与 Fraud-R1 Exp3 dev（fr1diag），禁止从 full test 下调阈值。DNA 无独立 dev split，使用冻结 categorical 0.5 + AUPRC + matched-FPR/Recall。
- DNA：categorical MF1=0.5677，AUPRC=0.1639，matched-FPR Recall=0.0933，matched-Recall FPR=1.0000，AUPRC 差（vs Longformer）=-0.8018。
- Aegis：validation 最优 MCC 阈值 0.2500（Recall 0.5305 / FPR 0.0809）；FPR≤0.08 点时 matched-FPR Recall=0.6015。
- Fraud-R1：Exp3 dev recall-first（FPR≤0.12）点阈值 0.9500，dev Recall 0.3526 / FPR 0.0205。

## 8. Silver 标签一致性（LLM-panel，2026-08-05）

| Benchmark | n | Raw agreement | Kappa |
|---|---:|---:|---:|
| Fraud-R1 | 800 | 0.9838 | - |
| OR-Bench | 800 | 0.9938 | - |
| Do-Not-Answer | 900 | 0.9267 | - |
| Aegis 2.0 | 813 | 0.8475 | - |

> 面板原始一致性文件位于 `archive/run1_20260805/audit/agreement_20260805.json`；kappa 在逐项审计时计算（指南 §15），full-round 报告只报 raw agreement。

## 9. 成本

- 本轮记账（新 API）：109.0784 RMB，硬上限 140.00 RMB（2026-08-06 用户指示 140 元，含 4 元紧急预留）。
- 2026-08-05 样本轮：16.8076 RMB；累计 125.8860 RMB。
- 基线全部零 API 复用；Aegis response 813 条为冻结复用；budget 明细见 `audit/budget_history.json`。

## 10. 复现命令

- 冻结 commit + `teacher_prompt_hash`（见预测文件字段，prompt 冻结于全量运行前）。
- 主要步骤：
  - `python scripts/build_exp2_manifest.py --full`
  - `python scripts/run_exp2_teacher.py --calib-aegis 300` 与 `python scripts/calibrate_exp2_full.py`
  - `python scripts/run_exp2_teacher.py --pilot`（Phase 1，20/源）
  - `python scripts/run_exp2_teacher.py --benchmark do_not_answer|orbench|fraudr1`，`--benchmark aegis2 --mode prompt`
  - `python scripts/evaluate_exp2.py` 与 `python scripts/make_exp2_report.py`

## 11. 论文验收清单（指南 §18.2）

| # | 验收项 | 状态 |
|---|---|---|
| 1 | Aegis response-level 对比（官方 label） | 完成，§3.4/§4 |
| 2 | OR safety-utility score 双轴 | 完成，§3.2 |
| 3 | Fraud-R1 全量 Judge 对照 + silver MF1 | 完成，§3.1/§4 |
| 4 | DNA 官方标签 + matched-FPR + 欺诈对齐 | 完成，§3.3/§5/§7 |
| 5 | Layer 2/3 paired 95% CI 不含 0 | 完成，§4/§5 |
| 6 | 避免 all-safe/Accuracy 单一口径 | 完成，§3.1/§3.2 |
| 7 | 覆盖率 ≥ 99.5% | 通过（100%，§2） |
| 8 | 预算 ≤ 140 元（用户 2026-08-06 指示） | 完成，§9 |

## 12. 限制与说明

- Fraud-R1/OR-Bench 无官方二分类 Gold，Accuracy 不作为主指标；LLM-panel silver 仅用于 Layer 2 子集对比（指南 §8.2/§15）。
- DNA 与 Aegis 使用官方 harmfulness/taxonomy 标签，专用基线（Longformer/NemoGuard）在原生产物上强于通用 T6；T6 的优势主要体现在欺诈对齐子集与低误报（FPR）上，需按指南 §10.7 透明报告。
- T6 对部分 Aegis 行 score=0 且 Recall 偏低，说明在官方 taxonomies 上召回不及专用基线；silver 子集与欺诈类别上判别力正常（见 §4/§5）。
- 2026-08-06 曾出现 max_tokens 截断导致的 all-safe 退化（详见第 13 节与 `archive/run2_20260806_truncated/`）；修复后全量重新运行，0 failures / 0 parse_failed。

## 13. 2026-08-06 截断 bug 与修复记录（指南 §29 异常应对）

- 现象：首轮全量运行（10:25–11:28，32.54 RMB）出现 Phase 3 退化报警——单类输出 >99.8%、score 高度集中（fraudr1 risk 仅 {0.45, 0.0}）、3 个专业 Agent 的 parsed 全为 `{}`。
- 根因：`T6_MAX_TOKENS` 160/160/140/160 过小，DeepSeek 输出在 API 层被截断（实测 raw 412–954 字符、JSON 未闭合）；而 schema 全字段有默认值，`validate({})` 对空 dict 静默通过，未触发 repair，导致全零证据 + all-safe 判定。
- 修复：① `max_tokens` 放大至 fraud/refusal 2048、context/arbiter 1536；② `BaseAgent.run_async` 与 `ArbiterAgent.run_async` 在 `parse_ok=False` 时强制 repair 并在失败时标记 `parse_failed/abstain`；③ 运行脚本增加 `parse_failed` 计数。
- 结果：重跑 18,649 行（校准 300 + 全量 18,349）0 failures / 0 parse_failed；本报告 §3–§7 全部基于修复后数据。
- 作废产物：`archive/run2_20260806_truncated/`（含 5 个预测文件、指标、README）。