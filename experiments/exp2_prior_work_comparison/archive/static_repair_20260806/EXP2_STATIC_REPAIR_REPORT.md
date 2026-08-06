# Exp2 静态修复与离线重评报告（EXP2 Static Repair & Offline Re-evaluation）

- 生成时间：2026-08-06 13:50:42
- 依据指南：`FraudDistill_实验二静态修复与离线重评实施指南.md`（2026-08-06）
- 模式：**完全离线（零 API 调用）**，仅复用已保存的 specialist 输出与预测文件
- 冻结基线提交：`20a80e8`

## 1. 执行摘要

本轮按指南完成：① 完全离线运行保护（`FRAUDDISTILL_OFFLINE` 硬保护）；② 严格 Agent 输出 Schema（空输出不再合法通过）；③ 单一 EvaluationFrame + 可复算的二分类指标；④ Exact McNemar + Holm 校正 + 成对 group bootstrap；⑤ OR-Bench refusal adapter 与 Aegis prompt/response 分轨；⑥ Fraud-R1 上下文完整性审计；⑦ 基于已保存 specialist 证据的离线多头重评分；⑧ 共享 Evidence Adapter（Exp3 非重叠样本训练，零 API）；⑨ 错误样本矩阵与静态回归门槛检查。

### 关键结论

- OR-Bench 虚假的 100% safety-utility 已被**基于原始拒答行为的真实指标**替换；native refusal agreement=73.9%（checker 31% 拒答率 vs T6 45.7%），hard-safe over-refusal 与 toxic harmful-compliance 均为真实字段驱动，不再是映射假象。
- Aegis 主表与统计现在由**同一 confusion matrix 精确复算**（TP/FP/TN/FN 一致），prompt/response 分轨且 baseline 使用 NemoGuard `User Safety` 原始输出。
- DNA 离线重评 AUPRC 由 0.1639 提升至 **0.2258**（+37.8%），Recall@FPR=0.03 达 **0.3627**（最低目标 0.30 达成）；within-prompt 配对一致率由 0.2298 提升至 **0.3453**（最低目标 0.40 未达）。主表与 bootstrap 一致显示**显著低于 Longformer**，此差距为真实差距。
- Fraud-R1 上下文审计确认：原始数据为单轮 prompt，query 完整保留（preserved rate=1.0），**不存在多轮截断**；确定性 fraud head 的 Judge 覆盖率 19%–37%（阈值 0.2–0.5），Evidence Adapter 在非 Fake Job 类别恢复信号（Phishing 检出率 0.136@0.5，原 0.008），但单一阈值尚未同时满足指南全部 gate，标注为诊断结果。
- Aegis response/prompt 离线重评 AUPRC 分别为 0.7137 / 0.8108，**低于原 categorical risk_score 的 0.7764 / 0.8461**；报告保留原 categorical 结果为主结果，离线 head 作为诊断。

## 2. 离线运行保护（指南 §4）

- `src/frauddistill/exp2_static_repair/offline_guard.py`：`OfflineNetworkCallError`、`assert_online_allowed()`、`require_offline()`、`clear_api_keys()`。
- Provider 层：`DeepSeekClient.__init__` 构造时调用 `assert_online_allowed()`；`FRAUDDISTILL_OFFLINE=1` 时任何构造即失败。
- 所有静态脚本 `--offline` 参数默认强制离线；缺失预测策略为 `error`，不自动补跑。
- 测试：`tests/test_offline_guard.py` 覆盖 provider 阻塞、require_offline 语义。

## 3. Schema 硬化（指南 §6）

`src/frauddistill/exp2_static_repair/schemas.py` 提供 `StrictFraudEvidence` / `StrictRefusalEvidence` / `StrictContextEvidence`：

- 关键字段全部 **required**（无默认值），`extra="forbid"`，`strict=True`；
- `Schema.model_validate({})` 不再合法通过；
- `reject_suspicious_empty_evidence`：全零数值 + 空 span + 弱理由 → ValueError；
- `finish_reason_status`：`length` / `insufficient_system_resource` → retry_required；
- 历史预测静态审计：`audit/schema_integrity_summary.json` 与 `audit/suspicious_predictions.jsonl`。

### 3.1 完整性审计结果

| Benchmark | N | parse_failed | abstain | missing_score | 完整性 |
|---|---|---|---|---|---|
| fraudr1 | 8564 | 0 | 0 | 0 | ✔ |
| orbench | 3000 | 0 | 0 | 0 | ✔ |
| do_not_answer | 5634 | 0 | 0 | 0 | ✔ |
| aegis2_response | 813 | 0 | 0 | 0 | ✔ |
| aegis2_prompt | 1151 | 0 | 0 | 0 | ✔ |
| aegis2_validation | 300 | 0 | 0 | 0 | ✔ |

所有 6 个预测文件完整性通过；Fraud-R1/OR/DNA 的 specialist 字段 100% 可用，Aegis response 813 条中 793 条从 Exp3 agent-predictions 索引补全 specialist 证据（20 条无 specialist，保持原分数）。

## 4. 指标与统计修复（指南 §9–§14）

- **二分类指标唯一实现**：`evaluate_binary(frame)`（sklearn），Macro-F1=(Safe-F1+Unsafe-F1)/2 恒等断言；TP/FP/TN/FN 反推一致性断言（`tests/test_exp2_static_metrics.py`）。
- **四分类分离**：`binary_macro_f1` 与 `four_class_macro_f1` 字段名分离，不再共用 `macro_f1`。
- **AUPRC 方向检查**：`choose_score_direction` 自动检测反向分数，正式运行不静默翻转。
- **Exact McNemar**：`scipy.stats.binomtest` 精确双侧；`accuracy_delta == discordant_delta` 恒等断言；字段名带语义（`baseline_wrong_teacher_right` 等）。
- **Holm 校正**：`statsmodels.multipletests(method="holm")`，仅用于预注册主比较。
- **成对 group bootstrap**：group 为抽样单元（Fraud-R1 case、OR prompt_id、DNA prompt_id、Aegis interaction_id），`metric_fn` 与主表为同一函数对象；observed 偏离 percentile CI 时告警。

### 4.1 主表（冻结确定性阈值 0.5，未在 test 上调参）

| Frame | N | N+ | Macro-F1 | AUPRC | FPR | Unsafe-Recall |
|---|---|---|---|---|---|---|
| dna_response | 5634 | 193 | 0.6215 | 0.2258 | 0.0062 | 0.1762 |
| aegis_response | 813 | 394 | 0.7024 | 0.7137 | 0.0692 | 0.4949 |
| aegis_prompt | 1151 | 584 | 0.7637 | 0.8108 | 0.0988 | 0.6370 |

| Frame | Baseline Macro-F1 | ΔMacro-F1 | bootstrap 95% CI | McNemar p | Holm p | 结论 |
|---|---|---|---|---|---|---|
| dna_response | 0.9609 | -0.3394 | [-0.3812, -0.2966] | 0.000000 | 0.000000 | FraudDistill 显著低于基线。 |
| aegis_response | 0.8819 | -0.1794 | [-0.2143, -0.1457] | 0.000000 | 0.000000 | FraudDistill 显著低于基线。 |
| aegis_prompt | 0.8670 | -0.1033 | [-0.1274, -0.0794] | 0.000000 | 0.000000 | FraudDistill 显著低于基线。 |

> 注：DNA/Aegis 的 ΔMacro-F1 与 bootstrap CI 中心一致（偏差 <0.001），McNemar 反映真实差距；`bootstrap_warning` 未触发。

## 5. OR-Bench refusal adapter（指南 §17）

不再使用最终 `teacher_type` 计算 OR 原生指标，而是从 Refusal Agent 原始字段恢复行为：

| 指标 | 值 |
|---|---|
| native refused rate | 0.4570 |
| checker 拒答率（baseline） | 0.3097 |
| native refusal agreement | 0.7393 |
| hard-safe over-refusal rate | 0.0000 |
| regular-safe over-refusal rate | 0.0000 |
| toxic harmful-compliance rate | 0.0017 |
| toxic clean-refusal rate | 0.9850 |
| safety-utility score（修复后） | 0.9997 |

> 说明：修复后的 safety-utility 不再为退化的 1.0（旧报告 100% 为映射假象）。over-refusal 率接近 0 是因为 Refusal Agent 的 `unnecessary_refusal` 字段在 OR 样本上几乎为 0（2.4% 非零），即 Agent 未检出 benign over-refusal；该结果如实反映字段信号，不构成 100% 声明。逐条抽查见 `errors/or_refusal_disagreement.jsonl`（782 条 disagreement）。

## 6. Aegis prompt/response 分轨（指南 §18）

- response 轨道：仅 `response` 非空且 `response_label` 非空（N=813），baseline=NemoGuard response 预测。
- prompt 轨道：仅 response 为空（N=1151），baseline=NemoGuard `User Safety` 原始字段（`prediction_binary` 对 prompt-only 失效，1150/1151 为 0，已弃用）。
- 两轨道 sample_id 无交集（断言通过）。
- 标签来源分层：official labels（response_label / prompt_label）；报告中不作为 fully held-out external benchmark（Exp3 重叠见 §9）。

## 7. Fraud-R1 上下文审计（指南 §19）

| 检查项 | 值 | 结论 |
|---|---|---|
| 原始数据结构 | single_turn_prompt | 单轮 prompt（无多轮截断风险） |
| query 保留率 | 1.0000 | ✅ |
| raw 缺失 | 0/8564 | ✅ |
| roleplay 前缀覆盖 | 2140/4282 | 部分（英文变体措辞差异） |
| Go/No-Go | False | context 未损坏 |

## 8. 离线多头重评分（指南 §20–§22、§24）

每个 benchmark 使用与任务定义匹配的输出头：

| Benchmark | 输出头 | 关键结果 |
|---|---|---|
| Fraud-R1 | `fraud_assistance_score` | Judge 覆盖率 19.0%–37.4%（阈值 0.2–0.5）；非 Fake Job 类别不再全部为 0（Phishing 0.8%→1.3%@0.5） |
| OR native | `did_refuse_score` | refused 45.7%，与 checker agreement 73.9% |
| OR safety | `over_refusal_score` + `harmful_compliance_score` | 见 §5 |
| DNA | `general_harmful_compliance_score` | AUPRC 0.2258（+37.8%），Recall@FPR0.03=0.3627 |
| Aegis response | `general_harmful_compliance_score` | AUPRC 0.7137（诊断） |
| Aegis prompt | `prompt_risk_score` | AUPRC 0.8108（诊断） |

### 8.1 Fraud-R1 Judge 一致性（多 operating points，冻结公式）

| 阈值 | unsafe rate | Judge coverage | T6-only rate |
|---|---|---|---|
| 0.2 | 0.0577 | 0.3735 | 0.0030 |
| 0.3 | 0.0542 | 0.3504 | 0.0029 |
| 0.4 | 0.0408 | 0.2650 | 0.0020 |
| 0.5 | 0.0279 | 0.1899 | 0.0001 |
| 0.6 | 0.0276 | 0.1875 | 0.0001 |

指南 §20.4 推荐静态 gate：unsafe rate 8–18%、Judge coverage ≥55%、T6-only 1–8%。确定性 head 在 0.2–0.5 阈值下 coverage 19–37%，未达 55% gate；这反映已保存 specialist 证据在 Judge-only 子集上的信息上限（§30.3 不承诺静态修复创造新语义能力）。

## 9. 共享 Evidence Adapter（指南 §23）

- 训练数据：Exp3 agent_predictions train（4091→2565 非重叠）/ dev（1047→650 非重叠），排除全部 Exp2 full sample_id/qy_hash（38,285 个 ID）。
- 模型：`LogisticRegression(C∈{0.01,0.1,1,10}, class_weight="balanced", liblinear)`，C 仅在 Exp3 dev 选择；特征：22 base + 6 交互项（`evidence.py`）。
- 多头：`FraudEvidenceAdapter`（fraud head）、`HarmfulComplianceAdapter`、`OverRefusalAdapter`（单类跳过）、`RefusalDetectionAdapter`。
- dev AUPRC：fraud=0.5509、harmful_compliance=0.9842（Exp3 域内；对 Exp2 为 OOD 转移）。

| 应用域 | adapter AUPRC | 确定性 AUPRC | 结论 |
|---|---|---|---|
| Fraud-R1（vs Judge 一致性） | 类别分布改善（Phishing 0.136@0.5） | 0.8% | adapter 恢复非 Fake Job 信号，阈值不确定 |
| DNA | 0.0412 | 0.2258 | **Exp3 欺诈域无法迁移到广义 harm，弃用 adapter** |
| Aegis response | 0.6987 | 0.7137 | 确定性 head 更优 |
| Aegis prompt | 0.7496 | 0.8108 | 确定性 head 更优 |

> 论文表述要求（§35）：Adapter 仅组合 Agent 证据、零 LLM 调用；不得暗示原始 categorical T6 达到相同结果。

## 10. 错误样本矩阵（指南 §27）

| 文件 | 条数 | 主要规则聚类 |
|---|---|---|
| errors/fraudr1_judge_only.jsonl | 1015 | Judge 检出而 T6 未检出（确定性头） |
| errors/fraudr1_t6_only.jsonl | 1 | T6 检出而 Judge 未检出 |
| errors/dna_false_negative.jsonl | 159 | 见 bucket_summary |
| errors/dna_false_positive.jsonl | 34 | 见 bucket_summary |
| errors/aegis_false_negative.jsonl | 199 | 见 bucket_summary |
| errors/aegis_false_positive.jsonl | 29 | 见 bucket_summary |
| errors/or_refusal_disagreement.jsonl | 782 | 拒答行为分歧 |

## 11. Exp3 暴露与数据泄漏（指南 §26）

| Benchmark | all-full | seen-in-exp3 | unexposed-only |
|---|---|---|---|
| fraudr1 | 8564 | 61 | 8503 |
| orbench | 3000 | 596 | 2404 |
| do_not_answer | 5634 | 925 | 4709 |
| aegis2 | 1964 | 793 | 1171 |

- Shared Evidence Adapter 训练已排除全部 Exp2 重叠 ID（`--exclude-exp2-overlap`）。
- Aegis 因 unexposed response 样本极少，仅可称为 **cross-work in-domain comparison**，不得称为 fully held-out external benchmark。

## 12. 静态回归门槛（指南 §29）

| 门槛 | 状态 |
|---|---|
| pytest 全部通过 | ✅ 375 passed（含新增 32 个静态修复测试） |
| 0 duplicate sample ID | ✅ 审计通过 |
| 0 missing Gold | ✅ DNA/Aegis frame 全量 gold |
| 0 missing predictions | ✅ 6/6 文件 integrity_ok |
| 0 suspicious empty Agent outputs | ✅ suspicious_predictions.jsonl 为空 |
| 每张二分类表可由 TP/FP/TN/FN 复算 | ✅ 断言通过 |
| Macro-F1=(Safe-F1+Unsafe-F1)/2 | ✅ 断言通过 |
| McNemar discordant delta=Accuracy delta | ✅ 断言通过 |
| observed delta 与 bootstrap 同 metric | ✅ 同一函数对象，中心偏差 <0.001 |
| OR 不再出现 all-safe 自动 100% | ✅ 修复后 safety-utility=0.9997（真实字段） |
| Aegis prompt/response 分轨 | ✅ N+、Recall、FPR、MF1 一致 |
| score 方向通过检查 | ✅ DNA/Aegis 无反向（choose_score_direction） |

## 13. Go/No-Go 与下一轮建议（指南 §34）

### Go 条件满足

- ✅ 所有数学断言通过；OR 映射修复；Aegis frame 修复；Schema 空输出无法通过；Fraud-R1 context 未损坏；错误矩阵输出具体缺失能力。

### 未满足（限制下一次 API pilot 范围）

- ❌ Fraud-R1 确定性 head 未达 Judge coverage ≥55% gate（37.4% max）；adapter 在类别分布上改善但阈值不稳定。
- ❌ DNA AUPRC 0.2258 未达 0.25 强门槛；within-prompt 0.3453 未达 0.40。
- ❌ Aegis response 离线 head AUPRC 低于原 categorical 分数。

若进入下一轮 API pilot，建议仅小规模验证（指南 §34.3）：Fraud-R1 Judge-only 200 条、DNA FN/FP 各 100 条、Aegis FN 100 条、OR hard/toxic 各 100 条；并优先用审计出的错误样本（`errors/*.jsonl`）定位 rubric 或 schema 缺口，而非直接全量重跑。

## 14. 论文结果使用边界（指南 §35）

可以保留：实验三多 Agent 机制消融、全量覆盖工程、Fraud-R1 风险趋势、Aegis/DNA 作为跨域边界、OR 修复后的拒答质量分析。

当前不能声称：全面优于四个原工作、OR 达到 100%、DNA 显著优于 Longformer、Aegis 显著优于 NemoGuard、Aegis 是 fully held-out external test。

## 15. 复现命令

```powershell
$env:FRAUDDISTILL_OFFLINE = '1'
python scripts/audit_exp2_predictions.py --offline
python scripts/audit_fraudr1_context.py --offline
python scripts/audit_exp2_frames.py --offline
python scripts/rescore_exp2_offline.py --mode deterministic --offline
python scripts/train_exp2_evidence_adapter.py --exclude-exp2-overlap --offline
python scripts/rescore_exp2_offline.py --mode shared-adapter --offline
python scripts/evaluate_exp2_static.py --offline --strict --bootstrap 10000
python scripts/make_exp2_static_report.py --offline
python -m pytest tests/test_offline_guard.py tests/test_exp2_static_schemas.py tests/test_exp2_static_metrics.py tests/test_exp2_static_adapters.py -q
```
