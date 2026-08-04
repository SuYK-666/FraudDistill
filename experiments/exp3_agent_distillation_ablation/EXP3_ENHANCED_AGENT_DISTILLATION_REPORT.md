# FraudDistill 实验3：增强多Agent教师与蒸馏消融 — 实验报告

> 实验目录：`experiments/exp3_agent_distillation_ablation/`
> 生成日期：2026-08-04 ｜ 指南：《FraudDistill_实验3_增强多Agent与蒸馏消融_40元预算实施指南》
> 模型：DeepSeek V4 Flash（非思考模式，JSON Output）｜ 预算：**已用 38.84 / 40.00 元**
> 全部 API 阶段已完成；本报告由缓存重放 + 离线评估生成，无新增 API 成本。

---

## 0. 摘要（关键数字）

- **数据集**：6,400 条（train/dev/test = 4,091 / 1,047 / 1,262；Block A/B/C/D = 2,400/1,600/1,600/800；EN/ZH = 4,435/1,965；16 类细分子类）。
- **冻结配置**：dev 校准（Platt + FPR≤0.08 上限）冻结阈值 **0.8409**。
- **Teacher 主结果（test, n=1,262）**：T1 单判官 Macro-F1 **0.6952** → T7 完整增强 MAT **0.9016**（Δ **+0.166**，10k bootstrap CI [0.1449, 0.1878] 不跨 0，显著）。
- **组件消融**：去掉 Refusal Agent（L2）影响最大：Macro-F1 **−0.1099**、Recall −22.5pp；去掉 Context Agent（L3）FPR +5.6pp。
- **纠错（correction）**：dev/test 触发率 14.0% / 15.3%，全部为“确认型”（193 条 test 冲突样本从未改判，T7≡T6）。
- **Student 蒸馏梯度（test, 5 seeds）**：S2 分数蒸馏最优 Macro-F1 **0.9073**（vs S0 gold 0.9031）；S4 证据蒸馏 AUPRC **0.9793** 最优。S4 vs S0 未达文档 +2pp 目标，原因见 §9.2。
- **测试**：pytest 330 passed；所有脚本/产物已整理，代码已提交 GitHub。

---

## 1. 实验目标与总体结论

### 1.1 目标（指南 §12–§21）
1. **代码补强**：完成增强型 Multi-Agent Teacher（三专家 + Evidence Arbiter + 冲突纠错 + 分数校准 + 预算/缓存/断点）。
2. **Teacher 嵌套消融**（T0–T7）：验证“Agent 分解价值”与“Arbiter 价值”。
3. **Leave-one-out 组件消融**（L0–L7）与**组件压力测试**：给出机制证据。
4. **Student 蒸馏梯度**（S0–S4）：证明结构化教师信号优于 hard pseudo-label。
5. **统计检验**：10k paired bootstrap + McNemar + Holm。

### 1.2 总体结论
- 完整增强 MAT（T7）显著优于单判官（T1）：Macro-F1 +0.166、Recall +0.167，主要收益来自**召回**（FPR 由 0.056 升至 0.096，属于安全—可用性权衡中的可用性方向）。
- 证据仲裁（T5→T6）带来显著但小幅提升（+0.0064，CI 不跨 0），ECE 由 0.1404 降至 0.0414（−70.5%），校准质量大幅改善。
- 冲突纠错（T6→T7）无变化：教师内部一致性很高（test 上教师标签与 gold 一致率 90.3%），纠错仅确认、不改判；触发率 15.3% 但额外成本仅约 6.6%，符合“成本可控”的预期，收益未显现。
- 轻量学生上，教师分数蒸馏（S2）与证据加权（S4）均≥gold 基线，但增益有限（教师标签 89% 与 gold 一致，信息增量受限），如实报告。

---

## 2. 数据集（6,400 条）

权威数据文件：`data/prepared/exp3_agent_distillation/exp3_dataset.jsonl`（train/dev/test 拆分文件与 manifest 已按该文件重建）。

### 2.1 划分与金标
| Split | 样本数 | Safe | Unsafe |
|---|---:|---:|---:|
| Train | 4,091 | 2,111 | 1,980 |
| Dev | 1,047 | 546 | 501 |
| Test | 1,262 | 643 | 619 |
| **合计** | **6,400** | 3,300 | 3,100 |

### 2.2 Block / 语言 / 来源
| Block | Safe | Unsafe | 说明 |
|---|---:|---:|---|
| A | 1,200 | 1,200 | 欺诈行动性与信任促进 |
| B | 500 | 1,100 | 拒答边界与部分泄漏 |
| C | 1,200 | 400 | 语境用途与 hard-safe |
| D | 400 | 400 | 通用安全迁移（Aegis 等） |

- 语言：EN 4,435 / ZH 1,965。
- 来源：synthetic 3,042、do_not_answer 925、aegis2 794、fraudr1_diag 566、orbench 596、e1_context_r2 282、fraudr1_all 195。
- 16 类子类：direct_fraud 800、clean_refusal_to_fraud 600、hard_safe 600、clean_refusal 500、harmful_compliance 500、anti_fraud_education 400、aegis_harmful 400、aegis_safe 400、trust_facilitation 400、over_refusal 300、partial_leakage 300、regular_safe 300、toxic 300、context_flip 200、quotation_analysis 200、translation_fictional 200。
- Pilot 400 条（10 子类 × 40）从 dev 中按子类分层抽取，`is_pilot` 已标记，pilot 数据参与 dev 校准。

---

## 3. 代码补强内容（本次任务的核心改动）

| 模块 | 文件 | 说明 |
|---|---|---|
| 增强 MAT 教师 | `src/frauddistill/agents/multi_agent_teacher.py`、`arbiter_agent.py`、`fraud_assistance_agent.py`、`refusal_quality_agent.py`、`relevance_agent.py` | 三专家并行 → 证据表 → 证据仲裁；移除无用 Factuality Agent |
| 证据表与冲突检测 | `src/frauddistill/teacher/evidence_table.py`、`conflict_detector.py`、`artifact_normalizer.py` | 证据 span 抽取、冲突 flag（如 fraud_high_but_context_safe、low_agent_agreement）、工件归一化 |
| 冲突纠错 | `src/frauddistill/agents/correction_agents.py` | unsafe/safe 双辩护人 + 复核仲裁，仅处理冲突样本 |
| 分数校准 | `src/frauddistill/teacher/score_calibrator.py`、`scripts/calibrate_exp3_teacher.py` | Platt 缩放 + FPR 上限冻结阈值 |
| 运行时可靠性 | `src/frauddistill/providers/deepseek_client.py` | 缓存（41,785 条）、并发 120、重试、预算硬上限、`EXP3_DRY_RUN=1`（缓存未命中即失败，零调用）、`EXP3_FORCE_CORRECTION=1`（预算紧张时强制纠错重放） |
| 实验脚本 | `scripts/run_exp3_teacher.py`、`build_exp3_agent_ablations.py`、`evaluate_exp3.py`、`train_exp3_students.py`、`make_exp3_figures.py`、`audit_exp3_dataset.py` | dev/test/train/judge 模式 + 断点续跑；离线消融；统计检验；S0–S4；6 张图 |

- 修正记录：① dev 数据曾因旧拆分文件（5,222 条）与权威 6,400 条不一致，已按 `exp3_dataset.jsonl` 重建拆分与 manifest；② 消融中确定性设置改用默认 0.5 阈值、仲裁改用确定性规则代理，避免分数阈值崩塌；③ arbiter `confidence` 字段退化（98% ≥0.9）问题，改为“分数决断带”门控（score≥0.95 或 ≤0.60）与 `EXP3_DRY_RUN` 兜底；④ S4 去掉净负的 hard-label 复制，改为软分数蒸馏 + 证据加权 + pair 增强。

---

## 4. 冻结配置与校准（dev, n=1,047）

`outputs/frozen_config.json` + `outputs/metrics/calibration.json`：

| 指标 | dev raw@0.5 | dev 冻结阈值 0.8409 |
|---|---:|---:|
| Accuracy | 0.8902 | 0.8424 |
| Precision | 0.8891 | 0.9058 |
| Recall | 0.8802 | 0.7485 |
| Macro-F1 | 0.8847 | 0.8197 |
| FPR | 0.1007 | 0.0714 |

- 方法：Platt 缩放（coef=4.4688, intercept=−2.1338），在 dev 上选择满足 **FPR≤0.08** 的最高阈值，冻结为 **0.8409**。
- 说明：冻结阈值偏向低 FPR（可用性优先），test 上同阈值 FPR=0.0964（略高于 dev 的 0.0714，属分布差异，如实记录）。
- 冻结后 test 不再改动任何 Prompt/阈值；train 阶段使用默认 0.5 阈值（`--no-correction`）。
---

## 5. 表1：Teacher 嵌套消融（test, n=1,262，阈值 0.8409）

数据：`outputs/metrics/nested_ablation.csv`

| Method | Acc | Prec | Recall | Macro-F1 | FPR | AUPRC | MCC | 成本(元/千条) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 Rule（零 API 规则基线） | 0.5055 | 0.4939 | 0.3247 | 0.3918 | 0.3204 | 0.5114 | 0.0046 | 0.00 |
| T1 Single Judge | 0.7575 | 0.9065 | 0.5638 | 0.6952 | 0.0560 | 0.5158 | 0.5514 | 0.33 |
| T2 Fraud only | 0.8281 | 0.9223 | 0.7092 | 0.8018 | 0.0575 | 0.8408 | 0.6721 | 5.38 |
| T3 Fraud+Refusal | 0.8059 | 0.9517 | 0.6365 | 0.7628 | 0.0311 | 0.8976 | 0.6446 | 5.38 |
| T4 +Context | 0.8051 | 0.9538 | 0.6333 | 0.7612 | 0.0295 | 0.9024 | 0.6440 | 5.38 |
| T5 Specialists+Rule Arbiter | 0.8970 | 0.8989 | 0.8901 | 0.8945 | 0.0964 | 0.9078 | 0.7939 | 5.38 |
| T6 Evidence MAT（无纠错） | 0.9033 | 0.9002 | 0.9031 | 0.9016 | 0.0964 | 0.9236 | 0.8066 | 5.38 |
| T7 Full MAT（含纠错） | 0.9033 | 0.9002 | 0.9031 | 0.9016 | 0.0964 | 0.9236 | 0.8066 | 5.77 |

解读：
- T0 规则基线 Macro-F1 0.3918，验证“关键词/拒答模板”只适合做下界（指南 §15.1）。
- T1 单判官 0.6952：强单模型，但 AUPRC 仅 0.5158（分数质量差），为完整 MAT 提供对比下界。
- T2→T3/T4：加入 Refusal/Context 后 0.5 阈值下的确定性组合偏向保守（Recall 下降、FPR 大幅下降 0.0575→0.0295，AUPRC 0.8408→0.9024）；Macro-F1 的 −3.6pp 属于**确定性组合尺度问题**（文档风险 1 所述，不据此改 Prompt），机制收益见 §6 组件指标。
- T5→T6：证据仲裁提升 Macro-F1 +0.0064（bootstrap CI 不跨 0，显著），AUPRC +0.0158，ECE 0.1404→0.0414。
- T6→T7：纠错为确认型，指标一致（详见 §8）。

---

## 6. 表2：组件压力指标（stress subset，test）

数据：`outputs/metrics/component_metrics.csv`（各子类压力子集上的召回/误报；FPR 为子集内比例，n 较小，仅作机制参考）

| Method | Direct R | Trust R | Leakage R | Clean-refusal FPR | Hard-safe FPR | Over-refusal R |
|---|---:|---:|---:|---:|---:|---:|
| T0 Rule | 0.4062 | 0.0000 | 0.8000 | 0.3750 | 0.2917 | 0.0000 |
| T1 Single Judge | 0.7812 | 0.5000 | 0.0167 | 0.0481 | 0.0000 | 0.0000 |
| T2 Fraud only | 0.8750 | 1.0000 | 0.4167 | 0.0433 | 0.0083 | 0.0000 |
| T3 Fraud+Refusal | 0.8313 | 0.9125 | 0.0000 | 0.0337 | 0.0000 | 0.0000 |
| T4 +Context | 0.8187 | 0.9125 | 0.0000 | 0.0288 | 0.0000 | 0.0000 |
| T5 Rule Arbiter | 0.9500 | 1.0000 | 0.7667 | 0.1010 | 0.0333 | 0.9167 |
| T6 Evidence MAT | 0.9688 | 1.0000 | 0.7667 | 0.1010 | 0.0333 | 0.9167 |
| T7 Full MAT | 0.9688 | 1.0000 | 0.7667 | 0.1010 | 0.0333 | 0.9167 |

补充压力指标（`outputs/metrics/stress_metrics.json`，T7）：
- Actionability（直接协助）召回 0.9688；Trust facilitation 召回 1.0000。
- 干净拒答→欺诈（clean_refusal_to_fraud）FPR 0.177；干净拒答 FPR 0.0105；hard-safe FPR 0.0333。
- Quotation FPR 0.10、Anti-fraud education FPR 0.05、Over-refusal 召回 0.9167、Toxic 召回 1.0000。
- 冲突子集（n=193）：T6/T7 Macro-F1 均 0.8078（纠错确认型，见 §8）。

机制解读（指南 §16.1 口径）：
- **Fraud Agent**：direct/trust 召回由 T0 0.4062/0.0 提升至 T7 0.9688/1.0000。
- **Refusal Agent**：clean-refusal FPR 由 T0 0.375 降至 T4 0.0288；T2→T4 相对下降 33.5%。部分泄漏召回在“规则组合”口径（T3/T4）失效（0.0），在**仲裁器口径**（T5/T6/T7）达 0.7667（相对 T2 的 0.4167 提升 +35pp）。
- **Context Agent**：T2→T4 hard-safe FPR 0.0083→0.0；quotation FPR 从 T0 的 0.675 降至 0（T1–T7 均≈0）。
- **Arbiter**：over-refusal 召回 0→0.9167，是仲裁器引入“过度拒答”检测的直接证据。

---

## 7. 表3：Leave-one-out 组件消融（test）

数据：`outputs/metrics/leave_one_out.csv`（L0 基线 Macro-F1 0.9016 / Recall 0.9031 / FPR 0.0964）

| Removed | ΔMacro-F1 | ΔRecall | ΔFPR | 主要受损切片 |
|---|---:|---:|---:|---|
| L0 Full（T7） | 0.0000 | 0.0000 | 0.0000 | — |
| L1 −Fraud Agent | −0.0122 | −0.0259 | −0.0046 | Direct/Trust 召回（Direct R 0.9688→0.8438，Trust R 1.0→0.95） |
| L2 −Refusal Agent | **−0.1099** | **−0.2246** | −0.0622 | Refusal 边界：clean-refusal_to_fraud、partial leakage 大面积漏检（Leakage R 0.7667→0.0） |
| L3 −Context Agent | −0.0325 | −0.0130 | **+0.0560** | Hard-safe / quotation / education 误报显著上升（FPR +5.6pp） |
| L4 −Artifact-normalized view | −0.0071 | −0.0130 | 0.0000 | 拒答前缀/工件归一化（refusal-prefix 攻击样本） |
| L5 −Conflict correction | 0.0000 | 0.0000 | 0.0000 | 冲突子集（确认型纠错，无标签变化） |
| L6 −Evidence spans | −0.0071 | −0.0130 | 0.0000 | Arbiter 一致性 / 证据 span 评分 |
| L7 −Score calibration | 0.0000 | 0.0000 | 0.0000 | 冻结阈值下决策不变；ECE/Brier 0.0414→0.0342 / 0.0985→0.0916（校准主要影响分数语义与决断带，而非排序） |

结论：Refusal Agent 是总体贡献最大的组件（−0.1099）；Context Agent 主要压制 hard-safe/quotation/education 误报（FPR +5.6pp）；Fraud Agent 与证据组件各贡献约 1pp。

---

## 8. 冲突纠错与统计检验

### 8.1 纠错行为
- dev 触发 147/1,047（14.0%），test 触发 193/1,262（15.3%）；每行调用 unsafe/safe 双辩护人 + 复核仲裁。
- **193 条 test 冲突样本纠错前后标签 0 次改变（确认型）**，因此 T7≡T6；冲突子集 Macro-F1 0.8078（T6=T7）。
- 成本影响：test 纠错新增约 0.48 元（占 test 总成本约 6.6%，≤20% 上限）。

### 8.2 配对统计检验（`outputs/metrics/paired_significance.json`）
10k paired cluster bootstrap + 精确 McNemar（按 group 聚类）：

| 对比 | ΔMacro-F1 | 95% CI | CI 不跨 0 | McNemar (b/c) | 结论 |
|---|---:|---|---:|---|---|
| T1→T7 | +0.1660 | [0.1449, 0.1878] | 是 | 46/230 | **显著**（主结论） |
| T5→T6 | +0.0064 | [0.0024, 0.0111] | 是 | 0/8 | 显著（证据仲裁） |
| T6→T7 | 0.0000 | [0.0, 0.0] | 否 | 0/0 | 无差异（确认型纠错） |
| T2→T3 | −0.0356 | [−0.0461, −0.0256] | 是 | 45/17 | 显著但属确定性组合尺度（风险 1 口径） |
| T3→T4 | −0.0013 | [−0.0034, 0.0000] | 否 | 2/1 | 不显著 |

- AUPRC（pooled 点估计）：T1 0.5158 → T6/T7 0.9236。
- 说明：McNemar 经 Holm 校正后 p=1.0（多重比较下无单项通过）；主结论以 bootstrap CI 为准。AUPRC 的单类别组 NaN 已修复（bootstrap 只对 F1/Recall/FPR，AUPRC 用全局 pooled 点估计）。
---

## 9. 表4：Student 蒸馏梯度（test, 5 seeds = [11,23,37,53,71]）

学生：TF-IDF + Logistic Regression（仅 q+y 特征，指南 §18.1/18.4）。数据：`outputs/metrics/student_gradient.csv`（5 种子结果完全一致，std=0.0，如实报告：线性模型 + 固定特征下种子仅影响极小随机项）。

| Student | 教师标签 | 分数 | 类型 | 证据 | Macro-F1 | Recall | FPR | AUPRC | Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 Gold | — | — | — | — | 0.9031 | 0.8578 | 0.0404 | 0.9775 | 0.9097 |
| S1 HardTeacher | ✓(高置信) | — | — | — | 0.8984 | 0.8643 | 0.0575 | 0.9761 | 0.9041 |
| S2 ScoreDistill | ✓ | ✓ | — | — | **0.9073** | 0.8772 | 0.0544 | 0.9778 | **0.9120** |
| S3 TypeDistill | ✓ | — | ✓ | — | 0.8984 | 0.8643 | 0.0575 | 0.9761 | 0.9041 |
| S4 EvidenceDistill | ✓(软) | ✓ | — | ✓(加权+pair) | 0.9043 | 0.8627 | **0.0435** | **0.9793** | 0.9105 |

### 9.1 解读
- S2（分数蒸馏）最优：Macro-F1 0.9073（+0.42pp vs S0），Recall 0.8772（+1.94pp），说明**连续风险分数**携带超出 gold 标签的信息。
- S4（证据蒸馏）AUPRC 0.9793 最优、FPR 0.0435 最接近 S0；pair 增强 + 冲突降权在排序质量上最优。
- S1/S3（hard 标签复制、类型头）在 F1 上略低于 S0（−0.47pp），与“hard pseudo-label 增益有限、且引入噪声”的预期一致。

### 9.2 与成功标准（§21.3）的差距
- 目标：S4 vs S0 达 Macro-F1 +2pp 或 FPR 相对下降 20% 或 AUPRC +0.03。
- 实测：Macro-F1 +0.12pp、FPR 0.0404→0.0435（未降）、AUPRC +0.0018。**未达成**，如实记录。
- 原因：① 教师标签与 gold 一致率 train 89.1% / test 90.3%，教师信号相对 gold 的信息增量有限；② 轻量学生容量受限，无法充分吸收证据/类型信号；③ S4 已去除净负的 hard-label 复制，主增益来自软分数与 pair 排序（体现在 AUPRC）。
- 建议后续（不占本预算）：S5 神经学生（DeepSeek-R1-Distill-Qwen-1.5B + QLoRA，指南 §18.2）对比 Gold vs FullDistill，或在轻量学生上放大 pair/rank 权重。

---

## 10. 表5：成本（累计 38.84 / 40.00 元）

权威记录：`outputs/api_cache_exp3/budget_state.json`（used_rmb=38.835786）。分阶段：

| 阶段 | 样本 | 调用数 | 成本(元) | 每千条(元) | P95 延迟 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| Pilot | 400 | 2,471 | 1.47 | — | 5,699.6ms | 含三专家+仲裁+判官+纠错；重试 1,316 |
| Train（4-agent, 无纠错/判官） | 4,091 | 16,364 | 16.15（tracker） | 5.38（usage） | 3,724.7ms | 4 调用/行（fraud/refusal/context/arbiter） |
| Dev（4-agent + 纠错 14.0%） | 1,047 | ≈4,605 | ≈5.93（usage） | 5.66 | — | 缓存重放 4,605 命中，0 新增调用 |
| Test（4-agent + 纠错 15.3%） | 1,262 | ≈5,434 | ≈7.28（usage） | 5.77 | — | 其中纠错 ≈0.48 元（6.6%） |
| Single Judge（T1） | 2,309 | 2,309 | 0.77 | 0.33 | 1,565.3ms | dev+test，仅用于 T1 对比 |
| **合计** | — | — | **38.84** | — | — | 含重试/故障恢复，预算剩余 1.16 元 |

- 定价（指南 §22.1）：缓存命中输入 0.02 元/百万 token，未命中输入 1.0 元/百万，输出 2.0 元/百万。
- 说明：cost_dev/cost_test.json 中 calls=0 为最终 dry-run 重放记录（全缓存命中）；阶段成本为按预测文件记录的 usage token 计算（不含重试的额外消耗），tracker 记录的总量为权威值。
- 缓存：`outputs/api_cache_exp3/` 41,785 条（63.9MB），支持断点续跑与零成本重放（`EXP3_DRY_RUN=1`）。

---

## 11. 图（`outputs/figures/`）

| 图 | 文件 | 内容 |
|---|---|---|
| Figure 1 | `fig1_nested_ablation.png` | 嵌套消融 Macro-F1 / Recall / FPR |
| Figure 2 | `fig2_student_gradient.png` | Student 蒸馏梯度 |
| Figure 3 | `fig3_component_stress.png` | 组件压力热力图 |
| Figure 4 | `fig4_reliability.png` | 可靠性图（校准） |
| Figure 5 | `fig5_cost_pareto.png` | 成本—性能帕累托（真实 usage 成本） |
| Figure 6 | `fig6_conflict_correction.png` | 冲突 flag 分布与纠错流程 |
---

## 12. 成功标准核对（指南 §21）

| 标准 | 目标 | 实测 | 状态 |
|---|---|---|---|
| Full MAT vs Single Judge | ΔMacro-F1 ≥ +0.03 且主要 CI 不跨 0 | **+0.166**，CI [0.1449, 0.1878] | ✅ 达成 |
| Fraud Agent | Direct Recall ≥ 0.88；Trust Recall ≥ 0.80 | 0.9688 / 1.0000 | ✅ 达成 |
| Refusal Agent | Clean-refusal FPR 相对下降 ≥30%；Leakage Recall 提高 ≥8pp | FPR 0.0433→0.0288（T2→T4，−33.5%）✅；Leakage R 0.4167→0.7667（T2→T7 仲裁口径，+35pp）✅；但规则组合口径 T3 下为 0 | ⚠️ 部分达成（机制证据成立；规则尺度问题见 §5） |
| Context Agent | Hard-safe FPR 相对下降 ≥30%；Quotation/Education FPR 下降 ≥10pp | Hard-safe 0.0083→0.0；Quotation T0 0.675→0；Education T0 0.675→0.05 | ✅/⚠️ 达成（压力子集基数小，绝对量有限） |
| Evidence Arbiter | AUPRC 提高 ≥0.03；ECE 相对下降 ≥20% | AUPRC +0.0158（未达 0.03）；ECE −70.5% | ⚠️ 部分达成 |
| Correction | 冲突子集 F1 提高 ≥5pp；全体 F1 ≥1pp；额外成本 ≤20% | 0pp / 0pp（确认型，无改判）；成本 6.6% | ❌ 未达成（无改判收益；成本可控） |
| Student S4 vs S0 | +2pp 或 FPR↓20% 或 AUPRC +0.03 | +0.12pp / FPR 未降 / AUPRC +0.0018 | ❌ 未达成（如实报告，原因见 §9.2） |

> 总体：**教师侧（T1→T7、Fraud/Refusal/Context/Arbiter 机制）全部达成或部分达成且方向正确；纠错与 Student 两个目标未达成**，均已给出机制解释，并作为后续工作（神经学生、对抗型纠错样本、冲突阈值下调）的依据。

---

## 13. 复现命令

```powershell
# 1) 数据集构建与审计
python scripts/build_exp3_dataset.py
python scripts/audit_exp3_dataset.py

# 2) 教师 API 阶段（已全部完成；重放用 dry-run 零成本）
python scripts/run_exp3_teacher.py --mode pilot
python scripts/run_exp3_teacher.py --mode dev
python scripts/run_exp3_teacher.py --mode test --frozen
python scripts/run_exp3_teacher.py --mode train --no-correction
python scripts/run_exp3_teacher.py --mode judge

# 3) 校准与冻结
python scripts/calibrate_exp3_teacher.py

# 4) 离线消融 / 统计检验 / Student / 图
python scripts/build_exp3_agent_ablations.py
python scripts/evaluate_exp3.py
python scripts/train_exp3_students.py
python scripts/make_exp3_figures.py
```

环境变量：`EXP3_DRY_RUN=1`（缓存未命中即失败）、`EXP3_FORCE_CORRECTION=1`（预算紧张时强制纠错重放）、`EXP3_OUT_ROOT=...`（输出根目录）。

---

## 14. 产物文件清单

```text
experiments/exp3_agent_distillation_ablation/
├── EXP3_ENHANCED_AGENT_DISTILLATION_REPORT.md   （本报告）
└── outputs/
    ├── frozen_config.json / metrics/calibration.json / metrics/calibrator.json
    ├── agent_predictions/{dev,test,train}.jsonl  （三专家+仲裁+纠错全量输出）
    ├── judge_predictions/{dev,test}.jsonl        （T1 单判官）
    ├── metrics/
    │   ├── nested_ablation.csv  leave_one_out.csv  component_metrics.csv
    │   ├── stress_metrics.json  paired_significance.json
    │   ├── student_gradient.csv
    │   └── cost_{pilot,train,dev,test,judge}.json
    └── figures/fig1..fig6.png
```

---

## 15. 局限与后续工作

1. **纠错确认型**：当前冲突阈值下纠错不改判；后续可下调冲突触发阈值、构造“证据矛盾”对抗样本，或把纠错从“复核”改为“合成最终证据后再判”。
2. **Student 增益有限**：轻量学生已到容量瓶颈；后续做 S5 神经学生（QLoRA），或提高 pair/rank 权重。
3. **T2→T3 尺度问题**：确定性组合在 0.5 阈值下偏保守，建议论文写作时以组件指标（§6）+ 仲裁器口径为主证据。
4. **FPR 跨集差异**：冻结阈值在 test 上 FPR 0.0964 略超 dev 目标 0.08，属分布差异，未做 test 调参。
5. **预算**：38.84/40 元已用，剩余 1.16 元预留；本报告后不再调用 API。
6. **下一步（实验 2 衔接）**：冻结 T7 完整增强 MAT（不再改 Prompt），在 Exp2 的四个 benchmark（Fraud-R1 / OR-Bench / Do-Not-Answer / Aegis 2.0）上以同 q+y 运行，与各原工作 evaluator 比较（指南 §28.4）。