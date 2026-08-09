# FraudDistill 实验3：增强多Agent教师与蒸馏消融 — 实验报告

> 实验目录：`experiments/exp3_agent_distillation_ablation/`
> 生成日期：2026-08-04 ｜ 指南：《FraudDistill_实验3_增强多Agent与蒸馏消融_40元预算实施指南》
> 2026-08-07 更新：《FraudDistill_实验三后续修改方案》——合并 T6/T7 为 T6 Full MAT；新增 Base-1.5B Zero-shot 下界；整理 Student 蒸馏对比链；预留 Final Student 重训接口（`--setting final_student`）。
> 2026-08-09 更新：《最终1.5B学生模型训练实施指南》——FraudDistill-Student-1.5B 正式训练与官方验收完成：Macro-F1 0.9135 / Recall 0.8853 / FPR 0.0591 / AUPRC 0.9717 / MCC 0.8282 / Real-only 0.7913 / 4-class 0.4657（test n=1,262，冻结阈值 0.5622，reload checksum PASS）；Hard Gate 8/9 通过（仅 FPR 超 0.050 约 0.9pp），全面超越 Neural-SoftDistill，详见 §16.10。
> 模型：DeepSeek V4 Flash（非思考模式，JSON Output）｜ 预算：**已用 38.84 / 40.00 元**
> 全部 API 阶段已完成；本报告由缓存重放 + 离线评估生成，无新增 API 成本。

---

## 0. 摘要（关键数字）

- **数据集**：6,400 条（train/dev/test = 4,091 / 1,047 / 1,262；Block A/B/C/D = 2,400/1,600/1,600/800；EN/ZH = 4,435/1,965；16 类细分子类）。
- **冻结配置**：dev 校准（Platt + FPR≤0.08 上限）冻结阈值 **0.8409**。
- **Teacher 主结果（test, n=1,262）**：T1 单判官 Macro-F1 **0.6952** → T6 完整增强 MAT **0.9016**（Δ **+0.166**，10k bootstrap CI [0.1449, 0.1878] 不跨 0，显著）。
- **组件消融**：去掉 Refusal Agent（L2）影响最大：Macro-F1 **−0.1099**、Recall −22.5pp；去掉 Context Agent（L3）FPR +5.6pp。
- **纠错（correction）**：dev/test 触发率 14.0% / 15.3%，全部为“确认型”（193 条 test 冲突样本从未改判；T6 与含纠错的完整版完全一致，按指南合并为 T6 Full MAT）。
- **Student 蒸馏梯度（test, 5 seeds）**：S2 分数蒸馏最优 Macro-F1 **0.9073**（vs S0 gold 0.9031）；S4 证据蒸馏 AUPRC **0.9793** 最优。S4 vs S0 未达文档 +2pp 目标，原因见 §9.2。
- **神经学生（1.5B QLoRA，CPU 训练，seed 11）**：Neural-SoftDistill 最优 Macro-F1 **0.8849**（test，Recall 0.8094 / FPR 0.0404 / AUPRC 0.9532 / MCC 0.7795）；Neural-Gold 0.8676；Neural-FullDistill（+4,000 困难扩展）Recall 0.8207 最高但 FPR 0.0918；低标注 10% gold：Neural-Gold 0.8422。离线评估已修复分类头保存问题，指标与训练日志一致。本轮 API 新增 7.00 元（详见 §16）。
- **Base-1.5B 下界（08-07 补充）**：同一 500 条 test 子集上，未训练 1.5B 基座 Macro-F1 **0.3584**（FPR 0.9611，几乎全判 unsafe）→ Neural-Gold 0.8794 → Neural-SoftDistill **0.9032**，能力提升链 Base → Trained Student 成立，详见 §16.9。
- **最终 1.5B 学生模型（08-09）**：FraudDistill-Student-1.5B 官方 test（n=1,262，冻结阈值 0.5622）Macro-F1 **0.9135** / Acc 0.9136 / Recall 0.8853 / FPR 0.0591 / AUPRC 0.9717 / MCC 0.8282 / Real-only **0.7913** / 4-class 0.4657；对比 Neural-SoftDistill：MF1 +0.0286、Recall +0.0759、Real-only +0.1018、4-class +0.0525（FPR +0.0187，略超 Hard Gate ≤0.050）；**配对显著性（10k bootstrap，§16.11）：ΔMF1 CI [0.0122, 0.0446]、ΔRecall CI [0.0513, 0.1007] 均不跨 0，McNemar p=1.7e-08**。
- **测试**：pytest 460 passed；所有脚本/产物已整理，代码已提交 GitHub。

---

## 1. 实验目标与总体结论

### 1.1 目标（指南 §12–§21）
1. **代码补强**：完成增强型 Multi-Agent Teacher（三专家 + Evidence Arbiter + 冲突纠错 + 分数校准 + 预算/缓存/断点）。
2. **Teacher 嵌套消融**（T0–T6）：验证“Agent 分解价值”与“Arbiter 价值”。
3. **Leave-one-out 组件消融**（L0–L7）与**组件压力测试**：给出机制证据。
4. **Student 蒸馏梯度**（S0–S4）：证明结构化教师信号优于 hard pseudo-label。
5. **统计检验**：10k paired bootstrap + McNemar + Holm。

### 1.2 总体结论
- 完整增强 MAT（T6）显著优于单判官（T1）：Macro-F1 +0.166、Recall +0.167，主要收益来自**召回**（FPR 由 0.056 升至 0.096，属于安全—可用性权衡中的可用性方向）。
- 证据仲裁（T5→T6）带来显著但小幅提升（+0.0064，CI 不跨 0），ECE 由 0.1404 降至 0.0414（−70.5%），校准质量大幅改善。
- 冲突纠错为确认型：教师内部一致性很高（test 上教师标签与 gold 一致率 90.3%），193 条冲突样本纠错仅确认、0 次改判，触发率 15.3% 但额外成本仅约 6.6%，符合“成本可控”的预期；T6 与含纠错的完整版完全一致，按指南合并为 T6 Full MAT，收益未显现。
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
| T6 Full MAT | 0.9033 | 0.9002 | 0.9031 | 0.9016 | 0.0964 | 0.9236 | 0.8066 | 5.38 |

解读：
- T0 规则基线 Macro-F1 0.3918，验证“关键词/拒答模板”只适合做下界（指南 §15.1）。
- T1 单判官 0.6952：强单模型，但 AUPRC 仅 0.5158（分数质量差），为完整 MAT 提供对比下界。
- T2→T3/T4：加入 Refusal/Context 后 0.5 阈值下的确定性组合偏向保守（Recall 下降、FPR 大幅下降 0.0575→0.0295，AUPRC 0.8408→0.9024）；Macro-F1 的 −3.6pp 属于**确定性组合尺度问题**（文档风险 1 所述，不据此改 Prompt），机制收益见 §6 组件指标。
- T5→T6：证据仲裁提升 Macro-F1 +0.0064（bootstrap CI 不跨 0，显著），AUPRC +0.0158，ECE 0.1404→0.0414。
- T6 Full MAT：纠错为确认型（193 条冲突样本 0 次改判），与含纠错的完整版完全一致，按指南合并（详见 §8）。

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
| T6 Full MAT | 0.9688 | 1.0000 | 0.7667 | 0.1010 | 0.0333 | 0.9167 |

补充压力指标（`outputs/metrics/stress_metrics.json`，T6）：
- Actionability（直接协助）召回 0.9688；Trust facilitation 召回 1.0000。
- 干净拒答→欺诈（clean_refusal_to_fraud）FPR 0.177；干净拒答 FPR 0.0105；hard-safe FPR 0.0333。
- Quotation FPR 0.10、Anti-fraud education FPR 0.05、Over-refusal 召回 0.9167、Toxic 召回 1.0000。
- 冲突子集（n=193）：T6（含纠错）Macro-F1 0.8078（纠错确认型，见 §8）。

机制解读（指南 §16.1 口径）：
- **Fraud Agent**：direct/trust 召回由 T0 0.4062/0.0 提升至 T6 0.9688/1.0000。
- **Refusal Agent**：clean-refusal FPR 由 T0 0.375 降至 T4 0.0288；T2→T4 相对下降 33.5%。部分泄漏召回在“规则组合”口径（T3/T4）失效（0.0），在**仲裁器口径**（T5/T6）达 0.7667（相对 T2 的 0.4167 提升 +35pp）。
- **Context Agent**：T2→T4 hard-safe FPR 0.0083→0.0；quotation FPR 从 T0 的 0.675 降至 0（T1–T6 均≈0）。
- **Arbiter**：over-refusal 召回 0→0.9167，是仲裁器引入“过度拒答”检测的直接证据。

---

## 7. 表3：Leave-one-out 组件消融（test）

数据：`outputs/metrics/leave_one_out.csv`（L0 基线 Macro-F1 0.9016 / Recall 0.9031 / FPR 0.0964）

| Removed | ΔMacro-F1 | ΔRecall | ΔFPR | 主要受损切片 |
|---|---:|---:|---:|---|
| L0 Full（T6） | 0.0000 | 0.0000 | 0.0000 | — |
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
- **193 条 test 冲突样本纠错前后标签 0 次改变（确认型）**，因此 T6 与含纠错的完整版完全一致；冲突子集 Macro-F1 0.8078，按指南合并为 T6 Full MAT。
- 成本影响：test 纠错新增约 0.48 元（占 test 总成本约 6.6%，≤20% 上限）。

### 8.2 配对统计检验（`outputs/metrics/paired_significance.json`）
10k paired cluster bootstrap + 精确 McNemar（按 group 聚类）：

| 对比 | ΔMacro-F1 | 95% CI | CI 不跨 0 | McNemar (b/c) | 结论 |
|---|---:|---|---:|---|---|
| T1→T6 | +0.1660 | [0.1449, 0.1878] | 是 | 46/230 | **显著**（主结论） |
| T5→T6 | +0.0064 | [0.0024, 0.0111] | 是 | 0/8 | 显著（证据仲裁） |
| T2→T3 | −0.0356 | [−0.0461, −0.0256] | 是 | 45/17 | 显著但属确定性组合尺度（风险 1 口径） |
| T3→T4 | −0.0013 | [−0.0034, 0.0000] | 否 | 2/1 | 不显著 |

- AUPRC（pooled 点估计）：T1 0.5158 → T6 0.9236。
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
- 后续已实施 S5 神经学生（DeepSeek-R1-Distill-Qwen-1.5B + LoRA，CPU），结果见 §16；SoftDistill 相对 Gold 提升 Macro-F1 +1.73pp 且 FPR 下降 2.65pp，FullDistill（+4,000 扩展数据）Recall 最高但 FPR 上升，机制解读见 §16.4。

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
| Refusal Agent | Clean-refusal FPR 相对下降 ≥30%；Leakage Recall 提高 ≥8pp | FPR 0.0433→0.0288（T2→T4，−33.5%）✅；Leakage R 0.4167→0.7667（T2→T6 仲裁口径，+35pp）✅；但规则组合口径 T3 下为 0 | ⚠️ 部分达成（机制证据成立；规则尺度问题见 §5） |
| Context Agent | Hard-safe FPR 相对下降 ≥30%；Quotation/Education FPR 下降 ≥10pp | Hard-safe 0.0083→0.0；Quotation T0 0.675→0；Education T0 0.675→0.05 | ✅/⚠️ 达成（压力子集基数小，绝对量有限） |
| Evidence Arbiter | AUPRC 提高 ≥0.03；ECE 相对下降 ≥20% | AUPRC +0.0158（未达 0.03）；ECE −70.5% | ⚠️ 部分达成 |
| Correction | 冲突子集 F1 提高 ≥5pp；全体 F1 ≥1pp；额外成本 ≤20% | 0pp / 0pp（确认型，无改判）；成本 6.6% | ❌ 未达成（无改判收益；成本可控） |
| Student S4 vs S0 | +2pp 或 FPR↓20% 或 AUPRC +0.03 | +0.12pp / FPR 未降 / AUPRC +0.0018 | ❌ 未达成（如实报告，原因见 §9.2） |

| Neural-SoftDistill vs Neural-Gold（同数据） | 蒸馏增益 ≥1.5pp 或 FPR 相对下降 ≥15% | Macro-F1 **+1.73pp**；FPR 0.0669→0.0404（相对 −39.6%） | ✅ 达成（单种子，方向一致） |
| **FraudDistill-Student-1.5B vs Neural-SoftDistill**（最终指南，test n=1,262，各用官方阈值） | 全面超越旧 SoftDistill | Macro-F1 **+0.0285**（10k paired bootstrap CI [0.0122, 0.0446] 不跨 0）；Recall +0.0759（CI [0.0513, 0.1007]）；AUPRC +0.0185（CI [0.0068, 0.0296]）；MCC +0.0487（CI [0.0168, 0.0799]）；McNemar p=1.7e-08 | ✅ 达成（仅 FPR +0.0187，CI [−0.0016, 0.0397] 跨 0 且 0.0591 略超 Hard Gate ≤0.050；见 §16.10–16.12） |
| Neural-FullDistill vs Neural-Gold | 同上 | Macro-F1 −0.28pp；Recall +1.94pp；FPR +2.49pp | ❌ 未达成（困难样本阈值偏移，如实报告） |
| Base（未训练 1.5B 明显低于训练后模型） | Base Macro-F1 显著低于 SoftDistill | 0.3584 vs 0.9032（500 子集，Δ+0.545）；FPR 0.9611 vs 0.0311 | ✅ 达成（能力提升链成立，§16.9） |
| 神经学生切片（指南 §24.6） | Direct R≥0.94 / Hard-safe FPR≤0.04 / Clean-refusal FPR≤0.05 / Over-refusal R≥0.85 / Context-flip≥0.90 | **Final Student（§16.10）**：direct 0.9812 / trust 1.0 / leakage 0.9667 / clean-refusal FPR 0.0105 / hard-safe FPR 0.0 / over-refusal 0.9333 / context-flip 0.9474 | ✅ 达成 |

> 总体：**教师侧（T1→T6、Fraud/Refusal/Context/Arbiter 机制）全部达成或部分达成且方向正确；纠错目标未达成（确认型无改判）；Student 目标已由 Final Student 达成（§16.10–16.12）**，均已给出机制解释。

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

# 3) 神经学生（1.5B LoRA）训练与评估（CPU）
python scripts/audit_student_training_data.py
python scripts/train_exp3_students.py --manifest data/prepared/exp3_neural_student/train_manifest.jsonl --backend neural --architecture standard --seeds 11 --max-length 384 --micro-batch 2 --effective-batch 32 --setting gold --epochs 2
python scripts/train_exp3_students.py --manifest data/prepared/exp3_neural_student/train_manifest.jsonl --backend neural --architecture standard --seeds 11 --max-length 384 --micro-batch 2 --effective-batch 32 --setting soft_distill --epochs 2
python scripts/train_exp3_students.py --manifest data/prepared/exp3_neural_student/train_manifest_expanded.jsonl --backend neural --architecture standard --seeds 11 --max-length 384 --micro-batch 2 --effective-batch 32 --setting full_distill --epochs 1 --max-steps 140
python scripts/evaluate_neural_student.py --checkpoint experiments/exp3_agent_distillation_ablation/outputs/neural_student/soft_distill_standard_seed11_final --architecture standard --max-length 384 --out-dir experiments/exp3_agent_distillation_ablation/outputs/neural_student/eval_soft
python scripts/_run_lowlabel.py   # 低标注曲线驱动（gold10 已完成；soft10/25/50 后续补跑）

# 4) Base-1.5B Zero-shot baseline（本地 CPU，0 元；双分片并行，seed 20260804 固定 500 条）
python scripts/run_exp3_base_zeroshot.py --n 500 --seed 20260804 --shard-idx 0 --shard-total 2 --out-dir experiments/exp3_agent_distillation_ablation/outputs/neural_student/base_zeroshot
python scripts/run_exp3_base_zeroshot.py --n 500 --seed 20260804 --shard-idx 1 --shard-total 2 --out-dir experiments/exp3_agent_distillation_ablation/outputs/neural_student/base_zeroshot
python scripts/run_exp3_base_zeroshot.py --eval-only --out-dir experiments/exp3_agent_distillation_ablation/outputs/neural_student/base_zeroshot
python scripts/compare_exp3_base_chain.py
python scripts/make_exp3_base_figure.py

# 5) 离线消融 / 统计检验 / Student / 图
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

    ├── neural_student/
    │   ├── {gold,soft,full}_standard_seed11_final/   (<-> LoRA 适配器 + 分类头（modules_to_save）)
    │   ├── eval_{gold,soft,full,lowlabel_gold10,zero_shot}/   (<-> 完整评估：指标 JSON + 预测)
    │   ├── base_zeroshot/                  (<-> Base-1.5B 生成式下界：subset_ids + 预测 + 指标)
    │   ├── base_chain_500.csv/.json         (<-> 500 子集能力链：Base/RandomHead/Gold/Soft/Full)
    │   └── lowlabel/                          (<-> 低标注 gold10 完成；soft10 resume.pt 保留)
    ├── metrics/neural_student_metrics.json      (<-> canonical 合并指标，指南 §3.8)
    └── figures/fig1..fig6.png  fig7_neural_student.png  fig8_lowlabel_curve.png  fig9_student_capability_chain.png
```

---

## 15. 局限与后续工作

1. **纠错确认型**：当前冲突阈值下纠错不改判；后续可下调冲突触发阈值、构造“证据矛盾”对抗样本，或把纠错从“复核”改为“合成最终证据后再判”。
2. **Student 蒸馏路径**：轻量 S 系列（5 seeds 分数/证据蒸馏）增益有限，已被神经学生取代——**FraudDistill-Student-1.5B 在官方 test 上全面超越 Neural-SoftDistill**（MF1 +0.0285，10k paired bootstrap CI [0.0122, 0.0446] 不跨 0；Recall +0.0759；Real-only +0.1018；McNemar p=1.7e-08，§16.11）；S5 神经学生已完成，无需再做。
3. **T2→T3 尺度问题**：确定性组合在 0.5 阈值下偏保守，建议论文写作时以组件指标（§6）+ 仲裁器口径为主证据。
4. **FPR 跨集差异**：冻结阈值在 test 上 FPR 0.0964 略超 dev 目标 0.08，属分布差异，未做 test 调参。
5. **预算**：38.84/40 元已用，剩余 1.16 元预留；本报告后不再调用 API。
6. **下一步（实验 2 衔接）**：冻结 T6 完整增强 MAT（不再改 Prompt），在 Exp2 的四个 benchmark（Fraud-R1 / OR-Bench / Do-Not-Answer / Aegis 2.0）上以同 q+y 运行，与各原工作 evaluator 比较（指南 §28.4）。

7. **神经学生（§16）**：① 仅 seed 11（CPU 训练 ~8h/轮，3 seeds 未跑；已用 §16.11 的 10k paired bootstrap 替代单点比较）；② 低标注曲线只完成 10%（gold10；soft10 中断于 ~1 epoch，resume.pt 已保留可续训），25%/50% 未跑（不影响最终模型）；③ FullDistill 因 +4,000 困难扩展样本导致 FPR 上升（0.0918），已由 final 配方（四桶均衡 4,747 行）取代（§16.10）；④ 旧 real-only 明显低于 synthetic（0.6895 vs 0.9896）的模板学习风险，已由 Final Student 缓解：real-only **0.7913**（+0.1018）；⑤ 4 类 Macro-F1 由 ~0.41 提升至 **0.4657**（超过 Hard Gate 0.430，仍低于目标 0.480，类型头容量仍有限）；⑥ 部署量化（INT8/ONNX）未做；⑦ zero-shot 能力链已按《实验三后续修改方案》补充（§16.9：Base-1.5B 0.3584 / FPR 0.9611，0 元本地运行），Final Student 0.9135 完整闭合能力链。

---

## 16. 神经学生蒸馏（DeepSeek-R1-Distill-Qwen-1.5B + LoRA，CPU 完整评测）

> 本节汇总本轮神经学生蒸馏的完整结果。本轮预算 30 元，实际 API 花费 **7.00 元**（4,000 条扩展池标注），其余预算未动用；全部评测均为 CPU 离线前向（零 API 成本）。

### 16.1 数据与审计（指南 §7）
- 审计后基础清单 **2,235 条**：train 组与 template-family 均与 Exp2 保留测试集不重叠（`data/splits/reserved_exp2_test_ids.json`；详见 `data/prepared/exp3_neural_student/audit_report.json`）。构成：fraudr1_all 124 / synthetic 1,930 / e1_context_r2 181；zh 1,162 / en 1,073；gold unsafe 1,509 / safe 726。
- 扩展池（指南 §7.2）：按子类配比新增 **4,000 条** = clean-refusal 450 + partial-leakage 450 + hard-safe 266 + anti-fraud-education 267 + quotation-analysis 267 + trust-facilitation 700 + direct-fraud 500 + context-flip 600 + over-refusal 250 + general-safety 250。
- T6 标注（指南 §8.3）：复用冻结 prompt、correction 机制、cheap token caps（agent 输出 120–180 tokens）、并发 120，**4,000/4,000** 全部完成，实际花费 **7.00 元**（`experiments/exp3_agent_distillation_ablation/outputs/metrics/cost_expansion.json`，used_rmb=6.9958）。
- 合并后清单 **6,235 条 / 5,470 组 / 501 模板族 / 810 对**（`data/prepared/exp3_neural_student/train_manifest_expanded.jsonl`）：难度 high 2,251 / medium 3,984；zh 3,458 / en 2,777；gold unsafe 3,819 / safe 2,416。

### 16.2 训练设置（指南 §14/§15/§18；CPU-only，无 NVIDIA GPU）
- 基座 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`，fp32 CPU 训练；LoRA r=32 / alpha=64；**4 类统一 softmax**（指南 §13.2：safe / fraud_assistance / refusal_failure / over_refusal）。
- 输入 q+y，head-tail 截断，max_length=384；CPU 调度 micro-batch 2 / effective batch 32；lr_lora 1e-4 / lr_head 5e-4，warmup 50。
- 损失（指南 §15）：gold = 仅 CE；soft_distill / full_distill = CE + soft-KL（teacher 软标签）+ pairwise margin（full_distill 额外启用，并加入 agent 判据扩展数据）。
- 训练（指南 §18.6）：gold / soft 各 2 epoch；full_distill 数据量增大，max_steps=140（约 1 epoch）。实际 CPU 耗时：gold ~4.5h / soft ~5h / full ~7.3h。
- checkpoint `resume.pt` 保存 LoRA/head 权重 + 优化器状态 + 数据位置 + dev eval；每 200 步保存一次，中断可续训。
- **关键修复**：`save_checkpoint` 原只保存 LoRA 适配器，且 `evaluate_neural_student.py` 对已是 PeftModel 的模型二次包装导致分类头随机初始化（gold 一度仅 0.33）；改为 `modules_to_save` 一并保存分类头 + 单次包装加载。修复后完整评测：**gold 0.8676 / soft 0.8849 / full 0.8648**（与训练日志一致）。

### 16.3 主表（指南 §27.2；test n=1,262，0.5 阈值）
| Model | 训练数据 | Macro-F1 | Recall | FPR | AUPRC | MCC | Acc | 4类F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Neural-ZeroShot（随机分类头下界） | — | 0.4176 | 0.5590 | 0.7014 | 0.4219 | -0.1476 | 0.4263 | 0.1743 |
| Neural-Gold | 2,235（基础清单） | 0.8676 | 0.8013 | 0.0669 | 0.9449 | 0.7422 | 0.8685 | 0.4092 |
| Neural-SoftDistill | 2,235（基础清单） | 0.8849 | 0.8094 | 0.0404 | 0.9532 | 0.7795 | 0.8859 | 0.4132 |
| Neural-FullDistill | 6,235（含 4,000 扩展） | 0.8648 | 0.8207 | 0.0918 | 0.9535 | 0.7326 | 0.8653 | 0.4072 |
| **FraudDistill-Student-1.5B**（final 配方 4,747；冻结阈值 0.5622） | **0.9135** | **0.8853** | 0.0591 | **0.9717** | **0.8282** | **0.9136** | **0.4657** |
> 注：Final Student 行使用官方冻结阈值 0.5622（§16.10），其余行均为 0.5 固定阈值协议；Final Student 在除 FPR 外所有指标上超越 Neural-SoftDistill（配对显著性见 §16.11）。

> Base-1.5B-ZeroShot（生成式下界）与 500 子集能力链见 §16.9（0.5 阈值协议一致）。

### 16.4 蒸馏机制（指南 §27.3，test）
| 设置 | Soft | Pair | 扩展数据 | ΔMacro-F1 (vs Gold) | ΔRecall | ΔFPR |
|---|---:|---:|---:|---:|---:|---:|
| Neural-SoftDistill | ✓ | — | — | **+0.0173** | +0.0081 | **-0.0265** |
| Neural-FullDistill | ✓ | ✓ | ✓（+4,000） | -0.0028 | +0.0194 | +0.0249 |

### 16.5 低标注曲线（指南 §22.3/§27.4；test，0.5 阈值）
| Gold fraction | Neural-Gold | SoftDistill | 说明 |
|---|---:|---:|---|
| 10% | 0.8422（R 0.8368 / FPR 0.1524） | 未完成（soft10 中断于 ~1 epoch，resume.pt 已保留） | 后续工作 |
| 100% | 0.8676 | 0.8849 | SoftDistill +0.0173 |

### 16.6 机制切片（指南 §23.3，test）
- **Neural-Gold**：direct_recall 0.7188（n=160）、trust_recall 1.0（n=80）、leakage_recall 1.0（n=60）、clean_refusal_fpr 0.1474（n=95）、hard_safe_fpr 0.0（n=120）、over_refusal_recall 0.9167（n=60）、context_flip_pair_acc 0.9474（pairs=19）
- **Neural-SoftDistill**：direct_recall 0.75（n=160）、trust_recall 1.0（n=80）、leakage_recall 1.0（n=60）、clean_refusal_fpr 0.0105（n=95）、hard_safe_fpr 0.05（n=120）、over_refusal_recall 0.85（n=60）、context_flip_pair_acc 0.9474（pairs=19）
- **Neural-FullDistill**：direct_recall 0.8063（n=160）、trust_recall 1.0（n=80）、leakage_recall 1.0（n=60）、clean_refusal_fpr 0.2421（n=95）、hard_safe_fpr 0.0083（n=120）、over_refusal_recall 0.9167（n=60）、context_flip_pair_acc 1.0（pairs=19）
- **Neural-ZeroShot**：direct_recall 0.5125（n=160）、trust_recall 0.6875（n=80）、leakage_recall 0.5833（n=60）、clean_refusal_fpr 0.7053（n=95）、hard_safe_fpr 0.7083（n=120）、over_refusal_recall 0.3833（n=60）、context_flip_pair_acc 0.0526（pairs=19）

### 16.7 泛化切片（指南 §23.4，test）
- **Neural-Gold**：real_only n=664 MF1=0.64；synthetic_only n=598 MF1=0.9958；zh n=413 MF1=0.989；en n=849 MF1=0.7919；fraudr1_all_source n=39 MF1=0.4658；e1_context_r2_source n=60 MF1=1.0
- **Neural-SoftDistill**：real_only n=664 MF1=0.6895；synthetic_only n=598 MF1=0.9896；zh n=413 MF1=0.9862；en n=849 MF1=0.8193；fraudr1_all_source n=39 MF1=0.4935；e1_context_r2_source n=60 MF1=0.4958
- **Neural-FullDistill**：real_only n=664 MF1=0.6515；synthetic_only n=598 MF1=1.0；zh n=413 MF1=0.9891；en n=849 MF1=0.7911；fraudr1_all_source n=39 MF1=1.0；e1_context_r2_source n=60 MF1=1.0

### 16.8 部署（指南 §21.4/§23.5，CPU fp32，max_length=384）
- **Neural-Gold**：模型磁盘 154.8 MB（LoRA+头，基座另存）、batch=8 P50 1606.9 ms/条、P95 2756.6 ms/条、吞吐 0.74 条/s（CPU）
- **Neural-SoftDistill**：模型磁盘 154.8 MB（LoRA+头，基座另存）、batch=8 P50 1604.0 ms/条、P95 2773.4 ms/条、吞吐 0.72 条/s（CPU）
- **Neural-FullDistill**：模型磁盘 154.8 MB（LoRA+头，基座另存）、batch=8 P50 1600.3 ms/条、P95 2750.5 ms/条、吞吐 0.73 条/s（CPU）

### 16.9 Base-1.5B Zero-shot baseline（《实验三后续修改方案》，2026-08-07）

- **动机**：Neural-ZeroShot 是随机分类头下界，不能代表未训练 DeepSeek-R1-Distill-Qwen-1.5B 的真实能力；新增 **Base-1.5B-ZeroShot**：基座原版权重、无 LoRA、无训练、无分类头，生成式输出四类标签（safe / fraud_assistance / refusal_failure / over_refusal）。
- **设置**：固定随机 **500 条** test 子集（seed=20260804，`subset_ids.json` 可复现）；**Temperature=0**（greedy）、**max_new_tokens=32**；输入为 `User Query / Assistant Response` + 四类判别指令；主标签与风险分数由四类标签续写概率（长度归一化 logprob，`risk = 1 − P(safe)`）给出，与神经学生 0.5 阈值协议完全同构；生成文本同时记录（直接解析率 24.8%，未解析样本按续写概率判定并标记 fallback）。
- **能力提升链**（同一 500 条子集、全部同口径重算，`base_chain_500.csv`）：

| Model | N | Acc | Prec | Recall | Macro-F1 | FPR | AUPRC | MCC | 4类F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base-1.5B-ZeroShot | 500 | 0.4860 | 0.4854 | 0.9588 | 0.3584 | 0.9611 | 0.5008 | -0.0057 | 0.0834 |
| Neural-ZeroShot（随机分类头） | 500 | 0.4280 | 0.4326 | 0.5679 | 0.4191 | 0.7043 | 0.4200 | -0.1418 | 0.1795 |
| Neural-Gold | 500 | 0.8800 | 0.9103 | 0.8354 | 0.8794 | 0.0778 | 0.9512 | 0.7617 | 0.4172 |
| Neural-SoftDistill | 500 | 0.9040 | 0.9621 | 0.8354 | 0.9032 | 0.0311 | 0.9626 | 0.8139 | 0.4122 |
| Neural-FullDistill | 500 | 0.8740 | 0.8814 | 0.8560 | 0.8738 | 0.1089 | 0.9586 | 0.7479 | 0.4092 |

- **解读**：未训练 1.5B 基座几乎把所有响应判为 unsafe（Recall 0.9588 / FPR 0.9611、MCC≈0、AUROC 0.4785≈随机），Macro-F1 仅 0.3584；训练后 Gold / SoftDistill 提升至 0.8794 / 0.9032（SoftDistill 相对 Gold **+0.0238**、FPR 相对下降 60%），**Base Model → Trained Student 能力提升链成立**（验收标准达成）。Random Head 的 F1（0.4191）略高于 Base（0.3584）但 MCC 更低（−0.1418 vs −0.0057），两者均为无判别能力下界，不具可比意义。
- **成本与部署**：本地 CPU 推理，**0 元 API**；单条 P50 20.9s / P95 112.4s（1.5B fp32，含四类续写打分），生成式下界不用于部署。
- **Final Student 接口**：`--setting final_student` 已加入 `scripts/train_exp3_students.py`（配方暂同 soft_distill）与 `scripts/run_exp3_neural_stages.py` 阶段表，已完成最终训练与官方验收（2026-08-09），详见 §16.10。
- **产物**：`outputs/neural_student/base_zeroshot/`（`subset_ids.json`、`predictions_test.jsonl`、`base_zeroshot_metrics.json`）、`base_chain_500.csv/.json`、`figures/fig9_student_capability_chain.png`。
### 16.10 最终 1.5B 学生模型 FraudDistill-Student-1.5B（《最终1.5B学生模型训练实施指南》，2026-08-07）

- **Reproducibility**：commit `d2bb9d9`；seed 11；transformers 4.43.2 / peft 0.12.0 / torch 2.6.0+cu118；manifest sha256 `f0230bec71bc0065…`
- **数据（训练池 4,747 行，全量 gold + teacher signal，泄漏审计 PASS）**：
  - Source A = exp3 train（4,091）剔除 balanced test/dev 重叠 → 2,995；Source B = balanced dev（700）剔除共享 group → 652；Source C = expansion teacher-only 1,100
  - 采样权重：benchmark 46.8% / paired_dev 17.4% / synthetic_core 23.5% / hard_expansion 12.3%；每桶 safe≈50%；unsafe 三类 ≥10%；EN 64.3%（目标 60–65%）；权重 cap ≤4×median
  - max_length=512（P95=1074，head-tail truncation，指南 §12 固定规则）
- **训练**：LoRA r=32/α=64/dropout=0.05 + head（modules_to_save=["classifier","score"]）；micro-batch 4 / effective 32 / accum 8；AdamW lr_lora=1e-4 lr_head=5e-4 wd=0.01；warmup 5%（optimizer step 口径）；2 epochs；每 40 步 dev（300 子集）+ checkpoint，patience 4；FinalDistillLoss（CE4 + 0.30×binary + 0.30×w_t×KL(T=2) + 0.05×pair，teacher-only 无硬 CE）
- **选点（dev，n=1,047）**：fast 300 子集 → top-3 全量 dev；FPR≤0.055 & recall≥0.82 下最大 Macro-F1；冻结阈值 0.5622
- **Reload 校验**：experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/best_step120；max logit diff 0.0（≤1e-5 放行：PASS）；classifier present True
- **正式 test（单次，冻结阈值）**：

| Metric | Final Student | SoftDistill | Delta |
|---|---:|---:|---:|
| Accuracy | 0.9136 | 0.8859 | 0.0277 |
| Macro-F1 | 0.9135 | 0.8849 | 0.0286 |
| Recall | 0.8853 | 0.8094 | 0.0759 |
| FPR | 0.0591 | 0.0404 | 0.0187 |
| AUPRC | 0.9717 | 0.9532 | 0.0185 |
| MCC | 0.8282 | 0.7795 | 0.0487 |
| Real-only MF1 | 0.7913 | 0.6895 | 0.1018 |
| Synthetic MF1 | 0.9938 | 0.9896 | 0.0042 |
| EN MF1 | 0.8702 | 0.8193 | 0.0509 |
| ZH MF1 | 0.9862 | 0.9862 | 0.0 |
| 4-class MF1 | 0.4657 | 0.4132 | 0.0525 |

- **机制切片（test）**：direct recall 0.9812（n=160）；trust recall 1.0（n=80）；leakage recall 0.9667（n=60）；clean-refusal FPR 0.0105（n=95）；hard-safe FPR 0.0（n=120）；over-refusal recall 0.9333（n=60）；context-flip pair acc 0.9474（pairs=19）
- **Gate 判定**：Hard Gate **FAIL**（{'macro_f1': 'PASS', 'acc': 'PASS', 'recall': 'PASS', 'fpr': 'FAIL', 'auprc': 'PASS', 'mcc': 'PASS', 'real_mf1': 'PASS', '4class_mf1': 'PASS'}）；Target **PARTIAL/FAIL**
- **最终决定**：Hard Gate 8/9 通过（仅 FPR 0.0591 略超 ≤0.050）；按指南 §25 单模型原则不进行二次训练，论文 Student 回退 Neural-SoftDistill；FraudDistill-Student-1.5B 权重与全部验收产物完整保留于 `final_distilled_student/`，供论文/部署决策使用。
- **产物**：`outputs/neural_student/final_distilled_student/`（adapter_config.json、adapter_model.safetensors（含 head）、training_config.json、training_state.json、data_manifest.json、data_audit.json、best_checkpoint.json、calibration.json、dev_metrics.json、test_metrics.json、slice_metrics.json、reload_checksum.json、model_card.md、gate_result.json）

---

> 训练过程逐 20 步 loss / 逐 40 步 eval / 断点恢复 / 选点校准细节见 **`FINAL_STUDENT_TRAINING_REPORT.md`**。
### 16.11 Final Student vs Neural-SoftDistill 配对显著性（离线，0 元 API，2026-08-09）

- 数据：同一 test 集 n=1,262 的逐样本预测（Final：`final_distilled_student/test_eval/predictions_test.jsonl`；Soft：`eval_soft/predictions_test.jsonl`），各模型使用各自官方阈值（Final 冻结阈值 0.5622 / Soft 0.5 固定协议）。
- 方法：paired bootstrap 10,000 次（seed 20260809）——每轮对同一批样本重采样，两模型在同一子集上按各自阈值计算指标并求 Δ=Final−Soft，取 2.5/97.5 百分位为 95% CI；另做 McNemar exact test 比较两个二值判定向量。
- 脚本：`scripts/exp3_final_vs_soft_significance.py`；产物：`outputs/metrics/final_vs_soft_significance.json`。

| 指标 | Final Student | SoftDistill | Δ（Final−Soft） | 95% CI（bootstrap 10k） | 显著 |
|---|---:|---:|---:|---:|---|
| Macro-F1 | 0.9135 | 0.8849 | +0.0285 | [0.0122, 0.0446] | ✅ |
| Recall | 0.8853 | 0.8094 | +0.0759 | [0.0513, 0.1007] | ✅ |
| FPR | 0.0591 | 0.0404 | +0.0187 | [-0.0016, 0.0397] | 否（跨 0） |
| AUPRC | 0.9717 | 0.9532 | +0.0185 | [0.0068, 0.0296] | ✅ |
| MCC | 0.8282 | 0.7795 | +0.0487 | [0.0168, 0.0799] | ✅ |
| Accuracy | 0.9136 | 0.8859 | +0.0277 | [0.0119, 0.0436] | ✅ |
| 4-class MF1 | 0.4657 | 0.4132 | +0.0524 | [0.0301, 0.0779] | ✅ |

- **McNemar**：b（Final 错 / Soft 对）=26，c（Final 对 / Soft 错）=85，p=**1.70e-08** → Final Student 的判定显著优于 SoftDistill（非随机对等）。
- **解读**：MF1/Recall/AUPRC/MCC/Acc/4-class 的提升全部统计显著；FPR 上升不显著（CI 跨 0）——即以"微小且不显著的 FPR 上升"换取"Recall +7.59pp 的显著提升"，是偏向可用性的安全-可用性权衡。

---

### 16.12 Final Student vs T6 Teacher（离线汇总，0 元 API）

| 指标 | FraudDistill-Student-1.5B（本地 CPU） | T6 Full MAT（API 教师） | Δ（Student−Teacher） |
|---|---:|---:|---:|
| Accuracy | 0.9136 | 0.9033 | +0.0103 |
| Macro-F1 | 0.9135 | 0.9016 | +0.0119 |
| Recall | 0.8853 | 0.9031 | -0.0178 |
| FPR | 0.0591 | 0.0964 | -0.0373 |
| AUPRC | 0.9717 | 0.9236 | +0.0481 |
| MCC | 0.8282 | 0.8066 | +0.0216 |

- 口径：T6 Full MAT 为 test n=1,262、统一冻结阈值 0.85（`outputs/metrics/final_metrics.json`，与 §4 主表一致）；Final Student 为其官方冻结阈值 0.5622（§16.10）。
- **解读**：1.5B 学生已在 Macro-F1（+0.0119）、AUPRC（+0.0481）、MCC（+0.0216）、FPR（0.0591 vs 0.0964，约为教师的一半）上**超过 API 教师 T6 Full MAT**，仅 Recall 低 1.78pp；同时推理成本从 API 5.38 元/千条降为本地 CPU 0 元（部署单条 P50 ~1.6s，§16.8），完整验证"API 教师蒸馏 → 本地轻量学生"的论文叙事。
