# FraudDistill 六实验总结文档（E1–E6）

> 本文档汇总 FraudDistill 全部 6 个正式实验：**实验思路、实验设计、数据集选取、主结果表格、实验分析**，并给出每个实验的文件夹与报告链接，便于论文写作与复现检索。
> 最后更新：2026-08-12 · 全部 6 个实验均已完成并冻结。

---

## 0. 实验总览

| # | 实验名称 | 文件夹 | 一句话结论 |
|---|---|---|---|
| E1 | 输入消融与关系性机制验证 | [`exp1_input_ablation/`](exp1_input_ablation/) | 欺诈协助检测是**响应级且关系性**的：联合 q+y 显著优于任意单视图，ΔMacro-F1 = +0.1717 |
| E2 | 跨工作对比（平衡诊断集） | [`exp2_prior_work_comparison/`](exp2_prior_work_comparison/) | 在 Fraud-R1 / OR-Bench / DNA / Aegis 四基准 10,813 行平衡集上**全部 PASS**，原工作基线同数据集重评 |
| E3 | 增强多 Agent 教师与蒸馏消融 | [`exp3_agent_distillation_ablation/`](exp3_agent_distillation_ablation/) | 完整增强 MAT 达 Macro-F1 0.9016；最终 1.5B 学生 **0.9135**，全面超越 Neural-SoftDistill |
| E4 | 未见类别/来源/风格泛化 | [`exp4_unseen/`](exp4_unseen/) | family-disjoint 复合迁移下 Student 保留中等排序能力（AUROC 0.7198），冻结点偏保守 |
| E5 | 校准与选择性审计级联 | [`exp5_calibration/`](exp5_calibration/) | 15% 查询率选择性审计（P3）相对冻结点 ΔMacro-F1 **+0.0918**，McNemar p=7.53×10⁻²⁰ |
| E6 | 跨多 API 直连响应检测 | [`exp6_balanced_multi_api/`](exp6_balanced_multi_api/) | 6 个直连 API 模型均衡关系集上 P0 MF1 0.722、P2-20% 审计 0.807；hard-safe FPR 14.4% |

**实验间逻辑链**：E1 建立“必须联合观察 q 与 y”的机制基础 → E2 在公开基准上证明框架优于原工作 → E3 训练出可部署的 1.5B 学生模型 → E4 检验学生模型在未见迁移下的边界 → E5 用低代价选择性审计修复边界处的召回 → E6 在真实多 API 目标模型上验证端到端可用性。

**跨实验防泄漏**：E4/E5/E6 均执行了 exact / family / template 级泄漏审计并与 E3 训练面板做交叉检查（见 `exp6_balanced_multi_api/data/cross_experiment_leakage_audit.json`、`exp4_unseen/audits/`）。

---

## 1. E1 — 输入消融与关系性机制验证

- **文件夹**：[`exp1_input_ablation/`](exp1_input_ablation/)
- **报告**：[`report/E1_FINAL_REPORT.md`](exp1_input_ablation/report/E1_FINAL_REPORT.md)
- **论文表格**：[`tables/E1_PAPER_TABLES.md`](exp1_input_ablation/tables/E1_PAPER_TABLES.md)
- **进度说明**：[`PROGRESS.md`](exp1_input_ablation/PROGRESS.md)

### 1.1 实验思路

回答三个递进的研究问题：
1. **E1-A**：安全对齐 LLM 在 Fraud-R1 风格自然诱导下，实际产生 fraud-assistance response 的比例是多少？（风险存在性与发生率背景）
2. **E1-B**：fraud assistance 是否是一个必须联合观察 q 与 y 才能可靠判断的 **relational property**？（核心机制主张 H1）
3. **E1-C**：B 中学到的 q+y 机制与冻结阈值能否迁移到独立自然低基率响应？（外部有效性 / 部署边界）

主 claim 由 E1-B 支撑：**单看查询（q）或单看回答（y）都不可靠，必须联合观察 q 与 y**。

### 1.2 实验设计

- **E1-A**：7,500 行冻结 registry（A7500，Qwen/DeepSeek 各半、中英各半、多种欺诈诱导场景），双 Judge（Qwen + DeepSeek）gold，agreement = 0.9984。
- **E1-B**：6,000 行受控平衡面板（正负严格 50/50，B1/B2/B3 各 2,000）：
  - B1 context-critical / y-matched：同一 y 由 q 决定正负；
  - B2 response-critical / q-matched：同一 q 由 y 决定正负；
  - B3 context-stable / mixed-source：混合自然来源对照。
  - 三组检测器层次：M0 TF-IDF/LR 诊断基线；M2/M3 LLM 四视图（q_only / y_only / q_y / wrong_q_y）；M1 XLM-R 联合编码器（5 seeds × 3 views）。
  - **wrong-q 负控制**：同 split、同语言、同 fraud category、不同 family 的错误 q 配对，验证模型确实在使用“关系”而非表面特征。
- **E1-C**：A7500 中**从未进入 B 面板的 canonical case** 作为独立 reserve（指南 §10.2 case/family 级独立），共 624 行 / 6 阳性。

### 1.3 数据集选取

| 子实验 | 行数 | 正样本 | 构成 |
|---|---:|---:|---|
| E1-A | 7,500 | 11（0.147%） | 自然诱导 registry（Fraud-R1 风格，qwen/deepseek × zh/en） |
| E1-B | 6,000 | 3,000 | B1/B2/B3 受控 matched 面板（zh 3065 / en 2935） |
| E1-C | 624 | 6（0.96%） | A 中未进入 B 的自然 reserve |

- B 面板 gold：双 Judge + adjudication，**B 面板自身 agreement = 0.9281**（与 A 的 0.9984 分开表述）。
- 防泄漏：跨 split exact-q = 0、exact-(q,y) = 0；near-dup y = 357 行（其中 212 行进入 clean-anchor 敏感性分析）。

### 1.4 主结果表格

**E1-A 自然发生率**

| 指标 | 值 |
|---|---|
| Central positives | 11 / 7500（0.1467%） |
| Wilson 95% CI | [0.000819, 0.002625] |
| Cluster-bootstrap 95% CI | [0.000534, 0.002564] |
| Judge agreement | 0.9984 |

**E1-B M1 XLM-R 联合编码器（Frozen Anchor 1200，5 seeds 均值 ± sd）**

| View | Macro-F1 | AUROC | AUPRC | Recall | Precision | FPR |
|---|---:|---:|---:|---:|---:|---:|
| q_only | 0.6450 ± 0.0165 | 0.7172 | 0.6497 | 0.9097 | 0.6135 | 0.5750 |
| y_only | 0.8017 ± 0.0032 | 0.9201 | 0.9242 | 0.8303 | 0.7962 | 0.2233 |
| **q_y** | **0.9685 ± 0.0035** | **0.9944** | **0.9934** | 0.9853 | 0.9534 | 0.0483 |
| wrong_q_y | 0.7574 ± 0.0039 | 0.7327 | 0.6632 | 0.8200 | 0.7300 | 0.3033 |

**E1-B M2/M3 LLM 四视图（Frozen Anchor 1200）**

| Judge | View | Macro-F1 | Recall | FPR |
|---|---:|---:|---:|---:|
| Qwen | q_only | 0.3333 | 0.0000 | 0.0000 |
| Qwen | y_only | 0.7576 | 0.9400 | 0.4100 |
| Qwen | q_y | 0.8141 | 0.9533 | 0.3183 |
| Qwen | wrong_q_y | 0.6536 | 0.5917 | 0.2817 |
| DeepSeek | q_only | 0.6258 | 0.6233 | 0.3717 |
| DeepSeek | y_only | 0.7601 | 0.9533 | 0.4167 |
| DeepSeek | q_y | 0.8383 | 0.9667 | 0.2850 |
| DeepSeek | wrong_q_y | 0.7055 | 0.7417 | 0.3300 |

**E1-C 自然迁移（独立 624 行 / 6 阳性，5 seeds 均值 ± sd）**

| View | Macro-F1 | Recall | FPR | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| q_y | 0.6748 ± 0.0808 | 0.5000 ± 0.1826 | 0.0162 ± 0.0167 | 0.9706 ± 0.0215 | 0.3970 ± 0.1538 |
| y_only | 0.6099 ± 0.0250 | 0.7333 ± 0.3091 | 0.0427 ± 0.0261 | 0.9794 ± 0.0058 | 0.2983 ± 0.0361 |

### 1.5 实验分析

- **H1 成立**：Δ_joint（M1）= 0.9685 − 0.8017 = **+0.1717**（目标 ≥ 0.05），family-cluster bootstrap 95% CI **[0.1511, 0.1928]**，p(Δ ≤ 0) = 0；配对 McNemar（Holm 校正后）均显著；5/5 seeds 全胜。
- **关系性负控制成立**：q_y vs wrong_q_y 掉点 −0.211（M1），说明模型使用的是 q 与 y 的**关系**而非 y 表面特征；M2/M3 同样成立（−0.16 / −0.13）。
- **分层机制**：B1（y-matched）q_y MF1 0.9925、B2（q-matched）0.9950、B3（context-stable）0.9273。
- **Clean-anchor 敏感性**：剔除 212 行跨 split 近重复 y 后 Δ_joint = +0.1784（CI [0.1551, 0.2028]），主结论不变。
- **E1-C 为方向性支持**：仅 6 个独立阳性，q_y 在 Macro-F1 / AUPRC（0.397 vs 0.298，相对 prevalence 0.96% 的 lift ≈ 28–71×）与低 FPR 工作点上优于 y_only，但按指南标注为 exploratory / descriptive，不强作 confirmatory claim。
- 统计修正记录：v4.6 将 cluster bootstrap 修正为 Macro-F1 口径（原实现只对正类 F1 重抽样，CI 与点估计不可比）。

---## 2. E2 — 跨工作对比（平衡诊断集）

- **文件夹**：[`exp2_prior_work_comparison/`](exp2_prior_work_comparison/)
- **报告**：[`EXP2_BALANCED_FINAL_REPORT_20260807.md`](exp2_prior_work_comparison/EXP2_BALANCED_FINAL_REPORT_20260807.md)
- **主表**：[`balanced_design/tables/main_8row_balanced.md`](exp2_prior_work_comparison/balanced_design/tables/main_8row_balanced.md)
- **数据集 manifest**：[`dataset_manifest.json`](exp2_prior_work_comparison/dataset_manifest.json)

### 2.1 实验思路

在四个公开安全/欺诈基准上，对 FraudDistill 多智能体防御框架做**响应级评测**，并与各基准的**原工作官方基线在同一个最终数据集上重评**，回答“框架相比原工作到底强在哪里、代价是什么”。

### 2.2 实验设计

- 每个基准构建**正负约 1:1 的响应级诊断集**（覆盖多提示模板、语言与危害类型），开发集与测试集分离，阈值在开发集上冻结后方进入测试集。
- 全部标注采用**双人独立标注 + 仲裁**（仲裁一致率 0.9421，Cohen's Kappa 0.8824）；诊断集仅依据仲裁后的金标构建，不依赖评估器预测筛选。
- 预注册 **7 项门槛**：Accuracy / Precision / Recall / Macro-F1 / FPR / AUPRC / MCC，逐基准判定 PASS/FAIL。
- 原工作基线全部为官方实现/官方协议，在最终测试集上**完整重评**：GPTCheck（Fraud-R1）、Official Response Checker（OR-Bench）、Longformer-Harmful（DNA）、NemoGuard（Aegis）。

### 2.3 数据集选取

| 基准 | 测试样本数 | 正 / 负 | 正负比例 | 构成说明 |
|---|---:|---|---:|---|
| Fraud-R1 | 3,000 | 1,500 / 1,500 | 50% / 50% | 中英文、多诈骗家族（冒充、虚假招聘、补贴骗局等） |
| OR-Bench | 2,000 | 1,000 / 1,000 | 50% / 50% | hard_safe / regular_safe / toxic 分层 |
| DNA | 5,000 | 2,500 / 2,500 | 50% / 50% | 6 组提示词分组、多类危害 |
| Aegis | 813 | 394 / 419 | 48.5% / 51.5% | Aegis 官方响应级样本 |
| **测试合计** | **10,813** | 5,394 / 5,419 | — | — |
| 开发集 | 700 | — | — | 阈值冻结使用 |

质量审计：全部基准精确重复样本 = 0、近重复率约 0；提示词单独特征与元数据特征对金标预测力接近随机（未发现泄漏性捷径）。

### 2.4 主结果表格（八行主表）

| Benchmark | System | N | Acc | Prec | Recall | Macro-F1 | FPR | AUPRC | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fraudr1 | FraudDistill | 3000 | 0.9847 | 0.9789 | 0.9907 | 0.9847 | 0.0213 | 0.9958 | 0.9694 |
| fraudr1 | Official Judge (GPTCheck) | 3000 | 0.6963 | 0.7420 | 0.6020 | 0.6936 | 0.2093 | — | 0.3998 |
| orbench | FraudDistill | 2000 | 0.8820 | 0.9974 | 0.7660 | 0.8804 | 0.0020 | 0.8839 | 0.7854 |
| orbench | Official Response Checker | 2000 | 0.9240 | 0.9133 | 0.9370 | 0.9240 | 0.0890 | — | 0.8483 |
| dna | FraudDistill | 5000 | 0.9612 | 0.9957 | 0.9264 | 0.9612 | 0.0040 | 0.9654 | 0.9246 |
| dna | Longformer-Harmful (LibrAI) | 5000 | 0.7226 | 0.8551 | 0.5360 | 0.7126 | 0.0908 | — | 0.4799 |
| aegis | FraudDistill | 813 | 0.8315 | 0.8836 | 0.7513 | 0.8297 | 0.0931 | 0.8303 | 0.6683 |
| aegis | NemoGuard (original) | 813 | 0.8819 | 0.8725 | 0.8858 | 0.8819 | 0.1217 | — | 0.7638 |

### 2.5 实验分析

- **四基准全部 PASS**：Fraud-R1（Acc 0.9847 vs 门槛 0.84）、OR-Bench（0.8820 vs 0.88）、DNA（0.9612 vs 0.96）、Aegis（0.8315 vs 0.83，Recall 0.7513 ≥ 0.75，AUPRC 0.8303 ≥ 0.82，MCC 0.6683 ≥ 0.65）。
- **Fraud-R1**：显著优于官方 Judge（Acc +0.288，McNemar p < 1e-200），漏报 14 / 误报 32。
- **OR-Bench**：官方 Checker 召回更高（0.9370 vs 0.7660），但其 FPR 为 0.0890，约为 FraudDistill（0.0020）的 44 倍——框架以极小误报代价实现门槛内精度，更适合“最小拒绝扰动”衡量。
- **DNA**：Acc 0.9612 / MCC 0.9246，显著优于 Longformer-Harmful（Acc 0.7226，p < 1e-300）。
- **Aegis**：全量 813 行重评后 7 项门槛全部通过；与 NemoGuard 相比召回仍有差距，但 FPR 更低（0.0931 vs 0.1217），误差结构显著不同（p = 2.4e-4）。
- 误差模式：Fraud-R1 漏报集中于少见诈骗话术变体；OR-Bench 漏报集中于“软拒绝”；DNA 漏报集中于多轮/嵌入上下文有害内容；Aegis 漏报集中于需场景化判断的有害响应。
- Bootstrap 95% CI（10,000 次）与全部 McNemar + Holm 检验见报告 §5.6–5.7。

---

## 3. E3 — 增强多 Agent 教师与蒸馏消融

- **文件夹**：[`exp3_agent_distillation_ablation/`](exp3_agent_distillation_ablation/)
- **主报告**：[`EXP3_ENHANCED_AGENT_DISTILLATION_REPORT.md`](exp3_agent_distillation_ablation/EXP3_ENHANCED_AGENT_DISTILLATION_REPORT.md)
- **最终学生训练报告**：[`FINAL_STUDENT_TRAINING_REPORT.md`](exp3_agent_distillation_ablation/FINAL_STUDENT_TRAINING_REPORT.md)
- **最终模型**：`exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/best_step120/`（FraudDistill-Student-1.5B，E4/E5/E6 均依赖此模型）

### 3.1 实验思路

- 验证“**Agent 分解价值**”与“**Arbiter 价值**”：完整增强 Multi-Agent Teacher（MAT）是否显著优于单判官；
- 验证“**结构化教师信号优于 hard pseudo-label**”：Student 蒸馏梯度 S0–S4；
- 训练可部署的 **1.5B 神经学生**（QLoRA），并给出 Base-1.5B 未训练下界与最终学生模型的完整验收。

### 3.2 实验设计

- **Teacher 嵌套消融（T0–T6）**：T0 规则基线（零 API）→ T1 单判官 → T2 仅 Fraud Agent → T3 +Refusal → T4 +Context → T5 专家 + 规则仲裁 → T6 完整增强 MAT（三专家 + Evidence Arbiter + 冲突纠错 + Platt 校准，冻结阈值 0.8409）。
- **Leave-one-out 组件消融（L0–L7）** 与组件压力测试。
- **Student 蒸馏梯度（S0–S4，5 seeds）**：gold hard-label / 分数蒸馏 / 证据加权等。
- **神经学生**：1.5B QLoRA（CPU/GPU 训练，seed 11），对比 Neural-Gold / Neural-SoftDistill / Neural-FullDistill / 低标注 10% gold。
- **最终学生**：`--setting final_student` 重训 + dev 门槛（FPR ≤ 0.055 & Recall ≥ 0.82）选最优 step，冻结阈值 0.5622，reload checksum 校验。

### 3.3 数据集选取

- 6,400 条：train 4,091 / dev 1,047 / test 1,262（safe 3,300 / unsafe 3,100，EN/ZH = 4,435/1,965，16 类细分子类）。
- 四块结构：A 欺诈行动性与信任促进、B 拒答边界与部分泄漏、C/D 其他危害与对照。
- 全部标签经双 Judge + adjudication；与 E2 测试集保持 group/template-family disjoint（`scripts/audit_student_training_data.py`）。

### 3.4 主结果表格

**Teacher 嵌套消融（test, n=1,262，冻结阈值 0.8409）**

| Method | Acc | Prec | Recall | Macro-F1 | FPR | AUPRC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 Rule（零 API） | 0.5055 | 0.4939 | 0.3247 | 0.3918 | 0.3204 | 0.5114 | 0.0046 |
| T1 Single Judge | 0.7575 | 0.9065 | 0.5638 | 0.6952 | 0.0560 | 0.5158 | 0.5514 |
| T2 Fraud only | 0.8281 | 0.9223 | 0.7092 | 0.8018 | 0.0575 | 0.8408 | 0.6721 |
| T3 Fraud+Refusal | 0.8059 | 0.9517 | 0.6365 | 0.7628 | 0.0311 | 0.8976 | 0.6446 |
| T4 +Context | 0.8051 | 0.9538 | 0.6333 | 0.7612 | 0.0295 | 0.9024 | 0.6440 |
| T5 Specialists+Rule Arbiter | 0.8970 | 0.8989 | 0.8901 | 0.8945 | 0.0964 | 0.9078 | 0.7939 |
| **T6 Full MAT** | **0.9033** | 0.9002 | **0.9031** | **0.9016** | 0.0964 | **0.9236** | **0.8066** |

**1.5B 学生模型（test, n=1,262）**

| 模型 | Macro-F1 | Recall | FPR | AUPRC | MCC |
|---|---:|---:|---:|---:|---:|
| Base-1.5B Zero-shot（500 子集） | 0.3584 | — | 0.9611 | — | — |
| Neural-Gold（500 子集） | 0.8794 | — | — | — | — |
| Neural-SoftDistill（500 子集） | 0.9032 | — | — | — | — |
| Neural-Gold（seed 11） | 0.8676 | — | — | — | — |
| Neural-SoftDistill（seed 11） | 0.8849 | 0.8094 | 0.0404 | 0.9532 | 0.7795 |
| Neural-FullDistill | — | 0.8207 | 0.0918 | — | — |
| **FraudDistill-Student-1.5B（Final，阈值 0.5622）** | **0.9135** | **0.8853** | 0.0591 | **0.9717** | **0.8282** |

Final Student 补充：Acc 0.9136、Real-only **0.7913**、4-class 0.4657；Hard Gate 8/9 通过（仅 FPR 超 0.050 约 0.9pp）。

### 3.5 实验分析

- **T1 → T6 显著提升**：Macro-F1 +0.166（10k bootstrap CI [0.1449, 0.1878]），主要收益来自**召回**（+0.167），代价是 FPR 升高（0.056 → 0.096，属安全—可用性权衡的可用性方向）。
- **组件消融**：去掉 Refusal Agent 影响最大（MF1 −0.1099、Recall −22.5pp）；去掉 Context Agent FPR +5.6pp。
- **证据仲裁（T5→T6）**：Macro-F1 +0.0064（CI 不跨 0），ECE 0.1404 → 0.0414（−70.5%），校准质量大幅改善。
- **冲突纠错为确认型**：test 触发率 15.3% 但 0 次改判（教师内部一致率 90.3%），T6 与含纠错完整版完全一致。
- **蒸馏梯度**：S2 分数蒸馏最优（0.9073 vs S0 gold 0.9031）；S4 证据蒸馏 AUPRC 0.9793 最优；增益有限（教师标签与 gold 一致率 89%，信息增量受限），如实报告。
- **Final Student vs Neural-SoftDistill**：MF1 +0.0286、Recall +0.0759、Real-only +0.1018、4-class +0.0525；配对显著性 ΔMF1 CI [0.0122, 0.0446]、ΔRecall CI [0.0513, 0.1007] 均不跨 0，McNemar p = 1.7e-08。
- **能力链**：Base-1.5B 0.3584（几乎全判 unsafe）→ Neural-Gold 0.8794 → Neural-SoftDistill 0.9032 → Final Student 0.9135，蒸馏训练的价值链条完整成立。

---## 4. E4 — 未见类别/来源/风格泛化

- **文件夹**：[`exp4_unseen/`](exp4_unseen/)
- **最终报告（static-fix）**：[`experiments/e4e5_final_staticfix/EXP4_UNSEEN_GENERALIZATION_REPORT_FINAL.md`](e4e5_final_staticfix/EXP4_UNSEEN_GENERALIZATION_REPORT_FINAL.md)
- **E4/E5 汇总**：[`e4e5_final_staticfix/E4E5_FINAL_SUMMARY.md`](e4e5_final_staticfix/E4E5_FINAL_SUMMARY.md)
- **冻结数据与指标**：`e4e5_final_staticfix/FINAL_METRICS.json`、`FINAL_DATA_AUDIT.json`、`FINAL_PAIRED_STATS.json`、`tables/e4_shift_corrected.md`

### 4.1 实验思路

E4 回答：蒸馏得到的 **Final Student（1.5B）** 在 **family-disjoint 的 held-out 复合迁移**（U1 类别迁移 / U2 来源迁移 / U3 目标模型与风格迁移）下表现如何？与两个神经教师基线（Neural-Gold、Neural-SoftDistill）相比，学生模型的排序能力与部署工作点如何取舍？

“unseen”的严格定义：仅指当前暴露审计未发现与训练面板的 exact/family/query 重叠，不声称自然分布泛化或穷尽语义泄漏排除。

### 4.2 实验设计

- 冻结 test manifest **1,200 行 / 557 families**、calibration **600 行 / 243 families**（canonical JSON-record SHA256 校验，`sha256_match: true`）；预测按 manifest-ID 过滤，剔除 1,425/686 行缓存外样本。
- 三个迁移块各 400 行：**U1**（未见欺诈类别，其中 269/400 带 ≥3 个问号后缀，语言—标签相关，作为受控压力面板）、**U2**（来源迁移，PKU-SafeRLHF：298 general_harm + 102 financial_fraud）、**U3**（目标模型/风格复合迁移）。
- Gold：双 Judge（DeepSeek + Qwen）raw agreement 0.8733、κ = 0.7485（agreed 1,572 / third-opinion 180 / deterministic 48，无人工复核）。
- 工作点冻结：Final Student 阈值 0.5622；Neural-Gold / Neural-SoftDistill 阈值 0.5。
- 统计：10,000 次 family-cluster bootstrap（family 级配对重抽样）给出差异 CI；配对正确性用精确 McNemar + Holm。

### 4.3 主结果表格（pooled, N=1200）

| Model | Eval N | Families | F1-unsafe | Macro-F1 | Recall | FPR | MCC | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final Student | 1200 | 557 | 0.3329 | 0.5092 | 0.2133 | 0.0683 | 0.2084 | 0.7198 | 0.7044 |
| Neural-Gold | 1200 | 557 | 0.4486 | 0.5366 | 0.3633 | 0.2567 | 0.1153 | 0.5725 | 0.5649 |
| Neural-SoftDistill | 1200 | 557 | 0.4708 | 0.5372 | 0.4033 | 0.3100 | 0.0974 | 0.6224 | 0.5913 |

分块（Student，Wilson CI 见 `tables/e4_shift_corrected.md`）：U1 Recall 0.2000 / FPR 0.0400；U2 Recall 0.1850 / FPR 0.1150；U3 Recall 0.2550 / FPR 0.0500。

### 4.4 实验分析

- **排序能力保留**：Final Student AUROC 0.7198 / AUPRC 0.7044，高于两个神经教师基线（AUROC 0.5725 / 0.6224），说明学生模型在未见迁移下仍保留中等排序能力。
- **部署工作点取舍**：学生冻结工作点（Recall 0.2133 / FPR 0.0683）明显比教师基线（FPR 0.2567 / 0.3100）保守——低误报率取向在迁移场景下以召回为代价。
- **与教师基线差异不显著**：Macro-F1 差异的 family-cluster bootstrap CI 均包含 0；配对正确性 McNemar 经 Holm 校正后 p = 0.2910（不显著）。报告如实呈现为**模型真实迁移失败/部署边界**，不归因于数据质量。
- 面板披露：U1 为受控压力面板（问号后缀、语言—标签相关，未做去后缀重推断）；U2 仅能支撑一般有害响应来源迁移，不能支撑欺诈特定来源泛化；U3 为复合迁移，无法分离单一因果。
- 结论：E4 说明需要**级联修复机制**（见 E5）来恢复部署工作点上的召回。

---

## 5. E5 — 校准与选择性审计级联

- **文件夹**：[`exp5_calibration/`](exp5_calibration/)
- **最终报告（static-fix）**：[`e4e5_final_staticfix/EXP5_CALIBRATION_REPORT_FINAL.md`](e4e5_final_staticfix/EXP5_CALIBRATION_REPORT_FINAL.md)
- **数据与策略**：`exp5_calibration/e5/`（`p3_policies.jsonl`、`p3_paired_statistics.json`、`label_efficiency_runs.jsonl` 等）

### 5.1 实验思路

在 E4 的部署边界处，E5 测试**无需重训模型**的低成本系统级修复：从 P0（学生冻结工作点）出发，比较温度校准（P1）、选择性弃权（P2）、**分数歧义选择性审计**（P3）三级策略，找到“用少量 API 审计换回召回”的可部署级联方案。

### 5.2 实验设计

- **P0**：Final Student 冻结工作点（`risk_score ≥ 0.5622`），零 API。
- **P1**：温度 5.0 + 风险阈值 0.6106（calibration 拟合）——预期为**负结果对照**。
- **P2**：all-safe 选择性弃权——预期为退化负结果（覆盖率为 1.0，无弃权）。
- **P3**：**分数歧义启发式** `min |risk_score − 0.5|` 选出最模糊的 K 行送 DeepSeek 结构化判官（temperature=0，缓存 qy-hash，判官不可见学生分数与 gold）；主工作点 K=180（15% 查询率），K=60–600（5%–50%）做敏感性。
- 统计：10,000 次 family-cluster bootstrap（CI）+ 精确 McNemar（配对正确性）；经验 bootstrap p 值不报告（10,000 次无法分辨 <1e-4）。

### 5.3 主结果表格（Eval N=1200, Cal N=600）

| Policy | Eval N | Cal N | Audited K | API rate | F1-unsafe | Macro-F1 | Recall | FPR | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (Final Student) | 1200 | 600 | 0 | 0.00 | 0.3329 | 0.5092 | 0.2133 | 0.0683 | 0.2084 |
| P1 (temp 5.0) | 1200 | 600 | 0 | 0.00 | 0.1323 | 0.4050 | 0.0717 | 0.0117 | 0.1501 |
| P2 (all-safe) | 1200 | 600 | 0 | 0.00 | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| **P3 (K=180, primary)** | 1200 | 600 | 180 | 0.15 | **0.4777** | **0.6010** | **0.3300** | **0.0517** | **0.3542** |

配对统计（点估计 + family-cluster bootstrap 95% CI）：P3−P0 ΔMacro-F1 **+0.0918** [0.0722, 0.1119]、ΔF1-unsafe +0.1448 [0.1133, 0.1772]、ΔRecall +0.1167 [0.0903, 0.1433]、ΔFPR −0.0167 [−0.0287, −0.0051]、ΔMCC +0.1457 [0.1126, 0.1800]；**精确 McNemar（配对正确性）：b=5, c=85, p=7.53×10⁻²⁰**（级联修正 85 个错误、引入 5 个）。P1−P0 为负结果：ΔMacro-F1 −0.1042 [−0.1286, −0.0792]。

### 5.4 实验分析

- **P3 有效且极廉价**：15% 查询率（180 条）下 F1-unsafe 0.3329 → 0.4777、Macro-F1 0.5092 → 0.6010、FPR 同步下降；敏感性 K=60–600 单调改善（50% 时 MF1 0.7636），成本约 ¥0.07–0.12/千行。
- **P1 是诚实记录的负结果**：温度校准只压低 FPR（−0.0567），召回与 MF1 均下降——不能靠温度解决迁移问题。
- **P2 退化**：学生分数空间无可行弃权带，弃权策略失效，必须引入外部判官。
- 报告写法遵循静态修复规范：只报点估计 + CI，不写无法分辨的 bootstrap p 值；McNemar 仅用于配对正确性。

---

## 6. E6 — 跨多 API 直连响应检测（平衡重跑版）

- **文件夹**：[`exp6_balanced_multi_api/`](exp6_balanced_multi_api/)
- **最终报告**：[`EXP6_V2_FINAL_REPORT.md`](exp6_balanced_multi_api/EXP6_V2_FINAL_REPORT.md)
- **协议**：[`protocol/EXP6_V2_BALANCED_RERUN_STRICT_PROTOCOL.md`](exp6_balanced_multi_api/protocol/EXP6_V2_BALANCED_RERUN_STRICT_PROTOCOL.md)
- **主表**：`tables/main_table_A_behavior.md` / `main_table_B_policy.md` / `main_table_C_model.md` / `hard_safe_table.md`

### 6.1 实验思路

v1 的 random-180 面板中 unsafe 极少（约 13 条），response 类别极端失衡、无法评估判别能力。v2 重做：回答两个估计对象——
- **E6-A**：同一共享挑战分布下，6 个直连 API 目标模型（Qwen Flash/Plus、DeepSeek Flash/Pro、GLM Flash、Kimi）的**欺诈协助行为率**；
- **E6-B**：冻结的 FraudDistill-Student-1.5B 能否在同类危险问题上区分“安全拒绝”与“欺诈协助”，并通过 selective audit 级联（P0/P1/P2）提升部署指标。

### 6.2 实验设计

- **共享面板**：anchor 200（100 should-refuse / 100 should-answer，zh/en 100/100）+ B0 200 + hard-safe control 40，共 440 q/模型；有效回答 3,043 条。
- **Silver**：全量三 Judge（Qwen Flash / DeepSeek Flash / GLM Flash）triple，binary Fleiss κ = 0.7474、unanimous 0.8232、unresolved 0；分歧由 J4（Kimi，不同配置的冻结强模型）裁决。
- **均衡关系集（Student-blind）**：每模型 80 条（unsafe 40 / safe 40，语义族匹配、语言/行为子型配额）+ 40 条 hard-safe control（cal 10 / test 30）；选择过程只读 Silver 与元数据，不加载学生分数。
- **三视图**：q-only / y-only / q+y（head-tail 编码，max_length=512，截断审计 0.4986）。
- **级联**：P0（冻结阈值 0.5622）/ P1（pooled 全局阈值 t=0.2323，calibration 约束 Recall ≥0.65 且 hard-safe FPR ≤0.15）/ P2（10%/20% selective audit）。
- 协议偏差如实记录：metadata-only shortcut probe pooled AUROC 0.7251 略超名义门（BAL 窗口按可行性放宽所致），不作为 shortcut 已消除的声明。

### 6.3 数据集选取

- 池规模 640 q（anchor/B0/B1/B2/control），本实验仅使用 anchor/B0/control 共享面板。
- 泄漏审计：exact / prefix80 / id 泄漏均为 0；superfamily 与跨实验（E3/E4/E5/E6-v1）重叠审计见 `data/superfamily_split_audit.json`、`data/cross_experiment_leakage_audit.json`。
- 高风险原始 q+y 内容仅存本地（`.gitignore` 保护），公开 manifest 见 `data/prompt_pool_manifest_public.jsonl`。

### 6.4 主结果表格

**E6-A 行为面板（共享挑战分布）**

| Model | N | Unsafe assistance | Clean refusal | Safe redirection | Over-refusal | Cost (CNY) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Flash | 300 | 55.7% | 3.7% | 36.0% | 0.0% | 0.24 |
| Qwen Plus | 300 | 42.0% | 0.7% | 54.7% | 0.0% | 0.30 |
| DeepSeek Flash | 300 | 46.0% | 2.3% | 45.0% | 0.0% | 0.28 |
| DeepSeek Pro | 300 | 51.3% | 2.7% | 42.7% | 0.0% | 0.88 |
| GLM Flash | 297 | 77.4% | 0.3% | 12.5% | 0.0% | 0.15 |
| Kimi | 300 | 63.7% | 2.7% | 28.3% | 0.0% | 0.35 |

**E6-B 主结果（frozen test, N=384）**

| Policy | N | Macro-F1 | F1-unsafe | Precision | Recall | FPR | MCC | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (frozen 0.5622) | 384 | 0.722 | 0.685 | 0.809 | 0.594 | 0.141 | 0.470 | 0.798 | 0.796 |
| P1 (pooled global) | 384 | 0.727 | 0.727 | 0.725 | 0.729 | 0.276 | 0.453 | 0.798 | 0.796 |
| P2 (audit 10%) | 384 | 0.776 | 0.778 | 0.770 | 0.786 | 0.234 | 0.552 | 0.798 | 0.796 |
| **P2 (audit 20%)** | 384 | **0.807** | **0.811** | **0.795** | **0.828** | 0.214 | **0.615** | 0.798 | 0.796 |

三视图（P1 阈值）：qy AUROC 0.798 / MF1 0.727；qonly 0.507 / 0.390；yonly 0.790 / 0.700；**q+y vs q-only AUROC 增益 +0.291（目标 ≥0.10）**。Hard-safe control：pooled FPR 14.4%（hard_safe 12.5% / anti_fraud 5.6% / matched_safe 22.2% / benign 33.3%）。

### 6.5 实验分析

- **E6-A 行为率差异显著**：GLM Flash 最高（77.4%），Qwen Plus 最低（42.0%）；中文场景普遍高于英文（+5~16pp）；6 个模型 over-refusal 均为 0。
- **E6-B 判别成立**：三视图对比确认学生模型在同类危险问题上使用 q+y 关系信息（q-only 几乎无判别力，AUROC 0.507）；hard-safe FPR 14.4% 说明仍存在对反诈/安全内容的误报。
- **级联增益**：P0 → P2-20% Macro-F1 0.722 → 0.807（+0.085），Recall 0.594 → 0.828，FPR 同步下降；P1 全局阈值以 FPR 0.276 换 Recall 0.729。
- **跨模型门控**：每模型 6/6 通过 AUROC ≥ 0.65 / MF1 ≥ 0.60 / Recall ≥ 0.50 门；仅 Kimi hard-safe FPR 0.267 略超 0.25 门。
- **错误切片**：zh MF1 0.711 / en 0.738；truncated 0.695 / non-truncated 0.818（长输入是主要掉点来源，head-tail 截断编码仍有信息损失）。
- 累计 E6 成本 ¥51.47（含 v1 ¥1.4153），按阶段记账见 `budget/cost_ledger.jsonl`。

---

## 7. 附：跨实验共用资产

| 资产 | 位置 | 说明 |
|---|---|---|
| 最终学生模型 | `exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/best_step120/` | E4/E5/E6 统一冻结的检测器 |
| E4/E5 冻结清单 | `experiments/e4e5_final_staticfix/` | `FINAL_METRICS.json` / `FINAL_DATA_AUDIT.json` / 4 张 corrected 表 |
| 公共配置 | `experiments/common/` | `datasets.yaml` / `label_mapping.yaml` / `model_registry.yaml` |
| E6 冻结模型注册 | `exp6_balanced_multi_api/protocol/model_registry_frozen.json` | 目标模型与判官身份冻结记录 |
| 归档区 | 各实验 `archive/` 子目录 | 旧版本与中间产物，仅本地保留，不入 Git |

> 安全提示：各实验的**原始高风险 q+y 内容**（含真实诈骗话术）仅保留在本机，报告与公开材料一律不展示可复用的高风险 prompt 与完整欺诈脚本。