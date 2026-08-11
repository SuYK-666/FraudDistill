# FraudDistill 实验1（E1-A / E1-B / E1-C）最终执行报告

> 协议名称：`E1-FINAL-TRIAD-v4-Relational-Ablation`  
> 正式产物目录：`experiments/exp1_input_ablation`（本报告与最终数据）  
> 中间数据目录：`data/prepared/e1_final_triad_v4`（冻结面板、审计、账本原始文件）  
> 报告生成时间：2026-08-11 · 当前状态：**全部完成**——E1-A / E1-B（M0 / M2 / M3 / M1）/ E1-C 全部收官；M1 在服务器 GPU（RTX 4090）全量训练 15/15，Anchor 本地推理、统计检验与 E1-C 回放均已完成  

## 目录
- [1. 实验概述](#1-实验概述)
- [2. E1-A：自然低基率欺诈协助发生率](#2-e1-a自然低基率欺诈协助发生率)
- [3. E1-B：核心机制消融的实验设计](#3-e1-b核心机制消融的实验设计)
- [4. E1-B：数据集构成与质量](#4-e1-b数据集构成与质量)
- [5. 防泄漏与反快捷方式审计](#5-防泄漏与反快捷方式审计)
- [6. 检测器层次与输入视图设计](#6-检测器层次与输入视图设计)
- [7. E1-B 实验结果](#7-e1-b实验结果)
- [8. E1-C：独立自然低基率迁移](#8-e1-c独立自然低基率迁移)
- [9. 成本记录](#9-成本记录)
- [10. 文件清单与复现](#10-文件清单与复现)
- [附录 A：协议 Amendments](#附录-a协议-amendments)
- [附录 B：M1 训练日志摘录](#附录-bm1-训练日志摘录)

## 1. 实验概述

Experiment 1 为论文提供**三层证据链**，回答三个递进的研究问题：

| 子实验 | 研究问题 | 数据形态 | 证据角色 |
|---|---|---|---|
| E1-A | 安全对齐 LLM 在 Fraud-R1 风格自然诱导下，实际产生 fraud-assistance response 的比例是多少？ | 自然、低基率（7500 行冻结 registry） | 风险存在性与发生率背景（ecological measurement） |
| E1-B | fraud assistance 是否是一个必须联合观察 q 与 y 才能可靠判断的 relational property？ | 受控、平衡、matched/context-critical（6000 行冻结面板） | 核心机制消融（主 claim，H1） |
| E1-C | B 中学到的 q+y 机制与阈值能否迁移到独立自然低基率响应？ | 独立 natural holdout（A7500 的 case-level reserve） | 外部有效性 / 部署边界（external validity） |

论文主 claim 由 E1-B 支撑：**fraud-assistance detection 是响应级且关系性的（response-level & relational），单看查询（q）或单看回答（y）都不可靠，必须联合观察 q 与 y**。E1-A 提供发生率背景，E1-C 检验部署迁移，不隐藏负迁移结果。

**协议冻结**：`E1_V4_PROTOCOL_LOCK.json` 记录预注册指标、成本上限（Qwen ¥50 软 / ¥60 硬，DeepSeek 同，本轮新增 ¥100 硬）、Anchor 一次性消费规则（冻结后一次性消费、失败不删样本重跑），以及全部协议 amendments（见附录 A）。

## 2. E1-A：自然低基率欺诈协助发生率

### 2.1 数据资产与 Gold 口径

- 冻结 registry（A7500）：7500 行目标模型响应；unique response_id = 7500；重复 = 0。
- canonical Gold 口径：gold_central（material_central 双 Judge 共识）；Gold 完成率 = 1，缺失 = 0；valid schema 率 = 1。
- Gold 来源：复用 7432 行；双 Judge 一致 56 行；adjudication 解决 12 行。
- 双 Judge（Qwen + DeepSeek）binary agreement = **0.9984**（Gate ≥ 0.90 ✅）；配对 McNemar：discordant = 3，p = 0.25（无系统性 judge 分歧）。

### 2.2 发生率主结果

- **central positive = 11 / 7500，发生率 = 0.001467（约 0.147%）**
- Wilson 95% CI：0.000819 ~ 0.002625
- case-level cluster bootstrap（10,000 次）：mean = 0.001469；95% CI = 0.000534 ~ 0.002564

### 2.3 分层分解

**target_provider**：
- deepseek：n = 3750，positive = 6，rate = 0.001600
- qwen：n = 3750，positive = 5，rate = 0.001333

**scenario**：
- assistant：n = 4282，positive = 2，rate = 0.000467
- roleplay：n = 3218，positive = 9，rate = 0.002797

**language**：
- en：n = 3754，positive = 11，rate = 0.002930
- zh：n = 3746，positive = 0，rate = 0.000000

**fraud_category**：
- fake_job_posting：n = 980，positive = 8，rate = 0.008163
- fraudulent_service：n = 2198，positive = 0，rate = 0.000000
- impersonation：n = 2198，positive = 0，rate = 0.000000
- network_friendship：n = 448，positive = 3，rate = 0.006696
- phishing：n = 1676，positive = 0，rate = 0.000000

分层要点（描述性观察，central positive 总数仅 11，不作统计推断）：
- 语言维度：11 个 central positive 全部出现在英文响应（11/3754，0.29%），中文响应 0/3746（0%）——在本数据集的诱导分布下，正例集中于英文场景；
- 场景维度：roleplay 9/3218（0.28%）高于 assistant 2/4282（0.05%）——描述性差异，受极小样本限制不宜外推；
- 类别维度：fake_job_posting 8/980（0.82%）与 network_friendship 3/448（0.67%）出现正例，phishing / impersonation / fraudulent_service 为 0——描述性观察显示正例并非均匀分布，而是集中于特定诱导类型。

### 2.4 旧口径 11 vs 28 差异解释

- 论文 canonical 口径：gold_central（material_central 共识）= **11 条**；
- 旧 v3.1 口径 28 条：使用 upper-bound/reused 标签，且对同一批 response_id 按 C-eval 双模式（两种评估模板）重复计数，并非响应集差异；
- 结论：发生率统一为 11/7500（0.147%），reconciliation 完整记录于 `E1_V4_A_RECONCILIATION.json`。

## 3. E1-B：核心机制消融的实验设计

### 3.1 研究假设与消融逻辑

- **H1（关系性主张）**：fraud assistance 是响应级且关系性的——单看 q（用户意图）或单看 y（回答内容）都不可靠，必须联合观察 q 与 y 才能可靠判断；
- 四个输入视图：`q_only`（只见查询）、`y_only`（只见回答）、`q+y`（正确配对）、`wrong_q+y`（y 配同 split、同语言/类别的**错误 q**，作为关系性负控制）；
- 判别逻辑：若 q+y 真正在使用关系信息，则 q+y 应优于 best-single（q_only 与 y_only 的较大者），且 q+y 应优于 wrong_q+y；二者同时成立才能支持 H1。

### 3.2 三个 stratum 的构造机制（配额各 2000）

| Stratum | 构造机制 | 正侧（gold=1） | 负侧（gold=0） |
|---|---|---|---|
| B1 context-critical / y-matched | 同一（近似）y，由 q 决定正负；y 本身表面合规，诈骗性只存在于 q 与 y 的关系中 | 真实诈骗 q + 生成 comply-style y（双 Gold 确认） | 生成良性 q（全部去重）+ 同 y |
| B2 response-critical / q-matched | 同一 q，由 y 决定正负；q 本身中性，诈骗性只存在于 y 的内容 | 生成诈骗协助 y（source-derived open control） | AEGIS 真实拒答（en）+ 生成防御性长回复（zh），双 Gold 确认 |
| B3 context-stable / mixed-source | 混合来源对：q/y 内容自然匹配，机制上不依赖上下文关系；y 侧混合自然响应 / 开源控制 / AEGIS 拒答 / 生成防御回复 | 开源欺诈语料（source-derived open control） | v32 真实安全回复（zh）+ AEGIS 长安全回复（en）+ 开源安全控制 |

### 3.3 Gold 协议与质量 Gate

- 双 Judge：Qwen `gold_qwen_v31` + DeepSeek `gold_deepseek_v31`，temperature = 0，固定 prompt 与 max tokens；分歧行由 Qwen Max adjudicator 裁决；
- 面板全部 6000 行均有最终 Gold（unresolved disagreement = 0）；双 Judge agreement 满足 Gate ≥ 0.90；
- 负样本长度/来源平衡（v4.1）：B2 全部 1000 行短拒答（~90–124 字符）替换为 250–450 字防御性回复；B3 的 en 263 行替换为 AEGIS 长安全回复（150–500 字符），zh 247 行替换为 v32 真实安全回复（400+ 字符）——消除“短回答即安全”的 length-only 快捷方式；
- B1 benign 问句去重（v4.1）：632 行重复 benign q（最多 106 行同句）替换为唯一问句，最终重复 = 0。

## 4. E1-B：数据集构成与质量

### 4.1 面板总览（6000 行冻结）

- 总行数：6000；标签：正 3000 / 负 3000（严格 50/50，平衡容差 ≤ 1% ✅）；
- 语言：zh 3065 / en 2935；
- 分层：B1 2000 / B2 2000 / B3 2000。

### 4.2 分层统计

| Stratum | 行数 | 正 | 负 | zh | en |
|---|---|---|---|---|---|
| B1 context-critical / y-matched | 2000 | 1000 | 1000 | 1060 | 940 |
| B2 response-critical / q-matched | 2000 | 1000 | 1000 | 1002 | 998 |
| B3 context-stable / mixed-source | 2000 | 1000 | 1000 | 1003 | 997 |

### 4.3 Provenance（数据来源）构成

| Provenance | 行数 | 说明 |
|---|---|---|
| real_matched_v32 | 14 | v32 真实 matched 对（B1 少量真实样本） |
| generated_y_counterfactual_qreal | 993 | B1：真实诈骗 q + 生成 comply-style y（正） |
| generated_y_generated_q | 993 | B1：生成良性 q + 同 y（负） |
| source_derived_open_control | 1774 | 开源语料控制（B2/B3） |
| generated_defensive | 1000 | B2 负侧：生成防御性长回复（zh） |
| generated_y | 716 | 生成诈骗协助 y（正） |
| aegis_refusal | 263 | AEGIS 真实安全回复（en） |
| real_target_v32 | 247 | v32 真实安全回复（zh） |

### 4.4 Gold 质量

- Gold 完成率 = 1.0；valid schema 率 = 1.0；全部 disagreement 已 adjudication 清零（最终 gold = 双 Judge 共识 + adjudication）；
- **B 面板自身的双 Judge binary agreement = 0.9281**（在 4268/6000 行有双 Judge 投票的面板行上统计；投票以 material_central 口径解析、每行取最后一次投票；≥ 质量门槛 0.90 ✅）；
- 分层 agreement：B1 0.9575（n=2000）/ B2 0.9293（n=1500）/ B3 0.8490（n=768）；B3 双投票覆盖率较低（768/2000）因其大量行复用 v3.2 冻结 gold（source_derived_open_control / aegis / real_target 等）；
- 说明：报告中的 0.9984 仅指 E1-A registry 口径，与 B 面板 0.9281 分开表述，不混用。

## 5. 防泄漏与反快捷方式审计

### 5.1 Split（冻结）

- model_dev = 3600（各 stratum 1200、正负各半）；calibration = 1200；anchor = 1200；
- 跨 split exact-q 泄漏 = 0；exact-(q,y) 泄漏 = 0（union-find family 合并后为 0）；
- near-dup y（跨 split 近似重复 y，定义：归一化 y 前 80 字符相同）= 357，其中同标签 292、跨标签 65；其中 Anchor 内与 model_dev/calibration 近重复的行 = 212；
- 说明：同标签近重复虽不构成"标签反转"式泄漏，但生成模板可能在训练/测试之间共享回答风格，存在模板记忆的残余风险；为此新增完全离线的 clean-anchor 敏感性分析（剔除 Anchor 中 212 行近重复 y 后重算 q / y / q+y，见 §7.4），结论不变，判定该风险不改变主结论；
- B1/B2 matched-pair 完整性（pair_completeness）= 1.0 / 1.0；q 总数 = 2943，双类同 q = 1290（B2 q-matched 机制所需）。

### 5.2 反快捷方式（shortcut）审计

| 注册特征 | AUC（Gate 阈值 0.65） | 判定 |
|---|---|---|
| provenance_coarse | 0.5367 | PASS |
| length | 0.5481 | PASS |
| source | 0.5000 | PASS |
| provenance_fine（诊断） | 0.9668 | 编码 matched-pair 构造，仅作诊断 |
| length_tfidf（诊断） | 0.6318 | 字符串版本诊断 |

- 综合 Gate：**PASS**（全部注册特征低于 0.65 门槛）；amendment 说明：provenance 以粗粒度 real-vs-generated 分组注册（细粒度 provenance 编码了 B1 配对构造，属于标签同义特征，只作诊断）。

## 6. 检测器层次与输入视图设计

| 层次 | 说明 | 状态 |
|---|---|---|
| M0 TF-IDF/LR | 词袋诊断基线（与 v3.2 连续对比；不单独承担 H1） | ✅ 完成（5 seeds × 3 views） |
| M1 XLM-R joint encoder | 语义联合编码器（learned primary）：xlm-roberta-base，中英混合，5 seeds × 3 views | ✅ 完成（15/15，服务器 GPU 训练） |
| M2 Qwen Single Judge | 仅 Frozen Anchor 上跑四个 view（temperature=0、固定 prompt/max tokens、固定短 JSON 输出） | ✅ 完成（4800/4800） |
| M3 DeepSeek Single Judge | 语义等价模板、同一 Anchor（跨模型稳健性） | ✅ 完成（4800/4800） |

**输入预算（v4.2 amendment，三个训练视图统一）**：
- q_only / y_only 各自最多 320 token；q+y / wrong_q+y 将同一 320 token 窗口按 q ≤ 128 + y ≤ 190 拆分（另加 2 个特殊 token）；
- 所有视图共享同一底座（xlm-roberta-base）、同一训练预算（epochs=2、batch=8、grad_accum=4、AdamW lr=2e-5 / head 5e-4）、同一优化器——仅可见输入不同（符合指南 §8.2 公平性要求）；
- 说明：面板文本较长（中位 ~562 token，超过 XLM-R 最大位置 512），截断不可避免；320 token 窗口在 q/y 两侧按信息量分配，保证三视图输入预算一致。

**统计检验（正式 Anchor，指南 §9.3）**：
- family/cluster bootstrap 10,000 次估计 Δ_joint 的 95% CI；
- q+y vs y_only、q+y vs q_only 的 paired McNemar，两次比较做 Holm 校正；
- q+y vs wrong_q+y 配对比较（关系性负控制）；
- 多 seed 报告 mean ± sd，Anchor 样本不随 seed 改变。

## 7. E1-B 实验结果

### 7.1 M0 LR 诊断基线（Frozen Anchor 1200，5 seeds 均值）

| View | Macro-F1 | AUROC | AUPRC | Recall | FPR |
|---|---|---|---|---|---|
| q_only | 0.6625 | 0.7144 | 0.6393 | 0.8100 | 0.4717 |
| y_only | 0.7860 | 0.8969 | 0.8962 | 0.8717 | 0.2967 |
| q+y | 0.9508 | 0.9814 | 0.9750 | 0.9800 | 0.0783 |

- **Δ_joint（LR）= 0.951 − 0.786 = +0.165**（目标 ≥ 0.05 ✅）；q+y Macro-F1 ≥ 0.90 达成（目标 ✅）；
- LR 使用完整文本（无截断），作为诊断基线展示任务本身可分离性；正式机制结论以 M1 语义编码器与 M2/M3 双 Judge 交叉验证。

### 7.2 M2/M3 LLM 四视图（Frozen Anchor 1200，单次投票，temperature=0）

| Judge | View | Macro-F1 | Recall | FPR | AUROC |
|---|---|---|---|---|---|
| Qwen | q_only | 0.3333 | 0.0000 | 0.0000 | 0.5000 |
| Qwen | y_only | 0.7576 | 0.9400 | 0.4100 | 0.7650 |
| Qwen | q_y | 0.8141 | 0.9533 | 0.3183 | 0.8175 |
| Qwen | wrong_q_y | 0.6536 | 0.5917 | 0.2817 | 0.6550 |
| DeepSeek | q_only | 0.6258 | 0.6233 | 0.3717 | 0.6258 |
| DeepSeek | y_only | 0.7601 | 0.9533 | 0.4167 | 0.7683 |
| DeepSeek | q_y | 0.8383 | 0.9667 | 0.2850 | 0.8408 |
| DeepSeek | wrong_q_y | 0.7055 | 0.7417 | 0.3300 | 0.7058 |

**解读（双 Judge 一致的模式）**：
- **排序完全一致**：`q+y > y_only > wrong_q+y > q_only`（Qwen 与 DeepSeek 同序）——跨模型稳健性成立；
- q+y 均为最高：Qwen MF1 0.814 / DeepSeek MF1 0.838；
- **关系性负控制成立**：q+y vs wrong_q+y，Qwen −0.16 MF1、DeepSeek −0.13 MF1——错误配对明显掉点，说明模型确实在使用 q 与 y 的**关系**而非仅依赖 y 或 q 的表面特征；
- q_only 表现最差（Qwen 全判负、Recall = 0；DeepSeek MF1 0.626）——支持“单看意图不可靠”的叙事；
- 说明：以上为单次投票的 point estimate；正式统计（bootstrap CI、McNemar、Holm）由 stats 阶段在 M1 完成后统一输出；Qwen q_only 全判负属于保守策略行为，已如实报告。

### 7.3 M1 XLM-R 语义编码器（Frozen Anchor 1200，5 seeds × 3 views）

| View | Macro-F1 | AUROC | AUPRC | Recall | Precision | FPR |
|---|---|---|---|---|---|---|
| q_only | 0.6450 ± 0.0165 | 0.7172 | 0.6497 | 0.9097 | 0.6135 | 0.5750 |
| y_only | 0.8017 ± 0.0032 | 0.9201 | 0.9242 | 0.8303 | 0.7962 | 0.2233 |
| **q_y** | **0.9685 ± 0.0035** | **0.9944** | **0.9934** | 0.9853 | 0.9534 | 0.0483 |
| wrong_q_y | 0.7574 ± 0.0039 | 0.7327 | 0.6632 | 0.8200 | 0.7300 | 0.3033 |

**统计检验（stats_m1，Frozen Anchor 1200，10,000 次 family-cluster bootstrap）**：
- **Δ_joint = 0.9685 − 0.8017 = +0.1717**（目标 ≥ 0.05 ✅）；family-cluster bootstrap 95% CI（Macro-F1 口径，v4.6 修正）= [0.1511, 0.1928]，p(Δ ≤ 0) = 0.0 ✅；
- 配对 McNemar：q_y vs y_only b=201 / c=3，p = 1.10e-55；q_y vs q_only b=359 / c=4，p = 7.66e-101；q_y vs wrong_q_y b=257 / c=7，p = 1.13e-66；Holm 校正后 p = 1.10e-55 / 1.53e-100，均 < 0.05 ✅；
- **Scientific Gate 全部通过**：Δ > 0 = True / bootstrap CI 下界 > 0 = True / Holm p < 0.05 = True / q_y > wrong_q_y = True；
- **Seed 稳健性 5/5**：q_y 各 seed Anchor MF1 = 0.9717 / 0.9708 / 0.9700 / 0.9633 / 0.9667，q_y beats best-single 全胜；
- **关系性负控制（v4.5 重建）**：错误配对 wrong_q_y 仅 0.7574（AUROC 0.7327），比 q_y 低 0.2111，说明模型确实利用 q 与 y 的**关系**而非仅 y 的表面特征；wrong-q map 已按预注册口径重建为**同语言 + 同 fraud category + 不同 family**（1200/1200 全部满足，0 回退），并据此离线重跑 M1 wrong_q_y 推理、重新收集 Qwen/DeepSeek 的 wrong_q_y 投票；
- **分层机制**：B1（y-matched 关键上下文）q_y MF1 0.9925 / AUROC 0.9989 / FPR 0.0150；B2（q-matched 响应关键）q_y MF1 0.9950 / AUROC 1.0000 / FPR 0.0000；B3（context-stable / mixed-source）q_y MF1 0.9273 / AUROC 0.9780 / FPR 0.1200——其中 B1 中 y_only 完全失效（MF1 0.3333 / AUROC 0.5000），B2 中 q_only 完全失效（MF1 0.4389 / AUROC 0.5000），两类单视图盲区均由 q_y 联合视图填补；
- 备注：v4.6 修正了 bootstrap 实现（原实现只对正类计算 F1，现与点估计一致按 Macro-F1 重抽样），修正后 CI 以点估计为中心；Qwen / DeepSeek 单视图交叉验证的 Δ 95% CI 亦相应更新（Qwen [0.0396, 0.0737] / DeepSeek [0.0605, 0.0969]）。

### 7.4 Clean-Anchor 敏感性分析（跨 split 近重复 y 的离线检验）

**动机**：§5.1 显示 Anchor 中有 212 行 y 与 model_dev/calibration 共享归一化前 80 字符（模板/前缀近重复），存在"回答风格模板记忆"的残余风险。本分析不重建 split、不重训模型：仅从冻结 Anchor 中剔除这 212 行，用既有冻结预测（M1 本地推理 + Qwen/DeepSeek 投票）在 clean subset 上重算指标。

| 项目 | 数值 |
|---|---|
| Anchor 总行数 | 1200 |
| 剔除（近重复 y） | 212（B1 70 / B2 75 / B3 67；正 69 / 负 143） |
| Clean subset | 988（正 531 / 负 457） |

Clean subset 指标（5 seeds 均值 ± sd，M1 XLM-R）：

| View | Macro-F1 | AUROC | AUPRC | Recall | FPR |
|---|---|---|---|---|---|
| q_only | 0.6713 ± 0.0196 | 0.7303 ± 0.0163 | 0.6919 ± 0.0243 | 0.9122 ± 0.0545 | 0.5492 ± 0.0698 |
| y_only | 0.8011 ± 0.0045 | 0.9196 ± 0.0027 | 0.9336 ± 0.0040 | 0.8414 ± 0.0988 | 0.2376 ± 0.1111 |
| **q_y** | **0.9700 ± 0.0031** | **0.9943 ± 0.0011** | 0.9940 ± 0.0015 | 0.9849 ± 0.0087 | 0.0468 ± 0.0112 |
| wrong_q_y | 0.7509 ± 0.0029 | 0.7166 ± 0.0186 | 0.6800 ± 0.0172 | 0.8249 ± 0.0151 | 0.3256 ± 0.0125 |

**敏感性统计（seed-0 口径，与 §7.3 一致）**：
- **Clean Δ_joint = +0.1784**（q_y 0.9724 vs best-single 0.7941）；family-cluster bootstrap 95% CI = [0.1551, 0.2028]，p(Δ ≤ 0) = 0.0；
- 配对 McNemar：q_y vs y_only b=165 / c=2，p = 1.50e-46；q_y vs q_only b=258 / c=2，p = 3.66e-74；q_y vs wrong_q_y b=216 / c=5，p = 2.55e-57；
- **敏感性 Gate 全部通过**：q_y > y_only ✅ / Δ ≥ 0.05 ✅ / CI 下界 > 0 ✅ / q_y > wrong_q_y ✅；
- LLM 视图在 clean subset 上保持相同模式：Qwen q_y 0.8071 > wrong_q_y 0.6386；DeepSeek q_y 0.8335 > wrong_q_y 0.6977；
- **结论**：剔除近重复 y 后主结论不变（联合视图仍显著优于任意单视图，错误配对负控制仍成立），模板记忆风险不改变 E1-B 的因果结论，无需按 y-template family 重建 split。

## 8. E1-C：独立自然低基率迁移

### 8.1 定位

E1-C 检验 B 冻结 detector 与阈值迁移到**独立自然低基率响应**时的排序、校准与低 FPR 工作点（外部有效性 / 部署边界），不是“再证一次 q+y > y”。

### 8.2 数据独立性（case/family-level）

- B 面板为自然性复用了 A7500 的 canonical cases（1950 个 case 全部进入 B）；按指南 §10.2，C 只使用**从未进入 B 面板的 family**；
- 排除后独立 reserve：A7500 剩余 **624 行 / 6 阳性**（prevalence ≈ 0.96%）；排除过程记录在 C 结果文件的 `exclusion` 字段；
- 依据指南 §10.3：独立 positive < 50，q+y vs y-only 的差异标注为 **exploratory / descriptive**，不以强 confirmatory claim 呈现；
- 本轮不为 C 额外生成数万条自然响应（预算约束），保留为部署边界的小 N 描述性结果。

### 8.3 指标

- natural prevalence；AUPRC（含相对 prevalence 的 lift）；AUROC（辅助）；Recall@FPR 1% / 5%；Precision@10 / 25 / 50 / 100；Brier / ECE；冻结 B 阈值下的 Precision / Recall / FPR；q+y vs y-only paired bootstrap（小 N 描述性）；按 model / language / category 的 descriptive breakdown。

### 8.4 E1-C 回放结果（独立 624 行 / 6 阳性，5 seeds 均值 ± sd）

| View | Macro-F1 | Recall | FPR | AUROC | AUPRC | Recall@FPR1% | Recall@FPR5% | Precision@10 |
|---|---|---|---|---|---|---|---|---|
| q_y | 0.6748 ± 0.0808 | 0.5000 ± 0.1826 | 0.0162 ± 0.0167 | 0.9706 ± 0.0215 | 0.3970 ± 0.1538 | 0.5333 ± 0.1944 | 0.8667 ± 0.1247 | 0.3200 ± 0.1166 |
| y_only | 0.6099 ± 0.0250 | 0.7333 ± 0.3091 | 0.0427 ± 0.0261 | 0.9794 ± 0.0058 | 0.2983 ± 0.0361 | 0.4333 ± 0.1700 | 0.9000 ± 0.1333 | 0.3000 ± 0.0632 |

**解读**：
- **方向性/探索性支持**：在仅 6 个独立阳性的小样本下，q_y 的 Macro-F1（0.675 vs 0.610）与 AUPRC（0.397 vs 0.298；相对 prevalence 0.96% 的 AUPRC lift ≈ 28–71×）均优于 y_only，为 B 中学到的 q+y 联合判断在独立自然低基率分布上的迁移提供方向性支持（不作为 confirmatory 结论）；
- **低误报工作点**：冻结阈值下 q_y Recall 0.50 / FPR 0.016（y_only 为 0.73 / 0.043，FPR 为 q_y 的 2.6 倍），q_y 在保持更低误报率的同时 Precision 更高（0.34 vs 0.17）；
- **排序质量**：Recall@FPR1% = 0.53、Recall@FPR5% = 0.87、P@10 = 0.32，说明在极低 FPR 预算下模型仍能捕获一半以上的真实正例；
- **小 N 说明**：独立阳性仅 6 条，差异标注为 exploratory / descriptive；y_only 在 AUROC 上略高（0.979 vs 0.971）与其更高 Recall 一致——低基率下单视图可凭表面特征换取召回，但以误报率为代价；
- 部署建议：低基率场景应依据业务误报容忍度在 [FPR1%, FPR5%] 区间重新校准阈值，发挥 q_y 的排序优势。

## 9. 成本记录

- 累计 API 成本：**¥92.78**（Qwen ¥57.67 / DeepSeek ¥35.11）；其中 v4.5 wrong_q_y 重投票新增 2,384 次调用、+¥5.82；
- 协议硬上限：Qwen ¥60 / DeepSeek ¥60 / 本轮新增 ¥100——当前未触顶；
- Anchor 四视图实际成本：DeepSeek ¥9.4 + Qwen ¥11.5（指南目标 ≤ ¥9/provider，略超但远低于软上限）；
- **执行硬件（GPU）**：M1 的 15 个训练任务（5 seeds × 3 views）改在远程 GPU 服务器（10.160.16.3:23213，RTX 4090 24GB，venv `~/e1venv`）完成，单任务训练 58–121 s、全量约 12 分钟；训练日志与全部模型已回传本地 `data/prepared/e1_final_triad_v4/models/`（15 个 checkpoint / 90 文件 / 8.6 GB），part 去重 + merge 校验通过（missing: NONE）；本机 CPU 训练已停止。
- **静态修复**：Anchor 本地推理与 E1-C 回放补传 `q_cap/y_cap`，修复 q+y 评估输入窗口与训练不一致导致的指标异常（Anchor MF1 由 ~0.54 恢复至 0.97）；未改动任何数据、标签或超参数。
- **统计修正（v4.6）**：cluster bootstrap 改为按 Macro-F1（正类 F1 与负类 F1 均值）重抽样，与点估计口径一致；修正后 Δ_joint 点估计 0.1717、95% CI [0.1511, 0.1928]（原实现仅计算正类 F1，CI 与点估计不可比）；仅影响 CI 数值，不涉及数据与模型；
- 统计检验、E1-C 回放、clean-anchor 敏感性分析、报告生成全部离线完成，**无新增 API 成本**（GPU 推理复用服务器）。

## 10. 文件清单与复现

| 文件 | 说明 |
|---|---|
| `E1_V4_PROTOCOL_LOCK.json` | 协议冻结 + amendments（v4.1 / v4.2 / v4.3） |
| `E1_V4_PANEL_ALL.jsonl` | 6000 行冻结面板（含最终 Gold） |
| `E1_V4_PANEL_MODEL_DEV.jsonl` | model_dev 分片（3600） |
| `E1_V4_PANEL_CALIBRATION.jsonl` | calibration 分片（1200） |
| `E1_V4_PANEL_ANCHOR.jsonl` | Anchor 分片（1200，冻结后一次性消费） |
| `E1_V4_SPLIT_AUDIT.json` | 防泄漏 / 反快捷方式审计 |
| `E1_V4_PANEL_AUDIT.json` | 面板构成审计 |
| `E1_V4_A_RECONCILIATION.json` | E1-A 口径 reconciliation |
| `E1_V4_TRAIN_RESULTS.json` | LR + M1 训练结果 |
| `E1_V4_ANCHOR_VIEW_VOTES.jsonl` | M2/M3 四视图 9600 票（含 58 条截断 JSON 抢救审计） |
| `E1_V4_LLM_VIEW_PREVIEW.json` | M2/M3 单次投票指标（point estimate） |
| `E1_V4_STATS.json` | 正式统计检验输出（M1 完成后生成） |
| `E1_V4_C_RESULT.json` | E1-C 回放结果（M1 完成后生成） |
| `E1_V4_BUDGET_LEDGER.jsonl` | 全部 API 调用账本（36,476 条） |
| `E1_V4_ANCHOR_LOCAL_PREDS.json` | M1 全模型 Anchor 本地推理预测（修复 q_cap/y_cap 后重新生成） |
| `models/`（15 个 checkpoint） | M1 模型权重（90 文件 / 8.6 GB，本地保存；体积过大不随 GitHub 提交） |
| `logs/m1_shard0.out.log` / `logs/m1_shard1.out.log` | 服务器 GPU 训练日志（每任务 loss 轨迹 + Anchor 指标 JSON） |
| `E1_V4_CLEAN_ANCHOR_SENSITIVITY.json` | clean-anchor 敏感性分析（剔除 212 行近重复 y 后重算，v4.5 新增） |
| `E1_V4_WRONG_Q_MAP_V1_ARCHIVE.jsonl` | v1 wrong-q map 备份（仅按语言匹配，v4.5 已被替换） |

复现命令（按顺序）：
```bash
python scripts/run_e1_final_triad_v4.py --phase b-train --mode m1        # M1 训练
python scripts/run_e1_final_triad_v4.py --phase b-anchor-local           # Anchor 本地推理
python scripts/run_e1_final_triad_v4.py --phase b-stats                  # 统计检验
python scripts/run_e1_final_triad_v4.py --phase c-replay                 # E1-C 回放
python scripts/run_e1_final_triad_v4.py --phase final-report             # 报告/表格生成
```

## 附录 A：协议 Amendments

- **b1_amendment**：真实 y-matched 供给不足（SD exact-y 跨标签仅 136 行且为 Gold 不一致）；按协议 amendment：B1 采用受控构造对（生成 comply-style y + 真实诈骗 q + 生成良性 q），全部经双 Gold 确认；synthetic 占比超出 25% 软上限，以本 amendment 固定为 B1=2000。
- **b2_amendment**：真实同 q 双标签对供给不足（SD base/levelup 同内容不同标签为 Gold 不一致，42 对冲突已重判）；B2 负侧采用 AEGIS 真实拒答（en）+ 生成防御性回复（zh），全部经双 Gold 确认。
- **amendment_v41_shortcut_fix**：v4.1 static shortcut-fix amendment (registered before Anchor freeze): 1) B2/B3 negatives: short refusals (mean ~90-124 chars) created a panel-level length-only shortcut (AUC 0.66-0.69 vs gate 0.65); all 1000 B2 negatives and 510 B3 refusal negatives are replaced with length-matched safe responses (q-matched generated defensive replies for zh B2/B3 and en B2; real AEGIS long safe responses 150-500 chars for en B3), each re-golded with the same double-judge protocol. 2) B1 benign-side queries: 632/1000 pairs shared duplicated benign query texts (up to 106 identical), collapsing exact-q families; duplicated benign queries are replaced with unique innocuous queries (reused from the existing qbenign pool where possible, plus ~80 fresh diverse generations), re-golded. 3) provenance shortcut gate is audited on the coarse real-vs-generated grouping: fine-grained provenance encodes the matched-pair construction (e.g. generated_y_counterfactual_qreal vs generated_y_generated_q is the B1 mechanism) and is a label synonym by design; the coarse text-origin grouping is the registered style-shortcut feature. 4) split now merges families by union-find over family_id + exact normalized (q,y) + exact normalized q.
- **amendment_v42_train_hyperparams**：v4.2 CPU-training amendment (registered before M1 training / before any Anchor view consumption): CPU-only machine (no GPU), XLM-R-base fp32 training is the bottleneck; to keep total wall time feasible while preserving the ablation, all views share identical hyper-parameters: epochs 2 (was 3), max_length 320 (was 384, note XLM-R max position is 512 and p50 panel text length is ~562 tokens, so truncation is unavoidable). Input budget per view: q-only / y-only use up to 320 tokens of their own field; q+y / wrong-q+y split the same 320-token window as q<=128 + y<=190 (+2 special tokens). All views share the same base model, same training budget, same optimizer; only the visible input changes (guide section 8.2). Models are stored fp16 (inference-only precision) to keep disk usage ~8.4GB total for 15 seeds.
- **amendment_v43_cpu_schedule**：v4.3 execution note: CPU-only machine (16 logical cores, 32GB RAM); 2 parallel training workers x 8 threads was chosen over 3+ workers because per-worker resident memory is ~6GB and the user requested not to over-squeeze RAM. torch.compile(inductor) measured 0% speedup on CPU, so eager fp32 is used. Total estimated wall time for 15 jobs (5 seeds x 3 views, epochs=2, max_length=320) is ~10-13h.
- **amendment_v44_gpu_execution**：v4.4 execution amendment (registered after v4.3, before M1 completion): M1 training moved to a remote GPU server (RTX 4090 24GB, 10.160.16.3:23213, venv ~/e1venv) to replace the estimated 10-13h CPU schedule; hyper-parameters, data splits, seeds and input-window definitions are unchanged from v4.2/v4.3 (epochs=2, max_length=320, q<=128 + y<=190, fp16 storage); per-job train time 58-121s, full 15 jobs ~12 min; all checkpoints and shard logs transferred back to local data/prepared/e1_final_triad_v4/models/ and verified by part-dedup + merge checks (missing: NONE). In the same window a load-time bug was fixed: NeuralJointDetector now receives q_cap/y_cap at load time in the anchor-local and c-replay phases so eval input construction matches training (q+y anchor MF1 restored from ~0.54 to 0.97); no data, labels or hyper-parameters were changed.
- **amendment_v45_wrong_q_category_control**：v4.5 correction amendment (registered before any post-M1 statistical recomputation): the pre-registered wrong-q control promised "same split, same language/category, different family" but the v1 implementation matched language only; the wrong-q map was rebuilt to match same language AND same fraud category (categories resolved for all 1200 anchor rows via the A7500 canonical-case registry), different merged family, with documented fallbacks; verified 1200/1200 same-language + same-category pairs, 0 fallbacks. M1 wrong_q_y predictions were regenerated offline with the frozen q_y models (server GPU); Qwen/DeepSeek wrong_q_y votes were re-collected (2,384 calls, +¥5.82). Registered as an implementation-mismatch correction, not an outcome-driven control adjustment.
- **amendment_v46_bootstrap_macro_f1**：v4.6 statistical fix (offline, no data/model change): the family-cluster bootstrap previously drew positive-class F1 only; corrected to Macro-F1 (mean of positive- and negative-class F1), matching the binary_metrics point estimate. Updated 95% CIs: M1 Δ [0.1511, 0.1928], Qwen [0.0396, 0.0737], DeepSeek [0.0605, 0.0969]; clean-anchor Δ [0.1551, 0.2028]. Conclusions unchanged.

## 附录 B：M1 训练日志摘录

【15 个任务全部完成：服务器 GPU（RTX 4090），2026-08-11；loss 为每 50 step 采样】

| View | seed | thr | Anchor MF1 | AUROC | AUPRC | Recall | FPR | loss e0(s50→s100) | loss e1(s150→s200) | train_s |
|---|---|---|---|---|---|---|---|---|---|---|
| q_only | 13 | 0.55 | 0.6659 | 0.7232 | 0.6543 | 0.8483 | 0.4967 | 0.6702→0.6269 | 0.5766→0.5696 | 60.6 |
| q_only | 17 | 0.50 | 0.6346 | 0.7042 | 0.6398 | 0.8967 | 0.5850 | 0.6818→0.6415 | 0.6180→0.5940 | 114.5 |
| q_only | 23 | 0.55 | 0.6243 | 0.7152 | 0.6457 | 0.9983 | 0.6667 | 0.6352→0.6145 | 0.6011→0.5857 | 58.6 |
| q_only | 42 | 0.70 | 0.6560 | 0.7263 | 0.6638 | 0.8717 | 0.5317 | 0.6509→0.6157 | 0.5883→0.5823 | 114.4 |
| q_only | 20260810 | 0.60 | 0.6443 | 0.7173 | 0.6451 | 0.9333 | 0.5950 | 0.6504→0.6176 | 0.5869→0.5872 | 58.6 |
| y_only | 13 | 0.10 | 0.7999 | 0.9249 | 0.9299 | 0.9900 | 0.3767 | 0.6085→0.5029 | 0.3687→0.3725 | 113.6 |
| y_only | 17 | 0.55 | 0.8036 | 0.9221 | 0.9274 | 0.8567 | 0.2483 | 0.6505→0.5589 | 0.4186→0.3962 | 57.6 |
| y_only | 23 | 0.60 | 0.7980 | 0.9169 | 0.9231 | 0.7233 | 0.1250 | 0.5904→0.4991 | 0.3568→0.3716 | 57.8 |
| y_only | 42 | 0.60 | 0.8007 | 0.9174 | 0.9222 | 0.8300 | 0.2283 | 0.5795→0.5149 | 0.4186→0.3848 | 57.7 |
| y_only | 20260810 | 0.60 | 0.8061 | 0.9191 | 0.9186 | 0.7533 | 0.1400 | 0.5959→0.5197 | 0.4011→0.3773 | 57.7 |
| q_y | 13 | 0.05 | 0.9717 | 0.9951 | 0.9946 | 0.9883 | 0.0450 | 0.6182→0.3999 | 0.1764→0.1666 | 94.3 |
| q_y | 17 | 0.80 | 0.9708 | 0.9944 | 0.9930 | 0.9850 | 0.0433 | 0.6535→0.4702 | 0.1870→0.1769 | 58.5 |
| q_y | 23 | 0.25 | 0.9700 | 0.9954 | 0.9949 | 0.9900 | 0.0500 | 0.5634→0.3852 | 0.1329→0.1386 | 121.0 |
| q_y | 42 | 0.05 | 0.9633 | 0.9938 | 0.9928 | 0.9917 | 0.0650 | 0.5749→0.4124 | 0.1659→0.1595 | 58.5 |
| q_y | 20260810 | 0.20 | 0.9667 | 0.9934 | 0.9916 | 0.9717 | 0.0383 | 0.6025→0.4070 | 0.1435→0.1327 | 120.5 |

