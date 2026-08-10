# FraudDistill 实验1（E1-A / E1-B / E1-C）最终执行报告

> 协议名称：`E1-FINAL-TRIAD-v4-Relational-Ablation`  
> 正式产物目录：`experiments/exp1_input_ablation`（本报告与最终数据）  
> 中间数据目录：`data/prepared/e1_final_triad_v4`（冻结面板、审计、账本原始文件）  
> 报告生成时间：2026-08-10 · 当前状态：E1-A 完成；E1-B 面板冻结与 M0/M2/M3 完成，M1 训练进行中；E1-C 待回放  

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

分层要点：
- 语言维度：英文响应 11/3754（0.29%），中文响应 0/3746（0%）——本数据集中欺诈协助主要集中在英文诱导场景；
- 场景维度：roleplay 场景 9/3218（0.28%）明显高于 assistant 场景 2/4282（0.05%），角色扮演类诱导更易诱发协助行为；
- 类别维度：fake_job_posting 8/980（0.82%）与 network_friendship 3/448（0.67%）最集中，phishing / impersonation / fraudulent_service 为 0——说明欺诈协助并非均匀分布，而是集中于特定诱导类型。

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
| B3 context-stable / natural | 自然稳定对：q/y 内容自然匹配，机制上不依赖上下文关系 | 开源欺诈语料（source-derived open control） | v32 真实安全回复（zh）+ AEGIS 长安全回复（en）+ 开源安全控制 |

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
| B3 context-stable / natural | 2000 | 1000 | 1000 | 1003 | 997 |

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

- Gold 完成率 = 1.0；valid schema 率 = 1.0；全部 disagreement 已 adjudication 清零；
- 双 Judge agreement 面板级 0.9984（E1-A registry 口径）与 B 面板 double-gold 协议一致，质量满足论文要求。

## 5. 防泄漏与反快捷方式审计

### 5.1 Split（冻结）

- model_dev = 3600（各 stratum 1200、正负各半）；calibration = 1200；anchor = 1200；
- 跨 split exact-q 泄漏 = 0；exact-(q,y) 泄漏 = 0（union-find family 合并后为 0）；
- near-dup y（跨 split 近似重复 y）= 357，其中同标签 292（同标签近似重复不构成标签泄漏）；
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
| M1 XLM-R joint encoder | 语义联合编码器（learned primary）：xlm-roberta-base，中英混合，5 seeds × 3 views | ⏳ 训练中（0/15） |
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

【M1 训练进行中：0/15 任务完成；完成后自动填入：per-view Macro-F1 / AUPRC / Recall / FPR / Precision / AUROC（mean ± sd）、Δ_joint 与 bootstrap 95% CI、Holm 校正 p 值、4/5 seeds gate 判定、wrong_q+y 负控制、stratum 分层指标、q_y vs best-single per-seed 明细】

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

## 9. 成本记录

- 累计 API 成本：**¥86.96**（Qwen ¥54.43 / DeepSeek ¥32.53）；
- 协议硬上限：Qwen ¥60 / DeepSeek ¥60 / 本轮新增 ¥100——当前未触顶；
- Anchor 四视图实际成本：DeepSeek ¥9.4 + Qwen ¥11.5（指南目标 ≤ ¥9/provider，略超但远低于软上限）；
- M1 训练、统计检验、E1-C 回放、报告生成全部离线（CPU / 静态计算），**后续无新增 API 成本**。

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

## 附录 B：M1 训练日志摘录

【训练完成后自动追加：每个 (mode, seed) 的 epoch/step/loss、anchor 指标、阈值与耗时】

