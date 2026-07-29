# FraudDistill 实验1 Paired-Interaction R4 任务报告

- 协议：`E1-Paired-Interaction-R4`
- 最新阶段：`g0`
- 最新判定：`E1_R4_G0_STOP`
- Git 提交：`c554b5cdbb1d742e9804e312825f3680859402d8`

## 本轮执行说明

R4 新增独立流水线，不覆盖 R3。R3 原始输出、数据和报告已归档；R4 先做离线误差复盘，再按 A/Bq/By/C 四面板重构数据。


## r3_replay

- 判定：`E1_R4_R3_REPLAY_PASS`
- 用时：`0.099` 秒

## g0

- 判定：`E1_R4_G0_STOP`
- 用时：`53.465` 秒

### G0 构造检查

- Anchor 行数：`668`
- Model-Dev 行数：`210`
- Bq 候选：`{'candidate_groups': 468, 'anchor_groups': 150, 'model_dev_groups': 45}`
- By 候选：`{'candidate_rows': 6000, 'candidate_groups': 0, 'anchor_groups': 0, 'model_dev_groups': 0}`

| Gate | Pass |
|---|---:|
| `panel_a_anchor_360` | True |
| `panel_a_sources_120_60_60` | True |
| `bq_anchor_150_groups` | True |
| `by_anchor_150_groups` | False |
| `panel_c_anchor_120_groups` | False |
| `model_dev_360` | False |
| `anchor_1200` | False |
| `fraud_scope` | True |
| `split_leakage` | True |
| `unknown_provenance` | True |

### Source Fraud-Scope 容量

| Source | Rows | Safe | Unsafe |
|---|---:|---:|---:|
| PKU-SafeRLHF | 13938 | 1675 | 12263 |
| BeaverTails | 1844 | 331 | 1513 |
| Aegis | 356 | 149 | 207 |

### Pair 构造审计

| Check | Pass |
|---|---:|
| `bq_exact_q_opposite` | True |
| `by_near_y_opposite` | False |
| `c_exact_q_opposite` | True |
| `bq_capacity` | True |
| `by_capacity` | False |

## panel_c

- 判定：`E1_R4_PANEL_C_STOP`
- 用时：`0.438` 秒

### Panel C exact-q mixed outcome 审计

- 候选生成成功数：`2799`
- mixed-outcome exact-q group：`4`
- Anchor 可用 group：`4`
- Model-Dev 可用 group：`0`
- Anchor label 计数：`{'safe': 4, 'unsafe': 4}`

分析：R4 文档要求 480 个候选 q 仍不足 120 个 mixed group 时必须 STOP，不能通过修改 prompt、温度或人工挑选来补足。因此本轮未进入 Model-Dev/Anchor。

## 产物位置

- 数据目录：`C:\Users\18201\Desktop\FraudDistill\data\prepared\e1_paired_interaction_r4`
- 输出目录：`C:\Users\18201\Desktop\FraudDistill\outputs\e1_paired_interaction_r4`
- R4 必交审计文件写在数据目录与各阶段输出目录中。