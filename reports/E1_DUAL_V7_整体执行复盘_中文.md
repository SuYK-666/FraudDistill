# FraudDistill E1 v7 整体执行复盘

- 协议：`E1-DUAL-NATURAL-DIAGNOSTIC-v1.0`
- 本轮执行范围：P0 canonical/cache 修复 + P1 800-call blind Pilot
- 最新判定：`E1_V7_PILOT_STOP`
- 最新代码提交基线：`73388c5465b0ba3be1f8144819b295b4471642d1`

## P0 结果

P0 为无 API 阶段，已通过。

| 项目 | 结果 |
|---|---:|
| Fraud-R1 raw rows | 4,282 |
| canonical cases | 2,141 |
| q hash missing | 0 |
| q hash collision | 0 |
| Pilot cases | 200 |
| Pilot 分层 | 2 语言 x 5 类别 x 20 |
| 旧 R4.1 Qwen cache rows | 4,282 |
| 可映射旧 cache rows | 4,282 |
| 可解释 canonical cases | 2,141 |
| 旧 cache 中至少 2 个 ok replicate 的 canonical | 2,025 |
| secret scan | PASS |
| prompt parity | PASS |

P0 修复确认了 R4.1 中 “4,282 个候选” 实际对应约 2,141 个 canonical q 的两次自然采样。新协议后续不再把 base/level-up 顶层重复样本当作完全独立 q。

## P1 Pilot 执行结果

Pilot 按协议只做自然响应生成、三状态参考标注和容量判定，未运行 q-only/y-only/q+y FraudDistill 指标。

目标模型生成：

| Model | Expected | Success | Completion |
|---|---:|---:|---:|
| Qwen | 400 | 400 | 1.0000 |
| DeepSeek | 400 | 400 | 1.0000 |

参考标注：

| 项目 | 数值 |
|---|---:|
| expected label rows | 1,600 |
| current-schema label rows | 1,600 |
| label API success | 1,600 |
| consensus SUCCESS | 761 |
| consensus FAILURE | 22 |
| consensus UNCERTAIN | 17 |

模型容量判定：

| Model | Completion | Valid | Agreement | Uncertain/Invalid | Risk events | EN risk | ZH risk | Risk categories | Mixed groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek | 1.0000 | 1.0000 | 0.9700 | 0.0300 | 20 | 8 | 12 | 2 | 5 |
| Qwen | 1.0000 | 1.0000 | 0.9875 | 0.0125 | 2 | 0 | 2 | 1 | 0 |

## STOP 原因

本轮 STOP 不是 API 失败，也不是缓存或 canonical 问题，而是 Pilot 容量 Gate 不满足：

- 单模型风险事件要求：`>=40`，实际 DeepSeek 20、Qwen 2。
- 每种语言风险事件要求：`>=15`，实际 DeepSeek en 8/zh 12，Qwen en 0/zh 2。
- 风险类别覆盖要求：`>=4/5`，实际 DeepSeek 2 类、Qwen 1 类。
- mixed-outcome canonical group 要求：整体 `>=20`，实际 5。

因此协议规定的 `GO_FULL_E1` 不成立，不能进入全量 E1-A/E1-B，也不能构建 Model-Dev/Anchor。

## 重要分析

这轮结果提供了一个清晰方向：新三状态标注协议本身可用，标注一致率和无效率已明显优于 R4.1 的旧二元标签。DeepSeek 作为目标模型比 Qwen 更容易产生风险事件，但 200-case Pilot 下仍不足以支撑后续诊断面板的容量需求。

如果继续推进，下一步应优先考虑文档中的多轮 CONTINUE 路线，而不是修改 q/y/q+y 评估器。当前自然首轮风险事件太少，直接进入全量会导致 diagnostic mixed-outcome 容量不足。

## 数据保留

本轮所有原始与中间数据均已本地保留：

- `data/prepared/e1_dual_v7/CANONICAL_CASES.jsonl`
- `data/prepared/e1_dual_v7/CANONICAL_MAP.csv`
- `data/prepared/e1_dual_v7/PILOT_CASES.jsonl`
- `data/prepared/e1_dual_v7/PILOT_RESPONSES.jsonl`
- `data/prepared/e1_dual_v7/PILOT_REFERENCE_LABELS.jsonl`
- `data/prepared/e1_dual_v7/PILOT_CONSENSUS.jsonl`
- `data/prepared/e1_dual_v7/PILOT_MODEL_DECISION.json`
- `outputs/e1_dual_v7`

这些数据目录受 `.gitignore` 保护，不提交 GitHub；代码、配置、测试和报告会提交。
