# FraudDistill 实验 1 V8 A2C 最终效果报告

生成时间：2026-07-30  
协议：`E1-V8-A2C-OFFICIAL-DELTA-STAGE-PRESSURE-v1.0`  
数据目录：`data/prepared/e1_v8_a2c`  
代码基线：`247a9d7e2904a5e4b4128405471cfab11fdd84e8`

## 1. 最终结论

本轮实验应当继续执行到当前终点，且已经完成到协议允许的最大范围。最终结论为：

| 模块 | 状态 | 结论 |
|---|---|---|
| P0 数据与工程 Gate | PASS | Fraud-R1 四阶段 canonical、split、prompt、secret scan 均合格 |
| 模型参数探测 | PASS | Qwen 与 DeepSeek 的 target / labeler 模型均可调用 |
| A-Delta 官方 Prompt 对齐 | GREEN | 官方 Role-play 明显提升 DeepSeek 自然风险容量 |
| C-ISO 四阶段压力 | AMBER | 质量合格，风险容量有价值但不足以直接进入 1,200 行 Tier-A Anchor |
| Probe 构建 | STOP | q-only 结构上限过高，当前 Probe 面板不能用于 q/y/q+y 结论 |
| 是否进入 Full E1 | false | 当前不建议直接全量 Anchor，应先修复 Probe 面板构造与 C 容量结构 |

最终 Decision：

```json
{
  "p0": "PASS",
  "dataset_terms_gate": "PASS",
  "model_param_probe": "PASS",
  "a_quality": "PASS",
  "a_capacity": "GREEN",
  "c_quality": "PASS",
  "c_capacity": "AMBER",
  "probe": "STOP",
  "diagnostic_primary_target": "deepseek",
  "recommended_full_route": "REPAIR",
  "eligible_for_full_e1": false
}
```

核心判断：本轮不是完全失败。A 方向已经拿到强结果，能支撑“官方 Prompt 对齐后 DeepSeek 自然风险显著增加”的论文叙事；C 方向也证明四阶段自然回答能产生一定风险容量和 mixed groups，但当前阶段分布不理想，Probe 面板 q-only 泄漏过强，因此不能直接进入最终 Full。

## 2. 数据与完整性

P0 数据审计全部通过：

| 指标 | 数值 |
|---|---:|
| raw rows | 4,282 |
| canonical case | 2,141 |
| English | 1,071 |
| Chinese | 1,070 |
| Pilot | 200 |
| Model-Dev | 400 |
| Frozen Anchor | 1,541 |
| split overlap | 0 |
| 每 case stage 数 | 4 |
| stage id | 0/1/2/3 |

类别分布：

| 类别 | canonical 数 |
|---|---:|
| fake job posting | 300 |
| fraudulent service | 600 |
| impersonation | 600 |
| network friendship | 169 |
| phishing | 472 |

data type 分布：

| data type | canonical 数 |
|---|---:|
| message | 1,369 |
| email | 472 |
| job posting | 300 |

目标回答完成情况：

| track | 目标回答 | 状态 |
|---|---:|---|
| A-Delta | 1,600 | 完成 |
| C-ISO stage 1-3 | 2,400 | 完成 |
| 合计 | 4,000 | 完成 |

目标回答质量：

| track | expected | observed | completion | valid response | provenance | truncation |
|---|---:|---:|---:|---:|---:|---:|
| A-Delta | 1,600 | 1,600 | 100.0% | 100.0% | 100.0% | 2.625% |
| C-ISO stage 1-3 | 2,400 | 2,400 | 100.0% | 100.0% | 100.0% | 12.625% |

注意：C-ISO 的 truncation 比 A-Delta 高，说明后续若要进入 Full，应考虑降低 `max_tokens` 触发 length 的风险，或在报告中单独分析 length finish_reason 对标签状态的影响。

## 3. A-Delta 结果

A-Delta 主问题是：同期 A0 parity control 与 A1 official treatment 相比，官方 Role-play 是否提升自然风险容量。

标签质量：

| 视图 | state agreement | binary agreement | kappa | uncertain | 质量 |
|---|---:|---:|---:|---:|---|
| O official-y | 93.50% | 97.125% | 0.816 | 6.50% | PASS |
| X contextual-qy | 94.438% | 96.25% | 0.747 | 5.563% | PASS |

A-Delta 核心表：

| arm/model | n | O failure | O failure rate | O continue | X failure | X failure rate | X assist |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 parity / DeepSeek | 400 | 20 | 5.00% | 99 | 32 | 8.00% | 32 |
| A1 official / DeepSeek | 400 | 27 | 6.75% | 67 | 55 | 13.75% | 55 |
| A0 parity / Qwen | 400 | 13 | 3.25% | 10 | 17 | 4.25% | 17 |
| A1 official / Qwen | 400 | 10 | 2.50% | 22 | 14 | 3.50% | 14 |

A-Delta 容量 Gate：

| 指标 | 结果 | A-GREEN 门槛 | 状态 |
|---|---:|---:|---|
| DeepSeek A1 X failure | 55/400 = 13.75% | ≥32/400 | PASS |
| DeepSeek English risk | 32 | ≥12 | PASS |
| DeepSeek Chinese risk | 23 | ≥12 | PASS |
| DeepSeek risk category coverage | 5/5 | ≥4/5 | PASS |
| global mixed canonical | 28 | ≥20 | PASS |
| conservative Round-1 projection | 548.72 | ≥400 | PASS |
| label quality | PASS | PASS | PASS |

A-Delta 最终等级：GREEN。

分析：

DeepSeek 在官方 Role-play 下的 X contextual failure 从 A0 的 8.00% 提升到 A1 的 13.75%，绝对提升 5.75pp，相对提升约 71.9%。这说明 V7 的低风险率确实有一部分来自 prompt 未完全官方对齐，而不是目标模型绝对安全。DeepSeek A1 同时覆盖中英文和全部 5 类风险类别，global mixed canonical 达到 28，具备独立自然诊断价值。

Qwen 的风险率仍保持较低。A1 Qwen X failure 为 3.50%，低于 A0 的 4.25%。这支持把 Qwen 作为强安全对照，而不是诊断主模型。模型选择规则下，DeepSeek 应作为后续 E1-B 的 diagnostic primary target。

论文叙事建议：A 方向可以作为“官方对齐后的自然发生率主线”。措辞应限定为 official-alignment bundle effect，不能声称是某一个单独 prompt 因素导致提升。

## 4. C-ISO 四阶段结果

C-ISO 主问题是：四阶段自然压力是否提供足够 E1-B 风险容量与 exact-q mixed groups。

标签质量：

| 视图 | state agreement | binary agreement | kappa | uncertain | 质量 |
|---|---:|---:|---:|---:|---|
| O official-y | 96.219% | 98.00% | 0.782 | 3.781% | PASS |
| X contextual-qy | 97.875% | 98.188% | 0.805 | 2.125% | PASS |

C-ISO 总体容量：

| 指标 | 结果 | C-GREEN 门槛 | 状态 |
|---|---:|---:|---|
| target response completion | 3,200/3,200 | ≥99.5% | PASS |
| X strict failure | 128 | ≥160 | 未达 GREEN |
| X strict failure rate | 4.00% | ≥5.00% | 未达 GREEN |
| global mixed q-stage groups | 64 | ≥60 | PASS |
| risk category coverage | 5/5 | 5/5 | PASS |
| English X failure | 73 | ≥60 | PASS |
| Chinese X failure | 55 | ≥60 | 未达 GREEN |
| 每 stage X failure | stage2=7, stage3=9 | 每 stage ≥20 | 未达 GREEN |
| conservative X projection | 350.72 | ≥600 | 未达 GREEN |

C-ISO 最终等级：AMBER。

stage × model 表：

| stage | model | n | O failure | O rate | O continue | X failure | X rate | X assist |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 BASE | DeepSeek | 400 | 27 | 6.75% | 67 | 55 | 13.75% | 55 |
| 0 BASE | Qwen | 400 | 10 | 2.50% | 22 | 14 | 3.50% | 14 |
| 1 CREDIBILITY | DeepSeek | 400 | 18 | 4.50% | 47 | 42 | 10.50% | 42 |
| 1 CREDIBILITY | Qwen | 400 | 1 | 0.25% | 5 | 1 | 0.25% | 1 |
| 2 URGENCY | DeepSeek | 400 | 0 | 0.00% | 9 | 7 | 1.75% | 7 |
| 2 URGENCY | Qwen | 400 | 0 | 0.00% | 0 | 0 | 0.00% | 0 |
| 3 EMOTION | DeepSeek | 400 | 3 | 0.75% | 6 | 9 | 2.25% | 9 |
| 3 EMOTION | Qwen | 400 | 0 | 0.00% | 0 | 0 | 0.00% | 0 |

X strict failure by stage：

| stage | X failure |
|---:|---:|
| 0 BASE | 69 |
| 1 CREDIBILITY | 43 |
| 2 URGENCY | 7 |
| 3 EMOTION | 9 |

X conservative projection：

| model | stage | observed failures | Wilson lower | projected lower |
|---|---:|---:|---:|---:|
| DeepSeek | 0 | 55 | 0.1072 | 165.17 |
| DeepSeek | 1 | 42 | 0.0786 | 121.16 |
| DeepSeek | 2 | 7 | 0.0085 | 13.10 |
| DeepSeek | 3 | 9 | 0.0119 | 18.31 |
| Qwen | 0 | 14 | 0.0210 | 32.30 |
| Qwen | 1 | 1 | 0.0004 | 0.68 |
| Qwen | 2 | 0 | 0.0000 | 0.00 |
| Qwen | 3 | 0 | 0.0000 | 0.00 |
| 合计 |  | 128 |  | 350.72 |

分析：

C-ISO 的质量已经合格，说明双标签器机制在四阶段数据上可用；但容量和趋势没有达到预期。最关键的问题是风险集中在 stage0 和 stage1，stage2、stage3 反而显著下降。这与协议预期的“后期风险高于 Base”相反，也解释了 conservative projection 只有 350.72。

这不是简单增加样本就能完全解决的问题。若 full 按当前分布扩展，stage2/3 仍会贡献很少风险，Anchor 会偏向 BASE/CREDIBILITY，不利于形成稳定的四阶段叙事。C-ISO 当前仍有价值，因为它产生了 128 个 X strict failure、64 个 global mixed q-stage groups 和 5/5 类别覆盖；但它更适合 Tier-B 或作为 Model-Dev 修复依据，而不是直接进入 Tier-A Full。

## 5. Probe 构建结果

Probe 目标是验证 q-only、y-only、q+y 是否能形成预期梯度。但当前 Probe 没有运行 evaluator，因为面板结构 Gate 已经 STOP。

Probe panel 构造结果：

| 指标 | 数值 |
|---|---:|
| panel size | 240 |
| positive | 120 |
| negative | 120 |
| q-group majority oracle accuracy | 0.9833 |
| 允许上限 | ≤0.72 |
| 结论 | PROBE_STRUCTURE_STOP |

Probe panel 分布：

| 维度 | 分布 |
|---|---|
| language | en 131；zh 109 |
| stage | stage0 99；stage1 64；stage2 37；stage3 40 |
| model | DeepSeek 168；Qwen 72 |
| category | fake job posting 96；network friendship 43；phishing 41；impersonation 33；fraudulent service 27 |

分析：

q-group majority oracle accuracy 达到 0.9833，说明同一个 q-stage group 内几乎没有形成充分的正负混合。任何只看 q 的确定性方法都可能接近直接猜中标签，因此继续运行 q/y/q+y evaluator 会导致 q-only 虚高，不能检验“q+y 相对 y-only 的互补性”。

这不是 evaluator 能解决的问题，而是面板构造和数据结构问题。当前 risk rows 主要来自 DeepSeek、stage0/1，而 safe rows 大量来自 Qwen、stage2/3；标签与 stage/model/q-group 强绑定，导致 q-only 泄漏过强。按协议必须停止 Probe，并在 Model-Dev 或下一轮修复面板构造规则。

## 6. 是否需要继续跑

当前不需要继续跑 API。理由如下：

1. Qwen 已恢复后，缺失标签已经补齐到质量 Gate PASS。
2. A-Delta 已经 GREEN，继续重复 A 不会增加主结论。
3. C-ISO 已经 AMBER，主要瓶颈是 stage2/3 风险不足和 projection 不达标，不是标签缺失。
4. Probe 已经结构 STOP，运行 evaluator 会得到不合规结果。
5. 最终 Decision 已经明确 `eligible_for_full_e1=false`，继续 full 会浪费 API 且不符合预注册流程。

因此当前应停止，并把本轮作为“V8 Pilot 完成但不放行 Full”的结果。

## 7. 后续建议

推荐下一步不直接扩 full，而是先修复 E1-B 数据结构：

1. 保留 A-Delta 作为自然发生率主线。A-GREEN 是本轮最强结果，DeepSeek official Role-play 的自然风险率足够支持论文叙事。
2. C-ISO 暂定为 AMBER 数据源。它能提供风险样本和 mixed groups，但不能直接承诺 1,200 行 Tier-A Anchor。
3. 重新设计 Probe builder。核心目标是降低 q-group majority oracle accuracy 到 ≤0.72，优先增加 exact-q mixed pairs，而不是简单平衡正负类。
4. 优先围绕 DeepSeek 构造诊断主样本。Qwen 在 stage1-3 几乎没有风险，可作为安全对照，但不适合作为风险容量来源。
5. 对 stage2/3 做单独复盘。当前后期压力没有产生预期风险增长，可能是 Fraud-R1 level-up 文本本身更容易触发防御，也可能是 isolated-stage renderer 缺少前序上下文导致后期信息不自然。
6. 若要进入 Full，应先在 Model-Dev 上验证新的面板构造规则，而不是直接把当前 Pilot 规则扩大。

## 8. 最终判断

本轮适合在论文中保留两类结论：

第一，官方 Prompt 对齐会显著改变自然风险测量。DeepSeek 从 A0 的 8.00% X failure 提升到 A1 的 13.75%，并覆盖中英文与 5 类风险，是可以写入主线的实证结果。

第二，四阶段自然压力目前只能作为 AMBER 证据。它提供了 128 个 strict contextual failure 和 64 个 mixed q-stage groups，但风险分布集中在前两阶段，conservative projection 不足，Probe 又因 q-only 结构泄漏停止。因此它不能直接支撑 q+y≈0.90 的最终主张。

最终建议：停止本轮继续调用 API，进入下一轮“Probe 面板与 C-ISO 风险结构修复”设计。
