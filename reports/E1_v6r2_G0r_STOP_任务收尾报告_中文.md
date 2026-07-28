# FraudDistill E1 v6r2 G0r STOP 任务收尾报告

## 1. 本轮任务范围

本轮严格依据 `FraudDistill_E1_v6r2_G0r_STOP复盘与快速高标准推进计划.md` 执行，目标是先完成项目整理和历史结果归档，再按 v6r2 独立协议重建 E1 数据准入流程，并运行 G0r2 Gate。文档明确要求：若 G0r2 不通过，则不得进入 smoke、Pilot 或 Formal，因此本轮在 G0r2 触发 STOP 后停止后续实验阶段。

## 2. 归档与整理

| 项目 | 处理结果 |
| --- | --- |
| 旧 `outputs/e1_relation_gate_v6r1` | 已归档 |
| 旧 `data/prepared/e1_relation_gate_v6r1` | 已归档 |
| 旧 `reports/` 内容 | 已归档并从当前报告目录移除 |
| 归档目录 | `archive/pre_v6r2_20260728_223145` |
| 当前报告目录 | 仅保留本轮 v6r2 收尾报告 |

说明：原始数据和运行产物按项目 `.gitignore` 策略保留在本机，不提交到 GitHub；代码、配置、测试和本报告提交。

## 3. 代码整改摘要

| 模块 | 修改内容 |
| --- | --- |
| `configs/experiments/e1_relation_gate_v6r2.yaml` | 新增 v6r2 协议配置、master 规模、R2 容量门槛和 G0r2 Gate |
| `scripts/build_e1_relation_manifests.py` | 增加 `--protocol v6r2` 分支 |
| `scripts/run_e1_relation_v6r2.py` | 新增 v6r2 阶段运行入口；非 G0 阶段在 G0r2 PASS 前锁定 |
| `src/frauddistill/exp1_ccfa/relation_manifest.py` | 新增 v6r2 manifest 构建、Aegis train/validation/test 全 split 加载、泄漏 supercomponent、R1/R2/R3 互斥、R2 候选边普查、最大基数/最小代价匹配、master split 审计 |
| `tests/test_e1_relation_v6r2.py` | 新增 exact query、exact answer、subset overlap、max-cardinality matching 单元测试 |

## 4. 数据源容量

| 数据源 | 行数 | safe | unsafe | components |
| --- | ---: | ---: | ---: | ---: |
| PKU-SafeRLHF | 164,196 | 79,481 | 84,715 | 46,135 |
| Aegis/Nemotron-V2 | 10,730 | 6,550 | 4,180 | 10,730 |
| BeaverTails | 336,984 | 150,180 | 186,804 | 26,480 |

本轮已经修复 Aegis 仅加载 test split 的问题，当前 Aegis/Nemotron-V2 纳入 train、validation、test 三个 split。容量侧主要瓶颈不再是原始 source 行数，而是严格泄漏 supercomponent 去重、R1/R2/R3 互斥和 R2 双边匹配后，可用于高质量上下文碰撞组的有效容量不足。

## 5. G0r2 Gate 结果

最终机器判定：`E1_V6R2_STOP`

| Gate | 目标 | 实际 | 结论 |
| --- | ---: | ---: | --- |
| R1 groups | ≥ 3,800 | 3,800 | PASS |
| R2 max matching groups | ≥ 2,250 | 821 | FAIL |
| R3 unique rows | ≥ 24,900 | 12,266 | FAIL |
| R1/R2/R3 supercomponent overlap | 0 | 0 | PASS |
| master counts | 达到完整目标 | 未达到 | FAIL |
| pilot/formal test disjoint | true | true | PASS |
| duplicate audit | true | true | PASS |

## 6. Master Split 生成结果

| Split | 目标行数 | 实际行数 | safe | unsafe | components |
| --- | ---: | ---: | ---: | ---: | ---: |
| master_train | 20,000 | 16,166 | 2,316 | 13,850 | 14,666 |
| master_model_dev | 3,000 | 810 | 405 | 405 | 510 |
| master_calibration | 3,000 | 400 | 200 | 200 | 200 |
| pilot_test | 1,800 | 600 | 300 | 300 | 300 |
| formal_test | 9,000 | 3,000 | 1,500 | 1,500 | 1,500 |

master 未达标是 R2 和 R3 容量不足的直接结果。当前 split 逻辑没有伪造或重复填充容量，而是在严格互斥和去重后输出真实可用规模。

## 7. R2 双边匹配审计

| 指标 | 门槛 | 实际 | 结论 |
| --- | ---: | ---: | --- |
| candidate edge count | > selected groups | 22,852 | PASS |
| max matching groups | ≥ 2,250 | 821 | FAIL |
| selected groups | 2,250 | 821 | FAIL |
| q selector SMD | ≤ 0.10 | 0.0228 | PASS |
| y selector SMD | ≤ 0.10 | 0.0356 | PASS |
| log answer length SMD | ≤ 0.10 | 0.0064 | PASS |
| refusal gap | ≤ 0.05 | 0.0171 | PASS |
| independent q AUROC | 0.47-0.55 | 0.5090 | PASS |
| independent y AUROC | 0.47-0.55 | 0.5119 | PASS |
| largest row source | ≤ 0.70 | 0.4939 | PASS |
| largest source-pair | ≤ 0.50 | 0.4068 | PASS |
| cross-source groups | ≥ 0.40 | 0.7333 | PASS |
| source-pair types | ≥ 5 | 6 | PASS |
| third-source share | ≥ 0.05 | 0.2217 | PASS |

R2 的平衡性、source 多样性、q/y 单侧不可分性均达到要求，但最大匹配容量只有 821 组，低于 v6r2 的 2,250 组准入门槛。因此从统计设计角度看，当前 R2 质量是合格的，数量不足是唯一阻断点。

## 8. R2 Source Pair 分布

| Source pair | groups |
| --- | ---: |
| BeaverTails / PKU-SafeRLHF | 334 |
| Aegis/Nemotron-V2 / PKU-SafeRLHF | 217 |
| PKU-SafeRLHF / PKU-SafeRLHF | 130 |
| Aegis/Nemotron-V2 / BeaverTails | 51 |
| Aegis/Nemotron-V2 / Aegis/Nemotron-V2 | 48 |
| BeaverTails / BeaverTails | 41 |

跨源 group 比例为 0.7333，最大 source-pair 占比为 0.4068，第三 source 行占比为 0.2217。该结果说明 v6r2 的 source 多样性约束有效，当前 STOP 不是由单一数据源垄断造成。

## 9. 未进入后续阶段的原因

本轮未运行 smoke、Pilot、Formal，原因如下：

1. v6r2 文档要求 G0r2 是硬 Gate。
2. G0r2 最终未通过，机器判定为 `E1_V6R2_STOP`。
3. 未通过项为 `r2_max_matching`、`r3_unique_rows`、`master_counts`。
4. 在 STOP 条件下继续训练或报告模型效果会违反当前协议，也会污染论文叙事所需的可复现证据链。

## 10. 结论与后续建议

本轮整改完成了 v6r2 所要求的关键工程修复：Aegis 全 split 载入、严格泄漏 supercomponent、R1/R2/R3 互斥、R2 候选边普查、最大基数匹配和完整 Gate 判定。结果显示，R2 质量门基本全部通过，但严格条件下可匹配容量不足；R3 在 supercomponent 互斥后也不足以支撑 full master 规模。

下一轮若继续推进，应优先扩大 R2 和 R3 的有效 fraud_core 候选池，而不是放宽性能门槛。可考虑补充更多公开 fraud/safety 数据源、生成并审计新的 fraud_core 语义组件，或调整 R2 构造策略以增加可匹配边密度；这些改动完成前，不建议启动 smoke/Pilot/Formal。

## 11. 验证记录

| 验证项 | 结果 |
| --- | --- |
| G0r2 runner | 已执行，返回 STOP |
| G0r2 机器判定 | `outputs/e1_relation_gate_v6r2/g0/E1_V6R2_DECISION.json` |
| G0r2 机器报告 | `outputs/e1_relation_gate_v6r2/g0/E1_V6R2_REPORT_CN.md` |
| 编译检查 | `python -m compileall scripts src tests` 通过 |
| 单元测试 | `pytest -q` 通过，101 passed |
