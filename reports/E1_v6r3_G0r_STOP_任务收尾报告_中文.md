# FraudDistill E1 v6r3 G0r3 任务收尾报告

## 1. 本轮范围

本轮依据 `FraudDistill_E1_v6r2_STOP复盘与v6r3下一步行动指南.md` 执行，已将 v6r2 结果冻结归档，并新建独立协议 `E1_CPU_CCF-A_v6r3`。v6r3 的重点不是继续沿用 v6r2 小修，而是重新定义 G0 数据准入：R2 改为 leakage supercomponent 级 matching，R3 改为严格 50/50 balanced capacity，master 缩放为 9k/1.5k/1.5k，并尝试纳入预注册的 WildGuardTrain。

## 2. 归档与整理

| 项目 | 结果 |
| --- | --- |
| v6r2 `outputs/e1_relation_gate_v6r2` | 已归档 |
| v6r2 `data/prepared/e1_relation_gate_v6r2` | 已归档 |
| v6r2 `reports/` | 已归档 |
| 第一次 v6r3 临时输出 | 已归档 |
| 归档目录 | `archive/pre_v6r3_20260728_232701` |

当前正式保留的运行产物位于：

| 类型 | 路径 |
| --- | --- |
| G0r3 数据产物 | `data/prepared/e1_relation_gate_v6r3` |
| G0r3 决策产物 | `outputs/e1_relation_gate_v6r3/g0` |

## 3. 代码整改

| 文件 | 内容 |
| --- | --- |
| `configs/experiments/e1_relation_gate_v6r3.yaml` | 新增 v6r3 独立配置、样本配额、R2/R3/G0 Gate |
| `src/frauddistill/exp1_ccfa/relation_manifest_v6r3.py` | 新增 v6r3 manifest 构建、WildGuard admission、component-level R2 matching、R3 balanced capacity、split quota audit |
| `scripts/run_e1_relation_v6r3.py` | 新增 v6r3 runner、G0 决策、stage lock 状态机 |
| `scripts/build_e1_relation_manifests.py` | 增加 `--protocol v6r3` |
| `tests/test_e1_relation_v6r3.py` | 增加 row UID、component matching、top-level STOP、split balance 测试 |

本轮还修正了 R3 选择逻辑：当 safe balanced capacity 不足时，不再继续单边拿满 unsafe，而是按 `min(requested_per_label, k_max)` 保持真实 50/50 平衡容量。

## 4. Clean Run 记录

| 项目 | 值 |
| --- | --- |
| clean-run 提交 | `8f685c5444ba1a29d30b18a7d43652cd93ace3e5` |
| 运行时 `git status` | clean |
| G0r3 决策 | `E1_V6R3_G0_STOP` |
| 后续 stage | 未运行，smoke 验证返回 `E1_V6R3_SMOKE_LOCKED` |

## 5. 数据源审计

| Source | rows | safe | unsafe | components | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| PKU-SafeRLHF | 164,196 | 79,481 | 84,715 | 46,135 | loaded |
| Aegis/Nemotron-V2 | 10,730 | 6,550 | 4,180 | 10,730 | loaded |
| BeaverTails | 336,984 | 150,180 | 186,804 | 26,480 | loaded |
| WildGuardTrain | 0 | 0 | 0 | 0 | FAIL |

WildGuardTrain 未能加载，错误为 Hugging Face gated dataset 需要认证：

```text
DatasetNotFoundError: Dataset 'allenai/wildguardmix' is a gated dataset on the Hub. You must be authenticated to access it.
```

因此 `required_source_failures` 与 `wildguard_admitted` 均未通过。这是本轮 G0r3 的硬阻断项之一。

## 6. G0r3 Gate 总览

| Gate | 目标 | 实际 | 结论 |
| --- | ---: | ---: | --- |
| git status clean | true | true | PASS |
| R1 groups | ≥ 3,800 | 3,800 | PASS |
| R2 true max matching | ≥ 2,250 | 759 | FAIL |
| R2 selected groups | 2,250 | 759 | FAIL |
| R3 balanced capacity | ≥ 12,000 | 760 | FAIL |
| R3 selected rows | 11,800 | 760 | FAIL |
| R3 label balance | 50/50 | 380/380 | PASS |
| R3 source count | ≥ 3 | 3 | PASS |
| R3 largest source | ≤ 0.70 | 0.7013 | FAIL |
| duplicate audit | 0 hit | 0 hit | PASS |
| R1/R2/R3 overlap | 0 | 0 | PASS |
| split audit | exact | not exact | FAIL |

最终未通过顶层 checks：

```text
required_source_failures
wildguard_admitted
r2_true_max_matching
r2_selected_groups
r2_audit
r3_balanced_capacity
r3_selected_rows
r3_largest_source
split_audit
```

## 7. R2 Component-Level Matching

| Policy | edges | max matching groups |
| --- | ---: | ---: |
| C0 | 8,939 | 660 |
| C1 | 8,974 | 660 |
| C2 | 8,974 | 660 |
| C3 | 12,527 | 717 |
| C4 | 14,121 | 759 |

最终选择 C4 soft family + soft refusal policy，其平衡质量较好，但容量仍不足：

| 指标 | 实际 | Gate |
| --- | ---: | --- |
| q SMD | 0.0315 | PASS |
| y SMD | 0.0390 | PASS |
| answer length SMD | 0.0061 | PASS |
| refusal gap | 0.0092 | PASS |
| q AUROC | 0.5137 | PASS |
| y AUROC | 0.5133 | PASS |
| largest row source | 0.5013 | PASS |
| largest source-pair | 0.3478 | PASS |
| cross-source group rate | 0.7378 | PASS |
| source-pair types | 6 | PASS |
| third-source share | 0.2233 | PASS |

R2 结论：v6r3 证明了 v6r2 的 821 不是严格 component-level 容量；在每个 supercomponent 只能使用一次后，真实最大匹配降为 759。质量控制仍然通过，但数量远低于 2,250，因此不得进入模型实验。

## 8. R3 Balanced Capacity

| 指标 | 值 |
| --- | ---: |
| safe-only supercomponents | 229 |
| unsafe-only supercomponents | 11,664 |
| dual-label supercomponents | 151 |
| max balanced R3 rows | 760 |
| selected rows | 760 |
| safe / unsafe | 380 / 380 |

R3 结论：v6r3 修正后，当前三源 fraud_core 的安全标签严重不足，真实平衡容量只有 760 行，而不是 v6r2 任意首行选择得到的 12,266 行。这个结果解释了为什么 v6r2 的 master_train 和 smoke_eval 会出现严重 unsafe 偏斜。

## 9. Split 结果

| Split | rows | safe | unsafe |
| --- | ---: | ---: | ---: |
| master_train | 3,960 | 1,980 | 1,980 |
| master_model_dev | 600 | 300 | 300 |
| master_calibration | 600 | 300 | 300 |
| pilot_test | 918 | 459 | 459 |
| formal_test | 3,000 | 1,500 | 1,500 |

这些 split 是容量不足下的真实可构造结果，均未达到 v6r3 冻结目标规模。因此 split audit 失败是预期结果，不应通过重复样本或放宽 Gate 伪造完整规模。

## 10. 结论

本轮已完成 v6r3 协议工程化和一次 clean G0r3。结果为严格 STOP，且没有运行 smoke/Pilot/Formal。当前最关键的事实是：

1. WildGuardTrain 因 gated dataset 未授权无法纳入，预注册的新数据源未满足 admission。
2. R2 component-level 最大匹配只有 759 组，低于 2,250。
3. R3 平衡容量只有 760 行，低于 12,000。
4. 现有三源公开数据在严格 response-level fraud_core、supercomponent 去重和 50/50 标签平衡下，无法支撑 v6r3 的完整 master/test 规模。

后续若继续推进，应先解决 WildGuardMix 的 Hugging Face 访问授权，或更换同等预注册、可公开访问、具备 response-level harm label 的数据源；在数据源 admission 通过前，不建议进入模型侧 smoke/Pilot/Formal。

## 11. 验证

| 验证项 | 结果 |
| --- | --- |
| `python -m compileall scripts src tests` | PASS |
| `pytest -q` | PASS，105 passed |
| clean G0r3 | STOP，产物已保留 |
| smoke stage lock | PASS，返回 `E1_V6R3_SMOKE_LOCKED` |

