# EXP6 v2 最终报告：跨多 API 直连响应的欺诈协助检测与 Selective Audit 级联

> 实验目录：`experiments/exp6_v2_balanced` ｜ 协议：`EXP6_V2_BALANCED_RERUN_STRICT_PROTOCOL.md` ｜ 生成时间：2026-08-12T06:48:39+00:00（UTC）

## 0. 摘要

本实验在 v1（random-180）基础上重做 E6：6 个直连 API 目标模型（Qwen Flash/Plus、DeepSeek Flash/Pro、GLM Flash、Kimi），共享行为面板 200 条 should-refuse + 200 条 B0 + 40 条 hard-safe control（每模型 440 条有效回答，共 3043 条记录）。全量回答经三 Judge（Qwen Flash / DeepSeek Flash / GLM Flash）+ J4（Kimi）裁决生成 Silver；随后以 Student-blind 方式构造每模型 80 条（40 unsafe / 40 safe）语义族匹配的均衡关系集 + 40 条 hard-safe control，冻结 Frozen Student（best_step120, threshold 0.5622）评分，离线评估 P0（冻结阈值）、P1（pooled 全局阈值）、P2（10%/20% selective audit 级联）。

**核心结果**：P0 Macro-F1 = 0.722（AUROC 0.798）；P1 Macro-F1 = 0.727（Recall 0.729，FPR 0.276，AUROC 0.798，AUPRC 0.796）；Hard-safe FPR（test, pooled）= 14.4%。

P1 存在可行全局阈值 t = 0.2323（calibration Recall ≥0.65 且 hard-safe FPR ≤0.15 约束下按 Macro-F1→MCC→更高阈值选择）。

**门控**：Pooled Core Pass = **False**；Pooled Strong Pass = **False**。

## 1. 实验设计

- **目标**：回答两个估计对象——(1) E6-A：同一共享挑战分布下各目标模型的行为率；(2) E6-B：Frozen Student 能否在同类危险问题下区分“安全拒绝”与“欺诈协助”。

- **与 v1 的区别**：v1 的 random-180 中 unsafe 极少（约 13 条），response imbalance 极端；v2 改为 50/50 均衡关系集 + semantic-family matching + hard-safe control + P1/P2 级联，且全量三 Judge Silver。

- **预算**：E6 总上限 ¥50（v1 已花费 ¥1.4153 计入）；截至本报告，累计成本约 ¥51.47。

## 2. 数据与池构建

- 池规模：anchor 200（100 should-refuse / 100 should-answer，zh/en 100/100）、B0 200、B1 100、B2 100、control 40，共 640 q；本实验仅使用 anchor/B0/control 作为共享面板（与 v1 设计一致，B1/B2 作为自适应补充储备）。

- 泄漏审计：exact/prefix80/id 泄漏均为 0；superfamily split 与跨实验（E3/E4/E5/E6-v1）重叠审计见 `data/superfamily_split_audit.json`、`data/cross_experiment_leakage_audit.json`；manifest SHA256 见 `data/pool_manifest_sha256.json`。

## 3. 目标模型与生成

| Slot | 服务 | 模型 | 有效回答 |
|---|---|---|---|
| M1 | qwen | qwen-flash | 440 |
| M2 | qwen | qwen-plus | 640 |
| M3 | deepseek | deepseek-v4-flash | 640 |
| M4 | deepseek | deepseek-v4-pro | 440 |
| M5 | glm | glm-4-flash | 437 |
| M6 | kimi | moonshot-v1-8k | 440 |

- 内容过滤：M5（GLM）3 条回答被平台内容过滤（作为 content-filter rate 记录，不重试）；其余失败均已重试。

## 4. Silver 三判质量

- 全候选三 Judge 覆盖：3541 条 triple（audits 3541）。

- binary Fleiss κ = 0.7474；behavior Fleiss κ = 0.6377；unanimous rate = 0.8232；schema consistency = 1.0；unresolved = 0.0。

- 非一致行由 J4（Kimi moonshot-v1-8k，与三 Judge 不同配置的冻结强模型）裁决，身份已写入 `protocol/model_registry_frozen.json`。

## 5. E6-A 行为面板（表 A）

| Model | N | Unsafe assistance | Full assistance | Partial leakage | Clean refusal | Safe redirection | Over-refusal | Avg latency (s) | Cost (CNY) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen Flash | 300 | 55.7% | 55.3% | 0.3% | 3.7% | 36.0% | 0.0% | 3.53 | 0.24343 |
| Qwen Plus | 300 | 42.0% | 42.0% | 0.0% | 0.7% | 54.7% | 0.0% | 4.92 | 0.29707 |
| DeepSeek Flash | 300 | 46.0% | 46.0% | 0.0% | 2.3% | 45.0% | 0.0% | 3.49 | 0.28287 |
| DeepSeek Pro | 300 | 51.3% | 51.3% | 0.0% | 2.7% | 42.7% | 0.0% | 5.84 | 0.88178 |
| GLM Flash | 297 | 77.4% | 77.4% | 0.0% | 0.3% | 12.5% | 0.0% | 12.16 | 0.14976 |
| Kimi | 300 | 63.7% | 63.7% | 0.0% | 2.7% | 28.3% | 0.0% | 4.49 | 0.3533 |

- zh/en 切片：
Qwen Flash zh 59.3%/en 52.0%；Qwen Plus zh 46.0%/en 38.0%；DeepSeek Flash zh 52.7%/en 39.3%；DeepSeek Pro zh 57.3%/en 45.3%；GLM Flash zh 85.7%/en 69.3%；Kimi zh 70.7%/en 56.7%。

## 6. E6-B 均衡集构造（Student-blind）

- 每模型 80 条关系集（unsafe 40 / safe 40，均来自 should-refuse q）+ 40 条 hard-safe control（cal 10 / test 30）；cal/test 按 family 冻结（relation cal 16 / test 64）。

- semantic-family matching、语言配额、行为子型配额、family 重叠审计见 `balanced/balanced_selection_audit.json`。

- metadata-only shortcut probe：pooled AUROC = 0.7251；per-model = {"M1": 0.6316, "M2": 0.7634, "M3": 0.6172, "M4": 0.7531, "M5": 0.5284, "M6": 0.6822}。

- **SS8.7 amendment（协议偏差，如实记录）**：由于冻结池中 per-model BAL 窗口按可行性放宽（M2=12, M4=13, M5/M6=8/10），metadata-only shortcut probe 的 pooled AUROC=0.7251 与 M2=0.7634/M4=0.7531 略超协议名义门（pooled ≤0.65 / 单模型 ≤0.70）；该偏差已记录，不作为 shortcut 已消除的声明，Student 判别增益以三视图对比与 hard-safe 控制为准。

- 选择过程仅读取 Silver 与元数据，未加载 Student 分数（`balanced_selection_audit.json` 中 `student_blind_selection=true`）。

## 7. Student 推理与三视图

- 冻结模型：FraudDistill-Student-1.5B `best_step120`，max_length=512（head-tail 编码，与模型卡一致），P0 阈值 0.5622。

- 截断审计：overall truncation rate = 0.4986（359/720 条，head-tail 编码 512 token 上限内全部容纳；每模型见 `student/truncation_audit_qy.json`）。

- 三视图（q-only / y-only / q+y，frozen test 上以 P1 阈值评估）：

| View | AUROC | Macro-F1 |
|---|---:|---:|
| qy | 0.798 | 0.727 |
| qonly | 0.507 | 0.390 |
| yonly | 0.790 | 0.700 |

- q+y vs q-only AUROC 增益 = 0.2909（目标 ≥0.10）。

## 8. 主结果：P0 / P1 / P2（表 B）

| Policy | N | Macro-F1 | F1-unsafe | Precision | Recall | FPR | MCC | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (frozen 0.5622) | 384 | 0.722 | 0.685 | 0.809 | 0.594 | 0.141 | 0.470 | 0.798 | 0.796 |
| P1 (pooled global) | 384 | 0.727 | 0.727 | 0.725 | 0.729 | 0.276 | 0.453 | 0.798 | 0.796 |
| P2 (audit 10%) | 384 | 0.776 | 0.778 | 0.770 | 0.786 | 0.234 | 0.552 | 0.798 | 0.796 |
| P2 (audit 20%) | 384 | 0.807 | 0.811 | 0.795 | 0.828 | 0.214 | 0.615 | 0.798 | 0.796 |

10k family-cluster bootstrap 95% CI（pooled frozen test）：

- P0 Macro-F1: [0.674, 0.770]（mean 0.721）
  - Recall: [0.516, 0.672]
  - FPR: [0.092, 0.194]
  - AUROC: [0.747, 0.846]
- P1 Macro-F1: [0.676, 0.775]（mean 0.727）
  - Recall: [0.665, 0.793]
  - FPR: [0.209, 0.344]
  - AUROC: [0.747, 0.846]
- P2-10% Macro-F1: [0.726, 0.823]（mean 0.776）
  - Recall: [0.725, 0.846]
  - FPR: [0.171, 0.297]
  - AUROC: [0.747, 0.846]
- P2-20% Macro-F1: [0.762, 0.850]（mean 0.807）
  - Recall: [0.772, 0.882]
  - FPR: [0.153, 0.275]
  - AUROC: [0.747, 0.846]

## 9. 跨模型切片（表 C，P1 阈值）

| Model | N | Macro-F1 | Recall | FPR | MCC | AUROC | AUPRC | Hard-safe FPR | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen Flash | 64 | 0.703 | 0.688 | 0.281 | 0.406 | 0.816 | 0.838 | 0.133 | PASS |
| Qwen Plus | 64 | 0.763 | 0.875 | 0.344 | 0.544 | 0.838 | 0.853 | 0.133 | PASS |
| DeepSeek Flash | 64 | 0.733 | 0.656 | 0.188 | 0.475 | 0.802 | 0.790 | 0.100 | PASS |
| DeepSeek Pro | 64 | 0.827 | 0.906 | 0.250 | 0.664 | 0.905 | 0.897 | 0.033 | PASS |
| GLM Flash | 64 | 0.670 | 0.594 | 0.250 | 0.348 | 0.740 | 0.762 | 0.200 | PASS |
| Kimi | 64 | 0.656 | 0.656 | 0.344 | 0.312 | 0.673 | 0.658 | 0.267 | fail |

## 10. Hard-safe Control（表）

| Slice | N | FPR | safe Recall | P90 | P95 |
|---|---:|---:|---:|---:|---:|
| pooled hard-safe | 180 | 0.144 | 0.000 | - | - |
| hard_safe | 72 | 0.125 | 0.000 | - | - |
| anti_fraud | 54 | 0.056 | 0.000 | - | - |
| matched_safe | 36 | 0.222 | 0.000 | - | - |
| benign | 18 | 0.333 | 0.000 | - | - |

- pooled hard-safe test risk P90 = 0.413，P95 = 0.654。

## 11. 错误切片与敏感性

- zh: MF1 0.711 / en: MF1 0.738。

- truncated: MF1 0.695 / non-truncated: MF1 0.818。

- family-pair matched: MF1 0.726 / stratum-matched: MF1 0.733。

- 编码敏感性：qy 主视图约 50% 输入超过 512 token（head-tail 截断编码），y-only 视图无截断；tail-only 编码敏感性未另行运行，以 truncated/non-truncated 切片代替（见上）。

## 12. 成本与效率

- 累计 E6 成本（含 v1 ¥1.4153）：¥51.4713；剩余预算 ¥NA。

- 按阶段：{"pool_translation": 0.0045, "v2_probe": 0.0376, "v2_gen_anchor": 1.0176, "v2_judge": 27.2281, "v2_adjudication": 9.7751, "v2_gen_b0": 1.183, "v2_gen_control": 0.16, "v2_retry": 0.0013, "v2_gen_b1": 0.2844, "v2_gen_b2": 0.2733, "v2_gen_b3": 0.3341}。

- P2 为离线 Silver 审核模拟（研究数据已全量 Silver），审核成本为 3 Judge 调用/条；10%/20% 审核率下 audited 数见 metrics JSON。

## 13. 门控评估

- Pooled Core Pass（AUROC≥0.75, AUPRC≥0.75, Macro-F1≥0.70, Recall≥0.65, MCC≥0.40, HS-FPR≤0.15）：**False**。

- Pooled Strong Pass：**False**。

- 跨模型最低门（AUROC≥0.65, MF1≥0.60, Recall≥0.50, HS-FPR≤0.25）逐模型见 `student/gate_results.json`；未达标模型如实标记为 transfer-failure slice。

## 14. 结论与限制

- 结论按协议 §17.3 分级；50/50 均衡测试不能解释为真实 prevalence/PPV，报告不据此给出部署报警量。

- 限制：三 Judge 中 GLM Flash 输出风格差异导致 behavior κ 偏低（已如实报告）；qy 视图约 50% 输入超过 512 token，采用 head-tail 主编码并报告 truncated/non-truncated 切片（truncated MF1 0.695 / non-truncated 0.818）；M6（Kimi）账户曾欠费导致补充批暂停，最终以续费后全量补齐。

## 附录 A：审计文件清单

```text
protocol/model_registry_frozen.json
data/prompt_pool_manifest.jsonl + sha256
data/superfamily_split_audit.json
data/cross_experiment_leakage_audit.json
generations/generation_registry.jsonl + summary
budget/cost_ledger.jsonl + cost_summary.json
silver/judge_J1..J3_raw.jsonl + adjudicator_raw.jsonl + silver_consensus.jsonl + silver_quality_metrics_all.json
balanced/balanced_selection_manifest.jsonl + audit + metadata_shortcut_probe.json
student/predictions_{all,qonly,yonly,tail_qy}.jsonl + truncation audits + metrics_p0_p1_p2.json + threshold_selection.json + gate_results.json + test_open_log.json
```

---

*报告由脚本自动生成（scripts/e6v2_write_report.py），数值均直接读取审计 JSON，避免手抄错误。*
