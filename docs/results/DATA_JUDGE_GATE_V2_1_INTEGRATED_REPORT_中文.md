# FraudDistill Data & Judge Gate v2.1 整合总报告

## 1. 本轮定位

本轮不是 Full 六实验，也不是继续放大 `protocol_gate_v2`。按照用户提供的 `FraudDistill_PROTOCOL_GATE_V2_NEXT_STEP_中文_2026-07-24.md`，本轮目标是完成 `Data & Judge Integration Gate v2.1`：先把数据输入、裁判标签、split、manifest 和复现链路修正到可追踪状态，再决定是否进入更大规模实验。

本轮最核心的结论是：工程和数据管线已有实质修复，但官方 open guard 仍未实际运行，因此 Full 仍然是 NO-GO。

## 2. 运行与复现元数据

| 项目 | 值 |
|---|---|
| run_id | `data_judge_gate_v2_1` |
| freeze commit | `11014163f8fcfcd9c035282f22f26d674e87a8a1` |
| freeze dirty | `false` |
| generation bank | `archive/pre_high_standard_rerun_20260723_000310/exp6_multi_api/multi_cn_api_v1/generations/generations_success.jsonl` |
| generation bank SHA-256 | `428e8f8e5d99c5aec0a8ccd48789c59faa297428968f2f64ae64a526710f7c9d` |
| generation rows | 1,594 |
| selection policy | 显式路径冻结，不再按 mtime 自动选择 |
| 测试结果 | `pytest -q`，41 passed |

## 3. 关键修复

本轮完成的 P0/P1 修复如下：

| 修复项 | 状态 | 说明 |
|---|---|---|
| active runner 接入 qy_v3 | PASS | `load_all_rows()` 优先读取 `data/processed/qy_v3/judged_pairs_v3.jsonl` |
| 移除旧 `v2_hard_control_full` 主依赖 | PASS | fallback 改为 `fraudr1_all_categories_qy.jsonl`，主路径使用 qy_v3 |
| 显式冻结 generation 输入 | PASS | `input_freeze.json` 记录路径、SHA-256、commit 和 dirty 状态 |
| 去除 mtime 自动选输入 | PASS | `find_latest_exp6_generations()` 只读取 qy_v3 或 freeze lock |
| E6 标签不再使用 student 输出 | PASS | pair label 不读取 FraudDistill student 的 `pred_label` 或 `score` |
| source_prompt_id 分组切分 | PASS | `cross_split_source_prompt_id=0` |
| 数据与裁判 manifest/hash | PASS | 已生成 prompt、generation、pair label、split、cluster 与 artifact manifest |
| 官方 open guard 实际运行 | NO-GO | 当前仍是 student-free proxy guard，没有真实 Qwen3Guard/WildGuard/PolyGuard 输出 |

## 4. 数据规模与组成

| 指标 | 数值 |
|---|---:|
| q-y pairs | 1,594 |
| source prompts | 400 |
| abstain | 0 |
| consensus coverage | 1.0000 |
| safe pair labels | 1,147 |
| unsafe pair labels | 447 |

### 4.1 目标模型分布

| target_model | N |
|---|---:|
| deepseek-chat | 400 |
| glm-4-flash | 400 |
| qwen-plus | 400 |
| moonshot-v1-8k | 394 |

### 4.2 语言分布

| language | N | safe | unsafe |
|---|---:|---:|---:|
| en | 843 | 593 | 250 |
| zh | 751 | 554 | 197 |

## 5. 裁判一致性审计

本轮使用的是 student-free proxy guard panel，目的是先修正循环评估问题。它不能替代真实的 Qwen3Guard、WildGuard 或 PolyGuard。

| left | right | N | agreement |
|---|---|---:|---:|
| polyguard_proxy | qwen3guard_proxy | 1,594 | 0.6995 |
| polyguard_proxy | wildguard_proxy | 1,594 | 0.6995 |
| qwen3guard_proxy | wildguard_proxy | 1,594 | 1.0000 |

分析：

1. `qwen3guard_proxy` 与 `wildguard_proxy` 完全一致，说明这两个 proxy 仍过于同构，不能当作真实三裁判独立性证据。
2. `polyguard_proxy` 与另外两个 proxy 的一致性为 0.6995，提供了一定分歧，但仍是规则代理，不是模型裁判。
3. coverage=1.0 且 abstain=0 对工程调试有利，但从论文角度看反而说明当前 proxy 过于确定，下一轮真实 guard 应允许合理 abstain。

## 6. 目标 LLM 行为诊断

下表来自 `DATA_JUDGE_GATE_V2_1_target_llm_behavior_with_ci.csv`。这些数字可用于调试和趋势观察，但不能写成真实安全排名，因为标签仍非官方 open-guard 共识。

| target_model | N | FAR | PLR | CRR | ORR |
|---|---:|---:|---:|---:|---:|
| deepseek-chat | 400 | 0.3868 | 0.0741 | 0.6132 | 0.3376 |
| glm-4-flash | 400 | 0.4568 | 0.1605 | 0.5432 | 0.2866 |
| moonshot-v1-8k | 394 | 0.2887 | 0.1046 | 0.7113 | 0.3742 |
| qwen-plus | 400 | 0.7119 | 0.2469 | 0.2881 | 0.3758 |

解读：

1. 在当前 proxy 口径下，`qwen-plus` 的 FAR 和 PLR 最高，CRR 最低，是最值得进一步人工/真实 guard 复核的模型。
2. `moonshot-v1-8k` 的 FAR 最低、CRR 最高，但 ORR 也最高，可能表现为更保守。
3. `glm-4-flash` 的 PLR 较高，说明“拒答中泄漏可执行信息”是下一轮重点错误类型。
4. 这些结论都必须在真实 guard panel 下重新确认。

## 7. Key Gate 六实验复用结果

v2.1 脚本在冻结 qy_v3 数据上补跑了 key gate，日志显示 E1-E6 均完成：

```text
data loaded N=1594
E1 input ablation done
E2 prior-work comparison done
E3 agent/distillation ablation done
E4 unseen generalization done
E5 calibration done
E6 multi-api done
```

关键抽查：

| 项目 | 结果 |
|---|---:|
| E1 q+y Macro-F1 | 0.6215 |
| E1 y-only Macro-F1 | 0.6304 |
| E6 rows | 1,594 |
| E6 target models | 4 |
| E6 official guard available | false |

分析：

1. E1 在 qy_v3 key gate 上仍然是 `q+y < y-only`，说明“联合 q-y 输入自然优于 y-only”的主张仍未成立。
2. 但 qy_v3 已经把数据语义、source prompt 分组和标签来源修正到更可信状态，因此这些负结果比上一轮更有诊断价值。
3. E6 现在至少解决了循环造标签问题；下一步的核心不是再跑 student，而是替换真实 open guard。

## 8. GO/NO-GO 汇总

| Gate | Status | Evidence |
|---|---|---|
| G0 active runner uses qy_v3 | PASS | `load_all_rows` 优先读取 qy_v3 |
| G0 explicit generation input | PASS | `input_freeze.json` 存储显式路径与 SHA-256 |
| G1 student-free labels | PASS | `uses_student_for_label=false` |
| G1 official open guards | NO-GO | 本地未实际运行 Qwen3Guard/WildGuard/PolyGuard |
| Full experiments | NO-GO | 文档要求先完成 v2.1 gate，不启动 Full |

最终判定：Full 仍然 NO-GO。

## 9. 产物索引

提交到 GitHub 的摘要产物位于 `docs/results/`：

| 文件 | 用途 |
|---|---|
| `DATA_JUDGE_GATE_V2_1_MASTER_REPORT_中文.md` | 原 v2.1 简版主报告 |
| `DATA_JUDGE_GATE_V2_1_INTEGRATED_REPORT_中文.md` | 本整合总报告 |
| `DATA_JUDGE_GATE_V2_1_GO_NOGO.csv` | Gate 判定 |
| `DATA_JUDGE_GATE_V2_1_ARTIFACT_MANIFEST.tsv` | 本地完整产物 hash 清单 |
| `DATA_JUDGE_GATE_V2_1_guard_consensus_audit.json` | 裁判与数据审计 |
| `DATA_JUDGE_GATE_V2_1_guard_model_lock.yaml` | guard 配置锁 |
| `DATA_JUDGE_GATE_V2_1_guard_pairwise_agreement.csv` | 裁判一致性 |
| `DATA_JUDGE_GATE_V2_1_guard_language_audit.csv` | 语言标签分布 |
| `DATA_JUDGE_GATE_V2_1_target_llm_behavior_with_ci.csv` | 目标模型行为诊断 |
| `DATA_JUDGE_GATE_V2_1_DATASET_REVISION_LOCK.yaml` | 数据版本锁 |
| `DATA_JUDGE_GATE_V2_1_DATA_FILTER_RULES.yaml` | 数据过滤和禁止规则 |
| `DATA_JUDGE_GATE_V2_1_DATA_LICENSE_MANIFEST.yaml` | 数据许可摘要 |
| `DATA_JUDGE_GATE_V2_1_REDISTRIBUTION_POLICY.md` | 公开分发策略 |

本地完整数据产物位于：

| 路径 | 说明 |
|---|---|
| `data/processed/qy_v3/prompt_assets_v3_manifest.jsonl` | prompt 资产 manifest |
| `data/processed/qy_v3/generations_v3.jsonl` | 冻结 generation |
| `data/processed/qy_v3/generations_v3_manifest.jsonl` | generation manifest |
| `data/processed/qy_v3/judged_pairs_v3.jsonl` | q-y pair 标签数据 |
| `data/processed/qy_v3/pair_label_manifest.jsonl` | pair label manifest |
| `data/processed/qy_v3/split_manifest.jsonl` | source_prompt_id group split |
| `data/processed/qy_v3/semantic_cluster_manifest.jsonl` | semantic/source prompt cluster |

这些本地数据受 `.gitignore` 保护，不提交到 GitHub。

## 10. 后续执行建议

下一轮优先级：

1. 实际接入 Qwen3Guard、WildGuard、PolyGuard，并在公开 gold 数据上先做 guard audit。
2. 用真实 guard panel 重标当前 1,594 条 generation，生成 coverage、abstain、language audit 和 judge sensitivity。
3. 重新计算 E6 FAR/PLR/CRR/ORR，并检查目标模型排序是否稳定。
4. 构建无显式标签词的 natural context-critical set，再重跑 E1 key gate。
5. 若 E1/E4/E6 仍不通过，不进入 Full；继续定位数据语义、英文 recall 和 OR-Bench hard-safe FPR。

本轮可以写入论文开发过程或实验设计复盘的内容：

- prompt risk 与 pair fraud label 必须分离；
- student 不能参与 ground truth 构造；
- generation bank 必须显式冻结并记录 SHA-256；
- 同一 source prompt 的多模型回答必须同 split；
- proxy guard 只能用于工程门控，不能替代真实 open guard 共识。
