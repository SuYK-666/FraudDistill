# FraudDistill 六实验Protocol Gate v2总报告

## 运行说明

本报告汇总同一轮 `protocol_gate_v2` 的七组输出：数据审计和实验 1-6。每组原始预测、指标、配置、模型、审计和分实验报告均保留在原运行目录；本报告只做汇总，不重采样、不覆盖原始结果。产物清单 `outputs\PROTOCOL_GATE_V2_ARTIFACT_MANIFEST.tsv` 记录每个文件的路径、字节数和 SHA-256。

## 总结结论

- E1：q+y Macro-F1=0.8544，y-only=0.8741，增益=-0.0197。
- E2：输出 7 行 proxy/baseline 审计；官方 guard baseline 仍需单独接入。
- E3：当前定位为 sample reweighting proxy；Student-Gold Macro-F1=0.8722，Full sample-reweighting proxy Macro-F1=0.8733。
- E4：最弱 held-out=phishing_scams，Macro-F1=0.8468；language holdout 已排除 procedural rows。
- E6：覆盖 4 个已有目标模型 generation；pair label 为 proxy guard consensus，独立 guard 可用性=False。
- Gate：按用户新文档，本轮仍是 NO-GO Full；目标是完成 P0/P1 修复和门控产物，而不是启动完整大样本。

## 原始产物保留

完整产物清单随本报告生成并复制到 `docs/results/`。本地 `data/`、`outputs/`、`archive/` 按 `.gitignore` 不提交到 GitHub。
## 数据标签审计

运行目录：`outputs\audit_label_integrity\protocol_gate_v2`

# 标签与数据审计

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: 5b5e41ae098326ec96f5dace8c206103df41d374
tag_or_describe: paper-six-exp-ccfa-v1-5-g5b5e41a
git_dirty_at_run: False
run_id: protocol_gate_v2
run_date: 2026-07-24T09:11:18.815257+00:00
python_version: 3.12.2
config_path: pending
config_sha256: pending
split_hash: 
```

## 结论与分析
本审计分开记录 official_gold、weak_reference 和 teacher_signal，禁止将 teacher signal 写入 gold。

Aegis2.0、Do-Not-Answer 当前均含正负类；OR-Bench hard-safe 纯 safe 子集不会单独计算 Recall_unsafe。

duplicate_prompt_hashes 统一定义为落入重复 prompt group 的额外样本行数，即 sum(count-1)。cross-split 项只在使用 split 后的实验内计算。

## 表格
| dataset | n_total | n_safe | n_unsafe | reference_types | unknown_count | duplicate_prompt_hashes | can_compute_binary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Aegis2.0 | 1781 | 1024 | 757 | official_gold | 0 | 25 | True |
| Do-Not-Answer | 1488 | 767 | 721 | official_gold | 0 | 719 | True |
| Fraud-R1 | 1431 | 716 | 715 | weak_reference | 0 | 618 | True |
| OR-Bench | 300 | 300 | 0 | weak_reference | 0 | 0 | False |

## 实验1：输入消融

运行目录：`outputs\exp1_input_ablation\protocol_gate_v2`

# 实验1：q/y/q+y 输入边界消融

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: 5b5e41ae098326ec96f5dace8c206103df41d374
tag_or_describe: paper-six-exp-ccfa-v1-5-g5b5e41a
git_dirty_at_run: False
run_id: protocol_gate_v2
run_date: 2026-07-24T09:14:04.384467+00:00
python_version: 3.12.2
config_path: outputs\exp1_input_ablation\protocol_gate_v2\config_resolved.yaml
config_sha256: 9041de4b156862ef1aa000ec788aa55636cfd69c1474110d6e85aecd71254880
split_hash: b54bf44921292c813433655c703b3ba9b9ba8c8529cb9d3b6f5f70b5d3cef7b4
```

## 结论与分析
测试集 N=852。q+y Macro-F1=0.8544，最佳单侧 Macro-F1=0.8741，差值=-0.0197。

测试集类别组成：safe=452，unsafe=400。

q+y Recall_unsafe=0.8875；y_only Recall_unsafe=0.8800。

q_only/y_only/q+y 本轮改为三套独立输入模型；不再执行 q+y 与 y-only 的混合，也不再用 y-only 指标作为 q+y 的选择约束。阈值只在各自 dev split 上选择。

新增 Track B：Context-Critical paired benchmark，N=144。该轨道为 procedural weak benchmark，用于验证缺少 q 时 y-only 的信息缺失，不写成 official gold。

## 表格
## Track A Naturalistic
| Input | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_only | 0.635 | 0.6282 | 0.545 | 0.5837 | 0.6293 | 0.2854 | 0.6794 |
| y_only | 0.8744 | 0.8564 | 0.88 | 0.8681 | 0.8741 | 0.1305 | 0.9228 |
| q+y | 0.8545 | 0.818 | 0.8875 | 0.8513 | 0.8544 | 0.1748 | 0.8896 |

## 操作点补充
| Setting | Input | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0.5 | q_only | 0.5 | 0.6373 | 0.6096 | 0.6325 | 0.6209 | 0.6366 | 0.3584 | 0.6794 |
| fixed_0.5 | y_only | 0.5 | 0.8862 | 0.8111 | 0.9875 | 0.8906 | 0.886 | 0.2035 | 0.9228 |
| fixed_0.5 | q+y | 0.5 | 0.8462 | 0.7785 | 0.94 | 0.8516 | 0.846 | 0.2367 | 0.8896 |
| dev_optimal | q_only | 0.528436 | 0.635 | 0.6282 | 0.545 | 0.5837 | 0.6293 | 0.2854 | 0.6794 |
| dev_optimal | y_only | 0.643444 | 0.8744 | 0.8564 | 0.88 | 0.8681 | 0.8741 | 0.1305 | 0.9228 |
| dev_optimal | q+y | 0.560954 | 0.8545 | 0.818 | 0.8875 | 0.8513 | 0.8544 | 0.1748 | 0.8896 |
| matched_fpr_to_y_only_dev | q+y | 0.59284 | 0.8486 | 0.843 | 0.8325 | 0.8377 | 0.8479 | 0.1372 | 0.8896 |
| matched_recall_to_y_only_dev | q+y | 0.562237 | 0.8556 | 0.8199 | 0.8875 | 0.8523 | 0.8556 | 0.1726 | 0.8896 |
| matched_fpr_to_qy_dev | y_only | 0.60094 | 0.8862 | 0.8344 | 0.945 | 0.8863 | 0.8862 | 0.1659 | 0.9228 |
| matched_recall_to_qy_dev | y_only | 0.637395 | 0.8756 | 0.8517 | 0.89 | 0.8704 | 0.8754 | 0.1372 | 0.9228 |

## Track B Context-Critical
| Track | Input | Threshold | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Context-Critical | y_only | 0.384046 | 144 | 0.5 | 0.0 | 0.0 | 0.0 | 0.3333 | 0.0 | 0.5 |
| Context-Critical | q+y | 0.42648 | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |

## Pair consistency
| Input | Pair groups | Pair consistency accuracy |
| --- | --- | --- |
| q+y | 72 | 1.0 |
| y_only | 72 | 0.0 |

## 实验2：现有工作对比

运行目录：`outputs\exp2_prior_work_comparison\protocol_gate_v2`

# 实验2：现有工作对比审计与 proxy 重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: 5b5e41ae098326ec96f5dace8c206103df41d374
tag_or_describe: paper-six-exp-ccfa-v1-5-g5b5e41a
git_dirty_at_run: False
run_id: protocol_gate_v2
run_date: 2026-07-24T09:16:22.782971+00:00
python_version: 3.12.2
config_path: outputs\exp2_prior_work_comparison\protocol_gate_v2\config_resolved.yaml
config_sha256: 40f648ea874a8bd7d3158b18ccce8b2a9b756789d72b6cc317a83c7e5600c31d
split_hash: 
```

## 结论与分析
本轮不再把规则近似称为 official baseline。当前仓库缺少官方 evaluator/checkpoint，因此 E2 正式论文主张仍阻塞。

可用输出包括每块 label audit、coverage、proxy prediction 和 CI 文件；它们用于调试 FraudDistill，不用于论文中声称优于现有工作。

## 表格
| Dataset | Method | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | Original metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fraud-R1 | Reproducible conservative proxy | 325 | 0.44 | 0.4238 | 0.5933 | 0.4944 | 0.4334 | 0.6914 | 0.5559 | official baseline unavailable |
| Fraud-R1 | FraudDistill student proxy | 325 | 0.9477 | 0.9926 | 0.8933 | 0.9404 | 0.9469 | 0.0057 | 0.9956 | official baseline unavailable |
| Aegis2.0 | Reproducible conservative proxy | 360 | 0.5944 | 0.8889 | 0.0523 | 0.0988 | 0.4186 | 0.0048 | 0.4411 | official baseline unavailable |
| Aegis2.0 | FraudDistill student proxy | 360 | 0.8417 | 0.7637 | 0.9085 | 0.8299 | 0.8409 | 0.2077 | 0.8327 | official baseline unavailable |
| Do-Not-Answer | Reproducible conservative proxy | 301 | 0.4718 | 0.0 | 0.0 | 0.0 | 0.3205 | 0.0207 | 0.5942 | official baseline unavailable |
| Do-Not-Answer | FraudDistill student proxy | 301 | 0.8239 | 0.8705 | 0.7756 | 0.8203 | 0.8239 | 0.1241 | 0.9225 | official baseline unavailable |
| OR-Bench hard-safe | FraudDistill proxy | 60 |  |  |  |  |  | 0.25 |  | pure-safe FPR only |

## 实验3：Agent 与样本重加权

运行目录：`outputs\exp3_agent_distillation_ablation\protocol_gate_v2`

# 实验3：Agent 与蒸馏消融重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: 5b5e41ae098326ec96f5dace8c206103df41d374
tag_or_describe: paper-six-exp-ccfa-v1-5-g5b5e41a
git_dirty_at_run: False
run_id: protocol_gate_v2
run_date: 2026-07-24T09:19:26.595108+00:00
python_version: 3.12.2
config_path: outputs\exp3_agent_distillation_ablation\protocol_gate_v2\config_resolved.yaml
config_sha256: 8176af2fc1c613162ca44dd3b32363e178897924e68696a83ce520be40a16de1
split_hash: 875a03b7ea810d80b4d18c90904f12b72ceb66cea5289d3f1bc878a0e966cb82
```

## 结论与分析
修复点：teacher label 不再覆盖 gold，而是以附加 token/score/type/rank/context 信号进入训练；每个变体使用相同 train/dev/test manifest，并生成不同模型文件。

Agent 表拆成 nested ablation、leave-one-component-out 和组件压力指标三张表；全局梯度与专属边界下降需要同时成立才可写成不可替代性结论。

新增 Component Stress Benchmark：N=720，标签类型为 procedural weak stress，不写成 official gold；用于观察模块不可替代性的大效应量。

Teacher alignment train+dev: {'n': 4026, 'agreement': 0.5116741182314953, 'teacher_label_counts': Counter({'safe': 3094, 'unsafe': 932}), 'gold_counts': Counter({'safe': 2264, 'unsafe': 1762}), 'confusion_matrix': {'tp': 364, 'fp': 568, 'fn': 1398, 'tn': 1696}, 'unsafe_recall': 0.20658342792281498, 'unsafe_precision': 0.3905579399141631, 'safe_fpr': 0.2508833922261484}

## 表格
## Broad Student
| Variant | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Student-Gold | 0.5 | 0.8727 | 0.8165 | 0.9188 | 0.8646 | 0.8722 | 0.1639 | 0.9247 |
| Gold + teacher-confidence reweight | 0.499998 | 0.8747 | 0.8225 | 0.9142 | 0.8659 | 0.8742 | 0.1565 | 0.926 |
| Gold + soft-score reweight | 0.479197 | 0.8717 | 0.8135 | 0.9211 | 0.864 | 0.8713 | 0.1676 | 0.9271 |
| Gold + score/type reweight | 0.474278 | 0.8727 | 0.8139 | 0.9234 | 0.8652 | 0.8723 | 0.1676 | 0.9275 |
| Gold + score/type/rank reweight | 0.472522 | 0.8727 | 0.8152 | 0.9211 | 0.8649 | 0.8723 | 0.1657 | 0.9282 |
| Full sample-reweighting proxy | 0.464523 | 0.8737 | 0.8156 | 0.9234 | 0.8662 | 0.8733 | 0.1657 | 0.9286 |

## Broad Agent nested
| Variant | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single Judge | 0.4928 | 0.2857 | 0.0974 | 0.1453 | 0.3924 | 0.1934 | 0.4272 |
| Fraud only | 0.5544 | 0.4933 | 0.2575 | 0.3384 | 0.5013 | 0.2099 | 0.4832 |
| Fraud + Refusal | 0.5483 | 0.4789 | 0.2367 | 0.3168 | 0.4897 | 0.2044 | 0.4739 |
| Fraud + Refusal + Relevance | 0.6057 | 0.7374 | 0.1694 | 0.2755 | 0.5023 | 0.0479 | 0.5718 |
| Full fixed | 0.6088 | 0.7604 | 0.1694 | 0.277 | 0.5045 | 0.0424 | 0.5715 |
| Full learned | 0.6129 | 0.8 | 0.1671 | 0.2764 | 0.5061 | 0.0331 | 0.5794 |
| Full learned calibrated | 0.6129 | 0.8 | 0.1671 | 0.2764 | 0.5061 | 0.0331 | 0.5799 |

## Broad Leave-one-out
| Variant | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | 0.6129 | 0.8 | 0.1671 | 0.2764 | 0.5061 | 0.0331 | 0.5799 |
| Full - Fraud | 0.5595 | 0.5714 | 0.0186 | 0.036 | 0.3753 | 0.011 | 0.4971 |
| Full - Refusal | 0.6181 | 0.798 | 0.1833 | 0.2981 | 0.5179 | 0.0368 | 0.585 |
| Full - Relevance | 0.5852 | 0.5894 | 0.2065 | 0.3058 | 0.505 | 0.1142 | 0.5386 |
| Full - learned Arbiter | 0.6088 | 0.7604 | 0.1694 | 0.277 | 0.5045 | 0.0424 | 0.5708 |

## Broad Component pressure
| Subset | N | Full Recall_unsafe | Full - Fraud Recall_unsafe | Full - Refusal Recall_unsafe | Full - Relevance Recall_unsafe | Full - learned Arbiter Recall_unsafe | Full FPR | Full - Fraud FPR | Full - Refusal FPR | Full - Relevance FPR | Full - learned Arbiter FPR | Full Macro-F1 | Full - Fraud Macro-F1 | Full - Refusal Macro-F1 | Full - Relevance Macro-F1 | Full - learned Arbiter Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable fraud | 11 | 0.5 | 0.5 | 0.5 | 0.75 | 0.5 |  |  |  |  |  |  |  |  |  |  |
| partial leakage | 13 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |
| hard safe / anti-fraud | 218 |  |  |  |  |  | 0.0 | 0.0 | 0.0 | 0.2839 | 0.0323 |  |  |  |  |  |
| agent conflict | 126 |  |  |  |  |  |  |  |  |  |  | 0.4474 | 0.4474 | 0.4474 | 0.5697 | 0.4703 |

## Stress Agent nested
| Variant | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single Judge | 0.5 | 0.3333 | 0.3636 | 0.4444 | 0.4 | 0.325 | 0.7778 | 0.4394 |
| Fraud only | 0.6199 | 0.5417 | 0.5366 | 0.6111 | 0.5714 | 0.5394 | 0.5278 | 0.4746 |
| Fraud + Refusal | 0.8214 | 0.6389 | 0.7273 | 0.4444 | 0.5517 | 0.6247 | 0.1667 | 0.5686 |
| Fraud + Refusal + Relevance | 0.2439 | 0.7361 | 0.6545 | 1.0 | 0.7912 | 0.7164 | 0.5278 | 0.6851 |
| Full fixed | 0.2289 | 0.7361 | 0.6545 | 1.0 | 0.7912 | 0.7164 | 0.5278 | 0.6851 |
| Full learned | 0.1639 | 0.875 | 0.8 | 1.0 | 0.8889 | 0.873 | 0.25 | 0.7555 |
| Full learned calibrated | 0.217676 | 0.875 | 0.8 | 1.0 | 0.8889 | 0.873 | 0.25 | 0.7555 |

## Stress Leave-one-out
| Variant | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full | 0.217676 | 0.875 | 0.8 | 1.0 | 0.8889 | 0.873 | 0.25 | 0.7555 |
| Full - Fraud | 0.1156 | 0.875 | 0.8 | 1.0 | 0.8889 | 0.873 | 0.25 | 0.8384 |
| Full - Refusal | 0.217676 | 0.875 | 0.8 | 1.0 | 0.8889 | 0.873 | 0.25 | 0.7232 |
| Full - Relevance | 0.5 | 0.7431 | 0.7465 | 0.7361 | 0.7413 | 0.743 | 0.25 | 0.6928 |
| Full - learned Arbiter | 0.272276 | 0.7361 | 0.6545 | 1.0 | 0.7912 | 0.7164 | 0.5278 | 0.6851 |

## Stress Student
| Variant | Threshold | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stress Gold | 0.49728 | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| + hard label | 0.497359 | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| + soft | 0.497409 | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| + type | 0.497444 | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| + rank | 0.49748 | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Full multi-task context | 0.497532 | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |

## 实验4：未见泛化

运行目录：`outputs\exp4_unseen\protocol_gate_v2`

# 实验4：unseen 泛化重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: 5b5e41ae098326ec96f5dace8c206103df41d374
tag_or_describe: paper-six-exp-ccfa-v1-5-g5b5e41a
git_dirty_at_run: False
run_id: protocol_gate_v2
run_date: 2026-07-24T09:22:05.426435+00:00
python_version: 3.12.2
config_path: outputs\exp4_unseen\protocol_gate_v2\config_resolved.yaml
config_sha256: dac387a8ba82969d8d742830d1c0176b8f994620258e7969000538c172e68532
split_hash: 
```

## 结论与分析
本轮改为 leave-one-category-out，并将 OR-Bench pure-safe 子集只按 FPR/specificity 解读。

当前 Fraud-R1 hard-control 公开数据仍不足以覆盖每个 held-out 类 300 unsafe + 300 safe 的论文强门槛；报告保留规模限制。

新增五类 procedural LOCO、source holdout、language holdout 表；这些是扩展弱评测，用来观察趋势，不能替代官方五类 gold benchmark。

## 表格
## 原始 Fraud-R1/OR-Bench
| Setting | Held-out | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leave-one-category-out | fake_job_postings | 395 | 0.9899 | 0.9744 | 1.0 | 0.987 | 0.9894 | 0.0165 | 0.9975 |
| Leave-one-category-out | impersonation | 402 | 0.898 | 1.0 | 0.763 | 0.8656 | 0.8917 | 0.0 | 0.9915 |
| Leave-one-category-out | phishing_scams | 634 | 0.858 | 0.8606 | 0.9179 | 0.8883 | 0.8468 | 0.2377 | 0.9735 |
| Source hard-safe | OR-Bench hard-safe | 300 | 0.6167 |  |  |  |  | 0.3833 |  |

## Procedural five-category LOCO
| Setting | Held-out | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Procedural five-category LOCO | fake_job_postings | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Procedural five-category LOCO | fraudulent_services | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Procedural five-category LOCO | impersonation | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Procedural five-category LOCO | online_relationships | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Procedural five-category LOCO | phishing_scams | 144 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |

## Source holdout
| Setting | Held-out | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Source holdout | Aegis2.0 | 1781 | 0.5272 | 0.4605 | 0.6539 | 0.5404 | 0.5268 | 0.5664 | 0.4902 |
| Source holdout | Do-Not-Answer | 1488 | 0.5181 | 0.5526 | 0.0291 | 0.0553 | 0.366 | 0.0222 | 0.5549 |
| Source holdout | Fraud-R1 | 1431 | 0.515 | 0.5755 | 0.1119 | 0.1874 | 0.4209 | 0.0824 | 0.5445 |

## Language holdout
| Setting | Held-out | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language holdout | en | 4287 | 0.5696 | 0.25 | 0.0005 | 0.0011 | 0.3634 | 0.0012 | 0.3843 |
| Language holdout | zh | 713 | 0.6746 | 0.798 | 0.4514 | 0.5766 | 0.6562 | 0.1102 | 0.7941 |

## 实验5：概率校准

运行目录：`outputs\exp5_calibration\protocol_gate_v2`

# 实验5：阈值与概率校准重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: 5b5e41ae098326ec96f5dace8c206103df41d374
tag_or_describe: paper-six-exp-ccfa-v1-5-g5b5e41a
git_dirty_at_run: False
run_id: protocol_gate_v2
run_date: 2026-07-24T09:24:21.450509+00:00
python_version: 3.12.2
config_path: outputs\exp5_calibration\protocol_gate_v2\config_resolved.yaml
config_sha256: 5f4a9b0400a83a6df978396243635be857fbfd67fda67e304d6b008c49c1cc6c
split_hash: 454fb1f45d75bcc6f34abccdedb68cd61c024fd325667cfc89aacad4444df920
```

## 结论与分析
修复点：本轮校准对象切换为 Full context auxiliary distillation proxy，不再使用上一轮 AUPRC 较低的 raw q+y 模型。

FPR cap 表述为 dev-UCB 约束，并单独报告 observed test FPR 与 test UCB；不再把 dev 约束写成 test 保证。

主方法使用 Platt；Isotonic 因小样本/离散分数容易阈值退化，移出主表。

## 表格
## Calibration
| Method | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | ECE | Brier | Observed test FPR UCB95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default raw 0.5 | 0.5 | 0.8278 | 0.7651 | 0.8916 | 0.8235 | 0.8277 | 0.2244 | 0.8674 | 0.1229 | 0.1386 | 0.2584 |
| Platt default 0.5 | 0.5 | 0.8217 | 0.774 | 0.8537 | 0.8119 | 0.8212 | 0.2044 | 0.8674 | 0.0422 | 0.1221 | 0.2374 |
| Platt dev-UCB FPR<=0.01 | 0.956725 | 0.6276 | 0.9571 | 0.1816 | 0.3052 | 0.5254 | 0.0067 | 0.8674 | 0.0422 | 0.1221 | 0.0166 |
| Platt dev-UCB FPR<=0.05 | 0.84418 | 0.7338 | 0.8794 | 0.4743 | 0.6162 | 0.7062 | 0.0533 | 0.8674 | 0.0422 | 0.1221 | 0.0736 |
| Platt dev-UCB FPR<=0.10 | 0.604914 | 0.8144 | 0.8006 | 0.7832 | 0.7918 | 0.8122 | 0.16 | 0.8674 | 0.0422 | 0.1221 | 0.1904 |

## Cascade operating points
| Operating point | Stage A | Stage B | Target FPR cap | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | Observed test FPR UCB95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| High recall | full-context high-recall score | dev-UCB false-positive suppressor | 0.1 | 0.604914 | 0.8144 | 0.8006 | 0.7832 | 0.7918 | 0.8122 | 0.16 | 0.8674 | 0.1904 |
| Deployment | full-context high-recall score | dev-UCB false-positive suppressor | 0.05 | 0.84418 | 0.7338 | 0.8794 | 0.4743 | 0.6162 | 0.7062 | 0.0533 | 0.8674 | 0.0736 |
| Strict | full-context high-recall score | dev-UCB false-positive suppressor | 0.01 | 0.956725 | 0.6276 | 0.9571 | 0.1816 | 0.3052 | 0.5254 | 0.0067 | 0.8674 | 0.0166 |

## 实验6：多 API

运行目录：`outputs\exp6_multi_api\protocol_gate_v2`

# 实验6：多 API 目标模型重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: 07a3afd304d668b306421f6eb74759973864e4d9
tag_or_describe: paper-six-exp-ccfa-v1-6-g07a3afd
git_dirty_at_run: False
run_id: protocol_gate_v2
run_date: 2026-07-24T09:25:50.531625+00:00
python_version: 3.12.2
config_path: outputs\exp6_multi_api\protocol_gate_v2\config_resolved.yaml
config_sha256: 7f0603b0ee6d47ca874c0da919177ff8c3bac81e85a3c34e98b510df67fb17f5
split_hash: 
```

## 结论与分析
修复点：新生成回复不再继承旧 gold_label；旧标签只作为 prompt_risk_label/should_refuse 辅助字段，pair_fraud_label 由回答拒答、实质帮助和部分泄漏规则重新计算。

student_vs_pair_silver 是 gate 阶段的确定性 guard proxy，不是最终 Qwen3Guard/WildGuard/HarmBench 共识；因此当前仍不能写成目标 LLM 安全排名。

Leave-One-Model-Out 改为使用 pair_silver_label，并保留该表仅作 prompt/model 泛化诊断。

## 表格
## Student vs pair silver
| Target LLM | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | Pair label provenance | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 0.79 | 0.4038 | 0.6562 | 0.5 | 0.6835 | 0.1845 | 0.4063 | deterministic_guard_proxy | 1.0 |
| glm-4-flash | 400 | 0.6775 | 0.2374 | 0.5893 | 0.3385 | 0.5626 | 0.3081 | 0.278 | deterministic_guard_proxy | 1.0 |
| moonshot-v1-8k | 394 | 0.703 | 0.2246 | 0.7561 | 0.3464 | 0.5771 | 0.3031 | 0.2485 | deterministic_guard_proxy | 1.0 |
| qwen-plus | 400 | 0.75 | 0.5106 | 0.4706 | 0.4898 | 0.6621 | 0.1544 | 0.4337 | deterministic_guard_proxy | 1.0 |

## Behavior
| Target LLM | N | FAR/UAR | RFR | CRR | Partial leakage rate | ORR | Avg latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 0.2058 | 0.2593 | 0.2263 | 0.0535 | 0.3376 | 8.1258 |
| glm-4-flash | 400 | 0.214 | 0.284 | 0.3333 | 0.07 | 0.2866 | 16.349 |
| moonshot-v1-8k | 394 | 0.1339 | 0.1799 | 0.3682 | 0.046 | 0.3742 | 9.9152 |
| qwen-plus | 400 | 0.4074 | 0.5761 | 0.2016 | 0.1687 | 0.3758 | 12.5722 |

## Leave-one-model-out
| Held-out model | N | threshold_dev_n | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 239 | 0.8675 | 0.6038 | 0.5 | 0.547 | 0.7347 | 0.0625 | 0.6021 |
| glm-4-flash | 400 | 239 | 0.8775 | 0.5455 | 0.75 | 0.6316 | 0.7791 | 0.1017 | 0.7528 |
| moonshot-v1-8k | 394 | 241 | 0.9391 | 0.7179 | 0.6829 | 0.7 | 0.8331 | 0.0312 | 0.6934 |
| qwen-plus | 400 | 239 | 0.7925 | 0.8065 | 0.2451 | 0.3759 | 0.6258 | 0.0201 | 0.522 |

## Coverage
| Target LLM | N | safe | unsafe | zh | en | available_in_current_bank |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 0 | 0 | 188 | 212 | True |
| glm-4-flash | 400 | 0 | 0 | 188 | 212 | True |
| moonshot-v1-8k | 394 | 0 | 0 | 187 | 207 | True |
| qwen-plus | 400 | 0 | 0 | 188 | 212 | True |

生成时间：2026-07-24T09:26:03.016125+00:00
