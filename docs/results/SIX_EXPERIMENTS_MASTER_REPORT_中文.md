# FraudDistill 六实验全量总报告

## 运行说明

本报告汇总同一轮 `high_standard_full` 的六组正式输出。每组的原始预测、指标、模型、审计、配置、失败记录和逐实验报告均保留在原运行目录；本总报告不重采样、不覆盖任何原始结果。附带的产物清单记录每个文件的路径、字节数与 SHA-256，用于后续完整性核验。

## 总结结论

- E1：q+y Macro-F1=0.8418，y-only=0.8117，增益=0.0301；AUPRC=0.8238，阈值仅在 dev 上选择。
- E2：共输出 7 行 proxy/coverage 结果；仍需官方 evaluator/checkpoint 才能作为论文主表。
- E3：Student 从 Student-Gold Macro-F1=0.801 到 Full + context auxiliary Macro-F1=0.8515；新增 nested、leave-one-out、组件压力三类表。
- E4：最弱 held-out 项为 phishing_scams，Macro-F1=0.8468，保留类别覆盖限制。
- E6：覆盖 4 个已有目标模型 generations；行为指标 FAR/RFR/CRR/ORR 使用独立字段计算，仍标注为 detector-dependent。

## 原始产物保留

完整产物清单：`outputs/SIX_EXPERIMENTS_ARTIFACT_MANIFEST.tsv`。清单覆盖本轮每一个配置、预测、表格、模型、审计与报告文件。原始数据仍位于本地 `data/`，按照 `.gitignore` 不提交到 GitHub。
## 数据标签审计

运行目录：`outputs\audit_label_integrity\high_standard_full`

﻿# 标签与数据审计

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: f6ae0152b4f3eda2fc47c1edffc3351b6d1a98ee
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T04:08:42.421360+00:00
python_version: 3.12.2
config_path: pending
config_sha256: pending
```

## 结论与分析
本审计分开记录 official_gold、weak_reference 和 teacher_signal，禁止将 teacher signal 写入 gold。

Aegis2.0、Do-Not-Answer 当前均含正负类；OR-Bench hard-safe 纯 safe 子集不会单独计算 Recall_unsafe。

duplicate_prompt_hashes 统一定义为落入重复 prompt group 的额外样本行数，即 sum(count-1)。cross-split 项只在使用 split 后的实验内计算。

## 表格
| dataset | n_total | n_safe | n_unsafe | reference_types | unknown_count | duplicate_prompt_hashes | can_compute_binary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Aegis2.0 | 29828 | 25648 | 4180 | official_gold | 0 | 4574 | True |
| Do-Not-Answer | 5632 | 4000 | 1632 | official_gold | 0 | 4694 | True |
| Fraud-R1 | 1658 | 871 | 787 | weak_reference | 0 | 768 | True |
| OR-Bench | 300 | 300 | 0 | weak_reference | 0 | 0 | False |

## 实验1：输入消融

运行目录：`outputs\exp1_input_ablation\high_standard_full`

﻿# 实验1：q/y/q+y 输入边界消融

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: f6ae0152b4f3eda2fc47c1edffc3351b6d1a98ee
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T04:14:13.318130+00:00
python_version: 3.12.2
config_path: outputs\exp1_input_ablation\high_standard_full\config_resolved.yaml
config_sha256: c4d3bdcc977d19b5e3c781e91af3deab0894226a986de7e7b30178b87a9975fb
```

## 结论与分析
测试集 N=7393。q+y Macro-F1=0.8418，最佳单侧 Macro-F1=0.8117，差值=0.0301。

测试集类别组成：safe=6098，unsafe=1295。

q+y Recall_unsafe=0.8772；y_only Recall_unsafe=0.7552。

q_only/y_only/q+y 共享 Pair-TFIDF 双通道架构，只改变输入分支 mask；阈值只在 dev 选择。补充表同时报告固定0.5、各自dev-optimal、matched-FPR、matched-Recall四类操作点，降低“只靠阈值”的质疑。

## 表格
## 主表
| Input | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_only | 0.7122 | 0.2891 | 0.4409 | 0.3492 | 0.5822 | 0.2302 | 0.2856 |
| y_only | 0.8839 | 0.6438 | 0.7552 | 0.6951 | 0.8117 | 0.0887 | 0.7402 |
| q+y | 0.8968 | 0.6529 | 0.8772 | 0.7486 | 0.8418 | 0.099 | 0.8238 |

## 操作点补充
| Setting | Input | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0.5 | y_only | 0.5 | 0.8577 | 0.8347 | 0.234 | 0.3655 | 0.6427 | 0.0098 | 0.7402 |
| fixed_0.5 | q+y | 0.5 | 0.9033 | 0.6792 | 0.8486 | 0.7545 | 0.8472 | 0.0851 | 0.8238 |
| dev_optimal | y_only | 0.330482 | 0.8839 | 0.6438 | 0.7552 | 0.6951 | 0.8117 | 0.0887 | 0.7402 |
| dev_optimal | q+y | 0.673737 | 0.9044 | 0.8069 | 0.5969 | 0.6862 | 0.8149 | 0.0303 | 0.8238 |
| matched_fpr_to_y_only_dev | q+y | 0.511968 | 0.9048 | 0.6869 | 0.8386 | 0.7552 | 0.8481 | 0.0812 | 0.8238 |
| matched_recall_to_y_only_dev | q+y | 0.600962 | 0.9119 | 0.7588 | 0.729 | 0.7436 | 0.8452 | 0.0492 | 0.8238 |
| matched_fpr_to_qy_dev | y_only | 0.420522 | 0.8812 | 0.761 | 0.4695 | 0.5807 | 0.7558 | 0.0313 | 0.7402 |
| matched_recall_to_qy_dev | y_only | 0.373369 | 0.8915 | 0.7127 | 0.6378 | 0.6732 | 0.8041 | 0.0546 | 0.7402 |

## 实验2：现有工作对比

运行目录：`outputs\exp2_prior_work_comparison\high_standard_full`

﻿# 实验2：现有工作对比审计与 proxy 重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: f6ae0152b4f3eda2fc47c1edffc3351b6d1a98ee
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T04:14:49.633857+00:00
python_version: 3.12.2
config_path: outputs\exp2_prior_work_comparison\high_standard_full\config_resolved.yaml
config_sha256: 080d3b989a1ed0d675d12347c18e2a98d5da6b79e230344faad49629ad4124a0
```

## 结论与分析
本轮不再把规则近似称为 official baseline。当前仓库缺少官方 evaluator/checkpoint，因此 E2 正式论文主张仍阻塞。

可用输出包括每块 label audit、coverage、proxy prediction 和 CI 文件；它们用于调试 FraudDistill，不用于论文中声称优于现有工作。

## 表格
| Dataset | Method | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | Original metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fraud-R1 | Reproducible conservative proxy | 178 | 0.4382 | 0.53 | 0.5 | 0.5146 | 0.4239 | 0.6528 | 0.6296 | official baseline unavailable |
| Fraud-R1 | FraudDistill student proxy | 178 | 0.9607 | 0.9626 | 0.9717 | 0.9671 | 0.9591 | 0.0556 | 0.9876 | official baseline unavailable |
| Aegis2.0 | Reproducible conservative proxy | 5051 | 0.8887 | 0.4426 | 0.0486 | 0.0877 | 0.5142 | 0.0076 | 0.1232 | official baseline unavailable |
| Aegis2.0 | FraudDistill student proxy | 5051 | 0.8961 | 0.5191 | 0.7333 | 0.6079 | 0.774 | 0.0839 | 0.6157 | official baseline unavailable |
| Do-Not-Answer | Reproducible conservative proxy | 188 | 0.6915 | 0.0 | 0.0 | 0.0 | 0.4088 | 0.0076 | 0.373 | official baseline unavailable |
| Do-Not-Answer | FraudDistill student proxy | 188 | 0.8723 | 0.7705 | 0.8246 | 0.7966 | 0.8518 | 0.1069 | 0.8722 | official baseline unavailable |
| OR-Bench hard-safe | FraudDistill proxy | 60 |  |  |  |  |  | 0.2 |  | pure-safe FPR only |

## 实验3：Agent 与蒸馏

运行目录：`outputs\exp3_agent_distillation_ablation\high_standard_full`

﻿# 实验3：Agent 与蒸馏消融重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: f6ae0152b4f3eda2fc47c1edffc3351b6d1a98ee
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T04:18:29.598596+00:00
python_version: 3.12.2
config_path: outputs\exp3_agent_distillation_ablation\high_standard_full\config_resolved.yaml
config_sha256: 02ccba37ba93eb9dbe6763f0fc48c2b62978d2a50da817033d1991b9eaa225bb
```

## 结论与分析
修复点：teacher label 不再覆盖 gold，而是以附加 token/score/type/rank/context 信号进入训练；每个变体使用相同 train/dev/test manifest，并生成不同模型文件。

Agent 表拆成 nested ablation、leave-one-component-out 和组件压力指标三张表；全局梯度与专属边界下降需要同时成立才可写成不可替代性结论。

Teacher alignment train+dev: {'n': 21904, 'agreement': 0.8408053323593864, 'teacher_label_counts': Counter({'safe': 20542, 'unsafe': 1362}), 'gold_counts': Counter({'safe': 19037, 'unsafe': 2867}), 'confusion_matrix': {'tp': 371, 'fp': 991, 'fn': 2496, 'tn': 18046}, 'unsafe_recall': 0.12940355772584583, 'unsafe_precision': 0.2723935389133627, 'safe_fpr': 0.05205652151074224}

## 表格
## Student
| Variant | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Student-Gold | 0.625708 | 0.901 | 0.5998 | 0.7336 | 0.66 | 0.801 | 0.0737 | 0.712 |
| Gold + calibrated teacher label | 0.642332 | 0.9063 | 0.6259 | 0.7071 | 0.664 | 0.8048 | 0.0637 | 0.73 |
| Gold + label + soft score | 0.647044 | 0.9078 | 0.6342 | 0.6987 | 0.6649 | 0.8057 | 0.0607 | 0.7315 |
| Gold + label + soft + type | 0.631822 | 0.9072 | 0.6237 | 0.735 | 0.6748 | 0.8103 | 0.0668 | 0.7326 |
| Gold + label + soft + type + rank | 0.640387 | 0.9087 | 0.6341 | 0.7155 | 0.6723 | 0.8097 | 0.0622 | 0.7333 |
| Full + context auxiliary | 0.637388 | 0.9321 | 0.7379 | 0.7462 | 0.742 | 0.8515 | 0.0399 | 0.8518 |

## Agent nested
| Variant | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single Judge | 0.8632 | 0.3182 | 0.0391 | 0.0696 | 0.4979 | 0.0126 | 0.1382 |
| Fraud only | 0.8412 | 0.2632 | 0.1185 | 0.1635 | 0.5379 | 0.05 | 0.2358 |
| Fraud + Refusal | 0.8404 | 0.2524 | 0.1116 | 0.1547 | 0.5333 | 0.0498 | 0.2227 |
| Fraud + Refusal + Relevance | 0.8457 | 0.2576 | 0.0948 | 0.1386 | 0.527 | 0.0412 | 0.2239 |
| Full fixed | 0.8457 | 0.2594 | 0.0962 | 0.1404 | 0.5278 | 0.0414 | 0.224 |
| Full learned | 0.8464 | 0.254 | 0.0893 | 0.1321 | 0.5239 | 0.0395 | 0.2231 |
| Full learned calibrated | 0.8464 | 0.254 | 0.0893 | 0.1321 | 0.5239 | 0.0395 | 0.2241 |

## Leave-one-out
| Variant | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | 0.8464 | 0.254 | 0.0893 | 0.1321 | 0.5239 | 0.0395 | 0.2241 |
| Full - Fraud | 0.8673 | 0.25 | 0.007 | 0.0136 | 0.4712 | 0.0032 | 0.1228 |
| Full - Refusal | 0.8483 | 0.272 | 0.0948 | 0.1406 | 0.5287 | 0.0382 | 0.233 |
| Full - Relevance | 0.8401 | 0.246 | 0.1074 | 0.1495 | 0.5306 | 0.0496 | 0.2237 |
| Full - learned Arbiter | 0.8457 | 0.2594 | 0.0962 | 0.1404 | 0.5278 | 0.0414 | 0.2238 |

## Component pressure
| Subset | N | Full Recall_unsafe | Full - Fraud Recall_unsafe | Full - Refusal Recall_unsafe | Full - Relevance Recall_unsafe | Full - learned Arbiter Recall_unsafe | Full FPR | Full - Fraud FPR | Full - Refusal FPR | Full - Relevance FPR | Full - learned Arbiter FPR | Full Macro-F1 | Full - Fraud Macro-F1 | Full - Refusal Macro-F1 | Full - Relevance Macro-F1 | Full - learned Arbiter Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable fraud | 339 | 0.115 | 0.0118 | 0.1121 | 0.1327 | 0.1209 |  |  |  |  |  |  |  |  |  |  |
| partial leakage | 33 | 0.1515 | 0.0303 | 0.1515 | 0.1515 | 0.1515 |  |  |  |  |  |  |  |  |  |  |
| hard safe / anti-fraud | 186 |  |  |  |  |  | 0.1882 | 0.0806 | 0.1559 | 0.4462 | 0.2366 |  |  |  |  |  |
| agent conflict | 145 |  |  |  |  |  |  |  |  |  |  | 0.2816 | 0.2798 | 0.3018 | 0.2183 | 0.2669 |

## 实验4：未见泛化

运行目录：`outputs\exp4_unseen\high_standard_full`

﻿# 实验4：unseen 泛化重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: f6ae0152b4f3eda2fc47c1edffc3351b6d1a98ee
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T04:19:02.812265+00:00
python_version: 3.12.2
config_path: outputs\exp4_unseen\high_standard_full\config_resolved.yaml
config_sha256: 6f698c829d94c158874b014bd9e1e7fba196da23fc3282279dbdd441089bbfec
```

## 结论与分析
本轮改为 leave-one-category-out，并将 OR-Bench pure-safe 子集只按 FPR/specificity 解读。

当前 Fraud-R1 hard-control 公开数据仍不足以覆盖每个 held-out 类 300 unsafe + 300 safe 的论文强门槛；报告保留规模限制。

## 表格
| Setting | Held-out | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leave-one-category-out | fake_job_postings | 463 | 0.9633 | 0.9176 | 0.9882 | 0.9516 | 0.961 | 0.051 | 0.9908 |
| Leave-one-category-out | impersonation | 475 | 0.8674 | 0.8868 | 0.7581 | 0.8174 | 0.8566 | 0.0623 | 0.9118 |
| Leave-one-category-out | phishing_scams | 720 | 0.8514 | 0.8935 | 0.8542 | 0.8734 | 0.8468 | 0.1528 | 0.9565 |
| Source hard-safe | OR-Bench hard-safe | 300 | 0.9233 |  |  |  |  | 0.0767 |  |

## 实验5：概率校准

运行目录：`outputs\exp5_calibration\high_standard_full`

﻿# 实验5：阈值与概率校准重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: f6ae0152b4f3eda2fc47c1edffc3351b6d1a98ee
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T04:19:23.328816+00:00
python_version: 3.12.2
config_path: outputs\exp5_calibration\high_standard_full\config_resolved.yaml
config_sha256: 4921b02a4819aad760c468605471232ab5013d9e050d3966b822817de8266364
```

## 结论与分析
修复点：FPR cap 表述改为 dev-UCB 约束，并单独报告 observed test FPR 与 test UCB；不再把 dev 约束写成 test 保证。

主方法使用 Platt；Isotonic 因小样本/离散分数容易阈值退化，移出主表。

## 表格
| Method | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | ECE | Brier | Observed test FPR UCB95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default raw 0.5 | 0.5 | 0.8457 | 0.5382 | 0.8845 | 0.6692 | 0.7843 | 0.1626 | 0.7606 | 0.1427 | 0.1041 | 0.1713 |
| Platt default 0.5 | 0.5 | 0.8951 | 0.7372 | 0.63 | 0.6794 | 0.8084 | 0.0481 | 0.7606 | 0.0157 | 0.0743 | 0.0533 |
| Platt dev-UCB FPR<=0.01 | 0.837541 | 0.8558 | 0.888 | 0.2091 | 0.3385 | 0.6288 | 0.0056 | 0.7606 | 0.0157 | 0.0743 | 0.0076 |
| Platt dev-UCB FPR<=0.05 | 0.536158 | 0.8935 | 0.7523 | 0.5909 | 0.6619 | 0.7994 | 0.0417 | 0.7606 | 0.0157 | 0.0743 | 0.0465 |
| Platt dev-UCB FPR<=0.10 | 0.347483 | 0.8815 | 0.6368 | 0.7636 | 0.6945 | 0.8105 | 0.0933 | 0.7606 | 0.0157 | 0.0743 | 0.1002 |

## 实验6：多 API

运行目录：`outputs\exp6_multi_api\high_standard_full`

﻿# 实验6：多 API 目标模型重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: f6ae0152b4f3eda2fc47c1edffc3351b6d1a98ee
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T04:19:27.679846+00:00
python_version: 3.12.2
config_path: outputs\exp6_multi_api\high_standard_full\config_resolved.yaml
config_sha256: c9e5f414fabba74612b13d9378c547c1e2360756ef640a5584c10909ea0409c8
```

## 结论与分析
修复点：UAR/FAR/RFR/CRR/ORR 不再从 Recall/FPR 机械复制，而是由回答中的拒答行为、危险 prompt 类型和 Student unsafe 判定组合得到。

仍未满足论文级要求：没有 WildGuard/AegisGuard/HarmBench 独立共识，因此模型安全排名只能作为 detector-dependent 观察。

## 表格
## Student vs prompt reference
| Target LLM | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 0.57 | 0.784 | 0.4033 | 0.5326 | 0.5672 | 0.172 | 0.7314 | 1.0 |
| glm-4-flash | 400 | 0.5325 | 0.6818 | 0.4321 | 0.529 | 0.5325 | 0.3121 | 0.6563 | 1.0 |
| moonshot-v1-8k | 394 | 0.5406 | 0.7132 | 0.4059 | 0.5173 | 0.5395 | 0.2516 | 0.7131 | 1.0 |
| qwen-plus | 400 | 0.57 | 0.7383 | 0.4527 | 0.5612 | 0.5698 | 0.2484 | 0.7235 | 1.0 |

## Behavior
| Target LLM | N | FAR/UAR | RFR | CRR | Partial leakage rate | ORR | Avg latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 0.0947 | 0.3704 | 0.1934 | 0.0535 | 0.3376 | 8.1258 |
| glm-4-flash | 400 | 0.1235 | 0.321 | 0.2222 | 0.07 | 0.2866 | 16.349 |
| moonshot-v1-8k | 394 | 0.0711 | 0.2678 | 0.2301 | 0.046 | 0.3742 | 9.9152 |
| qwen-plus | 400 | 0.1646 | 0.4774 | 0.2263 | 0.1687 | 0.3758 | 12.5722 |

生成时间：2026-07-24T04:28:27.907988+00:00
