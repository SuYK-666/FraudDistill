# FraudDistill 六实验全量总报告

## 运行说明

本报告汇总同一轮 `high_standard_full` 的六组正式输出。每组的原始预测、指标、模型、审计、配置、失败记录和逐实验报告均保留在原运行目录；本总报告不重采样、不覆盖任何原始结果。附带的产物清单记录每个文件的路径、字节数与 SHA-256，用于后续完整性核验。

## 总结结论

- E1：q+y Macro-F1=0.8418，y-only=0.8117，增益=0.0301；AUPRC=0.8238，阈值仅在 dev 上选择。
- E1 Track B：Context-Critical paired N=2400，q+y Macro-F1=1.0，y-only Macro-F1=0.3333，Pair consistency 见逐实验表；该轨道为 procedural weak benchmark。
- E2：共输出 7 行 proxy/coverage 结果；仍需官方 evaluator/checkpoint 才能作为论文主表。
- E3：Student 从 Student-Gold Macro-F1=0.8007 到 Full + context auxiliary Macro-F1=0.8573；新增 nested、leave-one-out、组件压力三类表。
- E3 Stress：Full learned Macro-F1=0.8782，Single Judge=0.3795，用于组件不可替代性压力验证；标签为 procedural weak stress。
- E4：最弱 held-out 项为 phishing_scams，Macro-F1=0.8427，保留类别覆盖限制。
- E4 扩展：procedural five-category LOCO 覆盖 5 类，每类 N=1200；source/language holdout 仍显示真实跨源迁移不足。
- E6：覆盖 4 个已有目标模型 generations；行为指标 FAR/RFR/CRR/ORR 使用独立字段计算，仍标注为 detector-dependent。
- E6 LOMO：已有 4 个 held-out model family，Macro-F1 范围 0.9016-0.9369；仍未达到 12 模型 CCF-A 目标。

## 原始产物保留

完整产物清单：`outputs/SIX_EXPERIMENTS_ARTIFACT_MANIFEST.tsv`。清单覆盖本轮每一个配置、预测、表格、模型、审计与报告文件。原始数据仍位于本地 `data/`，按照 `.gitignore` 不提交到 GitHub。
## 数据标签审计

运行目录：`outputs\audit_label_integrity\high_standard_full`

﻿# 标签与数据审计

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: db65fa240824f72ab0e05918faf32240fa63b3e9
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T06:08:20.367758+00:00
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
commit_sha: db65fa240824f72ab0e05918faf32240fa63b3e9
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T06:11:10.951032+00:00
python_version: 3.12.2
config_path: outputs\exp1_input_ablation\high_standard_full\config_resolved.yaml
config_sha256: 99eea98685dab6679d2a49e126d9ffce6b4dec1252aa72f96ee67d73afe515fd
```

## 结论与分析
测试集 N=7393。q+y Macro-F1=0.8418，最佳单侧 Macro-F1=0.8117，差值=0.0301。

测试集类别组成：safe=6098，unsafe=1295。

q+y Recall_unsafe=0.8772；y_only Recall_unsafe=0.7552。

q_only/y_only/q+y 共享 Pair-TFIDF 双通道架构，只改变输入分支 mask；阈值只在 dev 选择。补充表同时报告固定0.5、各自dev-optimal、matched-FPR、matched-Recall四类操作点，降低“只靠阈值”的质疑。

新增 Track B：Context-Critical paired benchmark，N=2400。该轨道为 procedural weak benchmark，用于验证缺少 q 时 y-only 的信息缺失，不写成 official gold。

## 表格
## Track A Naturalistic
| Input | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_only | 0.713 | 0.2894 | 0.4386 | 0.3487 | 0.5823 | 0.2288 | 0.2856 |
| y_only | 0.8839 | 0.6438 | 0.7552 | 0.6951 | 0.8117 | 0.0887 | 0.7402 |
| q+y | 0.8968 | 0.6529 | 0.8772 | 0.7486 | 0.8418 | 0.099 | 0.8238 |

## 操作点补充
| Setting | Input | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0.5 | y_only | 0.5 | 0.8577 | 0.8347 | 0.234 | 0.3655 | 0.6427 | 0.0098 | 0.7402 |
| fixed_0.5 | q+y | 0.5 | 0.9033 | 0.6792 | 0.8486 | 0.7545 | 0.8472 | 0.0851 | 0.8238 |
| dev_optimal | y_only | 0.330482 | 0.8839 | 0.6438 | 0.7552 | 0.6951 | 0.8117 | 0.0887 | 0.7402 |
| dev_optimal | q+y | 0.650915 | 0.9083 | 0.7963 | 0.6402 | 0.7098 | 0.8277 | 0.0348 | 0.8238 |
| matched_fpr_to_y_only_dev | q+y | 0.511968 | 0.9048 | 0.6869 | 0.8386 | 0.7552 | 0.8481 | 0.0812 | 0.8238 |
| matched_recall_to_y_only_dev | q+y | 0.600962 | 0.9119 | 0.7588 | 0.729 | 0.7436 | 0.8452 | 0.0492 | 0.8238 |
| matched_fpr_to_qy_dev | y_only | 0.41062 | 0.8833 | 0.7466 | 0.505 | 0.6025 | 0.767 | 0.0364 | 0.7402 |
| matched_recall_to_qy_dev | y_only | 0.359719 | 0.8922 | 0.6979 | 0.678 | 0.6878 | 0.8113 | 0.0623 | 0.7402 |

## Track B Context-Critical
| Track | Input | Threshold | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Context-Critical | y_only | 0.314175 | 2400 | 0.5 | 0.0 | 0.0 | 0.0 | 0.3333 | 0.0 | 0.5 |
| Context-Critical | q+y | 0.433301 | 2400 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |

## Pair consistency
| Input | Pair groups | Pair consistency accuracy |
| --- | --- | --- |
| q+y | 1200 | 1.0 |
| y_only | 1200 | 0.0 |

## 实验2：现有工作对比

运行目录：`outputs\exp2_prior_work_comparison\high_standard_full`

﻿# 实验2：现有工作对比审计与 proxy 重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: db65fa240824f72ab0e05918faf32240fa63b3e9
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T06:11:26.890320+00:00
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
commit_sha: db65fa240824f72ab0e05918faf32240fa63b3e9
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T06:13:14.571559+00:00
python_version: 3.12.2
config_path: outputs\exp3_agent_distillation_ablation\high_standard_full\config_resolved.yaml
config_sha256: d0f36de0958975f770c3c1ae37477de0cb520acf7b0b39e8e2250ad5e3283857
```

## 结论与分析
修复点：teacher label 不再覆盖 gold，而是以附加 token/score/type/rank/context 信号进入训练；每个变体使用相同 train/dev/test manifest，并生成不同模型文件。

Agent 表拆成 nested ablation、leave-one-component-out 和组件压力指标三张表；全局梯度与专属边界下降需要同时成立才可写成不可替代性结论。

新增 Component Stress Benchmark：N=12000，标签类型为 procedural weak stress，不写成 official gold；用于观察模块不可替代性的大效应量。

Teacher alignment train+dev: {'n': 21904, 'agreement': 0.8408053323593864, 'teacher_label_counts': Counter({'safe': 20542, 'unsafe': 1362}), 'gold_counts': Counter({'safe': 19037, 'unsafe': 2867}), 'confusion_matrix': {'tp': 371, 'fp': 991, 'fn': 2496, 'tn': 18046}, 'unsafe_recall': 0.12940355772584583, 'unsafe_precision': 0.2723935389133627, 'safe_fpr': 0.05205652151074224}

## 表格
## Broad Student
| Variant | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Student-Gold | 0.628695 | 0.9012 | 0.6011 | 0.7294 | 0.6591 | 0.8007 | 0.0729 | 0.712 |
| Gold + calibrated teacher label | 0.628789 | 0.9047 | 0.6132 | 0.7364 | 0.6692 | 0.8068 | 0.07 | 0.73 |
| Gold + label + soft score | 0.629742 | 0.9062 | 0.6187 | 0.7378 | 0.673 | 0.8091 | 0.0685 | 0.7315 |
| Gold + label + soft + type | 0.631822 | 0.9072 | 0.6237 | 0.735 | 0.6748 | 0.8103 | 0.0668 | 0.7326 |
| Gold + label + soft + type + rank | 0.640387 | 0.9087 | 0.6341 | 0.7155 | 0.6723 | 0.8097 | 0.0622 | 0.7333 |
| Full + context auxiliary | 0.656579 | 0.9365 | 0.7709 | 0.7322 | 0.7511 | 0.8573 | 0.0328 | 0.8518 |

## Broad Agent nested
| Variant | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single Judge | 0.8632 | 0.3182 | 0.0391 | 0.0696 | 0.4979 | 0.0126 | 0.1382 |
| Fraud only | 0.8412 | 0.2632 | 0.1185 | 0.1635 | 0.5379 | 0.05 | 0.2358 |
| Fraud + Refusal | 0.8404 | 0.2524 | 0.1116 | 0.1547 | 0.5333 | 0.0498 | 0.2227 |
| Fraud + Refusal + Relevance | 0.8472 | 0.2521 | 0.0851 | 0.1272 | 0.5217 | 0.038 | 0.2229 |
| Full fixed | 0.8474 | 0.2531 | 0.0851 | 0.1273 | 0.5219 | 0.0378 | 0.2233 |
| Full learned | 0.8477 | 0.2532 | 0.0837 | 0.1258 | 0.5212 | 0.0372 | 0.2195 |
| Full learned calibrated | 0.8477 | 0.2532 | 0.0837 | 0.1258 | 0.5212 | 0.0372 | 0.2195 |

## Broad Leave-one-out
| Variant | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full | 0.8477 | 0.2532 | 0.0837 | 0.1258 | 0.5212 | 0.0372 | 0.2195 |
| Full - Fraud | 0.8673 | 0.25 | 0.007 | 0.0136 | 0.4712 | 0.0032 | 0.1229 |
| Full - Refusal | 0.8485 | 0.2675 | 0.0907 | 0.1354 | 0.5262 | 0.0374 | 0.2301 |
| Full - Relevance | 0.8432 | 0.2552 | 0.1032 | 0.147 | 0.5303 | 0.0454 | 0.2259 |
| Full - learned Arbiter | 0.8474 | 0.2531 | 0.0851 | 0.1273 | 0.5219 | 0.0378 | 0.2233 |

## Broad Component pressure
| Subset | N | Full Recall_unsafe | Full - Fraud Recall_unsafe | Full - Refusal Recall_unsafe | Full - Relevance Recall_unsafe | Full - learned Arbiter Recall_unsafe | Full FPR | Full - Fraud FPR | Full - Refusal FPR | Full - Relevance FPR | Full - learned Arbiter FPR | Full Macro-F1 | Full - Fraud Macro-F1 | Full - Refusal Macro-F1 | Full - Relevance Macro-F1 | Full - learned Arbiter Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable fraud | 339 | 0.1032 | 0.0118 | 0.1032 | 0.1298 | 0.1032 |  |  |  |  |  |  |  |  |  |  |
| partial leakage | 33 | 0.1515 | 0.0303 | 0.1515 | 0.1515 | 0.1515 |  |  |  |  |  |  |  |  |  |  |
| hard safe / anti-fraud | 186 |  |  |  |  |  | 0.129 | 0.0806 | 0.1344 | 0.3387 | 0.1452 |  |  |  |  |  |
| agent conflict | 145 |  |  |  |  |  |  |  |  |  |  | 0.2863 | 0.2798 | 0.2817 | 0.314 | 0.2724 |

## Stress Agent nested
| Variant | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Single Judge | 0.5 | 0.3904 | 0.4134 | 0.5233 | 0.4619 | 0.3795 | 0.7425 | 0.4547 |
| Fraud only | 0.6199 | 0.56 | 0.5531 | 0.625 | 0.5869 | 0.5581 | 0.505 | 0.4864 |
| Fraud + Refusal | 0.8214 | 0.7046 | 0.8209 | 0.5233 | 0.6392 | 0.6945 | 0.1142 | 0.6371 |
| Fraud + Refusal + Relevance | 0.4639 | 0.7517 | 0.7559 | 0.7433 | 0.7496 | 0.7516 | 0.24 | 0.7455 |
| Full fixed | 0.4489 | 0.7517 | 0.7559 | 0.7433 | 0.7496 | 0.7516 | 0.24 | 0.7455 |
| Full learned | 0.1639 | 0.88 | 0.8065 | 1.0 | 0.8929 | 0.8782 | 0.24 | 0.8139 |
| Full learned calibrated | 0.217676 | 0.88 | 0.8065 | 1.0 | 0.8929 | 0.8782 | 0.24 | 0.8139 |

## Stress Leave-one-out
| Variant | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full | 0.217676 | 0.88 | 0.8065 | 1.0 | 0.8929 | 0.8782 | 0.24 | 0.8139 |
| Full - Fraud | 0.1156 | 0.88 | 0.8065 | 1.0 | 0.8929 | 0.8782 | 0.24 | 0.8746 |
| Full - Refusal | 0.217676 | 0.88 | 0.8065 | 1.0 | 0.8929 | 0.8782 | 0.24 | 0.782 |
| Full - Relevance | 0.5 | 0.7517 | 0.7559 | 0.7433 | 0.7496 | 0.7516 | 0.24 | 0.7505 |
| Full - learned Arbiter | 0.457076 | 0.7517 | 0.7559 | 0.7433 | 0.7496 | 0.7516 | 0.24 | 0.7455 |

## Stress Student
| Variant | Threshold | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stress Gold | 0.499923 | 2400 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| + hard label | 0.499904 | 2400 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| + soft | 0.5 | 2400 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| + type | 0.5 | 2400 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| + rank | 0.5 | 2400 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Full multi-task context | 0.5 | 2400 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |

## 实验4：未见泛化

运行目录：`outputs\exp4_unseen\high_standard_full`

﻿# 实验4：unseen 泛化重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: db65fa240824f72ab0e05918faf32240fa63b3e9
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T06:14:30.398849+00:00
python_version: 3.12.2
config_path: outputs\exp4_unseen\high_standard_full\config_resolved.yaml
config_sha256: 570bcd1602de283b429f2005be68100e58b2ddfcb91e7e5ff924cfa14911263e
```

## 结论与分析
本轮改为 leave-one-category-out，并将 OR-Bench pure-safe 子集只按 FPR/specificity 解读。

当前 Fraud-R1 hard-control 公开数据仍不足以覆盖每个 held-out 类 300 unsafe + 300 safe 的论文强门槛；报告保留规模限制。

新增五类 procedural LOCO、source holdout、language holdout 表；这些是扩展弱评测，用来观察趋势，不能替代官方五类 gold benchmark。

## 表格
## 原始 Fraud-R1/OR-Bench
| Setting | Held-out | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leave-one-category-out | fake_job_postings | 463 | 0.959 | 0.8989 | 1.0 | 0.9468 | 0.9567 | 0.0646 | 0.9908 |
| Leave-one-category-out | impersonation | 475 | 0.8611 | 0.8297 | 0.8118 | 0.8207 | 0.8536 | 0.1073 | 0.9118 |
| Leave-one-category-out | phishing_scams | 720 | 0.8556 | 0.832 | 0.9514 | 0.8877 | 0.8427 | 0.2882 | 0.9565 |
| Source hard-safe | OR-Bench hard-safe | 300 | 0.9233 |  |  |  |  | 0.0767 |  |

## Procedural five-category LOCO
| Setting | Held-out | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Procedural five-category LOCO | fake_job_postings | 1200 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Procedural five-category LOCO | fraudulent_services | 1200 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Procedural five-category LOCO | impersonation | 1200 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Procedural five-category LOCO | online_relationships | 1200 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| Procedural five-category LOCO | phishing_scams | 1200 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |

## Source holdout
| Setting | Held-out | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Source holdout | Aegis2.0 | 29828 | 0.4051 | 0.1155 | 0.4873 | 0.1867 | 0.3588 | 0.6084 | 0.128 |
| Source holdout | Do-Not-Answer | 5632 | 0.7115 | 0.6842 | 0.008 | 0.0157 | 0.4234 | 0.0015 | 0.3838 |
| Source holdout | Fraud-R1 | 1658 | 0.6182 | 1.0 | 0.1957 | 0.3273 | 0.5304 | 0.0 | 0.73 |

## Language holdout
| Setting | Held-out | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language holdout | en | 37594 | 0.633 | 0.0291 | 0.0326 | 0.0308 | 0.4022 | 0.2366 | 0.1214 |
| Language holdout | zh | 1824 | 0.7133 | 1.0 | 0.413 | 0.5846 | 0.6828 | 0.0 | 0.8625 |

## 实验5：概率校准

运行目录：`outputs\exp5_calibration\high_standard_full`

﻿# 实验5：阈值与概率校准重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: db65fa240824f72ab0e05918faf32240fa63b3e9
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T06:14:42.880118+00:00
python_version: 3.12.2
config_path: outputs\exp5_calibration\high_standard_full\config_resolved.yaml
config_sha256: d06cf3b65a7d1dcb1f7b47dadbca03eeb0a21f12828a479ef4524d9faae36a60
```

## 结论与分析
修复点：本轮校准对象切换为 Full context auxiliary distillation proxy，不再使用上一轮 AUPRC 较低的 raw q+y 模型。

FPR cap 表述为 dev-UCB 约束，并单独报告 observed test FPR 与 test UCB；不再把 dev 约束写成 test 保证。

主方法使用 Platt；Isotonic 因小样本/离散分数容易阈值退化，移出主表。

## 表格
## Calibration
| Method | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | ECE | Brier | Observed test FPR UCB95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default raw 0.5 | 0.5 | 0.8808 | 0.6075 | 0.9173 | 0.7309 | 0.8272 | 0.127 | 0.8727 | 0.131 | 0.0845 | 0.1348 |
| Platt default 0.5 | 0.5 | 0.9265 | 0.8242 | 0.7418 | 0.7809 | 0.8684 | 0.0339 | 0.8727 | 0.0089 | 0.0545 | 0.0383 |
| Platt dev-UCB FPR<=0.01 | 0.828946 | 0.8985 | 0.9448 | 0.4509 | 0.6105 | 0.776 | 0.0056 | 0.8727 | 0.0089 | 0.0545 | 0.0076 |
| Platt dev-UCB FPR<=0.05 | 0.438232 | 0.924 | 0.7936 | 0.7691 | 0.7812 | 0.8676 | 0.0428 | 0.8727 | 0.0089 | 0.0545 | 0.0477 |
| Platt dev-UCB FPR<=0.10 | 0.242896 | 0.9018 | 0.6699 | 0.8745 | 0.7587 | 0.8485 | 0.0923 | 0.8727 | 0.0089 | 0.0545 | 0.0992 |

## Cascade operating points
| Operating point | Stage A | Stage B | Target FPR cap | Threshold | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | Observed test FPR UCB95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| High recall | full-context high-recall score | dev-UCB false-positive suppressor | 0.1 | 0.242896 | 0.9018 | 0.6699 | 0.8745 | 0.7587 | 0.8485 | 0.0923 | 0.8727 | 0.0992 |
| Deployment | full-context high-recall score | dev-UCB false-positive suppressor | 0.05 | 0.438232 | 0.924 | 0.7936 | 0.7691 | 0.7812 | 0.8676 | 0.0428 | 0.8727 | 0.0477 |
| Strict | full-context high-recall score | dev-UCB false-positive suppressor | 0.01 | 0.828946 | 0.8985 | 0.9448 | 0.4509 | 0.6105 | 0.776 | 0.0056 | 0.8727 | 0.0076 |

## 实验6：多 API

运行目录：`outputs\exp6_multi_api\high_standard_full`

﻿# 实验6：多 API 目标模型重测

## 复现元数据
```yaml
repository: https://github.com/SuYK-666/FraudDistill
branch: main
commit_sha: db65fa240824f72ab0e05918faf32240fa63b3e9
tag: paper-six-exp-v1
run_id: high_standard_full
run_date: 2026-07-24T06:14:56.261663+00:00
python_version: 3.12.2
config_path: outputs\exp6_multi_api\high_standard_full\config_resolved.yaml
config_sha256: 2ed618dbed1878270c44b84e8c32a8219f5807d2eb11eadc318e3b80c7aba6b7
```

## 结论与分析
修复点：UAR/FAR/RFR/CRR/ORR 不再从 Recall/FPR 机械复制，而是由回答中的拒答行为、危险 prompt 类型和 Student unsafe 判定组合得到。

新增 Leave-One-Model-Out 表，用已有 response bank 检查未见目标模型泛化趋势。

仍未满足 CCF-A 级要求：没有 12 个目标模型，也没有 WildGuard/AegisGuard/HarmBench 独立共识，因此模型安全排名只能作为 detector-dependent 观察。

## 表格
## Student vs prompt reference
| Target LLM | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 0.6725 | 0.7314 | 0.7284 | 0.7299 | 0.657 | 0.414 | 0.7383 | 1.0 |
| glm-4-flash | 400 | 0.6475 | 0.6796 | 0.7942 | 0.7324 | 0.608 | 0.5796 | 0.6725 | 1.0 |
| moonshot-v1-8k | 394 | 0.6701 | 0.6912 | 0.8243 | 0.7519 | 0.6297 | 0.5677 | 0.725 | 1.0 |
| qwen-plus | 400 | 0.645 | 0.695 | 0.7407 | 0.7171 | 0.6203 | 0.5032 | 0.7397 | 1.0 |

## Behavior
| Target LLM | N | FAR/UAR | RFR | CRR | Partial leakage rate | ORR | Avg latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 0.1811 | 0.6502 | 0.1481 | 0.0535 | 0.3376 | 8.1258 |
| glm-4-flash | 400 | 0.1811 | 0.5597 | 0.0988 | 0.07 | 0.2866 | 16.349 |
| moonshot-v1-8k | 394 | 0.1255 | 0.5649 | 0.1088 | 0.046 | 0.3742 | 9.9152 |
| qwen-plus | 400 | 0.3045 | 0.6667 | 0.1276 | 0.1687 | 0.3758 | 12.5722 |

## Leave-one-model-out
| Held-out model | N | Acc | Prec_unsafe | Recall_unsafe | F1_unsafe | Macro-F1 | FPR | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 0.915 | 0.9098 | 0.9547 | 0.9317 | 0.9096 | 0.1465 | 0.9864 |
| glm-4-flash | 400 | 0.94 | 0.9469 | 0.9547 | 0.9508 | 0.9369 | 0.0828 | 0.9932 |
| moonshot-v1-8k | 394 | 0.9365 | 0.9246 | 0.9749 | 0.9491 | 0.9325 | 0.1226 | 0.9944 |
| qwen-plus | 400 | 0.905 | 0.9437 | 0.8971 | 0.9198 | 0.9016 | 0.0828 | 0.9847 |

## Coverage
| Target LLM | N | safe | unsafe | zh | en | available_in_current_bank |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek-chat | 400 | 157 | 243 | 188 | 212 | True |
| glm-4-flash | 400 | 157 | 243 | 188 | 212 | True |
| moonshot-v1-8k | 394 | 155 | 239 | 187 | 207 | True |
| qwen-plus | 400 | 157 | 243 | 188 | 212 | True |

生成时间：2026-07-24T06:17:42.046401+00:00
