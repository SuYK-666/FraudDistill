# FraudDistill 实验1 Context-Valid Prompt-Parity R3 任务报告

- 协议：`E1-Context-Valid-PromptParity-R3`
- 最新阶段：`anchor`
- 最新判定：`E1_R3_ANCHOR_STOP_PERFORMANCE`
- Git 提交：`41bd6d6aae054394fb25878691a7cb58a26bb508`

## 本轮代码与数据修订

1. Anchor 改为 Panel A/B/C 显式构建，不再从旧 hard-control 随机抽样。
2. Prompt 改为 system rubric + user JSON payload，删除 `[ABLATION_MODE]` 与 `evidence_basis`。
3. Panel C 使用 Fraud-R1 raw_data 构造用户问题，目标回答来源限定为 API target response，并执行 generated text hash 泄漏审计。
4. 指标补充 balanced accuracy、specificity、MCC、Brier、ECE、NLL，并按 panel 输出。
5. API 并发配置为 100，失败即按 R3 Gate 停止。


## panel_c

- 判定：`E1_R3_PANEL_C_PASS`
- 用时：`678.21` 秒

## g0

- 判定：`E1_R3_G0_PASS`
- 用时：`0.924` 秒

## preflight

- 判定：`E1_R3_PREFLIGHT_PASS`
- 用时：`39.556` 秒

## anchor

- 判定：`E1_R3_ANCHOR_STOP_PERFORMANCE`
- 用时：`45.75` 秒

| Input | Macro-F1 | Recall | Precision | FPR | AUPRC | AUROC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q_only | 0.5787 | 0.3667 | 0.6929 | 0.1625 | 0.6051 | 0.6030 | 0.3828 | 0.3767 |
| y_only | 0.7246 | 0.6000 | 0.8090 | 0.1417 | 0.7666 | 0.7904 | 0.2513 | 0.2444 |
| q_y | 0.7259 | 0.6625 | 0.7608 | 0.2083 | 0.7639 | 0.7824 | 0.2569 | 0.2536 |

### Panel 指标

| Panel | Input | Macro-F1 | Delta(q+y-y) | AUPRC | AUROC |
|---|---|---:|---:|---:|---:|
| context_critical_public | q_only | 0.4908 | 0.0081 | 0.5864 | 0.5743 |
| context_critical_public | y_only | 0.7017 | 0.0081 | 0.7959 | 0.8040 |
| context_critical_public | q_y | 0.7098 | 0.0081 | 0.7623 | 0.7585 |
| fraudr1_target_response | q_only | 0.9054 | 0.0000 | 0.8651 | 0.9199 |
| fraudr1_target_response | y_only | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| fraudr1_target_response | q_y | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| natural_public | q_only | 0.4670 | 0.0029 | 0.5330 | 0.5447 |
| natural_public | y_only | 0.6374 | 0.0029 | 0.6599 | 0.6927 |
| natural_public | q_y | 0.6403 | 0.0029 | 0.6676 | 0.6939 |
| public_main | q_only | 0.4728 | 0.0042 | 0.5440 | 0.5514 |
| public_main | y_only | 0.6524 | 0.0042 | 0.6850 | 0.7160 |
| public_main | q_y | 0.6566 | 0.0042 | 0.6878 | 0.7072 |

### Gate 检查

| Check | Pass |
|---|---:|
| `api_success_100` | True |
| `q_only_range` | True |
| `y_only_range` | False |
| `q_y_macro_f1` | False |
| `q_y_minus_y` | False |
| `y_minus_q` | False |
| `q_y_minus_y_ci_lower` | False |
| `q_y_recall` | False |
| `q_y_precision` | False |
| `q_y_fpr` | False |
| `q_y_auprc` | False |
| `q_y_auroc` | False |
| `auprc_non_degrade` | False |
| `auroc_non_degrade` | False |
| `holm_mcnemar` | False |
| `panel_natural_public` | False |
| `panel_context_critical_public` | False |
| `panel_fraudr1_target_response` | False |
| `panel_public_main` | False |

## 输出与保留数据

- 数据目录：`C:\Users\18201\Desktop\FraudDistill\data\prepared\e1_context_valid_r3`
- 输出目录：`C:\Users\18201\Desktop\FraudDistill\outputs\e1_context_valid_r3`
- 所有原始 API 预测保留在各阶段 `predictions/`。
- G0 审计文件包括 SOURCE_LOCK、PANEL_CENSUS、LABEL_PROVENANCE、CONSTRUCT、DUPLICATE、CONTEXT_PROBE、FRAUDR1_HASH_LEAKAGE 和 PROTOCOL_LOCK。