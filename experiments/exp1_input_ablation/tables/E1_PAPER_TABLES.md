# E1 Paper Tables (v4)

## Table E1-A: Natural fraud-assistance prevalence (A7500)

| Metric | Value |
|---|---|
| Registry rows | 7500 |
| Central positives | 11 (rate 0.001467) |
| Wilson 95% CI | 0.000819 ~ 0.002625 |
| Cluster-bootstrap 95% CI | 0.000534 ~ 0.002564 |
| Judge agreement | 0.9984 |

| Stratum | N | Positive | Rate |
|---|---|---|---|
| provider: deepseek | 3750 | 6 | 0.001600 |
| provider: qwen | 3750 | 5 | 0.001333 |
| scenario: assistant | 4282 | 2 | 0.000467 |
| scenario: roleplay | 3218 | 9 | 0.002797 |
| language: en | 3754 | 11 | 0.002930 |
| language: zh | 3746 | 0 | 0.000000 |

## Table E1-B-M0: TF-IDF/LR diagnostic baseline (Frozen Anchor 1200, 5 seeds mean)

| View | Macro-F1 | AUROC | AUPRC | Recall | Precision | FPR |
|---|---|---|---|---|---|---|
| q_only | 0.6625 | 0.7144 | 0.6393 | 0.8100 | 0.6320 | 0.4717 |
| y_only | 0.7860 | 0.8969 | 0.8962 | 0.8717 | 0.7461 | 0.2967 |
| q+y | 0.9508 | 0.9814 | 0.9750 | 0.9800 | 0.9260 | 0.0783 |

## Table E1-B-M2/M3: LLM single-judge four views (Frozen Anchor 1200)

| Judge | View | Macro-F1 | AUROC | AUPRC | Recall | Precision | FPR |
|---|---|---|---|---|---|---|---|
| Qwen | q_only | 0.3333 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen | y_only | 0.7576 | 0.7650 | 0.6845 | 0.9400 | 0.6963 | 0.4100 |
| Qwen | q_y | 0.8141 | 0.8175 | 0.7380 | 0.9533 | 0.7497 | 0.3183 |
| Qwen | wrong_q_y | 0.6536 | 0.6550 | 0.6050 | 0.5917 | 0.6775 | 0.2817 |
| DeepSeek | q_only | 0.6258 | 0.6258 | 0.5788 | 0.6233 | 0.6265 | 0.3717 |
| DeepSeek | y_only | 0.7601 | 0.7683 | 0.6867 | 0.9533 | 0.6959 | 0.4167 |
| DeepSeek | q_y | 0.8383 | 0.8408 | 0.7632 | 0.9667 | 0.7723 | 0.2850 |
| DeepSeek | wrong_q_y | 0.7055 | 0.7058 | 0.6425 | 0.7417 | 0.6921 | 0.3300 |

## Table E1-B-M1: XLM-R joint encoder (pending M1 training)

【M1 训练完成后填入】

## Table E1-C: Natural distribution transfer (pending C replay)

【M1 训练完成后填入】
