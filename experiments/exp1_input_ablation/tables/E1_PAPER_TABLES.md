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

## Table E1-B-M1: XLM-R joint encoder (Frozen Anchor 1200, 5 seeds mean ± sd)

| View | Macro-F1 | AUROC | AUPRC | Recall | Precision | FPR |
|---|---|---|---|---|---|---|
| q_only | 0.6450 ± 0.0165 | 0.7172 | 0.6497 | 0.9097 | 0.6135 | 0.5750 |
| y_only | 0.8017 ± 0.0032 | 0.9201 | 0.9242 | 0.8303 | 0.7962 | 0.2233 |
| **q_y** | **0.9685 ± 0.0035** | **0.9944** | **0.9934** | 0.9853 | 0.9534 | 0.0483 |
| wrong_q_y | 0.7574 ± 0.0039 | 0.7327 | 0.6632 | 0.8200 | 0.7300 | 0.3033 |

Δ_joint = +0.1717 (family-cluster bootstrap 95% CI, Macro-F1 formulation [0.1511, 0.1928], p_below_zero = 0.0); exact McNemar q_y vs y_only p = 1.10e-55; Holm-adjusted p < 0.05; q_y beats best-single 5/5 seeds. Wrong-q control (v4.5): same language + same fraud category + different family (1200/1200); wrong_q_y MF1 0.7574.

Stratified q_y MF1: B1 (y-matched) 0.9925 / B2 (q-matched) 0.9950 / B3 (context-stable mixed-source) 0.9273.

## Table E1-B-M1-clean: clean-anchor sensitivity (exclude 212 near-dup-y rows, n=988, 5 seeds mean ± sd)

| View | Macro-F1 | AUROC | AUPRC | Recall | FPR |
|---|---|---|---|---|---|
| q_only | 0.6713 ± 0.0196 | 0.7303 | 0.6919 | 0.9122 | 0.5492 |
| y_only | 0.8011 ± 0.0045 | 0.9196 | 0.9336 | 0.8414 | 0.2376 |
| q_y | 0.9700 ± 0.0031 | 0.9943 | 0.9940 | 0.9849 | 0.0468 |
| wrong_q_y | 0.7509 ± 0.0029 | 0.7166 | 0.6800 | 0.8249 | 0.3256 |

Clean Δ_joint = +0.1784 (bootstrap 95% CI [0.1551, 0.2028], p_below_zero = 0.0); McNemar q_y vs y_only b=165/c=2 p=1.50e-46; all sensitivity gates pass.

## Table E1-C: Natural distribution transfer (independent 624 rows / 6 positives, 5 seeds mean ± sd)

| View | Macro-F1 | Recall | FPR | AUROC | AUPRC | Recall@FPR1% | Recall@FPR5% | Precision@10 |
|---|---|---|---|---|---|---|---|---|
| q_y | 0.6748 ± 0.0808 | 0.5000 ± 0.1826 | 0.0162 ± 0.0167 | 0.9706 ± 0.0215 | 0.3970 ± 0.1538 | 0.5333 ± 0.1944 | 0.8667 ± 0.1247 | 0.3200 ± 0.1166 |
| y_only | 0.6099 ± 0.0250 | 0.7333 ± 0.3091 | 0.0427 ± 0.0261 | 0.9794 ± 0.0058 | 0.2983 ± 0.0361 | 0.4333 ± 0.1700 | 0.9000 ± 0.1333 | 0.3000 ± 0.0632 |

Notes: q_y achieves higher Macro-F1 / AUPRC / lower FPR than y_only; frozen-threshold recall is conservative (0.50); differences are exploratory given only 6 independent positives.
