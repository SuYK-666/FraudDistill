# EXP2 Cross-Benchmark Report — FraudDistill Multi-Agent Teacher vs Official Baselines

> Generated 2026-08-03 · preregistration frozen before runs · all numbers from scripts (no manual entry).

## 1. Main 8-row table

| Benchmark | Method | N_pool | N_gold | Accuracy | Precision | Recall | Macro-F1 | FPR | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fraud-R1 | Fraud-R1 Official Judge | 8564 | 600 | 0.967 | 0.048 | 1.000 | 0.529 | 0.033 | 0.062 |
| Fraud-R1 | **FraudDistill-MAT (DeepSeek)** | 8564 | 600 | 0.998 | 0.000 | 0.000 | 0.500 | 0.000 | 0.003 |
| OR-Bench | OR-Bench Official Checker | 3000 | 600 | 0.658 | 0.120 | 0.650 | 0.431 | 0.341 | 0.123 |
| OR-Bench | **FraudDistill-MAT (DeepSeek)** | 3000 | 600 | 0.943 | 1.000 | 0.150 | 0.630 | 0.000 | 0.268 |
| Do-Not-Answer | Longformer-Harmful | 5634 | 5634 | 0.995 | 0.966 | 0.886 | 0.962 | 0.001 | 0.869 |
| Do-Not-Answer | **FraudDistill-MAT (DeepSeek)** | 5634 | 5634 | 0.905 | 0.102 | 0.228 | 0.535 | 0.071 | 0.126 |
| Aegis 2.0 | NemoGuard-8B | 1964 | 140 | 0.921 | 0.848 | 0.824 | 0.894 | 0.047 | 0.834 |
| Aegis 2.0 | **FraudDistill-MAT (DeepSeek)** | 1964 | 140 | 0.771 | 0.522 | 0.706 | 0.696 | 0.208 | 0.538 |

All methods evaluated on identical q+y pairs within each benchmark. Fraud-R1/OR-Bench metrics on independently audited subsets; large pools used for subgroup/cost analyses. abstain samples mapped to safe for primary metrics; coverage reported.

## 2. Paired significance

| Benchmark | ΔAcc [95% CI] | ΔMacro-F1 [95% CI] | ΔFPR [95% CI] | McNemar p | AUPRC Δ [95% CI] |
|---|---|---|---|---|---|
| Fraud-R1 | +0.032 [+0.018, +0.047] | -0.027 [-0.123, +0.023] | -0.033 [-0.048, -0.020] | 0.0000 | -0.115 [-0.575, +0.000] |
| OR-Bench | +0.285 [+0.243, +0.327] | +0.198 [+0.105, +0.298] | -0.341 [-0.380, -0.302] | 0.0000 | +0.117 [+0.000, +0.250] |
| Do-Not-Answer | -0.090 [-0.101, -0.080] | -0.427 [-0.452, -0.400] | +0.070 [+0.062, +0.079] | 0.0000 | -0.739 [-0.809, -0.662] |
| Aegis 2.0 | -0.150 [-0.221, -0.086] | -0.199 [-0.287, -0.113] | +0.161 [+0.088, +0.237] | 0.0001 | -0.236 [-0.418, -0.057] |

Clustered paired bootstrap (10,000 reps by group_id; AUPRC 2,000 reps); McNemar exact two-sided; Holm correction applied across benchmarks (see paired_significance.json).

## 3. Subgroup highlights

| Benchmark | Group | Subgroup | N | Gold rate | Baseline M-F1 | Teacher M-F1 | ΔM-F1 |
|---|---|---|---|---:|---:|---:|---:|
| Do-Not-Answer | category | Information Hazards | 1488 | 0.04 | 0.991 | 0.458 | -0.533 |
| Do-Not-Answer | category | Discrimination, Exclusion, Toxicity, Hat | 1056 | 0.04 | 0.983 | 0.505 | -0.479 |
| Do-Not-Answer | target_model | GPT4 | 939 | 0.02 | 0.989 | 0.524 | -0.465 |
| Do-Not-Answer | target_model | Claude | 939 | 0.02 | 0.907 | 0.453 | -0.454 |
| Do-Not-Answer | target_model | ChatGPT | 939 | 0.01 | 0.964 | 0.523 | -0.441 |
| Do-Not-Answer | language | English | 5634 | 0.03 | 0.962 | 0.535 | -0.427 |
| Do-Not-Answer | prompt_type | unknown | 5634 | 0.03 | 0.962 | 0.535 | -0.427 |
| OR-Bench | prompt_type | hard_safe | 250 | 0.10 | 0.209 | 0.603 | +0.394 |
| Do-Not-Answer | category | Misinformation Harms | 930 | 0.04 | 0.935 | 0.544 | -0.391 |
| Do-Not-Answer | target_model | vicuna-7b | 939 | 0.06 | 0.980 | 0.593 | -0.387 |
| Do-Not-Answer | target_model | ChatGLM2 | 939 | 0.09 | 0.958 | 0.607 | -0.351 |
| Do-Not-Answer | target_model | llama2-7b-chat | 939 | 0.00 | 0.785 | 0.438 | -0.347 |
| OR-Bench | category | deception | 77 | 0.05 | 0.381 | 0.700 | +0.319 |
| Do-Not-Answer | category | Human-Chatbot Interaction Harms | 702 | 0.03 | 0.860 | 0.548 | -0.311 |
| Do-Not-Answer | category | Malicious Uses | 1458 | 0.02 | 0.960 | 0.717 | -0.243 |
| OR-Bench | prompt_type | toxic | 200 | 0.05 | 0.500 | 0.731 | +0.231 |
| OR-Bench | category | illegal | 76 | 0.01 | 0.272 | 0.500 | +0.228 |
| OR-Bench | category | harassment | 76 | 0.08 | 0.429 | 0.643 | +0.214 |
| OR-Bench | language | English | 600 | 0.07 | 0.431 | 0.630 | +0.200 |
| OR-Bench | target_model | qwen-plus | 600 | 0.07 | 0.431 | 0.630 | +0.200 |
| Aegis 2.0 | language | English | 140 | 0.24 | 0.894 | 0.696 | -0.198 |
| Aegis 2.0 | category | general_safety | 140 | 0.24 | 0.894 | 0.696 | -0.198 |
| Aegis 2.0 | target_model | unknown | 140 | 0.24 | 0.894 | 0.696 | -0.198 |
| Aegis 2.0 | prompt_type | unknown | 140 | 0.24 | 0.894 | 0.696 | -0.198 |
| OR-Bench | category | privacy | 76 | 0.05 | 0.320 | 0.500 | +0.180 |
| OR-Bench | category | sexual | 35 | 0.31 | 0.583 | 0.714 | +0.131 |
| OR-Bench | category | hate | 77 | 0.10 | 0.490 | 0.611 | +0.121 |
| OR-Bench | prompt_type | regular_safe | 150 | 0.03 | 0.585 | 0.500 | -0.085 |
| OR-Bench | category | harmful | 77 | 0.08 | 0.421 | 0.500 | +0.079 |
| OR-Bench | category | self-harm | 36 | 0.00 | 0.458 | 0.500 | +0.042 |
| Fraud-R1 | language | Chinese | 300 | 0.00 | 0.539 | 0.500 | -0.039 |
| Fraud-R1 | target_model | qwen-plus | 600 | 0.00 | 0.529 | 0.500 | -0.029 |
| Fraud-R1 | prompt_type | unknown | 600 | 0.00 | 0.529 | 0.500 | -0.029 |
| Fraud-R1 | category | Phishing Scams | 120 | 0.00 | 0.475 | 0.500 | +0.025 |
| Fraud-R1 | category | Fake Job Postings | 120 | 0.01 | 0.521 | 0.500 | -0.021 |
| OR-Bench | category | unethical | 35 | 0.00 | 0.486 | 0.500 | +0.014 |
| OR-Bench | category | violence | 35 | 0.00 | 0.486 | 0.500 | +0.014 |
| Fraud-R1 | language | English | 300 | 0.00 | 0.490 | 0.500 | +0.010 |
| Fraud-R1 | category | Online Relationships | 120 | 0.00 | 0.492 | 0.500 | +0.008 |
| Fraud-R1 | category | Fraudulent Services | 120 | 0.00 | 0.500 | 0.500 | +0.000 |

Full subgroup table: `_metrics/subgroup_metrics.csv`.

## 4. Error analysis (paired)

- baseline_correct_teacher_wrong: 568
- baseline_wrong_teacher_correct: 228
- both_correct: 6142
- both_wrong: 36

Redacted sample-level errors: `_metrics/error_analysis.jsonl` and `error_analysis_redacted.md`.

## 5. Cost summary

| Benchmark | Method | N | input tok | output tok | est. RMB | mean latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| Fraud-R1 | baseline | 8564 | 10200247 | 10256 | 10.2208 | 3748.0 |
| Fraud-R1 | teacher | 8564 | 52209365 | 4123375 | 60.4561 | 10805.7 |
| OR-Bench | baseline | 3798 | 2262313 | 220600 | 2.7035 | 2910.2 |
| OR-Bench | teacher | 3000 | 9833421 | 1377750 | 12.5889 | 13593.1 |
| Do-Not-Answer | baseline | 5634 | 0 | 0 | 0.0 | 529.5 |
| Do-Not-Answer | teacher | 5634 | 11130528 | 2118060 | 15.3666 | 9489.9 |
| Aegis 2.0 | baseline | 143 | 0 | 0 | 0.0 | 167606.4 |
| Aegis 2.0 | teacher | 1964 | 3597069 | 660339 | 4.9177 | 11513.2 |

## 6. Deliverables index

- `experiments/exp2_prior_work_comparison/fraudr1/unified/fraudr1_eval.jsonl`
- `experiments/exp2_prior_work_comparison/fraudr1/baseline_predictions/`
- `experiments/exp2_prior_work_comparison/fraudr1/teacher_predictions/`
- `experiments/exp2_prior_work_comparison/fraudr1/human_audit/`
- `experiments/exp2_prior_work_comparison/orbench/unified/orbench_eval.jsonl`
- `experiments/exp2_prior_work_comparison/orbench/baseline_predictions/`
- `experiments/exp2_prior_work_comparison/orbench/teacher_predictions/`
- `experiments/exp2_prior_work_comparison/orbench/human_audit/`
- `experiments/exp2_prior_work_comparison/do_not_answer/unified/do_not_answer_eval.jsonl`
- `experiments/exp2_prior_work_comparison/do_not_answer/baseline_predictions/`
- `experiments/exp2_prior_work_comparison/do_not_answer/teacher_predictions/`
- `experiments/exp2_prior_work_comparison/do_not_answer/human_audit/`
- `experiments/exp2_prior_work_comparison/aegis2/unified/aegis2_eval.jsonl`
- `experiments/exp2_prior_work_comparison/aegis2/baseline_predictions/`
- `experiments/exp2_prior_work_comparison/aegis2/teacher_predictions/`
- `experiments/exp2_prior_work_comparison/aegis2/human_audit/`