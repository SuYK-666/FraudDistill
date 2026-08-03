| Benchmark | Method | N_pool | N_gold | Accuracy | Precision | Recall | Macro-F1 | FPR | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fraud-R1 | Fraud-R1 Official Judge | 8564 | 600 | 0.967 | 0.048 | 1.000 | 0.529 | 0.033 | 0.062 |
| Fraud-R1 | **FraudDistill-MAT (DeepSeek)** | 8564 | 600 | 0.998 | 0.000 | 0.000 | 0.500 | 0.000 | 0.003 |
| OR-Bench | OR-Bench Official Checker | 3000 | 600 | 0.658 | 0.120 | 0.650 | 0.431 | 0.341 | 0.123 |
| OR-Bench | **FraudDistill-MAT (DeepSeek)** | 3000 | 600 | 0.943 | 1.000 | 0.150 | 0.630 | 0.000 | 0.268 |
| Do-Not-Answer | Longformer-Harmful | 5634 | 5634 | 0.995 | 0.966 | 0.886 | 0.962 | 0.001 | 0.869 |
| Do-Not-Answer | **FraudDistill-MAT (DeepSeek)** | 5634 | 5634 | 0.905 | 0.102 | 0.228 | 0.535 | 0.071 | 0.126 |
| Aegis 2.0 | NemoGuard-8B | 1964 | 1964 | 0.951 | 0.870 | 0.886 | 0.922 | 0.033 | 0.812 |
| Aegis 2.0 | **FraudDistill-MAT (DeepSeek)** | 1964 | 1964 | 0.761 | 0.441 | 0.726 | 0.659 | 0.231 | 0.486 |