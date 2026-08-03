| Benchmark | Method | N_pool | N_gold | N+ | Accuracy | Precision | Recall | Macro-F1 | FPR | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fraud-R1 (balanced diag) | Official Judge (GPTCheck) | 600 | 600 | 300 | 0.505 | 0.508 | 0.303 | 0.543 | 0.293 | — |
| Fraud-R1 (balanced diag) | FraudDistill-MAT (4-agent) | 600 | 600 | 300 | 0.850 | 0.790 | 0.953 | 0.805 | 0.253 | 0.847 |
| Fraud-R1 (balanced diag) | **Budgeted Cascade (ours)** | 600 | 600 | 300 | 0.868 | 0.818 | 0.947 | 0.834 | 0.210 | 0.888 |
| OR-Bench | Official Response Checker | 3000 | 600 | 40 | 0.658 | 0.120 | 0.650 | 0.431 | 0.341 | — |
| OR-Bench | FraudDistill-MAT (4-agent) | 3000 | 600 | 40 | 0.943 | 1.000 | 0.150 | 0.630 | 0.000 | 0.268 |
| OR-Bench | **Budgeted Cascade (ours)** | 3000 | 600 | 40 | 0.930 | 0.479 | 0.575 | 0.739 | 0.045 | 0.065 |
| Do-Not-Answer | Longformer-Harmful | 5634 | 5634 | 193 | 0.995 | 0.966 | 0.886 | 0.962 | 0.001 | 0.966 |
| Do-Not-Answer | FraudDistill-MAT (4-agent) | 5634 | 5634 | 193 | 0.905 | 0.102 | 0.228 | 0.535 | 0.071 | 0.126 |
| Do-Not-Answer | **Budgeted Cascade (ours)** | 5634 | 5634 | 193 | 0.921 | 0.117 | 0.197 | 0.547 | 0.053 | 0.363 |
| Aegis 2.0 (valid q+y) | NemoGuard-8B (partial 694/813) | 813 | 813 | 394 | 0.808 | 0.872 | 0.708 | 0.842 | 0.098 | — |
| Aegis 2.0 (valid q+y) | FraudDistill-MAT (4-agent) | 1964 | 813 | 394 | 0.768 | 0.779 | 0.726 | 0.779 | 0.193 | 0.763 |
| Aegis 2.0 (valid q+y) | **Budgeted Cascade (ours)** | 813 | 813 | 394 | 0.677 | 0.874 | 0.388 | 0.743 | 0.053 | 0.773 |
