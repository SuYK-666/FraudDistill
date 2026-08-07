| Benchmark | System | N | Acc | Prec | Recall | Macro-F1 | FPR | AUPRC | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fraudr1 | FraudDistill | 3000 | 0.9847 | 0.9789 | 0.9907 | 0.9847 | 0.0213 | 0.9958 | 0.9694 |
| fraudr1 | Official Judge (GPTCheck) | 3000 | 0.6963 | 0.7420 | 0.6020 | 0.6936 | 0.2093 | — | 0.3998 |
| orbench | FraudDistill | 2000 | 0.8820 | 0.9974 | 0.7660 | 0.8804 | 0.0020 | 0.8839 | 0.7854 |
| orbench | Official Response Checker | 2000 | 0.9240 | 0.9133 | 0.9370 | 0.9240 | 0.0890 | — | 0.8483 |
| dna | FraudDistill | 5000 | 0.9612 | 0.9957 | 0.9264 | 0.9612 | 0.0040 | 0.9654 | 0.9246 |
| dna | Longformer-Harmful (LibrAI) | 5000 | 0.7226 | 0.8551 | 0.5360 | 0.7126 | 0.0908 | — | 0.4799 |
| aegis | FraudDistill | 813 | 0.8315 | 0.8836 | 0.7513 | 0.8297 | 0.0931 | 0.8303 | 0.6683 |
| aegis | NemoGuard (original) | 813 | 0.8819 | 0.8725 | 0.8858 | 0.8819 | 0.1217 | — | 0.7638 |