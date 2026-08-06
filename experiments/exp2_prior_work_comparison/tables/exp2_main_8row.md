| benchmark | method | n | n_positive | positive_rate | accuracy | precision | recall | macro_f1 | fpr | auprc | mcc | balanced_accuracy | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fraud-R1 Content | Baseline | 3000 | 61 | 0.0203 | 0.78 | 0.0811 | 0.9508 | 0.463 | 0.2235 | 0.0941 | 0.2409 | 0.8636 | - |
| Fraud-R1 Content | FraudDistill Evidence MAT | 3000 | 61 | 0.0203 | 0.8527 | 0.1152 | 0.9344 | 0.528 | 0.149 | 0.4752 | 0.2986 | 0.8927 | FAIL |
| OR-Bench | Baseline | 3000 | 40 | 0.0133 | 0.6943 | 0.028 | 0.65 | 0.3743 | 0.3051 | 0.0514 | 0.0856 | 0.6725 | - |
| OR-Bench | FraudDistill Evidence MAT | 3000 | 40 | 0.0133 | 0.9867 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0514 | 0.0 | 0.5 | FAIL |
| Do-Not-Answer | Baseline | 5634 | 193 | 0.0343 | 0.995 | 0.9661 | 0.886 | 0.9616 | 0.0011 | 0.9657 | 0.9227 | 0.9425 | - |
| Do-Not-Answer | FraudDistill Evidence MAT | 5634 | 193 | 0.0343 | 0.9666 | 0.6667 | 0.0518 | 0.5476 | 0.0009 | 0.12 | 0.1796 | 0.5254 | FAIL |
| Aegis response | Baseline | 813 | 394 | 0.4846 | 0.8819 | 0.8725 | 0.8858 | 0.8787 | 0.1217 | 0.4152 | 0.7638 | 0.882 | - |
| Aegis response | FraudDistill Evidence MAT | 813 | 394 | 0.4846 | 0.7528 | 0.8814 | 0.566 | 0.8089 | 0.0716 | 0.7718 | 0.5337 | 0.7472 | FAIL |
