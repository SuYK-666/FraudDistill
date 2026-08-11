# 主表2：Student 跨模型检测性能（vs LLM-Silver）

| Scope | Audit split | N | Precision | Recall | F1-unsafe | Macro-F1 | FPR | MCC | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All models / Random | Random | 180 | 0.091 | 0.154 | 0.114 | 0.509 | 0.120 | 0.027 | 0.695 | 0.123 |
| Qwen family / Random | Random | 60 | 0.143 | 0.333 | 0.200 | 0.564 | 0.105 | 0.155 | 0.772 | 0.254 |
| DeepSeek family / Random | Random | 60 | 0.000 | 0.000 | 0.000 | 0.478 | 0.052 | -0.043 | 0.737 | 0.087 |
| GLM + Kimi / Random | Random | 60 | 0.083 | 0.125 | 0.100 | 0.460 | 0.212 | -0.074 | 0.555 | 0.156 |
| Qwen Flash / Random | Random | 30 | 0.250 | 0.500 | 0.333 | 0.630 | 0.107 | 0.288 | 0.911 | 0.417 |
| Qwen Plus / Random | Random | 30 | 0.000 | 0.000 | 0.000 | 0.464 | 0.103 | -0.062 | 0.345 | 0.050 |
| DeepSeek Flash / Random | Random | 30 | 0.000 | 0.000 | 0.000 | 0.474 | 0.069 | -0.050 | 0.759 | 0.125 |
| DeepSeek Pro / Random | Random | 30 | 0.000 | 0.000 | 0.000 | 0.483 | 0.034 | -0.034 | 0.672 | 0.091 |
| GLM Flash / Random | Random | 30 | 0.000 | 0.000 | 0.000 | 0.388 | 0.240 | -0.224 | 0.568 | 0.209 |
| Kimi / Random | Random | 30 | 0.167 | 0.333 | 0.222 | 0.542 | 0.185 | 0.111 | 0.494 | 0.137 |
| All models / Boundary stress | Boundary-10 | 60 | 0.188 | 1.000 | 0.316 | 0.499 | 0.481 | 0.312 | 0.750 | 0.216 |

注：Random-30 与 Boundary-10 不混合为普通 IID 测试集；Boundary stress 为刻意困难的选择性样本，仅作诊断。