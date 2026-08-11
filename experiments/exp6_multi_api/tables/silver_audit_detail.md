# Silver 审核明细（每模型 Random-30 + Boundary-10）

| Slot | Model | Split | N | unsafe | safe | unresolved |
|---|---|---|---:|---:|---:|---:|
| M1 | Qwen Flash | random | 30 | 2 | 28 | 0 |
| M1 | Qwen Flash | boundary | 10 | 1 | 9 | 0 |
| M2 | Qwen Plus | random | 30 | 1 | 29 | 0 |
| M2 | Qwen Plus | boundary | 10 | 1 | 9 | 0 |
| M3 | DeepSeek Flash | random | 30 | 1 | 29 | 0 |
| M3 | DeepSeek Flash | boundary | 10 | 0 | 10 | 0 |
| M4 | DeepSeek Pro | random | 30 | 1 | 29 | 0 |
| M4 | DeepSeek Pro | boundary | 10 | 1 | 9 | 0 |
| M5 | GLM Flash | random | 30 | 5 | 25 | 0 |
| M5 | GLM Flash | boundary | 10 | 1 | 9 | 0 |
| M6 | Kimi | random | 30 | 3 | 27 | 0 |
| M6 | Kimi | boundary | 10 | 2 | 8 | 0 |

## 主要行为标签分布（Random audit，resolved）
| behavior | n |
|---|---:|
| clean_refusal | 73 |
| safe_redirection | 42 |
| benign_help | 35 |
| fraud_assistance | 18 |
| partial_leakage | 12 |