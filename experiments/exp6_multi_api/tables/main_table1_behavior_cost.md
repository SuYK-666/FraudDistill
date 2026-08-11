# 主表1：目标模型行为与成本（Student 全量 + Random Silver 抽样）

| Provider | Model | N(成功/请求) | Student-UAR ↓ | Mean risk | P90 risk | Silver-UAR ↓ | PLR ↓ | CRR ↑ | ORR ↓ | Coverage | Cost (¥) | P50 latency(s) | Truncation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen | Qwen Flash (qwen-flash) | 200/200 | 11.0% | 0.149 | 0.539 | 12.5% | 12.5% | 62.5% | 0.0% | 100.0% | 0.1142 | 4.0 | 33.0% |
| qwen | Qwen Plus (qwen-plus) | 200/200 | 8.0% | 0.131 | 0.432 | 6.7% | 0.0% | 80.0% | 0.0% | 100.0% | 0.1322 | 5.8 | 33.0% |
| deepseek | DeepSeek Flash (deepseek-v4-flash) | 200/200 | 12.0% | 0.132 | 0.539 | 9.1% | 0.0% | 81.8% | 0.0% | 100.0% | 0.1230 | 3.5 | 32.5% |
| deepseek | DeepSeek Pro (deepseek-v4-pro) | 200/200 | 12.0% | 0.131 | 0.548 | 0.0% | 0.0% | 81.2% | 0.0% | 100.0% | 0.4018 | 7.0 | 32.5% |
| glm_kimi | GLM Flash (glm-4-flash) | 199/200 | 36.0% | 0.254 | 0.904 | 21.4% | 42.9% | 21.4% | 0.0% | 99.5% | 0.0663 | 14.0 | 32.7% |
| glm_kimi | Kimi (moonshot-v1-8k) | 200/200 | 18.0% | 0.183 | 0.683 | 11.8% | 11.8% | 29.4% | 0.0% | 100.0% | 0.1525 | 5.6 | 31.5% |

注：Student-UAR 基于全部 should-refuse q（每模型约 100 条）；Silver-UAR/PLR/CRR/ORR 基于 Random-30 审核子集（每模型 30 条，should-refuse 约 15 条），为抽样估计，正文附 Wilson 95% CI。