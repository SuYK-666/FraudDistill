| Model | Optimizer Steps | Train s | Train h | Train Peak VRAM GB | Processed Train Tokens (est.) | Eval s | Eval Peak VRAM GB | Eval Input Tokens | P50 ms | P95 ms | Throughput Rows/s | Sec/1k Rows | Total GPU h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 140 | 186.6 | 0.052 | 6.4450 | 964073 | 87.96 | 1.5480 | 682476 | 24.33 | 24.86 | 142.03 | 7.04 | 0.0763 |
| 1.5B | 296 | 721.9 | 0.201 | 8.9520 | 2038326 | 198.21 | 4.2170 | 682476 | 28.97 | 29.33 | 55.22 | 18.11 | 0.2556 |
| 3B | 280 | 981.5 | 0.273 | 15.4600 | 1928146 | 283.1 | 7.5320 | 682476 | 36.27 | 37.47 | 30.95 | 32.31 | 0.3513 |
| 7B | 304 | 5891.27 | 1.636 | 15.6710 | 2093416 | 554.85 | 14.1520 | 682476 | 55.26 | 56.43 | 15.04 | 66.50 | 1.7906 |
| 0.6B (Qwen3) | 296 | 691.2 | 0.192 | 6.7370 | 2038326 | 188.7 | 2.2600 | 682476 | 44.38 | 45.10 | 29.56 | 33.83 | 0.2444 |
| 1.7B (Qwen3) | 220 | 787.7 | 0.219 | 8.0100 | 1514972 | 543.12 | 4.5970 | 682476 | 43.91 | 45.34 | 14.49 | 69.02 | 0.3697 |
| 4B (Qwen3) | 220 | 2810.3 | 0.781 | 17.0530 | 1514972 | 639.06 | 8.6580 | 682476 | 56.59 | 57.34 | 15.24 | 65.60 | 0.9582 |
| 0.8B (Qwen3.5) | 220 | 767.3 | 0.213 | 11.1990 | 1518663 | 423.13 | 2.5290 | 686504 | 65.30 | 68.40 | 74.25 | 13.47 | 0.3307 |
| 2B (Qwen3.5) | 296 | 1714.3 | 0.476 | 10.2120 | 2043292 | 264.14 | 5.0900 | 686504 | 67.45 | 68.19 | 12.70 | 78.72 | 0.5496 |
| 4B (Qwen3.5) | 180 | 5683.74 | 1.579 | 10.0000 | 1242542 | 143.3 | 9.2300 | 686504 | 70.65 | 72.91 | 23.52 | 42.52 | 1.6186 |

> Train s includes model reload, resume and CUDA-retry overhead; endpoint checkpoint saving is excluded. Qwen2.5-7B used 3 attempts/2 CUDA retries. Qwen3.5-4B used 9 attempts/8 retries and official evaluation batch=2. Qwen3.5-2B/4B use the training-best proxy checkpoint because full candidate selection was unavailable. Qwen3.5 latency/throughput use unoptimized reference kernels.
