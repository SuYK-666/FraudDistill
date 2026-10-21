| Model | Params (B) | Core MF1 | Online MF1 | Recall | FPR | Hard-safe FPR | P50/P95 ms | Rows/s | Sec/1k | Eval input tok | Peak VRAM GB | Train h | GPU h | Est. CNY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct student | 0.5 | 0.7873 | 0.6532 | 0.7500 | 0.4375 | 0.2111 | 24.33/24.86 | 142.03 | 7.04 | 682476 | 1.55 | 0.05 | 0.076 | 0.00 |
| Qwen2.5-1.5B-Instruct student | 1.5 | 0.9128 | 0.7656 | 0.7500 | 0.2188 | 0.0667 | 28.97/29.33 | 55.22 | 18.11 | 682476 | 4.22 | 0.20 | 0.256 | 0.00 |
| Qwen2.5-3B-Instruct student | 3.0 | 0.8828 | 0.7013 | 0.8906 | 0.4688 | 0.1889 | 36.27/37.47 | 30.95 | 32.31 | 682476 | 7.53 | 0.27 | 0.351 | 0.00 |
| Qwen2.5-7B-Instruct student | 7.0 | 0.8794 | 0.6269 | 0.4844 | 0.2135 | 0.1333 | 55.26/56.43 | 15.04 | 66.50 | 682476 | 14.15 | 1.64 | 1.791 | 0.00 |
| Qwen3-0.6B student | 0.6 | 0.9239 | 0.7713 | 0.8698 | 0.3229 | 0.1056 | 44.38/45.10 | 29.56 | 33.83 | 682476 | 2.26 | 0.19 | 0.244 | 0.00 |
| Qwen3-1.7B student | 1.7 | 0.8785 | 0.7606 | 0.8646 | 0.3385 | 0.1722 | 43.91/45.34 | 14.49 | 69.02 | 682476 | 4.60 | 0.22 | 0.370 | 0.00 |
| Qwen3-4B student | 4.0 | 0.9175 | 0.7438 | 0.8073 | 0.3177 | 0.0722 | 56.59/57.34 | 15.24 | 65.60 | 682476 | 8.66 | 0.78 | 0.958 | 0.00 |
| Qwen3.5-0.8B student | 0.8 | 0.9262 | 0.6154 | 0.3646 | 0.0729 | 0.0389 | 65.30/68.40 | 74.25 | 13.47 | 686504 | 2.53 | 0.21 | 0.331 | 0.00 |
| Qwen3.5-2B student | 2.0 | 0.9093 | 0.6819 | 0.5885 | 0.2188 | 0.0611 | 67.45/68.19 | 12.70 | 78.72 | 686504 | 5.09 | 0.48 | 0.550 | 0.00 |
| Qwen3.5-4B student | 4.0 | 0.9109 | 0.6094 | 0.3490 | 0.0625 | 0.0222 | 70.65/72.91 | 23.52 | 42.52 | 686504 | 9.23 | 1.58 | 1.619 | 0.00 |
| Original DeepSeek-R1-Distill-Qwen-1.5B student | 1.5 | 0.9135 | 0.7266 | 0.7292 | 0.2760 | 0.1444 | - | - | - | - | - | 7.96 | - | - |
| Original 1.5B + 20% real E6 audit | 1.5 | 0.9135 | 0.7812 | 0.7812 | 0.2188 | 0.1444 | - | - | - | - | - | 7.96 | - | - |

> Train h is end-to-end wall time including model reload, resume and CUDA-retry overhead; endpoint checkpoint saving is excluded. GPU h counts the newly rented GPU training run and its frozen evaluation; the reused DeepSeek student was trained earlier on CPU. Classifiers emit zero output tokens.
>
> Qwen2.5-7B used 3 training attempts and 2 CUDA retries. Qwen3.5-4B used 9 attempts and 8 CUDA retries, with official evaluation batch=2 after a cuBLAS error. Qwen3.5-2B/4B use the training-best proxy checkpoint because full candidate selection was unavailable. Qwen3.5 timing uses unoptimized reference kernels, so its deployment cost is conservative.
