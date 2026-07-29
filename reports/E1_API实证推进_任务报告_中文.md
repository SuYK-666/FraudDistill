# FraudDistill E1 API 实证推进报告

- 协议：`E1-API-Empirical-Restore-v1`
- 最新阶段：`anchor_api`
- 最新判定：`E1_ANCHOR_API_STOP`
- Git 提交：`5a157baafa395df0650486c80e76fc8b0b7cb830`

## 本轮关键修订

1. 将上一轮 CPU 结果定位为 `E1_CPU_DIAGNOSTIC_STOP_R0`，不再作为 q->y 假设的最终否定。
2. 新增真实 API 预检与 Anchor API Gate，失败行不回填，所有 API 响应保留模型、usage、request id、finish reason 和响应哈希。
3. API Judge 采用追加式 JSONL 缓存，重复运行只补缺失成功样本，避免重复付费调用。
4. Anchor API 未通过前不进入 full manifest、formal CPU 或全量 API。

## 阶段结果

### preflight_api
- 判定：`E1_PREFLIGHT_READY`

### anchor2400
- 判定：`E1_ANCHOR_READY`

### anchor_api
- 判定：`E1_ANCHOR_API_STOP`

| Input | Macro-F1 | Recall | Precision | FPR | Hard-safe FPR | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| q_only | 0.5458 | 0.7542 | 0.5452 | 0.6292 | 0.6292 | 0.5590 |
| y_only | 0.8149 | 0.6417 | 1.0000 | 0.0000 | 0.0000 | 0.4355 |
| q_y | 0.8775 | 0.7625 | 0.9946 | 0.0042 | 0.0042 | 0.4154 |

| Check | Pass |
|---|---:|
| `api_success_rate` | True |
| `q_only_min` | True |
| `q_only_max` | True |
| `y_only_min` | True |
| `q_y_min` | False |
| `q_y_minus_y` | True |
| `y_minus_q` | True |
| `hard_safe_fpr` | True |
| `q_y_minus_y_ci_positive` | True |

## DeepSeek 预检对照

- 模型：`deepseek-v4-pro`
- 样本数：`40`


| Input | Macro-F1 | Recall | Precision | FPR | Hard-safe FPR | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| q_only | 0.5807 | 0.3529 | 0.6000 | 0.1739 | 0.0000 | 0.4785 |
| y_only | 0.6238 | 0.2941 | 1.0000 | 0.0000 | 0.0000 | 0.4240 |
| q_y | 0.6238 | 0.2941 | 1.0000 | 0.0000 | 0.0000 | 0.4489 |

结论：DeepSeek 40 样本预检中 q+y 未优于 y-only，且 Macro-F1 明显低于 Qwen Anchor，因此本轮未继续执行 DeepSeek 480 全 Anchor。

## 数据与原始结果位置

- 数据目录：`C:\Users\18201\Desktop\FraudDistill\data\prepared\e1_restored_v1plus_final`
- 输出目录：`C:\Users\18201\Desktop\FraudDistill\outputs\e1_restored_v1plus_final`
- 预测原始 JSONL 保留在各阶段 `predictions/` 子目录。
- 旧输出已归档到 `archive/`，当前正式输出目录只保留本轮数据。