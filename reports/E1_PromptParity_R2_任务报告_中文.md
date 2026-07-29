# FraudDistill E1 Prompt-Parity R2 任务报告

- 协议：`E1-API-PromptParity-R2`
- 最新阶段：`anchor_api`
- 最新判定：`E1_ANCHOR_API_STOP_NEAR`
- Git 提交：`09ad94d8947b497355b6798591e0c098a0e28ec8`

## 本轮关键修订

1. 已归档上一轮 `E1_ANCHOR_QWENMAX_POSITIVE_BUT_CONFOUNDED_R0`。
2. 三种输入模式使用同一个 q->y 判定目标和同一套 Rubric，差异只来自字段隐藏。
3. 输出 schema 改为 `p_unsafe`，并强制 label-score-type 语义一致。
4. 缓存绑定 provider/model/base URL/temperature/max_tokens/thinking/prompt/schema/normalizer revision。
5. 本轮 API 并发配置为 80；若完整率不足则 Gate 停止。

## 阶段结果

### anchor2400
- 判定：`E1_ANCHOR_READY`

### preflight_api
- 判定：`E1_PREFLIGHT_READY`
- API 完整率下界：`1.0`

### anchor_api
- 判定：`E1_ANCHOR_API_STOP_NEAR`

| Input | Macro-F1 | Recall | Precision | FPR | Hard-safe FPR | AUPRC | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| q_only | 0.5860 | 0.7083 | 0.5743 | 0.5250 | 0.5250 | 0.5811 | 0.6118 |
| y_only | 0.8947 | 0.7917 | 1.0000 | 0.0000 | 0.0000 | 0.9767 | 0.9768 |
| q_y | 0.9033 | 0.8083 | 1.0000 | 0.0000 | 0.0000 | 0.9221 | 0.9210 |

| Check | Pass |
|---|---:|
| `api_success_rate` | True |
| `q_only_min` | True |
| `q_only_max` | True |
| `y_only_min` | True |
| `q_y_min` | True |
| `q_y_minus_y` | False |
| `y_minus_q` | True |
| `q_y_minus_y_ci_lower` | False |
| `q_y_recall` | True |
| `q_y_precision` | True |
| `q_y_fpr` | True |
| `hard_safe_fpr` | True |
| `q_y_auprc` | True |

## 结果分析

本轮最关键的工程目标已经达成：三种输入模式使用同一个 q->y 判定目标，`p_unsafe` 的语义被固定为 unsafe 概率，AUPRC/AUROC 不再出现上一轮“F1 很高但排序指标低于随机基线”的异常。Smoke 阶段 120/120 成功，Anchor 阶段 1440/1440 成功，说明固定 Qwen 快照、80 并发、thinking 关闭、缓存指纹和 schema 校验可以稳定运行。

从指标形态看，R2 明显改善了绝对性能：q+y Macro-F1 达到 0.9033，Recall 0.8083，Precision 1.0000，FPR 0.0000，AUPRC 0.9221，已经满足除“相对 y-only 增益”外的主要 Anchor 阻断门槛。相比 R0 的 q+y 0.8775，本轮 q+y 绝对值提升约 +2.58pp，并且分数语义更可信。

但本轮仍不能进入 Gold-8K/full formal，原因是 y-only 同时被 prompt-parity 和 `p_unsafe` schema 显著抬高到 0.8947，导致 q+y 相对 y-only 只提升 +0.0086，低于预注册的 +0.030；cluster bootstrap 的 q+y-y_only CI 下界为 -0.0065，没有通过 >0.010 的硬门。也就是说，本轮证明“API Judge 在 q+y 下可以达到高绝对性能”，但尚未证明“q 与 y 的互补信息带来足够稳定的额外收益”。

因此，当前判定为 `E1_ANCHOR_API_STOP_NEAR` 是严格且必要的：不能降低门槛，也不能继续 full；下一步只能在 model_dev/协议层继续审计 y-only 过强的原因，例如字段隐藏是否仍泄漏上下文、y-only 是否能从回答模板直接推断 gold、Anchor 是否包含过多 y 本身已足够显式的 unsafe。不得查看 formal_test 错误后调 prompt，也不得删除样本或修改 gold。

## 数据与审计文件

- 数据目录：`C:\Users\18201\Desktop\FraudDistill\data\prepared\e1_restored_v1plus_final`
- 输出目录：`C:\Users\18201\Desktop\FraudDistill\outputs\e1_restored_v1plus_final`
- 原始预测：各阶段 `predictions/`。
- API 审计：`E1_API_COMPLETENESS.json`、`E1_API_MODEL_AUDIT.csv`、`E1_API_USAGE.csv`、`E1_API_FAILURE_AUDIT.json`、`E1_CACHE_AUDIT.csv`。
- Anchor 输入 hash：`anchor_api/E1_ANCHOR480_INPUT_HASH.json`。
