# FraudDistill E1 v7 双层自然评测与输入诊断任务报告

- 协议：`E1-DUAL-NATURAL-DIAGNOSTIC-v1.0`
- 最新阶段：`pilot-decide`
- 最新判定：`E1_V7_PILOT_STOP`
- Git commit：`73388c5465b0ba3be1f8144819b295b4471642d1`

## 执行说明

本轮不再沿用 R4.1 的 Bctx/Panel C 容量假设，而是按新协议先完成 P0 canonical/cache 修复，再执行 800-call 盲化 Pilot。Pilot 阶段只计算完成率、有效率、双参考三状态一致率、风险事件容量、语言/类别覆盖和 exact-q mixed-outcome 容量，不计算 q-only/y-only/q+y F1。

## p0

- 判定：`E1_V7_G0_PASS`
- 用时：`0.803` 秒

| Gate | Pass |
|---|---:|
| `canonical_count_2141` | True |
| `old_4282_mapping_explainable` | True |
| `q_hash_missing_zero` | True |
| `pilot_200` | True |
| `pilot_only_not_anchor` | True |
| `secret_scan` | True |
| `prompt_parity` | True |

## pilot-generate

- 判定：`E1_V7_PILOT_GENERATE_PASS`
- 用时：`334.876` 秒

```json
{
  "expected_rows": 800,
  "rows": 800,
  "all_expected_present": true,
  "status_counts": {
    "ok": 800
  },
  "completion_rate_by_model": {
    "deepseek": 1.0,
    "qwen": 1.0
  }
}
```

## pilot-label

- 判定：`E1_V7_PILOT_LABEL_PASS`
- 用时：`34.894` 秒

```json
{
  "expected_label_rows": 1600,
  "label_rows": 1600,
  "all_expected_present": true,
  "label_status_counts": {
    "ok": 1600
  },
  "label_state_counts": {
    "SUCCESS": 1538,
    "FAILURE": 58,
    "CONTINUE": 4
  },
  "consensus_state_counts": {
    "SUCCESS": 761,
    "FAILURE": 22,
    "UNCERTAIN": 17
  }
}
```

## pilot-decide

- 判定：`E1_V7_PILOT_STOP`
- 用时：`0.121` 秒

### Pilot 容量表

| Model | Completion | Valid | Agreement | Uncertain/Invalid | Risk events | EN risk | ZH risk | Categories | Mixed groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek | 1.0000 | 1.0000 | 0.9700 | 0.0300 | 20 | 8 | 12 | 2 | 5 |
| qwen | 1.0000 | 1.0000 | 0.9875 | 0.0125 | 2 | 0 | 2 | 1 | 0 |

## 数据位置

- 数据目录：`C:\Users\18201\Desktop\FraudDistill\data\prepared\e1_dual_v7`
- 输出目录：`C:\Users\18201\Desktop\FraudDistill\outputs\e1_dual_v7`
- 原始 API response cache 使用 append-only JSONL，本地保留，不提交 GitHub。