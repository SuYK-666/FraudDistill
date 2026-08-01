# E1 V11 Executive Report

## 首屏结论

- 最终 decision code：`E1_V11_STOP_INVALID`
- A：`PASS`；B1：`EXPLORATORY`；B2：`EXPLORATORY`；B3：`EXPLORATORY`；C：`EXPLORATORY`
- 新增预算：Qwen 35.2619 元；DeepSeek 12.4167 元；总计 47.6786 元。
- 论文口径：A 自然行为冻结；B1 为风险富集 case-control；B2/B3 为机制辅助；C 为低基率压力 holdout。

## E1-A NATURAL

```json
{
  "gate": "PASS",
  "source": "V10 frozen natural behavior",
  "n": 3080,
  "by_model": {
    "deepseek": {
      "events": 15,
      "n": 1540,
      "rate": 0.00974025974025974,
      "wilson95": {
        "low": 0.005911560103550392,
        "high": 0.01600873399617203
      },
      "events_per_1k": 9.74025974025974,
      "lower_positive": 6,
      "upper_positive": 30
    },
    "qwen": {
      "events": 11,
      "n": 1540,
      "rate": 0.007142857142857143,
      "wilson95": {
        "low": 0.003993115996839215,
        "high": 0.012745298866325448
      },
      "events_per_1k": 7.142857142857143,
      "lower_positive": 6,
      "upper_positive": 36
    }
  },
  "quality": {
    "expected": 6164,
    "paired_response_count": 3080,
    "valid_json": 1.0,
    "observed_agreement": 0.9824675324675325,
    "positive_agreement": 0.3076923076923077,
    "kappa": 0.30220264317181084,
    "pabak": 0.964935064935065,
    "gwet_ac1": 0.9820179081395346,
    "cross_table": {
      "a_safe_b_safe": 3014,
      "a_safe_b_risk": 3,
      "a_risk_b_safe": 51,
      "a_risk_b_risk": 12
    },
    "adjudication_completion": 1.0,
    "discord_total": 54,
    "discord_done": 54,
    "passed": true
  }
}
```

## E1-B ENRICHED CASE-CONTROL

```json
{
  "B1": {
    "pair_n": 0,
    "row_n": 0,
    "positive_n": 0,
    "metrics_by_mode": [
      {
        "mode": "q-only",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      },
      {
        "mode": "y-only",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      },
      {
        "mode": "wrong-q+y",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      },
      {
        "mode": "q+y",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      }
    ],
    "delta_qy_y": 0.0,
    "delta_qy_wrong": 0.0,
    "q_only_pair_accuracy": 0.0
  },
  "B2": {
    "pair_n": 0,
    "row_n": 0,
    "positive_n": 0,
    "metrics_by_mode": [
      {
        "mode": "q-only",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      },
      {
        "mode": "y-only",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      },
      {
        "mode": "wrong-q+y",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      },
      {
        "mode": "q+y",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      }
    ],
    "delta_qy_y": 0.0,
    "delta_qy_wrong": 0.0,
    "q_only_pair_accuracy": 0.0
  },
  "B3": {
    "pair_n": 0,
    "row_n": 0,
    "positive_n": 0,
    "metrics_by_mode": [
      {
        "mode": "q-only",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      },
      {
        "mode": "y-only",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      },
      {
        "mode": "wrong-q+y",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      },
      {
        "mode": "q+y",
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0
      }
    ],
    "delta_qy_y": 0.0,
    "delta_qy_wrong": 0.0,
    "q_only_pair_accuracy": 0.0
  }
}
```

## E1-C PRESSURE HOLDOUT

```json
{
  "pair_n": 218,
  "row_n": 218,
  "positive_n": 0,
  "metrics_by_mode": [
    {
      "mode": "q-only",
      "n": 0,
      "tp": 0,
      "fp": 0,
      "tn": 0,
      "fn": 0,
      "macro_f1": 0.0,
      "balanced_accuracy": 0.0,
      "accuracy": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "fpr": 0.0,
      "auroc": 0.0,
      "auprc": 0.0,
      "brier": 0.0,
      "ece": 0.0,
      "prevalence": 0.0,
      "auprc_lift": 0,
      "alerts_per_1k": 0
    },
    {
      "mode": "y-only",
      "n": 218,
      "tp": 0,
      "fp": 213,
      "tn": 5,
      "fn": 0,
      "macro_f1": 0.022421524663677132,
      "balanced_accuracy": 0.011467889908256881,
      "accuracy": 0.022935779816513763,
      "precision": 0.0,
      "recall": 0.0,
      "fpr": 0.9770642201834863,
      "auroc": 0.0,
      "auprc": 0.0,
      "brier": 0.025876146788990827,
      "ece": 0.06119266055045872,
      "prevalence": 0.0,
      "auprc_lift": 0,
      "alerts_per_1k": 977.0642201834862
    },
    {
      "mode": "wrong-q+y",
      "n": 0,
      "tp": 0,
      "fp": 0,
      "tn": 0,
      "fn": 0,
      "macro_f1": 0.0,
      "balanced_accuracy": 0.0,
      "accuracy": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "fpr": 0.0,
      "auroc": 0.0,
      "auprc": 0.0,
      "brier": 0.0,
      "ece": 0.0,
      "prevalence": 0.0,
      "auprc_lift": 0,
      "alerts_per_1k": 0
    },
    {
      "mode": "q+y",
      "n": 218,
      "tp": 0,
      "fp": 203,
      "tn": 15,
      "fn": 0,
      "macro_f1": 0.06437768240343349,
      "balanced_accuracy": 0.034403669724770644,
      "accuracy": 0.06880733944954129,
      "precision": 0.0,
      "recall": 0.0,
      "fpr": 0.9311926605504587,
      "auroc": 0.0,
      "auprc": 0.0,
      "brier": 0.07965045871559633,
      "ece": 0.11587155963302753,
      "prevalence": 0.0,
      "auprc_lift": 0,
      "alerts_per_1k": 931.1926605504588
    }
  ],
  "delta_qy_y": 0.041956157739756354,
  "delta_qy_wrong": 0.06437768240343349,
  "q_only_pair_accuracy": 0.0,
  "prevalence": 0.0,
  "capacity_gate": "EXPLORATORY"
}
```
