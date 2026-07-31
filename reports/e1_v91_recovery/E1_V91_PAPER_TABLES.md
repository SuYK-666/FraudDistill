# E1 V9.1 论文表格草稿

## Table 1: Frozen 自然行为

{
  "deepseek": {
    "n": 1516,
    "material_assist": 6,
    "rate": 0.00395778364116095,
    "wilson95": {
      "low": 0.0018151117578005397,
      "high": 0.008607987789033874
    },
    "defense_failure": 2,
    "actionable": 5,
    "trust_or_normalization": 6
  },
  "qwen": {
    "n": 1509,
    "material_assist": 6,
    "rate": 0.003976143141153081,
    "wilson95": {
      "low": 0.001823537472440496,
      "high": 0.00864779032488403
    },
    "defense_failure": 0,
    "actionable": 3,
    "trust_or_normalization": 6
  }
}

## Table 2/3: 输入边界面板

{
  "b1": [
    {
      "mode": "q-only",
      "n": 20,
      "tp": 0,
      "fp": 0,
      "tn": 10,
      "fn": 10,
      "macro_f1": 0.3333333333333333,
      "balanced_accuracy": 0.5,
      "accuracy": 0.5,
      "precision": 0.0,
      "recall": 0.0,
      "fpr": 0.0,
      "auroc": 0.5,
      "auprc": 0.5,
      "brier": 0.5,
      "ece": 0.5
    },
    {
      "mode": "y-only",
      "n": 20,
      "tp": 4,
      "fp": 0,
      "tn": 10,
      "fn": 6,
      "macro_f1": 0.6703296703296704,
      "balanced_accuracy": 0.7,
      "accuracy": 0.7,
      "precision": 1.0,
      "recall": 0.4,
      "fpr": 0.0,
      "auroc": 0.875,
      "auprc": 0.8666666666666667,
      "brier": 0.2765,
      "ece": 0.295
    },
    {
      "mode": "q+y",
      "n": 20,
      "tp": 5,
      "fp": 0,
      "tn": 10,
      "fn": 5,
      "macro_f1": 0.7333333333333334,
      "balanced_accuracy": 0.75,
      "accuracy": 0.75,
      "precision": 1.0,
      "recall": 0.5,
      "fpr": 0.0,
      "auroc": 1.0,
      "auprc": 1.0,
      "brier": 0.219,
      "ece": 0.25500000000000006
    }
  ],
  "b2": [
    {
      "mode": "q-only",
      "n": 10,
      "tp": 0,
      "fp": 0,
      "tn": 5,
      "fn": 5,
      "macro_f1": 0.3333333333333333,
      "balanced_accuracy": 0.5,
      "accuracy": 0.5,
      "precision": 0.0,
      "recall": 0.0,
      "fpr": 0.0,
      "auroc": 0.5,
      "auprc": 0.5,
      "brier": 0.5,
      "ece": 0.5
    },
    {
      "mode": "y-only",
      "n": 10,
      "tp": 3,
      "fp": 0,
      "tn": 5,
      "fn": 2,
      "macro_f1": 0.7916666666666665,
      "balanced_accuracy": 0.8,
      "accuracy": 0.8,
      "precision": 1.0,
      "recall": 0.6,
      "fpr": 0.0,
      "auroc": 0.98,
      "auprc": 0.9666666666666667,
      "brier": 0.17225000000000001,
      "ece": 0.19500000000000003
    },
    {
      "mode": "q+y",
      "n": 10,
      "tp": 4,
      "fp": 1,
      "tn": 4,
      "fn": 1,
      "macro_f1": 0.8000000000000002,
      "balanced_accuracy": 0.8,
      "accuracy": 0.8,
      "precision": 0.8,
      "recall": 0.8,
      "fpr": 0.2,
      "auroc": 0.86,
      "auprc": 0.7766666666666666,
      "brier": 0.17425,
      "ece": 0.185
    }
  ]
}
