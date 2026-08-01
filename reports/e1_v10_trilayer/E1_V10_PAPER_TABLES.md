# E1 V10 论文表格草稿

## E1-A

{
  "n": 3080,
  "by_model": {
    "deepseek": {
      "n": 1540,
      "positive": 15,
      "rate": 0.00974025974025974,
      "wilson95": {
        "low": 0.005911560103550392,
        "high": 0.01600873399617203
      },
      "lower_positive": 6,
      "upper_positive": 30
    },
    "qwen": {
      "n": 1540,
      "positive": 11,
      "rate": 0.007142857142857143,
      "wilson95": {
        "low": 0.003993115996839215,
        "high": 0.012745298866325448
      },
      "lower_positive": 6,
      "upper_positive": 36
    }
  },
  "severity_distribution": {
    "0": 3052,
    "1": 2,
    "3": 7,
    "2": 19
  },
  "judge_cross_table": {
    "a_safe_b_safe": 3014,
    "a_safe_b_risk": 3,
    "a_risk_b_safe": 51,
    "a_risk_b_risk": 12
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

## E1-B

{
  "broad": {
    "pair_n": 8,
    "row_n": 16,
    "metrics_by_mode": [
      {
        "mode": "q-only",
        "n": 16,
        "tp": 0,
        "fp": 0,
        "tn": 8,
        "fn": 8,
        "macro_f1": 0.3333333333333333,
        "balanced_accuracy": 0.5,
        "accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.5,
        "auprc": 0.5,
        "brier": 0.488125,
        "ece": 0.4875
      },
      {
        "mode": "y-only",
        "n": 16,
        "tp": 6,
        "fp": 0,
        "tn": 8,
        "fn": 2,
        "macro_f1": 0.873015873015873,
        "balanced_accuracy": 0.875,
        "accuracy": 0.875,
        "precision": 1.0,
        "recall": 0.75,
        "fpr": 0.0,
        "auroc": 0.90625,
        "auprc": 0.8928571428571428,
        "brier": 0.17863749999999998,
        "ece": 0.19624999999999998
      },
      {
        "mode": "wrong-q+y",
        "n": 16,
        "tp": 6,
        "fp": 2,
        "tn": 6,
        "fn": 2,
        "macro_f1": 0.75,
        "balanced_accuracy": 0.75,
        "accuracy": 0.75,
        "precision": 0.75,
        "recall": 0.75,
        "fpr": 0.25,
        "auroc": 0.8125,
        "auprc": 0.8229166666666666,
        "brier": 0.3372875,
        "ece": 0.35124999999999995
      },
      {
        "mode": "q+y",
        "n": 16,
        "tp": 7,
        "fp": 0,
        "tn": 8,
        "fn": 1,
        "macro_f1": 0.9372549019607843,
        "balanced_accuracy": 0.9375,
        "accuracy": 0.9375,
        "precision": 1.0,
        "recall": 0.875,
        "fpr": 0.0,
        "auroc": 0.9140625,
        "auprc": 0.9375,
        "brier": 0.12116875,
        "ece": 0.143125
      }
    ],
    "delta_qy_y": 0.06423902894491129,
    "delta_qy_wrong": 0.18725490196078431,
    "q_only_pair_accuracy": 0.5
  },
  "context": {
    "pair_n": 6,
    "row_n": 12,
    "metrics_by_mode": [
      {
        "mode": "q-only",
        "n": 12,
        "tp": 0,
        "fp": 0,
        "tn": 6,
        "fn": 6,
        "macro_f1": 0.3333333333333333,
        "balanced_accuracy": 0.5,
        "accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.5,
        "auprc": 0.5,
        "brier": 0.48416666666666663,
        "ece": 0.48333333333333334
      },
      {
        "mode": "y-only",
        "n": 12,
        "tp": 6,
        "fp": 0,
        "tn": 6,
        "fn": 0,
        "macro_f1": 1.0,
        "balanced_accuracy": 1.0,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "fpr": 0.0,
        "auroc": 1.0,
        "auprc": 1.0,
        "brier": 0.07805,
        "ece": 0.10166666666666667
      },
      {
        "mode": "wrong-q+y",
        "n": 12,
        "tp": 6,
        "fp": 2,
        "tn": 4,
        "fn": 0,
        "macro_f1": 0.8285714285714285,
        "balanced_accuracy": 0.8333333333333333,
        "accuracy": 0.8333333333333334,
        "precision": 0.75,
        "recall": 1.0,
        "fpr": 0.3333333333333333,
        "auroc": 0.9166666666666666,
        "auprc": 0.875,
        "brier": 0.36128333333333335,
        "ece": 0.3816666666666666
      },
      {
        "mode": "q+y",
        "n": 12,
        "tp": 6,
        "fp": 1,
        "tn": 5,
        "fn": 0,
        "macro_f1": 0.916083916083916,
        "balanced_accuracy": 0.9166666666666667,
        "accuracy": 0.9166666666666666,
        "precision": 0.8571428571428571,
        "recall": 1.0,
        "fpr": 0.16666666666666666,
        "auroc": 1.0,
        "auprc": 1.0,
        "brier": 0.004858333333333336,
        "ece": 0.04916666666666668
      }
    ],
    "delta_qy_y": -0.08391608391608396,
    "delta_qy_wrong": 0.08751248751248752,
    "q_only_pair_accuracy": 0.5
  }
}

## E1-C

{
  "pair_n": 593,
  "row_n": 593,
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
      "n": 593,
      "tp": 3,
      "fp": 16,
      "tn": 574,
      "fn": 0,
      "macro_f1": 0.6294907841299593,
      "balanced_accuracy": 0.9864406779661017,
      "accuracy": 0.9730185497470489,
      "precision": 0.15789473684210525,
      "recall": 1.0,
      "fpr": 0.02711864406779661,
      "auroc": 0.9943502824858758,
      "auprc": 0.49707602339181284,
      "brier": 0.0031517706576728497,
      "ece": 0.00236087689713322
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
      "n": 593,
      "tp": 3,
      "fp": 12,
      "tn": 578,
      "fn": 0,
      "macro_f1": 0.6615296803652968,
      "balanced_accuracy": 0.9898305084745762,
      "accuracy": 0.9797639123102867,
      "precision": 0.2,
      "recall": 1.0,
      "fpr": 0.020338983050847456,
      "auroc": 0.9966101694915255,
      "auprc": 0.7333333333333333,
      "brier": 0.0015863406408094434,
      "ece": 0.0004890387858347386
    }
  ],
  "delta_qy_y": 0.03203889623533751,
  "delta_qy_wrong": 0.6615296803652968,
  "q_only_pair_accuracy": 0.0
}
