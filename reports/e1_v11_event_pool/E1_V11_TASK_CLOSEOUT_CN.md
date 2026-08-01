# E1 V11 任务收尾报告

{
  "decision": {
    "decision": "E1_V11_STOP_INVALID",
    "a_gate": "PASS",
    "b1_gate": "EXPLORATORY",
    "b2_gate": "EXPLORATORY",
    "b3_gate": "EXPLORATORY",
    "c_gate": "EXPLORATORY",
    "hard_gates_ok": false,
    "b1_qy_macro_f1": 0.0,
    "b1_delta_qy_y": 0.0,
    "c_auprc_relative_improvement": 0
  },
  "a": {
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
  },
  "b": {
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
  },
  "c": {
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
  },
  "budget": {
    "qwen_cny": 35.261947382400024,
    "deepseek_cny": 12.416672996639997,
    "total_cny": 47.67862037904002,
    "over_hard_cap": false,
    "hard_caps": {
      "qwen": 48.0,
      "deepseek": 48.0
    }
  },
  "stats": {
    "gold_quality": {
      "expected_tasks": 1522,
      "valid_tasks": 806,
      "completion": 0.5295663600525624,
      "valid_schema": 0.5295663600525624,
      "paired_n": 224,
      "observed_agreement": 0.9732142857142857,
      "positive_agreement": 0.0,
      "kappa": -0.013574660633485284,
      "pabak": 0.9464285714285714,
      "gwet_ac1": 0.9724872057318321,
      "uncertain_rate": 0.7135348226018396,
      "cross_table": {
        "a_safe_b_safe": 218,
        "a_safe_b_risk": 3,
        "a_risk_b_safe": 3,
        "a_risk_b_risk": 0
      },
      "passed": false
    },
    "paired_stats": {
      "B1": [
        {
          "comparison": "q+y vs y-only",
          "point_delta": 0.0,
          "ci_low": 0.0,
          "ci_high": 0.0,
          "status": "NA_missing_mode",
          "holm_p": 1.0
        },
        {
          "comparison": "q+y vs wrong-q+y",
          "point_delta": 0.0,
          "ci_low": 0.0,
          "ci_high": 0.0,
          "status": "NA_missing_mode",
          "holm_p": 1.0
        }
      ],
      "B2": [
        {
          "comparison": "q+y vs y-only",
          "point_delta": 0.0,
          "ci_low": 0.0,
          "ci_high": 0.0,
          "status": "NA_missing_mode",
          "holm_p": 1.0
        },
        {
          "comparison": "q+y vs wrong-q+y",
          "point_delta": 0.0,
          "ci_low": 0.0,
          "ci_high": 0.0,
          "status": "NA_missing_mode",
          "holm_p": 1.0
        }
      ],
      "B3": [
        {
          "comparison": "q+y vs y-only",
          "point_delta": 0.0,
          "ci_low": 0.0,
          "ci_high": 0.0,
          "status": "NA_missing_mode",
          "holm_p": 1.0
        },
        {
          "comparison": "q+y vs wrong-q+y",
          "point_delta": 0.0,
          "ci_low": 0.0,
          "ci_high": 0.0,
          "status": "NA_missing_mode",
          "holm_p": 1.0
        }
      ]
    }
  },
  "bias": {
    "decision": {
      "decision": "E1_V11_STOP_INVALID",
      "a_gate": "PASS",
      "b1_gate": "EXPLORATORY",
      "b2_gate": "EXPLORATORY",
      "b3_gate": "EXPLORATORY",
      "c_gate": "EXPLORATORY",
      "hard_gates_ok": false,
      "b1_qy_macro_f1": 0.0,
      "b1_delta_qy_y": 0.0,
      "c_auprc_relative_improvement": 0
    },
    "known_biases": [
      "B1 是风险富集 case-control，不代表自然 prevalence。",
      "候选检索可能造成 spectrum bias，已通过 screen-negative audit 披露。",
      "Qwen-Max 同时参与 adjudication 和主 evaluator，因此 primary headline 只使用 consensus-core Gold。",
      "若 C positive_n 不足，C 只能解释为低基率趋势。"
    ],
    "budget": {
      "qwen_cny": 35.261947382400024,
      "deepseek_cny": 12.416672996639997,
      "total_cny": 47.67862037904002,
      "over_hard_cap": false,
      "hard_caps": {
        "qwen": 48.0,
        "deepseek": 48.0
      }
    }
  },
  "analysis_text": "本轮最终判定为 `E1_V11_STOP_INVALID`。A 层沿用 V10 冻结自然行为结果，不重新估计自然率。V11 Gold completion=0.5296，observed agreement=0.9732，positive agreement=0.0000。B1 为风险富集 case-control 面板，不能解释为自然发生率；B2/B3 用于机制辅助。C 层按低基率指标解释，AUPRC 与 FPR 优先于 accuracy。"
}
