# E1 V10 失败与偏差审计

{
  "decision": {
    "decision": "E1_V10_BEHAVIOR_PASS_MECHANISM_EXPLORATORY",
    "p0": {
      "decision": "P0_PASS",
      "checks": {
        "eligible_provenance_complete": true,
        "hash_complete": true,
        "duplicate_response_zero": true,
        "b_c_overlap_zero": true,
        "secret_scan": true
      },
      "dedup": {
        "duplicate_response_id": 0,
        "duplicate_q_y_model": 0,
        "rows": 7482
      },
      "provenance": {
        "eligible_rows": 7360,
        "eligible_provenance_complete_rate": 1.0,
        "hash_complete_rate": 1.0
      },
      "split": {
        "b_c_canonical_overlap": 0
      }
    },
    "a": {
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
    },
    "b": {
      "broad_pairs": 8,
      "context_pairs": 6,
      "broad_qy": 0.9372549019607843,
      "context_qy": 0.916083916083916
    },
    "c": {
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
    },
    "budget": {
      "qwen_cny": 27.572333486399955,
      "deepseek_cny": 3.439632960000003,
      "total_cny": 31.01196644639996,
      "over_hard_cap": false,
      "hard_caps": {
        "qwen": 48.0,
        "deepseek": 48.0
      }
    }
  },
  "p0": {
    "decision": "P0_PASS",
    "checks": {
      "eligible_provenance_complete": true,
      "hash_complete": true,
      "duplicate_response_zero": true,
      "b_c_overlap_zero": true,
      "secret_scan": true
    },
    "dedup": {
      "duplicate_response_id": 0,
      "duplicate_q_y_model": 0,
      "rows": 7482
    },
    "provenance": {
      "eligible_rows": 7360,
      "eligible_provenance_complete_rate": 1.0,
      "hash_complete_rate": 1.0
    },
    "split": {
      "b_c_canonical_overlap": 0
    }
  },
  "budget": {
    "qwen_cny": 27.572333486399955,
    "deepseek_cny": 3.439632960000003,
    "total_cny": 31.01196644639996,
    "over_hard_cap": false,
    "hard_caps": {
      "qwen": 48.0,
      "deepseek": 48.0
    }
  },
  "protocol_deviations": [
    {
      "item": "b_panel_capacity_low",
      "impact": "mechanism claims downgraded to exploratory",
      "broad_pairs": 8,
      "context_pairs": 6
    }
  ]
}
