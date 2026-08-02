# E1 v3.1 A 层 Target 质量报告

```json
{
  "manifest_audit": {
    "case_audit": {
      "raw_prompt_rows": 4282,
      "base_rows": 2141,
      "canonical_cases": 2141,
      "by_language_category": {
        "zh|phishing": 236,
        "zh|network_friendship": 84,
        "zh|fake_job_posting": 150,
        "zh|fraudulent_service": 300,
        "zh|impersonation": 300,
        "en|fraudulent_service": 300,
        "en|impersonation": 300,
        "en|phishing": 236,
        "en|fake_job_posting": 150,
        "en|network_friendship": 85
      }
    },
    "history_audit": {
      "complete_prompt_pairs": 1541,
      "reused_responses": 3082,
      "rejects": {
        "not_v91": 4400
      },
      "provider_counts": {
        "deepseek": 1541,
        "qwen": 1541
      }
    },
    "canonical_cases": 2141,
    "assistant_prompt_instances": 2141,
    "roleplay_reused_prompt_instances": 1541,
    "roleplay_extra_prompt_instances": 68,
    "target_prompt_instances": 3750,
    "expected_target_responses": 7500,
    "reused_target_responses": 3082,
    "pending_target_calls": 4418,
    "stage_gt_0": 0,
    "prompt_instance_duplicates": 0,
    "exact_q_hash_conflicts": 0
  },
  "target_quality": {
    "new_response_rows": 4418,
    "valid_new_response_rows": 4418,
    "complete_new_pairs": 2209,
    "pending_target_calls_initial": 4418,
    "target_gate": "PASS"
  },
  "gold_quality": {
    "expected_responses": 4418,
    "completed_responses": 4418,
    "completion_rate": 1.0,
    "valid_vote_rate": 1.0,
    "both_vote_rows": 4418,
    "binary_agreement": 0.9995473064735174,
    "pabak": 0.9990946129470348,
    "cohen_kappa": 0.0,
    "gwet_ac1": 0.9995471014957029,
    "uncertain_rate": 0.0,
    "adjudicated_count": 2,
    "unresolved_disagreements": 0,
    "gate": "PASS"
  },
  "gold_coverage": {
    "new_responses": 4418,
    "new_gold_known": 4418,
    "reused_responses": 3082,
    "reused_gold_known": 3082,
    "reuse_stats": {
      "reused": 3080,
      "rejected_reuse_missing_gold": 2
    },
    "registry_rows": 7500,
    "registry_with_gold": 7500,
    "gold_quality": {
      "expected_responses": 4418,
      "completed_responses": 4418,
      "completion_rate": 1.0,
      "valid_vote_rate": 1.0,
      "both_vote_rows": 4418,
      "binary_agreement": 0.9995473064735174,
      "pabak": 0.9990946129470348,
      "cohen_kappa": 0.0,
      "gwet_ac1": 0.9995471014957029,
      "uncertain_rate": 0.0,
      "adjudicated_count": 2,
      "unresolved_disagreements": 0,
      "gate": "PASS"
    }
  },
  "behavior_stats": {
    "n_with_gold": 7500,
    "n_rows": 7500,
    "overall": {
      "lower": {
        "n": 7500,
        "positive": 12,
        "rate": 0.0016,
        "wilson_95": {
          "low": 0.0009155295004006922,
          "high": 0.002794764616750031
        }
      },
      "central": {
        "n": 7500,
        "positive": 28,
        "rate": 0.0037333333333333333,
        "wilson_95": {
          "low": 0.002584312137040857,
          "high": 0.005390464402294347
        }
      },
      "upper": {
        "n": 7500,
        "positive": 68,
        "rate": 0.009066666666666667,
        "wilson_95": {
          "low": 0.007158685741368445,
          "high": 0.011477296853427962
        }
      }
    },
    "by_model": [
      {
        "stratum": "deepseek",
        "n": 3750,
        "positive": 17,
        "rate": 0.004533333333333334,
        "wilson_95": {
          "low": 0.0028323802957402697,
          "high": 0.00724834880158884
        }
      },
      {
        "stratum": "qwen",
        "n": 3750,
        "positive": 11,
        "rate": 0.0029333333333333334,
        "wilson_95": {
          "low": 0.0016387443218198962,
          "high": 0.005245259465812753
        }
      }
    ],
    "by_setting": [
      {
        "stratum": "assistant",
        "n": 4282,
        "positive": 2,
        "rate": 0.00046707146193367583,
        "wilson_95": {
          "low": 0.00012809718775472288,
          "high": 0.0017015222454503042
        }
      },
      {
        "stratum": "roleplay",
        "n": 3218,
        "positive": 26,
        "rate": 0.008079552517091361,
        "wilson_95": {
          "low": 0.005519719370909722,
          "high": 0.011812436644344372
        }
      }
    ],
    "by_language": [
      {
        "stratum": "en",
        "n": 3754,
        "positive": 12,
        "rate": 0.0031965903036760787,
        "wilson_95": {
          "low": 0.001829563449366146,
          "high": 0.005579333154119484
        }
      },
      {
        "stratum": "zh",
        "n": 3746,
        "positive": 16,
        "rate": 0.004271222637479978,
        "wilson_95": {
          "low": 0.0026308566351502254,
          "high": 0.006927269812138584
        }
      }
    ],
    "by_category": [
      {
        "stratum": "fake_job_posting",
        "n": 980,
        "positive": 22,
        "rate": 0.022448979591836733,
        "wilson_95": {
          "low": 0.014871024262220752,
          "high": 0.033756179228512886
        }
      },
      {
        "stratum": "fraudulent_service",
        "n": 2198,
        "positive": 0,
        "rate": 0.0,
        "wilson_95": {
          "low": 0.0,
          "high": 0.0017446573209460821
        }
      },
      {
        "stratum": "impersonation",
        "n": 2198,
        "positive": 0,
        "rate": 0.0,
        "wilson_95": {
          "low": 0.0,
          "high": 0.0017446573209460821
        }
      },
      {
        "stratum": "network_friendship",
        "n": 448,
        "positive": 3,
        "rate": 0.006696428571428571,
        "wilson_95": {
          "low": 0.002279950166667115,
          "high": 0.019500828530312847
        }
      },
      {
        "stratum": "phishing",
        "n": 1676,
        "positive": 3,
        "rate": 0.0017899761336515514,
        "wilson_95": {
          "low": 0.0006089360221258108,
          "high": 0.0052496280518929146
        }
      }
    ],
    "model_setting_language": [
      {
        "target_model": "deepseek",
        "setting": "assistant",
        "language": "en",
        "n": 1071,
        "positive": 2,
        "rate": 0.0018674136321195146,
        "wilson_95": {
          "low": 0.0005122616388952097,
          "high": 0.0067831945945270675
        }
      },
      {
        "target_model": "deepseek",
        "setting": "assistant",
        "language": "zh",
        "n": 1070,
        "positive": 0,
        "rate": 0.0,
        "wilson_95": {
          "low": 0.0,
          "high": 0.003577305373283746
        }
      },
      {
        "target_model": "deepseek",
        "setting": "roleplay",
        "language": "en",
        "n": 806,
        "positive": 4,
        "rate": 0.004962779156327543,
        "wilson_95": {
          "low": 0.0019315698001096983,
          "high": 0.012690377115201198
        }
      },
      {
        "target_model": "deepseek",
        "setting": "roleplay",
        "language": "zh",
        "n": 803,
        "positive": 11,
        "rate": 0.0136986301369863,
        "wilson_95": {
          "low": 0.007666030346935103,
          "high": 0.02436189600487241
        }
      },
      {
        "target_model": "qwen",
        "setting": "assistant",
        "language": "en",
        "n": 1071,
        "positive": 0,
        "rate": 0.0,
        "wilson_95": {
          "low": 0.0,
          "high": 0.003573977156509145
        }
      },
      {
        "target_model": "qwen",
        "setting": "assistant",
        "language": "zh",
        "n": 1070,
        "positive": 0,
        "rate": 0.0,
        "wilson_95": {
          "low": 0.0,
          "high": 0.003577305373283746
        }
      },
      {
        "target_model": "qwen",
        "setting": "roleplay",
        "language": "en",
        "n": 806,
        "positive": 6,
        "rate": 0.007444168734491315,
        "wilson_95": {
          "low": 0.003416065422228595,
          "high": 0.01614511985440906
        }
      },
      {
        "target_model": "qwen",
        "setting": "roleplay",
        "language": "zh",
        "n": 803,
        "positive": 5,
        "rate": 0.0062266500622665,
        "wilson_95": {
          "low": 0.0026624989168631593,
          "high": 0.01449261708439594
        }
      }
    ],
    "mcnemar_qwen_vs_deepseek": {
      "n_pairs": 3750,
      "qwen_positive": 11,
      "deepseek_positive": 17,
      "qwen_only_positive": 8,
      "deepseek_only_positive": 14,
      "both_positive": 3,
      "p_exact_mcnemar": 0.28627872467041016
    },
    "cluster_bootstrap_risk_diff": {
      "point_risk_diff": -0.0016000000000000003,
      "qwen_rate": 0.0029333333333333334,
      "deepseek_rate": 0.004533333333333334,
      "low_95": -0.00400320256204964,
      "high_95": 0.0008004268943436498,
      "n_clusters": 3682,
      "iterations": 10000
    },
    "note": "exploratory stratifications require FDR/Holm correction; main table uses central endpoint."
  },
  "natural_metrics_reference": {
    "existing_n": 3080,
    "central_positive": 26,
    "central_wilson": {
      "low": 0.005767301750188672,
      "high": 0.012340457538464482
    }
  }
}
```
