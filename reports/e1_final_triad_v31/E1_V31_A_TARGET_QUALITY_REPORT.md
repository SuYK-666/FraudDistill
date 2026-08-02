# E1 v3.1 A Target 质量报告

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
