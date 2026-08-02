# E1 REAL v2 数据来源审计

```json
{
  "registry_rows": 3956,
  "source_derived_rows": 0,
  "real_target_response_rows": 3956,
  "real_target_response_ratio": 1.0,
  "by_provider": {
    "deepseek": 2358,
    "qwen": 1598
  },
  "by_source_dataset": {
    "Fraud-R1/V10-natural": 3604,
    "Fraud-R1/V8.1-P2-real-target": 352
  },
  "by_language": {
    "en": 2004,
    "zh": 1952
  },
  "v10_audit": {
    "source_file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v10_trilayer\\E1_V10_RESPONSE_REGISTRY.jsonl",
    "accepted": 3604,
    "rejects": {
      "missing_q_or_y": 3878
    },
    "real_target_response_ratio": 1.0,
    "source_derived_rows": 0,
    "by_provider": {
      "deepseek": 2006,
      "qwen": 1598
    },
    "by_language": {
      "en": 1804,
      "zh": 1800
    },
    "by_category": {
      "fake_job": 534,
      "fraudulent_service": 1044,
      "impersonation": 1046,
      "online_relationship": 192,
      "phishing": 788
    }
  },
  "v81_audit": {
    "source_file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v81_narrative_delta\\P2_TARGET_RESPONSES.jsonl",
    "accepted": 400,
    "rejects": {},
    "real_target_response_ratio": 1.0,
    "source_derived_rows": 0,
    "by_provider": {
      "deepseek": 400
    },
    "by_language": {
      "en": 200,
      "zh": 200
    },
    "by_category": {
      "fake_job": 80,
      "fraudulent_service": 80,
      "impersonation": 80,
      "online_relationship": 80,
      "phishing": 80
    }
  }
}
```
