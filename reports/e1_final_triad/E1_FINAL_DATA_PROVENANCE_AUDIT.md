# E1 FINAL 数据来源审计

```json
{
  "registry": {
    "source_file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v10_trilayer\\E1_V10_RESPONSE_REGISTRY.jsonl",
    "accepted": 3482,
    "rejects": {
      "missing_required": 4000
    },
    "conflicts": [],
    "conflict_count": 0,
    "by_target_provider": {
      "deepseek": 1941,
      "qwen": 1541
    },
    "by_language": {
      "en": 1742,
      "zh": 1740
    },
    "by_category": {
      "fake_job": 440,
      "fraudulent_service": 1040,
      "impersonation": 1040,
      "network_friendship": 178,
      "phishing": 784
    },
    "target_model_mismatch": 0,
    "duplicate_conflicts": 0
  },
  "panel_flow": {
    "input_registry": 3482,
    "pairs_requested": 600,
    "strata_requested": {
      "context_critical_positive": 300,
      "context_stable_positive": 300,
      "context_hard_negative": 300,
      "context_stable_negative": 300
    },
    "panel_rows": 1200,
    "pair_count": 600,
    "by_stratum": {
      "context_critical_positive": 300,
      "context_stable_negative": 300,
      "context_stable_positive": 300,
      "context_hard_negative": 300
    },
    "by_split": {
      "model_dev": 480,
      "calibration": 240,
      "anchor": 360,
      "reserve": 120
    },
    "by_risk_subtype_positive": {
      "DIRECT_PERPETRATOR_ASSIST": 240,
      "UNSAFE_SCAM_COMPLIANCE": 180,
      "TRUST_NORMALIZATION": 180
    },
    "source_boundary": "B is a pre-registered source-derived case-control mechanism panel seeded from real V10 q contexts; it is not used as natural prevalence."
  },
  "leakage": {
    "passed": true
  }
}
```
