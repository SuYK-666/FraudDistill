# E1 v3.1 预算报告

```json
{
  "pricing_sources": [
    {
      "provider": "qwen",
      "url": "https://help.aliyun.com/zh/model-studio/model-pricing",
      "snapshot_note": "Official Alibaba Cloud Model Studio pricing page checked before v3 implementation."
    },
    {
      "provider": "deepseek",
      "url": "https://api-docs.deepseek.com/quick_start/pricing/",
      "snapshot_note": "Official DeepSeek API pricing page recorded as required protocol source."
    }
  ],
  "hard_limits_cny": {
    "hard_stop_total_cny": 215,
    "e1_a_hard_stop_cny": 100,
    "e1_b_hard_stop_cny": 100,
    "e1_c_hard_stop_cny": 15,
    "qwen_hard_stop_cny": 120,
    "deepseek_hard_stop_cny": 90,
    "e1_a_target_hard_stop_cny": 35,
    "e1_a_gold_hard_stop_cny": 70
  },
  "effective_concurrency": {
    "user_requested_total": 120,
    "qwen": 24,
    "deepseek": 20,
    "adjudicator": 8,
    "reason": "v3 protocol forbids using total concurrency 120 as default; provider-level caps are enforced."
  },
  "ledger_policy": {
    "check_every_calls": 200,
    "hard_stop_required": true
  }
}
```

## 调用 ledger
```json
{
  "rows": 19961,
  "total_cost_cny": 54.2399,
  "by_provider_cost": {
    "qwen": 29.8126,
    "deepseek": 24.4273
  },
  "calls_by_phase": {
    "E1-A-target-generation-v31": 4418,
    "E1-A-gold-v31": 8838,
    "E1-A-gold-v31-reused": 4,
    "E1-B-counterfactual-generation-v31": 1799,
    "E1-B-gold-v31": 4902
  }
}
```
