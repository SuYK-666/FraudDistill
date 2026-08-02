# E1 v3 预算报告

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
    "deepseek_hard_stop_cny": 90
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
