# E1 Final Push 论文表格

{
  "table_e1_a": [
    {
      "model": "deepseek",
      "n": 1540,
      "central_events": 15,
      "central_risk": 0.00974025974025974,
      "ci_low": 0.005911560103550392,
      "ci_high": 0.01600873399617203,
      "events_per_1k": 9.74025974025974,
      "lower_positive": 6,
      "upper_positive": 30,
      "scope": "A0 frozen standard-prompt benchmark"
    },
    {
      "model": "qwen",
      "n": 1540,
      "central_events": 11,
      "central_risk": 0.007142857142857143,
      "ci_low": 0.003993115996839215,
      "ci_high": 0.012745298866325448,
      "events_per_1k": 7.142857142857142,
      "lower_positive": 6,
      "upper_positive": 36,
      "scope": "A0 frozen standard-prompt benchmark"
    }
  ],
  "pilot": {
    "decision": "PILOT_STOP",
    "central_positive": 0,
    "mixed_groups": 0,
    "stop_reason_codes": [
      "central_positive_below_gate",
      "mixed_group_below_gate"
    ]
  },
  "fallback": {
    "decision": "NO_EXPANSION_BOUNDED_CLOSEOUT",
    "reason": [
      "central_positive_below_gate",
      "mixed_group_below_gate"
    ],
    "external_fallback_rows": 3
  }
}
