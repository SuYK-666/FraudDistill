# E1 R2 数据来源审计

```json
{
  "flow": {
    "historical_scanned_rows": 95878,
    "missing_q": 59201,
    "accepted": 34660,
    "bad_status": 1543,
    "not_target_qwen_deepseek": 314,
    "missing_y": 160
  },
  "dedup_unique_rows": 17863,
  "by_version": {
    "e1_v8_a2c": 240,
    "e1_v81_narrative_delta": 2681,
    "e1_v91_recovery": 9760,
    "e1_v10_trilayer": 4004,
    "e1_v11_event_pool": 1178
  },
  "by_model": {
    "deepseek": 12889,
    "qwen": 4974
  },
  "by_language": {
    "en": 8859,
    "zh": 9004
  },
  "by_category": {
    "impersonation": 5071,
    "fake_job": 3128,
    "fraudulent_service": 3946,
    "phishing": 3926,
    "relationship_investment": 1792
  },
  "reject_samples": [
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 0,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 1,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 2,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 3,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 4,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 5,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 6,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 7,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 8,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 9,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 10,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 11,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 12,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 13,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 14,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 15,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 16,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 17,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 18,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 19,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 20,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 21,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 22,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 23,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 24,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 25,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 26,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 27,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 28,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 29,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 30,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 31,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 32,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 33,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 34,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 35,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 36,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 37,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 38,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 39,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 40,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 41,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 42,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 43,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 44,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 45,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 46,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 47,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 48,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 49,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 50,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 51,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 52,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 53,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 54,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 55,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 56,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 57,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 58,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 59,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 60,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 61,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 62,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 63,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 64,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 65,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 66,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 67,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 68,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 69,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 70,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 71,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 72,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 73,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 74,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 75,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 76,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 77,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 78,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 79,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 80,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 81,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 82,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 83,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 84,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 85,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 86,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 87,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 88,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 89,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 90,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 91,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 92,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 93,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 94,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 95,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 96,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 97,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 98,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 99,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 100,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 101,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 102,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 103,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 104,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 105,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 106,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 107,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 108,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 109,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 110,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 111,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 112,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 113,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 114,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 115,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 116,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 117,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 118,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 119,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 120,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 121,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 122,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 123,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 124,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 125,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 126,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 127,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 128,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 129,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 130,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 131,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 132,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 133,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 134,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 135,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 136,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 137,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 138,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 139,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 140,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 141,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 142,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 143,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 144,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 145,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 146,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 147,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 148,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 149,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 150,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 151,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 152,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 153,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 154,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 155,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 156,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 157,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 158,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 159,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 160,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 161,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 162,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 163,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 164,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 165,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 166,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 167,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 168,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 169,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 170,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 171,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 172,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 173,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 174,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 175,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 176,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 177,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 178,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 179,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 180,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 181,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 182,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 183,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 184,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 185,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 186,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 187,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 188,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 189,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 190,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 191,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 192,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 193,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 194,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 195,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 196,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 197,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 198,
      "reason": "missing_q"
    },
    {
      "file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\A_CONSENSUS.jsonl",
      "index": 199,
      "reason": "missing_q"
    }
  ]
}
```
