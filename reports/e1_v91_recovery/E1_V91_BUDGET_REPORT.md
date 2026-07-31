# E1 V9.1 预算报告

{
  "budget": {
    "qwen_cny": 34.40152999999996,
    "deepseek_cny": 21.421076000000053,
    "total_cny": 55.82260600000001,
    "over_hard_cap": false,
    "hard_caps": {
      "qwen": 49.0,
      "deepseek": 49.0
    }
  },
  "ledger_jsonl_audit": {
    "exists": true,
    "lines": 10783,
    "valid_json": 10772,
    "invalid_json": 11,
    "invalid_examples": [
      {
        "line": 277,
        "error": "Extra data: line 1 column 10 (char 9)"
      },
      {
        "line": 682,
        "error": "Expecting value: line 1 column 1 (char 0)"
      },
      {
        "line": 1147,
        "error": "Extra data: line 1 column 2 (char 1)"
      },
      {
        "line": 3073,
        "error": "Extra data: line 1 column 4 (char 3)"
      },
      {
        "line": 5237,
        "error": "Unterminated string starting at: line 1 column 1 (char 0)"
      }
    ]
  },
  "ledger_csv": "E1_V91_BUDGET_LEDGER.csv",
  "ledger_jsonl": "E1_V91_BUDGET_LEDGER.jsonl"
}
