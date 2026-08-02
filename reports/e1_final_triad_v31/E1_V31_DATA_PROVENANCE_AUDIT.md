# E1 v3.1 数据来源审计

```json
{
  "p0": {
    "protocol": "E1-FINAL-TRIAD-v3.1-7500-3200-RealPrevalence",
    "runtime_commit": "e367547377891ecf16d4b43e512b93dd3b8d0c53",
    "git_status": "",
    "git_clean": true,
    "budget": {
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
    },
    "source_audit": {
      "fraudr1_raw_prompts": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\raw\\fraudr1\\prompts.jsonl",
        "exists": true,
        "sha256": "38794c0599ee2d287facac23e68df5aeeb378b84a27f3a33c0f8aed162648967"
      },
      "fraudr1_raw_base_en": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\raw\\fraudr1\\repo\\dataset\\FP-base-full\\FP-base-English.json",
        "exists": true,
        "sha256": "5c95ffee1b4932648fefeb1939c1f08d0b68f3b148349a53563ba7319a281192"
      },
      "fraudr1_raw_base_zh": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\raw\\fraudr1\\repo\\dataset\\FP-base-full\\FP-base-Chinese.json",
        "exists": true,
        "sha256": "d5abc5203924f4ca46daf31e3346249ccc3c617a454a8774c66a2afe05fe823c"
      },
      "fraudr1_prompts": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\full\\prompts\\fraudr1_all_target_prompts.jsonl",
        "exists": true,
        "sha256": "c7a8fc3a4a0c830bc7da052a6aeec2af445d103eb6fa0ba7e86288f4a7de58db"
      },
      "v10_registry": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v10_trilayer\\E1_V10_RESPONSE_REGISTRY.jsonl",
        "exists": true,
        "sha256": "e9f7738543bcae19eabeac715de5316c16e059b2cde753b9e37b3a6a8c15c6b7"
      },
      "v10_gold": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v10_trilayer\\E1_V10_GOLD_CONSENSUS.jsonl",
        "exists": true,
        "sha256": "cfb7bbd10faf6da2cd6788fa73f7232d281524c889c3cb8a715c2a4c3520c788"
      },
      "v10_a_metrics": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v10_trilayer\\E1_V10_A_NATURAL_METRICS.json",
        "exists": true,
        "sha256": "92086e3bcdd30d9fa24e5a5ef874fb3053783f0a22344b7217dec5f2ef26ee44"
      },
      "v81_p2_targets": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v81_narrative_delta\\P2_TARGET_RESPONSES.jsonl",
        "exists": true,
        "sha256": "4d9ad02459d0e5041ee42c8c0c95e0da6c726aba6a9bd59c56f52bad521c46a5"
      },
      "v8_a2c_targets": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v8_a2c\\TARGET_RESPONSES.jsonl",
        "exists": true,
        "sha256": "619365c6f8bc6913c354791a6ffa82edadbaeaa8a035d79683662185609cccce"
      },
      "v10_pressure_targets": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v10_trilayer\\E1_V10_PRESSURE_TARGET_RESPONSES.jsonl",
        "exists": true,
        "sha256": "773169e2d730cbddf5a341969920ac21ba76e94aaf4b141e96e5aa71289ef8c6"
      },
      "or_bench_prompts": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\full\\prompts\\or_bench_hard_safe_prompts.jsonl",
        "exists": true,
        "sha256": "50fb00dd75f09b0519a73ddbe206e76a600117af4e54e1c87767748ea7429921"
      }
    },
    "secret_scan": {
      "passed": true,
      "returncode": 1,
      "matches": ""
    },
    "api_allowed_now": true,
    "gate": "PASS"
  },
  "license": {
    "download_date": "2026-08-02",
    "sources": [
      {
        "dataset": "Fraud-R1",
        "url": "https://github.com/kaustpradalab/Fraud-R1",
        "local_path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\raw\\fraudr1\\prompts.jsonl",
        "raw_text_public_report_policy": "do_not_redistribute_raw_text"
      },
      {
        "dataset": "OR-Bench",
        "url": "https://github.com/justincui03/OR-Bench",
        "local_path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\full\\prompts\\or_bench_hard_safe_prompts.jsonl",
        "raw_text_public_report_policy": "ids_hashes_statistics_only"
      },
      {
        "dataset": "Do-Not-Answer",
        "url": "https://github.com/Libr-AI/do-not-answer",
        "local_path": "",
        "raw_text_public_report_policy": "not_materialized"
      }
    ],
    "gate": "PASS_IDS_HASHES_STATISTICS_ONLY"
  }
}
```
