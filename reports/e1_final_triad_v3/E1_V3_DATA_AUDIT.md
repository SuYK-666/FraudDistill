# E1 v3 数据与来源审计

```json
{
  "p0": {
    "protocol": "E1-FINAL-TRIAD-v3.0-7500-3200-RealPrevalence",
    "runtime_commit": "e05bcb09483ffc94da4d38a74c1c0b7bf178dfd0",
    "git_status": "D reports/e1_real_triad_v2/E1_REAL_V2_BUDGET_REPORT.md\n D reports/e1_real_triad_v2/E1_REAL_V2_DATA_PROVENANCE_AUDIT.md\n D reports/e1_real_triad_v2/E1_REAL_V2_EXECUTIVE_REPORT_CN.md\n D reports/e1_real_triad_v2/E1_REAL_V2_FAILURE_BIAS_AUDIT_CN.md\n D reports/e1_real_triad_v2/E1_REAL_V2_FULL_ANALYSIS_REPORT_CN.md\n D reports/e1_real_triad_v2/E1_REAL_V2_GOLD_QUALITY_REPORT.md\n D reports/e1_real_triad_v2/E1_REAL_V2_PAPER_TABLES.md\n D reports/e1_real_triad_v2/E1_REAL_V2_REPRODUCTION_GUIDE.md\n D reports/e1_real_triad_v2/E1_REAL_V2_STATISTICAL_APPENDIX_CN.md\n D reports/e1_real_triad_v2/E1_REAL_V2_TASK_CLOSEOUT_CN.md\n D reports/e1_real_triad_v2/E1_REAL_V2_TASK_OVERVIEW_CN.md\n M src/frauddistill/e1_real_v2/pairlite_cpu_v2.py\n?? configs/experiments/e1_final_triad_v3.yaml\n?? reports/e1_final_triad_v3/\n?? scripts/run_e1_a7500.py\n?? scripts/run_e1_b3200.py\n?? scripts/run_e1_c_real_prevalence.py\n?? scripts/run_e1_final_triad_v3.py\n?? src/frauddistill/e1_final_v3/\n?? tests/e1_final_v3/",
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
    },
    "source_audit": {
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
    "license_audit": {
      "download_date": "2026-08-02",
      "sources": [
        {
          "dataset": "Fraud-R1",
          "url": "https://github.com/kaustpradalab/Fraud-R1",
          "local_path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\full\\prompts\\fraudr1_all_target_prompts.jsonl",
          "redistribution_policy": "final public artifacts should prefer IDs/hashes/statistics unless dataset license snapshot permits raw text redistribution"
        },
        {
          "dataset": "OR-Bench",
          "url": "https://github.com/justincui03/OR-Bench",
          "local_path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\full\\prompts\\or_bench_hard_safe_prompts.jsonl",
          "redistribution_policy": "open-control only; not natural prevalence"
        },
        {
          "dataset": "Do-Not-Answer",
          "url": "https://github.com/Libr-AI/do-not-answer",
          "local_path": "",
          "redistribution_policy": "not yet materialized in v3 local run"
        }
      ],
      "gate": "PASS_WITH_LOCAL_HASHES",
      "note": "License evidence is recorded as URLs and local file hashes. Raw text redistribution remains restricted in reports."
    },
    "gate": "PASS_DRY_RUN",
    "api_allowed_now": false,
    "api_block_reason": "P0 implementation run must be committed and reviewed before live API expansion. Current command performs reproducible dry-run/audit only."
  },
  "a": {
    "prompt_audit": {
      "source_file": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\full\\prompts\\fraudr1_all_target_prompts.jsonl",
      "accepted": 2000,
      "rejects": {
        "duplicate_q_hash": 2282
      },
      "unique_q_hashes": 2000
    },
    "existing_registry_audit": {
      "input_files": [
        "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_v10_trilayer\\E1_V10_RESPONSE_REGISTRY.jsonl"
      ],
      "accepted": 7482,
      "rejects": {},
      "by_source_dataset": {
        "V10-natural-real": 7482
      },
      "by_provider": {
        "deepseek": 3941,
        "qwen": 3541
      },
      "with_q_text": 3604,
      "missing_q_text": 3878,
      "source_derived_rows": 0
    },
    "quota_table": [
      {
        "language": "en",
        "category": "fake_job_posting",
        "existing_unique_q": 105,
        "target_unique_q": 375,
        "new_q_needed": 270,
        "new_q_selected": 150,
        "cell_ready": false
      },
      {
        "language": "en",
        "category": "fraudulent_service",
        "existing_unique_q": 247,
        "target_unique_q": 375,
        "new_q_needed": 128,
        "new_q_selected": 128,
        "cell_ready": true
      },
      {
        "language": "en",
        "category": "impersonation",
        "existing_unique_q": 247,
        "target_unique_q": 375,
        "new_q_needed": 128,
        "new_q_selected": 128,
        "cell_ready": true
      },
      {
        "language": "en",
        "category": "network_friendship",
        "existing_unique_q": 34,
        "target_unique_q": 375,
        "new_q_needed": 341,
        "new_q_selected": 85,
        "cell_ready": false
      },
      {
        "language": "en",
        "category": "phishing",
        "existing_unique_q": 183,
        "target_unique_q": 375,
        "new_q_needed": 192,
        "new_q_selected": 120,
        "cell_ready": false
      },
      {
        "language": "zh",
        "category": "fake_job_posting",
        "existing_unique_q": 113,
        "target_unique_q": 375,
        "new_q_needed": 262,
        "new_q_selected": 150,
        "cell_ready": false
      },
      {
        "language": "zh",
        "category": "fraudulent_service",
        "existing_unique_q": 245,
        "target_unique_q": 375,
        "new_q_needed": 130,
        "new_q_selected": 130,
        "cell_ready": true
      },
      {
        "language": "zh",
        "category": "impersonation",
        "existing_unique_q": 245,
        "target_unique_q": 375,
        "new_q_needed": 130,
        "new_q_selected": 130,
        "cell_ready": true
      },
      {
        "language": "zh",
        "category": "network_friendship",
        "existing_unique_q": 30,
        "target_unique_q": 375,
        "new_q_needed": 345,
        "new_q_selected": 84,
        "cell_ready": false
      },
      {
        "language": "zh",
        "category": "phishing",
        "existing_unique_q": 181,
        "target_unique_q": 375,
        "new_q_needed": 194,
        "new_q_selected": 194,
        "cell_ready": true
      }
    ],
    "pending_target_calls": 2598
  },
  "b": {
    "input_rows": 10164,
    "known_gold_rows": 3080,
    "by_stratum": {
      "unknown_gold": 7084,
      "context_stable_negative": 3048,
      "context_hard_negative": 6,
      "context_stable_positive": 25,
      "context_critical_positive": 1
    },
    "by_provenance": {
      "real_target_response": 10164
    },
    "quota_checks": {
      "context_stable_positive": false,
      "context_stable_negative": true,
      "context_critical_positive": false,
      "context_hard_negative": false
    },
    "real_target_response_rows": 10164,
    "formal_panel_ready": false,
    "classified_preview": [
      {
        "response_id": "p2|fraudr1_en_899|stage0|deepseek|rep3",
        "canonical_q_id": "fraudr1_en_899",
        "q_hash_recomputed": "ece4d90aa4be258b2b4de4f4a7783bb0b6fa5a385dfbda7be7956b937d9b1f5d",
        "y_hash": "d5b98a36c65fac61e03a6f9cc613fb787dff92321873a730ec044084e799b6ad",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fake_job_posting",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_262|stage1|deepseek|rep2",
        "canonical_q_id": "fraudr1_en_262",
        "q_hash_recomputed": "0026a820daf7e9d92f27873f3e3fd240e050dc78d069da58cd4612425bb35946",
        "y_hash": "86c3f296ff1939afc898ca2aed2e589b284bb2590340f708adea6afcc440809b",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_262|stage0|deepseek|rep1",
        "canonical_q_id": "fraudr1_en_262",
        "q_hash_recomputed": "28fa963f18820ad71a0b7f92b4acb372c70451034ac4a9efe085da484cea88f0",
        "y_hash": "4d2ce40ecb32b20418aa607959cea7dc86944a0e8128626a371fb399d687b155",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_579|stage1|deepseek|rep0",
        "canonical_q_id": "fraudr1_en_579",
        "q_hash_recomputed": "fbfc42cceab1cd7b2fd8ac30686ff940003815f2e826f1e87cbf73b5d549e4d7",
        "y_hash": "8aa78d8f6606d4d16c439ca86148bb06cb7109d81ca0b43871db1651ab16bea7",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "impersonation",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_873|stage0|deepseek|rep1",
        "canonical_q_id": "fraudr1_en_873",
        "q_hash_recomputed": "af57123003b7ee1c6a8b4116bbea03d813dc2d96782e235d0c4c92af98317bd7",
        "y_hash": "0054e5a294c2e9edc50b5413b9f3af545a0a2561af984dafee4dc704c8c94c32",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fake_job_posting",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_345|stage1|deepseek|rep0",
        "canonical_q_id": "fraudr1_en_345",
        "q_hash_recomputed": "6dc0d99389dd631effdd42b8a520e5b33a6fd6694047e765f9e525817df9eb4c",
        "y_hash": "b9b001f3962c29bc349ed2056b1cd93823bcfc2bd4cf97e90f264d515a70a56e",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "impersonation",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_24|stage0|deepseek|rep2",
        "canonical_q_id": "fraudr1_en_24",
        "q_hash_recomputed": "4b8dd25a5d095c6546b4ba1656b76f0546813e151e5995495a80b76ea7232832",
        "y_hash": "39cd90bbf909a4263a3e37fffdd738898f2398d508cfe43f19bfe574078c7702",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_262|stage1|deepseek|rep3",
        "canonical_q_id": "fraudr1_en_262",
        "q_hash_recomputed": "0026a820daf7e9d92f27873f3e3fd240e050dc78d069da58cd4612425bb35946",
        "y_hash": "1d81b520f66b849841f1d09a2c6c4626135a26e7daaa15c86839c126fdbd0f88",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_541|stage0|deepseek|rep3",
        "canonical_q_id": "fraudr1_en_541",
        "q_hash_recomputed": "0043ec7db80dac239c5ea09a4e3f469d0f33116b12eebd5d7c1cfa021b2f90a6",
        "y_hash": "fde8f15a39d607d320fb1d6f3d126f47d176174209d40d47d2e38546a71c3bfc",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_24|stage0|deepseek|rep1",
        "canonical_q_id": "fraudr1_en_24",
        "q_hash_recomputed": "4b8dd25a5d095c6546b4ba1656b76f0546813e151e5995495a80b76ea7232832",
        "y_hash": "3232dd262e034de6d26ad5256c13a763a703faae31e284ef1a6bda3281bbd661",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_579|stage1|deepseek|rep1",
        "canonical_q_id": "fraudr1_en_579",
        "q_hash_recomputed": "fbfc42cceab1cd7b2fd8ac30686ff940003815f2e826f1e87cbf73b5d549e4d7",
        "y_hash": "5028fa24ce408b5c94896c664514cf11ff1b3e99ddf55fa8aff18ce6f5ef2a7d",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "impersonation",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_24|stage0|deepseek|rep3",
        "canonical_q_id": "fraudr1_en_24",
        "q_hash_recomputed": "4b8dd25a5d095c6546b4ba1656b76f0546813e151e5995495a80b76ea7232832",
        "y_hash": "494c4a7d523b6d2b8859f9f707f0d5ac6fe4038cf5a455df6b4962289be8fd1f",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_579|stage0|deepseek|rep1",
        "canonical_q_id": "fraudr1_en_579",
        "q_hash_recomputed": "bcae848dc305434a3d48ea41760d8f8eecaa64a1b3eebae73f1d2e06c6823fae",
        "y_hash": "f7316e187917882874068265b3f25894e88465be26e339564c851d58fd50465f",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "impersonation",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_345|stage1|deepseek|rep3",
        "canonical_q_id": "fraudr1_en_345",
        "q_hash_recomputed": "6dc0d99389dd631effdd42b8a520e5b33a6fd6694047e765f9e525817df9eb4c",
        "y_hash": "1d047c079cc72091ada0e526aa2f4f9090a5f013629f00036e28f93de271ac21",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "impersonation",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_899|stage0|deepseek|rep1",
        "canonical_q_id": "fraudr1_en_899",
        "q_hash_recomputed": "ece4d90aa4be258b2b4de4f4a7783bb0b6fa5a385dfbda7be7956b937d9b1f5d",
        "y_hash": "d908451a008cb8be3f18495ebe37147f36dfcccdc5aa35a9a12a2d6d0f0de071",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fake_job_posting",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_541|stage0|deepseek|rep2",
        "canonical_q_id": "fraudr1_en_541",
        "q_hash_recomputed": "0043ec7db80dac239c5ea09a4e3f469d0f33116b12eebd5d7c1cfa021b2f90a6",
        "y_hash": "289006af828bf56fb8a576d04738fd038ff717e17599920641e721dcf07e981e",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_262|stage1|deepseek|rep0",
        "canonical_q_id": "fraudr1_en_262",
        "q_hash_recomputed": "0026a820daf7e9d92f27873f3e3fd240e050dc78d069da58cd4612425bb35946",
        "y_hash": "b96d84f9a37912f0cf03bb30d78bd37e7d3f37a8818dfdf0e9bf3347e317ef04",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_899|stage0|deepseek|rep0",
        "canonical_q_id": "fraudr1_en_899",
        "q_hash_recomputed": "ece4d90aa4be258b2b4de4f4a7783bb0b6fa5a385dfbda7be7956b937d9b1f5d",
        "y_hash": "059a5e6ac29bdec7e58f0fa4603d4675660ef94eeaf3860224350b9190d266d7",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fake_job_posting",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_262|stage1|deepseek|rep1",
        "canonical_q_id": "fraudr1_en_262",
        "q_hash_recomputed": "0026a820daf7e9d92f27873f3e3fd240e050dc78d069da58cd4612425bb35946",
        "y_hash": "3ae268f64279e96dfa2fd4e02d0be9e1fa217ab4cc30b7ae65e7740128ee057a",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      },
      {
        "response_id": "p2|fraudr1_en_262|stage0|deepseek|rep0",
        "canonical_q_id": "fraudr1_en_262",
        "q_hash_recomputed": "28fa963f18820ad71a0b7f92b4acb372c70451034ac4a9efe085da484cea88f0",
        "y_hash": "8e6a8d0e4a5d7aa5b2c5d786e913b519da3d908d7e2f2b7bdc727e9a29380f30",
        "target_provider": "deepseek",
        "language": "en",
        "fraud_category": "fraudulent_service",
        "source_dataset": "V8.1-P2-real",
        "gold_status": "UNKNOWN",
        "stratum": "unknown_gold"
      }
    ]
  }
}
```
