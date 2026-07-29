# FraudDistill 实验1 Paired-Interaction R4.1 任务报告

- 协议：`E1-PairedInteraction-R4.1-SimpleRecovery`
- 最新阶段：`panel_c`
- 最新判定：`E1_R41_PANEL_C_STOP`
- Git 提交：`737d40d9baf3349eac629b0bf380aa7c1e03a33d`

## 本轮整改摘要

R4.1 已按停机复盘要求移除 By near-y collision 与 Panel C exact-q mixed-outcome 依赖。Bctx 使用 DeepSeek Pro/Flash 对 exact-q 正负回答组进行独立上下文增益验证；Panel C 固定使用每个 Fraud-R1 base 的 sample_index=0 单回答，并在 safe/unsafe 与中英文之间做平衡选择。

## cache_audit

- 判定：`E1_R41_CACHE_AUDIT_PASS`
- 用时：`0.149` 秒

```json
{
  "archive": "C:\\Users\\18201\\Desktop\\FraudDistill\\archive\\E1_R4_G0_STOP_ARCHIVE_20260729_174715",
  "old_panel_c_labels_exists": true,
  "old_panel_c_generations_exists": true,
  "old_panel_c_labels_sha256": "645cd3d6d4c0f100065bfcb0c73eefbbcf1130490847ec1cfebc1008487fbe0a",
  "old_panel_c_sample0_rows": 261,
  "old_panel_c_sample0_label_counts": {
    "safe": 260,
    "unsafe": 1
  },
  "public_revision_main": [],
  "dirty_status": "D \"reports/E1_PairedInteraction_R4_\\344\\273\\273\\345\\212\\241\\346\\212\\245\\345\\221\\212_\\344\\270\\255\\346\\226\\207.md\"\n?? configs/experiments/e1_paired_interaction_r41.yaml\n?? scripts/run_e1_paired_interaction_r41.py\n?? tests/test_e1_paired_interaction_r41.py"
}
```

## bctx

- 判定：`E1_R41_BCTX_STOP`
- 用时：`269.959` 秒

```json
{
  "candidate_groups": 468,
  "validated_pass_groups": 130,
  "anchor_groups": 70,
  "model_dev_groups": 60,
  "source_counts_selected": {
    "PKU-SafeRLHF": 112,
    "BeaverTails": 18
  },
  "source_shares_selected": {
    "PKU-SafeRLHF": 0.8615384615384616,
    "BeaverTails": 0.13846153846153847
  },
  "selected_mean_context_gain": 0.29423076923076924,
  "selected_median_context_gain": 0.25,
  "selected_y_only_acc": 0.6423076923076924,
  "checks": {
    "capacity": false,
    "mean_context_gain": true,
    "median_context_gain": true,
    "y_only_acc": true,
    "source_share": false
  },
  "passed": false,
  "source_audit": {
    "sources": {
      "PKU-SafeRLHF": {
        "rows": 13938,
        "safe": 1675,
        "unsafe": 12263
      },
      "BeaverTails": {
        "rows": 1844,
        "safe": 331,
        "unsafe": 1513
      },
      "Aegis": {
        "rows": 356,
        "safe": 149,
        "unsafe": 207
      }
    },
    "failures": [],
    "source_files": {
      "PKU-SafeRLHF:train": {
        "dataset": "PKU-Alignment/PKU-SafeRLHF",
        "revision": "9421ffafec3fa40a1f1a7d567b4d525079477ecb",
        "rows_raw": 73907
      },
      "PKU-SafeRLHF:test": {
        "dataset": "PKU-Alignment/PKU-SafeRLHF",
        "revision": "9421ffafec3fa40a1f1a7d567b4d525079477ecb",
        "rows_raw": 8211
      },
      "BeaverTails:330k_train": {
        "dataset": "PKU-Alignment/BeaverTails",
        "revision": "8401fe609d288129cc684a9b3be6a93e41cfe678",
        "rows_raw": 300567
      },
      "BeaverTails:30k_test": {
        "dataset": "PKU-Alignment/BeaverTails",
        "revision": "8401fe609d288129cc684a9b3be6a93e41cfe678",
        "rows_raw": 3021
      },
      "Aegis:prepared_qy": {
        "path": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\full\\evaluation_qy\\aegis_qy.jsonl",
        "sha256": "e9c57b831d6c8baf11506406d5244c1aa8553fb34240fcc24a0962f831ab35a5"
      }
    }
  },
  "prediction_rows": 3744
}
```

## panel_c

- 判定：`E1_R41_PANEL_C_STOP`
- 用时：`4773.432` 秒

```json
{
  "candidate_rows": 1816,
  "available_by_language_label": {
    "en|safe": 781,
    "en|unsafe": 48,
    "zh|safe": 979,
    "zh|unsafe": 8
  },
  "anchor_rows": 176,
  "model_dev_rows": 60,
  "anchor_counts": {
    "zh|safe": 60,
    "zh|unsafe": 8,
    "en|safe": 60,
    "en|unsafe": 48
  },
  "model_dev_counts": {
    "zh|safe": 30,
    "en|safe": 30
  },
  "checks": {
    "anchor_240": false,
    "model_dev_120": false,
    "fixed_sample0": true,
    "one_row_per_base": true,
    "dev_anchor_disjoint": true
  },
  "passed": false,
  "old_sample0_audit": {
    "candidate_rows": 261,
    "available_by_language_label": {
      "en|safe": 133,
      "en|unsafe": 1,
      "zh|safe": 127,
      "zh|unsafe": 0
    },
    "anchor_rows": 121,
    "model_dev_rows": 60,
    "anchor_counts": {
      "zh|safe": 60,
      "en|safe": 60,
      "en|unsafe": 1
    },
    "model_dev_counts": {
      "zh|safe": 30,
      "en|safe": 30
    },
    "checks": {
      "anchor_240": false,
      "model_dev_120": false,
      "fixed_sample0": true,
      "one_row_per_base": true,
      "dev_anchor_disjoint": true
    },
    "passed": false
  },
  "roleplay_candidates": 4282,
  "roleplay_generated": 4282,
  "roleplay_labeled": 4165,
  "settings_x_label": {
    "r4_helpful_single_cache|safe": 260,
    "r4_helpful_single_cache|unsafe": 1,
    "fraudr1_roleplay_single|safe": 1500,
    "fraudr1_roleplay_single|unsafe": 55
  },
  "panel_c_split_audit": {
    "passed": true,
    "checks": {
      "fixed_sample0": true,
      "one_row_per_base": true,
      "dev_anchor_base_disjoint": true
    },
    "base_overlap_count": 0,
    "base_overlap_examples": []
  }
}
```

## 数据与复现位置

- 数据目录：`C:\Users\18201\Desktop\FraudDistill\data\prepared\e1_paired_interaction_r41`
- 输出目录：`C:\Users\18201\Desktop\FraudDistill\outputs\e1_paired_interaction_r41`
- 原始与中间 API 缓存保留在数据目录和输出目录中；Git 提交只跟踪代码、配置、测试和报告，不提交 data/outputs/archive/api_keys.py。