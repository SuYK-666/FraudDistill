# E1 R2 任务收尾

```json
{
  "data_dir": "C:\\Users\\18201\\Desktop\\FraudDistill\\data\\prepared\\e1_context_recovery_r2",
  "report_dir": "C:\\Users\\18201\\Desktop\\FraudDistill\\reports\\e1_context_recovery_r2",
  "commit": "51bba702b0ceae992e5c1ceb0d7236240eb6d98e",
  "last_results": [
    {
      "protocol": "E1-CONTEXT-RECOVERY-R2-v1.0",
      "phase": "p4-pilot-gate",
      "commit": "51bba702b0ceae992e5c1ceb0d7236240eb6d98e",
      "git_status": "D reports/e1_triad_final/E1_TRIAD_BUDGET_REPORT.md\n D reports/e1_triad_final/E1_TRIAD_DATA_PROVENANCE_AUDIT.md\n D reports/e1_triad_final/E1_TRIAD_EXECUTIVE_REPORT_CN.md\n D reports/e1_triad_final/E1_TRIAD_FAILURE_BIAS_AUDIT_CN.md\n D reports/e1_triad_final/E1_TRIAD_FULL_ANALYSIS_REPORT_CN.md\n D reports/e1_triad_final/E1_TRIAD_PAPER_TABLES.md\n D reports/e1_triad_final/E1_TRIAD_REPRODUCTION_GUIDE.md\n D reports/e1_triad_final/E1_TRIAD_STATISTICAL_APPENDIX_CN.md\n D reports/e1_triad_final/E1_TRIAD_TASK_CLOSEOUT_CN.md\n?? configs/experiments/e1_context_recovery_r2.yaml\n?? scripts/run_e1_context_recovery_r2.py\n?? src/frauddistill/e1_context_r2/\n?? tests/e1_context_r2/",
      "wall_seconds": 0.132,
      "decision": "E1_R2_STOP_CONTEXT_CAPACITY",
      "pilot_decision": {
        "decision": "STOP_CONTEXT_CAPACITY",
        "checks": {
          "gold_quality": true,
          "context_critical_positive_ge_30": false,
          "context_stable_positive_ge_20": false,
          "context_hard_negative_ge_20": false,
          "context_stable_negative_ge_60": true
        },
        "strata": {
          "context_stable_negative": 287,
          "unresolved": 12,
          "context_hard_negative": 1
        }
      },
      "gold_quality": {
        "expected": 1200,
        "completed": 1200,
        "valid": 1191,
        "invalid": 9,
        "completion": 1.0,
        "valid_rate": 0.9925,
        "strata": {
          "context_stable_negative": 287,
          "unresolved": 12,
          "context_hard_negative": 1
        },
        "passed": true
      }
    }
  ]
}
```
