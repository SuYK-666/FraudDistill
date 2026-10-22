# LightEval Supplementary Experiments

This directory contains the offline preparation and GPU execution framework for the two experiments in `LightEval_Supplementary_Experiments_Plan.md`.

## Fixed scope

### Experiment 1: model scale vs quality and cost

- Train Qwen2.5-Instruct students at 0.5B, 1.5B, 3B and 7B with one recipe.
- Train Qwen2.5 students at 0.5B, 1.5B, 3B and 7B with one recipe.
- Train Qwen3 students at 0.6B, 1.7B and 4B, plus Qwen3.5 students at 0.8B, 2B and 4B.
- The 14B arm is intentionally excluded because the available RTX 4090 D has 24 GB VRAM and cannot host a 14B BF16 model under the common-precision protocol.
- Reuse the existing DeepSeek-R1-Distill-Qwen-1.5B `best_step120` student.
- Select checkpoints on the 1,047-row development set.
- Evaluate all students on the same 1,262-row core test, 384-row frozen E6 relation test and 180-row hard-safe test.
- Calibrate with 96 relation rows plus 60 hard-safe calibration rows.
- Record quality, tokens, latency, throughput, VRAM, training time and cost.

### Experiment 2: training, calibration and audit budget

All rows below use the **same E6 384-row frozen relation test**:

| Policy | Training | Target calibration | Real audit |
|---|---:|---:|---:|
| A0 | no, base 1.5B category interface | no | 0 |
| A1 | yes | no | 0 |
| A2 | yes | yes | 0 |
| A3 | yes | yes | 5% = 19 |
| A4 | yes | yes | 10% = 38 |
| A5 | yes | yes | 15% = 57 |
| A6 | yes | yes | 20% = 76 |

**Table 9 is E6-based.** E5 supplies only compatible cache/interface conventions. The current E5 cache has zero exact `q+y` overlap with the selected E6 audit list, so no E5 label is copied into Table 9.

The 76 audit items are ranked once by `|student_score - 0.2323|`, then composite ID. A3-A6 are strict prefixes of this one frozen list. Reviewer input contains `q+y` only: no student score, automatic label, Gold/Silver label, family or model metadata.

## Current local status

- [x] Training/dev/test and E6 manifests frozen and hash-locked.
- [x] Counts validated: 4,747 / 1,047 / 1,262 / 96 / 60 / 384 / 180.
- [x] Target threshold reproduced from calibration: 0.2323.
- [x] 76-item audit list and 19/38/57/76 nested prefixes frozen.
- [x] E5 cache compatibility check performed; exact overlap = 0.
- [x] Ten new Qwen2.5/Qwen3/Qwen3.5 scale runs completed.
- [x] A0 GPU scoring, 76 real reviewer calls and A0-A6 replay completed.
- [x] Checkpoint selection, calibration, metrics, token/latency/VRAM and cost collection completed.
- [x] Paired family bootstrap, McNemar tests, Markdown tables and SVG figures completed.
- [x] Checkpoint SHA256 and frozen result artifacts copied locally; server model/run artifacts removed.

## Entry points

```text
scripts/supp_prepare.py              freeze and validate local assets
scripts/supp_scale_train.py          dry-run/execute five Qwen trainings
scripts/supp_scale_evaluate.py       select checkpoint, score, calibrate, cost
scripts/supp_ablation_score_a0.py    A0 on E6 384 rows
scripts/supp_audit.py                real/cached E6 boundary audit
scripts/supp_ablation_evaluate.py    A0-A6 tables and bootstrap
scripts/supp_report.py               Markdown/CSV and SVG figures
scripts/supp_run_all.py              orchestrated stages
```

## Final outputs

- [Final experiment report](results/FINAL_EXPERIMENT_REPORT.md)
- [Qwen2.5 rerun: four-group quality, latency and audit-token report](results/qwen25_budget_rerun_20261022/QWEN25_BUDGET_AUDIT_REPORT.md)
- [Qwen2.5 rerun checkpoint manifest](results/qwen25_budget_rerun_20261022/checkpoint_manifest.json)
- [Table 9: online budget audit](results/tables/table9_online_budget_audit.md)
- [A0-A6 ablation quality, tokens, latency and API cost](results/tables/table_ablation_budget.md)
- [Scale quality and deployment cost](results/tables/table_scale_quality_cost.md)
- [Machine-readable scale results](results/scale_complete/scale_complete.json)
- [Server cleanup verification](results/SERVER_CLEANUP_STATUS.md)

See [GPU_RUNBOOK.md](GPU_RUNBOOK.md) for the exact GPU sequence and [TASK_BREAKDOWN.md](TASK_BREAKDOWN.md) for the work breakdown.

## Machine safety

- Running without `--execute` never loads a model, downloads weights, or calls an API.
- Model weights are intentionally not stored by this preparation work.
- Raw `q+y` copies under `prepared/raw/` and trained/run artifacts under `runs/` are ignored by git.
- Raw API audit responses, model weights, checkpoints, training logs and raw `q+y` remain local and are ignored by git.
- Only aggregate token/latency/cost records needed to regenerate the public tables are committed.
- The frozen calibration is selected before E6 test scoring. Do not tune the threshold on the 384-row test set.
- Invalid or missing reviewer output keeps the automatic decision and is counted as a failed audit, never as a correct audit. Set `HF_HOME` on the GPU host before training so the five base models use one controlled cache location.
