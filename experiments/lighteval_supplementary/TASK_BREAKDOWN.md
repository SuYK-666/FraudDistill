# Task Breakdown

> Status: completed. This file is retained as the frozen protocol checklist; phrases such as “critical path” describe the execution plan, not pending work. Final results are in `results/FINAL_EXPERIMENT_REPORT.md`.

## A. Shared freeze and protocol

| ID | Task | Dependency | Output |
|---|---|---|---|
| F1 | Freeze 4,747-row training manifest | existing E3 final manifest | hash and count audit |
| F2 | Freeze 1,047 dev and 1,262 core test rows | existing E3 dataset | `core_dev.jsonl`, `core_test.jsonl` |
| F3 | Freeze 96 relation calibration + 60 hard-safe calibration | E6 | calibration bundle |
| F4 | Freeze 384 relation test + 180 hard-safe test | E6 | test bundle |
| F5 | Verify ID overlap and SHA256 | F1-F4 | `asset_lock.json` |
| F6 | Reproduce source and target thresholds | frozen E6 threshold record | 0.5622 / 0.2323 |
| F7 | Freeze maximum 76-item audit list | F6 + existing student scores | `audit_selection_manifest.jsonl` |
| F8 | Build 19/38/57/76 strict prefixes | F7 | `ablation_plan.json` |
| F9 | Strip score/label metadata from reviewer input | F7 | `prepared/raw/audit_inputs.jsonl` |
| F10 | Audit E5 cache compatibility | E5 cache + E6 q+y hashes | 598 inspected, 0 exact hits |

## B. Experiment 1: scale

| ID | Task | Dependency | Output |
|---|---|---|---|
| S1 | Validate model registry and memory thresholds | F1-F4 | `scale_model_specs.json` |
| S2 | Train Qwen2.5-0.5B-Instruct, seed 11 | GPU + S1 | LoRA checkpoints |
| S3 | Train Qwen2.5-1.5B-Instruct, seed 11 | GPU + S1 | LoRA checkpoints |
| S4 | Train Qwen2.5-3B-Instruct, seed 11 | GPU + S1 | LoRA checkpoints |
| S5 | Train Qwen2.5-7B-Instruct, seed 11 | GPU + S1 | LoRA checkpoints |
| S7 | Reuse DeepSeek-1.5B `best_step120` | existing checkpoint | no retraining |
| S8 | Fast dev selection then top-3 full-dev selection | per-model checkpoints | `checkpoint_selection.json` |
| S9 | Calibrate on 96 relation + 60 hard-safe rows | selected checkpoint | `threshold_selection.json` |
| S10 | Score 1,262 core test | S9 | core metrics + predictions |
| S11 | Score 384 E6 relation test | S9 | online metrics |
| S12 | Score 180 hard-safe test | S9 | hard-safe FPR |
| S13 | Measure batch=1 P50/P95, fixed-batch throughput, peak VRAM | S8 | `cost.json` |
| S14 | Merge five new scales + reused DeepSeek baseline | S10-S13 | scale table and curve |

## C. Experiment 2: progression and audit

| ID | Task | Dependency | Output |
|---|---|---|---|
| A0 | Score untrained DeepSeek-1.5B class interface on E6 384 | GPU or existing compatible run | A0 predictions |
| A1 | Apply trained-student source threshold 0.5622 | existing E6 scores | A1 metrics |
| A2 | Apply target threshold 0.2323 | frozen calibration | A2 metrics |
| A3 | Apply first 19 valid audits | real audit results | A3 metrics/cost |
| A4 | Apply first 38 audits | A3 prefix + 19 new | A4 metrics/cost |
| A5 | Apply first 57 audits | A4 prefix + 19 new | A5 metrics/cost |
| A6 | Apply all 76 audits | A5 prefix + 19 new | A6 metrics/cost |
| A7 | Count fixed errors, introduced errors and net corrections | A2-A6 | correction audit |
| A8 | Run 10,000 paired family bootstrap vs A2 | frozen final predictions | interval JSON |
| A9 | Render Table 9 and budget figure | A0-A8 | report artifacts |

## D. Acceptance checks before submission

| ID | Check |
|---|---|
| Q1 | Same 384 E6 IDs and same family IDs for every policy |
| Q2 | No threshold selected using the 384-row test set |
| Q3 | Audit prefix containment: `A3 ⊂ A4 ⊂ A5 ⊂ A6` and sizes 19/38/57/76 |
| Q4 | Reviewer input contains no student score, automatic label or reference label |
| Q5 | Failed/invalid audits fall back to automatic prediction and remain in the denominator |
| Q6 | Cache read time is not counted as online review latency |
| Q7 | Summary tokens include input and retry tokens; classifier output tokens are 0 |
| Q8 | All model scales use the same retained-token cap and timing batch |
| Q9 | Weight/model downloads happen only on the GPU host |
| Q10 | Tables are generated from machine-readable artifacts, not hand-copied numbers |

## E. Critical path on GPU

1. Run preparation/preflight.
2. Train the four approved Qwen scales sequentially; the 14B arm is out of scope for this 24 GB server.
3. Evaluate each finished scale immediately so training and scoring can overlap operationally.
4. Run A0 while another scale trains if sufficient GPU memory is available.
5. Freeze the 76 audit list is already complete; only live reviewer calls remain.
6. Replay A3-A6 only after all 76 reviewer results are durable.
7. Run 10,000 bootstrap and generate the final table/figures.
