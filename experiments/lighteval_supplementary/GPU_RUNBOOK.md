# GPU Runbook

> Status: completed. This runbook is retained for reproducing or resuming the GPU pipeline. The examples below include the memory-constrained 7B path; final results and cleanup status are in `results/FINAL_EXPERIMENT_REPORT.md` and `results/SERVER_CLEANUP_STATUS.md`.

This guide documents the reproducible GPU execution path, including the 24 GB memory-constrained 7B recovery procedure used in the completed run.

## 0. Transfer policy

The GPU host must contain the private training and E6 inputs plus the reusable DeepSeek adapter/tokenizer. Qwen base weights are downloaded only on the GPU host. All ten scale runs are complete; the 7B commands below illustrate the low-memory resume path if a rerun is required.

```bash
export HF_HOME=/home/<remote-user>/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:64
```

## 1. Validate frozen assets without loading a model

```bash
python scripts/supp_prepare.py
python scripts/supp_run_all.py --stage preflight
python scripts/supp_scale_train.py --models qwen25_7b
python scripts/supp_scale_evaluate.py --model-id qwen25_7b --dry-run
```

Do not continue if counts, IDs or hashes fail. The E6 audit target threshold is 0.2323 and the real-audit prefixes are 19/38/57/76.

## 2. Example: train the memory-constrained 7B model

```bash
python scripts/supp_scale_train.py \
  --models qwen25_7b --device cuda --dtype bf16 --execute
```

The command resumes from `resume.pt`; completed rows are not replayed. For 24 GB operation, the trainer freezes the base in BF16, autocasts activations, keeps trainable LoRA/head master weights in FP32, offloads AdamW moments during dev evaluation and stores the best trainable snapshot on CPU.

## 3. Select and evaluate 7B

```bash
python scripts/supp_scale_evaluate.py --model-id qwen25_7b \
  --device cuda --dtype bf16 --batch-size 1 --gpu-hour-cny 0
```

Replace `0` with the actual hourly price if required. The run selects checkpoints on the development set, calibrates on 96 relation plus 60 hard-safe rows, scores all frozen test sets and records latency, throughput, VRAM, tokens and checkpoint SHA256.

Outputs are written under `experiments/lighteval_supplementary/results/scale/qwen25_7b/`. Copy this directory to the local host immediately after success.

## 4. Existing audit experiment

The A0, real DeepSeek v2 audit, 10,000 family bootstrap and A0-A6 replay are already complete. The current paper-facing Table 9 is the user-requested Silver/reference-label replacement oracle simulation at 5%, 10%, 15%, 20%, 25% and 30%. The real DeepSeek v2 audit at 5%-20% is retained as a separate sensitivity analysis.

Key files:

```text
experiments/lighteval_supplementary/results/ablation_oracle/budget_sensitivity/table_extended_oracle_budget.md
experiments/lighteval_supplementary/results/tables/table9_online_budget_audit.md
experiments/lighteval_supplementary/results/tables/table9_real_api_audit.md
```

## 5. Final local assembly

After copying 7B results:

```bash
python scripts/supp_record_run.py --model-id qwen25_7b --run-dir experiments/lighteval_supplementary/runs/qwen25_7b/final_distilled_student --result-dir experiments/lighteval_supplementary/results/scale/qwen25_7b
python scripts/supp_report.py
python scripts/supp_compare_scales.py
python -m pytest -q
```

Run `supp_record_run.py` on the server before deleting the checkpoint, then copy the resulting JSON locally.

## 6. Non-negotiable scientific checks

- A3-A6 use the same frozen E6 384-row test and threshold 0.2323.
- A6 is the first 76 rows used by A3-A5.
- Audit selection is frozen before reviewer calls.
- Test labels never influence calibration, audit ranking or checkpoint selection.
- Invalid audits fall back to the automatic decision.
- Cache hits have zero incremental API cost and zero online audit latency.
- The paper-facing Table 9 remains explicitly labeled as a Silver/reference-label replacement oracle simulation; it is not a real external API audit.
- Real DeepSeek v2 audit rows must never be substituted into the oracle-simulation table. The historical K=77 oracle replay is 0.807208.
- Every model result is generated from machine-readable artifacts and accompanied by checkpoint SHA256 and source logs.
