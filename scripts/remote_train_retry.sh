#!/usr/bin/env bash
# Sequential-model trainer with resume-only retries on CUDA allocator failures.
set -u

repo_root="${1:?repo root required}"
model_id="${2:?model id required}"
python_bin="${3:-python}"
run_root="$repo_root/experiments/lighteval_supplementary/runs/$model_id"
student_dir="$run_root/final_distilled_student"
log_dir="$repo_root/logs"
attempt=0

mkdir -p "$log_dir"
cd "$repo_root" || exit 2

while [[ ! -f "$student_dir/final_distill_metrics.json" ]]; do
  attempt=$((attempt + 1))
  err_path="$run_root/train.err.log"
  if [[ -f "$err_path" ]]; then
    mv "$err_path" "$run_root/train.err.attempt${attempt}.log"
  fi

  echo "TRAINING_ATTEMPT_BEGIN model=$model_id attempt=$attempt $(date '+%F %T')"
  "$python_bin" scripts/supp_scale_train.py \
    --models "$model_id" --device cuda --dtype bf16 --execute \
    >> "$log_dir/${model_id}_train_retry.out" 2>&1
  rc=$?

  if [[ -f "$student_dir/final_distill_metrics.json" ]]; then
    echo "TRAINING_DONE model=$model_id attempt=$attempt"
    exit 0
  fi

  if [[ -f "$err_path" ]] && tail -n 300 "$err_path" | grep -Eq 'CUDA out of memory|INTERNAL ASSERT'; then
    echo "CUDA_MEMORY_RETRY model=$model_id attempt=$attempt rc=$rc $(date '+%F %T')"
    sleep 8
    continue
  fi

  echo "TRAINING_STOPPED_NON_OOM model=$model_id attempt=$attempt rc=$rc $(date '+%F %T')"
  exit "$rc"
done
