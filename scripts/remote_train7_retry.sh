#!/usr/bin/env bash
# Resume-only retry loop for the memory-constrained Qwen2.5-7B run.
set -u

repo_root="${1:?repo root required}"
run_root="$repo_root/experiments/lighteval_supplementary/runs/qwen25_7b"
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

  echo "TRAINING_ATTEMPT_BEGIN attempt=$attempt $(date '+%F %T')"
  python scripts/supp_scale_train.py \
    --models qwen25_7b --device cuda --dtype bf16 --execute \
    >> "$log_dir/train7b_retry.out" 2>&1
  rc=$?

  if [[ -f "$student_dir/final_distill_metrics.json" ]]; then
    echo "TRAINING_DONE attempt=$attempt"
    exit 0
  fi

  if [[ -f "$err_path" ]] && tail -n 300 "$err_path" | grep -Eq 'CUDA out of memory|INTERNAL ASSERT'; then
    echo "CUDA_MEMORY_RETRY attempt=$attempt rc=$rc $(date '+%F %T')"
    sleep 8
    continue
  fi

  echo "TRAINING_STOPPED_NON_OOM attempt=$attempt rc=$rc $(date '+%F %T')"
  exit "$rc"
done
