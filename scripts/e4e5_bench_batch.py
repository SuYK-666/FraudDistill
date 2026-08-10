# -*- coding: utf-8 -*-
"""Quick A/B benchmark: micro_batch size effect on per-row wall time (CPU).

Runs the Final Student checkpoint on a small fixed slice of frozen_test with
micro_batch in {6,12,24} (warmup + timed run each) and prints s/row.
Requires the main inference to be suspended for clean timing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
torch.set_num_threads(16)

from frauddistill.e4e5_v2.student_inference import load_checkpoint, predict_scores  # noqa: E402

CKPT = REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/best_step120"
MANIFEST = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL/manifests/frozen_test.jsonl"

def main() -> None:
    rows = [json.loads(ln) for ln in open(MANIFEST, encoding="utf-8")][:24]
    print(f"loaded {len(rows)} rows; loading checkpoint ...", flush=True)
    t0 = time.time()
    model, tokenizer = load_checkpoint(CKPT, max_length=512)
    print(f"checkpoint loaded in {time.time()-t0:.0f}s", flush=True)
    for mb in (6, 12, 24):
        # warmup
        t0 = time.time()
        predict_scores(model, tokenizer, rows[:8], max_length=512, micro_batch=mb)
        tw = time.time() - t0
        # timed
        t0 = time.time()
        predict_scores(model, tokenizer, rows[:8], max_length=512, micro_batch=mb)
        tt = time.time() - t0
        print(f"micro_batch={mb:2d}: warmup {tw:.1f}s, timed {tt:.1f}s -> {tt/8:.2f} s/row (8 rows)", flush=True)

if __name__ == "__main__":
    main()
