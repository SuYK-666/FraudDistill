# -*- coding: utf-8 -*-
"""Parallel runner for the E4 Base-1.5B zero-shot stage (300-row subset).

Starts the base zero-shot stage while the main inference pipeline is still
busy on earlier stages. Uses the exact same sampling (seed + n from yaml) and
the exact same output writer as the main pipeline, so when the main pipeline
reaches its own base stage it detects the existing file and skips it.

Usage: python scripts/e4e5_run_base_parallel.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402
from e4e5_run_inference import base_zeroshot_on_rows  # noqa: E402
from frauddistill.e4e5_v2.schemas import read_jsonl  # noqa: E402

PROTO = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs/experiments/exp4_unseen_student_v2.yaml").read_text(encoding="utf-8"))
    n = cfg["panel"]["pilot_n"]
    seed = cfg["seed"]
    out = PROTO / "predictions" / "base_zeroshot.jsonl"
    if out.exists():
        n_existing = sum(1 for _ in open(out, encoding="utf-8"))
        print(f"[base-parallel] already exists ({n_existing} rows), skip", flush=True)
        return
    rows = read_jsonl(PROTO / "manifests" / "frozen_test.jsonl")
    print(f"[base-parallel] start n={n} seed={seed} rows_total={len(rows)} -> {out}", flush=True)
    t0 = time.time()
    base_zeroshot_on_rows(rows, n=n, seed=seed, out_path=out)
    print(f"[base-parallel] DONE in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
