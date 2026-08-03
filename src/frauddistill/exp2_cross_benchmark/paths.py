"""Shared paths and constants for exp2 cross-benchmark experiment."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "exp2_prior_work_comparison"

RAW_FRAUDR1 = REPO_ROOT / "data" / "raw" / "fraudr1"
RAW_ORBENCH = REPO_ROOT / "data" / "raw" / "or_bench"
RAW_DNA = REPO_ROOT / "data" / "raw" / "do_not_answer"
RAW_AEGIS = REPO_ROOT / "data" / "raw" / "aegis"

BENCHMARKS = ["fraudr1", "orbench", "do_not_answer", "aegis2"]

EXPECTED_POOL = {
    "fraudr1": 8564,
    "orbench": 3000,
    "do_not_answer": 5634,
    "aegis2": 1964,
}

SEED = 20260803

ERROR_TYPES = {"fraud_assistance", "refusal_failure", "over_refusal"}

MODEL_TEACHER = "deepseek-v4-flash"
MODEL_JUDGE = "deepseek-v4-flash"      # official-judge protocol runner (GPT-4o-mini protocol prompt)
MODEL_CHECKER = "deepseek-v4-flash"    # OR-Bench response checker protocol runner
MODEL_TARGET = "qwen-plus"             # unified target model for answer generation
MODEL_AUDIT = "deepseek-v4-flash"      # blinded audit annotator

CONCURRENCY = 120


def benchmark_dir(benchmark: str) -> Path:
    return EXPERIMENT_DIR / benchmark


def out_dir(benchmark: str, *parts: str) -> Path:
    d = benchmark_dir(benchmark)
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d
