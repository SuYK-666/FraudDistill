"""Shared paths and constants for exp2 cross-benchmark experiment (guide 2026-08-05)."""
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

SEED = 20260805

ERROR_TYPES = {"fraud_assistance", "refusal_failure", "over_refusal"}

MODEL_TEACHER = "deepseek-v4-flash"
MODEL_JUDGE = "deepseek-v4-flash"      # official-judge protocol runner (GPT-4o-mini protocol prompt)
MODEL_CHECKER = "deepseek-v4-flash"    # OR-Bench response checker protocol runner
MODEL_TARGET = "qwen-plus"             # unified target model for answer generation
MODEL_AUDIT = "deepseek-v4-flash"      # blinded audit annotator A
MODEL_AUDIT_B = "deepseek-v4-pro"      # blinded audit annotator B (independent model)

CONCURRENCY = 120

# Guide 13.2: frozen T6 Evidence MAT output caps for exp2.
T6_MAX_TOKENS = {"fraud": 160, "refusal": 160, "context": 140, "arbiter": 160}

# Guide 21: budget.
BUDGET_HARD_CAP_RMB = 36.0
BUDGET_RESERVE_RMB = 4.0

# Exp3 exposure sources for the overlap audit (guide 7.2).
EXP3_DATASET = REPO_ROOT / "data" / "prepared" / "exp3_agent_distillation" / "exp3_dataset.jsonl"
EXP3_STUDENT_MANIFESTS = [
    REPO_ROOT / "data" / "prepared" / "exp3_neural_student" / "train_manifest.jsonl",
    REPO_ROOT / "data" / "prepared" / "exp3_neural_student" / "train_manifest_expanded.jsonl",
]
RESERVED_EXP2_IDS = REPO_ROOT / "data" / "splits" / "reserved_exp2_test_ids.json"

MANIFEST_DIR = EXPERIMENT_DIR / "manifests"
METRICS_DIR = EXPERIMENT_DIR / "metrics"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
TEACHER_T6_DIR = EXPERIMENT_DIR / "teacher_predictions_t6"
EXP3_AGENT_DIR = REPO_ROOT / "experiments" / "exp3_agent_distillation_ablation" / "outputs" / "agent_predictions"


def benchmark_dir(benchmark: str) -> Path:
    return EXPERIMENT_DIR / benchmark


def out_dir(benchmark: str, *parts: str) -> Path:
    d = benchmark_dir(benchmark)
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d
