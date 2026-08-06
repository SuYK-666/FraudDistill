"""Shared paths and constants for exp2 cross-benchmark experiment (full-coverage guide 2026-08-06)."""
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

SEED = 20260806

ERROR_TYPES = {"fraud_assistance", "refusal_failure", "over_refusal"}

MODEL_TEACHER = "deepseek-v4-flash"
MODEL_JUDGE = "deepseek-v4-flash"      # official-judge protocol runner (GPT-4o-mini protocol prompt)
MODEL_CHECKER = "deepseek-v4-flash"    # OR-Bench response checker protocol runner
MODEL_TARGET = "qwen-plus"             # unified target model for answer generation
MODEL_AUDIT = "deepseek-v4-flash"      # blinded audit annotator A
MODEL_AUDIT_B = "deepseek-v4-pro"      # blinded audit annotator B (independent model)

CONCURRENCY = 120

# Guide 13.2: frozen T6 Evidence MAT output caps for exp2.
# 2026-08-06 bugfix round 2 (user instruction): caps enlarged further so the
# specialist JSON can never be truncated at the API level. Model output for a
# full evidence JSON is ~300-500 tokens/agent; 2048/1536 leaves a wide margin
# for long Chinese spans. Parse failures are surfaced as parse_failed/abstain.
T6_MAX_TOKENS = {"fraud": 2048, "refusal": 2048, "context": 1536, "arbiter": 1536}

# Guide 20/21 + 2026-08-06 user amendment: hard stop at 140 RMB total for
# this round (32.54 already consumed by the truncated run; re-run budgeted
# within the remainder). Monitoring milestones at 100 / 120 / 136 RMB.
BUDGET_HARD_CAP_RMB = 140.0
BUDGET_RESERVE_RMB = 4.0
BUDGET_MILESTONES = (100.0, 120.0, 136.0)

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
AUDIT_DIR = EXPERIMENT_DIR / "audit"
TEACHER_T6_DIR = EXPERIMENT_DIR / "teacher_predictions_t6"
EXP3_AGENT_DIR = REPO_ROOT / "experiments" / "exp3_agent_distillation_ablation" / "outputs" / "agent_predictions"

# Full-coverage manifest suite (guide 26).
FULL_MANIFEST = MANIFEST_DIR / "full_manifest.jsonl"
AEGIS_RESPONSE_MANIFEST = MANIFEST_DIR / "aegis_response_manifest.jsonl"
AEGIS_PROMPT_MANIFEST = MANIFEST_DIR / "aegis_prompt_manifest.jsonl"
FRAUD_ALIGNED_MANIFEST = MANIFEST_DIR / "fraud_aligned_manifest.jsonl"
AEGIS_VALIDATION_MANIFEST = MANIFEST_DIR / "aegis_validation_manifest.jsonl"
CACHE_AUDIT_FILE = AUDIT_DIR / "cache_audit.json"
FULL_POOL_SUMMARY_FILE = AUDIT_DIR / "full_pool_summary.json"
BUDGET_FILE = AUDIT_DIR / "budget_state.json"
BUDGET_HISTORY_FILE = AUDIT_DIR / "budget_history.json"

# Prompt-only auxiliary evaluation (guide 3.2): input-risk fields only.
PROMPT_RISK_THRESHOLD = 0.5  # frozen categorical point; calibration may refine

# DNA official taxonomy (guide 5.3 / 10.4): fraud-aligned risk areas.
DNA_FRAUD_ALIGNED_AREAS = {"Malicious Uses"}
AEGIS_FRAUD_ALIGNED_CATEGORIES = {
    "Fraud/Deception", "Manipulation", "PII/Privacy", "Impersonation", "Illegal Activity",
}

# Silver labels from the 2026-08-05 blind panel (guide 15): LLM-panel labels,
# NOT human gold. Used only for the audited evaluator-analysis rows.
SILVER_LABEL_TAG = "20260805"


def benchmark_dir(benchmark: str) -> Path:
    return EXPERIMENT_DIR / benchmark


def out_dir(benchmark: str, *parts: str) -> Path:
    d = benchmark_dir(benchmark)
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d
