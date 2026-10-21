# -*- coding: utf-8 -*-
"""Reconcile per-attempt training timing with end-to-end wall time."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frauddistill.supplementary.common import REPO, read_json, write_json

SCALE_ROOT = REPO / "experiments" / "lighteval_supplementary" / "results" / "scale"
MODEL_IDS = [
    "qwen25_7b", "qwen3_0_6b", "qwen3_1_7b", "qwen3_4b",
    "qwen35_0_8b", "qwen35_2b", "qwen35_4b",
]


def _step_span(run_dir: Path) -> float | None:
    path = run_dir / "step_log.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2:
        return None
    return float(rows[-1]["ts"]) - float(rows[0]["ts"])


def _driver_attempts(run_dir: Path) -> tuple[int, int]:
    path = run_dir / "train_retry_driver.log"
    if not path.exists():
        return (1, 0)
    text = path.read_text(encoding="utf-8-sig")
    attempts = len(re.findall(r"TRAINING_ATTEMPT_BEGIN", text))
    retries = len(re.findall(r"CUDA_MEMORY_RETRY", text))
    return (max(attempts, 1), retries)


def reconcile(model_id: str) -> dict:
    result_dir = SCALE_ROOT / model_id
    summary_path = result_dir / "summary.json"
    cost_path = result_dir / "cost.json"
    record_path = result_dir / "checkpoint_record.json"
    summary = read_json(summary_path)
    cost = summary["cost"]
    last_attempt = float(cost["training"].get("wall_seconds") or 0.0)
    span = _step_span(result_dir)
    attempts, retries = _driver_attempts(result_dir)
    cumulative = max(last_attempt, span or 0.0)
    eval_seconds = float(cost["evaluation"].get("wall_seconds") or 0.0)

    training = cost["training"]
    training["last_attempt_wall_seconds"] = round(last_attempt, 3)
    training["step_timestamp_span_seconds"] = round(span, 3) if span is not None else None
    training["cumulative_wall_seconds"] = round(cumulative, 3)
    training["wall_seconds"] = round(cumulative, 3)
    training["wall_hours"] = round(cumulative / 3600.0, 5)
    training["billable_gpu_hours"] = round(cumulative / 3600.0, 5)
    training["attempts"] = attempts
    training["cuda_memory_retries"] = retries
    training["cost_basis"] = "step timestamps spanning resumptions, model reloads and retries; endpoint save excluded"
    cost["total_gpu_hours"] = round((cumulative + eval_seconds) / 3600.0, 5)
    price = cost.get("gpu_hour_cny")
    if price is not None:
        cost["estimated_total_gpu_cny"] = round(cost["total_gpu_hours"] * float(price), 6)
    write_json(summary_path, summary)
    write_json(cost_path, cost)
    if record_path.exists():
        record = read_json(record_path)
        record["cost"] = cost
        write_json(record_path, record)
    return {
        "model_id": model_id,
        "last_attempt_s": round(last_attempt, 3),
        "step_span_s": round(span, 3) if span is not None else None,
        "cumulative_s": round(cumulative, 3),
        "attempts": attempts,
        "retries": retries,
        "total_gpu_hours": cost["total_gpu_hours"],
    }


def main() -> int:
    rows = [reconcile(model_id) for model_id in MODEL_IDS]
    write_json(SCALE_ROOT.parent / "scale_complete" / "observed_training_costs.json", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
