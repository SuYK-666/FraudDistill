"""Create a compact, local-readable checkpoint and result record for one scale run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import REPO, read_json, sha256_file, write_json

SUPP_DIR = REPO / "experiments" / "lighteval_supplementary"


def _observed_optimizer_steps(run_dir: Path, training: dict) -> int | None:
    """Read the final global optimizer step without retaining training logs locally."""
    for key in ("optimizer_steps", "global_step", "steps_completed"):
        value = training.get(key)
        if value is not None:
            return int(value)
    log_path = run_dir / "step_log.jsonl"
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        try:
            return int(json.loads(line)["global_step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None

def build_record(model_id: str, run_dir: Path, result_dir: Path) -> dict:
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    checkpoint = Path(summary["checkpoint"])
    adapter = checkpoint / "adapter_model.safetensors"
    training_metrics = run_dir / "final_distill_metrics.json"
    training = read_json(training_metrics) if training_metrics.exists() else {}
    record = {
        "model_id": model_id,
        "checkpoint": str(checkpoint.relative_to(REPO)) if checkpoint.is_relative_to(REPO) else str(checkpoint),
        "checkpoint_step": int("".join(ch for ch in checkpoint.name if ch.isdigit()) or 0),
        "checkpoint_sha256": sha256_file(adapter) if adapter.exists() else None,
        "checkpoint_files": sorted(path.name for path in checkpoint.iterdir() if path.is_file()),
        "optimizer_steps_observed": _observed_optimizer_steps(run_dir, training),
        "training": {
            "wall_seconds": training.get("wall_seconds"),
            "peak_vram_gb": training.get("peak_vram_gb"),
            "train_input_tokens": training.get("train_input_tokens"),
            "rows": training.get("rows"),
            "epochs": training.get("epochs", 2),
            "device": training.get("device"),
            "dtype": training.get("dtype"),
        },
        "threshold_selection": summary.get("threshold_selection"),
        "metrics": summary.get("metrics"),
        "cost": summary.get("cost"),
        "artifacts": sorted(path.name for path in result_dir.iterdir() if path.is_file()),
        "note": "The checkpoint hash and metrics were recorded locally. The remote model/run checkpoint is removed after final evaluation; only frozen result artifacts remain on the local host.",
    }
    write_json(result_dir / "checkpoint_record.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args(argv)
    result = build_record(args.model_id, Path(args.run_dir), Path(args.result_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
