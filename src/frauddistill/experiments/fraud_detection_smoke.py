from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher
from frauddistill.eval.metrics import binary_metrics
from frauddistill.eval.rule_baseline import predict_rule
from frauddistill.utils.io import read_jsonl, write_jsonl
from frauddistill.utils.text import build_detector_input


def run_input_ablation(rows: list[dict]) -> dict:
    modes = ["q_only", "y_only", "q_y"]
    results = {}
    for mode in modes:
        mode_rows = [_with_mode_text(row, mode) for row in rows]
        preds = [predict_rule(row) for row in mode_rows]
        results[mode] = _metrics(rows, preds)
    return results


def run_teacher_distillation_prep(rows: list[dict]) -> tuple[dict, list[dict]]:
    teacher = MultiAgentTeacher(client=None)
    signals = [teacher.run(row) for row in rows]
    teacher_preds = [
        {"id": row["id"], "pred_label": row["teacher_label"], "pred_score": row["teacher_score"], "pred_type": row["teacher_type"]}
        for row in signals
    ]
    rule_preds = [predict_rule(row) for row in rows]
    return {
        "rule": _metrics(rows, rule_preds),
        "multi_agent_teacher_offline": _metrics(rows, teacher_preds),
        "teacher_signal_count": len(signals),
    }, signals


def run_deployment_generalization_prep(rows: list[dict]) -> dict:
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        category = str(row.get("metadata", {}).get("fraud_category", "unknown"))
        by_category.setdefault(category, []).append(row)
    category_metrics = {}
    for category, subset in by_category.items():
        if len(subset) >= 2:
            category_metrics[category] = _metrics(subset, [predict_rule(row) for row in subset])
    low_fpr = _recall_at_fpr(rows, predict_rule, thresholds=[0.10, 0.20, 0.25])
    return {"category_smoke": category_metrics, "low_fpr_smoke": low_fpr}


def run_all(input_file: str | Path, output_dir: str | Path, limit: int = 12) -> dict:
    rows = list(read_jsonl(input_file))[:limit]
    if not rows:
        raise ValueError(f"{input_file} is empty")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exp2, signals = run_teacher_distillation_prep(rows)
    results = {
        "sample_count": len(rows),
        "exp1_input_ablation": run_input_ablation(rows),
        "exp2_teacher_distillation_prep": exp2,
        "exp3_deployment_generalization_prep": run_deployment_generalization_prep(rows),
    }
    (out / "fraud_detection_smoke_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out / "rule_predictions.jsonl", [predict_rule(row) for row in rows])
    write_jsonl(out / "teacher_signals.jsonl", signals)
    return results


def _with_mode_text(row: dict, mode: str) -> dict:
    rendered = build_detector_input(row["user_query"], row["target_model_answer"], row.get("context"), mode)
    copy = dict(row)
    if mode == "q_only":
        copy["target_model_answer"] = rendered
    elif mode == "y_only":
        copy["user_query"] = ""
        copy["target_model_answer"] = rendered
    else:
        copy["target_model_answer"] = rendered
    return copy


def _metrics(rows: list[dict], preds: list[dict]) -> dict:
    return binary_metrics(
        [row["gold_label"] for row in rows],
        [pred["pred_label"] for pred in preds],
        [float(pred["pred_score"]) for pred in preds],
    )


def _recall_at_fpr(rows: list[dict], predictor: Callable[[dict], dict], thresholds: list[float]) -> dict:
    scored = [(row, predictor(row)) for row in rows]
    result = {}
    for threshold in thresholds:
        cutoff = 1.0
        best_recall = 0.0
        for candidate in sorted({float(pred["pred_score"]) for _, pred in scored}, reverse=True):
            preds = [
                {**pred, "pred_label": "unsafe" if float(pred["pred_score"]) >= candidate else "safe"}
                for _, pred in scored
            ]
            metrics = _metrics([row for row, _ in scored], preds)
            if metrics["fpr"] <= threshold and metrics["recall"] >= best_recall:
                best_recall = metrics["recall"]
                cutoff = candidate
        result[f"recall_at_fpr_{threshold:.2f}"] = best_recall
        result[f"threshold_at_fpr_{threshold:.2f}"] = cutoff
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/unified/fraud_focus_smoke.jsonl")
    parser.add_argument("--output_dir", default="outputs/fraud_detection_smoke")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    results = run_all(args.input_file, args.output_dir, args.limit)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
