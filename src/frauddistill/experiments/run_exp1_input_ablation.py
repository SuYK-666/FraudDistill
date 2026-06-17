from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher
from frauddistill.eval.metrics import binary_metrics
from frauddistill.eval.rule_baseline import predict_rule
from frauddistill.utils.io import read_jsonl, write_jsonl

INPUT_MODES = ["q_only", "y_only", "q_y"]


def run_exp1(input_file: str, output_dir: str, limit: int | None = None) -> dict:
    rows = list(read_jsonl(input_file))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("input dataset is empty")
    out = Path(output_dir)
    pred_dir = out / "predictions"
    table_dir = out / "tables"
    pred_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    methods: dict[str, Callable[[dict], dict]] = {
        "rule": predict_rule,
        "multi_agent_teacher_offline": _teacher_predictor(),
    }
    metrics_rows = []
    group_rows = []
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for mode in INPUT_MODES:
        mode_rows = [_project_input(row, mode) for row in rows]
        all_metrics[mode] = {}
        for method, predictor in methods.items():
            predictions = [_attach_gold(source, predictor(projected)) for source, projected in zip(rows, mode_rows)]
            write_jsonl(pred_dir / f"{method}_{mode}_predictions.jsonl", predictions)
            metrics = _metrics_from_predictions(predictions)
            all_metrics[mode][method] = metrics
            metrics_rows.append({"input_mode": mode, "method": method, **metrics})
            group_rows.extend(_group_metrics_rows(predictions, mode, method))

    summary = {
        "experiment": "exp1_input_ablation_fraud",
        "dataset": str(input_file),
        "sample_count": len(rows),
        "input_modes": INPUT_MODES,
        "methods": list(methods),
        "metrics": all_metrics,
        "dataset_stats": _dataset_stats(rows),
    }
    (table_dir / "exp1_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_metrics_csv(table_dir / "exp1_metrics.csv", metrics_rows)
    _write_group_metrics_csv(table_dir / "exp1_group_metrics.csv", group_rows)
    (table_dir / "exp1_group_metrics.json").write_text(json.dumps(group_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _teacher_predictor() -> Callable[[dict], dict]:
    teacher = MultiAgentTeacher(client=None)

    def predict(row: dict) -> dict:
        signal = teacher.run(row)
        return {
            "id": row["id"],
            "pred_label": signal["teacher_label"],
            "pred_score": signal["teacher_score"],
            "pred_type": signal["teacher_type"],
        }

    return predict


def _project_input(row: dict, mode: str) -> dict:
    projected = dict(row)
    if mode == "q_only":
        projected["target_model_answer"] = ""
    elif mode == "y_only":
        projected["user_query"] = ""
    elif mode == "q_y":
        pass
    else:
        raise ValueError(f"unknown mode: {mode}")
    return projected


def _attach_gold(source: dict, pred: dict) -> dict:
    return {
        "id": source["id"],
        "gold_label": source["gold_label"],
        "gold_risk_type": source.get("gold_risk_type"),
        "pred_label": pred["pred_label"],
        "pred_score": pred["pred_score"],
        "pred_type": pred["pred_type"],
        "source": source.get("source"),
        "language": source.get("language"),
        "metadata": source.get("metadata", {}),
    }


def _metrics_from_predictions(predictions: list[dict]) -> dict[str, float]:
    return binary_metrics(
        [row["gold_label"] for row in predictions],
        [row["pred_label"] for row in predictions],
        [float(row["pred_score"]) for row in predictions],
    )


def _dataset_stats(rows: list[dict]) -> dict:
    stats: dict[str, dict[str, int]] = {"by_label": {}, "by_language": {}, "by_category": {}, "by_variant": {}}
    for row in rows:
        _bump(stats["by_label"], row.get("gold_label", "unknown"))
        _bump(stats["by_language"], row.get("language", "unknown"))
        metadata = row.get("metadata", {})
        _bump(stats["by_category"], metadata.get("fraud_category", "unknown"))
        _bump(stats["by_variant"], metadata.get("fraudr1_variant", "unknown"))
    return stats


def _bump(counter: dict[str, int], key: object) -> None:
    text = str(key)
    counter[text] = counter.get(text, 0) + 1


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["input_mode", "method", "accuracy", "precision", "recall", "macro_f1", "fpr", "fnr", "auroc", "auprc"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _group_metrics_rows(predictions: list[dict], input_mode: str, method: str) -> list[dict]:
    output = []
    group_specs = {
        "fraud_category": lambda row: row.get("metadata", {}).get("fraud_category", "unknown"),
        "language": lambda row: row.get("language", "unknown"),
        "fraudr1_variant": lambda row: row.get("metadata", {}).get("fraudr1_variant", "unknown"),
    }
    for group_name, getter in group_specs.items():
        grouped: dict[str, list[dict]] = {}
        for row in predictions:
            grouped.setdefault(str(getter(row)), []).append(row)
        for group_value, rows in grouped.items():
            metrics = _metrics_from_predictions(rows)
            output.append(
                {
                    "input_mode": input_mode,
                    "method": method,
                    "group": group_name,
                    "value": group_value,
                    "n": len(rows),
                    **metrics,
                }
            )
    return output


def _write_group_metrics_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "input_mode",
        "method",
        "group",
        "value",
        "n",
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "fpr",
        "fnr",
        "auroc",
        "auprc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    summary = run_exp1(args.input_file, args.output_dir, args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
