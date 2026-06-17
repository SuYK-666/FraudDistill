from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from frauddistill.eval.metrics import binary_metrics
from frauddistill.utils.io import read_jsonl, write_jsonl


def finalize(output_dir: str) -> dict:
    out = Path(output_dir)
    pred_dir = out / "predictions"
    table_dir = out / "tables"
    qy_parts = sorted(pred_dir.glob("single_judge_api_qwen_q_y_part*_predictions.jsonl"))
    qy_parts.extend(sorted(pred_dir.glob("single_judge_api_qwen_q_y_q_y_part*_predictions.jsonl")))
    if qy_parts:
        rows = []
        for part in qy_parts:
            rows.extend(read_jsonl(part))
        write_jsonl(pred_dir / "single_judge_api_qwen_q_y_predictions.jsonl", rows)

    prediction_files = sorted(pred_dir.glob("*_predictions.jsonl"))
    metrics_rows = []
    group_rows = []
    for file in prediction_files:
        if "_part" in file.name:
            continue
        method, mode = _parse_name(file.name)
        rows = list(read_jsonl(file))
        metrics = _metrics(rows)
        metrics_rows.append({"input_mode": mode, "method": method, **metrics})
        group_rows.extend(_group_metrics(rows, mode, method))
    metrics_rows.sort(key=lambda row: (row["input_mode"], row["method"]))
    group_rows.sort(key=lambda row: (row["input_mode"], row["method"], row["group"], row["value"]))
    _write_csv(table_dir / "exp1_all_metrics.csv", metrics_rows)
    _write_csv(table_dir / "exp1_all_group_metrics.csv", group_rows)
    summary = {"metrics": metrics_rows, "group_metrics": group_rows}
    (table_dir / "exp1_all_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _parse_name(name: str) -> tuple[str, str]:
    stem = name.removesuffix("_predictions.jsonl")
    for mode in ["q_only", "y_only", "q_y"]:
        suffix = f"_{mode}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], mode
    return stem, "unknown"


def _metrics(rows: list[dict]) -> dict[str, float]:
    return binary_metrics(
        [row["gold_label"] for row in rows],
        [row["pred_label"] for row in rows],
        [float(row["pred_score"]) for row in rows],
    )


def _group_metrics(rows: list[dict], mode: str, method: str) -> list[dict]:
    output = []
    groups = {
        "fraud_category": lambda row: row.get("metadata", {}).get("fraud_category", "unknown"),
        "language": lambda row: row.get("language", "unknown"),
        "fraudr1_variant": lambda row: row.get("metadata", {}).get("fraudr1_variant", "unknown"),
    }
    for group, getter in groups.items():
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(str(getter(row)), []).append(row)
        for value, subset in buckets.items():
            output.append({"input_mode": mode, "method": method, "group": group, "value": value, "n": len(subset), **_metrics(subset)})
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["input_mode", "method", "group", "value", "n", "accuracy", "precision", "recall", "macro_f1", "fpr", "fnr", "auroc", "auprc"]
    fieldnames = [field for field in preferred if field in fieldnames] + [field for field in fieldnames if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/exp1_final")
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_dir)["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
