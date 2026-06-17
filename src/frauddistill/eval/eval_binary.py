from __future__ import annotations

import argparse
import json

from frauddistill.eval.metrics import binary_metrics
from frauddistill.utils.io import read_jsonl


def eval_prediction_rows(rows: list[dict]) -> dict[str, float]:
    y_true = [row.get("gold_label") or row.get("label") for row in rows]
    y_pred = [row.get("pred_label") or row.get("teacher_label") for row in rows]
    y_score = [float(row.get("pred_score", row.get("teacher_score"))) for row in rows if row.get("pred_score", row.get("teacher_score")) is not None]
    return binary_metrics(y_true, y_pred, y_score if len(y_score) == len(rows) else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", required=True)
    parser.add_argument("--output_file")
    args = parser.parse_args()
    metrics = eval_prediction_rows(list(read_jsonl(args.prediction_file)))
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
