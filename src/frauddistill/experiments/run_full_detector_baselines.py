from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from frauddistill.agents.multi_agent_teacher import MultiAgentTeacher
from frauddistill.eval.metrics import binary_metrics
from frauddistill.eval.rule_baseline import predict_rule
from frauddistill.utils.io import read_jsonl, write_jsonl


def run_full_detector_baselines(evaluation_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    eval_dir = Path(evaluation_dir)
    out = Path(output_dir)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(eval_dir.glob("*.jsonl"))
    teacher = MultiAgentTeacher(client=None)
    all_metrics = []
    manifest: dict[str, Any] = {"evaluation_dir": str(eval_dir), "output_dir": str(out), "datasets": {}}

    for dataset_path in datasets:
        dataset_name = dataset_path.stem
        rows = list(read_jsonl(dataset_path))
        rule_preds = []
        teacher_preds = []
        for row in tqdm(rows, desc=f"baseline {dataset_name}"):
            rule = predict_rule(row)
            signal = teacher.run(row)
            rule_preds.append({**rule, "method": "keyword_refusal_rule"})
            teacher_preds.append(
                {
                    "id": row["id"],
                    "pred_label": signal["teacher_label"],
                    "pred_score": signal["teacher_score"],
                    "pred_type": signal["teacher_type"],
                    "method": "frauddistill_teacher_offline",
                }
            )
        write_jsonl(pred_dir / f"{dataset_name}_rule_predictions.jsonl", rule_preds)
        write_jsonl(pred_dir / f"{dataset_name}_teacher_predictions.jsonl", teacher_preds)
        dataset_metrics = []
        for method, preds in [("keyword_refusal_rule", rule_preds), ("frauddistill_teacher_offline", teacher_preds)]:
            metrics = _metrics(rows, preds)
            record = {"dataset": dataset_name, "method": method, "n": len(rows), **metrics}
            dataset_metrics.append(record)
            all_metrics.append(record)
            all_metrics.extend(_grouped_metrics(rows, preds, dataset_name, method, "source"))
            all_metrics.extend(_grouped_metrics(rows, preds, dataset_name, method, "gold_risk_type"))
            all_metrics.extend(_metadata_grouped_metrics(rows, preds, dataset_name, method, "fraud_category"))
        manifest["datasets"][dataset_name] = {
            "rows": len(rows),
            "metrics": dataset_metrics,
            "rule_predictions": str(pred_dir / f"{dataset_name}_rule_predictions.jsonl"),
            "teacher_predictions": str(pred_dir / f"{dataset_name}_teacher_predictions.jsonl"),
        }

    _write_csv(out / "baseline_metrics.csv", all_metrics)
    (out / "baseline_metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "BASELINE_REPORT.md").write_text(_baseline_report(manifest, all_metrics), encoding="utf-8")
    (out / "baseline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _metrics(rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, float]:
    return binary_metrics(
        [str(row["gold_label"]) for row in rows],
        [str(pred["pred_label"]) for pred in preds],
        [float(pred["pred_score"]) for pred in preds],
    )


def _grouped_metrics(rows: list[dict[str, Any]], preds: list[dict[str, Any]], dataset: str, method: str, key: str) -> list[dict[str, Any]]:
    grouped: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for row, pred in zip(rows, preds, strict=True):
        value = str(row.get(key, "unknown"))
        if value not in grouped:
            grouped[value] = ([], [])
        grouped[value][0].append(row)
        grouped[value][1].append(pred)
    records = []
    for value, (sub_rows, sub_preds) in grouped.items():
        if len({row["gold_label"] for row in sub_rows}) < 2:
            continue
        records.append({"dataset": dataset, "method": method, "group_by": key, "group": value, "n": len(sub_rows), **_metrics(sub_rows, sub_preds)})
    return records


def _metadata_grouped_metrics(rows: list[dict[str, Any]], preds: list[dict[str, Any]], dataset: str, method: str, key: str) -> list[dict[str, Any]]:
    grouped: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for row, pred in zip(rows, preds, strict=True):
        value = str((row.get("metadata") or {}).get(key, "unknown"))
        if value not in grouped:
            grouped[value] = ([], [])
        grouped[value][0].append(row)
        grouped[value][1].append(pred)
    records = []
    for value, (sub_rows, sub_preds) in grouped.items():
        if len({row["gold_label"] for row in sub_rows}) < 2:
            continue
        records.append({"dataset": dataset, "method": method, "group_by": f"metadata.{key}", "group": value, "n": len(sub_rows), **_metrics(sub_rows, sub_preds)})
    return records


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in records for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _baseline_report(manifest: dict[str, Any], metrics: list[dict[str, Any]]) -> str:
    lines = [
        "# Full Detector Baseline Report",
        "",
        "This report evaluates existing offline detectors on complete prepared q+y evaluation sets. No API calls are used in this step.",
        "",
        "## Main Results",
        "",
        "| Dataset | Method | N | Accuracy | Unsafe Recall | Precision | FPR | Macro-F1 | AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if "group_by" in row:
            continue
        lines.append(
            f"| `{row['dataset']}` | `{row['method']}` | {row['n']} | {row.get('accuracy', 0):.4f} | "
            f"{row.get('recall', 0):.4f} | {row.get('precision', 0):.4f} | {row.get('fpr', 0):.4f} | "
            f"{row.get('macro_f1', 0):.4f} | {row.get('auprc', 0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Template",
            "",
            "- `keyword_refusal_rule` is a low-cost baseline. It is expected to have unstable precision/FPR on hard-safe and general-safety sets.",
            "- `frauddistill_teacher_offline` is the current non-API teacher approximation. It should outperform rules, but paper-level claims require API judge/guard votes and student distillation.",
            "- Large deviations should be diagnosed per dataset and per fraud category before final paper tables are frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dir", default="data/prepared/full/evaluation_qy")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    manifest = run_full_detector_baselines(args.evaluation_dir, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "datasets": list(manifest["datasets"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
