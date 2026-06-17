from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from frauddistill.eval.metrics import binary_metrics
from frauddistill.eval.rule_baseline import predict_rule
from frauddistill.utils.io import read_jsonl, write_jsonl
from frauddistill.utils.text import build_detector_input


def run_exp2(samples_file: str, qwen_teacher_file: str, offline_teacher_file: str, output_dir: str) -> dict:
    rows = list(read_jsonl(samples_file))
    qwen_teacher = {row["id"]: row for row in read_jsonl(qwen_teacher_file)}
    offline_teacher = {row["id"]: row for row in read_jsonl(offline_teacher_file)}
    splits = _split_rows(rows)
    out = Path(output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        write_jsonl(out / f"{split}.jsonl", split_rows)

    gold_model = _train_model(splits["train"], labels=[row["gold_label"] for row in splits["train"]])
    distill_labels = [_teacher_or_gold(row, qwen_teacher) for row in splits["train"]]
    distill_weights = [_distill_weight(row, qwen_teacher) for row in splits["train"]]
    distill_model = _train_model(splits["train"], labels=distill_labels, sample_weight=distill_weights)
    joblib.dump(gold_model, out / "models" / "student_gold.joblib")
    joblib.dump(distill_model, out / "models" / "student_agentdistill.joblib")

    methods = {
        "rule": lambda items: [predict_rule(row) for row in items],
        "multi_agent_teacher_offline": lambda items: [_teacher_prediction(row, offline_teacher, "multi_agent_teacher_offline") for row in items],
        "single_judge_api_qwen": lambda items: [_teacher_prediction(row, qwen_teacher, "single_judge_api_qwen") for row in items],
        "student_gold": lambda items: _model_predictions(gold_model, items),
        "student_agentdistill": lambda items: _model_predictions(distill_model, items),
    }
    metric_rows = []
    for split in ["dev", "test"]:
        for method, predictor in methods.items():
            preds = [_attach_gold(source, pred) for source, pred in zip(splits[split], predictor(splits[split]))]
            write_jsonl(out / "predictions" / f"{method}_{split}_predictions.jsonl", preds)
            metric_rows.append({"split": split, "method": method, **_metrics(preds)})

    _write_csv(out / "tables" / "exp2_metrics.csv", metric_rows)
    summary = {"dataset_stats": {k: len(v) for k, v in splits.items()}, "metrics": metric_rows}
    (out / "tables" / "exp2_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    buckets = {"train": [], "dev": [], "test": []}
    for idx, row in enumerate(rows):
        key = idx % 10
        if key < 6:
            buckets["train"].append(row)
        elif key < 8:
            buckets["dev"].append(row)
        else:
            buckets["test"].append(row)
    return buckets


def _train_model(rows: list[dict], labels: list[str], sample_weight: list[float] | None = None) -> Pipeline:
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ]
    )
    texts = [_text(row) for row in rows]
    y = np.array([1 if label == "unsafe" else 0 for label in labels])
    kwargs = {}
    if sample_weight is not None:
        kwargs["clf__sample_weight"] = np.array(sample_weight)
    model.fit(texts, y, **kwargs)
    return model


def _model_predictions(model: Pipeline, rows: list[dict]) -> list[dict]:
    texts = [_text(row) for row in rows]
    scores = model.predict_proba(texts)[:, 1]
    preds = []
    for row, score in zip(rows, scores):
        label = "unsafe" if score >= 0.5 else "safe"
        preds.append({"id": row["id"], "pred_label": label, "pred_score": float(score), "pred_type": "fraud_assistance" if label == "unsafe" else "safe"})
    return preds


def _text(row: dict) -> str:
    return build_detector_input(row["user_query"], row["target_model_answer"], row.get("context"), "q_y")


def _teacher_or_gold(row: dict, teacher: dict[str, dict]) -> str:
    pred = teacher.get(row["id"])
    return pred["pred_label"] if pred and pred.get("pred_score", 0) >= 0.7 else row["gold_label"]


def _distill_weight(row: dict, teacher: dict[str, dict]) -> float:
    pred = teacher.get(row["id"])
    if not pred:
        return 1.0
    score = float(pred.get("pred_score", 0.5))
    agree = pred.get("pred_label") == row["gold_label"]
    return 1.5 + abs(score - 0.5) if agree else 0.8


def _teacher_prediction(row: dict, teacher: dict[str, dict], name: str) -> dict:
    pred = teacher.get(row["id"])
    if pred:
        return {"id": row["id"], "pred_label": pred["pred_label"], "pred_score": float(pred["pred_score"]), "pred_type": pred.get("pred_type", "safe")}
    return {"id": row["id"], "pred_label": "safe", "pred_score": 0.0, "pred_type": "safe", "judge_error": f"missing {name}"}


def _attach_gold(source: dict, pred: dict) -> dict:
    return {
        "id": source["id"],
        "gold_label": source["gold_label"],
        "gold_risk_type": source.get("gold_risk_type"),
        "pred_label": pred["pred_label"],
        "pred_score": pred["pred_score"],
        "pred_type": pred.get("pred_type", "safe"),
        "source": source.get("source"),
        "language": source.get("language"),
        "metadata": source.get("metadata", {}),
    }


def _metrics(rows: list[dict]) -> dict[str, float]:
    return binary_metrics([row["gold_label"] for row in rows], [row["pred_label"] for row in rows], [float(row["pred_score"]) for row in rows])


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["split", "method", "accuracy", "precision", "recall", "macro_f1", "fpr", "fnr", "auroc", "auprc"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file", default="data/unified/exp1_fraudr1_full.jsonl")
    parser.add_argument("--qwen_teacher_file", default="outputs/exp1_final/predictions/single_judge_api_qwen_q_y_predictions.jsonl")
    parser.add_argument("--offline_teacher_file", default="outputs/exp1_final/predictions/multi_agent_teacher_offline_q_y_predictions.jsonl")
    parser.add_argument("--output_dir", default="outputs/exp2_final")
    args = parser.parse_args()
    print(json.dumps(run_exp2(args.samples_file, args.qwen_teacher_file, args.offline_teacher_file, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
