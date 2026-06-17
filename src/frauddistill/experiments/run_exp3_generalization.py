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

TRAIN_CATEGORIES = {"phishing_scams", "impersonation", "fake_job_postings"}
TEST_CATEGORIES = {"fraudulent_services", "online_relationships"}


def run_exp3(samples_file: str, output_dir: str) -> dict:
    rows = list(read_jsonl(samples_file))
    train_pool = [row for row in rows if _category(row) in TRAIN_CATEGORIES and row.get("metadata", {}).get("fraudr1_variant") == "base"]
    dev = [row for row in rows if _category(row) in TRAIN_CATEGORIES and row.get("metadata", {}).get("fraudr1_variant") == "levelup"]
    category_test = [row for row in rows if _category(row) in TEST_CATEGORIES]
    target_test = [row for row in rows if _category(row) in TRAIN_CATEGORIES and row.get("metadata", {}).get("fraudr1_variant") == "levelup"]

    out = Path(output_dir)
    for sub in ["models", "predictions", "tables"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train_base_seen_categories.jsonl", train_pool)
    write_jsonl(out / "dev_levelup_seen_categories.jsonl", dev)
    write_jsonl(out / "test_unseen_categories.jsonl", category_test)

    model = _train(train_pool)
    joblib.dump(model, out / "models" / "student_agentdistill_proxy.joblib")
    methods = {
        "rule": lambda items: [predict_rule(row) for row in items],
        "student_agentdistill_proxy": lambda items: _predict_model(model, items),
    }
    metric_rows = []
    for setting, split_rows in {"category_generalization": category_test, "target_style_generalization": target_test}.items():
        for method, predictor in methods.items():
            preds = [_attach_gold(row, pred) for row, pred in zip(split_rows, predictor(split_rows))]
            write_jsonl(out / "predictions" / f"{method}_{setting}_predictions.jsonl", preds)
            metric_rows.append({"setting": setting, "method": method, **_metrics(preds)})

    low_fpr_rows = _low_fpr_table(model, dev, category_test)
    _write_csv(out / "tables" / "exp3_metrics.csv", metric_rows)
    _write_csv(out / "tables" / "exp3_low_fpr.csv", low_fpr_rows)
    summary = {
        "dataset_stats": {"train": len(train_pool), "dev": len(dev), "category_test": len(category_test), "target_style_test": len(target_test)},
        "metrics": metric_rows,
        "low_fpr": low_fpr_rows,
    }
    (out / "tables" / "exp3_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _train(rows: list[dict]) -> Pipeline:
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ]
    )
    model.fit([_text(row) for row in rows], np.array([1 if row["gold_label"] == "unsafe" else 0 for row in rows]))
    return model


def _predict_model(model: Pipeline, rows: list[dict], threshold: float = 0.5) -> list[dict]:
    scores = model.predict_proba([_text(row) for row in rows])[:, 1]
    output = []
    for row, score in zip(rows, scores):
        label = "unsafe" if score >= threshold else "safe"
        output.append({"id": row["id"], "pred_label": label, "pred_score": float(score), "pred_type": "fraud_assistance" if label == "unsafe" else "safe"})
    return output


def _low_fpr_table(model: Pipeline, dev: list[dict], test: list[dict]) -> list[dict]:
    dev_scores = model.predict_proba([_text(row) for row in dev])[:, 1]
    test_scores = model.predict_proba([_text(row) for row in test])[:, 1]
    thresholds = [0.10, 0.20, 0.25]
    rows = []
    for max_fpr in thresholds:
        best_threshold = 1.0
        best_recall = 0.0
        for threshold in sorted(set(dev_scores), reverse=True):
            dev_preds = ["unsafe" if score >= threshold else "safe" for score in dev_scores]
            dev_metrics = binary_metrics([row["gold_label"] for row in dev], dev_preds, list(dev_scores))
            if dev_metrics["fpr"] <= max_fpr:
                test_preds = ["unsafe" if score >= threshold else "safe" for score in test_scores]
                test_metrics = binary_metrics([row["gold_label"] for row in test], test_preds, list(test_scores))
                if test_metrics["recall"] >= best_recall:
                    best_recall = test_metrics["recall"]
                    best_threshold = float(threshold)
        test_preds = ["unsafe" if score >= best_threshold else "safe" for score in test_scores]
        test_metrics = binary_metrics([row["gold_label"] for row in test], test_preds, list(test_scores))
        rows.append({"method": "student_agentdistill_proxy", "constraint": f"FPR<={max_fpr:.2f}", "threshold": best_threshold, **test_metrics})
    return rows


def _text(row: dict) -> str:
    return build_detector_input(row["user_query"], row["target_model_answer"], row.get("context"), "q_y")


def _category(row: dict) -> str:
    return str(row.get("metadata", {}).get("fraud_category", "unknown"))


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
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["setting", "method", "constraint", "threshold", "accuracy", "precision", "recall", "macro_f1", "fpr", "fnr", "auroc", "auprc"]
    fieldnames = [field for field in preferred if field in fieldnames] + [field for field in fieldnames if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file", default="data/unified/exp3_fraudr1_all_categories.jsonl")
    parser.add_argument("--output_dir", default="outputs/exp3_final")
    args = parser.parse_args()
    print(json.dumps(run_exp3(args.samples_file, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
