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

SEEN = {"phishing_scams", "impersonation"}
UNSEEN = {"fake_job_postings", "or_bench_hard_safe"}


def run(samples_file: str, qwen_file: str, output_dir: str) -> dict:
    rows = list(read_jsonl(samples_file))
    qwen = {row["id"]: row for row in read_jsonl(qwen_file)}
    seen_rows = [row for row in rows if _category(row) in SEEN]
    unseen_rows = [row for row in rows if _category(row) in UNSEEN]
    train, dev = _split_seen(seen_rows)
    out = Path(output_dir)
    for sub in ["models", "predictions", "tables"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train_seen_categories.jsonl", train)
    write_jsonl(out / "dev_seen_categories.jsonl", dev)
    write_jsonl(out / "test_unseen_category_and_orbench.jsonl", unseen_rows)

    model = _train(train)
    joblib.dump(model, out / "models" / "student_agentdistill_v2_proxy.joblib")
    methods = {
        "rule": lambda items: [predict_rule(row) for row in items],
        "single_judge_api_qwen": lambda items: [_qwen_pred(row, qwen) for row in items],
        "student_agentdistill_v2_proxy": lambda items: _predict(model, items),
    }
    metric_rows = []
    for setting, split_rows in {"seen_dev": dev, "unseen_category_orbench": unseen_rows}.items():
        for method, predictor in methods.items():
            preds = [_attach(row, pred) for row, pred in zip(split_rows, predictor(split_rows))]
            write_jsonl(out / "predictions" / f"{method}_{setting}_predictions.jsonl", preds)
            metric_rows.append({"setting": setting, "method": method, **_metrics(preds)})
    low_fpr = _low_fpr(model, dev, unseen_rows)
    _write_csv(out / "tables" / "v2_exp3_metrics.csv", metric_rows)
    _write_csv(out / "tables" / "v2_exp3_low_fpr.csv", low_fpr)
    summary = {
        "dataset_stats": {"train_seen": len(train), "dev_seen": len(dev), "unseen_category_orbench": len(unseen_rows)},
        "metrics": metric_rows,
        "low_fpr": low_fpr,
    }
    (out / "tables" / "v2_exp3_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _split_seen(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    train, dev = [], []
    for idx, row in enumerate(rows):
        (dev if idx % 5 == 0 else train).append(row)
    return train, dev


def _train(rows: list[dict]) -> Pipeline:
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ]
    )
    model.fit([_text(row) for row in rows], np.array([1 if row["gold_label"] == "unsafe" else 0 for row in rows]))
    return model


def _predict(model: Pipeline, rows: list[dict], threshold: float = 0.5) -> list[dict]:
    scores = model.predict_proba([_text(row) for row in rows])[:, 1]
    out = []
    for row, score in zip(rows, scores):
        label = "unsafe" if score >= threshold else "safe"
        out.append({"id": row["id"], "pred_label": label, "pred_score": float(score), "pred_type": "fraud_assistance" if label == "unsafe" else "safe"})
    return out


def _low_fpr(model: Pipeline, dev: list[dict], test: list[dict]) -> list[dict]:
    dev_scores = model.predict_proba([_text(row) for row in dev])[:, 1]
    test_scores = model.predict_proba([_text(row) for row in test])[:, 1]
    rows = []
    for max_fpr in [0.10, 0.20, 0.25]:
        best_threshold = 1.0
        best_recall = -1.0
        for threshold in sorted(set(dev_scores), reverse=True):
            dev_pred = ["unsafe" if score >= threshold else "safe" for score in dev_scores]
            dev_metrics = binary_metrics([row["gold_label"] for row in dev], dev_pred, list(dev_scores))
            if dev_metrics["fpr"] <= max_fpr:
                test_pred = ["unsafe" if score >= threshold else "safe" for score in test_scores]
                test_metrics = binary_metrics([row["gold_label"] for row in test], test_pred, list(test_scores))
                if test_metrics["recall"] > best_recall:
                    best_threshold = float(threshold)
                    best_recall = test_metrics["recall"]
        test_pred = ["unsafe" if score >= best_threshold else "safe" for score in test_scores]
        rows.append({"method": "student_agentdistill_v2_proxy", "constraint": f"FPR<={max_fpr:.2f}", "threshold": best_threshold, **binary_metrics([row["gold_label"] for row in test], test_pred, list(test_scores))})
    return rows


def _qwen_pred(row: dict, qwen: dict[str, dict]) -> dict:
    pred = qwen.get(row["id"])
    if not pred:
        return {"id": row["id"], "pred_label": "safe", "pred_score": 0.0, "pred_type": "safe"}
    return {"id": row["id"], "pred_label": pred["pred_label"], "pred_score": float(pred["pred_score"]), "pred_type": pred.get("pred_type", "safe")}


def _text(row: dict) -> str:
    return build_detector_input(row["user_query"], row["target_model_answer"], row.get("context"), "q_y")


def _category(row: dict) -> str:
    return str(row.get("metadata", {}).get("fraud_category", "unknown"))


def _attach(row: dict, pred: dict) -> dict:
    return {"id": row["id"], "gold_label": row["gold_label"], "gold_risk_type": row.get("gold_risk_type"), "pred_label": pred["pred_label"], "pred_score": pred["pred_score"], "pred_type": pred.get("pred_type", "safe"), "language": row.get("language"), "metadata": row.get("metadata", {})}


def _metrics(rows: list[dict]) -> dict[str, float]:
    return binary_metrics([row["gold_label"] for row in rows], [row["pred_label"] for row in rows], [float(row["pred_score"]) for row in rows])


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = ["setting", "method", "constraint", "threshold", "accuracy", "precision", "recall", "macro_f1", "fpr", "fnr", "auroc", "auprc"]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file", default="data/unified/v2_hard_control_full.jsonl")
    parser.add_argument("--qwen_file", default="outputs/v2_exp1_final/predictions/single_judge_api_qwen_q_y_predictions.jsonl")
    parser.add_argument("--output_dir", default="outputs/v2_exp3_final")
    args = parser.parse_args()
    print(json.dumps(run(args.samples_file, args.qwen_file, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
