from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "unified" / "v2_hard_control_full.jsonl"
OUT_ROOT = ROOT / "outputs"
SEED = 20260722
SEEN_CATEGORIES = {"phishing_scams", "impersonation"}
UNSEEN_CATEGORY = "fake_job_postings"
UNSEEN_SOURCE = "or_bench_hard_safe"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["exp4", "exp5", "all"])
    parser.add_argument("--run-id", default="deepseek_student_v2_2400")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    rows = read_jsonl(DATA_FILE)
    if args.limit:
        rows = balanced_limit(rows, args.limit)
    if args.command in {"exp4", "all"}:
        run_exp4(rows, args.run_id, args.bootstrap)
    if args.command in {"exp5", "all"}:
        run_exp5(rows, args.run_id, args.bootstrap)


def run_exp4(rows: list[dict], run_id: str, bootstrap_n: int) -> None:
    out = init_run("exp4_unseen", run_id)
    train, seen_dev, unseen_fake_job, unseen_orbench = make_exp4_splits(rows)
    write_jsonl(out / "train_manifest.jsonl", train)
    write_jsonl(out / "seen_dev_manifest.jsonl", seen_dev)
    write_jsonl(out / "unseen_category_fake_job_manifest.jsonl", unseen_fake_job)
    write_jsonl(out / "unseen_source_orbench_manifest.jsonl", unseen_orbench)

    model = train_model(train)
    joblib.dump(model, out / "models" / "student_tfidf_logreg.joblib")
    settings = [
        ("Seen Dev", "seen categories", seen_dev),
        ("Unseen Category", "fake job", unseen_fake_job),
        ("Unseen Source", "OR-Bench hard safe", unseen_orbench),
        ("Unseen Mixed", "fake job + OR-Bench", unseen_fake_job + unseen_orbench),
    ]
    table = []
    all_predictions = []
    for setting, source, split_rows in settings:
        preds = predict(model, split_rows)
        metric = metrics(split_rows, preds)
        ci = bootstrap(split_rows, preds, bootstrap_n)
        table.append({"Setting": setting, "Test source/category": source, "N": len(split_rows), **display(metric)})
        all_predictions.extend(attach(split_rows, preds, setting))
        write_json(out / f"{slug(setting)}_bootstrap.json", ci)
    write_jsonl(out / "exp4_predictions_test.jsonl", all_predictions)
    write_csv(out / "exp4_main_table.csv", table)
    write_json(out / "exp4_metrics.json", table)
    write_config(out, "exp4_unseen", {"dataset": str(DATA_FILE.relative_to(ROOT)), "bootstrap": bootstrap_n})
    write_exp4_report(out, table)


def run_exp5(rows: list[dict], run_id: str, bootstrap_n: int) -> None:
    out = init_run("exp5_calibration", run_id)
    train, calib_dev, threshold_dev, test = make_exp5_splits(rows)
    for name, split_rows in [("train", train), ("calibration_dev", calib_dev), ("threshold_dev", threshold_dev), ("test", test)]:
        write_jsonl(out / f"{name}_manifest.jsonl", split_rows)
    base_model = train_model(train)
    joblib.dump(base_model, out / "models" / "student_raw.joblib")

    calib_y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in calib_dev])
    raw_calib = scores(base_model, calib_dev)
    raw_threshold = scores(base_model, threshold_dev)
    raw_test = scores(base_model, test)

    calibrators = {
        "Default": Calibration("raw", None, 0.5, raw_test),
        "Dev threshold": Calibration("raw", None, select_threshold(threshold_dev, raw_threshold, None), raw_test),
        "Platt": fit_platt(raw_calib, calib_y, threshold_dev, raw_threshold, test, raw_test),
        "Isotonic": fit_isotonic(raw_calib, calib_y, threshold_dev, raw_threshold, test, raw_test),
        "Temperature": fit_temperature(raw_calib, calib_y, threshold_dev, raw_threshold, test, raw_test),
    }

    table = []
    predictions = []
    for method, cal in calibrators.items():
        test_scores = cal.test_scores
        preds = labels_from_scores(test_scores, cal.threshold)
        pred_rows = [{"pred_label": label, "pred_score": float(score), "pred_type": "fraud_assistance" if label == "unsafe" else "safe"} for label, score in zip(preds, test_scores)]
        m = metrics(test, pred_rows)
        m["ece"] = ece(test, test_scores)
        m["brier"] = float(brier_score_loss([1 if r["gold_label"] == "unsafe" else 0 for r in test], test_scores))
        table.append({"Method": method, "τ": round(cal.threshold, 6), **display_calibration(m)})
        predictions.extend(attach(test, pred_rows, method))

    fpr_rows = []
    for cap in [0.01, 0.05, 0.10]:
        threshold = select_threshold(threshold_dev, raw_threshold, cap)
        pred_labels = labels_from_scores(raw_test, threshold)
        pred_rows = [{"pred_label": label, "pred_score": float(score), "pred_type": "fraud_assistance" if label == "unsafe" else "safe"} for label, score in zip(pred_labels, raw_test)]
        fpr_rows.append({"Constraint": f"FPR<={cap:.2f}", "τ": round(threshold, 6), **display(metrics(test, pred_rows))})

    write_jsonl(out / "exp5_predictions_test.jsonl", predictions)
    write_csv(out / "exp5_calibration_table.csv", table)
    write_csv(out / "exp5_fpr_constraint_table.csv", fpr_rows)
    write_json(out / "exp5_metrics.json", {"calibration": table, "fpr_constraints": fpr_rows})
    write_config(out, "exp5_calibration", {"dataset": str(DATA_FILE.relative_to(ROOT)), "bootstrap": bootstrap_n, "splits": {"train": len(train), "calibration_dev": len(calib_dev), "threshold_dev": len(threshold_dev), "test": len(test)}})
    write_exp5_report(out, table, fpr_rows)


class Calibration:
    def __init__(self, kind: str, model, threshold: float, test_scores: np.ndarray):
        self.kind = kind
        self.model = model
        self.threshold = float(threshold)
        self.test_scores = test_scores


def fit_platt(raw_calib: np.ndarray, calib_y: np.ndarray, threshold_dev: list[dict], raw_threshold: np.ndarray, test: list[dict], raw_test: np.ndarray) -> Calibration:
    model = LogisticRegression(solver="lbfgs")
    model.fit(logit(raw_calib).reshape(-1, 1), calib_y)
    thresh_scores = model.predict_proba(logit(raw_threshold).reshape(-1, 1))[:, 1]
    threshold = select_threshold(threshold_dev, thresh_scores, None)
    test_scores = model.predict_proba(logit(raw_test).reshape(-1, 1))[:, 1]
    return Calibration("platt", model, threshold, test_scores)


def fit_isotonic(raw_calib: np.ndarray, calib_y: np.ndarray, threshold_dev: list[dict], raw_threshold: np.ndarray, test: list[dict], raw_test: np.ndarray) -> Calibration:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(raw_calib, calib_y)
    thresh_scores = model.predict(raw_threshold)
    threshold = select_threshold(threshold_dev, thresh_scores, None)
    test_scores = model.predict(raw_test)
    return Calibration("isotonic", model, threshold, test_scores)


def fit_temperature(raw_calib: np.ndarray, calib_y: np.ndarray, threshold_dev: list[dict], raw_threshold: np.ndarray, test: list[dict], raw_test: np.ndarray) -> Calibration:
    best_t, best_loss = 1.0, float("inf")
    logits = logit(raw_calib)
    for t in np.linspace(0.5, 5.0, 91):
        probs = sigmoid(logits / t)
        loss = log_loss(calib_y, probs)
        if loss < best_loss:
            best_t, best_loss = float(t), float(loss)
    thresh_scores = sigmoid(logit(raw_threshold) / best_t)
    threshold = select_threshold(threshold_dev, thresh_scores, None)
    test_scores = sigmoid(logit(raw_test) / best_t)
    return Calibration("temperature", {"temperature": best_t}, threshold, test_scores)


def make_exp4_splits(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    seen = [r for r in rows if category(r) in SEEN_CATEGORIES]
    unseen_fake_job = [r for r in rows if category(r) == UNSEEN_CATEGORY]
    unseen_orbench = [r for r in rows if category(r) == UNSEEN_SOURCE]
    random.Random(SEED).shuffle(seen)
    train, seen_dev = stratified_split(seen, 0.80)
    return train, seen_dev, unseen_fake_job, unseen_orbench


def make_exp5_splits(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["gold_label"]].append(row)
    for items in by_label.values():
        rng.shuffle(items)
    train, calib, threshold, test = [], [], [], []
    for items in by_label.values():
        n = len(items)
        train.extend(items[: int(n * 0.50)])
        calib.extend(items[int(n * 0.50) : int(n * 0.6667)])
        threshold.extend(items[int(n * 0.6667) : int(n * 0.8334)])
        test.extend(items[int(n * 0.8334) :])
    for split in [train, calib, threshold, test]:
        rng.shuffle(split)
    return train, calib, threshold, test


def stratified_split(rows: list[dict], train_frac: float) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["gold_label"]].append(row)
    train, rest = [], []
    for items in by_label.values():
        rng.shuffle(items)
        cut = int(len(items) * train_frac)
        train.extend(items[:cut])
        rest.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(rest)
    return train, rest


def train_model(rows: list[dict]) -> Pipeline:
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", C=1.0)),
        ]
    )
    model.fit([text(r) for r in rows], [1 if r["gold_label"] == "unsafe" else 0 for r in rows])
    return model


def predict(model: Pipeline, rows: list[dict], threshold: float = 0.5) -> list[dict]:
    s = scores(model, rows)
    labels = labels_from_scores(s, threshold)
    return [{"pred_label": label, "pred_score": float(score), "pred_type": "fraud_assistance" if label == "unsafe" else "safe"} for label, score in zip(labels, s)]


def scores(model: Pipeline, rows: list[dict]) -> np.ndarray:
    return model.predict_proba([text(r) for r in rows])[:, 1]


def select_threshold(rows: list[dict], s: np.ndarray, fpr_cap: float | None) -> float:
    best_threshold, best_key = 0.5, (-1.0, -1.0)
    for threshold in sorted(set(float(x) for x in s), reverse=True):
        pred_rows = [{"pred_label": label, "pred_score": float(score), "pred_type": "fraud_assistance" if label == "unsafe" else "safe"} for label, score in zip(labels_from_scores(s, threshold), s)]
        m = metrics(rows, pred_rows)
        if fpr_cap is not None and m["fpr_safe"] > fpr_cap:
            continue
        key = (m["recall_unsafe"], m["macro_f1"])
        if key > best_key:
            best_threshold, best_key = threshold, key
    return float(best_threshold)


def labels_from_scores(s: np.ndarray, threshold: float) -> list[str]:
    return ["unsafe" if float(x) >= threshold else "safe" for x in s]


def metrics(rows: list[dict], preds: list[dict]) -> dict:
    y_true = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows])
    y_pred = np.array([1 if p["pred_label"] == "unsafe" else 0 for p in preds])
    y_score = np.array([float(p["pred_score"]) for p in preds])
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_unsafe": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_unsafe": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_unsafe": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr_safe": float(((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
    }
    if len(set(y_true.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc_unsafe"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = None
        out["auprc_unsafe"] = None
    return out


def ece(rows: list[dict], s: np.ndarray, bins: int = 10) -> float:
    y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows])
    total = len(y)
    value = 0.0
    for lo in np.linspace(0.0, 0.9, bins):
        hi = lo + 0.1
        mask = (s >= lo) & (s < hi if hi < 1.0 else s <= hi)
        if mask.sum() == 0:
            continue
        value += float(mask.sum() / total * abs(y[mask].mean() - s[mask].mean()))
    return value


def bootstrap(rows: list[dict], preds: list[dict], n: int) -> dict:
    rng = random.Random(SEED)
    values = defaultdict(list)
    for _ in range(n):
        idxs = [rng.randrange(len(rows)) for _ in rows]
        sample_rows = [rows[i] for i in idxs]
        sample_preds = [preds[i] for i in idxs]
        m = metrics(sample_rows, sample_preds)
        for key in ["accuracy", "precision_unsafe", "recall_unsafe", "f1_unsafe", "macro_f1", "fpr_safe", "auprc_unsafe"]:
            if m.get(key) is not None:
                values[key].append(m[key])
    return {k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))} for k, v in values.items()}


def display(m: dict) -> dict:
    return {
        "Acc": round(m["accuracy"], 4),
        "Prec_unsafe": round(m["precision_unsafe"], 4),
        "Recall_unsafe": round(m["recall_unsafe"], 4),
        "F1_unsafe": round(m["f1_unsafe"], 4),
        "Macro-F1": round(m["macro_f1"], 4),
        "FPR": round(m["fpr_safe"], 4),
        "AUPRC": None if m.get("auprc_unsafe") is None else round(m["auprc_unsafe"], 4),
    }


def display_calibration(m: dict) -> dict:
    row = display(m)
    row["ECE"] = round(m["ece"], 4)
    row["Brier"] = round(m["brier"], 4)
    return row


def write_exp4_report(out: Path, table: list[dict]) -> None:
    lines = [
        "# 实验4：unseen 泛化实验报告",
        "",
        "## 实验设置",
        "本实验使用 `v2_hard_control_full.jsonl`。训练集只使用 seen categories：phishing_scams 与 impersonation；unseen category 测试 fake_job_postings；unseen source/hard-safe 测试 OR-Bench hard safe。该设置复用公开数据和自动生成回答，不新增人工标注。",
        "",
        "## 主表",
        csv_markdown(out / "exp4_main_table.csv"),
        "",
        "## 数据分析",
        "Seen Dev 衡量模型在已见欺诈类别上的基本拟合能力；Unseen Category 直接检验 fake job 类别迁移；Unseen Source 主要是 hard-safe 误报压力，不含 unsafe 样本，因此 Precision/Recall 类指标不可按标准 response 检测主表单独解释，报告中保留它是为了观察 FPR。",
        "如果 Unseen Mixed 的 Recall 高但 FPR 也偏高，结论应写为模型有跨类别召回能力，但仍依赖实验5的阈值控制才能部署。",
        "",
        "## 规模说明",
        "当前 hard-control 全量集为 2400 条，低于增强版文档建议的 unseen test 1500-3000 且每个 held-out 类至少 300 unsafe/300 safe 的完整论文规模；本结果可作为能用的 hard-control 正式版，但后续应继续扩充 WildGuard/Aegis/Do-Not-Answer gold 数据。",
    ]
    (out / "exp4_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_exp5_report(out: Path, table: list[dict], fpr_rows: list[dict]) -> None:
    lines = [
        "# 实验5：阈值与概率校准实验报告",
        "",
        "## 实验设置",
        "本实验将 hard-control 数据按标签分层拆分为 train、calibration dev、threshold dev、test。calibration dev 只拟合 Platt/Isotonic/Temperature 校准器；threshold dev 只选择阈值；test 只做最终评估，避免根据 test 反复调参。",
        "",
        "## 校准主表",
        csv_markdown(out / "exp5_calibration_table.csv"),
        "",
        "## FPR 约束表",
        csv_markdown(out / "exp5_fpr_constraint_table.csv"),
        "",
        "## 数据分析",
        "Default 反映原始 0.5 阈值；Dev threshold 展示只调阈值的收益；Platt、Isotonic 和 Temperature 分别测试概率校准后再选阈值的效果。需要分开解读阈值选择和概率校准：低 FPR 改善不等于概率已经校准，ECE/Brier 改善才支持校准结论。",
        "若多个 FPR 约束得到相同阈值，这是因为 threshold dev 上可用分数是离散的，并且同一个阈值已经同时满足多个约束；报告保留这种情况，不根据 test 重调。",
    ]
    (out / "exp5_report.md").write_text("\n".join(lines), encoding="utf-8")


def init_run(exp: str, run_id: str) -> Path:
    out = OUT_ROOT / exp / run_id
    for sub in ["models", "tables", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    (out / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (out / "environment.txt").write_text(f"python={sys.version}\ncreated={datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    return out


def attach(rows: list[dict], preds: list[dict], setting: str) -> list[dict]:
    return [
        {
            "id": row["id"],
            "setting": setting,
            "gold_label": row["gold_label"],
            "gold_risk_type": row.get("gold_risk_type"),
            "pred_label": pred["pred_label"],
            "pred_score": pred["pred_score"],
            "pred_type": pred["pred_type"],
            "source": row.get("source"),
            "language": row.get("language"),
            "metadata": row.get("metadata", {}),
        }
        for row, pred in zip(rows, preds)
    ]


def balanced_limit(rows: list[dict], limit: int) -> list[dict]:
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["gold_label"]].append(row)
    for items in by_label.values():
        rng.shuffle(items)
    out = by_label["unsafe"][: limit // 2] + by_label["safe"][: limit - limit // 2]
    rng.shuffle(out)
    return out


def category(row: dict) -> str:
    return str(row.get("metadata", {}).get("fraud_category", "unknown"))


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def text(row: dict) -> str:
    return f"[Q]\n{row.get('user_query','')}\n[A]\n{row.get('target_model_answer','')}"


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_config(out: Path, exp: str, extra: dict) -> None:
    data = {"experiment": exp, "run_date": datetime.now().date().isoformat(), **extra}
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    (out / f"{exp.split('_')[0]}_config.yaml").write_text(text, encoding="utf-8")
    (out / "config_resolved.yaml").write_text(text, encoding="utf-8")


def csv_markdown(path: Path) -> str:
    rows = list(csv.reader(path.open("r", encoding="utf-8")))
    return "\n".join([
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * len(rows[0])) + " |",
        *["| " + " | ".join(row) + " |" for row in rows[1:]],
    ])


if __name__ == "__main__":
    main()
