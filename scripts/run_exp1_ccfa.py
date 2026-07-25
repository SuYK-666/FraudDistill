from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frauddistill.eval.metrics import binary_metrics
from frauddistill.exp1_ccfa.exact_mcnemar import exact_mcnemar
from frauddistill.exp1_ccfa.pair_cross_encoder import (
    INPUT_MODES,
    XLMRPairCrossEncoder,
    labels_from_scores,
    select_threshold,
)
from frauddistill.exp1_ccfa.paired_cluster_bootstrap import paired_cluster_bootstrap_delta
from frauddistill.exp1_ccfa.semantic_components import (
    attach_semantic_components,
    explicit_label_token_audit,
    leakage_audit,
    split_by_component,
    write_component_manifest,
    write_schema,
)
from frauddistill.student.pair_tfidf import PairTfidfDetector
from frauddistill.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 CCF-A q-y boundary runner")
    parser.add_argument("command", choices=["preflight", "e1_1_dev", "smoke"])
    parser.add_argument("--config", default="configs/experiments/exp1_ccfa_v4.yaml")
    parser.add_argument("--input", default="data/processed/qy_v3/judged_pairs_v3.jsonl")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--backend", choices=["xlmr", "tfidf_dev"], default="xlmr")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("exp1_ccfa_v4_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or ROOT / "outputs" / "exp1_ccfa_v4" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.command == "preflight":
        result = preflight(args.input, config, output_dir)
    else:
        rows = load_rows(args.input, args.limit)
        if args.command == "smoke":
            rows = rows[: min(len(rows), args.limit or 60)]
            args.bootstrap_iterations = min(args.bootstrap_iterations, 200)
        result = run_dev(rows, config, output_dir, args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def preflight(input_path: str, config: dict, output_dir: Path) -> dict:
    rows = load_rows(input_path, 0)
    label_sources = sorted({str(row.get("label_provenance", "")) for row in rows})
    attached = attach_semantic_components(rows)
    token_audit = explicit_label_token_audit(attached)
    split_audit = leakage_audit({"all_local_dev": attached})
    result = {
        "experiment": config["experiment"]["id"],
        "input": input_path,
        "rows": len(rows),
        "components": len({row["semantic_component_id"] for row in attached}),
        "label_sources": label_sources,
        "formal_gate_status": "NO_GO",
        "formal_gate_reason": "local E1-1 data uses proxy/silver labels; real Guard/public-gold panels are not frozen yet",
        "xlmr_backbone": config["model"]["backbone"],
        "token_audit": token_audit,
        "split_audit": split_audit,
    }
    write_common_artifacts(output_dir, attached, config, result)
    return result


def run_dev(rows: list[dict], config: dict, output_dir: Path, args: argparse.Namespace) -> dict:
    rows = attach_semantic_components(rows)
    for subdir in ("data", "tables", "statistics", "predictions", "reports"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    train, model_dev, threshold_dev, split_audit = split_by_component(rows, args.seed)
    if not split_audit["passed"]:
        raise RuntimeError(f"semantic component leakage detected: {split_audit}")
    if len({row["exp1_label"] for row in train}) < 2:
        raise RuntimeError("training split has fewer than two labels")

    predictions_by_mode: dict[str, list[dict]] = {}
    metrics_rows: list[dict] = []
    for mode in tqdm(INPUT_MODES, desc="E1 modes"):
        predictions = train_and_predict_mode(mode, train, model_dev, threshold_dev, config, args)
        predictions_by_mode[mode] = predictions
        write_jsonl(output_dir / "predictions" / f"{mode}_predictions.jsonl", predictions)
        labels = [row["gold_label"] for row in predictions]
        preds = [row["pred_label"] for row in predictions]
        scores = [float(row["pred_score"]) for row in predictions]
        metrics = binary_metrics(labels, preds, scores)
        metrics_rows.append({"panel": "E1-1-local-dev", "input_mode": mode, **metrics})

    stats = compare_modes(predictions_by_mode, args.bootstrap_iterations, args.seed)
    gate = development_gate(metrics_rows, stats, args.backend)
    write_metrics(output_dir / "tables" / "e1_1_dev_metrics.csv", metrics_rows)
    (output_dir / "tables" / "e1_1_dev_metrics.json").write_text(json.dumps(metrics_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "statistics" / "mode_comparisons.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_common_artifacts(output_dir, rows, config, {"split_audit": split_audit, "development_gate": gate})
    write_report(output_dir, metrics_rows, stats, gate, rows)
    return {
        "experiment": config["experiment"]["id"],
        "backend": args.backend,
        "rows": len(rows),
        "train_rows": len(train),
        "model_dev_rows": len(model_dev),
        "threshold_dev_rows": len(threshold_dev),
        "output_dir": str(output_dir),
        "development_gate": gate,
    }


def train_and_predict_mode(mode: str, train: list[dict], model_dev: list[dict], threshold_dev: list[dict], config: dict, args: argparse.Namespace) -> list[dict]:
    train_labels = [row["exp1_label"] for row in train]
    if args.backend == "tfidf_dev":
        model = PairTfidfDetector()
        model.fit(train, train_labels)
        dev_scores = model.predict_proba(model_dev, mode).tolist()
        threshold = select_threshold([row["exp1_label"] for row in model_dev], dev_scores).threshold if model_dev else 0.5
        scores = model.predict_proba(threshold_dev, mode).tolist()
    else:
        model = XLMRPairCrossEncoder(
            model_name=config["model"]["backbone"],
            max_length=int(config["model"]["max_length"]),
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
        )
        model.fit(train, train_labels, mode)
        dev_scores = model.predict_scores(model_dev, mode)
        threshold = select_threshold([row["exp1_label"] for row in model_dev], dev_scores).threshold if model_dev else 0.5
        scores = model.predict_scores(threshold_dev, mode)
    labels = labels_from_scores(scores, threshold)
    return [
        {
            "id": row["id"],
            "semantic_component_id": row["semantic_component_id"],
            "gold_label": row["exp1_label"],
            "pred_label": label,
            "pred_score": score,
            "threshold": threshold,
            "input_mode": mode,
            "label_provenance": row.get("label_provenance"),
            "source": row.get("source"),
            "target_model": row.get("target_model"),
        }
        for row, label, score in zip(threshold_dev, labels, scores)
    ]


def compare_modes(predictions_by_mode: dict[str, list[dict]], iterations: int, seed: int) -> dict:
    y_only = predictions_by_mode["y_only"]
    q_y = predictions_by_mode["q_y"]
    gold = [row["gold_label"] for row in q_y]
    clusters = [row["semantic_component_id"] for row in q_y]
    y_pred = [row["pred_label"] for row in y_only]
    qy_pred = [row["pred_label"] for row in q_y]
    metric_fn = lambda y_true, pred: float(f1_score(y_true, pred, average="macro", zero_division=0))
    return {
        "q_y_vs_y_only_macro_f1_delta": paired_cluster_bootstrap_delta(
            gold, y_pred, qy_pred, clusters, metric_fn, iterations=iterations, seed=seed
        ),
        "q_y_vs_y_only_exact_mcnemar": exact_mcnemar(gold, y_pred, qy_pred),
    }


def development_gate(metrics_rows: list[dict], stats: dict, backend: str) -> dict:
    by_mode = {row["input_mode"]: row for row in metrics_rows}
    qy = by_mode["q_y"]["macro_f1"]
    best_single = max(by_mode["q_only"]["macro_f1"], by_mode["y_only"]["macro_f1"])
    delta = qy - best_single
    return {
        "backend": backend,
        "formal_pass": False,
        "formal_pass_reason": "E1-1 is a local development run and cannot satisfy P1/P2/P3 public/Guard/full-scale gates",
        "direction_positive": delta > 0,
        "q_y_macro_f1": qy,
        "best_single_macro_f1": best_single,
        "delta_vs_best_single": delta,
        "q_y_vs_y_only_ci_lower": stats["q_y_vs_y_only_macro_f1_delta"]["ci_lower"],
    }


def write_common_artifacts(output_dir: Path, rows: list[dict], config: dict, status: dict) -> None:
    for subdir in ("data", "tables", "statistics", "predictions", "reports"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "data" / "all_rows_with_semantic_components.jsonl", rows)
    write_component_manifest(output_dir / "data" / "semantic_components.jsonl", rows)
    write_schema(output_dir / "data" / "all_rows_schema.json", rows)
    (output_dir / "config_lock.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output_dir / "exp1_gate_checklist.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["panel", "input_mode", "accuracy", "precision", "recall", "macro_f1", "fpr", "fnr", "auroc", "auprc"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(output_dir: Path, metrics_rows: list[dict], stats: dict, gate: dict, rows: list[dict]) -> None:
    lines = [
        "# 实验1 E1-1 本地开发运行报告",
        "",
        "## 结论",
        "",
        "本轮只使用现有 1,594 条本地开发数据，标签来源不是正式真实 Guard/public gold，因此不能作为论文正式验收结果。",
        f"开发方向 Gate：{'正向' if gate['direction_positive'] else '未达正向'}；q+y Macro-F1={gate['q_y_macro_f1']:.4f}，相对最佳单侧差值={gate['delta_vs_best_single']:.4f}。",
        "",
        "## 数据规模",
        "",
        f"- 行数：{len(rows)}",
        f"- semantic components：{len({row['semantic_component_id'] for row in rows})}",
        f"- 标签来源：{', '.join(sorted({str(row.get('label_provenance')) for row in rows}))}",
        "",
        "## 指标表",
        "",
        "| panel | input_mode | Macro-F1 | Accuracy | Precision | Recall | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['panel']} | {row['input_mode']} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['fpr']:.4f} | {row['fnr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 配对统计",
            "",
            f"- q+y vs y-only cluster bootstrap delta 均值：{stats['q_y_vs_y_only_macro_f1_delta']['delta_mean']:.4f}",
            f"- 95% CI：[{stats['q_y_vs_y_only_macro_f1_delta']['ci_lower']:.4f}, {stats['q_y_vs_y_only_macro_f1_delta']['ci_upper']:.4f}]",
            f"- exact McNemar p-value：{stats['q_y_vs_y_only_exact_mcnemar']['p_value']:.6f}",
            "",
            "## 分析",
            "",
            "这份报告用于判断重构后的 E1 代码能否跑通 q-only、y-only、q+y 三个完全独立输入模式，并检查 semantic-component split 与配对统计是否生效。若 q+y 仍未稳定高于单侧输入，下一步应优先补 P2 context-critical 碰撞组和真实 Guard 高置信标签，而不是直接扩大当前代理标签数据。",
        ]
    )
    (output_dir / "reports" / "EXP1_CCF_A_E1_1_DEV_REPORT_中文.md").write_text("\n".join(lines), encoding="utf-8")


def load_rows(path: str, limit: int) -> list[dict]:
    rows = list(read_jsonl(path))
    if limit and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


if __name__ == "__main__":
    main()
