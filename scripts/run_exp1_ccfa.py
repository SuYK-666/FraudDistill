from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frauddistill.eval.metrics import binary_metrics
from frauddistill.exp1_ccfa.duplicate_audit import duplicate_audit
from frauddistill.exp1_ccfa.exact_mcnemar import exact_mcnemar
from frauddistill.exp1_ccfa.pair_cross_encoder import INPUT_MODES, XLMRPairCrossEncoder, labels_from_scores, select_threshold
from frauddistill.exp1_ccfa.pairlite_cpu import PAIRLITE_LEVELS, PairLiteCPUDetector
from frauddistill.exp1_ccfa.paired_cluster_bootstrap import paired_cluster_bootstrap_delta
from frauddistill.exp1_ccfa.public_gold import build_p3_from_local_qy, build_p3_v1, polyguard_rows, write_p3_manifest
from frauddistill.exp1_ccfa.semantic_components import (
    attach_semantic_components,
    explicit_label_token_audit,
    leakage_audit,
    split_by_component,
    write_component_manifest,
    write_schema,
)
from frauddistill.exp1_ccfa.stat_tests import holm_adjust, paired_cluster_permutation_delta
from frauddistill.student.pair_tfidf import PairTfidfDetector
from frauddistill.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 CCF-A q-y boundary runner")
    parser.add_argument("command", choices=["preflight", "smoke", "cpu_smoke", "g0_probe", "e1_1_dev", "e1_1_corrected", "build_public_gold", "evaluate_panel", "fit_from_frozen_train_splits", "predict_frozen_panel", "guard_audit", "e1_2_pilot", "freeze_full", "full_eval"])
    parser.add_argument("--config", default="configs/experiments/exp1_ccfa_v4.yaml")
    parser.add_argument("--input", default="data/processed/qy_v3/judged_pairs_v3.jsonl")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--backend", choices=["xlmr", "tfidf_dev", "pairlite_cpu"], default="xlmr")
    parser.add_argument("--pairlite_level", choices=list(PAIRLITE_LEVELS), default="L2")
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--panel", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--require_frozen_manifest", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime(f"{config['experiment']['id']}_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or ROOT / "outputs" / config["experiment"]["id"] / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "preflight":
        result = preflight(args.input, config_path, config, output_dir)
    elif args.command == "build_public_gold":
        result = build_public_gold(config_path, config, output_dir)
    elif args.command == "fit_from_frozen_train_splits":
        result = fit_from_frozen_train_splits(config_path, config, output_dir, args)
    elif args.command == "predict_frozen_panel":
        result = predict_frozen_panel(config_path, config, output_dir, args)
    elif args.command == "guard_audit":
        result = guard_audit(config_path, config, output_dir)
    elif args.command in {"e1_2_pilot", "freeze_full", "full_eval"}:
        result = gated_not_ready(args.command, config_path, config, output_dir, args)
    else:
        rows = load_rows(args.input, args.limit)
        if args.command in {"smoke", "cpu_smoke"}:
            rows = rows[: min(len(rows), args.limit or 60)]
            args.bootstrap_iterations = min(args.bootstrap_iterations, 200)
            seeds = [args.seed]
            if args.command == "cpu_smoke" and args.backend == "xlmr":
                args.backend = "pairlite_cpu"
        elif args.command == "g0_probe":
            seeds = parse_seeds(args.seeds, default=[20260724, 20260725, 20260726])
            args.bootstrap_iterations = max(args.bootstrap_iterations, 1000)
        elif args.command in {"e1_1_corrected", "evaluate_panel"}:
            seeds = parse_seeds(args.seeds, default=[20260724, 20260725, 20260726])
            args.bootstrap_iterations = max(args.bootstrap_iterations, 10000)
        else:
            seeds = [args.seed]
        result = run_multiseed(rows, config_path, config, output_dir, args, seeds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def preflight(input_path: str, config_path: Path, config: dict, output_dir: Path) -> dict:
    rows = attach_semantic_components(load_rows(input_path, 0))
    label_sources = sorted({str(row.get("label_provenance", "")) for row in rows})
    result = {
        "identity": run_identity(config_path, input_path),
        "experiment": config["experiment"]["id"],
        "input": input_path,
        "rows": len(rows),
        "components": len({row["semantic_component_id"] for row in rows}),
        "label_sources": label_sources,
        "formal_gate_status": "NO_GO",
        "formal_gate_reason": "E1-1 local data uses proxy labels; real Guard/public-gold panels are not frozen",
        "token_audit": explicit_label_token_audit(rows),
        "split_audit": leakage_audit({"all_local_dev": rows}),
    }
    write_common_artifacts(output_dir, rows, config, result)
    return result


def guard_audit(config_path: Path, config: dict, output_dir: Path) -> dict:
    result = {
        "identity": run_identity(config_path, ""),
        "experiment": config["experiment"]["id"],
        "command": "guard_audit",
        "formal_gate_status": "NO_GO",
        "formal_gate_reason": "real open Guard raw outputs are not present; proxy labels are not allowed for Guard Gate",
        "required_artifacts": ["guard_raw_outputs.jsonl", "guard_consensus.jsonl", "guard_audit_metrics.json"],
    }
    write_common_artifacts(output_dir, [], config, result)
    return result


def build_public_gold(config_path: Path, config: dict, output_dir: Path) -> dict:
    ensure_dirs(output_dir)
    if config["experiment"]["id"] == "exp1_ccfa_cpu_v5":
        rows, p3_audit = build_p3_v1(ROOT / "data" / "raw" / "aegis" / "test.json", polyguard_base_ids=1325)
        output_file = output_dir / "data" / "p3_external_public_gold_v1.jsonl"
        write_p3_manifest(output_file, rows)
        labels = _count_by(rows, "exp1_label")
        sources = _count_by(rows, "source")
        result = {
            "identity": run_identity(config_path, ""),
            "experiment": config["experiment"]["id"],
            "command": "build_public_gold",
            "rows": len(rows),
            "components": len({row["semantic_component_id"] for row in rows}),
            "by_label": labels,
            "by_source": sources,
            "output_file": str(output_file),
            "p3_audit": p3_audit,
            "formal_gate_status": "DATA_READY" if p3_audit["passed"] else "NO_GO",
        }
        write_common_artifacts(output_dir, rows, config, result)
        return result

    input_files = [
        ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "aegis_qy.jsonl",
        ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "do_not_answer_qy.jsonl",
    ]
    target_min = int(config["data_targets"]["p3_public_gold_rows_min"])
    rows = []
    try:
        rows.extend(polyguard_rows(max_rows=1325))
    except Exception as exc:
        (output_dir / "data" / "polyguard_download_error.txt").write_text(str(exc), encoding="utf-8")
    local_target = max(4614, target_min) - len(rows)
    if local_target > 0:
        rows.extend(build_p3_from_local_qy(input_files, max_rows=local_target))
    rows = attach_semantic_components(rows[: max(4614, target_min)])
    output_file = output_dir / "data" / "p3_external_public_gold.jsonl"
    write_p3_manifest(output_file, rows)
    labels: dict[str, int] = {}
    sources: dict[str, int] = {}
    for row in rows:
        labels[row["exp1_label"]] = labels.get(row["exp1_label"], 0) + 1
        sources[str(row.get("source"))] = sources.get(str(row.get("source")), 0) + 1
    result = {
        "identity": run_identity(config_path, ""),
        "experiment": config["experiment"]["id"],
        "command": "build_public_gold",
        "rows": len(rows),
        "components": len({row["semantic_component_id"] for row in rows}),
        "by_label": labels,
        "by_source": sources,
        "output_file": str(output_file),
        "p3_min_rows": target_min,
        "p3_row_gate": len(rows) >= target_min,
        "label_provenance": "external_public_official_gold",
        "formal_gate_status": "DATA_READY" if len(rows) >= target_min else "NO_GO",
    }
    write_common_artifacts(output_dir, rows, config, result)
    return result


def gated_not_ready(command: str, config_path: Path, config: dict, output_dir: Path, args: argparse.Namespace) -> dict:
    result = {
        "identity": run_identity(config_path, args.input),
        "experiment": config["experiment"]["id"],
        "command": command,
        "formal_gate_status": "NO_GO",
        "formal_gate_reason": "document requires E1-1.1 and real Guard audit before E1-2/Full; frozen manifests are absent",
        "require_frozen_manifest": bool(args.require_frozen_manifest),
    }
    write_common_artifacts(output_dir, [], config, result)
    return result


def run_multiseed(rows: list[dict], config_path: Path, config: dict, output_dir: Path, args: argparse.Namespace, seeds: list[int]) -> dict:
    rows = attach_semantic_components(rows)
    ensure_dirs(output_dir)
    all_metric_rows: list[dict] = []
    seed_summaries: list[dict] = []
    for seed in tqdm(seeds, desc="E1 seeds"):
        seed_dir = output_dir / f"seed_{seed}"
        ensure_dirs(seed_dir)
        train, model_dev, threshold_dev, split_audit = split_by_component(rows, seed)
        if not split_audit["passed"]:
            raise RuntimeError(f"semantic component leakage detected for seed {seed}: {split_audit}")
        duplicate = duplicate_audit({"train": train, "model_dev": model_dev, "threshold_dev": threshold_dev})
        predictions_by_mode: dict[str, list[dict]] = {}
        metrics_rows: list[dict] = []
        selected_configs: dict[str, dict] = {}
        for mode in tqdm(INPUT_MODES, desc=f"seed {seed} modes", leave=False):
            predictions, selected = train_and_predict_mode(mode, train, model_dev, threshold_dev, config, args, seed)
            predictions_by_mode[mode] = predictions
            selected_configs[mode] = selected
            write_jsonl(seed_dir / "predictions" / f"{mode}.jsonl", predictions)
            metrics = metrics_for_predictions(predictions)
            row = {"seed": seed, "panel": "E1-1.1-corrected-local-dev", "input_mode": mode, **metrics}
            metrics_rows.append(row)
            all_metric_rows.append(row)
        stats = compare_modes(predictions_by_mode, args.bootstrap_iterations, seed)
        gate = development_gate(metrics_rows, stats, args.backend)
        (seed_dir / "statistics" / "mode_comparisons.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        (seed_dir / "training_selected_config.json").write_text(json.dumps(selected_configs, ensure_ascii=False, indent=2), encoding="utf-8")
        (seed_dir / "split_leakage_audit.json").write_text(json.dumps(split_audit, ensure_ascii=False, indent=2), encoding="utf-8")
        (seed_dir / "data" / "duplicate_audit.json").write_text(json.dumps(duplicate, ensure_ascii=False, indent=2), encoding="utf-8")
        seed_summaries.append({"seed": seed, "split_audit": split_audit, "duplicate_audit": duplicate, "development_gate": gate})
    summary = summarize_seeds(all_metric_rows, seed_summaries)
    write_metrics(output_dir / "tables" / "metrics_by_seed.csv", all_metric_rows)
    (output_dir / "tables" / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    status = {"identity": run_identity(config_path, args.input), "seed_summaries": seed_summaries, "summary": summary}
    write_common_artifacts(output_dir, rows, config, status)
    write_report(output_dir, all_metric_rows, status, rows, args.backend, seeds)
    return {
        "experiment": config["experiment"]["id"],
        "backend": args.backend,
        "seeds": seeds,
        "rows": len(rows),
        "components": len({row["semantic_component_id"] for row in rows}),
        "output_dir": str(output_dir),
        "summary": summary,
    }


def train_and_predict_mode(
    mode: str,
    train: list[dict],
    model_dev: list[dict],
    threshold_dev: list[dict],
    config: dict,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[dict], dict]:
    labels = [row["exp1_label"] for row in train]
    if args.backend == "tfidf_dev":
        model = PairTfidfDetector()
        model.fit(train, labels, mode=mode)
        dev_scores = model.predict_proba(model_dev, mode).tolist()
        selected = {"backend": "tfidf_dev", "mode_specific_fit": True}
        scores = model.predict_proba(threshold_dev, mode).tolist()
    elif args.backend == "pairlite_cpu":
        model = PairLiteCPUDetector(
            level=args.pairlite_level,
            alpha=float(args.lr or config.get("pairlite_cpu", {}).get("default_alpha", 3e-5)),
            l1_ratio=float(config.get("pairlite_cpu", {}).get("default_l1_ratio", 0.05)),
            max_iter=int(args.epochs or config.get("pairlite_cpu", {}).get("default_max_iter", 40)),
            seed=seed,
        )
        model.fit(train, labels, mode=mode)
        dev_scores = model.predict_proba(model_dev, mode).tolist()
        selected = {"backend": "pairlite_cpu", "level": args.pairlite_level, "profile": model.profile.__dict__ if model.profile else {}}
        scores = model.predict_proba(threshold_dev, mode).tolist()
    else:
        lr = float(args.lr or config["model"]["lr_search"][1])
        epochs = int(args.epochs or config["model"]["epochs_search"][0])
        batch_size = int(args.batch_size or config["model"]["batch_size"])
        model = XLMRPairCrossEncoder(
            model_name=config["model"]["backbone"],
            max_length=int(config["model"]["max_length"]),
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        model.fit(train, labels, mode)
        dev_scores = model.predict_scores(model_dev, mode)
        selected = {"backend": "xlmr", "model": config["model"]["backbone"], "lr": lr, "epochs": epochs, "batch_size": batch_size}
        scores = model.predict_scores(threshold_dev, mode)
    threshold = select_threshold([row["exp1_label"] for row in model_dev], dev_scores).threshold if model_dev else 0.5
    pred_labels = labels_from_scores(scores, threshold)
    return [
        {
            "id": row["id"],
            "semantic_component_id": row["semantic_component_id"],
            "gold_label": row["exp1_label"],
            "pred_label": pred,
            "pred_score": score,
            "threshold": threshold,
            "input_mode": mode,
            "label_provenance": row.get("label_provenance"),
            "source": row.get("source"),
            "target_model": row.get("target_model"),
        }
        for row, pred, score in zip(threshold_dev, pred_labels, scores)
    ], {**selected, "threshold": threshold}


def fit_from_frozen_train_splits(config_path: Path, config: dict, output_dir: Path, args: argparse.Namespace) -> dict:
    rows = attach_semantic_components(load_rows(args.input, args.limit))
    train, model_dev, threshold_dev, split_audit = split_by_component(rows, args.seed)
    if not split_audit["passed"]:
        raise RuntimeError(f"semantic component leakage detected: {split_audit}")
    model_dir = Path(args.model_dir or output_dir / "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for mode in tqdm(INPUT_MODES, desc="fit frozen modes"):
        model = PairLiteCPUDetector(level=args.pairlite_level, seed=args.seed)
        model.fit(train, [row["exp1_label"] for row in train], mode=mode)
        dev_scores = model.predict_proba(model_dev, mode).tolist()
        threshold = select_threshold([row["exp1_label"] for row in model_dev], dev_scores).threshold if model_dev else 0.5
        artifact = model_dir / f"pairlite_{args.pairlite_level}_{mode}_seed{args.seed}.joblib"
        joblib.dump({"model": model, "mode": mode, "threshold": threshold, "seed": args.seed, "config": config}, artifact, compress=3)
        artifacts.append({"mode": mode, "artifact": str(artifact), "threshold": threshold, "profile": model.profile.__dict__ if model.profile else {}})
    result = {
        "identity": run_identity(config_path, args.input),
        "experiment": config["experiment"]["id"],
        "command": "fit_from_frozen_train_splits",
        "backend": "pairlite_cpu",
        "level": args.pairlite_level,
        "split_audit": split_audit,
        "duplicate_audit": duplicate_audit({"train": train, "model_dev": model_dev, "threshold_dev": threshold_dev}),
        "artifacts": artifacts,
    }
    write_common_artifacts(output_dir, rows, config, result)
    return result


def predict_frozen_panel(config_path: Path, config: dict, output_dir: Path, args: argparse.Namespace) -> dict:
    if not args.model_dir:
        raise ValueError("--model_dir is required for predict_frozen_panel")
    panel_path = Path(args.panel or args.input)
    rows = attach_semantic_components(load_rows(str(panel_path), args.limit))
    predictions_by_mode = {}
    metric_rows = []
    for mode in tqdm(INPUT_MODES, desc="predict frozen panel"):
        artifact = Path(args.model_dir) / f"pairlite_{args.pairlite_level}_{mode}_seed{args.seed}.joblib"
        payload = joblib.load(artifact)
        model = payload["model"]
        threshold = float(payload["threshold"])
        scores = model.predict_proba(rows, mode).tolist()
        preds = labels_from_scores(scores, threshold)
        predictions = [
            {
                "id": row["id"],
                "semantic_component_id": row["semantic_component_id"],
                "gold_label": row["exp1_label"],
                "pred_label": pred,
                "pred_score": score,
                "threshold": threshold,
                "input_mode": mode,
                "source": row.get("source"),
            }
            for row, pred, score in zip(rows, preds, scores)
        ]
        predictions_by_mode[mode] = predictions
        write_jsonl(output_dir / "predictions" / f"{mode}.jsonl", predictions)
        metric_rows.append({"seed": args.seed, "panel": panel_path.stem, "input_mode": mode, **metrics_for_predictions(predictions)})
    write_metrics(output_dir / "tables" / "frozen_panel_metrics.csv", metric_rows)
    result = {
        "identity": run_identity(config_path, str(panel_path)),
        "experiment": config["experiment"]["id"],
        "command": "predict_frozen_panel",
        "panel": str(panel_path),
        "rows": len(rows),
        "components": len({row["semantic_component_id"] for row in rows}),
        "metrics": metric_rows,
        "fit_called": False,
    }
    write_common_artifacts(output_dir, rows, config, result)
    return result


def compare_modes(predictions_by_mode: dict[str, list[dict]], iterations: int, seed: int) -> dict:
    metric_fn = lambda y_true, pred: float(f1_score(y_true, pred, average="macro", zero_division=0))
    qy = predictions_by_mode["q_y"]
    gold = [row["gold_label"] for row in qy]
    clusters = [row["semantic_component_id"] for row in qy]
    qy_pred = [row["pred_label"] for row in qy]
    comparisons = {}
    p_values = {}
    for baseline in ("q_only", "y_only"):
        base_pred = [row["pred_label"] for row in predictions_by_mode[baseline]]
        key = f"q_y_vs_{baseline}"
        comparisons[key] = {
            "cluster_bootstrap": paired_cluster_bootstrap_delta(gold, base_pred, qy_pred, clusters, metric_fn, iterations=iterations, seed=seed),
            "cluster_permutation": paired_cluster_permutation_delta(gold, base_pred, qy_pred, clusters, metric_fn, iterations=iterations, seed=seed),
            "supplementary_exact_mcnemar": exact_mcnemar(gold, base_pred, qy_pred),
        }
        p_values[key] = float(comparisons[key]["cluster_permutation"]["one_sided_p"])
    comparisons["holm_correction"] = holm_adjust(p_values)
    return comparisons


def development_gate(metrics_rows: list[dict], stats: dict, backend: str) -> dict:
    by_mode = {row["input_mode"]: row for row in metrics_rows}
    best_name = "q_only" if by_mode["q_only"]["macro_f1"] >= by_mode["y_only"]["macro_f1"] else "y_only"
    qy = by_mode["q_y"]["macro_f1"]
    best_single = by_mode[best_name]["macro_f1"]
    best_stats = stats[f"q_y_vs_{best_name}"]["cluster_bootstrap"]
    return {
        "backend": backend,
        "formal_pass": False,
        "formal_pass_reason": "E1-1.1 uses local proxy labels and is implementation validation only",
        "best_single_mode": best_name,
        "direction_positive": qy - best_single > 0,
        "q_y_macro_f1": qy,
        "best_single_macro_f1": best_single,
        "delta_vs_best_single": qy - best_single,
        "delta_best_ci_lower": best_stats["ci_lower"],
    }


def metrics_for_predictions(predictions: list[dict]) -> dict:
    return binary_metrics(
        [row["gold_label"] for row in predictions],
        [row["pred_label"] for row in predictions],
        [float(row["pred_score"]) for row in predictions],
    )


def summarize_seeds(metrics_rows: list[dict], seed_summaries: list[dict]) -> dict:
    qy_rows = [row for row in metrics_rows if row["input_mode"] == "q_y"]
    positives = [summary["development_gate"]["direction_positive"] for summary in seed_summaries]
    deltas = [float(summary["development_gate"]["delta_vs_best_single"]) for summary in seed_summaries]
    return {
        "seed_count": len(seed_summaries),
        "positive_direction_count": sum(1 for value in positives if value),
        "mean_q_y_macro_f1": sum(float(row["macro_f1"]) for row in qy_rows) / max(len(qy_rows), 1),
        "mean_delta_vs_best_single": sum(deltas) / max(len(deltas), 1),
        "min_delta_vs_best_single": min(deltas) if deltas else None,
        "formal_gate_status": "NO_GO",
        "formal_gate_reason": "real Guard/public-gold and E1-2 pilot gates are not complete",
    }


def write_report(output_dir: Path, metrics_rows: list[dict], status: dict, rows: list[dict], backend: str, seeds: list[int]) -> None:
    identity = status["identity"]
    lines = [
        "# 实验1 E1-1.1 代码纠偏与本地开发验证报告",
        "",
        "## 运行身份",
        "",
        f"- git_commit: {identity['git_commit']}",
        f"- dirty_worktree: {identity['dirty_worktree']}",
        f"- backend: {backend}",
        f"- seeds: {','.join(str(seed) for seed in seeds)}",
        f"- config_sha256: {identity['config_sha256']}",
        f"- data_manifest_sha256: {identity['data_manifest_sha256']}",
        "",
        "## 结论",
        "",
        "本轮只用于 E1-1.1 实现可信度验证，仍然使用本地 proxy/silver 标签，不能作为 E1 正式 PASS。",
        f"3-seed 方向计数：{status['summary']['positive_direction_count']}/{status['summary']['seed_count']}；mean q+y Macro-F1={status['summary']['mean_q_y_macro_f1']:.4f}；mean Δbest={status['summary']['mean_delta_vs_best_single']:.4f}。",
        "",
        "## 数据审计",
        "",
        f"- 总行数：{len(rows)}",
        f"- 总 semantic components：{len({row['semantic_component_id'] for row in rows})}",
        f"- 标签来源：{', '.join(sorted({str(row.get('label_provenance')) for row in rows}))}",
        "",
        "## 指标表",
        "",
        "| seed | input_mode | Macro-F1 | Accuracy | Precision | Recall | FPR | FNR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['seed']} | {row['input_mode']} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['fpr']:.4f} | {row['fnr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 分析",
            "",
            "本轮修复了上一版的主要审计问题：报告明确 backend 与真实 evaluation N，TF-IDF sanity baseline 改为 mode-specific fit，统计同时覆盖 q+y vs q-only、q+y vs y-only，并写出 cluster bootstrap、component-aware permutation、supplementary McNemar 与 Holm 校正。",
            "由于标签仍不是真实 Guard 或 public gold，本结果只能决定是否继续进入真实 Guard audit 与 E1-2 pilot，不能启动 Full，也不能启动实验 2。",
        ]
    )
    (output_dir / "reports" / "EXP1_CCF_A_E1_1_1_CORRECTED_REPORT_中文.md").write_text("\n".join(lines), encoding="utf-8")


def write_common_artifacts(output_dir: Path, rows: list[dict], config: dict, status: dict) -> None:
    ensure_dirs(output_dir)
    write_jsonl(output_dir / "data" / "all_rows_with_semantic_components.jsonl", rows)
    write_component_manifest(output_dir / "data" / "semantic_components.jsonl", rows)
    write_schema(output_dir / "data" / "all_rows_schema.json", rows)
    (output_dir / "config_lock.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output_dir / "exp1_gate_checklist.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "panel", "input_mode", "accuracy", "precision", "recall", "macro_f1", "fpr", "fnr", "auroc", "auprc"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def ensure_dirs(path: Path) -> None:
    for subdir in ("data", "tables", "statistics", "predictions", "reports"):
        (path / subdir).mkdir(parents=True, exist_ok=True)


def run_identity(config_path: Path, input_path: str) -> dict:
    return {
        "git_commit": git(["rev-parse", "HEAD"]),
        "dirty_worktree": bool(git(["status", "--short"])),
        "config_sha256": file_sha256(config_path),
        "data_manifest_sha256": file_sha256(Path(input_path)) if input_path and Path(input_path).exists() else "",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
    }


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(value: str, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_rows(path: str, limit: int) -> list[dict]:
    rows = list(read_jsonl(path))
    if limit and limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def _count_by(rows: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field))
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    main()
