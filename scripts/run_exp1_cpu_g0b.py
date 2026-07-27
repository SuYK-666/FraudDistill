from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frauddistill.eval.metrics import binary_metrics
from frauddistill.exp1_ccfa.pair_cross_encoder import labels_from_scores, select_threshold
from frauddistill.exp1_ccfa.pairlite_cpu import PAIRLITE_INPUT_MODES, PairLiteCPUDetector
from frauddistill.exp1_ccfa.paired_cluster_bootstrap import paired_cluster_bootstrap_delta
from frauddistill.utils.io import read_jsonl, write_jsonl


LEVELS = ("B0", "B1", "R1", "R2")
SEEDS = (20260724, 20260725, 20260726)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-manifest E1-CPU-v5 G0b")
    parser.add_argument("--manifest_dir", default="data/prepared/exp1_cpu_v5/g0b")
    parser.add_argument("--output_dir", default="outputs/exp1_ccfa_cpu_v5/g0b_run")
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--quick_grid", action="store_true")
    args = parser.parse_args()
    result = run_g0b(Path(args.manifest_dir), Path(args.output_dir), args.bootstrap_iterations, args.quick_grid)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_g0b(manifest_dir: Path, output_dir: Path, bootstrap_iterations: int, quick_grid: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("predictions", "tables", "models", "reports"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    train = _read(manifest_dir / "g0_train.jsonl")
    model_dev = _read(manifest_dir / "g0_model_dev.jsonl")
    threshold_dev = _read(manifest_dir / "g0_threshold_dev.jsonl")
    p1 = _read(manifest_dir / "g0_p1_mini.jsonl")
    p2 = _read(manifest_dir / "g0_p2_mini.jsonl")

    started = time.perf_counter()
    selected = select_all_configs(train, model_dev, quick_grid)
    (output_dir / "tables" / "selected_configs.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    selection_seconds = time.perf_counter() - started

    rows = []
    predictions: dict[tuple[str, str, int, str], list[dict]] = {}
    resource_rows = []
    for seed in tqdm(SEEDS, desc="G0b stability seeds"):
        for level in LEVELS:
            for mode in PAIRLITE_INPUT_MODES:
                key = f"{level}:{mode}"
                cfg = selected[key]["config"]
                model = _fit(train, level, mode, cfg, seed)
                threshold_scores = model.predict_proba(threshold_dev, mode).tolist()
                threshold = select_threshold([row["exp1_label"] for row in threshold_dev], threshold_scores).threshold
                for panel_name, panel_rows in (("P1-mini", p1), ("P2-mini", p2)):
                    scores = model.predict_proba(panel_rows, mode).tolist()
                    pred_labels = labels_from_scores(scores, threshold)
                    pred_rows = _prediction_rows(panel_rows, pred_labels, scores, threshold, level, mode, seed, panel_name)
                    predictions[(level, mode, seed, panel_name)] = pred_rows
                    write_jsonl(output_dir / "predictions" / f"{panel_name}_{level}_{mode}_seed{seed}.jsonl", pred_rows)
                    metrics = metrics_for_panel(pred_rows, panel_name)
                    rows.append({"panel": panel_name, "level": level, "mode": mode, "seed": seed, **metrics})
                if model.profile:
                    resource_rows.append({"level": level, "mode": mode, "seed": seed, **model.profile.__dict__})

    write_table(output_dir / "tables" / "metrics_by_level_mode_seed.csv", rows)
    write_table(output_dir / "tables" / "resource_profile.csv", resource_rows)
    comparisons = compare_global_best(predictions, selected, bootstrap_iterations)
    (output_dir / "tables" / "global_best_comparisons.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(rows, comparisons, selection_seconds)
    (output_dir / "tables" / "g0b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, summary, rows, comparisons, manifest_dir)
    return summary


def select_all_configs(train: list[dict], model_dev: list[dict], quick_grid: bool) -> dict:
    selected = {}
    for level in LEVELS:
        for mode in PAIRLITE_INPUT_MODES:
            best = None
            candidates = candidate_configs(level, mode, quick_grid)
            for cfg in tqdm(candidates, desc=f"select {level}:{mode}", leave=False):
                model = _fit(train, level, mode, cfg, 20260724)
                scores = model.predict_proba(model_dev, mode).tolist()
                labels = labels_from_scores(scores, 0.5)
                metrics = binary_metrics([row["exp1_label"] for row in model_dev], labels, scores)
                rank = (metrics["macro_f1"], metrics["recall"], -cfg["alpha"], -cfg["l1_ratio"], -cfg.get("cross_weight", 1.0))
                payload = {"config": cfg, "model_dev_metrics_at_0_5": metrics, "rank": rank}
                if best is None or rank > best["rank"]:
                    best = payload
            selected[f"{level}:{mode}"] = best
    return selected


def candidate_configs(level: str, mode: str, quick_grid: bool) -> list[dict]:
    if quick_grid:
        alphas = [1e-4]
        l1_ratios = [0.05]
        char_weights = [0.5]
        cross_weights = [1.0]
    else:
        alphas = [3e-5, 1e-4, 3e-4]
        l1_ratios = [0.0, 0.05]
        char_weights = [0.5, 1.0] if level in {"B1", "R1", "R2"} else [1.0]
        cross_weights = [0.5, 1.0, 2.0] if level in {"R1", "R2"} and mode == "q_y" else [1.0]
    result = []
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            for char_weight in char_weights:
                for cross_weight in cross_weights:
                    result.append(
                        {
                            "alpha": alpha,
                            "l1_ratio": l1_ratio,
                            "max_iter": 40,
                            "char_weight": char_weight,
                            "cross_weight": cross_weight,
                            "scalar_weight": 1.0,
                        }
                    )
    return result


def _fit(rows: list[dict], level: str, mode: str, cfg: dict, seed: int) -> PairLiteCPUDetector:
    model = PairLiteCPUDetector(
        level=level,
        alpha=float(cfg["alpha"]),
        l1_ratio=float(cfg["l1_ratio"]),
        max_iter=int(cfg["max_iter"]),
        seed=seed,
        char_weight=float(cfg.get("char_weight", 1.0)),
        cross_weight=float(cfg.get("cross_weight", 1.0)),
        scalar_weight=float(cfg.get("scalar_weight", 1.0)),
    )
    return model.fit(rows, [row["exp1_label"] for row in rows], mode=mode)


def compare_global_best(predictions: dict[tuple[str, str, int, str], list[dict]], selected: dict, iterations: int) -> dict:
    best_single = _best_key(selected, modes=("q_only", "y_only"))
    best_q = _best_key(selected, modes=("q_only",))
    best_y = _best_key(selected, modes=("y_only",))
    best_qy = _best_key(selected, modes=("q_y",))
    result = {"best_q": best_q, "best_y": best_y, "best_single": best_single, "best_qy": best_qy, "panels": {}}
    metric_fn = lambda y_true, pred: float(f1_score(y_true, pred, average="macro", zero_division=0))
    for panel in ("P1-mini", "P2-mini"):
        panel_rows = []
        for seed in SEEDS:
            qy = predictions[(best_qy["level"], best_qy["mode"], seed, panel)]
            single = predictions[(best_single["level"], best_single["mode"], seed, panel)]
            y = predictions[(best_y["level"], best_y["mode"], seed, panel)]
            gold = [row["gold_label"] for row in qy]
            qy_pred = [row["pred_label"] for row in qy]
            single_pred = [row["pred_label"] for row in single]
            y_pred = [row["pred_label"] for row in y]
            clusters = [row.get("cluster_id") or row["semantic_component_id"] for row in qy]
            panel_rows.append(
                {
                    "seed": seed,
                    "delta_global_best": metric_fn(gold, qy_pred) - metric_fn(gold, single_pred),
                    "delta_y": metric_fn(gold, qy_pred) - metric_fn(gold, y_pred),
                    "bootstrap_global_best": paired_cluster_bootstrap_delta(
                        gold, single_pred, qy_pred, clusters, metric_fn, iterations=iterations, seed=seed
                    ),
                    "bootstrap_y": paired_cluster_bootstrap_delta(gold, y_pred, qy_pred, clusters, metric_fn, iterations=iterations, seed=seed),
                }
            )
        result["panels"][panel] = panel_rows
    return result


def _best_key(selected: dict, modes: tuple[str, ...]) -> dict:
    candidates = []
    for key, payload in selected.items():
        level, mode = key.split(":")
        if mode in modes:
            candidates.append((payload["rank"], level, mode, payload))
    _, level, mode, payload = max(candidates, key=lambda item: item[0])
    return {"level": level, "mode": mode, "config": payload["config"], "model_dev_metrics_at_0_5": payload["model_dev_metrics_at_0_5"]}


def summarize(rows: list[dict], comparisons: dict, selection_seconds: float) -> dict:
    summary = {"selection_seconds": selection_seconds, "panels": {}, "comparisons": comparisons}
    for panel in ("P1-mini", "P2-mini"):
        panel_rows = [row for row in rows if row["panel"] == panel and row["level"] == comparisons["best_qy"]["level"] and row["mode"] == "q_y"]
        macro = [float(row["macro_f1"]) for row in panel_rows]
        recall = [float(row["recall"]) for row in panel_rows]
        fpr = [float(row["fpr"]) for row in panel_rows]
        deltas = comparisons["panels"][panel]
        summary["panels"][panel] = {
            "best_qy_mean_macro_f1": float(np.mean(macro)) if macro else 0.0,
            "best_qy_sd_macro_f1": float(np.std(macro, ddof=1)) if len(macro) > 1 else 0.0,
            "unsafe_recall_mean": float(np.mean(recall)) if recall else 0.0,
            "fpr_mean": float(np.mean(fpr)) if fpr else 0.0,
            "delta_global_best_mean": float(np.mean([row["delta_global_best"] for row in deltas])),
            "delta_y_mean": float(np.mean([row["delta_y"] for row in deltas])),
            "positive_global_best_seeds": sum(1 for row in deltas if row["delta_global_best"] > 0),
            "positive_y_seeds": sum(1 for row in deltas if row["delta_y"] > 0),
            "ci_global_best_min_lower": min(row["bootstrap_global_best"]["ci_lower"] for row in deltas),
            "ci_y_min_lower": min(row["bootstrap_y"]["ci_lower"] for row in deltas),
        }
    summary["g0b_gate"] = gate_status(summary)
    return summary


def gate_status(summary: dict) -> dict:
    p1 = summary["panels"]["P1-mini"]
    p2 = summary["panels"]["P2-mini"]
    checks = {
        "p1_qy_macro_f1_ge_0_78": p1["best_qy_mean_macro_f1"] >= 0.78,
        "p1_delta_global_best_ge_0_03": p1["delta_global_best_mean"] >= 0.03,
        "p1_ci_lower_gt_0": p1["ci_global_best_min_lower"] > 0,
        "p1_positive_3_of_3": p1["positive_global_best_seeds"] == 3,
        "p1_unsafe_recall_ge_0_72": p1["unsafe_recall_mean"] >= 0.72,
        "p1_fpr_le_0_25": p1["fpr_mean"] <= 0.25,
        "p2_qy_macro_f1_ge_0_80": p2["best_qy_mean_macro_f1"] >= 0.80,
        "p2_delta_y_ge_0_10": p2["delta_y_mean"] >= 0.10,
        "p2_ci_lower_gt_0": p2["ci_y_min_lower"] > 0,
        "p2_positive_3_of_3": p2["positive_y_seeds"] == 3,
    }
    return {"passed": all(checks.values()), "checks": checks}


def metrics_for_panel(pred_rows: list[dict], panel_name: str) -> dict:
    metrics = binary_metrics(
        [row["gold_label"] for row in pred_rows],
        [row["pred_label"] for row in pred_rows],
        [float(row["pred_score"]) for row in pred_rows],
    )
    if panel_name == "P2-mini":
        metrics["strict_group_consistency"] = group_consistency(pred_rows)
    return metrics


def group_consistency(rows: list[dict]) -> float:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("cluster_id") or row["semantic_component_id"]), []).append(row)
    if not groups:
        return 0.0
    passed = 0
    for members in groups.values():
        if all(row["gold_label"] == row["pred_label"] for row in members):
            passed += 1
    return passed / len(groups)


def _prediction_rows(rows: list[dict], pred_labels: list[str], scores: list[float], threshold: float, level: str, mode: str, seed: int, panel: str) -> list[dict]:
    return [
        {
            "id": row["id"],
            "semantic_component_id": row["semantic_component_id"],
            "cluster_id": row.get("context_collision_group_id") or row["semantic_component_id"],
            "gold_label": row["exp1_label"],
            "pred_label": pred,
            "pred_score": score,
            "threshold": threshold,
            "level": level,
            "mode": mode,
            "seed": seed,
            "panel": panel,
            "source": row.get("source"),
        }
        for row, pred, score in zip(rows, pred_labels, scores)
    ]


def write_report(output_dir: Path, summary: dict, rows: list[dict], comparisons: dict, manifest_dir: Path) -> None:
    lines = [
        "# E1-CPU-v5 G0b Gold-Panel Gate 报告",
        "",
        "## 结论",
        "",
        f"- G0b gate: {'PASS' if summary['g0b_gate']['passed'] else 'NO-GO'}",
        f"- BestSingle-CPU: {comparisons['best_single']['level']} / {comparisons['best_single']['mode']}",
        f"- BestQY: {comparisons['best_qy']['level']} / {comparisons['best_qy']['mode']}",
        f"- manifest_dir: {manifest_dir}",
        "",
        "## Panel 汇总",
        "",
        "| panel | q+y Macro-F1 | Δglobal-best | Δy | positive global | unsafe recall | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for panel, row in summary["panels"].items():
        lines.append(
            f"| {panel} | {row['best_qy_mean_macro_f1']:.4f} | {row['delta_global_best_mean']:.4f} | {row['delta_y_mean']:.4f} | "
            f"{row['positive_global_best_seeds']}/3 | {row['unsafe_recall_mean']:.4f} | {row['fpr_mean']:.4f} |"
        )
    lines.extend(["", "## Gate 检查", ""])
    for key, value in summary["g0b_gate"]["checks"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    (output_dir / "reports" / "E1_CPU_v5_G0b_报告_中文.md").write_text("\n".join(lines), encoding="utf-8")


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read(path: Path) -> list[dict]:
    return list(read_jsonl(path))


if __name__ == "__main__":
    main()
