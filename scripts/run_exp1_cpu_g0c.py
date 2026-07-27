from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frauddistill.eval.metrics import binary_metrics
from frauddistill.exp1_ccfa.exact_mcnemar import exact_mcnemar
from frauddistill.exp1_ccfa.pair_cross_encoder import labels_from_scores
from frauddistill.exp1_ccfa.pairlite_cpu import PAIRLITE_INPUT_MODES, PairLiteCPUDetector
from frauddistill.exp1_ccfa.paired_cluster_bootstrap import paired_cluster_bootstrap_delta
from frauddistill.exp1_ccfa.stat_tests import holm_adjust, paired_cluster_permutation_delta
from frauddistill.utils.io import read_jsonl, write_jsonl


CONFIG_PATH = ROOT / "configs" / "experiments" / "exp1_ccfa_cpu_v5.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1-CPU-v5 G0c with hard data-gate stop")
    parser.add_argument("--manifest_dir", default="data/prepared/exp1_cpu_v5/g0c")
    parser.add_argument("--output_dir", default="outputs/exp1_ccfa_cpu_v5/g0c_run")
    parser.add_argument("--bootstrap_iterations", type=int, default=None)
    parser.add_argument("--quick_grid", action="store_true")
    parser.add_argument("--allow_diagnostic", action="store_true")
    args = parser.parse_args()
    summary = run_g0c(ROOT / args.manifest_dir, ROOT / args.output_dir, args.bootstrap_iterations, args.quick_grid, args.allow_diagnostic)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("status") in {"DATA_GATE_FAIL", "FAIL"}:
        raise SystemExit(2)


def run_g0c(manifest_dir: Path, output_dir: Path, bootstrap_iterations: int | None, quick_grid: bool, allow_diagnostic: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("predictions", "tables", "reports"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    iterations = int(bootstrap_iterations or config["statistics"]["g0c_bootstrap_iterations"])
    data_audit = load_data_audit_or_fail(manifest_dir, allow_diagnostic)
    if not data_audit["passed"] and not allow_diagnostic:
        summary = {"status": "DATA_GATE_FAIL", "data_gate_passed": False, "manifest_dir": str(manifest_dir)}
        write_report(output_dir, summary, [], {}, data_audit, diagnostic=False)
        return summary

    train = read_rows(manifest_dir / "g0_train.jsonl")
    model_dev = read_rows(manifest_dir / "g0_model_dev.jsonl")
    threshold_dev = read_rows(manifest_dir / "g0_threshold_dev.jsonl")
    p1 = read_rows(manifest_dir / "g0_p1_mini.jsonl")
    p2 = read_rows(manifest_dir / "g0_p2_mini.jsonl")
    seeds = list(config["statistics"]["g0_seeds"])
    levels = list(config["pairlite_cpu"]["levels"])
    started = time.perf_counter()
    selected = select_all_configs(train, model_dev, config, quick_grid)
    (output_dir / "tables" / "selected_configs.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    resource_rows = []
    predictions: dict[tuple[str, str, int, str], list[dict]] = {}
    for seed in tqdm(seeds, desc="G0c stability seeds"):
        for level in levels:
            for mode in PAIRLITE_INPUT_MODES:
                cfg = selected[f"{level}:{mode}"]["config"]
                model = fit_model(train, level, mode, cfg, seed)
                threshold_scores = model.predict_proba(threshold_dev, mode).tolist()
                threshold_info = select_constrained_threshold([row["exp1_label"] for row in threshold_dev], threshold_scores, config["threshold_policy"])
                for panel_name, panel_rows in (("P1-mini", p1), ("P2-mini", p2)):
                    scores = model.predict_proba(panel_rows, mode).tolist()
                    pred_labels = labels_from_scores(scores, threshold_info["threshold"])
                    pred_rows = prediction_rows(panel_rows, pred_labels, scores, threshold_info, level, mode, seed, panel_name)
                    predictions[(level, mode, seed, panel_name)] = pred_rows
                    write_jsonl(output_dir / "predictions" / f"{panel_name}_{level}_{mode}_seed{seed}.jsonl", pred_rows)
                    rows.append({"panel": panel_name, "level": level, "mode": mode, "seed": seed, **metrics_for_panel(pred_rows, panel_name)})
                if model.profile:
                    resource_rows.append({"level": level, "mode": mode, "seed": seed, **model.profile.__dict__})

    write_table(output_dir / "tables" / "metrics_by_level_mode_seed.csv", rows)
    write_table(output_dir / "tables" / "resource_profile.csv", resource_rows)
    comparisons = compare_best(predictions, selected, seeds, iterations)
    (output_dir / "tables" / "global_best_comparisons.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(rows, comparisons, data_audit, config, time.perf_counter() - started, allow_diagnostic)
    (output_dir / "tables" / "g0c_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, summary, rows, comparisons, data_audit, allow_diagnostic)
    return summary


def load_data_audit_or_fail(manifest_dir: Path, allow_diagnostic: bool = False) -> dict:
    path = manifest_dir / "g0c_data_audit.json"
    if not path.exists():
        raise FileNotFoundError(f"missing data audit: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not audit.get("passed") and not allow_diagnostic:
        return audit
    return audit


def select_all_configs(train: list[dict], model_dev: list[dict], config: dict, quick_grid: bool) -> dict:
    selected = {}
    for level in config["pairlite_cpu"]["levels"]:
        for mode in PAIRLITE_INPUT_MODES:
            best = None
            for cfg in tqdm(candidate_configs(level, mode, config, quick_grid), desc=f"select {level}:{mode}", leave=False):
                model = fit_model(train, level, mode, cfg, int(config["statistics"]["g0_seeds"][0]))
                scores = model.predict_proba(model_dev, mode).tolist()
                threshold = select_constrained_threshold([row["exp1_label"] for row in model_dev], scores, config["threshold_policy"])
                metrics = threshold["metrics"]
                rank = (metrics["macro_f1"], metrics["recall"], -metrics["fpr"], -cfg["alpha"], -cfg.get("cross_weight", 1.0))
                payload = {"config": cfg, "model_dev_metrics_constrained": metrics, "rank": rank}
                if best is None or rank > best["rank"]:
                    best = payload
            selected[f"{level}:{mode}"] = best
    return selected


def candidate_configs(level: str, mode: str, config: dict, quick_grid: bool) -> list[dict]:
    cpu = config["pairlite_cpu"]
    alphas = [float(cpu["default_alpha"])] if quick_grid else [float(x) for x in cpu["alpha_grid"]]
    char_weights = [1.0]
    cross_weights = [1.0]
    if level in {"B1", "R1", "R2"}:
        char_weights = [float(cpu["char_weight_grid"][0])] if quick_grid else [float(x) for x in cpu["char_weight_grid"]]
    if level in {"R1", "R2"} and mode == "q_y":
        cross_weights = [float(cpu["cross_weight_grid"][1])] if quick_grid else [float(x) for x in cpu["cross_weight_grid"]]
    result = []
    for alpha in alphas:
        for char_weight in char_weights:
            for cross_weight in cross_weights:
                result.append(
                    {
                        "alpha": alpha,
                        "l1_ratio": 0.0,
                        "max_iter": 40,
                        "char_weight": char_weight,
                        "cross_weight": cross_weight,
                        "scalar_weight": 1.0,
                    }
                )
    return result


def fit_model(rows: list[dict], level: str, mode: str, cfg: dict, seed: int) -> PairLiteCPUDetector:
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


def select_constrained_threshold(labels: list[str], scores: list[float], policy: dict) -> dict:
    best = None
    least_violation = None
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = labels_from_scores(scores, float(threshold))
        metrics = binary_metrics(labels, pred, scores)
        feasible = metrics["recall"] >= float(policy["unsafe_recall_min"]) and metrics["fpr"] <= float(policy["fpr_max"])
        violation = max(0.0, float(policy["unsafe_recall_min"]) - metrics["recall"]) + max(0.0, metrics["fpr"] - float(policy["fpr_max"]))
        payload = {"threshold": float(threshold), "metrics": metrics, "feasible": feasible, "violation": float(violation)}
        if feasible and (best is None or (metrics["macro_f1"], metrics["recall"], -metrics["fpr"]) > (best["metrics"]["macro_f1"], best["metrics"]["recall"], -best["metrics"]["fpr"])):
            best = payload
        if least_violation is None or (violation, -metrics["macro_f1"]) < (least_violation["violation"], -least_violation["metrics"]["macro_f1"]):
            least_violation = payload
    return best or least_violation


def compare_best(predictions: dict[tuple[str, str, int, str], list[dict]], selected: dict, seeds: list[int], iterations: int) -> dict:
    best_q = best_key(selected, ("q_only",))
    best_y = best_key(selected, ("y_only",))
    best_single = best_key(selected, ("q_only", "y_only"))
    best_qy = best_key(selected, ("q_y",))
    relation = {"level": "R1", "mode": "q_y", "config": selected["R1:q_y"]["config"]}
    b1_qy = {"level": "B1", "mode": "q_y", "config": selected["B1:q_y"]["config"]}
    result = {"best_q": best_q, "best_y": best_y, "best_single": best_single, "best_qy": best_qy, "relation_qy": relation, "b1_qy": b1_qy, "panels": {}}
    p_values = {}
    for panel in ("P1-mini", "P2-mini"):
        seed_rows = []
        for seed in seeds:
            qy = predictions[(best_qy["level"], "q_y", seed, panel)]
            single = predictions[(best_single["level"], best_single["mode"], seed, panel)]
            y = predictions[(best_y["level"], "y_only", seed, panel)]
            relation_rows = predictions[("R1", "q_y", seed, panel)]
            b1_rows = predictions[("B1", "q_y", seed, panel)]
            gold = [row["gold_label"] for row in qy]
            metric_fn = lambda yt, pred: float(f1_score(yt, pred, average="macro", zero_division=0))
            seed_rows.append(
                {
                    "seed": seed,
                    "delta_best_single": metric_fn(gold, [r["pred_label"] for r in qy]) - metric_fn(gold, [r["pred_label"] for r in single]),
                    "delta_y": metric_fn(gold, [r["pred_label"] for r in qy]) - metric_fn(gold, [r["pred_label"] for r in y]),
                    "relation_gain_vs_b1_qy": metric_fn(gold, [r["pred_label"] for r in relation_rows]) - metric_fn(gold, [r["pred_label"] for r in b1_rows]),
                }
            )
        pooled = pooled_comparison(predictions, best_qy, best_single, best_y, relation, b1_qy, seeds, panel, iterations)
        result["panels"][panel] = {"per_seed": seed_rows, "pooled": pooled}
        p_values[f"{panel}:qy_vs_single"] = pooled["mcnemar_best_single"]["p_value"]
        p_values[f"{panel}:qy_vs_y"] = pooled["mcnemar_y"]["p_value"]
    result["holm"] = holm_adjust(p_values)
    return result


def pooled_comparison(predictions: dict, best_qy: dict, best_single: dict, best_y: dict, relation: dict, b1: dict, seeds: list[int], panel: str, iterations: int) -> dict:
    qy_rows = predictions[(best_qy["level"], "q_y", seeds[0], panel)]
    gold = [row["gold_label"] for row in qy_rows]
    clusters = [row["cluster_id"] for row in qy_rows]
    qy = pooled_labels(predictions, best_qy["level"], "q_y", seeds, panel)
    single = pooled_labels(predictions, best_single["level"], best_single["mode"], seeds, panel)
    y = pooled_labels(predictions, best_y["level"], "y_only", seeds, panel)
    rel = pooled_labels(predictions, relation["level"], relation["mode"], seeds, panel)
    b1_labels = pooled_labels(predictions, b1["level"], b1["mode"], seeds, panel)
    metric_fn = lambda yt, pred: float(f1_score(yt, pred, average="macro", zero_division=0))
    return {
        "macro_f1_qy": metric_fn(gold, qy),
        "delta_best_single": metric_fn(gold, qy) - metric_fn(gold, single),
        "delta_y": metric_fn(gold, qy) - metric_fn(gold, y),
        "relation_gain_vs_b1_qy": metric_fn(gold, rel) - metric_fn(gold, b1_labels),
        "bootstrap_best_single": paired_cluster_bootstrap_delta(gold, single, qy, clusters, metric_fn, iterations=iterations, seed=20260727),
        "bootstrap_y": paired_cluster_bootstrap_delta(gold, y, qy, clusters, metric_fn, iterations=iterations, seed=20260728),
        "permutation_best_single": paired_cluster_permutation_delta(gold, single, qy, clusters, metric_fn, iterations=max(500, iterations // 5), seed=20260729),
        "mcnemar_best_single": exact_mcnemar(gold, single, qy),
        "mcnemar_y": exact_mcnemar(gold, y, qy),
    }


def pooled_labels(predictions: dict, level: str, mode: str, seeds: list[int], panel: str) -> list[str]:
    seed_scores = []
    thresholds = []
    for seed in seeds:
        rows = predictions[(level, mode, seed, panel)]
        seed_scores.append([float(row["pred_score"]) for row in rows])
        thresholds.append(float(rows[0]["threshold"]))
    scores = np.mean(np.asarray(seed_scores, dtype=float), axis=0).tolist()
    return labels_from_scores(scores, float(np.mean(thresholds)))


def best_key(selected: dict, modes: tuple[str, ...]) -> dict:
    candidates = []
    for key, payload in selected.items():
        level, mode = key.split(":")
        if mode in modes:
            candidates.append((payload["rank"], level, mode, payload))
    _, level, mode, payload = max(candidates, key=lambda item: item[0])
    return {"level": level, "mode": mode, "config": payload["config"], "model_dev_metrics_constrained": payload["model_dev_metrics_constrained"]}


def summarize(rows: list[dict], comparisons: dict, data_audit: dict, config: dict, wall_seconds: float, diagnostic: bool) -> dict:
    summary = {
        "status": "DIAGNOSTIC" if diagnostic else "PENDING",
        "diagnostic_not_eligible_for_gate": bool(diagnostic),
        "data_gate_passed": bool(data_audit.get("passed")),
        "git_commit": git_commit(),
        "wall_seconds": wall_seconds,
        "selected": {key: comparisons[key] for key in ("best_q", "best_y", "best_single", "best_qy", "relation_qy", "b1_qy")},
        "panels": {},
    }
    for panel in ("P1-mini", "P2-mini"):
        qy_level = comparisons["best_qy"]["level"]
        panel_rows = [row for row in rows if row["panel"] == panel and row["level"] == qy_level and row["mode"] == "q_y"]
        pooled = comparisons["panels"][panel]["pooled"]
        summary["panels"][panel] = {
            "best_qy_mean_macro_f1": mean([row["macro_f1"] for row in panel_rows]),
            "best_qy_sd_macro_f1": sd([row["macro_f1"] for row in panel_rows]),
            "unsafe_recall_mean": mean([row["recall"] for row in panel_rows]),
            "fpr_mean": mean([row["fpr"] for row in panel_rows]),
            "strict_group_consistency_mean": mean([row.get("strict_group_consistency", 0.0) for row in panel_rows]) if panel == "P2-mini" else None,
            "pooled": pooled,
            "positive_best_single_seeds": sum(1 for row in comparisons["panels"][panel]["per_seed"] if row["delta_best_single"] > 0),
            "positive_y_seeds": sum(1 for row in comparisons["panels"][panel]["per_seed"] if row["delta_y"] > 0),
            "positive_relation_gain_seeds": sum(1 for row in comparisons["panels"][panel]["per_seed"] if row["relation_gain_vs_b1_qy"] > 0),
        }
    summary["g0c_gate"] = gate_status(summary, config)
    if not diagnostic:
        if summary["g0c_gate"]["passed"]:
            summary["status"] = "PASS"
        elif summary["g0c_gate"]["amber"]:
            summary["status"] = "AMBER"
        else:
            summary["status"] = "FAIL"
    return summary


def gate_status(summary: dict, config: dict) -> dict:
    gates = config["pass_gates"]["g0"]
    p1 = summary["panels"]["P1-mini"]
    p2 = summary["panels"]["P2-mini"]
    checks = {
        "data_gate_passed": summary["data_gate_passed"],
        "p1_qy_macro_f1_ge_0_78": p1["pooled"]["macro_f1_qy"] >= float(gates["p1_mini_q_y_macro_f1_min"]),
        "p1_delta_best_single_ge_0_03": p1["pooled"]["delta_best_single"] >= float(gates["p1_mini_delta_best_single_min"]),
        "p1_ci_lower_gt_0": p1["pooled"]["bootstrap_best_single"]["ci_lower"] > float(gates["ci_lower_min"]),
        "p1_positive_3_of_3": p1["positive_best_single_seeds"] >= int(gates["positive_seeds_min"]),
        "p1_unsafe_recall_ge_0_72": p1["unsafe_recall_mean"] >= float(gates["p1_mini_unsafe_recall_min"]),
        "p1_fpr_le_0_25": p1["fpr_mean"] <= float(gates["p1_mini_fpr_max"]),
        "p2_qy_macro_f1_ge_0_80": p2["pooled"]["macro_f1_qy"] >= float(gates["p2_mini_q_y_macro_f1_min"]),
        "p2_delta_y_ge_0_10": p2["pooled"]["delta_y"] >= float(gates["p2_mini_delta_y_only_min"]),
        "p2_ci_lower_gt_0": p2["pooled"]["bootstrap_y"]["ci_lower"] > float(gates["ci_lower_min"]),
        "p2_positive_3_of_3": p2["positive_y_seeds"] >= int(gates["positive_seeds_min"]),
        "p2_group_consistency_ge_0_70": (p2["strict_group_consistency_mean"] or 0.0) >= float(gates["p2_mini_group_consistency_min"]),
        "p2_relation_gain_ge_0_02": p2["pooled"]["relation_gain_vs_b1_qy"] >= float(gates["p2_relation_gain_min"]),
    }
    amber = (
        summary["data_gate_passed"]
        and p1["pooled"]["macro_f1_qy"] >= float(gates["amber_p1_macro_f1_min"])
        and p2["pooled"]["macro_f1_qy"] >= float(gates["amber_p2_macro_f1_min"])
        and p1["pooled"]["delta_best_single"] > 0
        and p2["pooled"]["delta_y"] > 0
        and p1["pooled"]["bootstrap_best_single"]["ci_lower"] > 0
        and p2["pooled"]["bootstrap_y"]["ci_lower"] > 0
        and p1["positive_best_single_seeds"] >= 3
        and p2["positive_y_seeds"] >= 3
        and p1["fpr_mean"] <= float(gates["amber_fpr_max"])
        and p2["pooled"]["relation_gain_vs_b1_qy"] >= float(gates["p2_relation_gain_min"])
    )
    return {"passed": all(checks.values()), "amber": amber, "checks": checks}


def metrics_for_panel(pred_rows: list[dict], panel_name: str) -> dict:
    metrics = binary_metrics([row["gold_label"] for row in pred_rows], [row["pred_label"] for row in pred_rows], [float(row["pred_score"]) for row in pred_rows])
    if panel_name == "P2-mini":
        metrics["strict_group_consistency"] = group_consistency(pred_rows)
    return metrics


def group_consistency(rows: list[dict]) -> float:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("cluster_id") or row["semantic_component_id"]), []).append(row)
    return sum(1 for members in groups.values() if all(row["gold_label"] == row["pred_label"] for row in members)) / max(len(groups), 1)


def prediction_rows(rows: list[dict], pred_labels: list[str], scores: list[float], threshold_info: dict, level: str, mode: str, seed: int, panel: str) -> list[dict]:
    return [
        {
            "id": row["id"],
            "semantic_component_id": row["semantic_component_id"],
            "cluster_id": row.get("context_collision_group_id") or row["semantic_component_id"],
            "gold_label": row["exp1_label"],
            "pred_label": pred,
            "pred_score": score,
            "threshold": threshold_info["threshold"],
            "threshold_feasible": threshold_info["feasible"],
            "level": level,
            "mode": mode,
            "seed": seed,
            "panel": panel,
            "source": row.get("source"),
        }
        for row, pred, score in zip(rows, pred_labels, scores)
    ]


def write_report(output_dir: Path, summary: dict, rows: list[dict], comparisons: dict, data_audit: dict, diagnostic: bool) -> None:
    lines = [
        "# E1-CPU-v5 G0c Data-Corrected Confirmation 报告",
        "",
        "## 结论",
        "",
        f"- 状态：{summary.get('status', 'DATA_GATE_FAIL')}",
        f"- 数据 Gate：{'PASS' if data_audit.get('passed') else 'FAIL'}",
    ]
    if diagnostic:
        lines.append("- DIAGNOSTIC / NOT ELIGIBLE FOR GATE：本次使用了 --allow_diagnostic，仅作诊断，不可作为正式验收。")
    if "selected" in summary:
        lines.extend(
            [
                f"- BestSingle：{summary['selected']['best_single']['level']} / {summary['selected']['best_single']['mode']}",
                f"- BestQY：{summary['selected']['best_qy']['level']} / {summary['selected']['best_qy']['mode']}",
                "",
                "## Panel 汇总",
                "",
                "| panel | pooled q+y Macro-F1 | ΔBestSingle | ΔY | CI lower | positive seeds | recall | FPR | consistency | relation gain |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for panel, item in summary["panels"].items():
            pooled = item["pooled"]
            ci = pooled["bootstrap_y" if panel == "P2-mini" else "bootstrap_best_single"]["ci_lower"]
            lines.append(
                f"| {panel} | {pooled['macro_f1_qy']:.4f} | {pooled['delta_best_single']:.4f} | {pooled['delta_y']:.4f} | "
                f"{ci:.4f} | {item['positive_best_single_seeds']}/3 | {item['unsafe_recall_mean']:.4f} | {item['fpr_mean']:.4f} | "
                f"{'' if item['strict_group_consistency_mean'] is None else f'{item['strict_group_consistency_mean']:.4f}'} | {pooled['relation_gain_vs_b1_qy']:.4f} |"
            )
        lines.extend(["", "## Metric Gate", ""])
        for key, value in summary["g0c_gate"]["checks"].items():
            lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", "## Data Gate", ""])
    for key, value in data_audit.get("gate", {}).items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", "## 数据审计摘要", ""])
    lines.append(json.dumps({"counts": data_audit.get("counts"), "fraud_primary_rates": data_audit.get("fraud_primary_rates"), "p2_audit": data_audit.get("p2_audit")}, ensure_ascii=False, indent=2))
    (output_dir / "reports" / "E1_CPU_v5_G0c_报告_中文.md").write_text("\n".join(lines), encoding="utf-8")


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
