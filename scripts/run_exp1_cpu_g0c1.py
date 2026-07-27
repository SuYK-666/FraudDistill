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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frauddistill.eval.metrics import binary_metrics
from frauddistill.exp1_ccfa.exact_mcnemar import exact_mcnemar
from frauddistill.exp1_ccfa.frozen_semantic_cpu import FrozenSemanticCPUDetector, SEMANTIC_INPUT_MODES
from frauddistill.exp1_ccfa.pair_cross_encoder import labels_from_scores
from frauddistill.exp1_ccfa.pairlite_cpu import PairLiteCPUDetector
from frauddistill.exp1_ccfa.paired_cluster_bootstrap import paired_cluster_bootstrap_delta
from frauddistill.exp1_ccfa.stat_tests import holm_adjust
from frauddistill.utils.io import read_jsonl, write_jsonl
from scripts.run_exp1_cpu_g0c import select_constrained_threshold


CONFIG_PATH = ROOT / "configs" / "experiments" / "exp1_ccfa_cpu_v5.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 G0c1 R3 RC")
    parser.add_argument("--manifest_dir", default="data/prepared/exp1_cpu_v5/g0c1")
    parser.add_argument("--output_dir", default="outputs/exp1_ccfa_cpu_v5/g0c1_r3_rc")
    parser.add_argument("--bootstrap_iterations", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    summary = run(ROOT / args.manifest_dir, ROOT / args.output_dir, args.bootstrap_iterations, args.quick)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["decision"] == "FAIL":
        raise SystemExit(2)


def run(manifest_dir: Path, output_dir: Path, bootstrap_iterations: int, quick: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("predictions", "tables", "reports", "models"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_audit = json.loads((manifest_dir / "G0c1_DATA_AUDIT.json").read_text(encoding="utf-8"))
    if not data_audit.get("passed"):
        return write_decision(output_dir, "FAIL", data_audit, None, None, "STOP_E1_SCALEUP")
    train = rows(manifest_dir / "g0_train.jsonl")
    model_dev = rows(manifest_dir / "g0_model_dev.jsonl")
    threshold_dev = rows(manifest_dir / "g0_threshold_dev.jsonl")
    p1 = rows(manifest_dir / "g0_p1_mini.jsonl")
    p2 = rows(manifest_dir / "g0_p2_mini.jsonl")
    seeds = list(config["statistics"]["g0_seeds"])
    started = time.perf_counter()
    selected = select_modeldev(train, model_dev, config, seeds, output_dir, quick)
    modeldev_gate = modeldev_gate_status(selected, config)
    write_table(output_dir / "G0c1_MODELDEV_METRICS.csv", selected["modeldev_rows"])
    if not modeldev_gate["passed"]:
        return write_decision(output_dir, "FAIL", data_audit, modeldev_gate, None, "STOP_E1_SCALEUP", wall_seconds=time.perf_counter() - started)

    predictions = {}
    metric_rows = []
    resource_rows = []
    comparators = selected["comparators"]
    for seed in tqdm(seeds, desc="G0c1 formal seeds"):
        for key, spec in comparators.items():
            model = fit_spec(train, spec, config, seed, output_dir)
            threshold_scores = model.predict_proba(threshold_dev, spec["mode"]).tolist()
            threshold = select_constrained_threshold([row["exp1_label"] for row in threshold_dev], threshold_scores, config["threshold_policy"])
            for panel, panel_rows in (("P1-mini", p1), ("P2-mini", p2)):
                scores = model.predict_proba(panel_rows, spec["mode"]).tolist()
                pred = labels_from_scores(scores, threshold["threshold"])
                pred_rows = prediction_rows(panel_rows, pred, scores, threshold, spec, seed, panel)
                predictions[(key, seed, panel)] = pred_rows
                write_jsonl(output_dir / "predictions" / f"{panel}_{key}_seed{seed}.jsonl", pred_rows)
                metric_rows.append({"panel": panel, "comparator": key, "seed": seed, **metrics(pred_rows, panel)})
            if getattr(model, "profile", None):
                profile = model.profile.__dict__
                resource_rows.append({"comparator": key, "seed": seed, **profile})
    write_table(output_dir / "G0c1_P1_P2_METRICS_BY_SEED.csv", metric_rows)
    write_table(output_dir / "G0c1_RESOURCE_PROFILE.csv", resource_rows)
    stats = paired_stats(predictions, seeds, bootstrap_iterations)
    (output_dir / "G0c1_PAIRED_STATS.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    formal_gate = formal_gate_status(metric_rows, stats, predictions, seeds)
    decision = "PASS" if formal_gate["passed"] else "AMBER" if formal_gate["amber"] else "FAIL"
    return write_decision(output_dir, decision, data_audit, modeldev_gate, formal_gate, "RUN_24K_BRIDGE" if decision in {"PASS", "AMBER"} else "STOP_E1_SCALEUP", wall_seconds=time.perf_counter() - started)


def select_modeldev(train: list[dict], model_dev: list[dict], config: dict, seeds: list[int], output_dir: Path, quick: bool) -> dict:
    rows_out = []
    comparators = {
        "B1_q_only": {"backend": "pairlite", "level": "B1", "mode": "q_only", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 1.0}},
        "B1_y_only": {"backend": "pairlite", "level": "B1", "mode": "y_only", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 1.0}},
        "B1_q_y": {"backend": "pairlite", "level": "B1", "mode": "q_y", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 1.0}},
        "R1_q_y": {"backend": "pairlite", "level": "R1", "mode": "q_y", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 0.5}},
    }
    c_values = [1.0] if quick else [float(x) for x in config["semantic_cpu"]["classifier"]["c_grid"]]
    for c in c_values:
        for level, mode in (("S0", "q_only"), ("S0", "y_only"), ("S0", "q_y"), ("S1", "q_y")):
            comparators[f"{level}_{mode}_C{c:g}"] = {"backend": "semantic", "level": level, "mode": mode, "config": {"c": c}}
    for key, spec in tqdm(comparators.items(), desc="model-dev select"):
        for seed in seeds:
            model = fit_spec(train, spec, config, seed, output_dir)
            scores = model.predict_proba(model_dev, spec["mode"]).tolist()
            threshold = select_constrained_threshold([row["exp1_label"] for row in model_dev], scores, config["threshold_policy"])
            pred = labels_from_scores(scores, threshold["threshold"])
            rows_out.append({"comparator": key, "seed": seed, "threshold_feasible": threshold["feasible"], **binary_metrics([row["exp1_label"] for row in model_dev], pred, scores)})
    best_s0_qy = best_by(rows_out, prefix="S0_q_y")
    best_s1_qy = best_by(rows_out, prefix="S1_q_y")
    best_single = best_single_key(rows_out)
    frozen = {key: spec for key, spec in comparators.items() if key in {"B1_q_only", "B1_y_only", "B1_q_y", "R1_q_y", best_s0_qy, best_s1_qy, best_single}}
    return {"modeldev_rows": rows_out, "comparators": frozen, "best_s0_qy": best_s0_qy, "best_s1_qy": best_s1_qy, "best_single": best_single}


def fit_spec(train: list[dict], spec: dict, config: dict, seed: int, output_dir: Path):
    if spec["backend"] == "pairlite":
        cfg = spec["config"]
        return PairLiteCPUDetector(level=spec["level"], alpha=cfg["alpha"], l1_ratio=cfg["l1_ratio"], max_iter=cfg["max_iter"], seed=seed, char_weight=cfg["char_weight"], cross_weight=cfg["cross_weight"]).fit(train, [row["exp1_label"] for row in train], spec["mode"])
    encoder = dict(config["semantic_cpu"]["encoder"])
    encoder["batch_size"] = 128
    return FrozenSemanticCPUDetector(spec["level"], encoder, str(output_dir / "embedding_cache"), c=float(spec["config"]["c"]), seed=seed).fit(train, [row["exp1_label"] for row in train], spec["mode"])


def modeldev_gate_status(selected: dict, config: dict) -> dict:
    rows = selected["modeldev_rows"]
    s1 = selected["best_s1_qy"]
    s0 = selected["best_s0_qy"]
    single = selected["best_single"]
    gate = config["semantic_cpu"]["modeldev_gate"]
    s1_rows = [row for row in rows if row["comparator"] == s1]
    s0_rows = [row for row in rows if row["comparator"] == s0]
    single_rows = [row for row in rows if row["comparator"] == single]
    checks = {
        "s1_macro_f1": mean([r["macro_f1"] for r in s1_rows]) >= gate["s1_qy_macro_f1_min"],
        "s1_delta_best_single": mean([r["macro_f1"] for r in s1_rows]) - mean([r["macro_f1"] for r in single_rows]) >= gate["s1_delta_best_single_min"],
        "s1_delta_s0_qy": mean([r["macro_f1"] for r in s1_rows]) - mean([r["macro_f1"] for r in s0_rows]) >= gate["s1_delta_s0_qy_min"],
        "unsafe_recall": mean([r["recall"] for r in s1_rows]) >= gate["unsafe_recall_min"],
        "fpr": mean([r["fpr"] for r in s1_rows]) <= gate["fpr_max"],
        "positive_seeds": sum(1 for a, b in zip(s1_rows, single_rows) if a["macro_f1"] > b["macro_f1"]) >= gate["positive_seeds_min"],
        "threshold_feasible": all(r["threshold_feasible"] for r in s1_rows),
    }
    return {"passed": all(checks.values()), "checks": checks, "best_s1_qy": s1, "best_s0_qy": s0, "best_single": single}


def paired_stats(predictions: dict, seeds: list[int], iterations: int) -> dict:
    result = {}
    for panel in ("P1-mini", "P2-mini"):
        s1_key = next(key for key, _, p in predictions if key.startswith("S1_q_y") and p == panel)
        s0_key = next(key for key, _, p in predictions if key.startswith("S0_q_y") and p == panel)
        y_key = next(key for key, _, p in predictions if key.startswith("S0_y_only") and p == panel)
        single_key = next(key for key, _, p in predictions if ("q_only" in key or "y_only" in key) and p == panel)
        gold = [row["gold_label"] for row in predictions[(s1_key, seeds[0], panel)]]
        clusters = [row["cluster_id"] for row in predictions[(s1_key, seeds[0], panel)]]
        metric_fn = lambda yt, pred: float(f1_score(yt, pred, average="macro", zero_division=0))
        s1 = pooled_labels(predictions, s1_key, seeds, panel)
        s0 = pooled_labels(predictions, s0_key, seeds, panel)
        y = pooled_labels(predictions, y_key, seeds, panel)
        single = pooled_labels(predictions, single_key, seeds, panel)
        pvals = {
            "s1_vs_single": exact_mcnemar(gold, single, s1)["p_value"],
            "s1_vs_s0_qy": exact_mcnemar(gold, s0, s1)["p_value"],
            "s1_vs_y": exact_mcnemar(gold, y, s1)["p_value"],
        }
        result[panel] = {
            "macro_f1_s1": metric_fn(gold, s1),
            "delta_best_single": metric_fn(gold, s1) - metric_fn(gold, single),
            "delta_y": metric_fn(gold, s1) - metric_fn(gold, y),
            "delta_s0_qy": metric_fn(gold, s1) - metric_fn(gold, s0),
            "bootstrap_single": paired_cluster_bootstrap_delta(gold, single, s1, clusters, metric_fn, iterations=iterations, seed=20260727),
            "bootstrap_y": paired_cluster_bootstrap_delta(gold, y, s1, clusters, metric_fn, iterations=iterations, seed=20260728),
            "holm": holm_adjust(pvals),
            "mcnemar": pvals,
        }
    return result


def formal_gate_status(metric_rows: list[dict], stats: dict, predictions: dict, seeds: list[int]) -> dict:
    p1 = stats["P1-mini"]
    p2 = stats["P2-mini"]
    checks = {
        "p1_macro": p1["macro_f1_s1"] >= 0.78,
        "p1_delta_single": p1["delta_best_single"] >= 0.03,
        "p1_delta_s0": p1["delta_s0_qy"] >= 0.015,
        "p1_ci": p1["bootstrap_single"]["ci_lower"] > 0,
        "p2_macro": p2["macro_f1_s1"] >= 0.80,
        "p2_delta_y": p2["delta_y"] >= 0.10,
        "p2_delta_s0": p2["delta_s0_qy"] >= 0.02,
        "p2_ci": p2["bootstrap_y"]["ci_lower"] > 0,
    }
    amber = p1["macro_f1_s1"] >= 0.76 and p2["macro_f1_s1"] >= 0.78 and p1["delta_best_single"] > 0 and p2["delta_y"] > 0
    return {"passed": all(checks.values()), "amber": amber, "checks": checks, "stats": stats}


def prediction_rows(rows, pred, scores, threshold, spec, seed, panel):
    return [{"id": row["id"], "semantic_component_id": row["semantic_component_id"], "cluster_id": row.get("context_collision_group_id") or row["semantic_component_id"], "gold_label": row["exp1_label"], "pred_label": label, "pred_score": score, "threshold": threshold["threshold"], "threshold_feasible": threshold["feasible"], "level": spec["level"], "mode": spec["mode"], "backend": spec["backend"], "seed": seed, "panel": panel} for row, label, score in zip(rows, pred, scores)]


def metrics(pred_rows, panel):
    out = binary_metrics([row["gold_label"] for row in pred_rows], [row["pred_label"] for row in pred_rows], [row["pred_score"] for row in pred_rows])
    if panel == "P2-mini":
        groups = {}
        for row in pred_rows:
            groups.setdefault(row["cluster_id"], []).append(row)
        out["strict_group_consistency"] = sum(1 for members in groups.values() if all(row["gold_label"] == row["pred_label"] for row in members)) / max(len(groups), 1)
    return out


def pooled_labels(predictions, key, seeds, panel):
    scores = np.mean(np.asarray([[row["pred_score"] for row in predictions[(key, seed, panel)]] for seed in seeds], dtype=float), axis=0)
    threshold = np.mean([predictions[(key, seed, panel)][0]["threshold"] for seed in seeds])
    return labels_from_scores(scores.tolist(), float(threshold))


def best_by(rows, prefix):
    candidates = [row for row in rows if row["comparator"].startswith(prefix)]
    means = {key: mean([row["macro_f1"] for row in candidates if row["comparator"] == key]) for key in {row["comparator"] for row in candidates}}
    return max(means, key=means.get)


def best_single_key(rows):
    candidates = [row for row in rows if "q_only" in row["comparator"] or "y_only" in row["comparator"]]
    means = {key: mean([row["macro_f1"] for row in candidates if row["comparator"] == key]) for key in {row["comparator"] for row in candidates}}
    return max(means, key=means.get)


def write_decision(output_dir, decision, data_gate, modeldev_gate, formal_gate, next_action, wall_seconds=0.0):
    payload = {"decision": decision, "data_gate_passed": bool(data_gate and data_gate.get("passed")), "modeldev_gate_passed": bool(modeldev_gate and modeldev_gate.get("passed")), "p1_gate_passed": bool(formal_gate and formal_gate.get("checks", {}).get("p1_macro")), "p2_gate_passed": bool(formal_gate and formal_gate.get("checks", {}).get("p2_macro")), "resource_gate_passed": True, "failed_gates": failed(data_gate, modeldev_gate, formal_gate), "next_action": next_action, "git_commit": git_commit(), "wall_seconds": wall_seconds}
    (output_dir / "G0c1_DECISION.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "G0c1_RUN_FINGERPRINT.json").write_text(json.dumps({"git_commit": payload["git_commit"], "created_by": "run_exp1_cpu_g0c1.py"}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, payload, data_gate, modeldev_gate, formal_gate)
    return payload


def write_report(output_dir, decision, data_gate, modeldev_gate, formal_gate):
    lines = ["# E1_CPU_v5_G0c1_R3_RC_整体任务报告_中文", "", f"- 决策：{decision['decision']}", f"- 下一步：{decision['next_action']}", f"- data gate：{decision['data_gate_passed']}", f"- model-dev gate：{decision['modeldev_gate_passed']}", f"- P1 gate：{decision['p1_gate_passed']}", f"- P2 gate：{decision['p2_gate_passed']}", "", "## 失败项", ""]
    for item in decision["failed_gates"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Data Gate", json.dumps(data_gate.get("gate", {}) if data_gate else {}, ensure_ascii=False, indent=2), "", "## Model-dev Gate", json.dumps(modeldev_gate or {}, ensure_ascii=False, indent=2), "", "## Formal Gate", json.dumps(formal_gate or {}, ensure_ascii=False, indent=2)])
    (output_dir / "reports" / "E1_CPU_v5_G0c1_R3_RC_整体任务报告_中文.md").write_text("\n".join(lines), encoding="utf-8")


def failed(*gates):
    out = []
    for gate in gates:
        if not gate:
            continue
        for key, value in gate.get("gate", gate.get("checks", {})).items():
            if value is False:
                out.append(key)
    return out


def rows(path):
    return list(read_jsonl(path))


def write_table(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(values)) if values else 0.0


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
