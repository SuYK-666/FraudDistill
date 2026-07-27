from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import joblib
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
from frauddistill.exp1_ccfa.frozen_semantic_cpu import FrozenSemanticCPUDetector
from frauddistill.exp1_ccfa.pair_cross_encoder import labels_from_scores
from frauddistill.exp1_ccfa.pairlite_cpu import PairLiteCPUDetector
from frauddistill.exp1_ccfa.paired_cluster_bootstrap import paired_cluster_bootstrap_delta
from frauddistill.exp1_ccfa.resource_profile import ResourceProfiler, resource_gate
from frauddistill.exp1_ccfa.stat_tests import holm_adjust
from frauddistill.utils.io import read_jsonl, write_jsonl
from scripts.run_exp1_cpu_g0c import select_constrained_threshold


CONFIG_PATH = ROOT / "configs" / "experiments" / "exp1_ccfa_cpu_v5.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 CPU G0c2")
    parser.add_argument("--manifest_dir", default="data/prepared/exp1_cpu_v5/g0c2")
    parser.add_argument("--output_dir", default="outputs/exp1_ccfa_cpu_v5/g0c2")
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260724, 20260725, 20260726])
    parser.add_argument("--bootstrap_iterations", type=int, default=5000)
    parser.add_argument("--run_p1_even_if_p2_fails", action="store_true")
    parser.add_argument("--one_model_fix_consumed", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    summary = run(ROOT / args.manifest_dir, ROOT / args.output_dir, args.seeds, args.bootstrap_iterations, args.quick, args.one_model_fix_consumed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["decision"] == "STOP_OR_NARROW_CLAIM":
        raise SystemExit(2)


def run(manifest_dir: Path, output_dir: Path, seeds: list[int], bootstrap_iterations: int, quick: bool = False, one_model_fix_consumed: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("models", "predictions", "reports", "tables"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    profiler = ResourceProfiler(output_dir)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    audit = json.loads((manifest_dir / "G0c2_DATA_AUDIT.json").read_text(encoding="utf-8"))
    if not audit["base_data_gate"]["passed"]:
        profile = resource_gate(profiler.finish(), config["resource_gates"])
        return write_decision(output_dir, audit, None, None, None, profile, "STOP_OR_NARROW_CLAIM")
    train = rows(manifest_dir / "g0_train.jsonl")
    model_dev = rows(manifest_dir / "g0_model_dev.jsonl")
    threshold_dev = rows(manifest_dir / "g0_threshold_dev.jsonl")
    p1 = rows(manifest_dir / "g0_p1_mini.jsonl")
    p2 = rows(manifest_dir / "g0_p2_dvm_300.jsonl")

    selected = select_modeldev(train, model_dev, config, seeds, output_dir, quick)
    write_table(output_dir / "G0c2_MODELDEV_METRICS_BY_SEED.csv", selected["modeldev_rows"])
    modeldev_gate = modeldev_gate_status(selected, config)
    write_json(output_dir / "G0c2_SELECTED_MODELS.json", selected_summary(selected))
    panels = [("P1", p1)]
    if modeldev_gate["passed"] and audit["p2_data_gate"]["passed"] and p2:
        panels.append(("P2-DVM", p2))
    predictions: dict[tuple[str, int, str], list[dict]] = {}
    metric_rows = []
    threshold_rows = []
    resource_rows = []
    for seed in tqdm(seeds, desc="G0c2 formal seeds"):
        for key, spec in selected["comparators"].items():
            model = fit_spec(train, spec, config, seed, output_dir)
            model_path = output_dir / "models" / f"{key}_seed{seed}.joblib"
            joblib.dump({"model": model, "spec": spec, "seed": seed}, model_path)
            threshold_scores = model.predict_proba(threshold_dev, spec["mode"]).tolist()
            threshold = select_constrained_threshold([row["exp1_label"] for row in threshold_dev], threshold_scores, config["threshold_policy"])
            threshold_rows.append({"comparator": key, "seed": seed, **threshold})
            for panel_name, panel_rows in panels:
                scores = model.predict_proba(panel_rows, spec["mode"]).tolist()
                pred = labels_from_scores(scores, threshold["threshold"])
                pred_rows = prediction_rows(panel_rows, pred, scores, threshold, spec, seed, panel_name)
                predictions[(key, seed, panel_name)] = pred_rows
                write_jsonl(output_dir / "predictions" / f"{panel_name}_{key}_seed{seed}.jsonl", pred_rows)
                metric_rows.append({"panel": panel_name, "comparator": key, "seed": seed, **metrics(pred_rows, panel_name)})
            if getattr(model, "profile", None):
                resource_rows.append({"comparator": key, "seed": seed, **model.profile.__dict__})
            profiler.sample()
    p1_rows = [row for row in metric_rows if row["panel"] == "P1"]
    p2_rows = [row for row in metric_rows if row["panel"] == "P2-DVM"]
    write_table(output_dir / "G0c2_P1_METRICS_BY_SEED.csv", p1_rows)
    write_table(output_dir / "G0c2_P2_METRICS_BY_SEED.csv", p2_rows)
    write_table(output_dir / "G0c2_THRESHOLDS.csv", threshold_rows)
    write_json(output_dir / "G0c2_THRESHOLDS.json", threshold_rows)
    write_table(output_dir / "G0c2_RESOURCE_PROFILE.csv", resource_rows)
    stats = paired_stats(predictions, selected, seeds, bootstrap_iterations)
    write_json(output_dir / "G0c2_PAIRED_STATS.json", stats)
    p1_gate = p1_gate_status(stats.get("P1", {}), p1_rows, selected, seeds)
    p2_gate = p2_gate_status(stats.get("P2-DVM"), p2_rows, selected, seeds) if p2_rows else {"status": "NOT_RUN", "checks": {}}
    profile = resource_gate(profiler.finish(), config["resource_gates"])
    return write_decision(output_dir, audit, modeldev_gate, p1_gate, p2_gate, profile, decide(modeldev_gate, p1_gate, audit, p2_gate, profile, one_model_fix_consumed))


def select_modeldev(train: list[dict], model_dev: list[dict], config: dict, seeds: list[int], output_dir: Path, quick: bool) -> dict:
    comparators = {
        "B1_q_only": {"backend": "pairlite", "level": "B1", "mode": "q_only", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 1.0}},
        "B1_y_only": {"backend": "pairlite", "level": "B1", "mode": "y_only", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 1.0}},
        "B1_q_y": {"backend": "pairlite", "level": "B1", "mode": "q_y", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 1.0}},
        "R1_q_y": {"backend": "pairlite", "level": "R1", "mode": "q_y", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 0.5}},
    }
    c_values = [1.0] if quick else [float(x) for x in config["semantic_cpu"]["classifier"]["c_grid"]]
    relation_weights = [1.0] if quick else [float(x) for x in config["semantic_cpu"]["classifier"].get("relation_weight_grid", [1.0])]
    for c in c_values:
        for level, mode in (("S0", "q_only"), ("S0", "y_only"), ("S0", "q_y"), ("S1", "q_y")):
            weights = relation_weights if level == "S1" else [1.0]
            for relation_weight in weights:
                suffix = f"C{c:g}" if relation_weight == 1.0 else f"C{c:g}_RW{relation_weight:g}"
                comparators[f"{level}_{mode}_{suffix}"] = {"backend": "semantic", "level": level, "mode": mode, "config": {"c": c, "relation_weight": relation_weight}}
    rows_out = []
    for key, spec in tqdm(comparators.items(), desc="G0c2 model-dev"):
        for seed in seeds:
            model = fit_spec(train, spec, config, seed, output_dir)
            scores = model.predict_proba(model_dev, spec["mode"]).tolist()
            threshold = select_constrained_threshold([row["exp1_label"] for row in model_dev], scores, config["threshold_policy"])
            pred = labels_from_scores(scores, threshold["threshold"])
            rows_out.append({"comparator": key, "seed": seed, "threshold": threshold["threshold"], "threshold_feasible": threshold["feasible"], **binary_metrics([row["exp1_label"] for row in model_dev], pred, scores)})
    best_s0_qy = best_by(rows_out, "S0_q_y")
    best_s1_qy = best_by(rows_out, "S1_q_y")
    best_single = best_single_key(rows_out)
    frozen = {key: spec for key, spec in comparators.items() if key in {"B1_q_only", "B1_y_only", "B1_q_y", "R1_q_y", best_s0_qy, best_s1_qy, best_single}}
    return {"modeldev_rows": rows_out, "comparators": frozen, "best_s0_qy": best_s0_qy, "best_s1_qy": best_s1_qy, "best_single": best_single}


def fit_spec(train: list[dict], spec: dict, config: dict, seed: int, output_dir: Path):
    if spec["backend"] == "pairlite":
        cfg = spec["config"]
        return PairLiteCPUDetector(level=spec["level"], alpha=cfg["alpha"], l1_ratio=cfg["l1_ratio"], max_iter=cfg["max_iter"], seed=seed, char_weight=cfg["char_weight"], cross_weight=cfg["cross_weight"]).fit(train, [row["exp1_label"] for row in train], spec["mode"])
    encoder = dict(config["semantic_cpu"]["encoder"])
    encoder["batch_size"] = 128
    return FrozenSemanticCPUDetector(spec["level"], encoder, str(output_dir / "embedding_cache"), c=float(spec["config"]["c"]), relation_weight=float(spec["config"].get("relation_weight", 1.0)), seed=seed).fit(train, [row["exp1_label"] for row in train], spec["mode"])


def modeldev_gate_status(selected: dict, config: dict) -> dict:
    rows_out = selected["modeldev_rows"]
    gate = config["semantic_cpu"]["modeldev_gate"]
    s1_rows = by_comparator(rows_out, selected["best_s1_qy"])
    s0_rows = by_comparator(rows_out, selected["best_s0_qy"])
    single_rows = by_comparator(rows_out, selected["best_single"])
    checks = {
        "s1_macro_f1": mean([r["macro_f1"] for r in s1_rows]) >= gate["s1_qy_macro_f1_min"],
        "s1_delta_best_single": mean([r["macro_f1"] for r in s1_rows]) - mean([r["macro_f1"] for r in single_rows]) >= gate["s1_delta_best_single_min"],
        "s1_delta_s0_qy": mean([r["macro_f1"] for r in s1_rows]) - mean([r["macro_f1"] for r in s0_rows]) >= gate["s1_delta_s0_qy_min"],
        "unsafe_recall": mean([r["recall"] for r in s1_rows]) >= gate["unsafe_recall_min"],
        "fpr": mean([r["fpr"] for r in s1_rows]) <= gate["fpr_max"],
        "positive_seeds": sum(1 for a, b in zip(s1_rows, single_rows) if a["macro_f1"] > b["macro_f1"]) >= gate["positive_seeds_min"],
        "threshold_feasible": all(r["threshold_feasible"] for r in s1_rows),
    }
    return {"passed": all(checks.values()), "checks": checks, "best_s1_qy": selected["best_s1_qy"], "best_s0_qy": selected["best_s0_qy"], "best_single": selected["best_single"]}


def paired_stats(predictions: dict, selected: dict, seeds: list[int], iterations: int) -> dict:
    result = {}
    for panel in sorted({panel for _, _, panel in predictions}):
        s1_key = selected["best_s1_qy"]
        s0_key = selected["best_s0_qy"]
        single_key = selected["best_single"]
        y_key = "B1_y_only" if ("B1_y_only", seeds[0], panel) in predictions else next(key for key, _, p in predictions if "y_only" in key and p == panel)
        gold = [row["gold_label"] for row in predictions[(s1_key, seeds[0], panel)]]
        clusters = [row["cluster_id"] for row in predictions[(s1_key, seeds[0], panel)]]
        metric_fn = lambda yt, pred: float(f1_score(yt, pred, average="macro", zero_division=0))
        s1 = pooled_labels(predictions, s1_key, seeds, panel)
        s0 = pooled_labels(predictions, s0_key, seeds, panel)
        single = pooled_labels(predictions, single_key, seeds, panel)
        y = pooled_labels(predictions, y_key, seeds, panel)
        pvals = {"s1_vs_best_single": exact_mcnemar(gold, single, s1)["p_value"], "s1_vs_s0": exact_mcnemar(gold, s0, s1)["p_value"], "s1_vs_y": exact_mcnemar(gold, y, s1)["p_value"]}
        result[panel] = {
            "best_single_key": single_key,
            "s0_key": s0_key,
            "s1_key": s1_key,
            "y_key": y_key,
            "macro_f1_s1": metric_fn(gold, s1),
            "delta_best_single": metric_fn(gold, s1) - metric_fn(gold, single),
            "delta_s0_qy": metric_fn(gold, s1) - metric_fn(gold, s0),
            "delta_y": metric_fn(gold, s1) - metric_fn(gold, y),
            "bootstrap_single": paired_cluster_bootstrap_delta(gold, single, s1, clusters, metric_fn, iterations=iterations, seed=20260727),
            "bootstrap_y": paired_cluster_bootstrap_delta(gold, y, s1, clusters, metric_fn, iterations=iterations, seed=20260728),
            "mcnemar": pvals,
            "holm": holm_adjust(pvals),
        }
    return result


def p1_gate_status(stats: dict, rows_out: list[dict], selected: dict, seeds: list[int]) -> dict:
    if not stats:
        return {"status": "FAIL", "checks": {"present": False}}
    s1_rows = [row for row in rows_out if row["comparator"] == selected["best_s1_qy"]]
    checks = {
        "p1_macro": stats["macro_f1_s1"] >= 0.78,
        "p1_delta_single": stats["delta_best_single"] >= 0.03,
        "p1_delta_s0": stats["delta_s0_qy"] >= 0.015,
        "p1_ci": stats["bootstrap_single"]["ci_lower"] > 0,
        "unsafe_recall": mean([row["recall"] for row in s1_rows]) >= 0.72,
        "fpr": mean([row["fpr"] for row in s1_rows]) <= 0.25,
        "positive_seeds": sum(1 for row in s1_rows if row["macro_f1"] > 0) == len(seeds),
        "worst_seed": min([row["macro_f1"] for row in s1_rows], default=0.0) >= 0.75,
    }
    if all(checks.values()):
        status = "PASS"
    elif stats["macro_f1_s1"] >= 0.76 and stats["delta_best_single"] > 0:
        status = "AMBER"
    else:
        status = "FAIL"
    return {"status": status, "checks": checks, "stats": stats}


def p2_gate_status(stats: dict | None, rows_out: list[dict], selected: dict, seeds: list[int]) -> dict:
    if not stats:
        return {"status": "NOT_RUN", "checks": {}}
    s1_rows = [row for row in rows_out if row["comparator"] == selected["best_s1_qy"]]
    checks = {
        "p2_macro": stats["macro_f1_s1"] >= 0.80,
        "p2_delta_single": stats["delta_best_single"] >= 0.10,
        "p2_delta_y": stats["delta_y"] >= 0.10,
        "p2_delta_s0": stats["delta_s0_qy"] >= 0.02,
        "p2_ci_y": stats["bootstrap_y"]["ci_lower"] > 0,
        "worst_seed": min([row["macro_f1"] for row in s1_rows], default=0.0) >= 0.76,
        "positive_seeds": len([row for row in s1_rows if row["macro_f1"] > 0]) == len(seeds),
    }
    status = "PASS" if all(checks.values()) else "AMBER" if stats["macro_f1_s1"] >= 0.76 and stats["delta_y"] > 0 else "FAIL"
    return {"status": status, "checks": checks, "stats": stats}


def decide(modeldev_gate: dict, p1_gate: dict, data_audit: dict, p2_gate: dict, resources: dict, one_model_fix_consumed: bool) -> str:
    if not modeldev_gate.get("passed") or not resources.get("passed"):
        return "STOP_OR_NARROW_CLAIM" if one_model_fix_consumed else "ONE_MODEL_FIX"
    if p1_gate.get("status") in {"PASS", "AMBER"} and data_audit["p2_data_gate"]["passed"] and p2_gate.get("status") in {"PASS", "AMBER"}:
        return "RUN_24K_BRIDGE"
    if p1_gate.get("status") == "FAIL" or not data_audit["p2_data_gate"]["passed"] or p2_gate.get("status") == "FAIL":
        return "STOP_OR_NARROW_CLAIM"
    return "ONE_MODEL_FIX"


def write_decision(output_dir: Path, data_audit: dict, modeldev_gate: dict | None, p1_gate: dict | None, p2_gate: dict | None, resource: dict, decision: str) -> dict:
    payload = {
        "base_data_gate": "PASS" if data_audit["base_data_gate"]["passed"] else "FAIL",
        "modeldev_gate": "PASS" if modeldev_gate and modeldev_gate.get("passed") else "FAIL",
        "p1_gate": (p1_gate or {}).get("status", "NOT_RUN"),
        "p2_data_gate": "PASS" if data_audit["p2_data_gate"]["passed"] else "FAIL",
        "p2_model_gate": (p2_gate or {}).get("status", "NOT_RUN"),
        "resource_gate": "PASS" if resource.get("passed") else "FAIL",
        "decision": decision,
        "failed_checks": failed_checks(data_audit, modeldev_gate, p1_gate, p2_gate, resource),
        "git_commit": git_commit(),
        "data_fingerprint": data_audit.get("data_fingerprint", ""),
        "encoder_revision": yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["semantic_cpu"]["encoder"]["revision"],
    }
    write_json(output_dir / "G0c2_DECISION.json", payload)
    write_json(output_dir / "G0c2_RUN_FINGERPRINT.json", {"git_commit": payload["git_commit"], "encoder_revision": payload["encoder_revision"], "created_by": "run_exp1_cpu_g0c2.py"})
    write_json(output_dir / "G0c2_RESOURCE_GATE.json", resource)
    write_report(output_dir, payload, data_audit, modeldev_gate, p1_gate, p2_gate, resource)
    return payload


def write_report(output_dir: Path, decision: dict, data_audit: dict, modeldev_gate: dict | None, p1_gate: dict | None, p2_gate: dict | None, resource: dict) -> None:
    lines = [
        "# E1_CPU_v5_G0c2_整体任务报告_中文",
        "",
        "## 执行结论",
        f"- 最终决策：{decision['decision']}",
        f"- base data Gate：{decision['base_data_gate']}",
        f"- model-dev Gate：{decision['modeldev_gate']}",
        f"- P1 Gate：{decision['p1_gate']}",
        f"- P2 data Gate：{decision['p2_data_gate']}",
        f"- P2 model Gate：{decision['p2_model_gate']}",
        f"- resource Gate：{decision['resource_gate']}",
        "",
        "## 失败项",
    ]
    lines.extend([f"- {item}" for item in decision["failed_checks"]] or ["- 无"])
    lines.extend(["", "## Table A：数据漏斗与 Gate", "```json", json.dumps({"base": data_audit.get("base_data_gate"), "p2": data_audit.get("p2_data_gate"), "counts": data_audit.get("counts")}, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## Table B：Model-dev", "详见 `G0c2_MODELDEV_METRICS_BY_SEED.csv`。", "```json", json.dumps(modeldev_gate or {}, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## Table C：P1/P2 主结果", "详见 `G0c2_P1_METRICS_BY_SEED.csv`、`G0c2_P2_METRICS_BY_SEED.csv` 和 `G0c2_PAIRED_STATS.json`。", "```json", json.dumps({"p1": p1_gate, "p2": p2_gate}, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## Table D：P2 balance", "```json", json.dumps(data_audit.get("p2_dvm_audit", {}), ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## Table E：资源", "```json", json.dumps(resource, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## 分析", "本轮严格将 P2 数据 Gate 与 P1/R3 模型评估解耦。P2-DVM 的构建只使用 q-only/y-only nuisance selector、长度和拒答等单视图变量，不读取任何 q+y/S1 预测；P1 作为自然 fraud-core 面板独立给出三种子结果。若 P2-DVM 未通过，结论应缩小为自然面板证据和 P2 构建诊断，而不应继续调测试集。"])
    (output_dir / "reports" / "E1_CPU_v5_G0c2_整体任务报告_中文.md").write_text("\n".join(lines), encoding="utf-8")


def prediction_rows(rows, pred, scores, threshold, spec, seed, panel):
    return [{"id": row["id"], "semantic_component_id": row["semantic_component_id"], "cluster_id": row.get("context_collision_group_id") or row["semantic_component_id"], "gold_label": row["exp1_label"], "pred_label": label, "pred_score": score, "threshold": threshold["threshold"], "threshold_feasible": threshold["feasible"], "level": spec["level"], "mode": spec["mode"], "backend": spec["backend"], "seed": seed, "panel": panel} for row, label, score in zip(rows, pred, scores)]


def metrics(pred_rows, panel):
    out = binary_metrics([row["gold_label"] for row in pred_rows], [row["pred_label"] for row in pred_rows], [row["pred_score"] for row in pred_rows])
    if panel == "P2-DVM":
        groups: dict[str, list[dict]] = {}
        for row in pred_rows:
            groups.setdefault(row["cluster_id"], []).append(row)
        out["strict_group_consistency"] = sum(1 for members in groups.values() if all(row["gold_label"] == row["pred_label"] for row in members)) / max(len(groups), 1)
    return out


def pooled_labels(predictions, key, seeds, panel):
    scores = np.mean(np.asarray([[row["pred_score"] for row in predictions[(key, seed, panel)]] for seed in seeds], dtype=float), axis=0)
    threshold = np.mean([predictions[(key, seed, panel)][0]["threshold"] for seed in seeds])
    return labels_from_scores(scores.tolist(), float(threshold))


def best_by(rows_out, prefix):
    candidates = [row for row in rows_out if row["comparator"].startswith(prefix)]
    means = {key: mean([row["macro_f1"] for row in candidates if row["comparator"] == key]) for key in {row["comparator"] for row in candidates}}
    return max(means, key=means.get)


def best_single_key(rows_out):
    candidates = [row for row in rows_out if "q_only" in row["comparator"] or "y_only" in row["comparator"]]
    means = {key: mean([row["macro_f1"] for row in candidates if row["comparator"] == key]) for key in {row["comparator"] for row in candidates}}
    return max(means, key=means.get)


def by_comparator(rows_out, key):
    return [row for row in rows_out if row["comparator"] == key]


def selected_summary(selected: dict) -> dict:
    return {key: selected[key] for key in ("best_s1_qy", "best_s0_qy", "best_single")}


def failed_checks(data_audit, modeldev_gate, p1_gate, p2_gate, resource):
    out = []
    for prefix, gate in (("base", data_audit.get("base_data_gate", {})), ("p2_data", data_audit.get("p2_data_gate", {})), ("modeldev", modeldev_gate or {}), ("p1", p1_gate or {}), ("p2_model", p2_gate or {}), ("resource", resource or {})):
        for key, value in gate.get("checks", {}).items():
            if value is False:
                out.append(f"{prefix}:{key}")
    return out


def rows(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_table(path: Path, rows_out: list[dict]) -> None:
    if not rows_out:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows_out for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)


def mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
