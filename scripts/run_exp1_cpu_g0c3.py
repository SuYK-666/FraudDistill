from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
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


CONFIG_PATH = ROOT / "configs" / "experiments" / "exp1_ccfa_cpu_v5.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 CPU G0c3 final-freeze")
    parser.add_argument("--manifest_dir", default="data/prepared/exp1_cpu_v5/g0c3")
    parser.add_argument("--output_dir", default="outputs/exp1_ccfa_cpu_v5/g0c3")
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260724, 20260725, 20260726])
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow_dirty", action="store_true")
    args = parser.parse_args()
    summary = run(ROOT / args.manifest_dir, ROOT / args.output_dir, args.seeds, args.bootstrap_iterations, args.quick, args.allow_dirty)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["decision"] == "STOP_RELATION_CLAIM":
        raise SystemExit(2)


def run(manifest_dir: Path, output_dir: Path, seeds: list[int], bootstrap_iterations: int, quick: bool = False, allow_dirty: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("models", "predictions", "reports", "tables", "embedding_cache"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    dirty = git_status()
    if dirty and not allow_dirty:
        raise SystemExit("G0c3 formal run requires clean git tree. Commit code first or pass --allow_dirty only for smoke.")
    profiler = ResourceProfiler(output_dir)
    audit = read_json(manifest_dir / "G0c3_DATA_AUDIT.json")
    train = rows(manifest_dir / "g0_train.jsonl")
    model_dev = rows(manifest_dir / "g0_model_dev.jsonl")
    threshold_dev = rows(manifest_dir / "g0_threshold_dev.jsonl")
    p1 = rows(manifest_dir / "g0_p1_natural_mixed.jsonl")
    p2 = rows(manifest_dir / "g0_p2_dvm_core.jsonl")
    p3 = rows(manifest_dir / "p3_v1.jsonl")
    panels = [("P1-Natural-Mixed", p1), ("P3-Public-Gold", p3)]
    if len(p2) >= int(config["data_policy"]["p2_dvm"].get("limited_groups_min", 200)) * 2:
        panels.insert(1, ("P2-DVM-Core", p2))

    selected = select_modeldev(train, model_dev, config, output_dir, quick)
    write_table(output_dir / "G0c3_MODELDEV_METRICS_BY_SEED.csv", selected["modeldev_rows"])
    write_json(output_dir / "G0c3_SELECTED_MODELS.json", selected_summary(selected))

    predictions: dict[tuple[str, str, str], list[dict]] = {}
    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    resource_rows: list[dict] = []
    run_specs = [("FULL_TRAIN_DETERMINISTIC", train)]
    for seed in seeds:
        run_specs.append((str(seed), resample_train_components(train, seed)))

    for train_seed, train_rows in tqdm(run_specs, desc="G0c3 train runs"):
        for key, spec in selected["comparators"].items():
            model = fit_spec(train_rows, spec, config, int(seeds[0]) if train_seed == "FULL_TRAIN_DETERMINISTIC" else int(train_seed), output_dir)
            joblib.dump({"model": model, "spec": spec, "train_seed": train_seed}, output_dir / "models" / f"{key}_seed{train_seed}.joblib")
            threshold_scores = model.predict_proba(threshold_dev, spec["mode"]).tolist()
            threshold = select_source_balanced_threshold(threshold_dev, threshold_scores, config["threshold_policy"])
            threshold_rows.append({"comparator": key, "train_seed": train_seed, **threshold})
            for panel_name, panel_rows in panels:
                scores = model.predict_proba(panel_rows, spec["mode"]).tolist()
                pred = labels_from_scores(scores, threshold["threshold"])
                pred_rows = prediction_rows(panel_rows, pred, scores, threshold, spec, train_seed, panel_name)
                predictions[(key, train_seed, panel_name)] = pred_rows
                write_jsonl(output_dir / "predictions" / f"{panel_name}_{key}_seed{train_seed}.jsonl", pred_rows)
                metric_rows.append({"panel": panel_name, "comparator": key, "train_seed": train_seed, **metrics(pred_rows, panel_name)})
            if getattr(model, "profile", None):
                resource_rows.append({"comparator": key, "train_seed": train_seed, **model.profile.__dict__})
            profiler.sample()
    write_metric_tables(output_dir, metric_rows)
    write_table(output_dir / "G0c3_THRESHOLDS.csv", threshold_rows)
    write_json(output_dir / "G0c3_THRESHOLDS.json", threshold_rows)
    write_table(output_dir / "G0c3_RESOURCE_PROFILE.csv", resource_rows)
    stats = paired_stats(predictions, selected, [name for name, _ in run_specs], bootstrap_iterations)
    write_json(output_dir / "G0c3_PAIRED_STATS.json", stats)
    seed_direction = seed_direction_summary(metric_rows, selected)
    write_json(output_dir / "G0c3_SEED_DIRECTION.json", seed_direction)
    resource = resource_gate(profiler.finish(), config["resource_gates"])
    write_json(output_dir / "G0c3_RESOURCE_GATE.json", resource)
    protocol = protocol_lock(config, audit, manifest_dir, dirty)
    write_json(output_dir / "G0c3_PROTOCOL_LOCK.json", protocol)
    decision = decide(stats, metric_rows, audit, resource, selected, seed_direction)
    payload = {
        "decision": decision,
        "correctness_gate": correctness_gate(audit, resource, dirty),
        "selected": selected_summary(selected),
        "git_commit": git_commit(),
        "git_status_porcelain": dirty,
        "source_tree_sha256": source_tree_sha256(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "data_fingerprint": audit.get("data_fingerprint"),
        "encoder_revision": config["semantic_cpu"]["encoder"]["revision"],
        "dataset_revisions": config["data_policy"].get("dataset_revisions", {}),
        "seed_direction": seed_direction,
    }
    write_json(output_dir / "G0c3_DECISION.json", payload)
    write_json(output_dir / "G0c3_RUN_FINGERPRINT.json", payload)
    write_report(output_dir, payload, audit, stats, metric_rows, resource)
    return payload


def select_modeldev(train: list[dict], model_dev: list[dict], config: dict, output_dir: Path, quick: bool) -> dict:
    comparators = {
        "PairLite_q_only": {"backend": "pairlite", "level": "B1", "mode": "q_only", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 1.0}},
        "PairLite_y_only": {"backend": "pairlite", "level": "B1", "mode": "y_only", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 1.0}},
        "PairLite_q_y": {"backend": "pairlite", "level": "R1", "mode": "q_y", "config": {"alpha": 0.0003, "l1_ratio": 0.0, "max_iter": 40, "char_weight": 1.0, "cross_weight": 0.5}},
        "E_q_only": {"backend": "semantic", "level": "S0", "mode": "q_only", "config": {"c": 1.0}},
        "E_y_only": {"backend": "semantic", "level": "S0", "mode": "y_only", "config": {"c": 1.0}},
        "S0_q_y": {"backend": "semantic", "level": "S0", "mode": "q_y", "config": {"c": 1.0}},
        "S1_q_y": {"backend": "semantic", "level": "S1", "mode": "q_y", "config": {"c": 1.0}},
        "S2_joint_q_y": {"backend": "semantic", "level": "S2", "mode": "q_y", "config": {"c": 1.0}},
    }
    rows_out = []
    keys = list(comparators) if not quick else ["E_q_only", "E_y_only", "S0_q_y", "S1_q_y", "S2_joint_q_y"]
    for key in tqdm(keys, desc="G0c3 model-dev"):
        spec = comparators[key]
        model = fit_spec(train, spec, config, 20260724, output_dir)
        scores = model.predict_proba(model_dev, spec["mode"]).tolist()
        threshold = select_source_balanced_threshold(model_dev, scores, config["threshold_policy"])
        pred = labels_from_scores(scores, threshold["threshold"])
        rows_out.append({"comparator": key, "train_seed": "MODELDEV_SELECTION", "threshold": threshold["threshold"], "threshold_feasible": threshold["feasible"], **binary_metrics([row["exp1_label"] for row in model_dev], pred, scores)})
    full_candidates = [row for row in rows_out if row["comparator"] in {"S1_q_y", "S2_joint_q_y"}]
    best_full = max(full_candidates, key=lambda row: row["macro_f1"])["comparator"]
    best_single = max([row for row in rows_out if row["comparator"] in {"E_q_only", "E_y_only", "PairLite_q_only", "PairLite_y_only"}], key=lambda row: row["macro_f1"])["comparator"]
    frozen = {key: comparators[key] for key in keys}
    return {"modeldev_rows": rows_out, "comparators": frozen, "best_full": best_full, "best_single": best_single, "s0": "S0_q_y", "semantic_y": "E_y_only"}


def fit_spec(train: list[dict], spec: dict, config: dict, seed: int, output_dir: Path):
    if spec["backend"] == "pairlite":
        cfg = spec["config"]
        return PairLiteCPUDetector(level=spec["level"], alpha=cfg["alpha"], l1_ratio=cfg["l1_ratio"], max_iter=cfg["max_iter"], seed=seed, char_weight=cfg["char_weight"], cross_weight=cfg["cross_weight"]).fit(train, [row["exp1_label"] for row in train], spec["mode"])
    encoder = dict(config["semantic_cpu"]["encoder"])
    encoder["batch_size"] = 128
    encoder["pair_prefix"] = config["semantic_cpu"].get("joint_pair", {}).get("pair_prefix", encoder.get("query_prefix", ""))
    if spec["level"] == "S2":
        encoder["max_length"] = int(config["semantic_cpu"].get("joint_pair", {}).get("max_length", 384))
    return FrozenSemanticCPUDetector(spec["level"], encoder, str(output_dir / "embedding_cache"), c=float(spec["config"]["c"]), seed=seed).fit(train, [row["exp1_label"] for row in train], spec["mode"])


def resample_train_components(train: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for row in train:
        key = (str(row.get("exp1_label")), str(row.get("source")), str(row.get("prompt_risk_domain")))
        strata.setdefault(key, {}).setdefault(str(row.get("semantic_component_id")), []).append(row)
    out: list[dict] = []
    for components in strata.values():
        ids = sorted(components)
        sampled = rng.choice(ids, size=len(ids), replace=True)
        for component in sampled:
            out.extend(components[str(component)])
    return out[: len(train)]


def select_source_balanced_threshold(rows: list[dict], scores: list[float], policy: dict) -> dict:
    labels = [row["exp1_label"] for row in rows]
    y_true = np.asarray([1 if label == "unsafe" else 0 for label in labels])
    score_arr = np.asarray(scores, dtype=float)
    sources = [str(row.get("source")) for row in rows]
    source_counts = {source: sources.count(source) for source in set(sources)}
    weights = np.asarray([1.0 / source_counts[source] for source in sources], dtype=float)
    thresholds = sorted(set(score_arr.tolist()))
    best = {"threshold": 0.5, "weighted_macro_f1": -1.0, "feasible": False}
    curve = []
    for threshold in thresholds:
        pred = (score_arr >= threshold).astype(int)
        recall = float(((pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
        fpr = float(((pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
        weighted = float(f1_score(y_true, pred, average="macro", sample_weight=weights, zero_division=0))
        feasible = recall >= float(policy["unsafe_recall_min"]) and fpr <= float(policy["fpr_max"])
        curve.append({"threshold": float(threshold), "weighted_macro_f1": weighted, "recall": recall, "fpr": fpr, "feasible": feasible})
        if feasible and weighted > best["weighted_macro_f1"]:
            best = {"threshold": float(threshold), "weighted_macro_f1": weighted, "recall": recall, "fpr": fpr, "feasible": True}
    if not best["feasible"] and curve:
        best = max(curve, key=lambda row: row["weighted_macro_f1"])
        best["feasible"] = False
    best["source_balanced"] = True
    return best


def paired_stats(predictions: dict, selected: dict, train_seeds: list[str], iterations: int) -> dict:
    result = {}
    primary_seed = "FULL_TRAIN_DETERMINISTIC"
    full_key = selected["best_full"]
    for panel in sorted({panel for _, _, panel in predictions}):
        gold = [row["gold_label"] for row in predictions[(full_key, primary_seed, panel)]]
        clusters = [row["cluster_id"] for row in predictions[(full_key, primary_seed, panel)]]
        metric_fn = lambda yt, pred: float(f1_score(yt, pred, average="macro", zero_division=0))
        full = [row["pred_label"] for row in predictions[(full_key, primary_seed, panel)]]
        comparators = {
            "semantic_q": "E_q_only",
            "semantic_y": "E_y_only",
            "s0": selected["s0"],
            "best_single": selected["best_single"],
        }
        panel_stats = {"full_key": full_key, "primary_macro_f1": metric_fn(gold, full)}
        pvals = {}
        for name, key in comparators.items():
            if (key, primary_seed, panel) not in predictions:
                continue
            baseline = [row["pred_label"] for row in predictions[(key, primary_seed, panel)]]
            delta = metric_fn(gold, full) - metric_fn(gold, baseline)
            panel_stats[f"delta_{name}"] = delta
            panel_stats[f"bootstrap_{name}"] = paired_cluster_bootstrap_delta(gold, baseline, full, clusters, metric_fn, iterations=iterations, seed=20260727)
            pvals[f"full_vs_{name}"] = exact_mcnemar(gold, baseline, full)["p_value"]
        panel_stats["mcnemar"] = pvals
        panel_stats["holm"] = holm_adjust(pvals)
        result[panel] = panel_stats
    return result


def decide(stats: dict, metric_rows: list[dict], audit: dict, resource: dict, selected: dict, seed_direction: dict) -> str:
    if not correctness_gate(audit, resource, git_status())["passed"]:
        return "STOP_RELATION_CLAIM"
    p1 = stats.get("P1-Natural-Mixed", {})
    p2 = stats.get("P2-DVM-Core", {})
    p3 = stats.get("P3-Public-Gold", {})
    p1_ok = p1.get("primary_macro_f1", 0) >= 0.765 and p1.get("delta_semantic_y", -1) >= 0.010 and p1.get("delta_s0", -1) >= 0.003 and panel_mean(metric_rows, "P1-Natural-Mixed", selected["best_full"], "recall") >= 0.72 and panel_mean(metric_rows, "P1-Natural-Mixed", selected["best_full"], "fpr") <= 0.25
    p2_groups = audit.get("p2_dvm_audit", {}).get("groups", 0)
    p2_ok = bool(p2) and p2_groups >= 200 and p2.get("primary_macro_f1", 0) >= 0.78 and p2.get("delta_semantic_y", -1) >= 0.05 and p2.get("delta_s0", -1) >= 0.01
    p3_ok = p3.get("delta_semantic_y", 0) >= -0.01
    p1_seed_ok = seed_direction_gate(seed_direction, "P1-Natural-Mixed", ["E_y_only", selected["s0"]], min_positive_rate=0.75)
    p2_seed_ok = "P2-DVM-Core" not in stats or seed_direction_gate(seed_direction, "P2-DVM-Core", ["E_y_only", selected["s0"]], min_positive_rate=0.75)
    p3_seed_ok = seed_direction_gate(seed_direction, "P3-Public-Gold", ["E_y_only"], min_noninferior_rate=0.75, margin=-0.01)
    p1_ok = p1_ok and p1_seed_ok
    p2_ok = p2_ok and p2_seed_ok
    p3_ok = p3_ok and p3_seed_ok
    if p1_ok and p2_ok and p3_ok and p2_groups >= 300:
        return "FULL_GO"
    if p1_ok and p2_ok and p3_ok and 200 <= p2_groups < 300:
        return "FULL_GO_LIMITED_P2"
    return "STOP_RELATION_CLAIM"


def seed_direction_summary(metric_rows: list[dict], selected: dict) -> dict:
    by_key = {(row["panel"], row["comparator"], row["train_seed"]): row for row in metric_rows}
    panels = sorted({row["panel"] for row in metric_rows})
    baselines = ["E_y_only", selected["s0"], selected["best_single"]]
    full = selected["best_full"]
    out: dict[str, dict] = {}
    for panel in panels:
        out[panel] = {}
        seeds = sorted({row["train_seed"] for row in metric_rows if row["panel"] == panel and row["comparator"] == full})
        for baseline in baselines:
            deltas = []
            for seed in seeds:
                full_row = by_key.get((panel, full, seed))
                base_row = by_key.get((panel, baseline, seed))
                if not full_row or not base_row:
                    continue
                delta = float(full_row["macro_f1"]) - float(base_row["macro_f1"])
                deltas.append({"train_seed": seed, "delta_macro_f1": delta, "positive": delta > 0, "noninferior_m1pp": delta >= -0.01})
            if deltas:
                out[panel][baseline] = {
                    "count": len(deltas),
                    "positive_seeds": sum(1 for row in deltas if row["positive"]),
                    "positive_rate": sum(1 for row in deltas if row["positive"]) / len(deltas),
                    "noninferior_m1pp_seeds": sum(1 for row in deltas if row["noninferior_m1pp"]),
                    "noninferior_m1pp_rate": sum(1 for row in deltas if row["noninferior_m1pp"]) / len(deltas),
                    "deltas": deltas,
                }
    return out


def seed_direction_gate(seed_direction: dict, panel: str, baselines: list[str], min_positive_rate: float | None = None, min_noninferior_rate: float | None = None, margin: float = -0.01) -> bool:
    panel_rows = seed_direction.get(panel, {})
    for baseline in baselines:
        row = panel_rows.get(baseline)
        if not row:
            return False
        if min_positive_rate is not None and float(row.get("positive_rate", 0.0)) < min_positive_rate:
            return False
        if min_noninferior_rate is not None:
            deltas = row.get("deltas", [])
            if not deltas:
                return False
            rate = sum(1 for item in deltas if float(item["delta_macro_f1"]) >= margin) / len(deltas)
            if rate < min_noninferior_rate:
                return False
    return True


def correctness_gate(audit: dict, resource: dict, dirty: str) -> dict:
    checks = {
        "clean_git_tree": dirty == "",
        "base_data_gate": bool(audit.get("base_data_gate", {}).get("passed")),
        "p2_row_component_unique": bool(audit.get("base_data_gate", {}).get("checks", {}).get("p2_row_component_unique")),
        "resource": bool(resource.get("passed")),
        "cpu_only": not bool(resource.get("profile", {}).get("cuda_available")),
    }
    return {"passed": all(checks.values()), "checks": checks}


def prediction_rows(rows, pred, scores, threshold, spec, train_seed, panel):
    return [{"id": row["id"], "semantic_component_id": row["semantic_component_id"], "cluster_id": row.get("context_collision_group_id") or row["semantic_component_id"], "gold_label": row["exp1_label"], "pred_label": label, "pred_score": score, "threshold": threshold["threshold"], "threshold_feasible": threshold["feasible"], "level": spec["level"], "mode": spec["mode"], "backend": spec["backend"], "train_seed": train_seed, "panel": panel} for row, label, score in zip(rows, pred, scores)]


def metrics(pred_rows, panel):
    out = binary_metrics([row["gold_label"] for row in pred_rows], [row["pred_label"] for row in pred_rows], [row["pred_score"] for row in pred_rows])
    if panel == "P2-DVM-Core":
        groups: dict[str, list[dict]] = {}
        for row in pred_rows:
            groups.setdefault(row["cluster_id"], []).append(row)
        out["strict_group_consistency"] = sum(1 for members in groups.values() if all(row["gold_label"] == row["pred_label"] for row in members)) / max(len(groups), 1)
    return out


def write_metric_tables(output_dir: Path, metric_rows: list[dict]) -> None:
    write_table(output_dir / "G0c3_P1_METRICS_BY_SEED.csv", [row for row in metric_rows if row["panel"] == "P1-Natural-Mixed"])
    write_table(output_dir / "G0c3_P2_METRICS_BY_SEED.csv", [row for row in metric_rows if row["panel"] == "P2-DVM-Core"])
    write_table(output_dir / "G0c3_P3_METRICS_BY_SEED.csv", [row for row in metric_rows if row["panel"] == "P3-Public-Gold"])


def write_report(output_dir: Path, decision: dict, audit: dict, stats: dict, metric_rows: list[dict], resource: dict) -> None:
    lines = [
        "# E1_CPU_v5_G0c3_FINAL_FREEZE_整体任务报告_中文",
        "",
        "## 执行结论",
        f"- 最终决策：{decision['decision']}",
        f"- Git commit：{decision['git_commit']}",
        f"- P2 groups：{audit.get('p2_dvm_audit', {}).get('groups', 0)}",
        f"- 选择 full comparator：{decision['selected']['best_full']}",
        "",
        "## 数据与正确性",
        "```json",
        json.dumps({"base": audit.get("base_data_gate"), "p2": audit.get("p2_data_gate"), "counts": audit.get("counts")}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 模型结果",
        "```json",
        json.dumps(stats, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 资源",
        "```json",
        json.dumps(resource, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 分析",
        "G0c3 按 final-freeze 协议执行：P2 先于 P1 冻结，P2 匹配以 component 为容量节点，runner 同时评测 P1/P2/P3，并保留 semantic q-only、semantic y-only、S0、S1、S2 与 PairLite 基线。最终决策严格由 FULL_GO / FULL_GO_LIMITED_P2 / STOP_RELATION_CLAIM 三态给出。",
    ]
    (output_dir / "reports" / "E1_CPU_v5_G0c3_FINAL_FREEZE_整体任务报告_中文.md").write_text("\n".join(lines), encoding="utf-8")


def panel_mean(rows: list[dict], panel: str, comparator: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["panel"] == panel and row["comparator"] == comparator and row["train_seed"] != "FULL_TRAIN_DETERMINISTIC"]
    return float(np.mean(values)) if values else 0.0


def protocol_lock(config: dict, audit: dict, manifest_dir: Path, dirty: str) -> dict:
    return {
        "protocol": "G0c3-FINAL-FREEZE",
        "created_at_unix": time.time(),
        "git_commit": git_commit(),
        "git_status_porcelain": dirty,
        "source_tree_sha256": source_tree_sha256(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "manifest_sha256": audit.get("manifest_sha256", {}),
        "dataset_revisions": config["data_policy"].get("dataset_revisions", {}),
        "encoder_revision": config["semantic_cpu"]["encoder"]["revision"],
        "manifest_dir": str(manifest_dir),
    }


def selected_summary(selected: dict) -> dict:
    return {key: selected[key] for key in ("best_full", "best_single", "s0", "semantic_y")}


def rows(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_table(path: Path, rows_out: list[dict]) -> None:
    if not rows_out:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows_out for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def source_tree_sha256() -> str:
    h = hashlib.sha256()
    for path in sorted([*Path("src").rglob("*.py"), *Path("scripts").rglob("*.py"), *Path("configs").rglob("*.yaml")]):
        if path.exists():
            h.update(str(path).encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
