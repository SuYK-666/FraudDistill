from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frauddistill.eval.metrics import binary_metrics
from frauddistill.exp1_ccfa.pair_cross_encoder import labels_from_scores
from frauddistill.exp1_ccfa.pairlite_cpu import PairLiteCPUDetector
from frauddistill.exp1_ccfa.paired_cluster_bootstrap import paired_cluster_bootstrap_delta
from frauddistill.exp1_ccfa.resource_profile import ResourceProfiler, resource_gate
from frauddistill.utils.io import read_jsonl, write_jsonl


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_relation_gate_v6.yaml"


COMPARATORS = {
    "M1_q_only": {"level": "B1", "mode": "q_only", "input": "q"},
    "M2_y_only": {"level": "B1", "mode": "y_only", "input": "y"},
    "M3_additive_q_y": {"level": "B1", "mode": "q_y", "input": "q+y"},
    "M4_relation_full": {"level": "R2", "mode": "q_y", "input": "q+y+relation"},
    "PairLite_R2": {"level": "R2", "mode": "q_y", "input": "q+y+relation"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FraudDistill E1 Relation-Gate v6 pilot")
    parser.add_argument("--manifest_dir", default="data/prepared/e1_relation_gate_v6")
    parser.add_argument("--output_dir", default="outputs/e1_relation_gate_v6/pilot")
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    args = parser.parse_args()
    summary = run(ROOT / args.manifest_dir, ROOT / args.output_dir, args.bootstrap_iterations)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["decision"] != "E1_G1_PILOT_PASS":
        raise SystemExit(2)


def run(manifest_dir: Path, output_dir: Path, bootstrap_iterations: int) -> dict:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("predictions", "models", "reports", "tables"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    profiler = ResourceProfiler(output_dir)
    census = json.loads((manifest_dir / "E1_G0_DATA_CENSUS.json").read_text(encoding="utf-8"))
    if not census.get("passed"):
        decision = {"decision": "E1_G0_STOP", "reason": "G0 data census did not pass", "g0": census}
        write_json(output_dir / "E1_DECISION.json", decision)
        write_report(output_dir, decision, [], {}, {}, resource_profile(output_dir, profiler, config))
        return decision
    train = list(read_jsonl(manifest_dir / "train.jsonl"))
    calibration = list(read_jsonl(manifest_dir / "calibration_dev.jsonl"))
    pilot = list(read_jsonl(manifest_dir / "pilot_test.jsonl"))
    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    all_predictions: dict[tuple[str, int], list[dict]] = {}
    seeds = [int(seed) for seed in config["data"]["pilot"]["seeds"]]
    for seed in tqdm(seeds, desc="E1-G1 pilot seeds"):
        train_seed_rows = resample_rows(train, seed)
        for name, spec in tqdm(COMPARATORS.items(), desc=f"seed {seed}", leave=False):
            model = fit_pairlite(train_seed_rows, spec, config, seed)
            calibrator = calibrate(model, calibration, spec)
            scores = predict_scores(model, calibrator, pilot, spec)
            threshold = {"threshold": 0.5, "policy": "calibrated_fixed_0.5"}
            pred = labels_from_scores(scores, 0.5)
            pred_rows = prediction_rows(pilot, pred, scores, name, seed, threshold)
            all_predictions[(name, seed)] = pred_rows
            write_jsonl(output_dir / "predictions" / f"pilot_{name}_seed{seed}.jsonl", pred_rows)
            metric_rows.extend(metrics_by_subset(pred_rows, name, seed))
            threshold_rows.append({"comparator": name, "seed": seed, **threshold})
            if name in {"M2_y_only", "M4_relation_full"}:
                joblib.dump({"model": model, "calibrator": calibrator, "spec": spec, "seed": seed}, output_dir / "models" / f"{name}_seed{seed}.joblib")
            profiler.sample()
        shuffled = shuffle_queries(pilot, seed)
        relation = fit_pairlite(train_seed_rows, COMPARATORS["M4_relation_full"], config, seed)
        relation_calibrator = calibrate(relation, calibration, COMPARATORS["M4_relation_full"])
        shuffled_scores = predict_scores(relation, relation_calibrator, shuffled, COMPARATORS["M4_relation_full"])
        shuffled_pred = labels_from_scores(shuffled_scores, 0.5)
        write_jsonl(output_dir / "predictions" / f"pilot_M4_relation_full_q_shuffle_seed{seed}.jsonl", prediction_rows(shuffled, shuffled_pred, shuffled_scores, "M4_relation_full_q_shuffle", seed, {"threshold": 0.5}))
    write_table(output_dir / "E1_METRICS_BY_SEED.csv", metric_rows)
    write_table(output_dir / "E1_CALIBRATION.csv", threshold_rows)
    paired = paired_effects(all_predictions, seeds, bootstrap_iterations)
    write_json(output_dir / "E1_PAIRED_EFFECTS.json", paired)
    shuffle_rows = q_shuffle_results(output_dir, all_predictions, seeds)
    write_table(output_dir / "E1_Q_SHUFFLE_RESULTS.csv", shuffle_rows)
    loso = leave_one_source_out(all_predictions, seeds)
    write_table(output_dir / "E1_LOSO_SOURCE_RESULTS.csv", loso)
    resource = resource_profile(output_dir, profiler, config)
    decision = decide(metric_rows, paired, shuffle_rows, resource, config)
    payload = {
        "decision": decision,
        "wall_seconds_observed": time.perf_counter() - started,
        "git_commit": git_commit(),
        "g0_census": census,
        "paired_effects": paired,
        "resource": resource,
    }
    write_json(output_dir / "E1_DECISION.json", payload)
    write_json(output_dir / "E1_RESOURCE_PROFILE.json", resource)
    write_failure_index(output_dir / "E1_FAILURE_CASE_INDEX.csv", all_predictions)
    write_report(output_dir, payload, metric_rows, paired, {"q_shuffle": shuffle_rows, "loso": loso}, resource)
    return payload


def fit_pairlite(rows: list[dict], spec: dict, config: dict, seed: int) -> PairLiteCPUDetector:
    cfg = config["model"]["pairlite"]
    return PairLiteCPUDetector(
        level=spec["level"],
        alpha=float(cfg["alpha"]),
        l1_ratio=float(cfg["l1_ratio"]),
        max_iter=int(cfg["max_iter"]),
        seed=seed,
        word_features=int(cfg["word_features"]),
        char_features=int(cfg["char_features"]),
        hash_features=int(cfg["hash_features"]),
        top_k_cross=int(cfg["top_k_cross"]),
    ).fit(rows, [row["exp1_label"] for row in rows], spec["mode"])


def calibrate(model: PairLiteCPUDetector, rows: list[dict], spec: dict):
    scores = model.predict_proba(rows, spec["mode"])
    labels = np.asarray([1 if row["exp1_label"] == "unsafe" else 0 for row in rows], dtype=np.int8)
    if len(set(labels.tolist())) < 2:
        return None
    # Sigmoid calibration is fit only on calibration_dev; threshold remains fixed at 0.5.
    calibrator = CalibratedClassifierCV(estimator=model.classifier, method="sigmoid", cv="prefit")
    features = model.features(rows, spec["mode"])
    calibrator.fit(features, labels)
    return calibrator


def predict_scores(model: PairLiteCPUDetector, calibrator, rows: list[dict], spec: dict) -> list[float]:
    if calibrator is None:
        return model.predict_proba(rows, spec["mode"]).tolist()
    return calibrator.predict_proba(model.features(rows, spec["mode"]))[:, 1].tolist()


def metrics_by_subset(rows: list[dict], comparator: str, seed: int) -> list[dict]:
    out = []
    for subset in ("R1", "R2", "R3", "ALL"):
        subset_rows = rows if subset == "ALL" else [row for row in rows if row.get("e1_subset") == subset]
        if not subset_rows:
            continue
        gold = [row["gold_label"] for row in subset_rows]
        pred = [row["pred_label"] for row in subset_rows]
        score = [float(row["unsafe_score"]) for row in subset_rows]
        base = binary_metrics(gold, pred, score)
        base.update(extra_metrics(gold, pred, score))
        out.append({"subset": subset, "comparator": comparator, "seed": seed, **base})
    r_scores = [row["macro_f1"] for row in out if row["subset"] in {"R1", "R2", "R3"}]
    if len(r_scores) == 3:
        out.append({"subset": "E1_SCORE", "comparator": comparator, "seed": seed, "macro_f1": float(np.mean(r_scores))})
    return out


def extra_metrics(gold: list[str], pred: list[str], score: list[float]) -> dict:
    y = np.asarray([1 if label == "unsafe" else 0 for label in gold], dtype=int)
    p = np.asarray([1 if label == "unsafe" else 0 for label in pred], dtype=int)
    s = np.asarray(score, dtype=float)
    result = {
        "specificity": float(((p == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "brier": float(brier_score_loss(y, np.clip(s, 0.0, 1.0))),
        "ece": expected_calibration_error(y, s),
    }
    if len(set(y.tolist())) == 2:
        result["auroc"] = float(roc_auc_score(y, s))
        result["auprc"] = float(average_precision_score(y, s))
    return result


def expected_calibration_error(y: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(scores)
    y_sorted = y[order]
    s_sorted = scores[order]
    chunks = np.array_split(np.arange(len(scores)), min(bins, len(scores)))
    ece = 0.0
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        conf = float(np.mean(s_sorted[chunk]))
        acc = float(np.mean(y_sorted[chunk]))
        ece += (len(chunk) / max(len(scores), 1)) * abs(acc - conf)
    return float(ece)


def prediction_rows(rows: list[dict], pred: list[str], scores: list[float], comparator: str, seed: int, threshold: dict) -> list[dict]:
    out = []
    for row, label, score in zip(rows, pred, scores):
        out.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "e1_subset": row.get("e1_subset"),
                "cluster_id": row.get("cluster_id") or row.get("semantic_component_id"),
                "semantic_component_id": row.get("semantic_component_id"),
                "relation_group_id": row.get("relation_group_id"),
                "gold_label": row.get("exp1_label"),
                "pred_label": label,
                "unsafe_score": float(score),
                "comparator": comparator,
                "seed": seed,
                "threshold": threshold.get("threshold", 0.5),
            }
        )
    return out


def paired_effects(predictions: dict[tuple[str, int], list[dict]], seeds: list[int], iterations: int) -> dict:
    result = {}
    for seed in seeds:
        rel = predictions[("M4_relation_full", seed)]
        y = predictions[("M2_y_only", seed)]
        additive = predictions[("M3_additive_q_y", seed)]
        q = predictions[("M1_q_only", seed)]
        for base_name, base_rows in (("y_only", y), ("q_only", q), ("additive", additive)):
            key = f"seed{seed}_relation_vs_{base_name}"
            result[key] = paired_cluster_bootstrap_delta(
                [row["gold_label"] for row in base_rows],
                [row["pred_label"] for row in base_rows],
                [row["pred_label"] for row in rel],
                [row["cluster_id"] for row in rel],
                lambda gold, pred: float(f1_score(gold, pred, average="macro", zero_division=0)),
                iterations=iterations,
                seed=seed,
            )
    return result


def q_shuffle_results(output_dir: Path, predictions: dict[tuple[str, int], list[dict]], seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        rel = predictions[("M4_relation_full", seed)]
        y = predictions[("M2_y_only", seed)]
        shuffled = list(read_jsonl(output_dir / "predictions" / f"pilot_M4_relation_full_q_shuffle_seed{seed}.jsonl"))
        rel_score = e1_score(rel)
        y_score = e1_score(y)
        shuffle_score = e1_score(shuffled)
        rows.append({"seed": seed, "relation_e1_score": rel_score, "y_only_e1_score": y_score, "q_shuffle_e1_score": shuffle_score, "shuffle_drop": rel_score - shuffle_score, "shuffle_minus_y": shuffle_score - y_score})
    return rows


def leave_one_source_out(predictions: dict[tuple[str, int], list[dict]], seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        rel = predictions[("M4_relation_full", seed)]
        y = predictions[("M2_y_only", seed)]
        for source in sorted({str(row.get("source")) for row in rel}):
            rel_keep = [row for row in rel if str(row.get("source")) != source]
            y_keep = [row for row in y if str(row.get("source")) != source]
            rows.append({"seed": seed, "left_out_source": source, "relation_e1_score": e1_score(rel_keep), "y_only_e1_score": e1_score(y_keep), "delta": e1_score(rel_keep) - e1_score(y_keep)})
    return rows


def e1_score(rows: list[dict]) -> float:
    scores = []
    for subset in ("R1", "R2", "R3"):
        subset_rows = [row for row in rows if row.get("e1_subset") == subset]
        if subset_rows:
            scores.append(float(f1_score([r["gold_label"] for r in subset_rows], [r["pred_label"] for r in subset_rows], average="macro", zero_division=0)))
    return float(np.mean(scores)) if scores else 0.0


def decide(metric_rows: list[dict], paired: dict, shuffle_rows: list[dict], resource: dict, config: dict) -> str:
    gate = config["gates"]["pilot"]
    by = {(row["comparator"], row["seed"], row["subset"]): row for row in metric_rows}
    seeds = sorted({row["seed"] for row in metric_rows})
    relation_scores = [by[("M4_relation_full", seed, "E1_SCORE")]["macro_f1"] for seed in seeds if ("M4_relation_full", seed, "E1_SCORE") in by]
    y_scores = [by[("M2_y_only", seed, "E1_SCORE")]["macro_f1"] for seed in seeds if ("M2_y_only", seed, "E1_SCORE") in by]
    q_scores = [by[("M1_q_only", seed, "E1_SCORE")]["macro_f1"] for seed in seeds if ("M1_q_only", seed, "E1_SCORE") in by]
    r1_q_auroc = [by[("M1_q_only", seed, "R1")].get("auroc", 1.0) for seed in seeds if ("M1_q_only", seed, "R1") in by]
    checks = {
        "q_only_lt_y_only": np.mean(q_scores) < np.mean(y_scores) if q_scores and y_scores else False,
        "r1_q_only_auroc_range": bool(r1_q_auroc) and min(r1_q_auroc) >= float(gate["r1_q_only_auroc_min"]) and max(r1_q_auroc) <= float(gate["r1_q_only_auroc_max"]),
        "overall_q_only_auroc": all(by[("M1_q_only", seed, "ALL")].get("auroc", 1.0) <= float(gate["overall_q_only_auroc_max"]) for seed in seeds if ("M1_q_only", seed, "ALL") in by),
        "y_only_range": bool(y_scores) and float(gate["y_only_score_min"]) <= np.mean(y_scores) <= float(gate["y_only_score_max"]),
        "relation_score": bool(relation_scores) and np.mean(relation_scores) >= float(gate["relation_score_min"]),
        "relation_delta": bool(relation_scores and y_scores) and np.mean(relation_scores) - np.mean(y_scores) >= float(gate["relation_delta_y_min"]),
        "seed_direction": all(r > y for r, y in zip(relation_scores, y_scores)),
        "q_shuffle": bool(shuffle_rows) and min(row["shuffle_drop"] for row in shuffle_rows) >= float(gate["q_shuffle_drop_min"]) and max(abs(row["shuffle_minus_y"]) for row in shuffle_rows) <= float(gate["q_shuffle_y_gap_max"]),
        "resource": bool(resource.get("passed")),
    }
    write_json(ROOT / "outputs" / "e1_relation_gate_v6" / "pilot" / "E1_PILOT_GATE_CHECKS.json", checks)
    return "E1_G1_PILOT_PASS" if all(checks.values()) else "E1_G1_PILOT_STOP"


def resample_rows(rows: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(rows))
    return [rows[int(i)] for i in indices]


def shuffle_queries(rows: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    queries = [row.get("user_query", "") for row in rows]
    shuffled = rng.permutation(queries)
    return [dict(row, user_query=str(query)) for row, query in zip(rows, shuffled)]


def resource_profile(output_dir: Path, profiler: ResourceProfiler, config: dict) -> dict:
    profile = profiler.finish()
    profile["hardware"] = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()}
    profile["deploy_bundle_mb"] = dir_mb(output_dir / "models")
    profile["results_mb"] = dir_mb(output_dir) - profile["deploy_bundle_mb"]
    gate_config = {"cpu_only": True, "peak_ram_mb_max": config["gates"]["pilot"]["peak_ram_mb_max"], "g0_wall_time_minutes_max": config["gates"]["pilot"]["wall_minutes_max"], "artifact_mb_max": 300}
    gated = resource_gate(profile, gate_config)
    gated.update(profile)
    return gated


def dir_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / (1024 * 1024)


def write_report(output_dir: Path, decision: dict, metric_rows: list[dict], paired: dict, controls: dict, resource: dict) -> None:
    report = output_dir / "reports" / "E1_CPU_CCF-A_v6_Pilot_整体任务报告_中文.md"
    lines = [
        "# FraudDistill E1 Relation-Gate v6 Pilot 整体任务报告",
        "",
        f"- 决策：`{decision.get('decision')}`",
        f"- Git commit：`{decision.get('git_commit', git_commit())}`",
        f"- CPU-only：`{not resource.get('cuda_available', False)}`",
        f"- Peak RSS MB：`{resource.get('peak_rss_mb')}`",
        f"- Wall seconds：`{resource.get('wall_seconds')}`",
        "",
        "## 结果解读",
        "",
        "本轮严格按 v6 冻结方案执行：G0 只做 response-level gold 的数据可行性审计；G1 使用 development pool 进行 pilot，不读取或调节 formal test。报告中的 PASS/STOP 是由门槛自动计算得到，未根据测试结果改标签、改阈值或改子集比例。",
        "",
        "## 主表",
        "",
        "| Model | Input | Seed | R1 F1 | R2 F1 | R3 F1 | E1-Score | AUPRC | Recall | FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparator in COMPARATORS:
        for seed in sorted({row["seed"] for row in metric_rows}):
            values = {row["subset"]: row for row in metric_rows if row["comparator"] == comparator and row["seed"] == seed}
            if not values:
                continue
            all_row = values.get("ALL", {})
            lines.append(
                f"| {comparator} | {COMPARATORS[comparator]['input']} | {seed} | "
                f"{values.get('R1', {}).get('macro_f1', 0):.4f} | {values.get('R2', {}).get('macro_f1', 0):.4f} | {values.get('R3', {}).get('macro_f1', 0):.4f} | "
                f"{values.get('E1_SCORE', {}).get('macro_f1', 0):.4f} | {all_row.get('auprc', 0):.4f} | {all_row.get('recall', 0):.4f} | {all_row.get('fpr', 0):.4f} |"
            )
    lines.extend(["", "## 关系增益与控制实验", "", "paired effects、q-shuffle、leave-one-source-out 的完整原始文件分别保存在 `E1_PAIRED_EFFECTS.json`、`E1_Q_SHUFFLE_RESULTS.csv`、`E1_LOSO_SOURCE_RESULTS.csv`。"])
    if paired:
        deltas = [row.get("delta_mean", 0.0) for key, row in paired.items() if "relation_vs_y_only" in key]
        lines.append(f"- Relation vs y-only 平均 bootstrap delta：{float(np.mean(deltas)) if deltas else 0.0:.4f}")
    if controls.get("q_shuffle"):
        drops = [row["shuffle_drop"] for row in controls["q_shuffle"]]
        lines.append(f"- q-shuffle 平均下降：{float(np.mean(drops)):.4f}")
    lines.extend(["", "## 结论", "", "若本轮为 STOP，原因应优先从公开 response-level fraud gold 的数量、关系可学习性、q 泄漏或 y-only 表面特征过强中分析；不得用 formal test 反复调参。"])
    report.write_text("\n".join(lines), encoding="utf-8")


def write_failure_index(path: Path, predictions: dict[tuple[str, int], list[dict]]) -> None:
    rows = []
    for (name, seed), pred_rows in predictions.items():
        if name != "M4_relation_full":
            continue
        for row in pred_rows:
            if row["gold_label"] != row["pred_label"]:
                rows.append({"id": row["id"], "source": row["source"], "subset": row["e1_subset"], "semantic_component_id": row["semantic_component_id"], "error_type": f"{row['gold_label']}_as_{row['pred_label']}", "seed": seed})
    write_table(path, rows)


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
