from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
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
from frauddistill.exp1_ccfa.residual_relation_cpu import ResidualRelationCPUDetector
from frauddistill.exp1_ccfa.resource_profile import ResourceProfiler, dir_size_mb, resource_gate
from frauddistill.utils.io import read_jsonl, write_jsonl


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_relation_gate_v6r1.yaml"


COMPARATORS = {
    "M1_q_only": {"backend": "pairlite", "level": "B1", "mode": "q_only", "input": "q"},
    "M2_y_only": {"backend": "pairlite", "level": "B1", "mode": "y_only", "input": "y"},
    "M3_additive_q_y": {"backend": "pairlite", "level": "B1", "mode": "q_y", "input": "q+y"},
    "M4_pairlite_r2": {"backend": "pairlite", "level": "R2", "mode": "q_y", "input": "q+y+sparse-cross"},
    "M5_residual_relation": {"backend": "residual", "mode": "q_y", "input": "y-logit+bounded-relation"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 Relation-Gate v6r1")
    parser.add_argument("--stage", choices=["g0", "smoke", "pilot", "formal"], required=True)
    parser.add_argument("--manifest_dir", default="data/prepared/e1_relation_gate_v6r1")
    parser.add_argument("--output_dir", default="outputs/e1_relation_gate_v6r1")
    parser.add_argument("--allow_formal_without_pass", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.stage == "g0":
        from frauddistill.exp1_ccfa.relation_manifest import write_relation_manifests_v6r1

        census = write_relation_manifests_v6r1(ROOT / args.manifest_dir, config, ROOT / "configs" / "data" / "FRAUD_TAXONOMY_LOCK.yaml", int(config["data"]["seed"]))
        g0_out = ROOT / args.output_dir / "g0"
        g0_out.mkdir(parents=True, exist_ok=True)
        decision = "E1_V6R1_G0R_PASS" if census.get("passed") else "E1_V6R1_STOP"
        payload = {"decision": decision, "stage": "g0", "git_commit": git_commit(), "census": census}
        write_json(g0_out / "E1_V6R1_DECISION.json", payload)
        write_g0_report(g0_out / "E1_V6R1_REPORT_CN.md", payload)
        print(json.dumps(census, ensure_ascii=False, indent=2, default=json_default))
        if not census["passed"]:
            raise SystemExit(2)
        return
    payload = run_stage(args.stage, ROOT / args.manifest_dir, ROOT / args.output_dir / args.stage, config, args.allow_formal_without_pass)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    if payload["decision"] not in {"E1_V6R1_SMOKE_PASS", "E1_V6R1_FULL_READY", "E1_V6R1_FULL_COMPLETE"}:
        raise SystemExit(2)


def run_stage(stage: str, manifest_dir: Path, output_dir: Path, config: dict, allow_formal_without_pass: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("predictions", "models", "reports", "tables"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    if stage == "formal" and not allow_formal_without_pass:
        pass_file = output_dir.parent / "pilot" / "E1_V6R1_DECISION.json"
        if not pass_file.exists() or json.loads(pass_file.read_text(encoding="utf-8")).get("decision") != "E1_V6R1_FULL_READY":
            payload = {"decision": "E1_V6R1_FORMAL_LOCKED", "reason": "pilot did not produce E1_V6R1_FULL_READY"}
            write_json(output_dir / "E1_V6R1_DECISION.json", payload)
            return payload
    profiler = ResourceProfiler(output_dir)
    cfg = config["data"][stage if stage != "formal" else "formal"]
    prefix = "smoke" if stage == "smoke" else ("pilot" if stage == "pilot" else "formal")
    train = rows(manifest_dir / f"{prefix}_train.jsonl")
    model_dev = rows(manifest_dir / f"{prefix}_model_dev.jsonl")
    calibration = rows(manifest_dir / f"{prefix}_calibration_dev.jsonl")
    test = rows(manifest_dir / f"{prefix}_test.jsonl")
    seeds = [int(seed) for seed in cfg["seeds"]]
    bootstrap_iterations = int(cfg["bootstrap_iterations"])
    metric_rows: list[dict] = []
    modeldev_rows: list[dict] = []
    threshold_rows: list[dict] = []
    all_predictions: dict[tuple[str, int], list[dict]] = {}
    selected_lambdas = {}
    for seed in tqdm(seeds, desc=f"E1 v6r1 {stage} seeds"):
        train_seed = shuffled(train, seed)
        seed_models: dict[str, object] = {}
        for name, spec in tqdm(COMPARATORS.items(), desc=f"seed {seed}", leave=False):
            model = fit_model(train_seed, spec, config, seed)
            if name == "M5_residual_relation":
                lambda_info = select_lambda(model, model_dev, seed_models["M2_y_only"], config)
                selected_lambdas[str(seed)] = lambda_info
                model.with_lambda(float(lambda_info["selected_lambda"]))
            dev_scores = scores(model, model_dev, spec)
            modeldev_rows.extend(metrics_by_subset(prediction_rows(model_dev, labels_from_scores(dev_scores, 0.5), dev_scores, name, seed, {"threshold": 0.5}), name, seed, prefix="model_dev"))
            threshold = select_threshold(calibration, scores(model, calibration, spec))
            threshold_rows.append({"comparator": name, "seed": seed, **threshold})
            test_scores = scores(model, test, spec)
            pred = labels_from_scores(test_scores, float(threshold["threshold"]))
            pred_rows = prediction_rows(test, pred, test_scores, name, seed, threshold)
            write_jsonl(output_dir / "predictions" / f"{prefix}_{name}_seed{seed}.jsonl", pred_rows)
            all_predictions[(name, seed)] = pred_rows
            metric_rows.extend(metrics_by_subset(pred_rows, name, seed))
            seed_models[name] = model
            if name in {"M2_y_only", "M5_residual_relation"}:
                joblib.dump({"model": model, "spec": spec, "threshold": threshold, "seed": seed}, output_dir / "models" / f"{name}_seed{seed}.joblib")
            profiler.sample()
        shuffled_test = stratified_q_derangement(test, seed)
        m5 = seed_models["M5_residual_relation"]
        shuffle_scores = scores(m5, shuffled_test, COMPARATORS["M5_residual_relation"])
        shuffle_pred = labels_from_scores(shuffle_scores, float([row for row in threshold_rows if row["comparator"] == "M5_residual_relation" and row["seed"] == seed][0]["threshold"]))
        write_jsonl(output_dir / "predictions" / f"{prefix}_M5_q_shuffle_seed{seed}.jsonl", prediction_rows(shuffled_test, shuffle_pred, shuffle_scores, "M5_q_shuffle", seed, {"threshold": 0.5}))
    write_table(output_dir / "E1_V6R1_MODELDEV_SELECTION.csv", modeldev_rows)
    write_json(output_dir / "E1_V6R1_SELECTED_LAMBDA.json", selected_lambdas)
    write_table(output_dir / "E1_V6R1_THRESHOLDS.csv", threshold_rows)
    write_json(output_dir / "E1_V6R1_CALIBRATION.json", {"thresholds": threshold_rows})
    write_table(output_dir / "E1_V6R1_METRICS_BY_SEED.csv", metric_rows)
    write_table(output_dir / "E1_V6R1_METRICS_BY_SUBSET_SOURCE_FAMILY.csv", metrics_by_stratum(all_predictions))
    paired = paired_e1_bootstrap(all_predictions, seeds, bootstrap_iterations)
    write_json(output_dir / "E1_V6R1_PAIRED_E1_BOOTSTRAP.json", paired)
    shuffle = q_shuffle_rows(output_dir, prefix, all_predictions, seeds)
    write_table(output_dir / "E1_V6R1_Q_SHUFFLE.csv", shuffle)
    source_excluded = source_excluded_rows(all_predictions, seeds)
    write_table(output_dir / "E1_V6R1_SOURCE_EXCLUDED.csv", source_excluded)
    write_failure_index(output_dir / "E1_V6R1_FAILURE_CASE_INDEX.csv", all_predictions)
    resource = resource_summary(output_dir, profiler, config, stage)
    write_json(output_dir / "E1_V6R1_RESOURCE_PROFILE.json", resource)
    decision = decide(stage, metric_rows, modeldev_rows, paired, shuffle, resource, selected_lambdas, config)
    payload = {"decision": decision, "stage": stage, "git_commit": git_commit(), "selected_lambdas": selected_lambdas, "resource": resource}
    write_json(output_dir / "E1_V6R1_DECISION.json", payload)
    write_model_card(output_dir / "E1_V6R1_MODEL_CARD.md", selected_lambdas, stage)
    write_report(output_dir / "E1_V6R1_REPORT_CN.md", payload, metric_rows, paired, shuffle)
    return payload


def fit_model(rows_: list[dict], spec: dict, config: dict, seed: int):
    if spec["backend"] == "residual":
        residual_cfg = {key: value for key, value in config["model"]["residual"].items() if key != "lambda_grid"}
        return ResidualRelationCPUDetector(seed=seed, pairlite_config=config["model"]["pairlite"], **residual_cfg).fit(rows_, [row["exp1_label"] for row in rows_], stratum_weights(rows_))
    cfg = config["model"]["pairlite"]
    return PairLiteCPUDetector(level=spec["level"], alpha=cfg["alpha"], l1_ratio=cfg["l1_ratio"], max_iter=cfg["max_iter"], seed=seed, word_features=cfg["word_features"], char_features=cfg["char_features"], hash_features=cfg["hash_features"], top_k_cross=cfg["top_k_cross"]).fit(rows_, [row["exp1_label"] for row in rows_], spec["mode"])


def scores(model, rows_: list[dict], spec: dict) -> list[float]:
    return model.predict_proba(rows_, spec.get("mode", "q_y")).tolist()


def select_lambda(model: ResidualRelationCPUDetector, model_dev: list[dict], y_model, config: dict) -> dict:
    y_scores = y_model.predict_proba(model_dev, "y_only").tolist()
    y_pred = labels_from_scores(y_scores, 0.5)
    y_rows = prediction_rows(model_dev, y_pred, y_scores, "M2_y_only", 0, {"threshold": 0.5})
    candidates = []
    for value in config["model"]["residual"]["lambda_grid"]:
        model.with_lambda(float(value))
        m_scores = model.predict_proba(model_dev).tolist()
        m_pred = labels_from_scores(m_scores, 0.5)
        m_rows = prediction_rows(model_dev, m_pred, m_scores, "M5_residual_relation", 0, {"threshold": 0.5})
        r1r2_gain = np.mean([subset_f1(m_rows, s) - subset_f1(y_rows, s) for s in ("R1", "R2")])
        r3_delta = subset_f1(m_rows, "R3") - subset_f1(y_rows, "R3")
        candidates.append({"lambda": float(value), "e1_score": e1_score(m_rows), "delta_y": e1_score(m_rows) - e1_score(y_rows), "r1_r2_gain": float(r1r2_gain), "r3_delta": float(r3_delta)})
    feasible = [row for row in candidates if row["lambda"] > 0 and row["r3_delta"] >= -0.005 and row["r1_r2_gain"] >= 0.03]
    selected = max(feasible or candidates, key=lambda row: (row["e1_score"], row["delta_y"]))
    return {"selected_lambda": selected["lambda"], "candidates": candidates, "selection_rule": "max model_dev E1 with R3 noninferiority and R1/R2 gain when feasible"}


def select_threshold(calibration: list[dict], score_values: list[float]) -> dict:
    thresholds = sorted(set([0.5, *np.quantile(np.asarray(score_values, dtype=float), np.linspace(0.05, 0.95, 37)).tolist()]))
    best = None
    fallback = None
    for threshold in thresholds:
        pred = labels_from_scores(score_values, float(threshold))
        pred_rows = prediction_rows(calibration, pred, score_values, "cal", 0, {"threshold": threshold})
        all_metrics = metric_dict(pred_rows)
        item = {"threshold": float(threshold), **all_metrics, "feasible": all_metrics["recall"] >= 0.85 and all_metrics["fpr"] <= 0.15}
        if item["feasible"] and (best is None or item["e1_score"] > best["e1_score"]):
            best = item
        if fallback is None or (item["recall"] >= 0.85, -item["fpr"], item["e1_score"]) > (fallback["recall"] >= 0.85, -fallback["fpr"], fallback["e1_score"]):
            fallback = item
    return best or dict(fallback, threshold_feasible=False)


def metrics_by_subset(pred_rows: list[dict], comparator: str, seed: int, prefix: str = "") -> list[dict]:
    out = []
    for subset in ("R1", "R2", "R3", "ALL"):
        subset_rows = pred_rows if subset == "ALL" else [row for row in pred_rows if row.get("e1_subset") == subset]
        if not subset_rows:
            continue
        base = metric_dict(subset_rows)
        out.append({"split": prefix or "test", "subset": subset, "comparator": comparator, "seed": seed, **base})
    out.append({"split": prefix or "test", "subset": "E1_SCORE", "comparator": comparator, "seed": seed, "macro_f1": e1_score(pred_rows)})
    return out


def metric_dict(pred_rows: list[dict]) -> dict:
    gold = [row["gold_label"] for row in pred_rows]
    pred = [row["pred_label"] for row in pred_rows]
    score_values = [float(row["unsafe_score"]) for row in pred_rows]
    result = binary_metrics(gold, pred, score_values)
    y = np.asarray([1 if label == "unsafe" else 0 for label in gold], dtype=int)
    s = np.asarray(score_values, dtype=float)
    result["e1_score"] = e1_score(pred_rows)
    result["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1)))
    result["ece"] = expected_calibration_error(y, s)
    if len(set(y.tolist())) == 2:
        result["auroc"] = float(roc_auc_score(y, s))
        result["auprc"] = float(average_precision_score(y, s))
    return result


def expected_calibration_error(y: np.ndarray, s: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(s)
    y = y[order]
    s = s[order]
    ece = 0.0
    for chunk in np.array_split(np.arange(len(s)), min(bins, len(s))):
        if len(chunk):
            ece += len(chunk) / max(len(s), 1) * abs(float(np.mean(y[chunk])) - float(np.mean(s[chunk])))
    return float(ece)


def prediction_rows(rows_: list[dict], pred: list[str], score_values: list[float], comparator: str, seed: int, threshold: dict) -> list[dict]:
    return [
        {
            "id": row.get("id"),
            "row_uid": row.get("row_uid"),
            "source": row.get("source"),
            "fraud_family_q_only": row.get("fraud_family_q_only") or row.get("fraud_family"),
            "e1_subset": row.get("e1_subset"),
            "cluster_id": row.get("relation_group_id") or row.get("cluster_id") or row.get("semantic_component_id"),
            "semantic_component_id": row.get("semantic_component_id"),
            "relation_group_id": row.get("relation_group_id"),
            "gold_label": row.get("exp1_label"),
            "pred_label": p,
            "unsafe_score": float(s),
            "comparator": comparator,
            "seed": seed,
            "threshold": threshold.get("threshold", 0.5),
        }
        for row, p, s in zip(rows_, pred, score_values)
    ]


def e1_score(pred_rows: list[dict]) -> float:
    values = [subset_f1(pred_rows, subset) for subset in ("R1", "R2", "R3") if any(row.get("e1_subset") == subset for row in pred_rows)]
    return float(np.mean(values)) if values else 0.0


def subset_f1(pred_rows: list[dict], subset: str) -> float:
    rows_ = [row for row in pred_rows if row.get("e1_subset") == subset]
    if not rows_:
        return 0.0
    return float(f1_score([row["gold_label"] for row in rows_], [row["pred_label"] for row in rows_], average="macro", zero_division=0))


def paired_e1_bootstrap(predictions: dict[tuple[str, int], list[dict]], seeds: list[int], iterations: int) -> dict:
    result = {}
    for seed in seeds:
        m2 = predictions[("M2_y_only", seed)]
        m5 = predictions[("M5_residual_relation", seed)]
        clusters = sorted({row["cluster_id"] for row in m5})
        by_cluster = {cluster: [idx for idx, row in enumerate(m5) if row["cluster_id"] == cluster] for cluster in clusters}
        rng = np.random.default_rng(seed)
        deltas = []
        for _ in range(iterations):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            idxs = [idx for cluster in sampled for idx in by_cluster[cluster]]
            sampled_m2 = [m2[idx] for idx in idxs]
            sampled_m5 = [m5[idx] for idx in idxs]
            deltas.append(e1_score(sampled_m5) - e1_score(sampled_m2))
        arr = np.asarray(deltas, dtype=float)
        result[f"seed{seed}_M5_vs_M2"] = {"iterations": iterations, "cluster_count": len(clusters), "delta_mean": float(np.mean(arr)), "ci_lower": float(np.quantile(arr, 0.025)), "ci_upper": float(np.quantile(arr, 0.975))}
    return result


def stratified_q_derangement(rows_: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = [dict(row) for row in rows_]
    strata: dict[tuple[str, str, str, str], list[int]] = {}
    for idx, row in enumerate(rows_):
        key = (str(row.get("e1_subset")), str(row.get("source")), str(row.get("fraud_family_q_only") or row.get("fraud_family")), str(row.get("exp1_label")))
        strata.setdefault(key, []).append(idx)
    for indices in strata.values():
        if len(indices) < 2:
            continue
        perm = indices[:]
        for _ in range(20):
            shuffled_idx = rng.permutation(indices).tolist()
            if all(a != b for a, b in zip(indices, shuffled_idx)):
                perm = shuffled_idx
                break
        queries = [rows_[idx].get("user_query", "") for idx in perm]
        for idx, query in zip(indices, queries):
            out[idx]["user_query"] = query
    return out


def q_shuffle_rows(output_dir: Path, prefix: str, predictions: dict[tuple[str, int], list[dict]], seeds: list[int]) -> list[dict]:
    rows_out = []
    for seed in seeds:
        m5 = predictions[("M5_residual_relation", seed)]
        m2 = predictions[("M2_y_only", seed)]
        shuffled_rows = rows(output_dir / "predictions" / f"{prefix}_M5_q_shuffle_seed{seed}.jsonl")
        rows_out.append({"seed": seed, "M5_original": e1_score(m5), "M5_q_shuffle": e1_score(shuffled_rows), "M2_y_only": e1_score(m2), "original_minus_shuffle": e1_score(m5) - e1_score(shuffled_rows), "shuffle_minus_y": e1_score(shuffled_rows) - e1_score(m2)})
    return rows_out


def source_excluded_rows(predictions: dict[tuple[str, int], list[dict]], seeds: list[int]) -> list[dict]:
    out = []
    for seed in seeds:
        m5 = predictions[("M5_residual_relation", seed)]
        m2 = predictions[("M2_y_only", seed)]
        for source in sorted({str(row.get("source")) for row in m5}):
            a = [row for row in m5 if str(row.get("source")) != source]
            b = [row for row in m2 if str(row.get("source")) != source]
            out.append({"seed": seed, "excluded_source": source, "M5_E1": e1_score(a), "M2_E1": e1_score(b), "delta": e1_score(a) - e1_score(b)})
    return out


def metrics_by_stratum(predictions: dict[tuple[str, int], list[dict]]) -> list[dict]:
    out = []
    for (name, seed), pred_rows in predictions.items():
        for field in ("source", "fraud_family_q_only"):
            for value in sorted({str(row.get(field)) for row in pred_rows}):
                subset = [row for row in pred_rows if str(row.get(field)) == value]
                if subset:
                    out.append({"comparator": name, "seed": seed, "stratum_type": field, "stratum": value, **metric_dict(subset)})
    return out


def decide(stage: str, metric_rows: list[dict], modeldev_rows: list[dict], paired: dict, shuffle: list[dict], resource: dict, selected_lambdas: dict, config: dict) -> str:
    if stage == "smoke":
        return "E1_V6R1_SMOKE_PASS" if metric_rows and not any(np.isnan(float(row.get("macro_f1", 0))) for row in metric_rows) else "E1_V6R1_STOP"
    seeds = sorted({int(row["seed"]) for row in metric_rows})
    m = {(row["comparator"], int(row["seed"]), row["subset"]): row for row in metric_rows}
    md = {(row["comparator"], int(row["seed"]), row["subset"]): row for row in modeldev_rows}
    if stage == "pilot":
        gdev = config["gates"]["gdev"]
        y_dev = [float(md[("M2_y_only", seed, "E1_SCORE")]["macro_f1"]) for seed in seeds if ("M2_y_only", seed, "E1_SCORE") in md]
        m5_dev = [float(md[("M5_residual_relation", seed, "E1_SCORE")]["macro_f1"]) for seed in seeds if ("M5_residual_relation", seed, "E1_SCORE") in md]
        gdev_pass = bool(y_dev and m5_dev) and gdev["y_only_min"] <= np.mean(y_dev) <= gdev["y_only_max"] and np.mean(m5_dev) >= gdev["m5_score_min"] and np.mean(m5_dev) - np.mean(y_dev) >= gdev["delta_min"] and all(float(info["selected_lambda"]) > 0 for info in selected_lambdas.values())
        gate = config["gates"]["pilot"]
        m5_scores = [float(m[("M5_residual_relation", seed, "E1_SCORE")]["macro_f1"]) for seed in seeds]
        m2_scores = [float(m[("M2_y_only", seed, "E1_SCORE")]["macro_f1"]) for seed in seeds]
        deltas = [m5 - m2 for m5, m2 in zip(m5_scores, m2_scores)]
        ci_lower = min(value["ci_lower"] for value in paired.values()) if paired else -1.0
        checks = {
            "gdev": gdev_pass,
            "m5_mean": np.mean(m5_scores) >= gate["m5_mean_min"],
            "m5_worst": min(m5_scores) >= gate["m5_worst_min"],
            "delta_mean": np.mean(deltas) >= gate["delta_mean_min"],
            "ci_lower": ci_lower >= gate["ci_lower_min"],
            "positive_seeds": all(delta > 0 for delta in deltas),
            "q_only_lt_y_only": all(float(m[("M1_q_only", seed, "E1_SCORE")]["macro_f1"]) < float(m[("M2_y_only", seed, "E1_SCORE")]["macro_f1"]) for seed in seeds),
            "r1_q_auroc": all(gate["r1_q_auroc_min"] <= float(m[("M1_q_only", seed, "R1")].get("auroc", 0.5)) <= gate["r1_q_auroc_max"] for seed in seeds),
            "subset_mins": all(float(m[("M5_residual_relation", seed, "R1")]["macro_f1"]) >= gate["r1_m5_min"] and float(m[("M5_residual_relation", seed, "R2")]["macro_f1"]) >= gate["r2_m5_min"] and float(m[("M5_residual_relation", seed, "R3")]["macro_f1"]) >= gate["r3_m5_min"] for seed in seeds),
            "r3_noninferior": all(float(m[("M5_residual_relation", seed, "R3")]["macro_f1"]) - float(m[("M2_y_only", seed, "R3")]["macro_f1"]) >= gate["r3_noninferiority_min"] for seed in seeds),
            "q_shuffle": bool(shuffle) and min(row["original_minus_shuffle"] for row in shuffle) >= gate["q_shuffle_drop_min"] and max(abs(row["shuffle_minus_y"]) for row in shuffle) <= gate["q_shuffle_y_gap_max"],
            "risk": all(float(m[("M5_residual_relation", seed, "ALL")]["recall"]) >= gate["recall_min"] and float(m[("M5_residual_relation", seed, "ALL")]["fpr"]) <= gate["fpr_max"] and float(m[("M5_residual_relation", seed, "ALL")].get("auprc", 0)) >= gate["auprc_min"] for seed in seeds),
            "calibration": all(float(m[("M5_residual_relation", seed, "ALL")]["ece"]) <= gate["ece_max"] and float(m[("M5_residual_relation", seed, "ALL")]["brier"]) <= gate["brier_max"] for seed in seeds),
            "resource": bool(resource.get("passed")),
        }
        write_json(ROOT / "outputs" / "e1_relation_gate_v6r1" / "pilot" / "E1_V6R1_GATE_CHECKS.json", checks)
        return "E1_V6R1_FULL_READY" if all(checks.values()) else "E1_V6R1_STOP"
    return "E1_V6R1_FULL_COMPLETE"


def resource_summary(output_dir: Path, profiler: ResourceProfiler, config: dict, stage: str) -> dict:
    profile = profiler.finish()
    profile["hardware"] = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()}
    profile["deploy_bundle_mb"] = dir_size_mb(output_dir / "models")
    gate = config["gates"]["pilot"] if stage in {"smoke", "pilot"} else {"peak_ram_mb_max": 8192, "wall_minutes_max": 120}
    return resource_gate(profile, {"cpu_only": True, "peak_ram_mb_max": gate.get("peak_ram_mb_max", 8192), "g0_wall_time_minutes_max": gate.get("wall_minutes_max", 120), "artifact_mb_max": 500})


def stratum_weights(rows_: list[dict]) -> np.ndarray:
    counts = {subset: sum(1 for row in rows_ if row.get("e1_subset") == subset) for subset in ("R1", "R2", "R3")}
    return np.asarray([1.0 / max(counts.get(str(row.get("e1_subset")), 0), 1) for row in rows_], dtype=float) * len(rows_) / 3.0


def shuffled(rows_: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [rows_[int(idx)] for idx in rng.permutation(len(rows_))]


def rows(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def write_table(path: Path, rows_: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows_[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def write_failure_index(path: Path, predictions: dict[tuple[str, int], list[dict]]) -> None:
    rows_ = []
    for (name, seed), pred_rows in predictions.items():
        if name != "M5_residual_relation":
            continue
        for row in pred_rows:
            if row["gold_label"] != row["pred_label"]:
                rows_.append({"id": row["id"], "row_uid": row["row_uid"], "source": row["source"], "split": "test", "error_type": f"{row['gold_label']}_as_{row['pred_label']}", "score": row["unsafe_score"], "seed": seed})
    write_table(path, rows_)


def write_model_card(path: Path, lambdas: dict, stage: str) -> None:
    path.write_text(f"# E1 v6r1 Residual-Relation CPU Model\n\nStage: {stage}\n\nSelected lambdas:\n\n```json\n{json.dumps(lambdas, ensure_ascii=False, indent=2, default=json_default)}\n```\n", encoding="utf-8")


def write_report(path: Path, payload: dict, metric_rows: list[dict], paired: dict, shuffle: list[dict]) -> None:
    lines = ["# FraudDistill E1 v6r1 任务报告", "", f"- 决策：`{payload['decision']}`", f"- 阶段：`{payload['stage']}`", f"- Git commit：`{payload['git_commit']}`", "", "## 主结果", "", "| Model | Seed | R1 | R2 | R3 | E1-Score | Recall | FPR | AUPRC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for comparator in COMPARATORS:
        for seed in sorted({int(row["seed"]) for row in metric_rows}):
            vals = {(row["comparator"], int(row["seed"]), row["subset"]): row for row in metric_rows}
            if (comparator, seed, "E1_SCORE") not in vals:
                continue
            all_row = vals.get((comparator, seed, "ALL"), {})
            lines.append(f"| {comparator} | {seed} | {float(vals.get((comparator, seed, 'R1'), {}).get('macro_f1', 0)):.4f} | {float(vals.get((comparator, seed, 'R2'), {}).get('macro_f1', 0)):.4f} | {float(vals.get((comparator, seed, 'R3'), {}).get('macro_f1', 0)):.4f} | {float(vals[(comparator, seed, 'E1_SCORE')]['macro_f1']):.4f} | {float(all_row.get('recall', 0)):.4f} | {float(all_row.get('fpr', 0)):.4f} | {float(all_row.get('auprc', 0)):.4f} |")
    lines.extend(["", "## 统计与控制", "", f"- Bootstrap 文件：`E1_V6R1_PAIRED_E1_BOOTSTRAP.json`", f"- q-shuffle 文件：`E1_V6R1_Q_SHUFFLE.csv`", "", "本报告按 v6r1 协议自动生成；若决策为 STOP，不解封 formal test。"])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_g0_report(path: Path, payload: dict) -> None:
    census = payload["census"]
    r2 = census.get("r2_balance", {})
    lines = [
        "# FraudDistill E1 v6r1 G0r 数据与实现 Gate 报告",
        "",
        f"- 决策：`{payload['decision']}`",
        f"- Git commit：`{payload['git_commit']}`",
        "",
        "## G0r 摘要",
        "",
        f"- R1 groups：{census.get('r1_groups')}",
        f"- R2 groups：{census.get('r2_groups')}",
        f"- R3 rows：{census.get('r3_rows')}",
        f"- cross-split component leakage passed：{census.get('leakage', {}).get('passed')}",
        f"- same row_uid duplicates：{census.get('same_row_uid_duplicates')}",
        "",
        "## R2 Balance",
        "",
        f"- R2 balance passed：{r2.get('passed')}",
        f"- q nuisance SMD：{r2.get('q_selector_smd')}",
        f"- y nuisance SMD：{r2.get('y_selector_smd')}",
        f"- log answer length SMD：{r2.get('log_answer_length_smd')}",
        f"- refusal gap：{r2.get('refusal_gap')}",
        f"- largest source rate：{r2.get('largest_source_rate')}",
        f"- independent q-only AUROC：{r2.get('independent_q_probe_auc')}",
        f"- independent y-only AUROC：{r2.get('independent_y_probe_auc')}",
        "",
        "## 结论",
        "",
        "本阶段未进入 smoke/Pilot。根据 v6r1 文档，R2 可用 groups 低于 3,500 或 R2 balance 不通过时必须 STOP，不能为了结果叙事降低匹配质量或继续解封 Pilot/Formal。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{value.__class__.__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
