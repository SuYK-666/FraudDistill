from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Iterable

from frauddistill.e1_v10.metrics import binom_two_sided, groupby, wilson


def prevalence(rows: list[dict[str, Any]], endpoint: str = "gold_central") -> dict[str, Any]:
    valid = [r for r in rows if int(r.get(endpoint, -1)) >= 0]
    k = sum(1 for r in valid if int(r[endpoint]) == 1)
    n = len(valid)
    return {
        "n": n,
        "positive": k,
        "rate": k / n if n else 0.0,
        "wilson_95": wilson(k, n) if n else {"low": 0.0, "high": 0.0},
    }


def stratified(rows: list[dict[str, Any]], key: str, endpoint: str = "gold_central") -> list[dict[str, Any]]:
    out = []
    for value, group in sorted(groupby(rows, key).items()):
        p = prevalence(group, endpoint)
        out.append({"stratum": str(value), **p})
    return out


def model_setting_language(rows: list[dict[str, Any]], endpoint: str = "gold_central") -> list[dict[str, Any]]:
    out = []
    for (model, setting, language), group in sorted(groupby(rows, "target_provider", "scenario", "language").items()):
        p = prevalence(group, endpoint)
        out.append({"target_model": str(model), "setting": str(setting), "language": str(language), **p})
    return out


def sensitivity_endpoints(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {name: prevalence(rows, endpoint) for name, endpoint in [("lower", "gold_lower"), ("central", "gold_central"), ("upper", "gold_upper")]}


def mcnemar_paired(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        if int(row.get("gold_central", -1)) < 0:
            continue
        paired[row["prompt_instance_id"]][row["target_provider"]] = int(row["gold_central"])
    b = 0
    c = 0
    for values in paired.values():
        if set(values) != {"qwen", "deepseek"}:
            continue
        if values["qwen"] == 1 and values["deepseek"] == 0:
            b += 1
        if values["qwen"] == 0 and values["deepseek"] == 1:
            c += 1
    qwen = sum(1 for v in paired.values() if v.get("qwen") == 1)
    ds = sum(1 for v in paired.values() if v.get("deepseek") == 1)
    n_pairs = sum(1 for v in paired.values() if set(v) == {"qwen", "deepseek"})
    return {
        "n_pairs": n_pairs,
        "qwen_positive": qwen,
        "deepseek_positive": ds,
        "qwen_only_positive": b,
        "deepseek_only_positive": c,
        "both_positive": sum(1 for v in paired.values() if v.get("qwen") == 1 and v.get("deepseek") == 1),
        "p_exact_mcnemar": binom_two_sided(b, c),
    }


def cluster_bootstrap_risk_diff(
    rows: list[dict[str, Any]],
    cluster_key: str = "canonical_case_id",
    iterations: int = 10000,
    seed: int = 20260802,
) -> dict[str, Any]:
    valid = [r for r in rows if int(r.get("gold_central", -1)) >= 0]
    clusters = list(groupby(valid, cluster_key).values())
    if not clusters:
        return {"point": 0.0, "low": 0.0, "high": 0.0, "n_clusters": 0}

    def rate_diff(sample: list[dict[str, Any]]) -> float:
        q = [r for r in sample if r["target_provider"] == "qwen"]
        d = [r for r in sample if r["target_provider"] == "deepseek"]
        rq = sum(1 for r in q if r["gold_central"] == 1) / len(q) if q else 0.0
        rd = sum(1 for r in d if r["gold_central"] == 1) / len(d) if d else 0.0
        return rq - rd

    rng = random.Random(seed)
    vals = []
    for _ in range(max(1, iterations)):
        sample = [r for c in (rng.choice(clusters) for _ in clusters) for r in c]
        vals.append(rate_diff(sample))
    vals.sort()
    q = [r for r in valid if r["target_provider"] == "qwen"]
    d = [r for r in valid if r["target_provider"] == "deepseek"]
    rq = sum(1 for r in q if r["gold_central"] == 1) / len(q) if q else 0.0
    rd = sum(1 for r in d if r["gold_central"] == 1) / len(d) if d else 0.0
    return {
        "point_risk_diff": rq - rd,
        "qwen_rate": rq,
        "deepseek_rate": rd,
        "low_95": vals[int(0.025 * (len(vals) - 1))],
        "high_95": vals[int(0.975 * (len(vals) - 1))],
        "n_clusters": len(clusters),
        "iterations": iterations,
    }


def a_behavior_stats(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    valid = [r for r in rows if int(r.get("gold_central", -1)) >= 0]
    return {
        "n_with_gold": len(valid),
        "n_rows": len(rows),
        "overall": {endpoint: prevalence(valid, f"gold_{endpoint}") for endpoint in ["lower", "central", "upper"]},
        "by_model": stratified(valid, "target_provider"),
        "by_setting": stratified(valid, "scenario"),
        "by_language": stratified(valid, "language"),
        "by_category": stratified(valid, "fraud_category"),
        "model_setting_language": model_setting_language(valid),
        "mcnemar_qwen_vs_deepseek": mcnemar_paired(valid),
        "cluster_bootstrap_risk_diff": cluster_bootstrap_risk_diff(valid, iterations=int(config["statistics"]["bootstrap_iterations"]), seed=int(config["experiment"]["seed"])),
        "note": "exploratory stratifications require FDR/Holm correction; main table uses central endpoint.",
    }
