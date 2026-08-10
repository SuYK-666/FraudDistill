# -*- coding: utf-8 -*-
"""v4 statistics: cluster bootstrap, paired McNemar + Holm, per-stratum metrics."""
from __future__ import annotations

import collections
import random
from typing import Any

from frauddistill.e1_v10.metrics import binary_metrics, holm_adjust


def eval_rows(anchor: list[dict[str, Any]], scores: list[float], threshold: float, mode: str) -> list[dict[str, Any]]:
    out = []
    for r, s in zip(anchor, scores):
        out.append({
            "response_id": r["response_id"],
            "family_id": r.get("family_id", ""),
            "stratum": r.get("stratum", ""),
            "gold": int(r["gold_central"]),
            "pred": int(s >= threshold),
            "score": float(s),
            "mode": mode,
        })
    return out


def delta_joint(by_mode: dict[str, list[dict[str, Any]]], anchor) -> dict[str, Any]:
    best_single = max(binary_metrics(by_mode["q_only"])["macro_f1"], binary_metrics(by_mode["y_only"])["macro_f1"])
    qy = binary_metrics(by_mode["q_y"])["macro_f1"]
    return {"q_y": qy, "best_single": best_single, "delta": qy - best_single}


def cluster_bootstrap_delta(by_mode: dict[str, list[dict[str, Any]]], iterations: int = 10000, seed: int = 13) -> dict[str, Any]:
    rng = random.Random(seed)
    fams = sorted({r["family_id"] for r in by_mode["q_y"]})
    rows_by_fam = {f: {m: [r for r in rows if r["family_id"] == f] for m, rows in by_mode.items()} for f in fams}
    vals = []
    for _ in range(iterations):
        chosen = [rng.choice(fams) for _ in fams]
        def mf(m):
            rows = [r for f in chosen for r in rows_by_fam[f].get(m, [])]
            return binary_metrics(rows)["macro_f1"] if rows else 0.0
        vals.append(mf("q_y") - max(mf("q_only"), mf("y_only")))
    vals.sort()
    point = delta_joint(by_mode, None)["delta"]
    return {
        "point": point,
        "ci95": [vals[int(0.025 * (iterations - 1))], vals[int(0.975 * (iterations - 1))]],
        "p_below_zero": sum(1 for v in vals if v <= 0) / iterations,
        "iterations": iterations,
    }


def paired_mcnemar(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    by_left = {r["response_id"]: r for r in left}
    by_right = {r["response_id"]: r for r in right}
    b = c = 0
    for rid in set(by_left) & set(by_right):
        l, r_ = by_left[rid]["pred"], by_right[rid]["pred"]
        g = by_left[rid]["gold"]
        if l == g and r_ != g:
            b += 1
        elif l != g and r_ == g:
            c += 1
    from scipy.stats import binomtest
    if b + c == 0:
        p = 1.0
    else:
        p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    return {"b": b, "c": c, "n_discordant": b + c, "p": p}


def per_stratum(by_mode: dict[str, list[dict[str, Any]]], strata: list[str]) -> dict[str, Any]:
    out = {}
    for st in strata:
        out[st] = {m: binary_metrics([r for r in rows if r["stratum"] == st]) for m, rows in by_mode.items()}
    return out


def aggregate_results(anchor, preds_by_mode: dict[str, list[dict[str, Any]]], iterations: int = 10000, seed: int = 13) -> dict[str, Any]:
    metrics = {m: binary_metrics(rows) for m, rows in preds_by_mode.items()}
    dj = delta_joint(preds_by_mode, anchor)
    cb = cluster_bootstrap_delta(preds_by_mode, iterations=iterations, seed=seed)
    mcn_qy_vs_y = paired_mcnemar(preds_by_mode["q_y"], preds_by_mode["y_only"])
    mcn_qy_vs_q = paired_mcnemar(preds_by_mode["q_y"], preds_by_mode["q_only"])
    mcn_qy_vs_wrong = paired_mcnemar(preds_by_mode["q_y"], preds_by_mode["wrong_q_y"])
    holm = holm_adjust([mcn_qy_vs_y["p"], mcn_qy_vs_q["p"]])
    return {
        "metrics": metrics,
        "delta_joint": dj,
        "bootstrap": cb,
        "mcnemar": {"qy_vs_y": mcn_qy_vs_y, "qy_vs_q": mcn_qy_vs_q, "qy_vs_wrong": mcn_qy_vs_wrong},
        "holm": {"p_adj": holm},
        "strata": per_stratum(preds_by_mode, sorted({r["stratum"] for r in anchor})),
        "scientific_gate": {
            "delta>0": dj["delta"] > 0,
            "ci_lower>0": cb["ci95"][0] > 0,
            "holm_p<0.05": bool(holm) and all(p < 0.05 for p in holm),
            "qy>wrong": metrics["q_y"]["macro_f1"] > metrics["wrong_q_y"]["macro_f1"],
        },
    }
