from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from frauddistill.e1_v10.metrics import binary_metrics, binom_two_sided, gwet_ac1, holm_adjust, positive_agreement, wilson


def grouped(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    out = defaultdict(list)
    for row in rows:
        out[row.get(key)].append(row)
    return out


def bootstrap_metric(rows: list[dict[str, Any]], mode: str, *, cluster_key: str, metric: str, iterations: int, seed: int) -> dict[str, float]:
    mode_rows = [r for r in rows if r.get("mode") == mode]
    clusters = list(grouped(mode_rows, cluster_key).values())
    if not clusters:
        return {"point": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    vals = []
    for _ in range(iterations):
        sample = [r for _ in clusters for r in rng.choice(clusters)]
        vals.append(binary_metrics(sample)[metric])
    vals.sort()
    return {"point": binary_metrics(mode_rows)[metric], "low": vals[int(0.025 * (iterations - 1))], "high": vals[int(0.975 * (iterations - 1))]}


def paired_delta(rows: list[dict[str, Any]], left: str, right: str, *, cluster_key: str, iterations: int, seed: int) -> dict[str, float]:
    clusters = list(grouped(rows, cluster_key).values())
    rng = random.Random(seed)
    vals = []
    for _ in range(iterations):
        sample = [r for _ in clusters for r in rng.choice(clusters)]
        vals.append(binary_metrics([r for r in sample if r["mode"] == left])["macro_f1"] - binary_metrics([r for r in sample if r["mode"] == right])["macro_f1"])
    vals.sort()
    point = binary_metrics([r for r in rows if r["mode"] == left])["macro_f1"] - binary_metrics([r for r in rows if r["mode"] == right])["macro_f1"]
    return {"point": point, "low": vals[int(0.025 * (iterations - 1))], "high": vals[int(0.975 * (iterations - 1))]}


def mcnemar(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    by = {(r["response_id"], r["mode"]): r for r in rows}
    b = c = 0
    for rid in {r["response_id"] for r in rows}:
        l = by.get((rid, left))
        rr = by.get((rid, right))
        if not l or not rr:
            continue
        lc = int(l["pred"]) == int(l["gold"])
        rc = int(rr["pred"]) == int(rr["gold"])
        b += int(lc and not rc)
        c += int((not lc) and rc)
    return {"left": left, "right": right, "b": b, "c": c, "p_exact": binom_two_sided(b, c)}


def agreement(vote_pairs: list[tuple[int, int]]) -> dict[str, float]:
    overall = sum(a == b for a, b in vote_pairs) / max(1, len(vote_pairs))
    return {"overall_binary_agreement": overall, "positive_agreement": positive_agreement(vote_pairs), "gwet_ac1": gwet_ac1(vote_pairs)}


def prevalence_table(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    out = []
    for group, bucket in grouped(rows, group_key).items():
        n = len(bucket)
        k = sum(int(r.get("gold", 0)) for r in bucket)
        ci = wilson(k, n)
        out.append({"group": group, "n": n, "positive": k, "rate": k / n if n else 0, "wilson_low": ci["low"], "wilson_high": ci["high"]})
    return sorted(out, key=lambda r: str(r["group"]))


def holm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return holm_adjust(rows, "p_exact")
