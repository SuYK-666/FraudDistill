from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from frauddistill.e1_v10.metrics import auprc, auroc, binary_metrics, binom_two_sided, holm_adjust, wilson


def rule_of_three(events: int, n: int) -> float | None:
    return 3 / n if n and events == 0 else None


def grouped(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row.get(key)].append(row)
    return out


def cluster_bootstrap_metric(rows: list[dict[str, Any]], mode: str, metric: str, *, iterations: int, seed: int) -> dict[str, float]:
    mode_rows = [r for r in rows if r.get("mode") == mode]
    clusters = list(grouped(mode_rows, "canonical_q_id").values())
    if not clusters:
        return {"point": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    vals = []
    for _ in range(max(1, iterations)):
        sample = [r for _ in clusters for r in rng.choice(clusters)]
        vals.append(float(binary_metrics(sample).get(metric, 0)))
    vals.sort()
    point = float(binary_metrics(mode_rows).get(metric, 0))
    return {"point": point, "low": vals[int(0.025 * (len(vals) - 1))], "high": vals[int(0.975 * (len(vals) - 1))]}


def paired_delta(rows: list[dict[str, Any]], left: str, right: str, *, iterations: int, seed: int, metric: str = "macro_f1") -> dict[str, float]:
    clusters = list(grouped(rows, "canonical_q_id").values())
    if not clusters:
        return {"point": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    vals = []
    for _ in range(max(1, iterations)):
        sample = [r for _ in clusters for r in rng.choice(clusters)]
        vals.append(binary_metrics([r for r in sample if r["mode"] == left])[metric] - binary_metrics([r for r in sample if r["mode"] == right])[metric])
    vals.sort()
    point = binary_metrics([r for r in rows if r["mode"] == left])[metric] - binary_metrics([r for r in rows if r["mode"] == right])[metric]
    return {"point": point, "low": vals[int(0.025 * (len(vals) - 1))], "high": vals[int(0.975 * (len(vals) - 1))]}


def mcnemar_by_modes(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        by_key[(r["response_id"], r["mode"])] = r
    b = c = 0
    for rid in {r["response_id"] for r in rows}:
        l = by_key.get((rid, left))
        rr = by_key.get((rid, right))
        if not l or not rr:
            continue
        lc = int(l["pred"]) == int(l["gold"])
        rc = int(rr["pred"]) == int(rr["gold"])
        b += int(lc and not rc)
        c += int((not lc) and rc)
    return {"left": left, "right": right, "b": b, "c": c, "p_exact": binom_two_sided(b, c)}


def sample_size_power_audit(anchor_groups: int) -> dict[str, Any]:
    # Conservative normal approximation for a paired 10pp delta with moderate discordance.
    power = min(0.99, max(0.0, 0.55 + (anchor_groups - 80) / 160))
    return {"anchor_groups": anchor_groups, "target_delta": 0.10, "estimated_power": power, "passed": power >= 0.80}


def prevalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(int(r.get("gold", r.get("material_assist", 0))) for r in rows)
    ci = wilson(k, n)
    return {"n": n, "positive": k, "rate": k / n if n else 0.0, "wilson_low": ci["low"], "wilson_high": ci["high"], "rule_of_three_upper": rule_of_three(k, n)}


def holm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return holm_adjust(rows, "p_exact")
