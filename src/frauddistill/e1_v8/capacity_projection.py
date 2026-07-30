from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def wilson_lower(successes: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - adj) / denom)


def stratified_projection(rows: list[dict[str, Any]], planned_per_model_stage: dict[tuple[str, int], int]) -> dict[str, Any]:
    observed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        observed[(row["target_model"], int(row["stage_id"]))].append(row)
    parts = []
    total = 0.0
    for key, planned_n in sorted(planned_per_model_stage.items()):
        cell = observed.get(key, [])
        failures = sum(1 for row in cell if row.get("x_consensus_state") == "FAILURE")
        lower = wilson_lower(failures, len(cell))
        projected = planned_n * lower
        total += projected
        parts.append({"target_model": key[0], "stage_id": key[1], "observed_n": len(cell), "failures": failures, "wilson_lower": lower, "planned_n": planned_n, "projected_lower": projected})
    return {"projection_lower": total, "cells": parts}


def cluster_bootstrap_diff(rows: list[dict[str, Any]], group_key: str, left_field: str, right_field: str, iterations: int = 10000, seed: int = 20260729) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    keys = sorted(groups)
    if not keys:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rnd = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = [rnd.choice(keys) for _ in keys]
        left = []
        right = []
        for key in sample:
            for row in groups[key]:
                left.append(float(row.get(left_field, 0)))
                right.append(float(row.get(right_field, 0)))
        values.append((sum(left) / len(left)) - (sum(right) / len(right)))
    values.sort()
    return {"mean": sum(values) / len(values), "ci_low": values[int(0.025 * (len(values) - 1))], "ci_high": values[int(0.975 * (len(values) - 1))]}
