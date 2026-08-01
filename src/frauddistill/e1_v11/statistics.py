from __future__ import annotations

from collections import Counter
from typing import Any

from frauddistill.e1_v10.metrics import binary_metrics, cohen_kappa, gwet_ac1, positive_agreement, wilson


def gold_quality(pairs: list[tuple[int, int]], expected_tasks: int, valid_tasks: int, uncertain_n: int = 0) -> dict[str, Any]:
    agree = sum(1 for a, b in pairs if a == b)
    table = Counter(pairs)
    n = len(pairs)
    return {
        "expected_tasks": expected_tasks,
        "valid_tasks": valid_tasks,
        "completion": valid_tasks / expected_tasks if expected_tasks else 0,
        "valid_schema": valid_tasks / expected_tasks if expected_tasks else 0,
        "paired_n": n,
        "observed_agreement": agree / n if n else 0,
        "positive_agreement": positive_agreement(pairs),
        "kappa": cohen_kappa(pairs),
        "pabak": (2 * agree / n - 1) if n else 0,
        "gwet_ac1": gwet_ac1(pairs),
        "uncertain_rate": uncertain_n / n if n else 0,
        "cross_table": {
            "a_safe_b_safe": table[(0, 0)],
            "a_safe_b_risk": table[(0, 1)],
            "a_risk_b_safe": table[(1, 0)],
            "a_risk_b_risk": table[(1, 1)],
        },
    }


def gate_gold_quality(q: dict[str, Any]) -> bool:
    return (
        q.get("completion", 0) >= 0.995
        and q.get("valid_schema", 0) >= 0.990
        and q.get("observed_agreement", 0) >= 0.90
        and max(q.get("pabak", 0), q.get("gwet_ac1", 0)) >= 0.80
        and q.get("uncertain_rate", 1) <= 0.05
    )


def rate_ci(events: int, n: int) -> dict[str, Any]:
    ci = wilson(events, n)
    return {"events": events, "n": n, "rate": events / n if n else 0, "wilson95": ci, "events_per_1k": 1000 * events / n if n else 0}


def mode_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return binary_metrics(rows)
