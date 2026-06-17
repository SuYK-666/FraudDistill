from __future__ import annotations

from frauddistill.eval.rule_baseline import predict_rule


def predict_with_rule_fallback(rows: list[dict]) -> list[dict]:
    return [predict_rule(row) for row in rows]
