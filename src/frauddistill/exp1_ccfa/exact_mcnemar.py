from __future__ import annotations

from scipy.stats import binomtest


def exact_mcnemar(y_true: list[str], pred_a: list[str], pred_b: list[str]) -> dict[str, float | int]:
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a, and pred_b must have the same length")
    b = 0
    c = 0
    for gold, a, b_pred in zip(y_true, pred_a, pred_b):
        a_correct = a == gold
        b_correct = b_pred == gold
        if a_correct and not b_correct:
            b += 1
        elif b_correct and not a_correct:
            c += 1
    n = b + c
    p_value = 1.0 if n == 0 else binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "discordant": n, "p_value": float(p_value)}
