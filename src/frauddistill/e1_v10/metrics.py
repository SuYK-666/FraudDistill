from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return empty_metrics()
    y = [int(r["gold"]) for r in rows]
    p = [int(r["pred"]) for r in rows]
    s = [float(r.get("score", p[i])) for i, r in enumerate(rows)]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, p))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1p = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    nprec = tn / (tn + fn) if tn + fn else 0.0
    nrec = tn / (tn + fp) if tn + fp else 0.0
    f1n = 2 * nprec * nrec / (nprec + nrec) if nprec + nrec else 0.0
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "macro_f1": (f1p + f1n) / 2,
        "balanced_accuracy": (rec + nrec) / 2,
        "accuracy": (tp + tn) / len(rows),
        "precision": prec,
        "recall": rec,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "auroc": auroc(y, s),
        "auprc": auprc(y, s),
        "brier": sum((a - b) ** 2 for a, b in zip(y, s)) / len(y),
        "ece": ece(y, s),
    }


def empty_metrics() -> dict[str, float]:
    return {
        "n": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "brier": 0.0,
        "ece": 0.0,
    }


def auroc(y: list[int], s: list[float]) -> float:
    pos = [score for yy, score in zip(y, s) if yy == 1]
    neg = [score for yy, score in zip(y, s) if yy == 0]
    if not pos or not neg:
        return 0.0
    wins = sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auprc(y: list[int], s: list[float]) -> float:
    order = sorted(zip(s, y), key=lambda x: x[0], reverse=True)
    total_pos = sum(y)
    if total_pos == 0:
        return 0.0
    tp = 0
    fp = 0
    points = []
    i = 0
    while i < len(order):
        score = order[i][0]
        group = []
        while i < len(order) and order[i][0] == score:
            group.append(order[i][1])
            i += 1
        tp += sum(1 for yy in group if yy)
        fp += sum(1 for yy in group if not yy)
        points.append((tp / total_pos, tp / (tp + fp)))
    prev_r = 0.0
    area = 0.0
    for recall, precision in points:
        area += (recall - prev_r) * precision
        prev_r = recall
    return area


def ece(y: list[int], s: list[float], bins: int = 10) -> float:
    if not y:
        return 0.0
    out = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, score in enumerate(s) if lo <= score < hi or (i == bins - 1 and score == 1.0)]
        if idx:
            acc = sum(y[j] for j in idx) / len(idx)
            conf = sum(s[j] for j in idx) / len(idx)
            out += len(idx) / len(y) * abs(acc - conf)
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    if n == 0:
        return {"low": 0.0, "high": 0.0}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {"low": max(0.0, centre - half), "high": min(1.0, centre + half)}


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    agree = sum(a == b for a, b in pairs) / len(pairs)
    pa = sum(a for a, _ in pairs) / len(pairs)
    pb = sum(b for _, b in pairs) / len(pairs)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (agree - pe) / (1 - pe) if pe != 1 else 1.0


def gwet_ac1(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    agree = sum(a == b for a, b in pairs) / len(pairs)
    p = (sum(a for a, _ in pairs) + sum(b for _, b in pairs)) / (2 * len(pairs))
    pe = 2 * p * (1 - p)
    return (agree - pe) / (1 - pe) if pe != 1 else 1.0


def positive_agreement(pairs: list[tuple[int, int]]) -> float:
    both = sum(a == 1 and b == 1 for a, b in pairs)
    a_pos = sum(a == 1 for a, _ in pairs)
    b_pos = sum(b == 1 for _, b in pairs)
    return (2 * both / (a_pos + b_pos)) if a_pos + b_pos else 1.0


def binom_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) * (0.5**n) for i in range(k + 1)) * 2
    return min(1.0, prob)


def holm_adjust(rows: list[dict[str, Any]], p_key: str = "p_exact") -> list[dict[str, Any]]:
    ordered = sorted(enumerate(rows), key=lambda x: x[1].get(p_key, 1.0))
    adjusted = [1.0 for _ in rows]
    prev = 0.0
    m = len(rows)
    for rank, (idx, row) in enumerate(ordered):
        value = min(1.0, (m - rank) * row.get(p_key, 1.0))
        prev = max(prev, value)
        adjusted[idx] = prev
    return [{**row, "holm_p": adjusted[i]} for i, row in enumerate(rows)]


def groupby(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(k) for k in keys)].append(row)
    return out


def cluster_bootstrap_delta(
    rows: list[dict[str, Any]],
    cluster_key: str,
    left_mode: str,
    right_mode: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    clusters = list(groupby(rows, cluster_key).values())
    if not clusters:
        return {"point": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    vals = []
    for _ in range(max(1, iterations)):
        sample = [r for c in (rng.choice(clusters) for _ in clusters) for r in c]
        left = binary_metrics([r for r in sample if r.get("mode") == left_mode])["macro_f1"]
        right = binary_metrics([r for r in sample if r.get("mode") == right_mode])["macro_f1"]
        vals.append(left - right)
    vals.sort()
    point = binary_metrics([r for r in rows if r.get("mode") == left_mode])["macro_f1"] - binary_metrics([r for r in rows if r.get("mode") == right_mode])["macro_f1"]
    return {"point": point, "low": vals[int(0.025 * (len(vals) - 1))], "high": vals[int(0.975 * (len(vals) - 1))]}
