"""Evaluate budgeted cascade prediction files against gold (exp2 v2).

Usage:
  python scripts/eval_cascade_predictions.py --benchmark fraudr1_diag --glob 'cascade_pilot*.jsonl'
  python scripts/eval_cascade_predictions.py --benchmark all
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "exp2_prior_work_comparison"))

GOLD_JOIN = {
    "orbench": ("orbench/human_audit/human_audit_adjudicated.jsonl", "binary"),
}


def fix_gold(benchmark: str, rows: list) -> None:
    """Overwrite gold_binary in prediction rows from the adjudicated audit when available."""
    if benchmark not in GOLD_JOIN:
        return
    rel, key = GOLD_JOIN[benchmark]
    jpath = os.path.join(BASE, rel)
    if not os.path.exists(jpath):
        return
    join = {}
    with open(jpath, encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                join[str(j["id"])] = j.get(key)
            except Exception:
                pass
    for r in rows:
        g = join.get(str(r.get("id")))
        if g is not None:
            r["gold_binary"] = g


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def binary_metrics(rows):
    y_true = [r["gold_binary"] for r in rows if r.get("gold_binary") is not None]
    y_pred = [r.get("prediction_binary") for r in rows if r.get("gold_binary") is not None]
    n = len(y_true)
    if n == 0:
        return {"n": 0}
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    # AUPRC via simple rank-based computation
    scored = sorted(
        [(r.get("risk_score") or 0.0), t] for r, t in zip(
            [r for r in rows if r.get("gold_binary") is not None],
            y_true,
        )
    )
    pos = sum(1 for _, t in scored if t == 1)
    neg = n - pos
    auprc = 0.0
    if pos and neg:
        tp_r = 0
        fp_r = 0
        prev_score = None
        precs = []
        recs = []
        for s, t in sorted(scored, key=lambda x: -x[0]):
            if prev_score is not None and s != prev_score:
                precs.append(tp_r / (tp_r + fp_r) if tp_r + fp_r else 0.0)
                recs.append(tp_r / pos)
            tp_r += t
            fp_r += 1 - t
            prev_score = s
        precs.append(tp_r / (tp_r + fp_r) if tp_r + fp_r else 0.0)
        recs.append(tp_r / pos)
        auprc = 0.0
        for i in range(1, len(precs)):
            auprc += (recs[i] - recs[i - 1]) * precs[i]
    return {"n": n, "n_pos": pos, "acc": round(acc, 4), "prec": round(prec, 4),
            "recall": round(rec, 4), "macro_f1": round(f1, 4), "fpr": round(fpr, 4),
            "auprc": round(auprc, 4), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def usage_stats(rows):
    calls = sum(r.get("usage", {}).get("calls", 0) for r in rows)
    out = sum(r.get("usage", {}).get("output", 0) for r in rows)
    miss = sum(r.get("usage", {}).get("input_miss", 0) for r in rows)
    hit = sum(r.get("usage", {}).get("input_hit", 0) for r in rows)
    n = max(len(rows), 1)
    return {"calls": calls, "mean_calls": round(calls / n, 3),
            "mean_output_tokens": round(out / max(calls, 1), 1),
            "mean_input_miss": round(miss / n, 1), "mean_input_hit": round(hit / n, 1),
            "api_failed": sum(1 for r in rows if r.get("evaluation_status") == "api_failed"),
            "json_parse_fail": sum(1 for r in rows if r.get("evaluation_status") == "api_failed" and "parse" in str(r.get("reason", ""))),
            "routes": dict(collections.Counter(r.get("route") for r in rows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--glob", default="cascade_pilot*.jsonl")
    ap.add_argument("--dir", default="")
    args = ap.parse_args()

    if args.benchmark == "all":
        benches = ["fraudr1_diag", "orbench", "dna", "aegis2", "fraudr1"]
    else:
        benches = [args.benchmark]
    for b in benches:
        d = args.dir or os.path.join(BASE, b, "cascade_predictions")
        if not os.path.isdir(d):
            print(f"[{b}] no dir {d}")
            continue
        for path in sorted(glob.glob(os.path.join(d, args.glob))):
            rows = load(path)
            fix_gold(b, rows)
            rows = [r for r in rows if r.get("evaluation_status") != "invalid_qy"]
            m = binary_metrics(rows)
            u = usage_stats(rows)
            print(f"== {b} :: {os.path.basename(path)}")
            print("   metrics:", json.dumps(m, ensure_ascii=False))
            print("   usage:", json.dumps(u, ensure_ascii=False))


if __name__ == "__main__":
    main()
