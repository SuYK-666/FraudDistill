"""Oracle audit-budget sensitivity: reference-label replacement upper bound."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from frauddistill.e4e5_v2.cluster_bootstrap import exact_mcnemar, paired_cluster_bootstrap
from frauddistill.supplementary.common import REPO, binary_metrics, read_jsonl, write_json

ROOT = REPO / "experiments" / "lighteval_supplementary"
OUT = ROOT / "results" / "ablation_oracle" / "budget_sensitivity"
THRESHOLD = 0.2323
RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def main() -> int:
    rows = read_jsonl(ROOT / "prepared" / "raw" / "relation_test.jsonl")
    preds = {}
    for raw in (REPO / "experiments/exp6_balanced_multi_api/student/predictions_all.jsonl").open(encoding="utf-8"):
        item = json.loads(raw)
        if item.get("split") == "test" and item.get("relation") == "relation":
            preds[f"{item['slot']}::{item['prompt_id']}"] = item
    keys = [f"{r['slot']}::{r['prompt_id']}" for r in rows]
    gold = np.asarray([int(r["binary_label"] == "unsafe") for r in rows], dtype=int)
    family = [r["family_id"] for r in rows]
    base = np.asarray([int(preds[key]["risk_score"] >= THRESHOLD) for key in keys], dtype=int)
    ranked = sorted(keys, key=lambda key: (abs(preds[key]["risk_score"] - THRESHOLD), key))
    index = {key: i for i, key in enumerate(keys)}
    records = []
    policies = {}
    for rate in RATES:
        k = math.floor(rate * len(keys))
        pred = base.copy()
        for key in ranked[:k]:
            pred[index[key]] = gold[index[key]]
        metrics = binary_metrics(gold.tolist(), [p for p in pred], pred=pred.tolist(), threshold=0.5)
        item = {
            "rate": rate,
            "k": k,
            "metrics": metrics,
            "errors": metrics["fn"] + metrics["fp"],
            "delta_f1_vs_A2": None,
            "relative_f1_gain_vs_A2": None,
            "net_corrections": 0,
            "rows_per_net_correction": None,
        }
        if rate == 0:
            policies["A2"] = item
        else:
            base_errors = policies["A2"]["errors"]
            item["delta_f1_vs_A2"] = round(metrics["macro_f1"] - policies["A2"]["metrics"]["macro_f1"], 6)
            item["relative_f1_gain_vs_A2"] = round(
                (metrics["macro_f1"] - policies["A2"]["metrics"]["macro_f1"]) /
                policies["A2"]["metrics"]["macro_f1"], 6)
            item["net_corrections"] = base_errors - item["errors"]
            item["rows_per_net_correction"] = round(k / max(item["net_corrections"], 1), 3)
            item["mcnemar_vs_A2"] = exact_mcnemar(pred, base, gold)
            item["bootstrap_vs_A2"] = paired_cluster_bootstrap(
                gold, np.zeros(len(gold)), np.zeros(len(gold)), (0.5, 0.5), family,
                replicates=10000, seed=20260811, metric="macro_f1", pred_a=pred, pred_b=base)
            policies[f"{int(round(rate * 100))}pct"] = item
        records.append(item)
    previous = None
    for item in records:
        if item["rate"] == 0:
            previous = item
            continue
        k = item["k"]
        pred = base.copy()
        for key in ranked[:k]:
            pred[index[key]] = gold[index[key]]
        prev_k = previous["k"]
        prev_pred = base.copy()
        for key in ranked[:prev_k]:
            prev_pred[index[key]] = gold[index[key]]
        item["adjacent_bootstrap"] = paired_cluster_bootstrap(
            gold, np.zeros(len(gold)), np.zeros(len(gold)), (0.5, 0.5), family,
            replicates=10000, seed=20260812 + k, metric="macro_f1", pred_a=pred, pred_b=prev_pred)
        previous = item
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "extended_oracle_budget.json", {"threshold": THRESHOLD, "policies": policies})
    lines = [
        "# Extended Oracle Audit-Budget Sensitivity",
        "",
        "> Reference-label replacement simulation only; not real external API audit.",
        "",
        "| Audit rate | K | Macro-F1 | Recall | FPR | Delta F1 vs A2 | 95% bootstrap CI vs A2 | McNemar p vs A2 | Adjacent Delta F1 | Adjacent 95% CI | Relative F1 gain | Net corrections | Rows / correction |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for item in records:
        m = item["metrics"]
        if item["rate"] == 0:
            lines.append(f"| 0% | 0 | {m['macro_f1']:.4f} | {m['recall']:.4f} | {m['fpr']:.4f} | +0.0000 | - | - | - | - | +0.00% | +0 | - |")
        else:
            b = item["bootstrap_vs_A2"]; a = item["adjacent_bootstrap"]
            lines.append(
                f"| {item['rate']:.0%} | {item['k']} | {m['macro_f1']:.4f} | {m['recall']:.4f} | "
                f"{m['fpr']:.4f} | {item['delta_f1_vs_A2']:+.4f} | "
                f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {item['mcnemar_vs_A2']['p_exact']:.2e} | "
                f"{a['mean_diff']:+.4f} | [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] | "
                f"{item['relative_f1_gain_vs_A2']:+.2%} | {item['net_corrections']:+d} | {item['rows_per_net_correction']:.2f} |"
            )
    (OUT / "table_extended_oracle_budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "READY", "output": str(OUT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
