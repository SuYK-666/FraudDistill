# -*- coding: utf-8 -*-
"""E1 clean-anchor sensitivity: offline audit of cross-split near-duplicate y.

Excludes Anchor rows whose y shares an 80-char normalized prefix (or is an
exact normalized duplicate) with any model_dev/calibration y, then recomputes
q_only / y_only / q_y / wrong_q_y metrics from the frozen predictions on the
clean subset. Reports per-seed mean +/- sd, Delta_joint (point), family-cluster
bootstrap 95% CI (Macro-F1 formulation) and the sensitivity gates.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v3.io import norm
from frauddistill.e1_final_v4.judge_views import view_label
from frauddistill.e1_final_v4.stats import cluster_bootstrap_delta, delta_joint, paired_mcnemar
from frauddistill.e1_v10.metrics import binary_metrics

DATA = ROOT / "data" / "prepared" / "e1_final_triad_v4"
SEEDS = [13, 17, 23, 42, 20260810]
VIEWS = ["q_only", "y_only", "q_y", "wrong_q_y"]


def read_jsonl(p: pathlib.Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    dev = read_jsonl(DATA / "E1_V4_PANEL_MODEL_DEV.jsonl")
    cal = read_jsonl(DATA / "E1_V4_PANEL_CALIBRATION.jsonl")
    anchor = read_jsonl(DATA / "E1_V4_PANEL_ANCHOR.jsonl")
    local = json.loads((DATA / "E1_V4_ANCHOR_LOCAL_PREDS.json").read_text(encoding="utf-8"))
    votes = read_jsonl(DATA / "E1_V4_ANCHOR_VIEW_VOTES.jsonl")

    # near-duplicate y keys from dev + calibration
    dev_cal = dev + cal
    dev_cal_y_prefix = {norm(r.get("y_private") or "")[:80] for r in dev_cal if (r.get("y_private") or "").strip()}
    dev_cal_y_full = {norm(r.get("y_private") or "") for r in dev_cal if (r.get("y_private") or "").strip()}
    excluded = []
    for r in anchor:
        y = norm(r.get("y_private") or "")
        if not y:
            continue
        if y in dev_cal_y_full or y[:80] in dev_cal_y_prefix:
            excluded.append(r["response_id"])
    excl = set(excluded)
    keep = [r for r in anchor if r["response_id"] not in excl]
    print(f"anchor n={len(anchor)} | near-dup y excluded={len(excl)} | clean n={len(keep)}")

    from collections import Counter
    print("excluded by stratum:", dict(Counter(r["stratum"] for r in anchor if r["response_id"] in excl)))
    print("excluded by label:", dict(Counter(int(r["gold_central"]) for r in anchor if r["response_id"] in excl)))
    print("clean by label:", dict(Counter(int(r["gold_central"]) for r in keep)))

    rid_keep = set(keep)
    out: dict = {
        "n_anchor": len(anchor),
        "n_excluded_near_dup_y": len(excl),
        "n_clean": len(keep),
        "exclusion_rule": "y exact-normalized match OR normalized 80-char prefix match vs model_dev/calibration y",
        "excluded_by_stratum": dict(Counter(r["stratum"] for r in anchor if r["response_id"] in excl)),
        "excluded_by_label": dict(Counter(int(r["gold_central"]) for r in anchor if r["response_id"] in excl)),
        "clean_by_label": dict(Counter(int(r["gold_central"]) for r in keep)),
    }

    # ---- M1: per-seed metrics on clean subset
    m1 = {}
    for view in VIEWS:
        per_seed = []
        for seed in SEEDS:
            idx = SEEDS.index(seed)
            rows = [r for r in local[view][idx]["rows"] if r["response_id"] in rid_keep]
            per_seed.append(binary_metrics(rows))
        m1[view] = {
            "macro_f1_mean": st.mean(x["macro_f1"] for x in per_seed),
            "macro_f1_sd": st.stdev(x["macro_f1"] for x in per_seed),
            "per_seed": [{"seed": s, "macro_f1": x["macro_f1"], "auroc": x["auroc"], "auprc": x["auprc"], "recall": x["recall"], "fpr": x["fpr"], "n": x["n"]} for s, x in zip(SEEDS, per_seed)],
        }
    out["m1_clean"] = m1

    # ---- stats on seed-0 rows (consistent with phase_b_stats design)
    by_view = {v: [r for r in local[v][0]["rows"] if r["response_id"] in rid_keep] for v in VIEWS}
    dj = delta_joint(by_view, None)
    cb = cluster_bootstrap_delta(by_view, iterations=10000, seed=13)
    mcn = {k: paired_mcnemar(by_view["q_y"], by_view[v]) for k, v in [("qy_vs_y", "y_only"), ("qy_vs_q", "q_only"), ("qy_vs_wrong", "wrong_q_y")]}
    gates = {
        "qy_gt_y_only": m1["q_y"]["macro_f1_mean"] > m1["y_only"]["macro_f1_mean"],
        "delta_joint_ge_005": dj["delta"] >= 0.05,
        "ci_lower_gt_0": cb["ci95"][0] > 0,
        "qy_gt_wrong": m1["q_y"]["macro_f1_mean"] > m1["wrong_q_y"]["macro_f1_mean"],
    }
    out["stats_clean_seed0"] = {"delta_joint": dj, "bootstrap": cb, "mcnemar": mcn, "gates": gates}
    print("clean Δ_joint:", round(dj["delta"], 4), "CI:", [round(x, 4) for x in cb["ci95"]])
    print("gates:", gates)

    # ---- LLM views on clean subset (final votes)
    votes_by_provider: dict = {"qwen": {}, "deepseek": {}}
    for r in votes:
        prov = str(r.get("response_model", "") or "").lower()
        key = "qwen" if "qwen" in prov else ("deepseek" if "deepseek" in prov else None)
        if key is None or r["response_id"] not in rid_keep:
            continue
        view = str(r.get("judge", "")).replace("view_", "")
        lab = view_label(r)
        if lab is None:
            continue
        votes_by_provider[key].setdefault(view, {})[r["response_id"]] = lab
    llm = {}
    for prov, views in votes_by_provider.items():
        llm[prov] = {}
        for view in VIEWS:
            lab_map = views.get(view, {})
            rows = [{**a, "pred": lab_map.get(a["response_id"], 0), "score": float(lab_map.get(a["response_id"], 0)), "gold": int(a["gold_central"])}
                    for a in keep if a["response_id"] in lab_map]
            llm[prov][view] = binary_metrics(rows)
    out["llm_clean"] = llm

    (DATA / "E1_V4_CLEAN_ANCHOR_SENSITIVITY.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote E1_V4_CLEAN_ANCHOR_SENSITIVITY.json")


if __name__ == "__main__":
    main()
