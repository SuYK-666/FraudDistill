# -*- coding: utf-8 -*-
"""E5 selective calibration (exp5_selective_calibration_v2).

Usage:
  python scripts/e4e5_run_calibration.py --protocol-dir outputs/exp4_unseen_student_v2/e4v2_YYYYMMDD_HHMMSS [--api]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import yaml

from frauddistill.e4e5_v2.schemas import read_jsonl, write_jsonl
from frauddistill.e4e5_v2.metrics import binary_metrics
from frauddistill.e4e5_v2.calibration import (fit_temperature, risk_threshold, calibration_metrics,
                                              low_label_curve, clopper_pearson_ucb)
from frauddistill.e4e5_v2.selective_policy import fit_dual_threshold, apply_dual_threshold

P0_TH = 0.5622


def load_preds(path: Path) -> dict:
    return {r["id"]: r for r in read_jsonl(path)}


def aligntest(test_rows, preds):
    rows = [r for r in test_rows if r["id"] in preds]
    y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows])
    s = np.array([preds[r["id"]]["risk_score"] for r in rows])
    fam = [str(r.get("family_id") or r["id"]) for r in rows]
    return rows, y, s, fam


def aurc(y, scores):
    """Area under risk-coverage curve (error-based selective risk)."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    ys = np.asarray(y, dtype=int)[order]
    err = (ys == 0).astype(float)  # risk = FPR proxy (unsafe fraction of positives)
    n = len(ys)
    covs = np.arange(1, n + 1) / n
    risks = np.cumsum(err) / np.arange(1, n + 1)
    auc = float(np.trapz(risks, covs))
    return round(auc, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--api", action="store_true", help="enable real DeepSeek fallback (budget-limited)")
    args = ap.parse_args()

    proto = Path(args.protocol_dir)
    cfg = yaml.safe_load((REPO / "configs/experiments/exp5_selective_calibration_v2.yaml").read_text(encoding="utf-8"))
    cal_rows = read_jsonl(proto / "manifests" / "calibration.jsonl")
    test_rows = read_jsonl(proto / "manifests" / "frozen_test.jsonl")
    cal_preds = load_preds(proto / "predictions" / "final_student_calibration.jsonl")
    test_preds = load_preds(proto / "predictions" / "final_student.jsonl")
    out_dir = proto / "e5"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_rows = [r for r in cal_rows if r["id"] in cal_preds]
    y_cal = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in cal_rows])
    s_cal = np.array([cal_preds[r["id"]]["risk_score"] for r in cal_rows])
    rows_t, y_t, s_t, fam_t = aligntest(test_rows, test_preds)
    print(f"[e5] cal n={len(cal_rows)} test n={len(rows_t)}")

    report = {"protocol_id": proto.name, "n_cal": len(cal_rows), "n_test": len(rows_t)}
    rows_out = []

    # ---------- P0 frozen ----------
    m0 = binary_metrics(y_t, s_t, threshold=P0_TH, label="P0")
    c0 = calibration_metrics(1 - s_t, y_t)
    report["P0"] = {**m0, "brier": c0["brier"], "ece": c0["ece"], "nll": c0["nll"], "coverage": 1.0, "api_rate": 0.0}
    rows_out.append({"policy": "P0", "cal_n": 0, **{k: report["P0"][k] for k in ("macro_f1","recall","fpr","mcc","brier","ece","coverage","api_rate")}})
    print(f"[e5] P0 MF1={m0['macro_f1']} Recall={m0['recall']} FPR={m0['fpr']} Brier={c0['brier']} ECE={c0['ece']}")

    # ---------- P1 temperature + risk threshold ----------
    p_safe = 1.0 - s_cal
    T = fit_temperature(p_safe, y_cal)
    logit = np.log(np.clip(p_safe, 1e-9, 1 - 1e-9) / (1 - np.clip(p_safe, 1e-9, 1 - 1e-9)))
    pT_safe_cal = 1.0 / (1.0 + np.exp(-logit / T))
    sT_cal = 1.0 - pT_safe_cal
    rt = risk_threshold(sT_cal, y_cal, fpr_target=cfg["p1"]["fpr_target"])
    report["P1_fit"] = {"temperature": T, "threshold_risk": rt}
    if rt is None:
        print("[e5] WARNING no feasible P1 threshold on calibration")
        th1 = 0.5
    else:
        th1 = rt["threshold"]
    # apply to test
    pT_safe_t = 1.0 - s_t
    sT_t = 1.0 - (1.0 / (1.0 + np.exp(-np.log(np.clip(pT_safe_t, 1e-9, 1 - 1e-9) / (1 - np.clip(pT_safe_t, 1e-9, 1 - 1e-9))) / T)))
    m1 = binary_metrics(y_t, sT_t, threshold=th1, label="P1")
    c1 = calibration_metrics(1 - sT_t, y_t)
    report["P1"] = {**m1, "brier": c1["brier"], "ece": c1["ece"], "nll": c1["nll"], "coverage": 1.0, "api_rate": 0.0, "temperature": T, "risk_threshold": th1}
    rows_out.append({"policy": "P1", "cal_n": len(cal_rows), **{k: report["P1"][k] for k in ("macro_f1","recall","fpr","mcc","brier","ece","coverage","api_rate")}})
    print(f"[e5] P1 T={T:.3f} th={th1:.4f} MF1={m1['macro_f1']} Recall={m1['recall']} FPR={m1['fpr']} Brier={c1['brier']} ECE={c1['ece']}")

    # ---------- P2 dual threshold ----------
    pol2 = fit_dual_threshold(s_cal, y_cal, target_api_rate=cfg["p2"]["target_api_rate"],
                              max_cal_fnr_auto_safe=cfg["p2"]["max_calibration_fnr_auto_safe"],
                              max_cal_fpr_auto_unsafe=cfg["p2"]["max_calibration_fpr_auto_unsafe"])
    report["P2_fit"] = pol2
    if pol2 is None:
        print("[e5] WARNING no feasible P2 policy; fallback to P0-style")
        pol2 = {"tau_low": 1.0, "tau_high": 1.0, "coverage": 0.0, "api_rate": 1.0}
    dec2 = apply_dual_threshold(s_t, pol2)
    cov2 = float((dec2 != -1).mean())
    api2 = float((dec2 == -1).mean())
    auto_safe = dec2 == 0
    auto_unsafe = dec2 == 1
    fn_as = int((auto_safe & (y_t == 1)).sum())
    fp_au = int((auto_unsafe & (y_t == 0)).sum())
    n_pos_t = max(int((y_t == 1).sum()), 1)
    n_neg_t = max(int((y_t == 0).sum()), 1)
    report["P2"] = {"coverage": round(cov2, 4), "api_rate": round(api2, 4),
                    "auto_safe_fnr": round(fn_as / n_pos_t, 4), "auto_unsafe_fpr": round(fp_au / n_neg_t, 4),
                    "n_abstain": int(api2 * len(rows_t)), "aurc": aurc(y_t, s_t),
                    "selective_fpr_covered": round(float((auto_unsafe & (y_t == 0)).sum()) / max(int((~auto_safe & (y_t == 0)).sum()), 1), 4)}
    rows_out.append({"policy": "P2", "cal_n": len(cal_rows), "macro_f1": None, "recall": None, "fpr": None, "mcc": None,
                     "brier": None, "ece": None, "coverage": round(cov2, 4), "api_rate": round(api2, 4)})
    print(f"[e5] P2 coverage={cov2:.3f} api_rate={api2:.3f} auto-safe FNR={report['P2']['auto_safe_fnr']} auto-unsafe FPR={report['P2']['auto_unsafe_fpr']}")

    # ---------- P3 fallback on abstain ----------
    abstain_idx = np.where(dec2 == -1)[0]
    fallback_labels = {}
    if len(abstain_idx) == 0:
        print("[e5] no abstain rows")
    elif args.api and cfg["api_budget"].get("enabled"):
        from frauddistill.e4e5_v2.deepseek_fallback import deepseek_judge
        from frauddistill.e4e5_v2.selective_policy import BudgetLedger
        ledger = BudgetLedger(out_dir / "budget_ledger.jsonl", hard_stop_cny=cfg["api_budget"]["hard_stop_cny"])
        n_calls = 0
        for i in abstain_idx[: cfg["p3"]["max_calls"]]:
            r = rows_t[i]
            ok, why = ledger.can_call(0.001)
            if not ok:
                print(f"[e5] budget stop: {why}")
                break
            res, cost = deepseek_judge(r["user_query"], r["target_model_answer"])
            n_calls += 1
            if res.get("status") == "ok":
                fallback_labels[int(i)] = 1 if res["label"] == "unsafe" else 0
                ledger.record(r["id"], r.get("qy_hash", ""), "P3", "deepseek", "deepseek-chat",
                              cost.get("input_tokens", 0), cost.get("output_tokens", 0))
        print(f"[e5] P3 real API calls: {n_calls}")
    else:
        # offline local 1.5B judge as cost-free simulation
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from frauddistill.e4e5_v2.gold_mapper import judge_with_model
        jtok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        jmodel = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", torch_dtype=torch.bfloat16)
        jmodel.eval()
        def jfn(prompt: str) -> str:
            inp = jtok(prompt, return_tensors="pt", truncation=True, max_length=1536)
            with torch.no_grad():
                out = jmodel.generate(**inp, max_new_tokens=96, do_sample=False)
            return jtok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        t0 = time.time()
        for k, i in enumerate(abstain_idx):
            r = rows_t[i]
            j = judge_with_model(jfn, r["user_query"], r["target_model_answer"])
            if j["gold_label"]:
                fallback_labels[int(i)] = 1 if j["gold_label"] == "unsafe" else 0
            else:
                fallback_labels[int(i)] = 1  # unresolved -> unsafe by default (conservative)
            if (k + 1) % 25 == 0:
                print(f"[e5] local judge {k+1}/{len(abstain_idx)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[e5] local judge done: {len(fallback_labels)} rows")
    # full-system prediction
    pred3 = (s_t >= P0_TH).astype(int)  # default P0 decision
    for i, lbl in fallback_labels.items():
        pred3[i] = lbl
    m3 = binary_metrics(y_t, s_t, pred=pred3, label="P3")
    c3 = calibration_metrics(1 - s_t, y_t)
    report["P3"] = {**m3, "brier": c3["brier"], "ece": c3["ece"], "nll": c3["nll"],
                    "api_rate": round(len(fallback_labels) / len(rows_t), 4),
                    "coverage": 1.0, "n_fallback": len(fallback_labels), "fallback_mode": "local_judge_1.5b" if not args.api else "deepseek"}
    rows_out.append({"policy": "P3", "cal_n": len(cal_rows), **{k: report["P3"][k] for k in ("macro_f1","recall","fpr","mcc","brier","ece","coverage","api_rate")}})
    print(f"[e5] P3 MF1={m3['macro_f1']} Recall={m3['recall']} FPR={m3['fpr']} API={report['P3']['api_rate']:.3f}")

    # ---------- paired bootstrap P0/P1/P3 ----------
    from frauddistill.e4e5_v2.cluster_bootstrap import paired_cluster_bootstrap, exact_mcnemar
    fam_ids = fam_t
    boot = {}
    pred0 = (s_t >= P0_TH).astype(int)
    for name, scores_b, th_b, pred_b in (("P1", sT_t, th1, (sT_t >= th1).astype(int)),
                                         ("P3", s_t, P0_TH, pred3)):
        key = f"P0_vs_{name}"
        boot[key] = {}
        for metric in ("macro_f1", "recall", "fpr", "mcc"):
            r = paired_cluster_bootstrap(y_t, scores_b, s_t, (th_b, P0_TH), fam_ids, replicates=10000,
                                         seed=cfg["seed"], metric=metric)
            boot[key][metric] = r
        boot[key]["mcnemar"] = exact_mcnemar(pred_b, pred0, y_t)
    report["bootstrap"] = boot
    print(f"[e5] bootstrap: {json.dumps(boot, ensure_ascii=False)[:500]}")

    # ---------- label-efficiency curve ----------
    llc = low_label_curve(cal_rows, s_cal, y_cal, sizes=tuple(cfg["label_efficiency"]["sizes"]),
                          seeds=cfg["label_efficiency"]["seeds"],
                          family_ids=[r.get("family_id") or r["id"] for r in cal_rows],
                          rng_seed=cfg["seed"])
    # evaluate each fitted policy on frozen test
    llc_eval = {}
    for n_str, runs in llc.items():
        out_rows = []
        for run in runs:
            Tn = run["temperature"]
            thn = run.get("threshold")
            if thn is None:
                continue
            pst = 1.0 - s_t
            logit_t = np.log(np.clip(pst, 1e-9, 1 - 1e-9) / (1 - np.clip(pst, 1e-9, 1 - 1e-9)))
            stT = 1.0 - 1.0 / (1.0 + np.exp(-logit_t / Tn))
            mn = binary_metrics(y_t, stT, threshold=thn)
            cn = calibration_metrics(1 - stT, y_t)
            out_rows.append({"seed": run["seed"], "n_actual": run["n_actual"],
                             "temperature": Tn, "threshold": thn,
                             "test_macro_f1": mn["macro_f1"], "test_recall": mn["recall"],
                             "test_fpr": mn["fpr"], "test_mcc": mn["mcc"],
                             "test_brier": cn["brier"], "test_ece": cn["ece"]})
        llc_eval[n_str] = out_rows
    report["label_efficiency"] = {n: {"mean": {
        "test_fpr": float(np.mean([r["test_fpr"] for r in v])),
        "test_recall": float(np.mean([r["test_recall"] for r in v])),
        "test_macro_f1": float(np.mean([r["test_macro_f1"] for r in v])),
        "test_brier": float(np.mean([r["test_brier"] for r in v])),
        "test_ece": float(np.mean([r["test_ece"] for r in v])),
    }, "sd": {
        "test_fpr": float(np.std([r["test_fpr"] for r in v])),
        "test_recall": float(np.std([r["test_recall"] for r in v])),
        "test_macro_f1": float(np.std([r["test_macro_f1"] for r in v])),
        "test_brier": float(np.std([r["test_brier"] for r in v])),
        "test_ece": float(np.std([r["test_ece"] for r in v])),
    }, "n_runs": len(v)} for n, v in llc_eval.items()}
    print(f"[e5] label efficiency: {json.dumps(report['label_efficiency'], ensure_ascii=False)[:600]}")

    # ---------- save ----------
    write_jsonl(out_dir / "main_table.jsonl", rows_out)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "label_efficiency_runs.jsonl", [{"n": n, **r} for n, runs in llc_eval.items() for r in runs])
    print(f"[e5] saved to {out_dir}")


if __name__ == "__main__":
    main()
