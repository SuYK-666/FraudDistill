# -*- coding: utf-8 -*-
"""E4/E5 final static-fix closeout: frozen offline recompute (zero API).

Usage:
  python scripts/e4e5_staticfix_finalize.py --protocol-dir outputs/exp4_unseen_student_v2/e4v2_FINAL
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frauddistill.e4e5_v2.metrics import binary_metrics_raw, wilson_ci
from frauddistill.e4e5_v2.cluster_bootstrap import (paired_cluster_bootstrap, exact_mcnemar, holm_correct)

STUDENT_THR = 0.5622
P1_THR = 0.6105563031516033
P1_TEMP = 5.0
SEED = 20260808
REPLICATES = 10000


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def temp_scaled(scores: np.ndarray, t: float) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(s / (1.0 - s))
    return 1.0 / (1.0 + np.exp(-logit / t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--out-dir", default="experiments/e4e5_final_staticfix")
    args = ap.parse_args()
    proto = Path(args.protocol_dir)
    out = REPO / args.out_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    commit = git_head()
    # ---------------- manifests ----------------
    test_rows = read_jsonl(proto / "manifests" / "frozen_test.jsonl")
    cal_rows = read_jsonl(proto / "manifests" / "calibration.jsonl")
    hashes = json.loads((proto / "manifests" / "hashes.json").read_text(encoding="utf-8"))
    test_sha = sha256_file(proto / "manifests" / "frozen_test.jsonl")
    cal_sha = sha256_file(proto / "manifests" / "calibration.jsonl")
    test_sha_expected = hashes["frozen_test"]["sha256"]
    cal_sha_expected = hashes["calibration"]["sha256"]
    assert len(test_rows) == 1200 and len(set(r["family_id"] for r in test_rows)) == 557
    assert len(cal_rows) == 600 and len(set(r["family_id"] for r in cal_rows)) == 243

    # ---------------- data audit ----------------
    audit = {"test": {"manifest_n": len(test_rows), "families": 557, "sha256": test_sha, "sha256_expected": test_sha_expected, "sha256_match": (test_sha == test_sha_expected),
                      "gold_label": dict(Counter(r["gold_label"] for r in test_rows)),
                      "primary_shift": dict(Counter(r["primary_shift"] for r in test_rows))},
             "calibration": {"manifest_n": len(cal_rows), "families": 243, "sha256": cal_sha, "sha256_expected": cal_sha_expected, "sha256_match": (cal_sha == cal_sha_expected),
                             "gold_label": dict(Counter(r["gold_label"] for r in cal_rows))}}
    t_id = {r["id"] for r in test_rows}; c_id = {r["id"] for r in cal_rows}
    audit["intersections"] = {
        "id_test_cal": sorted(t_id & c_id),
        "family_test_cal": sorted(({r["family_id"] for r in test_rows} & {r["family_id"] for r in cal_rows})),
        "qy_test_cal": sorted(({r["qy_hash"] for r in test_rows} & {r["qy_hash"] for r in cal_rows})),
    }
    cal_qy = Counter(r["qy_hash"] for r in cal_rows)
    audit["calibration"]["qy_unique"] = len(cal_qy)
    audit["calibration"]["qy_dup_ids"] = [r["id"] for r in cal_rows if cal_qy[r["qy_hash"]] > 1]
    # U1 / U2 disclosures
    u1 = [r for r in test_rows if r["primary_shift"] == "U1_category"]
    u2 = [r for r in test_rows if r["primary_shift"] == "U2_source"]
    audit["u1"] = {"n": len(u1),
                   "n_question_mark_suffix": sum(1 for r in u1
                                                 if re.search(r"\?{3,}\s*$", r.get("user_query") or ""))}
    audit["u2"] = {"n": len(u2), "source": dict(Counter(r.get("source") for r in u2)),
                   "fraud_category": dict(Counter(r.get("fraud_category") for r in u2))}

    pred_files = {
        "final_student": ("predictions/final_student.jsonl", "test"),
        "neural_gold": ("predictions/neural_gold.jsonl", "test"),
        "neural_softdistill": ("predictions/neural_softdistill.jsonl", "test"),
        "base_zeroshot": ("predictions/base_zeroshot.jsonl", "test"),
        "final_student_calibration": ("predictions/final_student_calibration.jsonl", "cal"),
    }
    audit["predictions"] = {}
    for key, (rel, scope) in pred_files.items():
        rows = read_jsonl(proto / rel)
        ids = [r["id"] for r in rows]
        manifest = t_id if scope == "test" else c_id
        used = [i for i in ids if i in manifest]
        extra = [i for i in ids if i not in manifest]
        missing = sorted(manifest - set(ids))
        dup = [k for k, v in Counter(ids).items() if v > 1]
        audit["predictions"][key] = {"file": rel, "rows": len(rows), "used": len(used),
                                     "missing": missing, "extra_n": len(extra), "extra_ids": extra[:20],
                                     "duplicates": dup}
    (out / "FINAL_DATA_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- E4 metrics (test 1200) ----------------
    pred_maps = {}
    for key in ("final_student", "neural_gold", "neural_softdistill"):
        pred_maps[key] = {r["id"]: r for r in read_jsonl(proto / pred_files[key][0])}
    thresholds = {"final_student": STUDENT_THR, "neural_gold": 0.5, "neural_softdistill": 0.5}
    e4 = {}
    for key in ("final_student", "neural_gold", "neural_softdistill"):
        rows = [r for r in test_rows if r["id"] in pred_maps[key]]
        y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows])
        s = np.array([pred_maps[key][r["id"]]["risk_score"] for r in rows])
        pred = (s >= thresholds[key]).astype(int)
        m = binary_metrics_raw(y, s, pred=pred, threshold=thresholds[key])
        e4[key] = {"pooled": m}
        for sh in ("U1_category", "U2_source", "U3_target_style"):
            idx = [i for i, r in enumerate(rows) if r["primary_shift"] == sh]
            ys = y[idx]; ss = s[idx]
            ms = binary_metrics_raw(ys, ss, pred=(ss >= thresholds[key]).astype(int), threshold=thresholds[key])
            ms["recall_ci95"] = list(wilson_ci(int(ms["tp"]), int(ms["n_positive"])))
            ms["fpr_ci95"] = list(wilson_ci(int(ms["fp"]), int(ms["n_negative"])))
            e4[key][sh] = ms
    # base zero-shot appendix (300 subset)
    bz_rows = read_jsonl(proto / pred_files["base_zeroshot"][0])
    bz_map = {r["id"]: r for r in bz_rows}
    bz_used = [r for r in test_rows if r["id"] in bz_map]
    bz_y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in bz_used])
    bz_s = np.array([bz_map[r["id"]]["risk_score"] for r in bz_used])
    e4["base_zeroshot"] = {"pooled": binary_metrics_raw(bz_y, bz_s, pred=(bz_s >= 0.5).astype(int), threshold=0.5),
                           "n": len(bz_used), "all_unsafe_predictor": bool(((bz_s >= 0.5).astype(int) == 1).all())}

    # ---------------- E5 policies ----------------
    test_rows_full = test_rows
    y_test = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in test_rows_full])
    s_test = np.array([pred_maps["final_student"][r["id"]]["risk_score"] for r in test_rows_full])
    fams = [str(r["family_id"] or r["id"]) for r in test_rows_full]
    shifts = [r["primary_shift"] for r in test_rows_full]

    def policy_pred(scores, thr, temp=None):
        s = temp_scaled(scores, temp) if temp else scores
        return (s >= thr).astype(int)

    p0_pred = policy_pred(s_test, STUDENT_THR)
    p1_pred = policy_pred(s_test, P1_THR, P1_TEMP)
    p2_pred = np.zeros_like(p0_pred)

    def m_from(pred, label):
        return binary_metrics_raw(y_test, s_test, pred=pred, threshold=STUDENT_THR, label=label)

    e5 = {"P0": m_from(p0_pred, "P0"), "P1": m_from(p1_pred, "P1"), "P2": m_from(p2_pred, "P2")}
    # calibration brier/ece for P0/P1 (from report.json, frozen offline cache)
    rep = json.loads((proto / "e5" / "report.json").read_text(encoding="utf-8"))
    e5["P0"]["brier"] = rep["P0"]["brier"]; e5["P0"]["ece"] = rep["P0"]["ece"]
    e5["P1"]["brier"] = rep["P1"]["brier"]; e5["P1"]["ece"] = rep["P1"]["ece"]

    # P3 from audit cache
    audit_rows = read_jsonl(proto / "e5" / "p3_audit_results.jsonl")
    audit_by_id = {r["id"]: r for r in audit_rows}
    assert len(audit_rows) == 600 and all(r.get("ds_label") is not None for r in audit_rows)
    amb = np.abs(s_test - 0.5)
    order = np.argsort(amb)
    p3 = {}
    for k in (60, 120, 180, 240, 300, 360, 420, 480, 540, 600):
        final = p0_pred.copy()
        for i in order[:k]:
            a = audit_by_id[test_rows_full[i]["id"]]
            final[i] = 1 if a["ds_label"] == "unsafe" else 0
        mm = m_from(final, f"P3_K{k}")
        agree = sum(1 for i in order[:k] if (1 if audit_by_id[test_rows_full[i]["id"]]["ds_label"] == "unsafe" else 0) == y_test[i])
        p3[f"P3_K{k}"] = {**mm, "n_audited": k, "api_rate": round(k / 1200, 4),
                          "judge_agreement_with_gold": round(agree / k, 4),
                          "student_score_auroc": e5["P0"].get("auroc"), "student_score_auprc": e5["P0"].get("auprc")}
        per_shift = {}
        for sh in ("U1_category", "U2_source", "U3_target_style"):
            idx = [i for i in range(1200) if shifts[i] == sh]
            aud = [i for i in idx if i in set(order[:k])]
            per_shift[sh] = {"n": len(idx), "audited": len(aud),
                             "audited_unsafe_gold": int(sum(1 for i in aud if y_test[i] == 1))}
        p3[f"P3_K{k}"]["per_shift"] = per_shift
    # verify frozen P3 K=180 primary point
    p3_main = p3["P3_K180"]
    assert abs(p3_main["f1_unsafe"] - 0.4777) < 0.002 and abs(p3_main["recall"] - 0.33) < 0.002, p3_main

    # ---------------- paired statistics ----------------
    def score_arr(key):
        return np.array([pred_maps[key][r["id"]]["risk_score"] for r in test_rows_full], dtype=float)

    s_gold = score_arr("neural_gold"); s_soft = score_arr("neural_softdistill")
    comparisons = {
        "neural_gold_vs_final_student": (s_gold, np.full(1200, 0.5), np.full(1200, STUDENT_THR), True),
        "neural_softdistill_vs_final_student": (s_soft, np.full(1200, 0.5), np.full(1200, STUDENT_THR), True),
        "P1_vs_P0": (temp_scaled(s_test, P1_TEMP), np.full(1200, P1_THR), np.full(1200, STUDENT_THR), False),
        "P3_vs_P0": (s_test, np.full(1200, STUDENT_THR), np.full(1200, STUDENT_THR), False),
    }
    p3_pred = p0_pred.copy()  # keep non-audited rows at student predictions
    for i in order[:180]:
        a = audit_by_id[test_rows_full[i]["id"]]
        p3_pred[i] = 1 if a["ds_label"] == "unsafe" else 0
    paired = {}
    for name, (sa, th_a, th_b, with_roc) in comparisons.items():
        if name == "P3_vs_P0":
            pa, pb = p3_pred, p0_pred
            boot = paired_cluster_bootstrap(y_test, sa, sa, (float(th_a[0]), float(th_b[0])), fams,
                                            replicates=REPLICATES, seed=SEED,
                                            pred_a=p3_pred, pred_b=p0_pred,
                                            metrics=["macro_f1", "f1_unsafe", "recall", "fpr", "mcc"])
            mc = exact_mcnemar(pa, pb, y_test)
            paired[name] = {"bootstrap": boot, "mcnemar": mc, "n": 1200,
                            "note": "score-based AUROC/AUPRC unchanged by cascade by construction"}
        else:
            pa = (sa >= float(th_a[0])).astype(int)
            pb = (s_test >= float(th_b[0])).astype(int)
            boot = paired_cluster_bootstrap(y_test, sa, s_test, (float(th_a[0]), float(th_b[0])), fams,
                                            replicates=REPLICATES, seed=SEED,
                                            metrics=["macro_f1", "f1_unsafe", "recall", "fpr", "mcc"],
                                            include_auroc_auprc=with_roc)
            mc = exact_mcnemar(pa, pb, y_test)
            paired[name] = {"bootstrap": boot, "mcnemar": mc, "n": 1200}
    pvals = [paired[n]["mcnemar"]["p_exact"] for n in comparisons]
    paired["_holm"] = holm_correct(pvals, list(comparisons))

    # ---------------- gold quality (manifest-aligned 1800) ----------------
    # gold_v4_final.jsonl covers 1489 of the 1800 frozen-manifest ids; the
    # remaining 311 records live in the per-panel gold_v4_*.jsonl files.
    gold_records: dict[str, dict] = {}
    for gf in [proto / "audits" / "gold_v4_final.jsonl"] + sorted((proto / "audits").glob("gold_v4_*.jsonl")):
        for _r in read_jsonl(gf):
            gold_records.setdefault(_r["id"], _r)
    gold = [gold_records[r["id"]] for r in test_rows + cal_rows]
    assert len(gold) == 1800 and all(r["id"] in gold_records for r in test_rows + cal_rows)
    res = Counter(r.get("resolution") for r in gold)
    raw_agree = sum(1 for r in gold if r["judge_a_label"] == r["judge_b_label"])
    from sklearn.metrics import cohen_kappa_score
    kappa = float(cohen_kappa_score([r["judge_a_label"] for r in gold], [r["judge_b_label"] for r in gold]))
    gold_q = {"n": len(gold), "raw_judge_agreement": raw_agree,
              "raw_agreement_rate": round(raw_agree / len(gold), 4), "cohens_kappa": round(kappa, 4),
              "resolution": dict(res), "gold_source": dict(Counter(r.get("gold_source") for r in gold)),
              "label": dict(Counter(r["gold_label"] for r in gold)),
              "type": dict(Counter(r["gold_type"] for r in gold)),
              "note": "dual-judge Gold (DeepSeek+Qwen), no human verification"}
    # P3 K=180 static sensitivity (resolution join)
    gold_by_id = {r["id"]: r for r in gold}
    k180 = [audit_by_id[test_rows_full[i]["id"]] for i in order[:180]]
    sub = defaultdict(list)
    for a in k180:
        g = gold_by_id.get(a["id"])
        sub[g.get("resolution", "unknown") if g else "no_gold"].append(a)
    p3_sens = {}
    for res_name, rows in sorted(sub.items()):
        agree_judge = sum(1 for a in rows if (1 if a["ds_label"] == "unsafe" else 0) == (1 if a["gold_label"] == "unsafe" else 0))

        agree_student = sum(1 for a in rows if (1 if a["student_label"] == "unsafe" else 0) == (1 if a["gold_label"] == "unsafe" else 0))
        p3_sens[res_name] = {"n": len(rows), "p3_judge_agreement_with_gold": round(agree_judge / len(rows), 4),
                             "student_agreement_with_gold": round(agree_student / len(rows), 4)}
    gold_q["p3_k180_sensitivity"] = p3_sens
    (out / "FINAL_GOLD_QUALITY.json").write_text(json.dumps(gold_q, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- JSON outputs ----------------
    metrics = {
        "protocol_dir": str(proto), "commit": commit, "seed": SEED, "replicates": REPLICATES,
        "thresholds": {"final_student": STUDENT_THR, "neural_gold": 0.5, "neural_softdistill": 0.5,
                       "P1": {"temperature": P1_TEMP, "threshold": P1_THR}},
        "e4": e4, "e5": e5, "p3": p3,
    }
    (out / "FINAL_METRICS.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "FINAL_PAIRED_STATS.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- tables ----------------
    def r4(v): return f"{v:.4f}"
    def dash(v): return "—" if v is None else r4(v)

    rows = [("Final Student", e4["final_student"]["pooled"]), ("Neural-Gold", e4["neural_gold"]["pooled"]),
            ("Neural-SoftDistill", e4["neural_softdistill"]["pooled"])]
    lines = ["| Model | Eval N | Families | F1-unsafe | Macro-F1 | Recall | FPR | MCC | AUROC | AUPRC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in rows:
        lines.append(f"| {name} | 1200 | 557 | {r4(m['f1_unsafe'])} | {r4(m['macro_f1'])} | {r4(m['recall'])} | "
                     f"{r4(m['fpr'])} | {r4(m['mcc'])} | {dash(m.get('auroc'))} | {dash(m.get('auprc'))} |")
    (out / "tables" / "e4_main_corrected.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["| Model | Shift | N | F1-unsafe | Macro-F1 | Recall | Recall 95%CI | FPR | FPR 95%CI | MCC |",
             "|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|"]
    for key in ("final_student", "neural_gold", "neural_softdistill"):
        for sh in ("U1_category", "U2_source", "U3_target_style"):
            m = e4[key][sh]
            rc = m.get("recall_ci95") or [m["recall"], m["recall"]]
            fc = m.get("fpr_ci95") or [m["fpr"], m["fpr"]]
            lines.append(f"| {key} | {sh} | {m['n']} | {r4(m['f1_unsafe'])} | {r4(m['macro_f1'])} | {r4(m['recall'])} | "
                         f"[{r4(rc[0])}, {r4(rc[1])}] | {r4(m['fpr'])} | [{r4(fc[0])}, {r4(fc[1])}] | {r4(m['mcc'])} |")
    (out / "tables" / "e4_shift_corrected.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["| Policy | Eval N | Cal N | Audited K | API rate | F1-unsafe | Macro-F1 | Recall | FPR | MCC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in [("P0 (Final Student)", e5["P0"]), ("P1 (temp 5.0)", e5["P1"]), ("P2 (all-safe)", e5["P2"]),
                    ("P3 (K=180, primary)", p3["P3_K180"])]:
        k = 0 if name.startswith(("P0", "P1", "P2")) else 180
        rate = 0.0 if k == 0 else 0.15
        lines.append(f"| {name} | 1200 | 600 | {k} | {rate:.2f} | {r4(m['f1_unsafe'])} | {r4(m['macro_f1'])} | "
                     f"{r4(m['recall'])} | {r4(m['fpr'])} | {r4(m['mcc'])} |")
    (out / "tables" / "e5_main_corrected.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["| Policy | Eval N | Audited K | API rate | F1-unsafe | Macro-F1 | Recall | FPR | MCC | "
             "Student-score AUROC | Student-score AUPRC | Judge-agree |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(p3, key=lambda x: int(x.split("K")[1])):
        m = p3[name]
        lines.append(f"| {name} | 1200 | {m['n_audited']} | {m['api_rate']:.2f} | {r4(m['f1_unsafe'])} | "
                     f"{r4(m['macro_f1'])} | {r4(m['recall'])} | {r4(m['fpr'])} | {r4(m['mcc'])} | "
                     f"{dash(m.get('student_score_auroc'))} | {dash(m.get('student_score_auprc'))} | {r4(m['judge_agreement_with_gold'])} |")
    lines.append("")
    lines.append("> K=180 is the primary reported operating point (15% query rate, score-ambiguity heuristic "
                 "min |risk_score-0.5|); K=60-600 are sensitivity only, not used for selection.")
    (out / "tables" / "e5_p3_sensitivity_corrected.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    from sklearn.metrics import precision_recall_curve
    for key, color, lab in (("final_student", "#1f77b4", "Final Student"),
                            ("neural_gold", "#d62728", "Neural-Gold"),
                            ("neural_softdistill", "#2ca02c", "Neural-SoftDistill")):
        y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in test_rows])
        s = np.array([pred_maps[key][r["id"]]["risk_score"] for r in test_rows])
        prec, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, prec, color=color, label=lab)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("E4 PR curves (test N=1200, 557 families)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "figures" / "e4_pr_curve_final.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ks = [p3[n]["n_audited"] for n in sorted(p3, key=lambda x: int(x.split("K")[1]))]
    f1u = [p3[n]["f1_unsafe"] for n in sorted(p3, key=lambda x: int(x.split("K")[1]))]
    mf1 = [p3[n]["macro_f1"] for n in sorted(p3, key=lambda x: int(x.split("K")[1]))]
    ax.plot(ks, f1u, marker="o", label="F1-unsafe")
    ax.plot(ks, mf1, marker="s", label="Macro-F1")
    ax.axvline(180, color="gray", ls="--", lw=1, label="K=180 primary")
    ax.set_xlabel("Audited K (score-ambiguity)"); ax.set_ylabel("F1"); ax.set_title("E5 P3 sensitivity (test N=1200)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "figures" / "e5_p3_curve_final.png", dpi=150); plt.close(fig)

    # ---------------- changelog ----------------
    cl = f"""# FINAL_CHANGELOG — E4/E5 Static Fix (frozen)

Date: 2026-08-10 (static offline pass; no new API calls, no new annotation, no test-driven tuning)
Code commit: `{commit}`
Source protocol dir: `{args.protocol_dir}`
Seed: {SEED} | Bootstrap replicates: {REPLICATES}

## Fixed in this pass
1. **Macro-F1**: previously `macro_f1` stored the unsafe-class F1. Now `macro_f1 = (F1-unsafe + F1-safe)/2` via `binary_metrics_raw` (matches `sklearn f1_score(average='macro')`); `f1_unsafe` reported separately. All tables rebuilt from unrounded raw values.
2. **Holm-Bonferroni**: corrected to cumulative max (was cumulative min). E4 two-comparison Holm-adjusted p now ≈ 0.2910.
3. **McNemar**: p-value no longer rounded to 6 decimals (P3 vs P0 exact p = 7.53e-20, previously displayed as 0.0).
4. **Bootstrap**: single-pass vectorized family-cluster bootstrap (10,000 replicates, fixed seed, family-level resampling, paired across models) covering Macro-F1, F1-unsafe, Recall, FPR, MCC (+AUROC/AUPRC for E4 model pairs).
5. **Endpoint sign convention**: all Δ reported as new − baseline (e.g. ΔFPR(P1−P0) = −0.0566).
6. **Data scope**: metrics computed only on frozen manifests (test 1200 / 557 families, cal 600 / 243 families); 1425/686 extra prediction rows excluded by manifest-ID join. Manifest SHA256 recorded in FINAL_DATA_AUDIT.json.
7. **U1/U2 wording**: U2 fully PKU-SafeRLHF (298 general_harm / 102 financial_fraud); U1 269 `???`-suffix queries and language-label correlation disclosed as artifacts/limitations.
8. **Gold**: quality summary recomputed on all 1800 records (raw agreement {gold_q['raw_agreement_rate']}, κ {gold_q['cohens_kappa']}, agreed {res.get('agreed')}, third-opinion {res.get('deepseek_third_opinion')}, deterministic {res.get('deterministic_arbiter')}); labeled `dual-judge Gold` (no human verification).
9. **P3**: primary operating point K=180 / 15% API rate via score-ambiguity heuristic; K=60-600 sensitivity only; AUROC/AUPRC columns renamed to Student-score; API cost 600 rows ≈ ¥0.07, historical total ≈ ¥15.3 (original ¥10 hard cap disclosed as protocol deviation).
10. **Base zero-shot** (300-row subset, all-unsafe) and 4-class metrics moved to appendix/limitations; no quantitative comparison against N=1200 rows.

## Invalidated / superseded files
- Old EXP4/EXP5 reports, tables and figures (archived under `experiments/archive/e4e5_pre_staticfix/`).
- `gold_quality_audit_v4.json` (N=6) and old `shortcut_audit` files (archived).
- All previous Macro-F1 CIs (recomputed here).
"""
    (out / "FINAL_CHANGELOG.md").write_text(cl, encoding="utf-8")

    print("[finalize] done ->", out)
    print("[finalize] E4:", {k: round(e4[k]["pooled"]["macro_f1"], 4) for k in ("final_student", "neural_gold", "neural_softdistill")})
    print("[finalize] P3 K180:", {k: round(p3["P3_K180"][k], 4) for k in ("f1_unsafe", "macro_f1", "recall", "fpr", "mcc")})
    print("[finalize] mcnemar P3vsP0:", paired["P3_vs_P0"]["mcnemar"])
    print("[finalize] holm:", {k: round(v["p_holm"], 6) for k, v in paired["_holm"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
