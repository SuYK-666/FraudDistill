# -*- coding: utf-8 -*-
"""E4/E5 final reports + tables + figures (exp4_unseen_student_v2).

Usage:
  python scripts/e4e5_write_reports.py --protocol-dir outputs/exp4_unseen_student_v2/e4v2_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import yaml

from frauddistill.e4e5_v2.schemas import read_jsonl
from frauddistill.e4e5_v2.metrics import binary_metrics
from frauddistill.e4e5_v2.cluster_bootstrap import run_paired_statistics

COMP_TAGS = {
    "final_student": "Final Student (0.5622)",
    "neural_gold": "Neural-Gold (0.5)",
    "neural_softdistill": "Neural-SoftDistill (0.5)",
    "base_zeroshot": "Base-1.5B-ZeroShot",
}
THRESHOLDS = {"final_student": 0.5622, "neural_gold": 0.5, "neural_softdistill": 0.5, "base_zeroshot": 0.5}


def fmt_cell(v, dec=4, na="—"):
    if v is None:
        return na
    return f"{v:.{dec}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-dir", required=True)
    args = ap.parse_args()
    proto = Path(args.protocol_dir)
    test_rows = read_jsonl(proto / "manifests" / "frozen_test.jsonl")
    cal_rows = read_jsonl(proto / "manifests" / "calibration.jsonl")
    pred_dir = proto / "predictions"
    e5_dir = proto / "e5"
    out_tables = proto / "tables"
    out_figs = proto / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    # ---------------- E4 metrics ----------------
    e4_rows = []
    pred_maps = {}
    for key in ("final_student", "neural_gold", "neural_softdistill", "base_zeroshot"):
        fp = pred_dir / f"{key}.jsonl"
        if not fp.exists():
            print(f"[report] missing {key} predictions")
            continue
        preds = {r["id"]: r for r in read_jsonl(fp)}
        pred_maps[key] = preds
        rows = [r for r in test_rows if r["id"] in preds]
        y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows])
        s = np.array([preds[r["id"]]["risk_score"] for r in rows])
        m = binary_metrics(y, s, threshold=THRESHOLDS[key], label=key)
        e4_rows.append({"model": key, "scope": "pooled", "n": len(rows), **m})
        for shift in ("U1_category", "U2_source", "U3_target_style"):
            idx = [i for i, r in enumerate(rows) if r["primary_shift"] == shift]
            if idx:
                ms = binary_metrics(y[idx], s[idx], threshold=THRESHOLDS[key], label=f"{key}/{shift}")
                e4_rows.append({"model": key, "scope": shift, "n": len(idx), **ms})

    # E4 main table md
    with open(out_tables / "e4_main.md", "w", encoding="utf-8") as f:
        f.write("| Model | Scope | N | Macro-F1 | Recall | FPR | MCC | AUPRC | AUROC |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in e4_rows:
            f.write(f"| {COMP_TAGS.get(r['model'], r['model'])} | {r['scope']} | {r['n']} | "
                    f"{fmt_cell(r['macro_f1'])} | {fmt_cell(r['recall'])} | {fmt_cell(r['fpr'])} | "
                    f"{fmt_cell(r['mcc'])} | {fmt_cell(r.get('auprc'))} | {fmt_cell(r.get('auroc'))} |\n")

    # paired significance (pooled)
    if "final_student" in pred_maps and "neural_gold" in pred_maps:
        stats = run_paired_statistics(test_rows, pred_maps, THRESHOLDS,
                                      [("final_student", "neural_gold"), ("final_student", "neural_softdistill")])
        (out_tables / "e4_paired_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(out_tables / "e4_paired.md", "w", encoding="utf-8") as f:
            f.write("| Comparison | Metric | Δ mean | 95% CI | p (McNemar) | p (Holm) |\n|---|---|---:|---:|---:|---:|\n")
            holm = stats.get("_holm", {})
            for key in ("final_student_vs_neural_gold", "final_student_vs_neural_softdistill"):
                if key not in stats:
                    continue
                for metric in ("macro_f1", "recall", "fpr"):
                    b = stats[key][f"bootstrap_{metric}"]
                    mc = stats[key]["mcnemar"]
                    h = holm.get(key, {}).get("p_holm")
                    f.write(f"| {key} | {metric} | {b['mean_diff']:.4f} | [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] | {mc['p_exact']} | {h} |\n")

    # ---------------- E5 tables ----------------
    e5 = json.loads((e5_dir / "report.json").read_text(encoding="utf-8")) if (e5_dir / "report.json").exists() else None
    if e5:
        with open(out_tables / "e5_main.md", "w", encoding="utf-8") as f:
            f.write("| Policy | Cal N | MF1 | Recall | FPR | MCC | Brier | ECE | Coverage | API rate |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for pol in ("P0", "P1", "P2", "P3"):
                r = e5.get(pol)
                if not r:
                    continue
                f.write(f"| {pol} | {e5.get('P1_fit', {}).get('temperature') and 600 or 0 if pol == 'P1' else (600 if pol in ('P2','P3') else 0)} | ")
                f.write(f"{fmt_cell(r.get('macro_f1'))} | {fmt_cell(r.get('recall'))} | {fmt_cell(r.get('fpr'))} | {fmt_cell(r.get('mcc'))} | "
                        f"{fmt_cell(r.get('brier'))} | {fmt_cell(r.get('ece'))} | {fmt_cell(r.get('coverage'))} | {fmt_cell(r.get('api_rate'))} |\n")

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pr_curve(y, s, label):
        from sklearn.metrics import precision_recall_curve
        p, r, _ = precision_recall_curve(y, s)
        return p, r, label

    curves = []
    for key in ("final_student", "neural_gold", "neural_softdistill"):
        if key not in pred_maps:
            continue
        rows = [r for r in test_rows if r["id"] in pred_maps[key]]
        y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows])
        s = np.array([pred_maps[key][r["id"]]["risk_score"] for r in rows])
        curves.append(pr_curve(y, s, COMP_TAGS[key]))
    if curves:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for p, r, lbl in curves:
            ax.plot(r, p, lw=2, label=lbl)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("E4 PR Curves (Pooled Frozen Test)")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_figs / "e4_pr_curves.png", dpi=150)
        plt.close(fig)

    if e5:
        # reliability diagram P0 vs P1
        s_t = np.array([pred_maps["final_student"][r["id"]]["risk_score"] for r in test_rows])
        y_t = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in test_rows])
        p0 = 1 - s_t
        T = e5.get("P1_fit", {}).get("temperature", 1.0)
        logit = np.log(np.clip(p0, 1e-9, 1 - 1e-9) / (1 - np.clip(p0, 1e-9, 1 - 1e-9)))
        p1 = 1.0 / (1.0 + np.exp(-logit / T))
        fig, ax = plt.subplots(figsize=(6, 5))
        for p_safe, lbl in ((p0, "P0 raw"), (p1, f"P1 T={T:.2f}")):
            bins = np.linspace(0, 1, 11)
            mids, accs = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (p_safe >= lo) & (p_safe < hi)
                if m.sum() == 0:
                    continue
                mids.append((lo + hi) / 2)
                accs.append(1 - y_t[m].mean())
            ax.plot(mids, accs, marker="o", ms=4, label=lbl)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        ax.set_xlabel("Predicted P(safe)")
        ax.set_ylabel("Empirical P(safe)")
        ax.set_title("E5 Reliability Diagram (Frozen Test)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_figs / "e5_reliability.png", dpi=150)
        plt.close(fig)

        # label-efficiency curve
        le = e5.get("label_efficiency", {})
        if le:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sizes = sorted(int(n) for n in le)
            for ax, metric, title in ((axes[0], "test_fpr", "Test FPR vs Calibration Labels"),
                                      (axes[1], "test_ece", "Test ECE vs Calibration Labels")):
                mean = [le[str(n)]["mean"][metric] for n in sizes]
                sd = [le[str(n)]["sd"][metric] for n in sizes]
                ax.errorbar(sizes, mean, yerr=sd, marker="o", capsize=3)
                ax.set_xlabel("N calibration labels")
                ax.set_title(title)
                ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_figs / "e5_label_efficiency.png", dpi=150)
            plt.close(fig)

    print(f"[report] tables -> {out_tables}")
    print(f"[report] figures -> {out_figs}")


if __name__ == "__main__":
    main()
