# -*- coding: utf-8 -*-
"""Render frozen E4/E5 final reports from FINAL_*.json artifacts (zero API)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "e4e5_final_staticfix"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def r4(v, na="—"):
    return "—" if v is None else f"{v:.4f}"


def pct(v):
    return "—" if v is None else f"{v*100:.1f}%"


def fmt_p(p):
    return "p<1e-19" if p < 1e-19 else (f"p={p:.3g}" if p < 0.001 else f"p={p:.4f}")


def main():
    metrics = load("FINAL_METRICS.json")
    paired = load("FINAL_PAIRED_STATS.json")
    gold = load("FINAL_GOLD_QUALITY.json")
    audit = load("FINAL_DATA_AUDIT.json")
    e4 = metrics["e4"]; e5 = metrics["e5"]; p3 = metrics["p3"]
    commit = metrics["commit"]

    def boot_line(name, met):
        b = paired[name]["bootstrap"][met]
        return f"Δ{met} = {b['mean_diff']:+.4f} (95% CI [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}], {fmt_p(b['p_value_approx'])})"

    # ---------------- EXP4 report ----------------
    e4_main_rows = [("Final Student", e4["final_student"]["pooled"]),
                    ("Neural-Gold", e4["neural_gold"]["pooled"]),
                    ("Neural-SoftDistill", e4["neural_softdistill"]["pooled"])]
    rows = ["| Model | Eval N | Families | F1-unsafe | Macro-F1 | Recall | FPR | MCC | AUROC | AUPRC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in e4_main_rows:
        rows.append(f"| {name} | {m['n']} | 557 | {r4(m['f1_unsafe'])} | {r4(m['macro_f1'])} | {r4(m['recall'])} | "
                    f"{r4(m['fpr'])} | {r4(m['mcc'])} | {r4(m.get('auroc'))} | {r4(m.get('auprc'))} |")
    e4_main_md = "\n".join(rows)

    exp4 = f"""# EXP4 — Unseen Generalization (FINAL static-fix version)

> Frozen offline pass · 2026-08-10 · commit `{commit}` · zero new API calls
> Data: frozen test manifest **N=1200, 557 families** (SHA256 `{audit['test']['sha256_expected']}`),
> calibration **N=600, 243 families** (`{audit['calibration']['sha256_expected']}`); all extra cached
> prediction rows (1425/686) were excluded by manifest-ID join (see `FINAL_DATA_AUDIT.json`).
> Labels are **dual-judge Gold** (DeepSeek+Qwen with adjudication); no human verification.

## 1. Scope and terminology

E4 evaluates the distilled **Final Student** (1.5B) against two neural-teacher baselines
(Neural-Gold, Neural-SoftDistill) on **family-disjoint held-out composite shifts**
(U1 category, U2 source, U3 target/style). The term “unseen” means only that the current
exposure audit found no exact/family/query overlap with training panels; it does **not**
claim natural-distribution generalization or exhaustive semantic-leakage exclusion.

## 2. Main results (pooled, N=1200)

{e4_main_md}

Notes:
- Operating points are frozen: Final Student uses `risk_score >= 0.5622` (selected on
  calibration); Neural-Gold / Neural-SoftDistill use `>= 0.5`. Thresholded metrics describe
  deployment behavior; ranking ability is compared via AUROC/AUPRC.
- The Final Student retains moderate ranking ability (AUROC {r4(e4['final_student']['pooled'].get('auroc'))})
  but its frozen operating point trades recall for a low false-positive rate
  (Recall {r4(e4['final_student']['pooled']['recall'])}, FPR {r4(e4['final_student']['pooled']['fpr'])}).
- Neural-Gold and Neural-SoftDistill achieve higher unsafe-class F1 at 0.5 but at
  substantially higher FPR ({r4(e4['neural_gold']['pooled']['fpr'])} and {r4(e4['neural_softdistill']['pooled']['fpr'])});
  their Macro-F1 advantage over the Student is small and **not significant** at the family level
  (ΔMacro-F1 {e4['neural_gold']['pooled']['macro_f1']-e4['final_student']['pooled']['macro_f1']:+.4f},
  cluster-bootstrap 95% CI includes zero, Holm-adjusted p = {paired['_holm']['neural_gold_vs_final_student']['p_holm']:.4f}).

## 3. Paired family-cluster statistics (10,000 replicates, seed {metrics['seed']})

- Neural-Gold vs Final Student: {boot_line('neural_gold_vs_final_student', 'macro_f1')}
- Neural-SoftDistill vs Final Student: {boot_line('neural_softdistill_vs_final_student', 'macro_f1')}
- McNemar (unsafe-class agreement): Gold vs Student b={paired['neural_gold_vs_final_student']['mcnemar']['b']},
  c={paired['neural_gold_vs_final_student']['mcnemar']['c']}, {fmt_p(paired['neural_gold_vs_final_student']['mcnemar']['p_exact'])};
  Soft vs Student b={paired['neural_softdistill_vs_final_student']['mcnemar']['b']},
  c={paired['neural_softdistill_vs_final_student']['mcnemar']['c']}, {fmt_p(paired['neural_softdistill_vs_final_student']['mcnemar']['p_exact'])}.
- Holm-Bonferroni across the two E4 comparisons: adjusted p = {paired['_holm']['neural_gold_vs_final_student']['p_holm']:.4f} (both).
- Full per-metric CIs: `FINAL_PAIRED_STATS.json`; PR curves: `figures/e4_pr_curve_final.png`.

## 4. Per-shift breakdown (N=400 per shift, Wilson 95% CI for Recall/FPR)

See `tables/e4_shift_corrected.md`. Highlights:
- **U1 (category shift)**: Student Recall {r4(e4['final_student']['U1_category']['recall'])} / FPR {r4(e4['final_student']['U1_category']['fpr'])}.
- **U2 (source shift)**: Student Recall {r4(e4['final_student']['U2_source']['recall'])} / FPR {r4(e4['final_student']['U2_source']['fpr'])}.
- **U3 (target/style shift)**: Student Recall {r4(e4['final_student']['U3_target_style']['recall'])} / FPR {r4(e4['final_student']['U3_target_style']['fpr'])}.

## 5. Panel disclosures and limitations

- **U1 (N=400)**: {audit['u1']['n_question_mark_suffix']} of 400 queries contain a synthetic `?????` suffix, and the panel has a
  language–label correlation (language-only BA/AUROC ≈ 0.70). De-suffixed re-inference was not
  run in this round; U1 is interpreted as a **controlled stress panel**, not a natural-distribution estimate.
- **U2 (N=400)**: all rows come from **PKU-SafeRLHF** ({audit['u2']['fraud_category']['general_harm']} `general_harm`,
  {audit['u2']['fraud_category']['financial_fraud']} `financial_fraud`). U2 is a general harmful-response
  source shift; it is not used to claim fraud-specific source generalization.
- **U3 (N=400)**: target-model/style composite shift; effects of target model, source and style are
  not separable into a single causal claim.
- **Gold**: dual-judge Gold with raw judge agreement {gold['raw_agreement_rate']:.4f} (κ = {gold['cohens_kappa']:.4f},
  agreed {gold['resolution']['agreed']} / third-opinion {gold['resolution']['deepseek_third_opinion']} /
  deterministic {gold['resolution']['deterministic_arbiter']}, full 1800 records). No human verification.
- Metadata shortcuts were limited overall, but the U1 panel retained the language-label correlation and
  synthetic punctuation artifacts above; **no claim that all shortcut audits passed**.

## 6. Appendix — Base-1.5B zero-shot (300-row subset)

The Base-1.5B zero-shot run covers a 300-row subset only and behaves as an **all-unsafe predictor**
at its 0.5 operating point (F1-unsafe {r4(e4['base_zeroshot']['pooled']['f1_unsafe'])}, Recall 1.0, FPR 1.0,
MCC 0). It is not directly comparable with the N=1200 rows and is **excluded from the main table**.

## 7. Appendix — 4-class

Gold-type support over test+cal: safe {gold['type']['safe']}, fraud_assistance {gold['type']['fraud_assistance']},
over_refusal {gold['type']['over_refusal']}, refusal_failure {gold['type']['refusal_failure']}.
Classes with support < 30 (over_refusal, refusal_failure) are not stably comparable; no 4-class
conclusion is drawn in the main body.
"""
    (OUT / "EXP4_UNSEEN_GENERALIZATION_REPORT_FINAL.md").write_text(exp4, encoding="utf-8")

    # ---------------- EXP5 report ----------------
    e5_rows = [("P0 (Final Student)", e5["P0"], 0, 0.0), ("P1 (temp 5.0)", e5["P1"], 0, 0.0),
               ("P2 (all-safe)", e5["P2"], 0, 0.0), ("P3 (K=180, primary)", p3["P3_K180"], 180, 0.15)]
    rows = ["| Policy | Eval N | Cal N | Audited K | API rate | F1-unsafe | Macro-F1 | Recall | FPR | MCC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m, k, rate in e5_rows:
        rows.append(f"| {name} | 1200 | 600 | {k} | {rate:.2f} | {r4(m['f1_unsafe'])} | {r4(m['macro_f1'])} | "
                    f"{r4(m['recall'])} | {r4(m['fpr'])} | {r4(m['mcc'])} |")
    e5_main_md = "\n".join(rows)

    sens_md = (OUT / "tables" / "e5_p3_sensitivity_corrected.md").read_text(encoding="utf-8")

    exp5 = f"""# EXP5 — Calibration & Selective Audit (FINAL static-fix version)

> Frozen offline pass · 2026-08-10 · commit `{commit}` · zero new API calls
> Scope: P0 (frozen student), P1 (temperature calibration), P2 (selective abstention),
> P3 (score-ambiguity selective audit with cached DeepSeek decisions).

## 1. Main table (Eval N=1200, Cal N=600)

{e5_main_md}

- **P0** is the frozen operating point of the Final Student (`risk_score >= 0.5622`).
- **P1** (temperature 5.0 + risk threshold {metrics['thresholds']['P1']['threshold']:.4f}, fitted on
  calibration) is a **negative result**: it trades recall for lower FPR.
  ΔMacro-F1(P1−P0) = {e5['P1']['macro_f1']-e5['P0']['macro_f1']:+.4f}, ΔRecall = {e5['P1']['recall']-e5['P0']['recall']:+.4f},
  ΔFPR = {e5['P1']['fpr']-e5['P0']['fpr']:+.4f}, ΔBrier ≈ {e5['P1']['brier']-e5['P0']['brier']:+.4f},
  ΔECE ≈ {e5['P1']['ece']-e5['P0']['ece']:+.4f} (new − baseline).
- **P2** (all-safe selective abstention) is a degenerate negative result (coverage 1.0, no abstention).
- **P3** is the primary reported operating point: **K=180 / 15% query rate** selected by the
  **score-ambiguity heuristic** (min |risk_score − 0.5|, from cached decisions), **not** a
  pre-registered optimum and **not** re-selected from test.

## 2. P3 vs P0 (paired, 10,000 family-cluster replicates)

- {boot_line('P3_vs_P0', 'macro_f1')}  (ΔMacro-F1 point ≈ +0.0918)
- {boot_line('P3_vs_P0', 'f1_unsafe')}  (ΔF1-unsafe ≈ +0.1448)
- {boot_line('P3_vs_P0', 'recall')}  (ΔRecall ≈ +0.1167)
- {boot_line('P3_vs_P0', 'fpr')}  (ΔFPR ≈ −0.0167)
- {boot_line('P3_vs_P0', 'mcc')}  (ΔMCC ≈ +0.1458)
- McNemar: b={paired['P3_vs_P0']['mcnemar']['b']}, c={paired['P3_vs_P0']['mcnemar']['c']}
  ({fmt_p(paired['P3_vs_P0']['mcnemar']['p_exact'])}) — the cascade corrects 85 errors while introducing 5.
- Score-based AUROC/AUPRC are unchanged by the cascade by construction and are labeled
  **Student-score AUROC/AUPRC** in the sensitivity table.

## 3. P3 sensitivity (K=60–600, API rate 5–50%)

{sens_md}

The K=180 row is the **primary reported operating point**; the rest of the curve is sensitivity
analysis and was not used for selection.

## 4. Gold-quality / evaluator sensitivity (K=180 audited rows)

- Among the 180 audited rows, {gold['p3_k180_sensitivity']['agreed']['n']} have dual-judge-agreed Gold
  (P3-judge agreement {gold['p3_k180_sensitivity']['agreed']['p3_judge_agreement_with_gold']:.4f},
  Student agreement {gold['p3_k180_sensitivity']['agreed']['student_agreement_with_gold']:.4f});
  third-opinion {gold['p3_k180_sensitivity']['deepseek_third_opinion']['n']},
  deterministic {gold['p3_k180_sensitivity']['deterministic_arbiter']['n']}.
- This table only shows that P3 gains are not driven solely by adjudicated samples; it does **not**
  substitute for independent human evaluation.

## 5. Budget and protocol disclosures

- P3 primary requires 180 audited rows; the cached sensitivity sweep covers **600 rows** in total.
- P3 new API cost ≈ **¥0.07** (single-judge DeepSeek on 600 cached rows).
- Historical E4/E5 API spend across all phases ≈ **¥15.3**; the original ¥10 hard cap was **not**
  satisfied across all phases and is disclosed as a **protocol deviation**.
- DeepSeek participated in part of Gold construction and in the P3 audit; **evaluator dependence**
  is a stated limitation of P3.

## 6. Conclusion

The distilled student retained moderate ranking ability under family-disjoint held-out composite
shifts (AUROC {r4(e4['final_student']['pooled'].get('auroc'))}), but its frozen operating point traded
recall for a low false-positive rate. A score-ambiguity selective audit at a 15% query rate improved
unsafe-class F1, macro-F1, recall, FPR, and MCC using cached single-judge decisions. The cascade
result is interpreted as a **system-level recovery mechanism** rather than evidence of deployment
readiness.
"""
    (OUT / "EXP5_CALIBRATION_REPORT_FINAL.md").write_text(exp5, encoding="utf-8")

    # ---------------- summary ----------------
    summary = f"""# E4/E5 Final Summary (frozen static-fix)

> 2026-08-10 · commit `{commit}` · offline recompute, zero new API calls
> Frozen manifests: test **1200 / 557 families**, calibration **600 / 243 families**;
> predictions filtered by manifest-ID join (1425/686 cached rows excluded).
> Gold: dual-judge (DeepSeek+Qwen), raw agreement {gold['raw_agreement_rate']:.4f}, κ {gold['cohens_kappa']:.4f}
> (agreed {gold['resolution']['agreed']} / third-opinion {gold['resolution']['deepseek_third_opinion']} / deterministic {gold['resolution']['deterministic_arbiter']}; no human verification).

## E4 — family-disjoint held-out composite shifts

| Model | Eval N | Families | F1-unsafe | Macro-F1 | Recall | FPR | MCC | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final Student | 1200 | 557 | {r4(e4['final_student']['pooled']['f1_unsafe'])} | {r4(e4['final_student']['pooled']['macro_f1'])} | {r4(e4['final_student']['pooled']['recall'])} | {r4(e4['final_student']['pooled']['fpr'])} | {r4(e4['final_student']['pooled']['mcc'])} | {r4(e4['final_student']['pooled'].get('auroc'))} | {r4(e4['final_student']['pooled'].get('auprc'))} |
| Neural-Gold | 1200 | 557 | {r4(e4['neural_gold']['pooled']['f1_unsafe'])} | {r4(e4['neural_gold']['pooled']['macro_f1'])} | {r4(e4['neural_gold']['pooled']['recall'])} | {r4(e4['neural_gold']['pooled']['fpr'])} | {r4(e4['neural_gold']['pooled']['mcc'])} | {r4(e4['neural_gold']['pooled'].get('auroc'))} | {r4(e4['neural_gold']['pooled'].get('auprc'))} |
| Neural-SoftDistill | 1200 | 557 | {r4(e4['neural_softdistill']['pooled']['f1_unsafe'])} | {r4(e4['neural_softdistill']['pooled']['macro_f1'])} | {r4(e4['neural_softdistill']['pooled']['recall'])} | {r4(e4['neural_softdistill']['pooled']['fpr'])} | {r4(e4['neural_softdistill']['pooled']['mcc'])} | {r4(e4['neural_softdistill']['pooled'].get('auroc'))} | {r4(e4['neural_softdistill']['pooled'].get('auprc'))} |

## E5 — calibration & selective audit

| Policy | Eval N | Cal N | Audited K | API rate | F1-unsafe | Macro-F1 | Recall | FPR | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (Final Student) | 1200 | 600 | 0 | 0.00 | {r4(e5['P0']['f1_unsafe'])} | {r4(e5['P0']['macro_f1'])} | {r4(e5['P0']['recall'])} | {r4(e5['P0']['fpr'])} | {r4(e5['P0']['mcc'])} |
| P1 (temp 5.0) | 1200 | 600 | 0 | 0.00 | {r4(e5['P1']['f1_unsafe'])} | {r4(e5['P1']['macro_f1'])} | {r4(e5['P1']['recall'])} | {r4(e5['P1']['fpr'])} | {r4(e5['P1']['mcc'])} |
| P2 (all-safe) | 1200 | 600 | 0 | 0.00 | {r4(e5['P2']['f1_unsafe'])} | {r4(e5['P2']['macro_f1'])} | {r4(e5['P2']['recall'])} | {r4(e5['P2']['fpr'])} | {r4(e5['P2']['mcc'])} |
| P3 (K=180, primary) | 1200 | 600 | 180 | 0.15 | {r4(p3['P3_K180']['f1_unsafe'])} | {r4(p3['P3_K180']['macro_f1'])} | {r4(p3['P3_K180']['recall'])} | {r4(p3['P3_K180']['fpr'])} | {r4(p3['P3_K180']['mcc'])} |

## Key statistics (10,000 family-cluster bootstrap, seed {metrics['seed']})

- P3 vs P0: ΔMacro-F1 {paired['P3_vs_P0']['bootstrap']['macro_f1']['mean_diff']:+.4f}
  (CI [{paired['P3_vs_P0']['bootstrap']['macro_f1']['ci95'][0]:.4f}, {paired['P3_vs_P0']['bootstrap']['macro_f1']['ci95'][1]:.4f}]),
  ΔF1-unsafe {paired['P3_vs_P0']['bootstrap']['f1_unsafe']['mean_diff']:+.4f},
  ΔRecall {paired['P3_vs_P0']['bootstrap']['recall']['mean_diff']:+.4f},
  ΔFPR {paired['P3_vs_P0']['bootstrap']['fpr']['mean_diff']:+.4f},
  ΔMCC {paired['P3_vs_P0']['bootstrap']['mcc']['mean_diff']:+.4f};
  McNemar b=5, c=85, p=7.53×10⁻²⁰.
- P1 vs P0 (negative result): ΔMacro-F1 {paired['P1_vs_P0']['bootstrap']['macro_f1']['mean_diff']:+.4f},
  ΔRecall {paired['P1_vs_P0']['bootstrap']['recall']['mean_diff']:+.4f},
  ΔFPR {paired['P1_vs_P0']['bootstrap']['fpr']['mean_diff']:+.4f}.
- E4 model pairs: Holm-adjusted p = {paired['_holm']['neural_gold_vs_final_student']['p_holm']:.4f} (both comparisons).

## Conclusion

The distilled student retained moderate ranking ability under family-disjoint held-out composite
shifts (AUROC {r4(e4['final_student']['pooled'].get('auroc'))}), but its frozen operating point traded
recall for a low false-positive rate. A score-ambiguity selective audit at a 15% query rate improved
unsafe-class F1, macro-F1, recall, FPR, and MCC using cached single-judge decisions. The cascade
result is interpreted as a system-level recovery mechanism rather than evidence of deployment
readiness.

## Deliverables in this folder

- `FINAL_METRICS.json` / `FINAL_PAIRED_STATS.json` / `FINAL_DATA_AUDIT.json` / `FINAL_GOLD_QUALITY.json`
- `tables/` (4 corrected tables) · `figures/` (PR curve, P3 sensitivity)
- `FINAL_CHANGELOG.md` · `EXP4_UNSEEN_GENERALIZATION_REPORT_FINAL.md` · `EXP5_CALIBRATION_REPORT_FINAL.md`
- Old reports/tables/figures and N=6 gold audit archived under `experiments/archive/e4e5_pre_staticfix/`.
"""
    (OUT / "E4E5_FINAL_SUMMARY.md").write_text(summary, encoding="utf-8")
    print("[reports] wrote 3 reports to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
