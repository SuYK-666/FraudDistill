# -*- coding: utf-8 -*-
"""Render frozen E4/E5 final reports from FINAL_*.json artifacts (zero API)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "e4e5_final_staticfix"

# Unicode via escapes so the source file stays pure ASCII.
EM = "\u2014"   # em dash
EN = "\u2013"   # en dash
D = "\u0394"    # capital Delta
MIN = "\u2212"  # minus sign
AP = "\u2248"   # approximately equal
KA = "\u03ba"   # kappa
YEN = "\u00a5"  # yen
DOT = "\u00b7"  # middle dot
LQ = "\u201c"   # left quote
RQ = "\u201d"   # right quote
TIMES = "\u00d7"  # multiplication sign
P20 = "10\u207b\u00b2\u2070"  # 10^-20


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def r4(v, na=EM):
    return na if v is None else f"{v:.4f}"


def fmt_p(p):
    return "p<1e-19" if p < 1e-19 else (f"p={p:.3g}" if p < 0.001 else f"p={p:.4f}")


def dpoint(a, b):
    """Full-sample point estimate of a - b, rounded to 4 decimals."""
    return round(a - b, 4)


def fmt_d(v):
    """Format signed delta with a unicode minus sign."""
    s = f"{v:+.4f}"
    return s.replace("-", MIN)


def ci_of(comp, met):
    b = comp["bootstrap"][met]
    lo = f"{b['ci95'][0]:.4f}".replace("-", MIN)
    hi = f"{b['ci95'][1]:.4f}".replace("-", MIN)
    return f"[{lo}, {hi}]"


def boot_line(comp, met, point):
    return f"{D}{met} = {point:+.4f}, family-cluster bootstrap 95% CI {ci_of(comp, met)}"


def main():
    metrics = load("FINAL_METRICS.json")
    paired = load("FINAL_PAIRED_STATS.json")
    gold = load("FINAL_GOLD_QUALITY.json")
    audit = load("FINAL_DATA_AUDIT.json")
    e4 = metrics["e4"]; e5 = metrics["e5"]; p3 = metrics["p3"]
    rev = metrics["revisions"]
    src_rev = rev["source_experiment"][:7]
    fix_base = rev["static_fix_base"][:7]
    release = rev["release_tag"]

    def header_note(kind):
        return (f"> Frozen offline pass {DOT} 2026-08-10 {DOT} {kind} {DOT} zero new API calls\n"
                f"> Source experiment revision: `{src_rev}` "
                f"{DOT} Static-fix implementation base: `{fix_base}` {DOT} Artifact release tag: `{release}`\n"
                f"> Data: frozen test manifest **N=1200, 557 families** (canonical SHA256 `{audit['test']['sha256']}`), "
                f"calibration **N=600, 243 families** (`{audit['calibration']['sha256']}`); all extra cached "
                f"prediction rows (1425/686) were excluded by manifest-ID join (see `FINAL_DATA_AUDIT.json`).\n"
                f"> Labels are **dual-judge Gold** (DeepSeek+Qwen with adjudication); no human verification.")

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

    ng = paired["neural_gold_vs_final_student"]
    ns = paired["neural_softdistill_vs_final_student"]
    e4_pairs_md = (
        "| Comparison | Metric | Point estimate | Bootstrap 95% CI |\n"
        "|---|---|---:|---:|\n"
        f"| Neural-Gold vs Final Student | Macro-F1 | {fmt_d(dpoint(e4['neural_gold']['pooled']['macro_f1'], e4['final_student']['pooled']['macro_f1']))} | {ci_of(ng, 'macro_f1')} |\n"
        f"| Neural-Gold vs Final Student | F1-unsafe | {fmt_d(dpoint(e4['neural_gold']['pooled']['f1_unsafe'], e4['final_student']['pooled']['f1_unsafe']))} | {ci_of(ng, 'f1_unsafe')} |\n"
        f"| Neural-Gold vs Final Student | Recall | {fmt_d(dpoint(e4['neural_gold']['pooled']['recall'], e4['final_student']['pooled']['recall']))} | {ci_of(ng, 'recall')} |\n"
        f"| Neural-Gold vs Final Student | FPR | {fmt_d(dpoint(e4['neural_gold']['pooled']['fpr'], e4['final_student']['pooled']['fpr']))} | {ci_of(ng, 'fpr')} |\n"
        f"| Neural-Gold vs Final Student | MCC | {fmt_d(dpoint(e4['neural_gold']['pooled']['mcc'], e4['final_student']['pooled']['mcc']))} | {ci_of(ng, 'mcc')} |\n"
        f"| Neural-Gold vs Final Student | AUROC | {dpoint(e4['neural_gold']['pooled'].get('auroc'), e4['final_student']['pooled'].get('auroc')):+.4f} | {ci_of(ng, 'auroc')} |\n"
        f"| Neural-Gold vs Final Student | AUPRC | {dpoint(e4['neural_gold']['pooled'].get('auprc'), e4['final_student']['pooled'].get('auprc')):+.4f} | {ci_of(ng, 'auprc')} |\n"
        f"| Neural-SoftDistill vs Final Student | Macro-F1 | {fmt_d(dpoint(e4['neural_softdistill']['pooled']['macro_f1'], e4['final_student']['pooled']['macro_f1']))} | {ci_of(ns, 'macro_f1')} |\n"
        f"| Neural-SoftDistill vs Final Student | F1-unsafe | {fmt_d(dpoint(e4['neural_softdistill']['pooled']['f1_unsafe'], e4['final_student']['pooled']['f1_unsafe']))} | {ci_of(ns, 'f1_unsafe')} |\n"
        f"| Neural-SoftDistill vs Final Student | Recall | {fmt_d(dpoint(e4['neural_softdistill']['pooled']['recall'], e4['final_student']['pooled']['recall']))} | {ci_of(ns, 'recall')} |\n"
        f"| Neural-SoftDistill vs Final Student | FPR | {fmt_d(dpoint(e4['neural_softdistill']['pooled']['fpr'], e4['final_student']['pooled']['fpr']))} | {ci_of(ns, 'fpr')} |\n"
        f"| Neural-SoftDistill vs Final Student | MCC | {fmt_d(dpoint(e4['neural_softdistill']['pooled']['mcc'], e4['final_student']['pooled']['mcc']))} | {ci_of(ns, 'mcc')} |\n"
        f"| Neural-SoftDistill vs Final Student | AUROC | {dpoint(e4['neural_softdistill']['pooled'].get('auroc'), e4['final_student']['pooled'].get('auroc')):+.4f} | {ci_of(ns, 'auroc')} |\n"
        f"| Neural-SoftDistill vs Final Student | AUPRC | {dpoint(e4['neural_softdistill']['pooled'].get('auprc'), e4['final_student']['pooled'].get('auprc')):+.4f} | {ci_of(ns, 'auprc')} |\n"
    )
    ng_p = ng["mcnemar"]["p_exact"]
    ns_p = ns["mcnemar"]["p_exact"]

    exp4 = f"""# EXP4 {EM} Unseen Generalization (FINAL static-fix version)

{header_note('offline recompute')}

## 1. Scope and terminology

E4 evaluates the distilled **Final Student** (1.5B) against two neural-teacher baselines
(Neural-Gold, Neural-SoftDistill) on **family-disjoint held-out composite shifts**
(U1 category, U2 source, U3 target/style). The term {LQ}unseen{RQ} means only that the current
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
  substantially higher FPR ({r4(e4['neural_gold']['pooled']['fpr'])} and {r4(e4['neural_softdistill']['pooled']['fpr'])}).
  The family-cluster bootstrap confidence intervals for the Macro-F1 differences included zero.
  Separately, paired correctness differences were not significant under exact McNemar tests
  after Holm correction (adjusted p = {paired['_holm']['neural_gold_vs_final_student']['p_holm']:.4f} for both comparisons).

## 3. Paired family-cluster statistics (10,000 replicates, seed {metrics['seed']})

Point estimates are computed on the full sample; CIs come from the family-cluster bootstrap
(paired resampling at the family level, fixed seed). Empirical bootstrap p-values are not
reported (10,000 replicates cannot resolve p below ~1e-4); significance is assessed separately
with exact McNemar tests of paired correctness.

{e4_pairs_md}

- **McNemar test of paired correctness**: Neural-Gold vs Final Student b={ng['mcnemar']['b']},
  c={ng['mcnemar']['c']}, {fmt_p(ng_p)}; Neural-SoftDistill vs Final Student b={ns['mcnemar']['b']},
  c={ns['mcnemar']['c']}, {fmt_p(ns_p)}. Holm-Bonferroni across the two E4 comparisons:
  adjusted p = {paired['_holm']['neural_gold_vs_final_student']['p_holm']:.4f} (both).
- Full per-metric CIs: `FINAL_PAIRED_STATS.json`; PR curves: `figures/e4_pr_curve_final.png`.

## 4. Per-shift breakdown (N=400 per shift, Wilson 95% CI for Recall/FPR)

See `tables/e4_shift_corrected.md`. Highlights:
- **U1 (category shift)**: Student Recall {r4(e4['final_student']['U1_category']['recall'])} / FPR {r4(e4['final_student']['U1_category']['fpr'])}.
- **U2 (source shift)**: Student Recall {r4(e4['final_student']['U2_source']['recall'])} / FPR {r4(e4['final_student']['U2_source']['fpr'])}.
- **U3 (target/style shift)**: Student Recall {r4(e4['final_student']['U3_target_style']['recall'])} / FPR {r4(e4['final_student']['U3_target_style']['fpr'])}.

## 5. Panel disclosures and limitations

- **U1 (N=400)**: {audit['u1']['n_question_mark_suffix']} of 400 queries carry a trailing run of
  at least three `?` characters (regex `\\?{{3,}}\\s*$`), and the panel has a
  language{EN}label correlation (language-only BA/AUROC {AP} 0.70). De-suffixed re-inference was
  not run in this round; U1 is interpreted as a **controlled stress panel**, not a
  natural-distribution estimate.
- **U2 (N=400)**: all rows come from **PKU-SafeRLHF** ({audit['u2']['fraud_category']['general_harm']} `general_harm`,
  {audit['u2']['fraud_category']['financial_fraud']} `financial_fraud`). U2 is a general harmful-response
  source shift; it is not used to claim fraud-specific source generalization.
- **U3 (N=400)**: target-model/style composite shift; effects of target model, source and style are
  not separable into a single causal claim.
- **Gold**: dual-judge Gold with raw judge agreement {gold['raw_agreement_rate']:.4f} ({KA} = {gold['cohens_kappa']:.4f},
  agreed {gold['resolution']['agreed']} / third-opinion {gold['resolution']['deepseek_third_opinion']} /
  deterministic {gold['resolution']['deterministic_arbiter']}, full 1800 records). No human verification.
- Metadata shortcuts were limited overall, but the U1 panel retained the language-label correlation and
  synthetic punctuation artifacts above; **no claim that all shortcut audits passed**.

## 6. Appendix {EM} Base-1.5B zero-shot (300-row subset)

The Base-1.5B zero-shot run covers a 300-row subset only and behaves as an **all-unsafe predictor**
at its 0.5 operating point (F1-unsafe {r4(e4['base_zeroshot']['pooled']['f1_unsafe'])}, Recall 1.0, FPR 1.0,
MCC 0). It is not directly comparable with the N=1200 rows and is **excluded from the main table**.

## 7. Appendix {EM} 4-class

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

    p3c = paired["P3_vs_P0"]
    p1c = paired["P1_vs_P0"]
    delta_table = (
        "| Comparison | Point estimate | Bootstrap 95% CI |\n"
        "|---|---:|---:|\n"
        f"| {D}Macro-F1(P3{MIN}P0) | {fmt_d(dpoint(p3['P3_K180']['macro_f1'], e5['P0']['macro_f1']))} | {ci_of(p3c, 'macro_f1')} |\n"
        f"| {D}F1-unsafe(P3{MIN}P0) | {fmt_d(dpoint(p3['P3_K180']['f1_unsafe'], e5['P0']['f1_unsafe']))} | {ci_of(p3c, 'f1_unsafe')} |\n"
        f"| {D}Recall(P3{MIN}P0) | {fmt_d(dpoint(p3['P3_K180']['recall'], e5['P0']['recall']))} | {ci_of(p3c, 'recall')} |\n"
        f"| {D}FPR(P3{MIN}P0) | {fmt_d(dpoint(p3['P3_K180']['fpr'], e5['P0']['fpr']))} | {ci_of(p3c, 'fpr')} |\n"
        f"| {D}MCC(P3{MIN}P0) | {fmt_d(dpoint(p3['P3_K180']['mcc'], e5['P0']['mcc']))} | {ci_of(p3c, 'mcc')} |\n"
        f"| {D}Macro-F1(P1{MIN}P0) | {fmt_d(dpoint(e5['P1']['macro_f1'], e5['P0']['macro_f1']))} | {ci_of(p1c, 'macro_f1')} |\n"
        f"| {D}Recall(P1{MIN}P0) | {fmt_d(dpoint(e5['P1']['recall'], e5['P0']['recall']))} | {ci_of(p1c, 'recall')} |\n"
        f"| {D}FPR(P1{MIN}P0) | {fmt_d(dpoint(e5['P1']['fpr'], e5['P0']['fpr']))} | {ci_of(p1c, 'fpr')} |\n"
    )

    exp5 = f"""# EXP5 {EM} Calibration & Selective Audit (FINAL static-fix version)

{header_note('offline recompute')}

Scope: P0 (frozen student), P1 (temperature calibration), P2 (selective abstention),
P3 (score-ambiguity selective audit with cached DeepSeek decisions).

## 1. Main table (Eval N=1200, Cal N=600)

{e5_main_md}

- **P0** is the frozen operating point of the Final Student (`risk_score >= 0.5622`).
- **P1** (temperature 5.0 + risk threshold {metrics['thresholds']['P1']['threshold']:.4f}, fitted on
  calibration) is a **negative result**: it trades recall for lower FPR.
  {D}Macro-F1(P1{MIN}P0) = {fmt_d(dpoint(e5['P1']['macro_f1'], e5['P0']['macro_f1']))},
  {D}Recall(P1{MIN}P0) = {fmt_d(dpoint(e5['P1']['recall'], e5['P0']['recall']))},
  {D}FPR(P1{MIN}P0) = {fmt_d(dpoint(e5['P1']['fpr'], e5['P0']['fpr']))},
  {D}Brier {AP} {fmt_d(dpoint(e5['P1']['brier'], e5['P0']['brier']))},
  {D}ECE {AP} {fmt_d(dpoint(e5['P1']['ece'], e5['P0']['ece']))} (new {MIN} baseline, unrounded values).
- **P2** (all-safe selective abstention) is a degenerate negative result (coverage 1.0, no abstention).
- **P3** is the primary reported operating point: **K=180 / 15% query rate** selected by the
  **score-ambiguity heuristic** (min |risk_score {MIN} 0.5|, from cached decisions), **not** a
  pre-registered optimum and **not** re-selected from test.

## 2. P3 vs P0 and P1 vs P0 (paired, 10,000 family-cluster replicates)

Point estimates are full-sample values; CIs come from the family-cluster bootstrap (fixed seed).
Empirical bootstrap p-values are not reported (10,000 replicates cannot resolve p below ~1e-4).

{delta_table}

- **Exact McNemar test of paired correctness** (P3 vs P0): b={p3c['mcnemar']['b']}, c={p3c['mcnemar']['c']},
  p=7.53{TIMES}{P20} {EM} the cascade corrects 85 errors while introducing 5.
- Score-based AUROC/AUPRC are unchanged by the cascade by construction and are labeled
  **Student-score AUROC/AUPRC** in the sensitivity table.

## 3. P3 sensitivity (K=60{EN}600, API rate 5{EN}50%)

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
- P3 new API cost {AP} **{YEN}0.07** (single-judge DeepSeek on 600 cached rows).
- Historical E4/E5 API spend across all phases {AP} **{YEN}15.3**; the original {YEN}10 hard cap was **not**
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

> 2026-08-10 {DOT} offline recompute {DOT} zero new API calls
> Source experiment revision: `{src_rev}` {DOT} Static-fix implementation base: `{fix_base}`
> Artifact release tag: `{release}`
> Frozen manifests: test **1200 / 557 families**, calibration **600 / 243 families** (canonical JSON-record
> SHA256, verified `sha256_match: true`); predictions filtered by manifest-ID join (1425/686 cached rows excluded).
> Gold: dual-judge (DeepSeek+Qwen), raw agreement {gold['raw_agreement_rate']:.4f}, {KA} {gold['cohens_kappa']:.4f}
> (agreed {gold['resolution']['agreed']} / third-opinion {gold['resolution']['deepseek_third_opinion']} / deterministic {gold['resolution']['deterministic_arbiter']}; no human verification).

## E4 {EM} family-disjoint held-out composite shifts

| Model | Eval N | Families | F1-unsafe | Macro-F1 | Recall | FPR | MCC | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final Student | 1200 | 557 | {r4(e4['final_student']['pooled']['f1_unsafe'])} | {r4(e4['final_student']['pooled']['macro_f1'])} | {r4(e4['final_student']['pooled']['recall'])} | {r4(e4['final_student']['pooled']['fpr'])} | {r4(e4['final_student']['pooled']['mcc'])} | {r4(e4['final_student']['pooled'].get('auroc'))} | {r4(e4['final_student']['pooled'].get('auprc'))} |
| Neural-Gold | 1200 | 557 | {r4(e4['neural_gold']['pooled']['f1_unsafe'])} | {r4(e4['neural_gold']['pooled']['macro_f1'])} | {r4(e4['neural_gold']['pooled']['recall'])} | {r4(e4['neural_gold']['pooled']['fpr'])} | {r4(e4['neural_gold']['pooled']['mcc'])} | {r4(e4['neural_gold']['pooled'].get('auroc'))} | {r4(e4['neural_gold']['pooled'].get('auprc'))} |
| Neural-SoftDistill | 1200 | 557 | {r4(e4['neural_softdistill']['pooled']['f1_unsafe'])} | {r4(e4['neural_softdistill']['pooled']['macro_f1'])} | {r4(e4['neural_softdistill']['pooled']['recall'])} | {r4(e4['neural_softdistill']['pooled']['fpr'])} | {r4(e4['neural_softdistill']['pooled']['mcc'])} | {r4(e4['neural_softdistill']['pooled'].get('auroc'))} | {r4(e4['neural_softdistill']['pooled'].get('auprc'))} |

## E5 {EM} calibration & selective audit

| Policy | Eval N | Cal N | Audited K | API rate | F1-unsafe | Macro-F1 | Recall | FPR | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (Final Student) | 1200 | 600 | 0 | 0.00 | {r4(e5['P0']['f1_unsafe'])} | {r4(e5['P0']['macro_f1'])} | {r4(e5['P0']['recall'])} | {r4(e5['P0']['fpr'])} | {r4(e5['P0']['mcc'])} |
| P1 (temp 5.0) | 1200 | 600 | 0 | 0.00 | {r4(e5['P1']['f1_unsafe'])} | {r4(e5['P1']['macro_f1'])} | {r4(e5['P1']['recall'])} | {r4(e5['P1']['fpr'])} | {r4(e5['P1']['mcc'])} |
| P2 (all-safe) | 1200 | 600 | 0 | 0.00 | {r4(e5['P2']['f1_unsafe'])} | {r4(e5['P2']['macro_f1'])} | {r4(e5['P2']['recall'])} | {r4(e5['P2']['fpr'])} | {r4(e5['P2']['mcc'])} |
| P3 (K=180, primary) | 1200 | 600 | 180 | 0.15 | {r4(p3['P3_K180']['f1_unsafe'])} | {r4(p3['P3_K180']['macro_f1'])} | {r4(p3['P3_K180']['recall'])} | {r4(p3['P3_K180']['fpr'])} | {r4(p3['P3_K180']['mcc'])} |

## Key statistics (10,000 family-cluster bootstrap, seed {metrics['seed']})

Point estimates are full-sample values; CIs are family-cluster bootstrap 95% intervals.
Empirical bootstrap p-values are not reported (10,000 replicates cannot resolve p below ~1e-4).

- P3 vs P0: {D}Macro-F1 {fmt_d(dpoint(p3['P3_K180']['macro_f1'], e5['P0']['macro_f1']))} (95% CI {ci_of(p3c, 'macro_f1')}),
  {D}F1-unsafe {fmt_d(dpoint(p3['P3_K180']['f1_unsafe'], e5['P0']['f1_unsafe']))} (95% CI {ci_of(p3c, 'f1_unsafe')}),
  {D}Recall {fmt_d(dpoint(p3['P3_K180']['recall'], e5['P0']['recall']))} (95% CI {ci_of(p3c, 'recall')}),
  {D}FPR {fmt_d(dpoint(p3['P3_K180']['fpr'], e5['P0']['fpr']))} (95% CI {ci_of(p3c, 'fpr')}),
  {D}MCC {fmt_d(dpoint(p3['P3_K180']['mcc'], e5['P0']['mcc']))} (95% CI {ci_of(p3c, 'mcc')}).
- **Exact McNemar test of paired correctness** (P3 vs P0): b={p3c['mcnemar']['b']}, c={p3c['mcnemar']['c']},
  p=7.53{TIMES}{P20}.
- P1 vs P0 (negative result): {D}Macro-F1 {fmt_d(dpoint(e5['P1']['macro_f1'], e5['P0']['macro_f1']))} (95% CI {ci_of(p1c, 'macro_f1')}),
  {D}Recall {fmt_d(dpoint(e5['P1']['recall'], e5['P0']['recall']))} (95% CI {ci_of(p1c, 'recall')}),
  {D}FPR {fmt_d(dpoint(e5['P1']['fpr'], e5['P0']['fpr']))} (95% CI {ci_of(p1c, 'fpr')}).
- E4 model pairs: family-cluster bootstrap CIs for the Macro-F1 differences included zero;
  paired correctness differences were not significant under exact McNemar tests after Holm
  correction (adjusted p = {paired['_holm']['neural_gold_vs_final_student']['p_holm']:.4f} for both comparisons).

## Conclusion

The distilled student retained moderate ranking ability under family-disjoint held-out composite
shifts (AUROC {r4(e4['final_student']['pooled'].get('auroc'))}), but its frozen operating point traded
recall for a low false-positive rate. A score-ambiguity selective audit at a 15% query rate improved
unsafe-class F1, macro-F1, recall, FPR, and MCC using cached single-judge decisions. The cascade
result is interpreted as a system-level recovery mechanism rather than evidence of deployment
readiness.

## Deliverables in this folder

- `FINAL_METRICS.json` / `FINAL_PAIRED_STATS.json` / `FINAL_DATA_AUDIT.json` / `FINAL_GOLD_QUALITY.json`
- `tables/` (4 corrected tables) {DOT} `figures/` (PR curve, P3 sensitivity)
- `FINAL_CHANGELOG.md` {DOT} `EXP4_UNSEEN_GENERALIZATION_REPORT_FINAL.md` {DOT} `EXP5_CALIBRATION_REPORT_FINAL.md`
- Old reports/tables/figures and N=6 gold audit archived under `experiments/archive/e4e5_pre_staticfix/`.
"""
    (OUT / "E4E5_FINAL_SUMMARY.md").write_text(summary, encoding="utf-8")
    print("[reports] wrote 3 reports to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())