# EXP4 — Unseen Generalization (FINAL static-fix version)

> Frozen offline pass · 2026-08-10 · commit `2ffbf9b4c4e0b06500c34621727258ed72bbc0c7` · zero new API calls
> Data: frozen test manifest **N=1200, 557 families** (SHA256 `7f086e6500888378452f922378a772b84e111a1c56264ace3de0eab9598f5ecb`),
> calibration **N=600, 243 families** (`5c52724ca869f35558cddc1200498189763d72e48c7bb9f3b65023a0c647d4ec`); all extra cached
> prediction rows (1425/686) were excluded by manifest-ID join (see `FINAL_DATA_AUDIT.json`).
> Labels are **dual-judge Gold** (DeepSeek+Qwen with adjudication); no human verification.

## 1. Scope and terminology

E4 evaluates the distilled **Final Student** (1.5B) against two neural-teacher baselines
(Neural-Gold, Neural-SoftDistill) on **family-disjoint held-out composite shifts**
(U1 category, U2 source, U3 target/style). The term “unseen” means only that the current
exposure audit found no exact/family/query overlap with training panels; it does **not**
claim natural-distribution generalization or exhaustive semantic-leakage exclusion.

## 2. Main results (pooled, N=1200)

| Model | Eval N | Families | F1-unsafe | Macro-F1 | Recall | FPR | MCC | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final Student | 1200 | 557 | 0.3329 | 0.5092 | 0.2133 | 0.0683 | 0.2084 | 0.7198 | 0.7044 |
| Neural-Gold | 1200 | 557 | 0.4486 | 0.5366 | 0.3633 | 0.2567 | 0.1153 | 0.5725 | 0.5649 |
| Neural-SoftDistill | 1200 | 557 | 0.4708 | 0.5372 | 0.4033 | 0.3100 | 0.0974 | 0.6224 | 0.5913 |

Notes:
- Operating points are frozen: Final Student uses `risk_score >= 0.5622` (selected on
  calibration); Neural-Gold / Neural-SoftDistill use `>= 0.5`. Thresholded metrics describe
  deployment behavior; ranking ability is compared via AUROC/AUPRC.
- The Final Student retains moderate ranking ability (AUROC 0.7198)
  but its frozen operating point trades recall for a low false-positive rate
  (Recall 0.2133, FPR 0.0683).
- Neural-Gold and Neural-SoftDistill achieve higher unsafe-class F1 at 0.5 but at
  substantially higher FPR (0.2567 and 0.3100);
  their Macro-F1 advantage over the Student is small and **not significant** at the family level
  (ΔMacro-F1 +0.0274,
  cluster-bootstrap 95% CI includes zero, Holm-adjusted p = 0.2910).

## 3. Paired family-cluster statistics (10,000 replicates, seed 20260808)

- Neural-Gold vs Final Student: Δmacro_f1 = +0.0274 (95% CI [-0.0141, 0.0697], p=0.2062)
- Neural-SoftDistill vs Final Student: Δmacro_f1 = +0.0278 (95% CI [-0.0114, 0.0663], p=0.1680)
- McNemar (unsafe-class agreement): Gold vs Student b=206,
  c=183, p=0.2646;
  Soft vs Student b=228,
  c=197, p=0.1455.
- Holm-Bonferroni across the two E4 comparisons: adjusted p = 0.2910 (both).
- Full per-metric CIs: `FINAL_PAIRED_STATS.json`; PR curves: `figures/e4_pr_curve_final.png`.

## 4. Per-shift breakdown (N=400 per shift, Wilson 95% CI for Recall/FPR)

See `tables/e4_shift_corrected.md`. Highlights:
- **U1 (category shift)**: Student Recall 0.2000 / FPR 0.0400.
- **U2 (source shift)**: Student Recall 0.1850 / FPR 0.1150.
- **U3 (target/style shift)**: Student Recall 0.2550 / FPR 0.0500.

## 5. Panel disclosures and limitations

- **U1 (N=400)**: 269 of 400 queries contain a synthetic `?????` suffix, and the panel has a
  language–label correlation (language-only BA/AUROC ≈ 0.70). De-suffixed re-inference was not
  run in this round; U1 is interpreted as a **controlled stress panel**, not a natural-distribution estimate.
- **U2 (N=400)**: all rows come from **PKU-SafeRLHF** (298 `general_harm`,
  102 `financial_fraud`). U2 is a general harmful-response
  source shift; it is not used to claim fraud-specific source generalization.
- **U3 (N=400)**: target-model/style composite shift; effects of target model, source and style are
  not separable into a single causal claim.
- **Gold**: dual-judge Gold with raw judge agreement 0.8733 (κ = 0.7485,
  agreed 1572 / third-opinion 180 /
  deterministic 48, full 1800 records). No human verification.
- Metadata shortcuts were limited overall, but the U1 panel retained the language-label correlation and
  synthetic punctuation artifacts above; **no claim that all shortcut audits passed**.

## 6. Appendix — Base-1.5B zero-shot (300-row subset)

The Base-1.5B zero-shot run covers a 300-row subset only and behaves as an **all-unsafe predictor**
at its 0.5 operating point (F1-unsafe 0.6784, Recall 1.0, FPR 1.0,
MCC 0). It is not directly comparable with the N=1200 rows and is **excluded from the main table**.

## 7. Appendix — 4-class

Gold-type support over test+cal: safe 900, fraud_assistance 873,
over_refusal 17, refusal_failure 10.
Classes with support < 30 (over_refusal, refusal_failure) are not stably comparable; no 4-class
conclusion is drawn in the main body.
