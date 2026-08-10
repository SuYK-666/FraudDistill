# EXP5 — Calibration & Selective Audit (FINAL static-fix version)

> Frozen offline pass · 2026-08-10 · commit `86e348dd67df9f4277575dfa3589779f3080d882` · zero new API calls
> Scope: P0 (frozen student), P1 (temperature calibration), P2 (selective abstention),
> P3 (score-ambiguity selective audit with cached DeepSeek decisions).

## 1. Main table (Eval N=1200, Cal N=600)

| Policy | Eval N | Cal N | Audited K | API rate | F1-unsafe | Macro-F1 | Recall | FPR | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (Final Student) | 1200 | 600 | 0 | 0.00 | 0.3329 | 0.5092 | 0.2133 | 0.0683 | 0.2084 |
| P1 (temp 5.0) | 1200 | 600 | 0 | 0.00 | 0.1323 | 0.4050 | 0.0717 | 0.0117 | 0.1501 |
| P2 (all-safe) | 1200 | 600 | 0 | 0.00 | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| P3 (K=180, primary) | 1200 | 600 | 180 | 0.15 | 0.4777 | 0.6010 | 0.3300 | 0.0517 | 0.3542 |

- **P0** is the frozen operating point of the Final Student (`risk_score >= 0.5622`).
- **P1** (temperature 5.0 + risk threshold 0.6106, fitted on
  calibration) is a **negative result**: it trades recall for lower FPR.
  ΔMacro-F1(P1−P0) = -0.1042, ΔRecall = -0.1417,
  ΔFPR = -0.0567, ΔBrier ≈ -0.1029,
  ΔECE ≈ -0.1867 (new − baseline).
- **P2** (all-safe selective abstention) is a degenerate negative result (coverage 1.0, no abstention).
- **P3** is the primary reported operating point: **K=180 / 15% query rate** selected by the
  **score-ambiguity heuristic** (min |risk_score − 0.5|, from cached decisions), **not** a
  pre-registered optimum and **not** re-selected from test.

## 2. P3 vs P0 (paired, 10,000 family-cluster replicates)

- Δmacro_f1 = +0.0917 (95% CI [0.0722, 0.1119], p<1e-19)  (ΔMacro-F1 point ≈ +0.0918)
- Δf1_unsafe = +0.1446 (95% CI [0.1133, 0.1772], p<1e-19)  (ΔF1-unsafe ≈ +0.1448)
- Δrecall = +0.1165 (95% CI [0.0903, 0.1433], p<1e-19)  (ΔRecall ≈ +0.1167)
- Δfpr = -0.0167 (95% CI [-0.0287, -0.0051], p=0.0072)  (ΔFPR ≈ −0.0167)
- Δmcc = +0.1456 (95% CI [0.1126, 0.1800], p<1e-19)  (ΔMCC ≈ +0.1458)
- McNemar: b=5, c=85
  (p<1e-19) — the cascade corrects 85 errors while introducing 5.
- Score-based AUROC/AUPRC are unchanged by the cascade by construction and are labeled
  **Student-score AUROC/AUPRC** in the sensitivity table.

## 3. P3 sensitivity (K=60–600, API rate 5–50%)

| Policy | Eval N | Audited K | API rate | F1-unsafe | Macro-F1 | Recall | FPR | MCC | Student-score AUROC | Student-score AUPRC | Judge-agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3_K60 | 1200 | 60 | 0.05 | 0.3915 | 0.5445 | 0.2600 | 0.0683 | 0.2587 | 0.7198 | 0.7044 | 0.8667 |
| P3_K120 | 1200 | 120 | 0.10 | 0.4396 | 0.5746 | 0.3000 | 0.0650 | 0.3042 | 0.7198 | 0.7044 | 0.8583 |
| P3_K180 | 1200 | 180 | 0.15 | 0.4777 | 0.6010 | 0.3300 | 0.0517 | 0.3542 | 0.7198 | 0.7044 | 0.8722 |
| P3_K240 | 1200 | 240 | 0.20 | 0.5211 | 0.6288 | 0.3700 | 0.0500 | 0.3928 | 0.7198 | 0.7044 | 0.8708 |
| P3_K300 | 1200 | 300 | 0.25 | 0.5659 | 0.6573 | 0.4150 | 0.0517 | 0.4295 | 0.7198 | 0.7044 | 0.8767 |
| P3_K360 | 1200 | 360 | 0.30 | 0.6055 | 0.6834 | 0.4567 | 0.0517 | 0.4651 | 0.7198 | 0.7044 | 0.8806 |
| P3_K420 | 1200 | 420 | 0.35 | 0.6417 | 0.7065 | 0.5000 | 0.0583 | 0.4923 | 0.7198 | 0.7044 | 0.8690 |
| P3_K480 | 1200 | 480 | 0.40 | 0.6791 | 0.7320 | 0.5450 | 0.0600 | 0.5279 | 0.7198 | 0.7044 | 0.8750 |
| P3_K540 | 1200 | 540 | 0.45 | 0.7114 | 0.7530 | 0.5917 | 0.0717 | 0.5522 | 0.7198 | 0.7044 | 0.8648 |
| P3_K600 | 1200 | 600 | 0.50 | 0.7301 | 0.7636 | 0.6267 | 0.0900 | 0.5596 | 0.7198 | 0.7044 | 0.8500 |

> K=180 is the primary reported operating point (15% query rate, score-ambiguity heuristic min |risk_score-0.5|); K=60-600 are sensitivity only, not used for selection.


The K=180 row is the **primary reported operating point**; the rest of the curve is sensitivity
analysis and was not used for selection.

## 4. Gold-quality / evaluator sensitivity (K=180 audited rows)

- Among the 180 audited rows, 164 have dual-judge-agreed Gold
  (P3-judge agreement 0.8841,
  Student agreement 0.4634);
  third-opinion 13,
  deterministic 3.
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
shifts (AUROC 0.7198), but its frozen operating point traded
recall for a low false-positive rate. A score-ambiguity selective audit at a 15% query rate improved
unsafe-class F1, macro-F1, recall, FPR, and MCC using cached single-judge decisions. The cascade
result is interpreted as a **system-level recovery mechanism** rather than evidence of deployment
readiness.
