# E4/E5 Final Summary (frozen static-fix)

> 2026-08-10 · offline recompute · zero new API calls
> Source experiment revision: `86e348d` · Static-fix implementation base: `2ffbf9b`
> Artifact release tag: `e4e5-staticfix-v1`
> Frozen manifests: test **1200 / 557 families**, calibration **600 / 243 families** (canonical JSON-record
> SHA256, verified `sha256_match: true`); predictions filtered by manifest-ID join (1425/686 cached rows excluded).
> Gold: dual-judge (DeepSeek+Qwen), raw agreement 0.8733, κ 0.7485
> (agreed 1572 / third-opinion 180 / deterministic 48; no human verification).

## E4 — family-disjoint held-out composite shifts

| Model | Eval N | Families | F1-unsafe | Macro-F1 | Recall | FPR | MCC | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Final Student | 1200 | 557 | 0.3329 | 0.5092 | 0.2133 | 0.0683 | 0.2084 | 0.7198 | 0.7044 |
| Neural-Gold | 1200 | 557 | 0.4486 | 0.5366 | 0.3633 | 0.2567 | 0.1153 | 0.5725 | 0.5649 |
| Neural-SoftDistill | 1200 | 557 | 0.4708 | 0.5372 | 0.4033 | 0.3100 | 0.0974 | 0.6224 | 0.5913 |

## E5 — calibration & selective audit

| Policy | Eval N | Cal N | Audited K | API rate | F1-unsafe | Macro-F1 | Recall | FPR | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (Final Student) | 1200 | 600 | 0 | 0.00 | 0.3329 | 0.5092 | 0.2133 | 0.0683 | 0.2084 |
| P1 (temp 5.0) | 1200 | 600 | 0 | 0.00 | 0.1323 | 0.4050 | 0.0717 | 0.0117 | 0.1501 |
| P2 (all-safe) | 1200 | 600 | 0 | 0.00 | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| P3 (K=180, primary) | 1200 | 600 | 180 | 0.15 | 0.4777 | 0.6010 | 0.3300 | 0.0517 | 0.3542 |

## Key statistics (10,000 family-cluster bootstrap, seed 20260808)

Point estimates are full-sample values; CIs are family-cluster bootstrap 95% intervals.
Empirical bootstrap p-values are not reported (10,000 replicates cannot resolve p below ~1e-4).

- P3 vs P0: ΔMacro-F1 +0.0918 (95% CI [0.0722, 0.1119]),
  ΔF1-unsafe +0.1448 (95% CI [0.1133, 0.1772]),
  ΔRecall +0.1167 (95% CI [0.0903, 0.1433]),
  ΔFPR −0.0167 (95% CI [−0.0287, −0.0051]),
  ΔMCC +0.1457 (95% CI [0.1126, 0.1800]).
- **Exact McNemar test of paired correctness** (P3 vs P0): b=5, c=85,
  p=7.53×10⁻²⁰.
- P1 vs P0 (negative result): ΔMacro-F1 −0.1042 (95% CI [−0.1286, −0.0792]),
  ΔRecall −0.1417 (95% CI [−0.1712, −0.1123]),
  ΔFPR −0.0567 (95% CI [−0.0780, −0.0375]).
- E4 model pairs: family-cluster bootstrap CIs for the Macro-F1 differences included zero;
  paired correctness differences were not significant under exact McNemar tests after Holm
  correction (adjusted p = 0.2910 for both comparisons).

## Conclusion

The distilled student retained moderate ranking ability under family-disjoint held-out composite
shifts (AUROC 0.7198), but its frozen operating point traded
recall for a low false-positive rate. A score-ambiguity selective audit at a 15% query rate improved
unsafe-class F1, macro-F1, recall, FPR, and MCC using cached single-judge decisions. The cascade
result is interpreted as a system-level recovery mechanism rather than evidence of deployment
readiness.

## Deliverables in this folder

- `FINAL_METRICS.json` / `FINAL_PAIRED_STATS.json` / `FINAL_DATA_AUDIT.json` / `FINAL_GOLD_QUALITY.json`
- `tables/` (4 corrected tables) · `figures/` (PR curve, P3 sensitivity)
- `FINAL_CHANGELOG.md` · `EXP4_UNSEEN_GENERALIZATION_REPORT_FINAL.md` · `EXP5_CALIBRATION_REPORT_FINAL.md`
- Old reports/tables/figures and N=6 gold audit archived under `experiments/archive/e4e5_pre_staticfix/`.
