# FINAL_CHANGELOG — E4/E5 Static Fix (frozen)

Date: 2026-08-10 (static offline pass; no new API calls, no new annotation, no test-driven tuning)
Code commit: `2ffbf9b4c4e0b06500c34621727258ed72bbc0c7`
Source protocol dir: `outputs\exp4_unseen_student_v2\e4v2_FINAL`
Seed: 20260808 | Bootstrap replicates: 10000

## Fixed in this pass
1. **Macro-F1**: previously `macro_f1` stored the unsafe-class F1. Now `macro_f1 = (F1-unsafe + F1-safe)/2` via `binary_metrics_raw` (matches `sklearn f1_score(average='macro')`); `f1_unsafe` reported separately. All tables rebuilt from unrounded raw values.
2. **Holm-Bonferroni**: corrected to cumulative max (was cumulative min). E4 two-comparison Holm-adjusted p now ≈ 0.2910.
3. **McNemar**: p-value no longer rounded to 6 decimals (P3 vs P0 exact p = 7.53e-20, previously displayed as 0.0).
4. **Bootstrap**: single-pass vectorized family-cluster bootstrap (10,000 replicates, fixed seed, family-level resampling, paired across models) covering Macro-F1, F1-unsafe, Recall, FPR, MCC (+AUROC/AUPRC for E4 model pairs).
5. **Endpoint sign convention**: all Δ reported as new − baseline (e.g. ΔFPR(P1−P0) = −0.0566).
6. **Data scope**: metrics computed only on frozen manifests (test 1200 / 557 families, cal 600 / 243 families); 1425/686 extra prediction rows excluded by manifest-ID join. Manifest SHA256 recorded in FINAL_DATA_AUDIT.json.
7. **U1/U2 wording**: U2 fully PKU-SafeRLHF (298 general_harm / 102 financial_fraud); U1 269 `???`-suffix queries and language-label correlation disclosed as artifacts/limitations.
8. **Gold**: quality summary recomputed on all 1800 records (raw agreement 0.8733, κ 0.7485, agreed 1572, third-opinion 180, deterministic 48); labeled `dual-judge Gold` (no human verification).
9. **P3**: primary operating point K=180 / 15% API rate via score-ambiguity heuristic; K=60-600 sensitivity only; AUROC/AUPRC columns renamed to Student-score; API cost 600 rows ≈ ¥0.07, historical total ≈ ¥15.3 (original ¥10 hard cap disclosed as protocol deviation).
10. **Base zero-shot** (300-row subset, all-unsafe) and 4-class metrics moved to appendix/limitations; no quantitative comparison against N=1200 rows.

## Invalidated / superseded files
- Old EXP4/EXP5 reports, tables and figures (archived under `experiments/archive/e4e5_pre_staticfix/`).
- `gold_quality_audit_v4.json` (N=6) and old `shortcut_audit` files (archived).
- All previous Macro-F1 CIs (recomputed here).
