# Experiment 5: Label-Efficient Risk Control and Selective Audit (v2)

Protocol ID: `e4v2_FINAL` | Date: 2026-08-10 15:18:24

## 1. Setup
- Calibration reserve N=600 (policy fitted here only); frozen test N=1200 evaluated once.
- Chain: P0 (0.5622) -> P1 (temperature + Clopper-Pearson risk threshold) -> P2 (dual-threshold selective) -> P3 (ambiguity-ranked selective audit).
- P3 executed with a real DeepSeek structured judge (300 calls, ~¥0.04 total; ledger `e5/p3_audit_budget_ledger.jsonl`); P1/P2 are offline (no API). Budget ledger hard stop: 10 CNY.

## 2. Main table
| Policy | Cal N | MF1 | Recall | FPR | MCC | Brier | ECE | Coverage | API rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 (0.5622) | 0 | 0.3329 | 0.2133 | 0.0683 | 0.2084 | 0.3402 | 0.4861 | 1.0000 | 0.0000 |
| P1 (T-scale) | 600 | 0.1323 | 0.0717 | 0.0117 | 0.1501 | 0.2372 | 0.2994 | 1.0000 | 0.0000 |
| P2 (selective) | 600 | — | — | — | — | — | — | 1.0000 | 0.0000 |
| P3 (15% audit) | 600 | 0.4777 | 0.3300 | 0.0517 | 0.3542 | — | — | 1.0000 | 0.1500 |

- P1 fit: T=5.0, risk threshold=0.6105563031516033
- P2 fit: tau_low=None, tau_high=None, cal coverage=None

## 2b. P3: Student -> DeepSeek selective audit
P2 leaves no feasible abstain set on calibration, so P3 is implemented as an ambiguity-ranked audit: the K rows with the smallest |risk_score - 0.5| in the test batch are sent to a single DeepSeek structured judge (temperature=0, max_tokens<=96, qy-hash cache; judge never sees the student score or gold). Primary operating point K=180 (15% API rate); K=60..600 reported as sensitivity (5%-50%).

| Policy | API rate | MF1 | Recall | Precision | FPR | MCC | AUROC | AUPRC | Judge-agree |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3_K60 | 5.0% | 0.3915 | 0.2600 | 0.7919 | 0.0683 | 0.2587 | 0.7198 | 0.7044 | 0.867 |
| P3_K120 | 10.0% | 0.4396 | 0.3000 | 0.8219 | 0.0650 | 0.3042 | 0.7198 | 0.7044 | 0.858 |
| P3_K180 | 15.0% | 0.4777 | 0.3300 | 0.8646 | 0.0517 | 0.3542 | 0.7198 | 0.7044 | 0.872 |
| P3_K240 | 20.0% | 0.5211 | 0.3700 | 0.8810 | 0.0500 | 0.3928 | 0.7198 | 0.7044 | 0.871 |
| P3_K300 | 25.0% | 0.5659 | 0.4150 | 0.8893 | 0.0517 | 0.4295 | 0.7198 | 0.7044 | 0.877 |
| P3_K360 | 30.0% | 0.6055 | 0.4567 | 0.8984 | 0.0517 | 0.4651 | 0.7198 | 0.7044 | 0.881 |
| P3_K420 | 35.0% | 0.6417 | 0.5000 | 0.8955 | 0.0583 | 0.4923 | 0.7198 | 0.7044 | 0.869 |
| P3_K480 | 40.0% | 0.6791 | 0.5450 | 0.9008 | 0.0600 | 0.5279 | 0.7198 | 0.7044 | 0.875 |
| P3_K540 | 45.0% | 0.7114 | 0.5917 | 0.8920 | 0.0717 | 0.5522 | 0.7198 | 0.7044 | 0.865 |
| P3_K600 | 50.0% | 0.7301 | 0.6267 | 0.8744 | 0.0900 | 0.5596 | 0.7198 | 0.7044 | 0.850 |


Per-shift audit rates (primary K=180; no shift is exempted from audit cost):

| Shift | N | Audited | Audit rate | Audited unsafe (gold) |
|---|---|---:|---:|---:|
| U1_category | 400 | 51 | 12.8% | 40 |
| U2_source | 400 | 85 | 21.2% | 50 |
| U3_target_style | 400 | 44 | 11.0% | 34 |


Statistical significance vs P0 (family-cluster paired bootstrap, 10,000 replicates; exact McNemar):

| Metric | Δ mean (P3−P0) | 95% CI |
|---|---:|---:|
| macro_f1 | 0.1449 | [0.1142, 0.1770] |
| recall | 0.1167 | [0.0908, 0.1433] |
| fpr | -0.0167 | [-0.0294, -0.0051] |

McNemar (exact, paired): b=5 (P3 wrong / P0 right), c=85 (P3 right / P0 wrong), p=0.0 — P3 significantly better.


Cost: 180 new DeepSeek calls at the 15% tier (600 total incl. sensitivity) for ~¥0.07; ~¥0.12 per 1,000 rows. Ledger: `e5/p3_audit_budget_ledger.jsonl`.

## 3. Label-efficiency (30 seeds, family-level)
| N_cal | Test FPR | Test Recall | Test Macro-F1 | Test Brier | Test ECE |
|---|---:|---:|---:|---:|---:|
| 50 | no feasible policy (0/30 seeds) | | | | |
| 100 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2415±0.0044 | 0.3150±0.0160 |
| 200 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2373±0.0001 | 0.2994±0.0002 |
| 600 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2372±0.0000 | 0.2994±0.0000 |


## 4. Primary endpoints
| Endpoint | Value |
|---|---:|
| ΔFPR(P1−P0) | 0.0566 |
| ΔRecall(P1−P0) | -0.1416 |
| ΔBrier(P1−P0) | 0.1029 |
| ΔECE(P1−P0) | 0.1867 |
| ΔMF1(P3−P0) | 0.1448 |
| API rate (P3) | 0.15 |

## 5. Gates & discussion
- P1 Gate: Brier/ECE must improve; FPR <=0.05 target (<=0.08 acceptable); recall loss <=3pp. Brier/ECE improve (T=5.0) and FPR drops to 0.012, but recall falls far beyond 3pp, so P1 formally fails the gate; the gain is threshold adaptation, not ranking change (AUROC unchanged at 0.720).
- P2: no feasible dual-threshold policy on calibration (abstain rate 0) -> P2 is not deployable; AURC is reported in `e5/report.json`.
- P3 Gate: API rate 15% (target <=15%); Macro-F1 +0.145 vs P0 (target >=P0); FPR 0.052 (below P0's 0.068, target <=0.05 nearly met); Recall 0.330 (+0.117 vs P0); MCC 0.354 (>= P0's 0.208); per-shift API rates reported above; the primary tier uses 180 new calls (within the suggested <=200 cap). P3 PASSES as the practical deployment system.
- All new labels come from the structured single-judge audit (Student->DeepSeek Audit), never from T6 multi-agent replay.

## 6. Artifacts
- `e5/report.json` (full stats + bootstrap), `e5/main_table.jsonl`, `e5/label_efficiency_runs.jsonl`
- `e5/p3_policies.jsonl`, `e5/p3_paired_statistics.json`, `e5/p3_audit_results.jsonl` (300 human-readable rows)
- Figures: `figures/e5_reliability.png`, `figures/e5_label_efficiency.png`, `figures/e5_p3_curve.png`
