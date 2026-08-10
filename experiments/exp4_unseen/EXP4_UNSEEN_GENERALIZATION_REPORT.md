# Experiment 4: Strict Unseen Generalization of the Distilled Student (v2)

Protocol ID: `e4v2_FINAL` | Date: 2026-08-10 15:18:23

## 1. Setup
- Primary: FraudDistill-Student-1.5B (best_step120), frozen threshold 0.5622, max_length 512.
- Comparators: Neural-Gold (0.5, 384), Neural-SoftDistill (0.5, 384), Base-1.5B-ZeroShot (300-row fixed subset, seed 20260808).
- Frozen test N=1200 (Level-3 strict unseen), consumed once (TEST_CONSUME_TOKEN). Calibration reserve N=600 used only by E5.
- Shifts: U1 unseen category (elder_health_product, naked_chat_sextortion); U2 unseen source (Aegis validation, PKU-SafeRLHF); U3 unseen target model/style (SmolLM2-1.7B, Phi-3.5-mini).
- Exposure audit per guide 4.3 (exact/family/template gates) + near-duplicate scan; all formal rows passed.

## 2. Main results (pooled)
| model | n | macro_f1 | recall | fpr | mcc | auprc | auroc | 4cl-MF1 | StrictRecall |
|---|---|---|---|---|---|---|---|---|---|
| Final Student (0.5622) | 1200 | 0.3329 | 0.2133 | 0.0683 | 0.2084 | 0.7044 | 0.7198 | 0.2461 | 0.2163 |
| Neural-Gold (0.5) | 1200 | 0.4486 | 0.3633 | 0.2567 | 0.1153 | 0.5649 | 0.5725 | 0.2428 | 0.3633 |
| Neural-SoftDistill (0.5) | 1200 | 0.4708 | 0.4033 | 0.31 | 0.0974 | 0.5913 | 0.6224 | 0.2594 | 0.4152 |
| Base-1.5B-ZeroShot | 300 | 0.6784 | 1.0 | 1.0 | 0.0 | 0.4468 | — | 0.1602 | 1.0 |


## 3. Per-shift results
### U1_category (N=400)

| model | n | macro_f1 | recall | fpr | mcc | auprc | auroc |
|---|---|---|---|---|---|---|---|
| final_student | 400 | 0.3226 | 0.2 | 0.04 | 0.2462 | 0.774 | 0.7959 |
| neural_gold | 400 | 0.6507 | 0.68 | 0.41 | 0.2711 | 0.6169 | 0.6818 |
| neural_softdistill | 400 | 0.5564 | 0.53 | 0.375 | 0.1557 | 0.636 | 0.65 |
| base_zeroshot | 98 | 0.6099 | 1.0 | 1.0 | 0.0 | 0.4275 | — |

### U2_source (N=400)

| model | n | macro_f1 | recall | fpr | mcc | auprc | auroc |
|---|---|---|---|---|---|---|---|
| final_student | 400 | 0.2846 | 0.185 | 0.115 | 0.098 | 0.5821 | 0.5963 |
| neural_gold | 400 | 0.0566 | 0.03 | 0.03 | 0.0 | 0.4803 | 0.4688 |
| neural_softdistill | 400 | 0.2703 | 0.175 | 0.12 | 0.0776 | 0.6086 | 0.6403 |
| base_zeroshot | 106 | 0.6914 | 1.0 | 1.0 | 0.0 | 0.531 | — |

### U3_target_style (N=400)

| model | n | macro_f1 | recall | fpr | mcc | auprc | auroc |
|---|---|---|---|---|---|---|---|
| final_student | 400 | 0.3908 | 0.255 | 0.05 | 0.2851 | 0.7763 | 0.7567 |
| neural_gold | 400 | 0.4444 | 0.38 | 0.33 | 0.0522 | 0.5684 | 0.5731 |
| neural_softdistill | 400 | 0.5206 | 0.505 | 0.435 | 0.0701 | 0.5645 | 0.5936 |
| base_zeroshot | 96 | 0.7285 | 1.0 | 1.0 | 0.0 | 0.4438 | — |



## 4. System-level deployment view (Final Student + selective audit)
| Variant | API rate | n | macro_f1 | recall | precision | fpr | mcc | auprc | auroc |
|---|---|---|---|---|---|---|---|---|---|
| Student+Audit 60 | 0.05 | 60 | 0.3915 | 0.26 | 0.7919 | 0.0683 | 0.2587 | 0.7044 | 0.7198 |
| Student+Audit 120 | 0.1 | 120 | 0.4396 | 0.3 | 0.8219 | 0.065 | 0.3042 | 0.7044 | 0.7198 |
| Student+Audit 180 | 0.15 | 180 | 0.4777 | 0.33 | 0.8646 | 0.0517 | 0.3542 | 0.7044 | 0.7198 |
| Student+Audit 240 | 0.2 | 240 | 0.5211 | 0.37 | 0.881 | 0.05 | 0.3928 | 0.7044 | 0.7198 |
| Student+Audit 300 | 0.25 | 300 | 0.5659 | 0.415 | 0.8893 | 0.0517 | 0.4295 | 0.7044 | 0.7198 |
| Student+Audit 360 | 0.3 | 360 | 0.6055 | 0.4567 | 0.8984 | 0.0517 | 0.4651 | 0.7044 | 0.7198 |
| Student+Audit 420 | 0.35 | 420 | 0.6417 | 0.5 | 0.8955 | 0.0583 | 0.4923 | 0.7044 | 0.7198 |
| Student+Audit 480 | 0.4 | 480 | 0.6791 | 0.545 | 0.9008 | 0.06 | 0.5279 | 0.7044 | 0.7198 |
| Student+Audit 540 | 0.45 | 540 | 0.7114 | 0.5917 | 0.892 | 0.0717 | 0.5522 | 0.7044 | 0.7198 |
| Student+Audit 600 | 0.5 | 600 | 0.7301 | 0.6267 | 0.8744 | 0.09 | 0.5596 | 0.7044 | 0.7198 |


## 4. Paired significance (10k family-cluster bootstrap, exact McNemar, Holm)

| comparison | metric | delta | ci95 | mcnemar_p |
|---|---|---|---|---|
| final_student vs neural_gold | macro_f1 | -0.11566 | [-0.1772, -0.0532] | 0.264635 |
| final_student vs neural_gold | recall | -0.15004 | [-0.2055, -0.0938] | 0.264635 |
| final_student vs neural_gold | fpr | -0.18837 | [-0.2319, -0.1445] | 0.264635 |
| final_student vs neural_softdistill | macro_f1 | -0.13782 | [-0.1940, -0.0818] | 0.145522 |
| final_student vs neural_softdistill | recall | -0.18987 | [-0.2404, -0.1385] | 0.145522 |
| final_student vs neural_softdistill | fpr | -0.24175 | [-0.2892, -0.1938] | 0.145522 |

Holm-adjusted: {"final_student_vs_neural_gold": {"p_raw": 0.264635, "p_holm": 0.264635}, "final_student_vs_neural_softdistill": {"p_raw": 0.145522, "p_holm": 0.291044}}


## 5. Discussion
- Raw-model results above define the deployment boundary of the 1.5B student under strict unseen transfer (no tuning, no API): ranking is meaningful (AUROC 0.72) but recall is limited.
- Section 4 shows the practical system: routing the most ambiguous samples to a single DeepSeek audit lifts MF1 from 0.333 to 0.478 at 15% API rate (0.566 at 25%, 0.730 at 50%), while keeping FPR at 0.052-0.09. See Experiment 5 for the full P3 protocol, statistics and cost.
- Final Student vs Neural-Gold / Neural-SoftDistill: bootstrap CIs + McNemar above.
- U3 (target model/style shift) is expected to be the hardest shift.
- Base-1.5B zero-shot (300 subset) is the untrained reference (H4-a).

## 6. Artifacts
- Manifests/hashes: `manifests/`; predictions: `predictions/`; audits: `audits/`; tables/figures: `tables/`, `figures/`.
