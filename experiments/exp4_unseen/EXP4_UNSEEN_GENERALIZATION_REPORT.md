# Experiment 4: Strict Unseen Generalization of the Distilled Student (v2)

Protocol ID: `e4v2_probe3` | Date: 2026-08-09 13:32:57

## 1. Setup
- Primary: FraudDistill-Student-1.5B (best_step120), frozen threshold 0.5622, max_length 512.
- Comparators: Neural-Gold (0.5, 384), Neural-SoftDistill (0.5, 384), Base-1.5B-ZeroShot (300-row fixed subset, seed 20260808).
- Frozen test N=400 (Level-3 strict unseen), consumed once (TEST_CONSUME_TOKEN). Calibration reserve N=253 used only by E5.
- Shifts: U1 unseen category (elder_health_product, naked_chat_sextortion); U2 unseen source (Aegis validation, PKU-SafeRLHF); U3 unseen target model/style (SmolLM2-1.7B, Phi-3.5-mini).
- Exposure audit per guide 4.3 (exact/family/template gates) + near-duplicate scan; all formal rows passed.

## 2. Main results (pooled)
| model | n | macro_f1 | recall | fpr | mcc | auprc | auroc | 4cl-MF1 | StrictRecall |
|---|---|---|---|---|---|---|---|---|---|


## 3. Per-shift results



## 5. Discussion
- Final Student vs Neural-Gold / Neural-SoftDistill: bootstrap CIs + McNemar above.
- U3 (target model/style shift) is expected to be the hardest shift.
- Base-1.5B zero-shot (300 subset) is the untrained reference (H4-a).

## 6. Artifacts
- Manifests/hashes: `manifests/`; predictions: `predictions/`; audits: `audits/`; tables/figures: `tables/`, `figures/`.
