# E1-FINAL-TRIAD v3.2 均衡面板（B 重建）执行报告

- 协议：`E1-FINAL-TRIAD-v3.2-6000-BalancedV1Style`
- 运行提交：`4272f8dd7e05762667a48377f4c8915d5377db81`（worktree dirty）
- v3.2 新增 API 花费：¥29.70（硬停 ¥30）

## 1. 面板构成（v1 预印本 hard-control 风格，均衡）
- 总行数：6000；positive 3000 / negative 3000
- stratum：`{"hard_unsafe": 334, "unsafe_regular": 2666, "hard_safe_scam": 1271, "hard_safe_roleplay": 750, "safe_refusal": 450, "hard_safe_synthetic": 529}`
- provenance：`{"real_target_response": 1211, "counterfactual_synthetic": 852, "source_derived_open_control": 3937}`
- language：`{"en": 3037, "zh": 2963}`
- category：`{"network_friendship": 11, "fake_job_posting": 154, "phishing": 1262, "impersonation": 1630, "fake job posting": 705, "fraudulent service": 1246, "counterfactual": 558, "fraudulent_service": 434}`

## 2. Gold 质量
- {"expected_responses": 368, "completed_responses": 368, "completion_rate": 1.0, "valid_vote_rate": 0.9980847803881512, "both_vote_rows": 368, "binary_agreement": 0.9538043478260869, "pabak": 0.907608695652174, "cohen_kappa": 0.8407980456026057, "gwet_ac1": 0.9350177880495468, "uncertain_rate": 0.0, "adjudicated_count": 17, "unresolved_disagreements": 0, "gate": "PASS"}
- consensus：{"consensus_rows": 368, "known_rows": 368, "missing_vote_rows": 0, "sd_gold1": 2666, "sd_gold0": 1271, "real_gold1": 11, "real_gold0": 7489, "synth_gold1": 330, "synth_gold0": 777, "synth_pending": 0, "quality": {"expected_responses": 368, "completed_responses": 368, "completion_rate": 1.0, "valid_vote_rate": 0.9980847803881512, "both_vote_rows": 368, "binary_agreement": 0.9538043478260869, "pabak

## 3. Frozen Anchor 主结果（一次性消耗，5-seed）
| View | Macro-F1 (mean±sd) | AUPRC | FPR | Recall |
|---|---|---|---|---|
| q_only | 0.657±0.000 | 0.704 | 0.535 | 0.857 |
| y_only | 0.869±0.000 | 0.922 | 0.201 | 0.932 |
| q+y | 0.871±0.000 | 0.935 | 0.182 | 0.920 |

- q+y vs y-only McNemar：`{"b_y_correct_qy_wrong": 19, "c_y_wrong_qy_correct": 21, "p_value": 0.8746293123804207}`
- q+y cluster bootstrap 95% CI：`{"mean": 0.8714059154254966, "low": 0.8450502243363827, "high": 0.8972143910945747}`
- y-only cluster bootstrap 95% CI：`{"mean": 0.8692350261184845, "low": 0.8421535351212664, "high": 0.8954466287741549}`
- 错误转移：`{"y_wrong_qy_correct": 21, "y_correct_qy_wrong": 19}`

## 4. 反快捷方式审计
- `{"n_dev": 3847, "n_cal": 1076, "n_anchor": 1077, "exact_qy_cross_split": 0, "family_cross_split": 0, "label_provenance_shortcut_auc": 0.5561666666666667, "q_appears_in_both_classes": 916, "q_appears_in_both_classes_share": 0.4177, "gate": "PASS"}`

## 5. C 真实低基率回放（A7500）
- {"can_run_c": true, "n_rows": 7500, "prevalence": {"positive": 28, "rate": 0.0037333333333333333}, "y_only": {"n": 7500, "tp": 2, "fp": 10, "tn": 7462, "fn": 26, "macro_f1": 0.5487967914438503, "balanced_accuracy": 0.5350451208320588, "accuracy": 0.9952, "precision": 0.16666666666666666, "recall": 0.07142857142857142, "fpr": 0.0013383297644539614, "auroc": 0.9773392092382992, "auprc": 0.13770305644872222, "brier": 0.0073209408361445535, "ece": 0.053740857137765434, "recall_at_fpr_1pct": 0.39285714285714285, "recall_at_fpr_5pct": 0.8571428571428571, "recall_at_fpr_10pct": 0.9642857142857143, "precision_at_budget_10": 0.1, "precision_at_budget_25": 0.2, "precision_at_budget_50": 0.16, "precision_at_budget_100": 0.11, "precision_at_budget_200": 0.1, "threshold": 0.45}, "q_y": {"n": 7500, "tp": 0, "fp": 14, "tn": 7458, "fn": 28, "macro_f1": 0.49859606899318093, "balanced_accuracy": 0.4990631
- 说明：E1-C is NOT an unseen generalization experiment; it replays the frozen v3.2 B detector on the A7500 real distribution.

## 6. 结论
- 主 seed=13：q-only Macro-F1 = 0.657；y-only = 0.869；q+y = 0.871。叙事排序 成立（q-only < y-only < q+y）。