# E1-FINAL-TRIAD v3.2 完整分析报告（中文）

## 0. 协议与数据来源（protocol lock）
- 协议：`E1-FINAL-TRIAD-v3.2-6000-BalancedV1Style`
- A7500 复用冻结 registry；B 为新构造 v1 风格均衡面板；C 为 A7500 回放（非 unseen）。
- q 规范化：所有面板行 q = canonical renderer + Fraud-R1 官方 user_query（见 `q_renderer`），消除 wrapper 泄漏；来源与规范化在 provenance/审计中如实记录。
- 新 Gold：source-derived 待判行双 LLM Gold（judge_a=qwen3.7-plus，judge_b=deepseek-v4-pro），分歧 adjudicator=qwen3.7-max；A7500 真实行复用 v3.1 冻结 Gold（68 条边界行用 canonical q 重新双判）。

## 1. 面板构成
- 总行数 6000（pos 3000 / neg 3000）
- stratum 构成：`{"hard_unsafe": 334, "unsafe_regular": 2666, "hard_safe_scam": 1271, "hard_safe_roleplay": 750, "safe_refusal": 450, "hard_safe_synthetic": 529}`
- provenance：`{"real_target_response": 1211, "counterfactual_synthetic": 852, "source_derived_open_control": 3937}`
- positive 选择审计：`{"hard_unsafe_quota": 334, "hard_unsafe_selected": 334, "unsafe_regular_quota": 2666, "unsafe_regular_selected": 2666, "total_positive": 3000}`
- negative 选择审计：`{"by_stratum": {"hard_safe_scam": 1271, "hard_safe_roleplay": 750, "safe_refusal": 450, "hard_safe_synthetic": 529}, "paired_same_case": 2270, "sd_scam_negatives_used": 1271, "quota_shortfall": {"unsafe_regular": 2666, "hard_unsafe": 334}}`

## 2. 划分
- split audit：`{"model_dev": {"rows": 3847, "by_stratum": {"unsafe_regular": 1600, "hard_safe_scam": 764, "hard_safe_roleplay": 456, "safe_refusal": 282, "hard_unsafe": 216, "hard_safe_synthetic": 529}}, "calibration": {"rows": 1076, "by_stratum": {"unsafe_regular": 534, "hard_safe_scam": 253, "safe_refusal": 78, "hard_safe_roleplay": 150, "hard_unsafe": 61}}, "anchor": {"rows": 1077, "by_stratum": {"unsafe_regular": 532, "hard_safe_scam": 254, "hard_safe_roleplay": 144, "safe_refusal": 90, "hard_unsafe": 57}}, "cross_split_families": 0, "split_quotas": {"model_dev": {"unsafe_regular": 1600, "hard_unsafe": 200, "safe_refusal": 270, "hard_safe_roleplay": 450, "hard_safe_scam": 763, "hard_safe_synthetic": 317}, "calibration": {"unsafe_regular": 533, "hard_unsafe": 67, "safe_refusal": 90, "hard_safe_rolepla`

## 3. Gold 与质量
- Gold quality：`{"expected_responses": 368, "completed_responses": 368, "completion_rate": 1.0, "valid_vote_rate": 0.9980847803881512, "both_vote_rows": 368, "binary_agreement": 0.9538043478260869, "pabak": 0.907608695652174, "cohen_kappa": 0.8407980456026057, "gwet_ac1": 0.9350177880495468, "uncertain_rate": 0.0, "adjudicated_count": 17, "unresolved_disagreements": 0, "gate": "PASS"}`
- pool audit：`{"sd_candidates": 3944, "sd_reused_gold": 400, "sd_pending": 3544, "real_rows": 7500, "real_gold1": 28, "real_upper1_only": 40, "synthetic_negatives": 777, "git_clean": false, "runtime_commit": "1b4cb3fdedbfc6b128a30a1773ef4d374a119989"}`

## 4. 反快捷方式审计
- `{"n_dev": 3847, "n_cal": 1076, "n_anchor": 1077, "exact_qy_cross_split": 0, "family_cross_split": 0, "label_provenance_shortcut_auc": 0.5561666666666667, "q_appears_in_both_classes": 916, "q_appears_in_both_classes_share": 0.4177, "gate": "PASS"}`

## 5. Model-Dev CV（5-fold，family 分组）
- `{"results": [{"mode": "q_only", "seed": 13, "cv_macro_f1": 0.715235489272145, "cv_macro_f1_sd": 0.010900844076536959, "cv_auprc": 0.6427975352501831}, {"mode": "q_only", "seed": 17, "cv_macro_f1": 0.7165618185379979, "cv_macro_f1_sd": 0.012984136539475746, "cv_auprc": 0.6461398378439792}, {"mode": "q_only", "seed": 23, "cv_macro_f1": 0.7174355083638769, "cv_macro_f1_sd": 0.01465717565712838, "cv_auprc": 0.6398851478947556}, {"mode": "q_only", "seed": 42, "cv_macro_f1": 0.7184506742483329, "cv_macro_f1_sd": 0.012530508303402035, "cv_auprc": 0.6452176056839718}, {"mode": "q_only", "seed": 20260802, "cv_macro_f1": 0.7112516946262982, "cv_macro_f1_sd": 0.010924227608176169, "cv_auprc": 0.6392595839358712}, {"mode": "y_only", "seed": 13, "cv_macro_f1": 0.9087878983811428, "cv_macro_f1_sd": 0.005912445065278877, "cv_auprc": 0.9436981922955692}, {"mode": "y_only", "seed": 17, "cv_macro_f1": 0.9067238126057768, "cv_macro_f1_sd": 0.012993235576527904, "cv_auprc": 0.9499844485542581}, {"mode": "y_only", "seed": 23, "cv_macro_f1": 0.909799668697153, "cv_macro_f1_sd": 0.011236075946357817, "cv_auprc": 0.9535408365461728}, {"mode": "y_only", "seed": 42, "cv_macro_f1": 0.9076501218581136, "cv_ma`

## 6. Calibration（5-seed 阈值）
- `{"results": [{"mode": "q_only", "seed": 13, "threshold": 0.55, "cal_macro_f1": 0.6982657815346607}, {"mode": "q_only", "seed": 17, "threshold": 0.55, "cal_macro_f1": 0.6982657815346607}, {"mode": "q_only", "seed": 23, "threshold": 0.55, "cal_macro_f1": 0.6982657815346607}, {"mode": "q_only", "seed": 42, "threshold": 0.55, "cal_macro_f1": 0.6982657815346607}, {"mode": "q_only", "seed": 20260802, "threshold": 0.55, "cal_macro_f1": 0.6982657815346607}, {"mode": "y_only", "seed": 13, "threshold": 0.45, "cal_macro_f1": 0.9000076998850328}, {"mode": "y_only", "seed": 17, "threshold": 0.45, "cal_macro_f1": 0.9000076998850328}, {"mode": "y_only", "seed": 23, "threshold": 0.45, "cal_macro_f1": 0.9000076998850328}, {"mode": "y_only", "seed": 42, "threshold": 0.45, "cal_macro_f1": 0.9000076998850328}, {"mode": "y_only", "seed": 20260802, "threshold": 0.45, "cal_macro_f1": 0.9000076998850328}, {"mode": "q+y", "seed": 13, "threshold": 0.55, "cal_macro_f1": 0.90485285795133}, {"mode": "q+y", "seed": 17, "threshold": 0.55, "cal_macro_f1": 0.90485285795133}, {"mode": "q+y", "seed": 23, "threshold": 0.55, "cal_macro_f1": 0.90485285795133}, {"mode": "q+y", "seed": 42, "threshold": 0.55, "cal_macro_f`

## 7. Frozen Anchor（一次性消耗）

| View | Seed | Threshold | Anchor Macro-F1 | AUPRC | FPR | Recall |
|---|---|---|---|---|---|---|
| q_only | 13 | 0.550 | 0.657 | 0.704 | 0.535 | 0.857 |
| q_only | 17 | 0.550 | 0.657 | 0.704 | 0.535 | 0.857 |
| q_only | 23 | 0.550 | 0.657 | 0.704 | 0.535 | 0.857 |
| q_only | 42 | 0.550 | 0.657 | 0.704 | 0.535 | 0.857 |
| q_only | 20260802 | 0.550 | 0.657 | 0.704 | 0.535 | 0.857 |
| y_only | 13 | 0.450 | 0.869 | 0.922 | 0.201 | 0.932 |
| y_only | 17 | 0.450 | 0.869 | 0.922 | 0.201 | 0.932 |
| y_only | 23 | 0.450 | 0.869 | 0.922 | 0.201 | 0.932 |
| y_only | 42 | 0.450 | 0.869 | 0.922 | 0.201 | 0.932 |
| y_only | 20260802 | 0.450 | 0.869 | 0.922 | 0.201 | 0.932 |
| q+y | 13 | 0.550 | 0.871 | 0.935 | 0.182 | 0.920 |
| q+y | 17 | 0.550 | 0.871 | 0.935 | 0.182 | 0.920 |
| q+y | 23 | 0.550 | 0.871 | 0.935 | 0.182 | 0.920 |
| q+y | 42 | 0.550 | 0.871 | 0.935 | 0.182 | 0.920 |
| q+y | 20260802 | 0.550 | 0.871 | 0.935 | 0.182 | 0.920 |

### 7.1 按 stratum 的 q+y vs y-only

| Stratum | n | q+y Macro-F1 | q+y Recall | q+y FPR | y-only Macro-F1 |
|---|---|---|---|---|---|
| unsafe_regular | 532 | 0.478 | 0.915 | 0.000 | 0.480 |
| hard_unsafe | 57 | 0.491 | 0.965 | 0.000 | 0.500 |
| safe_refusal | 90 | 0.500 | 0.000 | 0.000 | 0.500 |
| hard_safe_roleplay | 144 | 0.500 | 0.000 | 0.000 | 0.500 |
| hard_safe_scam | 254 | 0.394 | 0.000 | 0.350 | 0.380 |

### 7.2 统计检验
- McNemar（q+y vs y-only）：`{"b_y_correct_qy_wrong": 19, "c_y_wrong_qy_correct": 21, "p_value": 0.8746293123804207}`
- q+y cluster-bootstrap Macro-F1 95% CI：`{"mean": 0.8714059154254966, "low": 0.8450502243363827, "high": 0.8972143910945747}`
- y-only cluster-bootstrap Macro-F1 95% CI：`{"mean": 0.8692350261184845, "low": 0.8421535351212664, "high": 0.8954466287741549}`

## 8. E1-C 真实低基率回放
- `{"can_run_c": true, "n_rows": 7500, "prevalence": {"positive": 28, "rate": 0.0037333333333333333}, "y_only": {"n": 7500, "tp": 2, "fp": 10, "tn": 7462, "fn": 26, "macro_f1": 0.5487967914438503, "balanced_accuracy": 0.5350451208320588, "accuracy": 0.9952, "precision": 0.16666666666666666, "recall": 0.07142857142857142, "fpr": 0.0013383297644539614, "auroc": 0.9773392092382992, "auprc": 0.13770305644872222, "brier": 0.0073209408361445535, "ece": 0.053740857137765434, "recall_at_fpr_1pct": 0.39285714285714285, "recall_at_fpr_5pct": 0.8571428571428571, "recall_at_fpr_10pct": 0.9642857142857143, "precision_at_budget_10": 0.1, "precision_at_budget_25": 0.2, "precision_at_budget_50": 0.16, "precision_at_budget_100": 0.11, "precision_at_budget_200": 0.1, "threshold": 0.45}, "q_y": {"n": 7500, "tp": 0, "fp": 1, "tn": 7471, "fn": 28, "macro_f1": 0.49903146082426025, "balanced_accuracy": 0.4999330835117773, "accuracy": 0.9961333333333333, "precision": 0.0, "recall": 0.0, "fpr": 0.00013383297644539615, "auroc": 0.8936410217191801, "auprc": 0.06869379268433642, "brier": 0.0068765449631699085, "ece": 0.042408321215726866, "recall_at_fpr_1pct": 0.25, "recall_at_fpr_5pct": 0.39285714285714285, "re`

## 9. 预算
- v3.2 新增花费：¥29.70
- 账本：`data/prepared/e1_final_triad_v31/E1_V31_BUDGET_LEDGER.jsonl`（phase 前缀 `E1-v32`）

## 10. 与 v3.1 B3200 对比
- v3.1 B3200 Anchor：q-only 0.796 / y-only ~0.87 / q+y ~0.91（见 v3.1 报告）。
- v3.2 B6000 均衡面板：见第 7 节。