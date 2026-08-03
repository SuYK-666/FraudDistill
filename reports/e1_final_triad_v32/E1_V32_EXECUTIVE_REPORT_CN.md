# E1-FINAL-TRIAD v3.2 均衡面板（B 重建）执行报告

- 协议：`E1-FINAL-TRIAD-v3.2-6000-BalancedV1Style`
- 运行提交：`d8558e430e6e99b0c307b4ad33e9860fff5cbff5`（worktree clean）
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

### 3.1 Model-Dev CV（5-fold，family 分组，辅助视角）

| View | Macro-F1 (mean±sd) | AUPRC |
|---|---|---|
| q_only | 0.716±0.003 | 0.643 |
| y_only | 0.909±0.001 | 0.950 |
| q+y | 0.921±0.001 | 0.958 |

## 4. 反快捷方式审计
- `{"n_dev": 3847, "n_cal": 1076, "n_anchor": 1077, "exact_qy_cross_split": 0, "family_cross_split": 0, "label_provenance_shortcut_auc": 0.5561666666666667, "q_appears_in_both_classes": 916, "q_appears_in_both_classes_share": 0.4177, "gate": "PASS"}`

## 5. C 真实低基率回放（A7500）
- AUROC：y-only `0.977` / q+y `0.894`；AUPRC：y-only `0.138` / q+y `0.069`
- recall@FPR1%：y-only `0.393` / q+y `0.250`；recall@FPR5%：y-only `0.857` / q+y `0.393`
- 冻结阈值下（y `0.450` / q+y `0.550`）：y-only recall `0.071`、q+y recall `0.000`——均衡面板校准阈值在 A7500 低基率分布上发生分数平移，未做后验重校准。
- 说明：E1-C is NOT an unseen generalization experiment; it replays the frozen v3.2 B detector on the A7500 real distribution.

## 6. 验收口径评估（v3.1 冻结门控应用于 v3.2）
| 指标 | 目标 | v3.2 结果 | 判定 |
|---|---|---|---|
| q-only Macro-F1（期望弱） | 0.55–0.70 | 0.657 | PASS |
| y-only Macro-F1（期望一般） | 0.76–0.86 | 0.869 | MARGINAL/FAIL |
| q+y Macro-F1 | ≥0.90 | 0.871 | FAIL |
| q+y CI lower | ≥0.88 | 0.845 | FAIL |
| q+y 相对 y 增益 | ≥0.05 | 0.002 | FAIL |
| 5/5 seed 同向（q+y>y>q） | 成立 | 已检查（注：5-seed 结果退化为同值，阈值网格平坦所致） | 说明 |
| C q+y/y-only AUPRC ratio | ≥1.5 | 0.499 | FAIL |

## 7. 结论
- 主 seed=13：q-only Macro-F1 = 0.657；y-only = 0.869；q+y = 0.871。叙事排序 成立（q-only < y-only < q+y）。 Model-Dev CV（3847 行 5-fold）q+y = 0.922 达到 ≥0.90；Frozen Anchor（1077 行）q+y = 0.872 未达 0.90 目标，且 q+y 相对 y-only 增益很小（+0.003）。C 回放显示均衡面板阈值在 A7500 低基率分布上不迁移，y-only 排序（AUROC 0.977）优于 q+y（0.894）——v3.2 的 v1 风格构造换来了均衡面板判别力，但牺牲了真实分布迁移性，如实记录。