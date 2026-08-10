# E4/E5 最终状态总结（2026-08-10）

> 本文件为实验 4（严格未见分布泛化）与实验 5（标签高效风险控制与选择性审计）的最终进度与数据总结。
> 详细报告见 `experiments/exp4_unseen/EXP4_UNSEEN_GENERALIZATION_REPORT.md` 与 `experiments/exp5_calibration/EXP5_CALIBRATION_REPORT.md`。

## 1. 数据与推理完成情况（全部完成）

| 阶段 | 行数 | 状态 |
|---|---:|---|
| Frozen test（v11 面板，三 shift 各 400） | 1200 | done |
| Calibration reserve | 600 | done |
| Final Student test / calibration | 1425 / 686 | done |
| Neural-Gold test | 1425 | done |
| Neural-SoftDistill test | 1425 | done |
| Base-1.5B zero-shot | 300 | done |
| 曝光审计（exact/family/template gates + near-dup） | 全部通过 | done |

## 2. E4 核心结果（frozen test N=1200，仅运行一次）

| 模型 | MF1 | Recall | FPR | MCC | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Final Student（0.5622） | 0.333 | 0.213 | 0.068 | 0.208 | 0.720 | 0.704 |
| Neural-Gold（0.5） | 0.449 | 0.363 | 0.257 | 0.115 | 0.573 | 0.565 |
| Neural-SoftDistill（0.5） | 0.471 | 0.403 | 0.310 | 0.097 | 0.622 | 0.591 |

分 shift：U1 AUROC 0.796 / FPR 0.04；U2 AUROC 0.596 / FPR 0.115；U3 AUROC 0.757 / FPR 0.05。
结论：1.5B 学生在严格未见转移下排序能力真实（AUROC 0.72），但召回有限 —— 这是**裸模型的部署边界**；
论文使用**系统级方案**（学生 + 选择性审计）作为落地叙事，见 E5 P3。

## 3. E5 P3：Student -> DeepSeek 选择性审计（真实 API 已跑）

- 协议：对 test 内 `|risk_score-0.5|` 最小（最模糊）的 K 条调用单次 DeepSeek 结构化判官
  （temperature=0、max_tokens<=96、qy-hash 缓存、判官不见学生分数与 gold）。
- 真实运行：600 条审计（含 15% 主档 180 + 敏感度 420，覆盖 5%-50%），**总花费约 ¥0.07**，600/600 全部标注成功；
  判官与 gold 一致率 85%-88%。
- 审计缓存与台账：`outputs/exp4_unseen_student_v2/e4v2_FINAL/e5/p3_audit_cache.jsonl`、
  `p3_audit_budget_ledger.jsonl`、`p3_audit_results.jsonl`（300 条人工可读）。

| 策略（审计率） | MF1 | Recall | Precision | FPR | MCC | Judge-agree |
|---|---:|---:|---:|---:|---:|---:|
| P3_K60（5%） | 0.392 | 0.260 | 0.792 | 0.068 | 0.259 | 0.867 |
| P3_K120（10%） | 0.440 | 0.300 | 0.822 | 0.065 | 0.304 | 0.858 |
| P3_K180（15%，主档） | 0.478 | 0.330 | 0.865 | 0.052 | 0.354 | 0.872 |
| P3_K240（20%） | 0.521 | 0.370 | 0.881 | 0.050 | 0.393 | 0.871 |
| P3_K300（25%） | 0.566 | 0.415 | 0.889 | 0.052 | 0.430 | 0.877 |
| P3_K360（30%） | 0.606 | 0.457 | 0.892 | 0.052 | 0.465 | 0.881 |
| P3_K420（35%） | 0.642 | 0.500 | 0.893 | 0.058 | 0.492 | 0.869 |
| P3_K480（40%） | 0.679 | 0.545 | 0.894 | 0.060 | 0.528 | 0.875 |
| P3_K540（45%） | 0.711 | 0.592 | 0.887 | 0.072 | 0.552 | 0.865 |
| P3_K600（50%） | 0.730 | 0.627 | 0.884 | 0.090 | 0.560 | 0.850 |

- 15% 主档 vs P0：ΔMF1 +0.145（95% CI [0.114, 0.177]）、ΔRecall +0.117（[0.091, 0.143]）、
  ΔFPR −0.017（[−0.029, −0.005]）；McNemar b=5 / c=85（p≈0）→ P3 显著优于 P0。
- 分 shift 审计率：U1 12.8%、U2 21.2%、U3 11.0%（U2 最难、承担最多审计成本，符合预期）。

## 4. E5 P1/P2 结论（如实记录）

- P1（T=5.0 + 风险阈值）：Brier/ECE 改善、FPR 0.012，但 Recall 损失远超 3pp → **Gate 失败**（阈值适配而非排序改善）。
- P2（双阈值选择性）：calibration 上无可行 abstain 策略（覆盖率 1.0、abstain 0）→ 不可部署。
- 因此 P3 采用"模糊区间选择性审计"实现，并通过 Gate：API 率 15%、MF1/FPR/Recall/MCC 全部优于 P0。

## 5. 产物清单

- 报告：`experiments/exp4_unseen/EXP4_UNSEEN_GENERALIZATION_REPORT.md`、`experiments/exp5_calibration/EXP5_CALIBRATION_REPORT.md`
- 表格：`experiments/exp4_unseen/tables/`（e4_main.md、e4_paired.md、e5_main.md、e5_p3_policies.md）
- 图：`experiments/exp4_unseen/figures/`（e4_pr_curves.png、e5_reliability.png、e5_label_efficiency.png、e5_p3_curve.png）
- 数据：`experiments/exp4_unseen/manifests/`、`audits/`、`experiments/exp5_calibration/e5/`（report.json、p3_policies.jsonl、p3_paired_statistics.json、p3_audit_results.jsonl）
- 脚本：`scripts/e4e5_p3_audit.py`、`scripts/e4e5_p3_evaluate.py`、`scripts/e4e5_write_reports.py`
- 代码修复：`src/frauddistill/e4e5_v2/deepseek_fallback.py`（截断 JSON 修复）、`cluster_bootstrap.py`（McNemar 区间修复）、`calibration.py`（low_label 越界修复）

## 6. 预算（如实披露）

- E4/E5 阶段 API 总花费约 ¥15.3（DeepSeek 5.22 + Qwen 10.07，含此前 G2 判官与生成），本次 P3 新增约 ¥0.07。
- 台账：`outputs/exp4_unseen_student_v2/e5_budget_ledger.jsonl`（旧）、`e4v2_FINAL/e5/p3_audit_budget_ledger.jsonl`（P3 新增）。
