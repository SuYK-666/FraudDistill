# 最终 1.5B 学生模型训练 — 完成状态（2026-08-09）

## 训练（已完成）
- 配方：scripts/train_exp3_students.py --setting final_distill，seed 11，LoRA r=32/alpha=64 + head（modules_to_save），2 epochs / 296 步，训练池 4,747 行；wall ≈ 8.0h（含 3 次断电续训，resume 确定性重放验证通过）。
- 训练中 dev（300 子集 @0.5）：step 40 MF1 0.8397 -> step 120 0.9123 -> step 200 0.9126（best）-> step 280 0.9118 -> epoch2 末 0.9085；loss 收敛正常。
- 最佳 checkpoint：best_step120（两阶段 dev 选点：fast 300 -> top-3 全量 dev，FPR<=0.055 & recall>=0.82 下最大 MF1）。

## 官方验收（已完成，产物在 outputs/neural_student/final_distilled_student/）
- 冻结阈值：0.5622（dev：MF1 0.9115 / FPR 0.0421 / Recall 0.8623）。
- Reload checksum：128 样本 max logit diff 0.0 -> PASS（修复了 legacy 加载路径的 PeftModel 双重包裹 bug，score head 未加载问题）。
- 正式 test（n=1,262，单次冻结阈值）：

| 指标 | 结果 | Hard Gate | 目标 |
|---|---:|---:|---:|
| Macro-F1 | 0.9135 | >=0.885 PASS | >=0.900 PASS |
| Accuracy | 0.9136 | >=0.885 PASS | >=0.900 PASS |
| Recall | 0.8853 | >=0.81 PASS | >=0.84 PASS |
| FPR | 0.0591 | <=0.050 FAIL（差 0.9pp） | <=0.040 FAIL |
| AUPRC | 0.9717 | >=0.950 PASS | >=0.960 PASS |
| MCC | 0.8282 | >=0.780 PASS | >=0.800 PASS |
| Real-only MF1 | 0.7913 | >=0.740 PASS | >=0.780 PASS |
| Synthetic MF1 | 0.9938 | >=0.950 PASS | 0.95-0.99 达标 |
| 4-class MF1 | 0.4657 | >=0.430 PASS | >=0.480 FAIL |

- 机制切片：direct recall 0.9812 / trust 1.0 / leakage 0.9667 / clean-refusal FPR 0.0105 / hard-safe FPR 0.0 / over-refusal 0.9333 / context-flip pair acc 0.9474。
- 对比 Neural-SoftDistill：MF1 +0.0286、Recall +0.0759、Real-only +0.1018、4-class +0.0525、MCC +0.0487、AUPRC +0.0185；FPR +0.0187。
- 结论：8/9 Hard Gate 通过（仅 FPR 一项未达 <=0.050）；按指南 §25 单模型原则不进行二次训练，论文 Student 回退 Neural-SoftDistill；FraudDistill-Student-1.5B 权重与全部验收产物完整保留，供复现/部署决策使用。

## 产物清单
training_config.json / training_state.json / data_manifest.json / data_audit.json / best_checkpoint.json / calibration.json / dev_metrics.json / reload_checksum.json / test_metrics.json / slice_metrics.json / gate_result.json / model_card.md / test_eval/（predictions_test.jsonl）+ best_step120 等 checkpoint 目录（adapter_config.json + adapter_model.safetensors）