# 最终 1.5B 学生模型训练 — 进行状态（2026-08-08 18:32 更新）

## 当前状态：已暂停（用户出门，等待重启）
- **断点**：`experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/resume.pt`
  - global_step=**160/296**（epoch 2 内，micro-step 87），no_improve=1，best=step 120（MF1 0.9123）
  - 保存时间 18:29（step 160 eval 完成后），**无损断点**
- 暂停状态备份：`outputs/train_final_student_pause_state.json`

## 重启方法（等用户指令）
```
python scripts/train_exp3_students.py --backend neural --setting final_distill --seeds 11
  --manifest data/prepared/exp3_neural_student/final_train_manifest.jsonl
  --max-length 512 --micro-batch 4 --effective-batch 32 --eval-steps 40 --patience 4
  --lora-r 32 --lora-alpha 64 --epochs 2 --eval-subset 300
  --out-root experiments/exp3_agent_distillation_ablation/outputs/neural_student
  --resume experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/resume.pt
```
同时重启 watchdog：`python scripts/watchdog_final_student.py`

## 已完成进度（54%）
| 评估点 | MF1 | Recall | FPR | AUPRC | MCC |
|---|---:|---:|---:|---:|---:|
| step 40 | 0.8397 | 0.8824 | 0.1951 | 0.9262 | 0.6843 |
| step 80 | 0.8863 | 0.9118 | 0.1341 | 0.9616 | 0.7745 |
| step 120（best） | 0.9123 | 0.8897 | 0.0671 | 0.9733 | 0.8250 |
| epoch1 结束 | 0.8995 | 0.9118 | 0.1098 | - | - |
| step 160 | 0.9094 | 0.9118 | 0.0915 | 0.9675 | 0.8189 |

- Best（step 120）：Acc 0.9133、Real-only MF1 0.7797、Synthetic 0.9924、direct_recall 0.97、trust 1.0、clean_refusal FPR 0.0、hard_safe FPR 0.04
- loss：epoch 1 单步均值 0.447，缓降无发散；epoch 2 开头波动 0.74-0.77
- 剩余：136 步训练 + 选点 + 校准 + reload 校验 + 正式 test；Gate 差距：FPR 0.067 vs ≤0.050

## 注意事项
- 恢复后确定性重放已验证（step 61-66 逐位一致），断点续训无损
- 训练完成后 watchdog 自动执行：选点（fast 300→top-3 全量）→ 阈值校准（FPR≤0.05 & recall≥0.82）→ reload checksum（128 条 ≤1e-5）→ 正式 test（单次冻结阈值）
- 之后运行 `python scripts/finalize_exp3_final_student.py` 打包产物 + 报告 §16.10