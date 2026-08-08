# 最终 1.5B 学生模型训练 — 进行状态（2026-08-08 10:10 更新）

## 当前状态：已暂停（用户指令，等待重启）
- **断点**：`experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/resume.pt`（保存于 step 60 边界，09:48:33）
- **进度**：global_step=60 / 296（epoch 0，micro-step 479），dev eval 已完成 1 次（step 40：macro_f1=0.8397 / recall=0.8824 / fpr=0.1951 / auprc=0.9262 / mcc=0.6843）
- **早停计数**：no_improve=0（无风险）
- 暂停状态备份：`outputs/train_final_student_pause_state.json`

## 重启方法（等用户指令后执行）
```
python scripts/train_exp3_students.py --backend neural --setting final_distill --seeds 11
  --manifest data/prepared/exp3_neural_student/final_train_manifest.jsonl
  --max-length 512 --micro-batch 4 --effective-batch 32 --eval-steps 40 --patience 4
  --lora-r 32 --lora-alpha 64 --epochs 2 --eval-subset 300
  --out-root experiments/exp3_agent_distillation_ablation/outputs/neural_student
  --resume experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/resume.pt
```
- 同时重启 watchdog：`python scripts/watchdog_final_student.py`
- 丢失的 step 60→66 会在恢复时确定性重放（固定 seed + 确定性采样器），结果与不暂停完全一致。

## 背景（第 2 轮，修复后）
- 修复了假性早停 Bug（eval/save 移入 optimizer-step 门控，commit `6bb8732`），smoke 验证通过。
- 确定性复现已验证：step 20/40 的 loss 与 dev 指标与第 1 轮逐位一致。
- 训练数据：4,747 行（benchmark 1,065 / synthetic_core 1,930 / paired_dev 652 / hard_expansion 1,100），seed 11。