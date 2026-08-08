# 最终 1.5B 学生模型训练 — 进行状态（2026-08-08 11:20 更新）

## 当前状态：训练中（已从断点恢复）
- **进程**：训练 PID 30144（11:19:19 启动），watchdog PID 3152。
- **恢复点**：`resume.pt`（step 60 边界，09:48:33）→ 已从 global_step=60 / epoch 0 / micro-step 479 继续，剩余约 236 步。
- **提速**：电源方案已切换为**卓越性能**（GUID a7c44061...）；`torch.set_num_threads(16)` 保持。
- **进度**：暂停前 step 40 dev eval：macro_f1=0.8397 / recall=0.8824 / FPR=0.1951 / AUPRC=0.9262 / MCC=0.6843（no_improve=0）。
- 预计完成：剩余 236 步 × ~3.1 min ≈ 12.2 小时 → 约 2026-08-08 23:30（训练完成后 watchdog 自动执行选点 → 校准 → reload 校验 → 正式 test）。

## 监控方式
- 步骤级日志：`.../final_distilled_student/step_log.jsonl`（每 20 步）、`eval_log.jsonl`（每次 dev eval）
- 训练：`outputs/train_final_distill.log` / `.err`（tqdm）
- Watchdog：`outputs/train_final_student_watchdog.log`

## 完成后
1. `python scripts/finalize_exp3_final_student.py`（打包 §28 产物 + 报告 §16.10 + Gate 判定）
2. 更新报告摘要与 §16.9 表述；commit + push GitHub
3. 验收标准（§21/44）：Hard Gate MF1≥0.885 / Acc≥0.885 / Recall≥0.81 / FPR≤0.050 / AUPRC≥0.950 / MCC≥0.780 / Real-only≥0.740 / 4-class≥0.430；目标 MF1≥0.900 / Real≥0.780 / FPR≤0.040