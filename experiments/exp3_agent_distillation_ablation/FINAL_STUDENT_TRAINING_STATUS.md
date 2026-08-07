# 最终 1.5B 学生模型训练 — 进行状态（2026-08-07 22:30）

> 本文件为训练期间的状态速览；最终结果由 `scripts/finalize_exp3_final_student.py` 写入正式报告 §16.10。

## 当前状态
- **正式训练已启动**：PID 32816，`--setting final_distill`，seed 11，2 epochs，max_length=512。
- **数据**：训练池 4,747 行（benchmark 1,065 / synthetic_core 1,930 / paired_dev 652 / hard_expansion 1,100），全量 gold + teacher signal，泄漏审计 PASS（train vs dev/test/balanced test 零重叠），采样权重达标，EN 63.9%。
- **修复**：optimizer-step 口径总步数（进度条 0/296、warmup 5%≈15 步、日志分母正确）——此前误用 micro-batch 总数 2,374，会导致 warmup 占训练 40%。
- **修复2（重要）**：FinalDistillLoss 原实现把 sample_weight（sampler 专用，总和≈1）当作 loss 重要性权重，分母 clamp_min(1.0) 导致 loss 被压缩约 188 倍（日志显示 0.004，实际应 ~4）；已改为批内均值（保留类别权重、teacher-only 掩码、w_t），离线验证 loss_total≈4.11 后重启训练。
- **监控**：watchdog（PID 31920）每 60s 检查；训练意外退出会自动 `--resume` 续跑（最多 3 次）；训练完成后自动执行：dev 选点（fast 300 → top-3 全量）→ 阈值校准（FPR≤0.05 & recall≥0.82 最大 MF1，否则 FPR≤0.06 回退）→ reload checksum（128 条，≤1e-5）→ 正式 test（单次，冻结阈值）。

## 日志位置
- 训练：`outputs/train_final_distill.log`（每 20 步一条详细日志）/ `.err`（tqdm 进度条）
- Watchdog：`outputs/train_final_student_watchdog.log`
- 状态：`outputs/train_final_student_watchdog_status.json`（完成时含 test 关键指标）

## 预计时间（CPU fp32，16 线程）
- 训练 296 optimizer steps × ≈3.1 min ≈ 15–16 小时（含末尾旧代码一次 raw test 约 78 min）
- 选点+校准 ≈ 3–4 小时；reload 校验 ≈ 15 min；正式 test ≈ 80 min
- 合计 ≈ 20–22 小时（约 2026-08-08 晚间完成）

## 完成后
1. `python scripts/finalize_exp3_final_student.py`（打包 §28 产物 + 生成报告 §16.10 + Gate 判定）
2. 更新本报告摘要与 §16.9 中 "待后续指令执行重训" 的表述
3. commit + push GitHub（模型 adapter 不进仓库，随 .gitignore 排除；manifest/audit 在 `data/prepared/exp3_neural_student/`）
4. 验收标准（§21/44）：Hard Gate MF1≥0.885 / Acc≥0.885 / Recall≥0.81 / FPR≤0.050 / AUPRC≥0.950 / MCC≥0.780 / Real-only≥0.740 / 4-class≥0.430；目标 MF1≥0.900 / Real≥0.780 / FPR≤0.040；不达标则论文回退 Neural-SoftDistill（禁止二次训练）