# 最终 1.5B 学生模型训练 — 进行状态（2026-08-08 06:35 更新）

> 本文件为训练期间的状态速览；最终结果由 `scripts/finalize_exp3_final_student.py` 写入正式报告 §16.10。

## 当前状态
- **正式训练已重启（第 2 轮，修复后）**：PID 21912（2026-08-08 06:26:58 启动），`--setting final_distill`，seed 11，2 epochs，max_length=512。
- **数据**：训练池 4,747 行（benchmark 1,065 / synthetic_core 1,930 / paired_dev 652 / hard_expansion 1,100），全量 gold + teacher signal，泄漏审计 PASS，采样权重达标，EN 63.9%。
- **重要修复（本轮）**：定位并修复了**假性早停 Bug**——`save_steps`/`eval_steps` 两个代码块原先在 grad_accum 门控之外，导致 step 边界（如 40）后的每一个 micro-batch 都会重复执行完整 dev eval（global_step 仍等于 40），连续 5 次相同 eval 后触发 `no_improve>=4` 早停，模型仅训练到 step 40 就退出（dev macro_f1=0.8397、FPR=0.195，远不达标）。
  - 修复：eval/save 块移入 `(step+1) % grad_accum == 0` 门控内，仅在 optimizer-step 边界执行一次（commit `6bb8732`）。
  - 附带修复：`best_state` 只克隆可训练参数（原克隆整个 6GB 基座导致内存压力）；`save_resume` 补存 `best_step` 键；新增 `step_log.jsonl` / `eval_log.jsonl` 步骤级日志。
  - **Smoke 验证通过**：400 行子集跑 12 步（eval-steps=4），`eval_log.jsonl` 恰好 2 条（step 4、8 各一次），无重复 eval，checkpoint/best 保存正常。
- **第 1 轮（含 Bug 的运行）已归档**：`outputs/neural_student/archive/final_distilled_student_broken_step40_20260808/`，不再作为正式结果。
- **监控**：watchdog（PID 1720）每 60s 检查；训练意外退出会自动 `--resume` 续跑（最多 3 次）；训练完成后自动执行：dev 选点（fast 300 → top-3 全量）→ 阈值校准（FPR≤0.05 & recall≥0.82 最大 MF1，否则 FPR≤0.06 回退）→ reload checksum（128 条，≤1e-5）→ 正式 test（单次，冻结阈值）。

## 日志位置
- 训练：`outputs/train_final_distill.log`（每 20 步一条详细日志）/ `.err`（tqdm 进度条）
- 步骤级日志：`.../final_distilled_student/step_log.jsonl`（每 20 步）/ `eval_log.jsonl`（每次 dev eval）
- Watchdog：`outputs/train_final_student_watchdog.log`；状态：`outputs/train_final_student_watchdog_status.json`

## 预计时间（CPU fp32，~14 线程，~3 min/step）
- 训练 296 optimizer steps ≈ 15 小时（约 2026-08-08 21:30 完成，每 40 步含一次 300 条 dev eval）
- 选点+校准 ≈ 2–3 小时；reload 校验 ≈ 15 min；正式 test ≈ 60–90 min
- 合计 ≈ 19–20 小时

## 完成后
1. `python scripts/finalize_exp3_final_student.py`（打包 §28 产物 + 生成报告 §16.10 + Gate 判定）
2. 更新本报告摘要与 §16.9 中 "待后续指令执行重训" 的表述
3. commit + push GitHub（模型 adapter 不进仓库，随 .gitignore 排除；manifest/audit 在 `data/prepared/exp3_neural_student/`）
4. 验收标准（§21/44）：Hard Gate MF1≥0.885 / Acc≥0.885 / Recall≥0.81 / FPR≤0.050 / AUPRC≥0.950 / MCC≥0.780 / Real-only≥0.740 / 4-class≥0.430；目标 MF1≥0.900 / Real≥0.780 / FPR≤0.040；不达标则论文回退 Neural-SoftDistill（禁止二次训练）