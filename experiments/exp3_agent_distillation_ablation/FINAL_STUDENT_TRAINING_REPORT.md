# FraudDistill-Student-1.5B 最终训练报告（2026-08-08 → 2026-08-09）

> 实验目录：`experiments/exp3_agent_distillation_ablation/`
> 指南：《FraudDistill_最终1.5B学生模型训练实施指南》（2026-08-07）
> 运行环境：CPU-only（无 NVIDIA GPU），fp32；电源计划：卓越性能
> 相关日志：`outputs/train_final_distill.log`（每 20 步摘要）、`outputs/train_final_distill.err`（tqdm 进度）、`outputs/train_final_student_watchdog.log`（选点/校验/正式 test 流水）
> 步级记录：`outputs/neural_student/final_distilled_student/step_log.jsonl`（每步 1 条）、`eval_log.jsonl`（每次 dev eval）

---

## 0. 摘要

- 训练完成：**2 epochs / 296 步 / 4,747 行**，wall ≈ **7.96 h**（含 3 次断电续训，resume 确定性重放验证通过）。
- 最佳 checkpoint：**`best_step120`**（两阶段 dev 选点：fast 300 → top-3 全量 dev，FPR≤0.055 & recall≥0.82 约束下最大 Macro-F1）。
- 冻结阈值：**0.5622**（dev：MF1 0.9115 / FPR 0.0421 / Recall 0.8623）。
- Reload 校验：128 样本 **max logit diff 0.0 → PASS**（修复 legacy 加载路径双重包裹 bug 后）。
- 正式 test（n=1,262，单次冻结阈值）：**Macro-F1 0.9135 / Acc 0.9136 / Recall 0.8853 / FPR 0.0591 / AUPRC 0.9717 / MCC 0.8282 / Real-only 0.7913 / 4-class 0.4657**。
- Gate 判定：Hard Gate **8/9 通过**（仅 FPR 0.0591 略超 ≤0.050）；按指南 §25 单模型原则不进行二次训练，论文 Student 回退 Neural-SoftDistill；本模型权重与全部验收产物完整保留。

---

## 1. 训练配置

```text
python scripts/train_exp3_students.py --backend neural --setting final_distill --seeds 11 \
  --manifest data/prepared/exp3_neural_student/final_train_manifest.jsonl \
  --max-length 512 --micro-batch 4 --effective-batch 32 \
  --eval-steps 40 --patience 4 --lora-r 32 --lora-alpha 64 \
  --epochs 2 --eval-subset 300 \
  --out-root experiments/exp3_agent_distillation_ablation/outputs/neural_student
```

| 项 | 值 |
|---|---|
| 基座模型 | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B（冻结，CPU fp32） |
| 架构 | standard（4 类统一 softmax：safe / fraud_assistance / refusal_failure / over_refusal） |
| LoRA | r=32 / alpha=64 / dropout=0.05；target: q/k/v/o/gate/up/down proj；modules_to_save=[classifier, score] |
| 输入 | q+y（user_query + target_model_answer），head-tail 截断，max_length=512（P95=1074） |
| 批大小 | micro-batch 4 / effective 32 / grad_accum 8 |
| 优化器 | AdamW；lr_lora 1e-4 / lr_head 5e-4；weight_decay 0.01；max_grad_norm 1.0；warmup 5%（optimizer step 口径） |
| 损失 | FinalDistillLoss = CE4 + 0.30×binary + 0.30×w_t×KL(T=2) + 0.05×pair（teacher-only 样本无硬 CE） |
| class_weights | [0.75, 0.75, 0.898, 1.5]（四类） |
| 数据（4,747 行） | benchmark 1,065 / synthetic_core 1,930 / paired_dev 652 / hard_expansion 1,100；每桶 safe≈50%；EN 64.3%；权重 cap ≤4×median |
| eval 协议 | 每 40 步 dev 300 子集（@0.5 阈值，仅监控）；epoch 结束全量 dev 1,047（@0.5）；patience 4 |
| seed | 11 |

---

## 2. 训练过程时间线（3 次启动）

| 启动 | 时间 | 说明 |
|---|---|---|
| 第 1 次 | 08-08 上午 | 从 step 0 开始；训练至断电中断 |
| 第 2 次 | 08-08 下午 | `--resume resume.pt` 续训；再次断电中断 |
| 第 3 次 | 08-08 19:44:38 | 从 **step 160**（epoch 2 内 micro-step 87）恢复；一直跑完 296/296 |
| 训练完成 | 08-09 03:43 | `final_distill_metrics.json` 写入，watchdog 接管 Stage B |

- 断电续训无损性验证：step 61–66 重放结果**逐位一致**（确定性验证通过）。
- 单步耗时：约 165–215 s/step（CPU fp32；tqdm 进度条显示的 ~523 s/step 是每个 micro-batch 刷新一次的假象，实际以 step 边界为准：296 步 / 7.56 h ≈ 170 s/step）。
- 每 20 步在 `outputs/train_final_distill.log` 输出 loss/grad/lr + 当窗口数据构成（safe 数 / EN 数 / source 桶 / teacher_only 数），便于实时监控。

---

## 3. 每 20 步 loss 明细（step_log.jsonl，单步值）

| step | loss_gold | loss_binary | loss_kl | loss_pair | loss_total | grad_norm | lr_lora | lr_head |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1.4189 | 1.1272 | 4.4183 | 0.0000 | 3.0826 | 15.3397 | 1e-4 | 5e-4 |
| 40 | 0.4108 | 0.3646 | 1.8729 | 0.0000 | 1.0821 | 12.9675 | 1e-4 | 5e-4 |
| 60 | 0.3082 | 0.2746 | 1.1339 | 0.0000 | 0.7308 | 5.7904 | 1e-4 | 5e-4 |
| 80 | 0.0344 | 0.0352 | 0.5771 | 0.0000 | 0.2181 | 7.0292 | 1e-4 | 5e-4 |
| 100 | 0.1238 | 0.1587 | 0.6365 | 0.0000 | 0.3624 | 9.2634 | 1e-4 | 5e-4 |
| 120 | 0.0953 | 0.1090 | 0.5817 | 0.0000 | 0.3026 | 11.1707 | 1e-4 | 5e-4 |
| 140 | 0.0104 | 0.0121 | 0.7531 | 0.0000 | 0.2400 | 3.9374 | 1e-4 | 5e-4 |
| 160 | 0.0165 | 0.0182 | 0.3723 | 0.0000 | 0.1337 | 3.6215 | 1e-4 | 5e-4 |
| 180 | 0.0986 | 0.0697 | 0.3397 | 0.0000 | 0.2214 | 9.7650 | 1e-4 | 5e-4 |
| 200 | 0.0043 | 0.0039 | 0.1938 | 0.0000 | 0.0636 | 1.9178 | 1e-4 | 5e-4 |
| 220 | 0.0222 | 0.0043 | 0.2761 | 0.0000 | 0.1063 | 2.5997 | 1e-4 | 5e-4 |
| 240 | 0.0390 | 0.0347 | 0.1819 | 0.0000 | 0.1040 | 4.9488 | 1e-4 | 5e-4 |
| 260 | 0.0464 | 0.0601 | 0.6031 | 0.0000 | 0.2454 | 4.8665 | 1e-4 | 5e-4 |
| 280 | 0.0254 | 0.0251 | 0.3637 | 0.0000 | 0.1420 | 4.2207 | 1e-4 | 5e-4 |
| 296 | 0.0286 | 0.0227 | 0.3085 | 0.0000 | 0.1279 | 4.8141 | 1e-4 | 5e-4 |
| 298（epoch2 结束） | 0.0067 | 0.0059 | 0.0954 | 0.0000 | 0.0371 | 2.4379 | 1e-4 | 5e-4 |

要点：

- **loss_total 快速下降**：step 20 时 3.08（warmup 初期，KL 项 4.42 主导）→ step 80 已降至 0.22 → 此后在 0.06–0.30 区间波动（单步噪声），epoch 2 末收敛至 0.037。
- **loss_kl（蒸馏项）**：从 4.42 单调降至 0.1–0.6 区间，是 loss_total 的主要构成，说明学生逐步贴近教师软信号。
- **loss_gold / loss_binary**：step 80 后进入低值区（0.004–0.16），波动来自不同 bucket/teacher-only 样本的混合。
- **loss_pair 恒为 0**：final_distill 配方中 pair 项权重极低（0.05）且该阶段无配对样本，符合配方设定。
- **grad_norm**：1.9–15.3 波动，无发散；max_grad_norm=1.0 裁剪生效。
- **lr**：全程恒定（lora 1e-4 / head 5e-4），warmup 在 optimizer step 口径下完成。

---

## 4. 每 40 步 dev eval 明细（final_distill_metrics.json history.dev）

监控口径：dev 300 子集 @0.5 阈值（step 149 / 298 为 **epoch 结束全量 dev 1,047** 行 @0.5）。

| step | epoch | MF1 | Recall | FPR | Acc | AUPRC | MCC | real MF1 | syn MF1 | EN MF1 | ZH MF1 | ECE | Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 1 | 0.8397 | 0.8824 | 0.1951 | 0.8400 | 0.9262 | 0.6843 | 0.6948 | 0.9475 | 0.7681 | 0.9888 | 0.0758 | 0.1131 |
| 80 | 1 | 0.8863 | 0.9118 | 0.1341 | 0.8867 | 0.9616 | 0.7745 | 0.7481 | 0.9766 | 0.8397 | 0.9773 | 0.0689 | 0.0850 |
| 120 | 1 | 0.9123 | 0.8897 | 0.0671 | 0.9133 | 0.9733 | 0.8250 | 0.7797 | 0.9924 | 0.8663 | 1.0000 | 0.0593 | 0.0711 |
| 149（epoch1 末） | 1 | 0.8995 | 0.9118 | 0.1098 | 0.9000 | 0.9697 | 0.7996 | 0.7545 | 1.0000 | 0.8492 | 1.0000 | 0.0765 | 0.0822 |
| 160 | 2 | 0.9094 | 0.9118 | 0.0915 | 0.9100 | 0.9675 | 0.8189 | 0.7739 | 1.0000 | 0.8634 | 1.0000 | 0.0626 | 0.0721 |
| 200 | 2 | 0.9126 | 0.9044 | 0.0793 | 0.9133 | 0.9701 | 0.8251 | 0.7768 | 1.0000 | 0.8676 | 1.0000 | 0.0676 | 0.0726 |
| 240 | 2 | 0.9123 | 0.8897 | 0.0671 | 0.9133 | 0.9641 | 0.8250 | 0.7797 | 0.9924 | 0.8663 | 1.0000 | 0.0695 | 0.0749 |
| 280 | 2 | 0.9118 | 0.8603 | 0.0427 | 0.9133 | 0.9684 | 0.8266 | 0.7493 | 1.0000 | 0.8633 | 1.0000 | 0.0708 | 0.0686 |
| 298（epoch2 末） | 2 | 0.9085 | 0.8603 | 0.0488 | 0.9100 | 0.9711 | 0.8194 | 0.7547 | 0.9924 | 0.8641 | 0.9888 | 0.0637 | 0.0690 |

要点：

- **收敛轨迹**：step 40→120 快速上升（MF1 0.8397→0.9123，FPR 0.1951→0.0671）；step 120 后进入平台期（MF1 0.908–0.913）。
- **FPR 持续下降**：0.1951 → 0.0427（step 280）/ 0.0488（epoch2 末），说明后段主要在压低误报。
- **Recall 权衡**：step 280 起 recall 降至 0.86（FPR 最低点），step 120/240 在 recall 0.8897 时 FPR 0.0671。
- **best 判定**：按指南 §16.1（FPR≤0.055 & recall≥0.82 下最大 MF1），`best_step120` 成为候选（全量 dev FPR 0.0531，唯一满足约束）；step 280 的 FPR 虽低但 recall 0.8603，MF1 略低且未进入 top-3 全量评估。
- **ZH 几乎完美**（0.977–1.000），EN 为瓶颈（0.768–0.868），real-only 是主要待提升项（0.69–0.78）。

机制切片（best_step120，dev 300 子集 @0.5）：direct_recall 0.9444 / trust_recall 1.0 / leakage_recall 0.9286 / clean_refusal_fpr 0.0 / hard_safe_fpr 0.0 / over_refusal_recall 0.6364 / context_flip_recall 1.0 / quotation_fpr 0.0（n 分别为 36/23/14/23/25/11/14/11）。

---

## 5. 两阶段 dev 选点（Stage B1，watchdog 自动执行）

- fast pass：dev 300 子集（seed 20260804 固定）对全部 7 个 checkpoint 打分排序；
- 全量 dev（1,047 行 @0.5）top-3：

| checkpoint | MF1 | Recall | FPR | real MF1 |
|---|---:|---:|---:|---:|
| checkpoint-200 / best_step200 | 0.9129 | 0.9002 | 0.0751 | 0.7789 |
| best_step120 | 0.9077 | 0.8663 | 0.0531 | 0.7681 |

- 约束筛选（FPR≤0.055 & recall≥0.82）：仅 **best_step120** 可行 → 选中（`best_checkpoint.json`，dev_gate_warning=false）。
- 阈值校准（指南 §20，全量 dev scores）：冻结阈值 **0.5622**（dev：MF1 0.9115 / FPR 0.0421 / Recall 0.8623，满足 FPR≤0.05 & recall≥0.82 下最大 MF1）。

---

## 6. Reload 校验（Stage B2）

- 首次执行 **FAIL**：max logit diff 14.28（128 样本）。
- 根因：`evaluate_final_student.py` legacy 加载路径对已是 PeftModel 的模型再次 `PeftModel.from_pretrained`，形成**双重包裹**，`modules_to_save`（score 分类头）权重静默加载失败 → 随机分类头 → 量级 O(1e1) 的 logit 差异（同一 bug 曾在早期 gold 训练中出现，`evaluate_neural_student.py` 已修复并在注释中警告，但 final 评估脚本沿用时再次引入）。
- 修复：改为 `model2.load_adapter(str(ckpt), adapter_name="default")` 原位加载（小样本验证与 canonical 路径 logit 完全一致）。
- 修复后：**max logit diff 0.0 → PASS**（`reload_checksum.json`：classifier_present=true, adapter_present=true）。
- 说明：canonical 加载路径（`load_checkpoint`：plain base + 单次 from_pretrained）始终正确，dev 选点与正式 test 均使用该路径，因此选点/校准结果不受影响。

---

## 7. 正式 test（Stage B3，n=1,262，单次冻结阈值 0.5622）

| 指标 | 结果 | Hard Gate（指南 §21.1） | 目标（指南 §22） |
|---|---:|---:|---:|
| Macro-F1 | 0.9135 | ≥0.885 ✅ | ≥0.900 ✅ |
| Accuracy | 0.9136 | ≥0.885 ✅ | ≥0.900 ✅ |
| Recall | 0.8853 | ≥0.81 ✅ | ≥0.84 ✅ |
| FPR | 0.0591 | ≤0.050 ❌（差 0.9pp） | ≤0.040 ❌ |
| AUPRC | 0.9717 | ≥0.950 ✅ | ≥0.960 ✅ |
| MCC | 0.8282 | ≥0.780 ✅ | ≥0.800 ✅ |
| Real-only MF1 | 0.7913 | ≥0.740 ✅ | ≥0.780 ✅ |
| Synthetic MF1 | 0.9938 | ≥0.950 ✅ | 0.95–0.99 ✅ |
| 4-class MF1 | 0.4657 | ≥0.430 ✅ | ≥0.480 ❌ |
| EN / ZH MF1 | 0.8702 / 0.9862 | — | — |

机制切片（test）：direct_recall 0.9812（n=160）/ trust_recall 1.0（n=80）/ leakage_recall 0.9667（n=60）/ clean_refusal_fpr 0.0105（n=95）/ hard_safe_fpr 0.0（n=120）/ over_refusal_recall 0.9333（n=60）/ quotation_fpr 0.0（n=40）/ context_flip_pair_acc 0.9474（pairs=19）。

对比 Neural-SoftDistill（同 test，soft 官方阈值 0.5）：MF1 +0.0286 / Recall +0.0759 / FPR +0.0187 / AUPRC +0.0185 / MCC +0.0487 / Real-only +0.1018 / 4-class +0.0525（配对显著性见主报告 §16.11）。

**Gate 判定**：Hard Gate 8/9 通过（仅 FPR 一项未达）；按指南 §25 单模型原则**不进行第二次正式训练**，论文 Student 回退 Neural-SoftDistill；FraudDistill-Student-1.5B 权重与全部验收产物完整保留于 `outputs/neural_student/final_distilled_student/`。

---

## 8. 产物清单（`outputs/neural_student/final_distilled_student/`）

```text
best_step120/                      <- 最终 checkpoint（adapter_config.json + adapter_model.safetensors 140.9MB + tokenizer）
checkpoint-{200,240,280}/ best_step{40,80,120,200}/   <- 同 run 其他 checkpoint
final/                             <- epoch2 结束 checkpoint（147.8MB）
training_config.json / training_state.json / data_manifest.json / data_audit.json
best_checkpoint.json / calibration.json / dev_metrics.json / reload_checksum.json
test_metrics.json / slice_metrics.json / gate_result.json / model_card.md
test_eval/                         <- 正式 test 预测（predictions_test.jsonl）+ test_metrics.json
step_log.jsonl / eval_log.jsonl / final_distill_metrics.json / resume.pt
```

复现入口：`scripts/train_exp3_students.py`（`--setting final_distill`）+ `scripts/evaluate_final_student.py`（`--select-best-on-dev` / `--reload-check` / `--split test --frozen-calibration`）+ `scripts/finalize_exp3_final_student.py`。