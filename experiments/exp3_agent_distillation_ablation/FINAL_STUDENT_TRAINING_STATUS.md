# 最终 1.5B 学生模型训练 — 进行状态（2026-08-08 19:53 更新）

## 当前状态：训练中（第 3 次启动，从 step 160 恢复）
- **进程**：训练 PID 21432（19:44:38 启动），watchdog PID 22836。
- **恢复点**：`resume.pt`（step 160/296，epoch 2 内 micro-step 87）→ 已确认恢复并正常推进（step 161-163 loss 波动正常）。
- **剩余**：~133 步 × ~3 min ≈ 6.7 小时 → 预计 2026-08-09 02:30 左右完成训练；随后 watchdog 自动执行：dev 选点（fast 300→top-3 全量）→ 阈值校准（FPR≤0.05 & recall≥0.82）→ reload checksum（128 条 ≤1e-5）→ 正式 test（单次冻结阈值）。

## 已完成进度（54%+）
| 评估点 | MF1 | Recall | FPR | AUPRC | MCC |
|---|---:|---:|---:|---:|---:|
| step 40 | 0.8397 | 0.8824 | 0.1951 | 0.9262 | 0.6843 |
| step 80 | 0.8863 | 0.9118 | 0.1341 | 0.9616 | 0.7745 |
| step 120（best） | 0.9123 | 0.8897 | 0.0671 | 0.9733 | 0.8250 |
| epoch1 结束 | 0.8995 | 0.9118 | 0.1098 | - | - |
| step 160 | 0.9094 | 0.9118 | 0.0915 | 0.9675 | 0.8189 |

- Best（step 120）：Acc 0.9133、Real-only MF1 0.7797、Synthetic 0.9924
- Gate 差距：FPR 0.067 vs ≤0.050（epoch 2 若持续不改善，patience=4 自动早停）
- 断点续训无损已验证（step 61-66 确定性重放逐位一致）

## 完成后
1. watchdog 自动跑完选点/校准/reload/正式 test（产物在 run dir）
2. `python scripts/finalize_exp3_final_student.py`（打包 §28 产物 + 报告 §16.10 + Gate 判定）
3. 更新报告摘要与 §16.9；commit + push GitHub