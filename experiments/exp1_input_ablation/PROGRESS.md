# 实验1 进度（PROGRESS）

> 更新时间：2026-08-11 · **全部完成**（M1 GPU 训练 15/15，统计与回放收官，报告已更新，代码已提交 GitHub）

| 子实验 | 状态 | 说明 |
|---|---|---|
| E1-A | ✅ 完成 | 11/7500（0.147%），Gate 全过 |
| E1-B 面板 | ✅ 冻结 | 6000 行，Split-Freeze Gate PASS |
| E1-B M0 LR | ✅ 完成 | q+y MF1 0.951，Δ=+0.165 |
| E1-B M2/M3 Anchor | ✅ 完成 | 9600/9600 票，四视图模式一致 |
| E1-B M1 XLM-R | ✅ 完成 | 15/15（GPU）；q_y MF1 0.9685 ± 0.0035 / AUROC 0.9944；Δ=+0.1717，CI [0.1511, 0.1928]（v4.6 Macro-F1 修正），Gate 全过，5/5 seeds |
| E1-C | ✅ 完成 | 独立 624 行 / 6 阳性；q_y MF1 0.6748 / AUROC 0.9706 / AUPRC 0.397 / Recall@FPR1% 0.533 |
| 统计检验 | ✅ 完成 | bootstrap CI [0.1511, 0.1928]（v4.6 修正）；McNemar / Holm 全部显著；clean-anchor 敏感性 Gate 全过（Δ=0.1784，CI [0.1551, 0.2028]） |
| 最终报告 / GitHub | ✅ 完成 | `report/E1_FINAL_REPORT.md` + `tables/E1_PAPER_TABLES.md` 已更新并提交 |

## 执行说明（GPU）
- 15 个 M1 训练任务在远程服务器（10.160.16.3:23213，RTX 4090 24GB，venv `~/e1venv`）完成，单任务 58–121 s、全量约 12 分钟；
- 模型与日志已回传 `data/prepared/e1_final_triad_v4/models/`（15 checkpoint / 90 文件 / 8.6 GB），merge 校验 missing: NONE；
- 本机 CPU 训练已停止；旧 CPU 模型与 GPU 暂存文件已归档；wrong-q 控制已按同语言+同类别重建（v4.5，1200/1200），bootstrap 已修正为 Macro-F1（v4.6）。

## 最终 M1 指标（Frozen Anchor 1200，5 seeds 均值 ± sd）

| View | Macro-F1 | AUROC | AUPRC | Recall | Precision | FPR |
|---|---|---|---|---|---|---|
| q_only | 0.6450 ± 0.0165 | 0.7172 | 0.6497 | 0.9097 | 0.6135 | 0.5750 |
| y_only | 0.8017 ± 0.0032 | 0.9201 | 0.9242 | 0.8303 | 0.7962 | 0.2233 |
| **q_y** | **0.9685 ± 0.0035** | **0.9944** | **0.9934** | 0.9853 | 0.9534 | 0.0483 |
| wrong_q_y | 0.7574 ± 0.0039 | 0.7327 | 0.6632 | 0.8200 | 0.7300 | 0.3033 |

## 日志与结果位置
- 训练日志：`data/prepared/e1_final_triad_v4/logs/m1_shard0.out.log`、`m1_shard1.out.log`
- 统计结果：`data/prepared/e1_final_triad_v4/E1_V4_STATS.json`
- E1-C 结果：`data/prepared/e1_final_triad_v4/E1_V4_C_RESULT.json`
