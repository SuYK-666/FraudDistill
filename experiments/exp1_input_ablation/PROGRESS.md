# 实验1 进度（PROGRESS）

> 更新时间：2026-08-11 00:30 · 训练进行中

| 子实验 | 状态 | 说明 |
|---|---|---|
| E1-A | ✅ 完成 | 11/7500（0.147%），Gate 全过 |
| E1-B 面板 | ✅ 冻结 | 6000 行，Split-Freeze Gate PASS |
| E1-B M0 LR | ✅ 完成 | q+y MF1 0.951，Δ=+0.165 |
| E1-B M2/M3 Anchor | ✅ 完成 | 9600/9600 票，四视图模式一致 |
| E1-B M1 XLM-R | ⏳ 训练中 | 0/15 任务，预计 14-19h（CPU） |
| E1-C | ⏳ 待回放 | 独立 reserve 624 行 / 6 阳性，离线 |
| 统计检验 | ⏳ 待运行 | M1 完成后 |
| 最终报告/GitHub | ⏳ 待完成 | 报告骨架已写入 report/ |

## 实时监控
- 状态脚本：`python scripts/e1_v4_status.py`
- 训练日志：`data/prepared/e1_final_triad_v4/logs/m1_shard0.out.log`、`m1_shard1.out.log`
- 每任务完成追加 `E1_V4_TRAIN_PROGRESS.jsonl`（mode/seed/Anchor 指标）
