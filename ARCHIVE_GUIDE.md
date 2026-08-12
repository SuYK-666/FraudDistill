# 项目归档说明（ARCHIVE GUIDE）

> 整理日期：2026-08-12
> 原则：**只保留最新版本/最终成品在正式目录，旧版本与临时产物统一移入各实验的 `archive/` 归档**（本地保留，不推送到 GitHub；`archive/` 已在 `.gitignore` 中）。

## 本次整理归档位置一览

| 原位置（已清空） | 归档去向 | 内容 |
|---|---|---|
| 仓库根目录 `exp6_v2_*.log`（8 个） | `experiments/exp6_v2_balanced/archive/dev_logs/` | E6 v2 各阶段运行日志（.err/.log） |
| 仓库根目录 `_e6v2_server_stage/` | `experiments/exp6_v2_balanced/archive/dev_logs/_e6v2_server_stage/` | 服务器暂存数据/脚本（含 `_e6v2_blocked_*` 等） |
| 仓库根目录 `logs/`（8 个 0KB 日志） | `experiments/archive/e4e5_run_logs_20260812/` | E4/E5 早期 u1/u3 空日志 |
| `scripts/` 下 `_` 前缀临时脚本/日志（58 个） | `experiments/exp6_v2_balanced/archive/dev_scripts/` | `_e6v2_*.log`、`_run_*.py`、`_watchdog_*.py`、`_e1_*.py` 等开发期脚本 |
| `experiments/exp4_unseen/audits/`（58 个中间文件） | `experiments/exp4_unseen/archive/audits_build_intermediates_20260812/` | g2_*/gold_v4_u* 等构建期判定/补丁数据 |
| `aegis2/human_audit/`（2 个带 `_20260805` 旧文件） | `experiments/exp2_prior_work_comparison/aegis2/archive/human_audit_20260805_legacy/` | Aegis v4 早期审计原始输出 |
| exp3 `outputs/neural_student/` 训练中间产物（7 个目录） | `outputs/neural_student/archive/training_runs_20260812/` | gold/soft/full/lowlabel/smoke 训练检查点与 resume.pt（约 13GB） |
| `final_distilled_student/` 非最优检查点（8 项） | `outputs/neural_student/archive/final_distilled_student_extra_ckpts_20260812/` | checkpoint-200/240/280、best_step40/80/200、final、resume.pt |

## 保留的最新成品（勿动）

- **E3 最终模型**：`experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/best_step120/`（E4/E5/E6 均依赖，另有 `gold/soft/full_*_final` 三个对比模型）
- **E4/E5 正式产物**：`experiments/exp4_unseen/manifests/`（frozen_test/calibration/panel_v11_additions）、`experiments/e4e5_final_staticfix/`（FINAL_* 全套）、`experiments/exp5_calibration/e5/`
- **E6 v2 正式产物**：`experiments/exp6_v2_balanced/balanced/`（8 个正式文件）、`silver/`、`student/`、`tables/`、`figures/`、`EXP6_V2_FINAL_REPORT.md`
- **E2 最终成品**：`experiments/exp2_prior_work_comparison/balanced_design/`、`aegis2/baseline_predictions/` 等
- **E1 最终成品**：`experiments/exp1_input_ablation/report/`、`tables/`

## 注意事项

- `scripts/_kimi_balance_loop.py` 与 `_kimi_balance_monitor.log` 为仍在运行的成本监控进程（PID 31340）所用，保持原位未动。
- `experiments/exp4_unseen/audits/` 保留 7 个正式审计摘要（`gold_v4_final.jsonl`、`near_duplicate_pairs.jsonl`、`exact_overlap.json`、`exposure_audit*.jsonl`、`payload_audit.json`、`gold_audit.jsonl`）。
- 归档内容均为 Git 未跟踪/已删除状态，需要复现中间过程时从上述归档路径取回即可。
