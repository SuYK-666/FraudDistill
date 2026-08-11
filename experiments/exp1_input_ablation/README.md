# E1 Input-Ablation（实验1）正式目录

本文件夹为 **Experiment 1（E1-A / E1-B / E1-C）** 的正式产物目录。

## 结构
- `report/` — 最终中文实验报告（含表格与图）
- `tables/` — 论文用表格（Markdown/CSV）
- `data/` — 冻结面板、审计、统计结果（含 protocol lock 与 final 数据集）
- `archive/` — 中间过程文件归档（task manifests、votes、pilot 等，已弃用/可复现）

## 实验概览
| 子实验 | 内容 | 状态 |
|---|---|---|
| E1-A | 自然低基率欺诈协助发生率（A7500 冻结 registry） | 数据完成，Gate 通过 |
| E1-B | q / y / q+y 输入边界关系消融（6000 行冻结面板） | 完成（15/15，Gate 全过，v4.6 CI 修正） |
| E1-C | 独立自然低基率迁移回放（case-level 独立 reserve） | 完成（离线回放，方向性/探索性支持） |

## 关键考核（指南 §9.5）
- Scientific Gate：Δ_joint>0、cluster-bootstrap CI 下界>0、Holm p<0.05、q+y>wrong-q+y、shortcut audits PASS
- Target Gate：q+y Macro-F1 ≥ 0.90、Δ_joint ≥ 0.05、5 seeds 中 ≥4/5 q+y>best-single
