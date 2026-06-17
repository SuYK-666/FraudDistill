# 数据准备状态

更新时间：2026-06-16

## 已就绪

| 数据 | 本地位置 | 状态 |
|---|---|---|
| Fraud-R1 GitHub 仓库 | `data/raw/fraudr1/repo` | 已存在 |
| Fraud-R1 base Chinese | `data/raw/fraudr1/repo/dataset/FP-base-full/FP-base-Chinese.json` | 已存在 |
| Fraud-R1 base English | `data/raw/fraudr1/repo/dataset/FP-base-full/FP-base-English.json` | 已存在 |
| Fraud-R1 levelup Chinese | `data/raw/fraudr1/repo/dataset/FP-levelup-full/FP-levelup-Chinese.json` | 已存在 |
| Fraud-R1 levelup English | `data/raw/fraudr1/repo/dataset/FP-levelup-full/FP-levelup-English.json` | 已存在 |
| Fraud focus smoke | `data/unified/fraud_focus_smoke.jsonl` | 已生成 12 条脱敏烟测样本 |

## 待正式实验前准备

| 数据 | 目标位置 | 用途 |
|---|---|---|
| Qwen target outputs | `data/generated_answers/fraudr1/qwen_outputs.jsonl` | 第一阶段 Target LLM 回答 |
| Fraud-R1 Qwen unified | `data/unified/fraudr1_qwen.jsonl` | 三个主实验的核心统一数据 |
| Fraud-R1 teacher signals | `data/teacher_signals/fraudr1_qwen_teacher.jsonl` | Student-AgentDistill 训练信号 |

这些文件需要调用 Target LLM 或 API Teacher，当前准备期没有批量生成。

## 辅助数据下载入口

主实验第一阶段以 Fraud-R1 为主。辅助数据可在正式实验扩展时下载：

```powershell
$env:PYTHONPATH='src'
python -m frauddistill.data.download_public --dataset do_not_answer
python -m frauddistill.data.download_public --dataset or_bench
python -m frauddistill.data.download_public --dataset aegis
```

辅助数据用途：

| 数据 | 用途 |
|---|---|
| Do-Not-Answer | 补充 `refusal_failure` 判断 |
| OR-Bench | 构造 hard safe cases，控制 `over_refusal` 和 FPR |
| Aegis / Nemotron | 抽取 fraud/deception/scams 相关安全样本 |

## 当前原则

- 不人工改写数据集正文。
- 不生成可复用欺诈脚本。
- 不把 smoke 占位标签当作正式 gold label。
- 正式实验前，需要使用目标模型真实回答替换 smoke placeholder。
