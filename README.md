# FraudDistill

FraudDistill 是一个面向 **LLM 欺诈协助输出检测** 的研究与实验框架。项目关注的不是传统短信、邮件或网页诈骗文本分类，也不是只判断用户 prompt 是否危险，而是检测：

```text
user_query q + target_model_answer y -> safe / unsafe, risk_type, risk_score
```

核心问题是：当目标大模型面对欺诈诱导、角色扮演、多轮上下文或越狱式输入时，它是否生成了具有欺诈协助风险的回答，以及如何用较低成本自动检测这类回答。

## 项目定位

当前版本只保留一条主线：

```text
Prompt 诱导 -> Target LLM 生成回答 -> 检测是否存在欺诈协助、拒答失败或过度拒答
```

小模型和多 Agent 都是检测手段，不是研究对象本身。

当前风险类型限定为：

| 类型 | 含义 |
|---|---|
| `fraud_assistance` | 回答为诈骗、钓鱼、冒充身份、虚假招聘、情感诈骗等提供可执行协助 |
| `refusal_failure` | 面对明显欺诈诱导请求时，模型没有拒绝并给出实质帮助 |
| `over_refusal` | 面对反诈教育、识别诈骗、防范诈骗等正常请求时错误拒绝 |
| `safe` | 正常回答、正确拒答或安全教育 |

幻觉、事实性错误、RAG 证据一致性等旧方向暂不作为当前主实验。

## 方法概览

```text
公开数据集 / benchmark
        -> 统一为 q + y 样本
        -> Multi-Agent Teacher 生成结构化 teacher signal
        -> 本地 Student Detector 学习 gold/weak label + teacher signal
        -> 低成本部署的欺诈协助输出检测器
```

保留的 Agent：

| Agent | 作用 |
|---|---|
| Fraud Assistance Agent | 判断回答是否提供欺诈协助，关注可执行性 |
| Refusal Quality Agent | 判断拒答失败和过度拒答 |
| Relevance Agent | 判断回答是否真正回应用户意图 |
| Arbiter Agent | 汇总 teacher signal |

Factuality Agent 代码暂保留，但第一阶段不作为主线实验组件。

## 数据集

主数据集：

| 数据集 | 用途 |
|---|---|
| Fraud-R1 | 核心数据来源，第一阶段聚焦 Phishing Scams、Impersonation、Fake Job Postings |

辅助数据集：

| 数据集 | 用途 |
|---|---|
| Do-Not-Answer | 补充危险请求应拒绝的基线能力 |
| Aegis / Nemotron Content Safety Dataset | 补充内容安全 safe/unsafe 样本，优先抽取 fraud/deception/scams 相关子类 |
| OR-Bench | 提供 hard safe cases，用于控制过度拒绝和误报 |

当前仓库已包含 Fraud-R1 原始仓库数据。正式实验前建议先构造：

```text
data/generated_answers/fraudr1/qwen_outputs.jsonl
data/unified/fraudr1_qwen.jsonl
data/teacher_signals/fraudr1_qwen_teacher.jsonl
```

注意：报告和公开材料中不展示可复用的高风险 prompt、诈骗话术或完整欺诈脚本。

## 项目结构

```text
FraudDistill/
├── configs/
│   ├── agents/
│   ├── data/
│   ├── experiments/
│   └── student/
├── data/
│   ├── raw/
│   ├── generated_answers/
│   ├── unified/
│   ├── teacher_signals/
│   └── predictions/
├── outputs/
├── src/frauddistill/
│   ├── agents/
│   ├── data/
│   ├── eval/
│   ├── experiments/
│   ├── student/
│   ├── target_llm/
│   ├── teacher/
│   └── utils/
└── tests/
```

## 安装

```powershell
pip install -r requirements.txt
pip install -e .
```

如后续需要本地 Student 微调，再安装：

```powershell
pip install -e ".[student]"
```

## API Key 与多模型接口

复制模板：

```powershell
Copy-Item api_keys.template.py api_keys.py
```

然后在 `api_keys.py` 中填写本地 key。`api_keys.py` 已在 `.gitignore` 中，不应提交。

本地接口统一使用 OpenAI-compatible Chat Completions 形式，后续只需要在 `api_keys.py` 填入 key。默认不会自动发起大规模 API 调用。

支持的 provider：

| Provider | 默认模型 | Base URL |
|---|---|---|
| Qwen / DashScope | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| DeepSeek | `deepseek-chat` | `https://api.deepseek.com` |
| OpenAI | `gpt-4.1-mini` | `https://api.openai.com/v1` |
| Gemini | `gemini-2.5-flash` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Kimi / Moonshot | `moonshot-v1-8k` | `https://api.moonshot.cn/v1` |
| GLM / ZhipuAI | `glm-4-flash` | `https://open.bigmodel.cn/api/paas/v4` |
| Doubao / Volcano Ark | `doubao-seed-1-6-flash` | `https://ark.cn-beijing.volces.com/api/v3` |
| OpenRouter | `openai/gpt-4.1-mini` | `https://openrouter.ai/api/v1` |

模型清单位于：

```text
configs/models.yaml
```

轻量阶段建议先只启用 2-3 个目标模型；正式扩展阶段再打开更多 provider，并记录实际模型版本、调用日期、参数和中转平台。

## 轻量准备与烟测

构造极小 Fraud-R1 聚焦版 smoke 数据。该命令只读取本地 Fraud-R1 prompt，并写入脱敏占位回答，不调用 API：

```powershell
python -m frauddistill.data.prepare_fraud_focus `
  --input_files `
  data/raw/fraudr1/repo/dataset/FP-base-full/FP-base-Chinese.json `
  data/raw/fraudr1/repo/dataset/FP-base-full/FP-base-English.json `
  data/raw/fraudr1/repo/dataset/FP-levelup-full/FP-levelup-Chinese.json `
  data/raw/fraudr1/repo/dataset/FP-levelup-full/FP-levelup-English.json `
  --output_file data/unified/fraud_focus_smoke.jsonl `
  --limit 12
```

跑三个实验的离线 smoke：

```powershell
python -m frauddistill.experiments.fraud_detection_smoke `
  --input_file data/unified/fraud_focus_smoke.jsonl `
  --output_dir outputs/fraud_detection_smoke `
  --limit 12
```

该 smoke 会生成：

```text
outputs/fraud_detection_smoke/fraud_detection_smoke_metrics.json
outputs/fraud_detection_smoke/rule_predictions.jsonl
```

## Phase A 补充实验流水线

新增补充实验围绕四类输出组织：

| 步骤 | 输出 | 说明 |
|---|---|---|
| Prompt pool | `data/prompts/*.jsonl` | 统一公开数据、hard safe 和反诈教育 prompt |
| Target generations | `data/generations/*.jsonl` | 多目标 LLM 回答，记录模型、参数、延迟和错误 |
| Raw votes | `data/labels/raw_votes/*.jsonl` | 规则拒答检测、离线 teacher、后续外部 judge/guard 投票 |
| Silver labels | `data/labels/silver/*.jsonl` | 聚合为 `silver_high`、`silver_medium`、`ambiguous` |

先做不调用 API 的极小链路测试：

```powershell
python -m frauddistill.target_llm.generate_responses `
  --input_file data/unified/fraud_focus_smoke.jsonl `
  --output_file data/generations/generations_smoke_dryrun.jsonl `
  --limit 2 `
  --dry_run

python -m frauddistill.labelers.run_auto_labelers `
  --input_file data/generations/generations_smoke_dryrun.jsonl `
  --output_file data/labels/raw_votes/smoke_votes.jsonl

python -m frauddistill.labelers.aggregator `
  --input_file data/labels/raw_votes/smoke_votes.jsonl `
  --output_file data/labels/silver/smoke_silver.jsonl
```

接入真实 API 后，去掉 `--dry_run`，并通过 `configs/models.yaml` 控制启用模型。可用 `--model qwen-plus` 对单个模型做小范围验证。

## 全量数据准备

全量数据准备只构造 prompt pool 和已有 q+y evaluation set，不调用任何模型 API：

```powershell
python -m frauddistill.data.prepare_full_experiment_data `
  --output_root data/prepared/full
```

主要产物：

```text
data/prepared/full/MANIFEST.md
data/prepared/full/prompts/all_target_prompts.jsonl
data/prepared/full/evaluation_qy/
```

当前全量准备包含 Fraud-R1、OR-Bench、Do-Not-Answer、Aegis、HaluEval、RAGTruth、HaluBench 和 FELM。WildGuardMix 是 Hugging Face gated dataset，需要账号授权后再下载；DetoxBench 暂未发现单一官方可下载数据包，因此不混入非官方数据。

## 六个正式实验入口

本轮论文实验统一入口为：

```powershell
python scripts/run_high_standard_rerun.py all --bootstrap 500
python scripts/write_six_experiment_master_report.py
```

当前 CCF-A 重定位审查先跑小规模 qwen 版，不直接覆盖为全量结论：

```powershell
python scripts/run_high_standard_rerun.py small --bootstrap 300 --small-limit 720 --api-provider qwen --api-probe-limit 6
python scripts/write_six_experiment_master_report.py --run-id ccfa_small_qwen
```

该命令会先把既有 `outputs/` 内容归档到 `archive/pre_ccfa_small_qwen_rerun_*`，然后只保留本轮小规模输出在 `outputs/*/ccfa_small_qwen/`。实验 1-5 会调用 qwen 生成 teacher probe 原始记录，但这些记录只作为训练期/审计信号，不写入 gold label，也不在 Student 推理时作为特征。实验 6 继续读取当前可用的 qwen、DeepSeek、Kimi、GLM generation bank。

全量版仍可按 `smoke -> pilot -> high_standard_full` 执行；smoke/pilot 自动归档，正式结果保留在 `outputs/*/high_standard_full/`。GitHub 上可查看报告副本：

```text
docs/results/SIX_EXPERIMENTS_MASTER_REPORT_中文.md
docs/reproduction/REPRODUCE_SIX_EXPERIMENTS.md
```

六个实验分别对应：

| 实验 | 输出目录 | 重点 |
|---|---|---|
| E1 输入边界消融 | `outputs/exp1_input_ablation/ccfa_small_qwen/` 或 `high_standard_full/` | q only、y only、q+y、matched-FPR、matched-Recall、McNemar |
| E2 现有工作对比 | `outputs/exp2_prior_work_comparison/ccfa_small_qwen/` 或 `high_standard_full/` | proxy coverage 与官方 baseline 缺口审计 |
| E3 Agent/蒸馏消融 | `outputs/exp3_agent_distillation_ablation/ccfa_small_qwen/` 或 `high_standard_full/` | nested ablation、leave-one-out、组件压力表、Student 梯度 |
| E4 unseen 泛化 | `outputs/exp4_unseen/ccfa_small_qwen/` 或 `high_standard_full/` | leave-one-category-out 与 hard-safe source holdout |
| E5 校准 | `outputs/exp5_calibration/ccfa_small_qwen/` 或 `high_standard_full/` | Platt、FPR-UCB 阈值、reliability 数据 |
| E6 多 API | `outputs/exp6_multi_api/ccfa_small_qwen/` 或 `high_standard_full/` | 多目标模型 generations、行为指标、detector-dependent 排名 |

`data/`、`outputs/`、`archive/`、模型文件和 `api_keys.py` 均不提交到 GitHub；公开仓库只保留代码、配置、复现说明和报告副本。

当前 CCF-A 重新定位版额外增加三类弱评测轨道：

| 轨道 | 文件 | 说明 |
|---|---|---|
| Context-Critical paired | `outputs/exp1_input_ablation/high_standard_full/tables/context_critical_table.csv` | 同一或高度相似回答在不同 q 下发生 safe/unsafe 语义反转，用于证明 q 的信息增益 |
| Component Stress | `outputs/exp3_agent_distillation_ablation/high_standard_full/tables/stress_agent_ablation.csv` | 程序化构造 actionable、partial leakage、hard-safe、conflict 压力集 |
| Procedural five-category LOCO | `outputs/exp4_unseen/high_standard_full/tables/procedural_loco5.csv` | 五类欺诈均含 safe/unsafe 对照，用于观察未见类别趋势 |

这些轨道标注为 `procedural_weak_*`，用于论文叙事中的受控机制验证；官方 gold 主张仍应以公开 benchmark 或官方 evaluator 为准。

## 旧版三个主实验

| 实验 | 配置 | 目的 |
|---|---|---|
| 实验一：输入边界消融 | `configs/experiments/exp1_input_ablation_fraud.yaml` | 比较 `q only`、`y only`、`q + y`，证明检测对象必须包含用户请求与模型回答 |
| 实验二：多 Agent 教师蒸馏 | `configs/experiments/exp2_agent_distillation_fraud.yaml` | 比较 Student-Gold 与 Student-AgentDistill |
| 实验三：轻量部署与泛化 | `configs/experiments/exp3_deployment_generalization_fraud.yaml` | 验证新欺诈类别、新 Target LLM 与低误报约束下的表现 |

默认配置不会启动大规模实验、训练或批量 API 调用。正式实验请在明确指令后运行。

## 当前实验产物

截至 2026-06-17，V1 与 V2 三个实验正式版均已完成。V1 主要用于管线验证；当前更推荐引用 V2 hard-control setting：

| 实验 | 输出目录 | 报告 |
|---|---|---|
| 实验一：输入边界消融 | `outputs/exp1_final/` | `outputs/exp1_final/EXP1_REPORT.md` |
| 实验二：多 Agent 教师蒸馏 | `outputs/exp2_final/` | `outputs/exp2_final/EXP2_REPORT.md` |
| 实验三：轻量部署与泛化 | `outputs/exp3_final/` | `outputs/exp3_final/EXP3_REPORT.md` |
| 实验一 V2：Hard-Control 输入边界消融 | `outputs/v2_exp1_final/` | `outputs/v2_exp1_final/EXP1_V2_REPORT.md` |
| 实验二 V2：Hard-Control 教师蒸馏 | `outputs/v2_exp2_final/` | `outputs/v2_exp2_final/EXP2_V2_REPORT.md` |
| 实验三 V2：Hard-Control 泛化与部署 | `outputs/v2_exp3_final/` | `outputs/v2_exp3_final/EXP3_V2_REPORT.md` |

V2 引入 Qwen 生成的多样 safe answers、反诈教育 hard safe、OR-Bench hard safe，以及 Qwen 漏检/Phishing 边界 hard unsafe。smoke、预演和 API 分片中间文件已归档到 `archive/`；主路径保留最终版数据、预测、表格和报告。

## Student Detector

推荐本地 Student：

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

配置文件：

```text
configs/student/deepseek_r1_distill_qwen_1_5b_qlora.yaml
```

本项目最多只建议部署一个本地小模型。第一阶段可先跑 Student-ZeroShot，再进行 LoRA/QLoRA 小样本微调。

## 测试

```powershell
pytest -q
```

测试覆盖 schema、转换器、离线 teacher、指标、API 客户端配置和端到端烟测。

## 安全与复现原则

- 不人工改写公开数据集正文。
- 不进行大规模人工标注。
- 不把 teacher signal 伪称为 gold label；若无官方 evaluator，应写作 weak supervision。
- 不展示可复用的诈骗 prompt、完整话术或欺诈脚本。
- 默认脚本不进行大规模训练或批量 API 调用。
- API Teacher 只用于训练期信号生成；最终目标是本地低成本 Student Detector。

## License

当前仓库尚未指定许可证。正式开源前建议补充 `LICENSE`，并再次确认所有外部数据集的许可条款。
