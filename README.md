# FraudDistill

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-473%20passed%20%7C%204%20skipped-brightgreen)
![Experiments](https://img.shields.io/badge/experiments-6%20completed-4E9A51)
![Student](https://img.shields.io/badge/student-1.5B%20QLoRA-blueviolet)
![Status](https://img.shields.io/badge/status-research%20prototype-lightgrey)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

FraudDistill 是一个面向 **LLM 欺诈协助输出检测**（fraud-assistance response detection）的研究与实验框架。

本项目检测的不是“用户 prompt 是否危险”，也不是传统的诈骗文本分类，而是目标大模型面对欺诈诱导时**实际生成的回答**是否构成欺诈协助：

```text
user_query q + target_model_answer y  ->  safe / unsafe, risk_type, risk_score
```

核心主张：欺诈协助是一个**响应级且关系性（response-level & relational）**的属性——必须联合观察 `q` 与 `y` 才能可靠判断，单看查询或单看回答都不可靠。

## 亮点

- **关系性检测范式**：三视图（q-only / y-only / q+y）联合评估，配 wrong-q 负控制，机制证据来自受控消融实验。
- **增强型 Multi-Agent Teacher**：三专家（欺诈协助 / 拒答质量 / 相关性）+ Evidence Arbiter + 冲突纠错 + Platt 校准，产出结构化教师信号。
- **可部署的 1.5B 学生模型**：FraudDistill-Student-1.5B（QLoRA 蒸馏），Macro-F1 0.9135，支持 head-tail 编码的低成本本地推理。
- **低成本级联修复**：分数歧义选择性审计（15% 查询率）可在不重训模型的情况下显著恢复部署召回。
- **完整实验体系**：E1 机制验证 → E2 跨工作对比 → E3 教师/学生蒸馏 → E4 未见泛化 → E5 校准级联 → E6 多 API 实测，全部含冻结数据、统计检验与复现材料。

## 核心任务定义

| 风险类型 | 含义 |
|---|---|
| `fraud_assistance` | 回答为诈骗、钓鱼、冒充身份、虚假招聘、情感诈骗等提供可执行协助 |
| `refusal_failure` | 面对明显欺诈诱导请求时，模型没有拒绝并给出实质帮助 |
| `over_refusal` | 面对反诈教育、识别诈骗、防范诈骗等正常请求时错误拒绝 |
| `safe` | 正常回答、正确拒答或安全教育 |

## 方法概览

```text
公开数据集 / benchmark
    -> 统一为 q + y 响应级样本
    -> Multi-Agent Teacher 生成结构化 teacher signal
    -> 本地 Student Detector 学习 gold/weak label + teacher signal
    -> 低成本部署的欺诈协助输出检测器
```

框架组件（代码位于 `src/frauddistill/`）：

| 组件 | 作用 |
|---|---|
| Fraud Assistance Agent | 判断回答是否提供欺诈协助，关注可执行性 |
| Refusal Quality Agent | 判断拒答失败和过度拒答 |
| Relevance Agent | 判断回答是否真正回应用户意图 |
| Evidence Arbiter | 汇总三专家证据，输出结构化分数与证据表 |
| Student Detector | 1.5B QLoRA 学生模型，本地推理（含 head-tail 截断编码） |

## 实验总览（E1–E6）

| # | 实验 | 目录 | 一句话目标 |
|---|---|---|---|
| E1 | 输入消融与关系性机制验证 | [`experiments/exp1_input_ablation/`](experiments/exp1_input_ablation/) | 证明 q+y 联合观察的必要性 |
| E2 | 跨工作对比（平衡诊断集） | [`experiments/exp2_prior_work_comparison/`](experiments/exp2_prior_work_comparison/) | 四基准同数据集对比原工作基线 |
| E3 | 增强多 Agent 教师与蒸馏消融 | [`experiments/exp3_agent_distillation_ablation/`](experiments/exp3_agent_distillation_ablation/) | 教师分解价值 + 1.5B 学生蒸馏 |
| E4 | 未见类别/来源/风格泛化 | [`experiments/exp4_unseen/`](experiments/exp4_unseen/) | family-disjoint 复合迁移评估 |
| E5 | 校准与选择性审计级联 | [`experiments/exp5_calibration/`](experiments/exp5_calibration/) | 低代价恢复部署召回 |
| E6 | 跨多 API 直连响应检测 | [`experiments/exp6_balanced_multi_api/`](experiments/exp6_balanced_multi_api/) | 真实目标模型行为率与判别 |

> 📖 每个实验的**实验思路、实验设计、数据集选取、主结果表格与实验分析**详见：[`experiments/EXPERIMENTS_SUMMARY.md`](experiments/EXPERIMENTS_SUMMARY.md)（六实验总结文档，含全部报告与数据链接）。

## 仓库结构

```text
FraudDistill/
├── configs/                  # 数据集 / 模型 / 阈值 / 实验配置
├── data/prepared/            # 各实验冻结面板与中间数据（raw 高风险内容不入库）
├── experiments/
│   ├── exp1_input_ablation/  # E1 输入消融（报告 + 论文表格）
│   ├── exp2_prior_work_comparison/  # E2 跨工作对比（平衡诊断集 + 八行主表）
│   ├── exp3_agent_distillation_ablation/  # E3 教师/学生蒸馏（最终模型）
│   ├── exp4_unseen/          # E4 未见泛化（冻结 test/calibration manifest）
│   ├── exp5_calibration/     # E5 校准与选择性审计
│   ├── exp6_balanced_multi_api/  # E6 多 API 平衡重跑
│   ├── e4e5_final_staticfix/ # E4/E5 最终静态修复产物
│   └── EXPERIMENTS_SUMMARY.md # 六实验总结文档
├── scripts/                  # 实验构建 / 评测 / 报告脚本
├── src/frauddistill/
│   ├── agents/               # 专家 Agent 与 Arbiter
│   ├── teacher/              # 教师信号、证据表、校准
│   ├── student/              # 学生模型（构建 / 训练 / 推理）
│   ├── providers/            # 多 API provider 客户端
│   ├── e4e5_v2/  e1_final_v4/  exp2_cross_benchmark/  # 实验实现
│   └── ...
├── tests/                    # pytest 测试套件（473 passed）
└── third_party/              # 第三方基线代码（不随包分发）
```

## 安装

```powershell
pip install -r requirements.txt
pip install -e .
```

本地学生模型训练/推理（PyTorch + Transformers）：

```powershell
pip install -e ".[student]"
```

## API Key 配置

```powershell
Copy-Item api_keys.template.py api_keys.py
```

然后在 `api_keys.py` 中填写本地 key（`api_keys.py` 已在 `.gitignore` 中，不应提交）。本地接口统一使用 OpenAI-compatible Chat Completions 形式；默认不会自动发起大规模 API 调用。

## 测试

```powershell
pytest -q
```

测试覆盖 schema、转换器、离线教师、指标、API 客户端配置与端到端烟测（473 passed / 4 skipped）。

## 安全与复现原则

- 不人工改写公开数据集正文；不把 teacher signal 伪称为 gold label（无官方 evaluator 时写作 weak supervision）。
- 不展示可复用的高风险 prompt、诈骗话术或完整欺诈脚本；**原始 q+y 数据仅保留在本机**，`.gitignore` 已保护相关路径。
- 默认脚本不进行大规模训练或批量 API 调用；预算、缓存与断点机制见 `src/frauddistill/providers/` 与各实验协议。
- 所有实验结果均附带冻结清单、数据审计（SHA256）、统计检验与复现材料，归档目录 `archive/` 不进入 Git 仓库。

## License

当前仓库尚未指定许可证；正式开源前将补充 `LICENSE`，并再次确认所有外部数据集的许可条款。