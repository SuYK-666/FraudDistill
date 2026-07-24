# Data & Judge Integration Gate v2.1 总报告

## 运行结论

本轮没有启动 Full。按照 NEXT_STEP 文档，本轮只做数据与裁判集成门控：冻结现有 1,594 条 generation，生成 qy_v3 pair 数据，去除 student 参与造标签，按 `source_prompt_id` 做 group split，并输出可追踪 manifest/hash。

## 关键数字

- 冻结 generation：1594 条
- source prompts：400 个
- target models：{'deepseek-chat': 400, 'moonshot-v1-8k': 394, 'qwen-plus': 400, 'glm-4-flash': 400}
- languages：{'en': 843, 'zh': 751}
- pair labels：{'safe': 1147, 'unsafe': 447}
- consensus coverage：1.0000
- abstain：0
- student used for label：False
- official guard available：False
- split cross-prompt leakage：{'cross_split_source_prompt_id': 0}

## 分析

这轮修复了上一轮最严重的循环评估问题：pair label 不再读取 FraudDistill student 的 `pred_label` 或 `score`。不过本地仍未实际运行 Qwen3Guard、WildGuard 或 PolyGuard 权重，因此 v2.1 只能判定为“结构性集成通过、官方 guard 仍 NO-GO”。E6 的行为表可用于调试 FAR/PLR/CRR，但不能写成目标 LLM 真实安全排名。

active runner 已优先读取 `data/processed/qy_v3/judged_pairs_v3.jsonl`，不再主动依赖旧 `v2_hard_control_full.jsonl`。输入冻结文件记录了 generation bank 的显式路径和 SHA-256，避免按修改时间混用旧模型或旧 generation。

## 产物

本地完整产物在 `outputs/data_judge_gate_v2_1/data_judge_gate_v2_1/` 与 `data/processed/qy_v3/`。GitHub 仅提交 `docs/results/DATA_JUDGE_GATE_V2_1_*` 摘要、审计和锁文件，不提交原始数据。
