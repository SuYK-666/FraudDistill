from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    sections = [
        "# FraudDistill 高标准六实验重测总览",
        "",
        "本轮已按复盘文档要求先归档旧结果，再执行 smoke、pilot、full 三阶段流程。smoke 与 pilot 已归档；当前 `outputs/*/high_standard_full/` 为完整重测结果。",
        "",
        "## 总体判断",
        "",
        "本轮修复了三个关键正确性问题：",
        "",
        "- E3 蒸馏不再灾难性退化，Full Distill 高于 Student-Gold，但增益仍小。",
        "- E5 低 FPR 表不再把 dev 约束伪写成 test 保证，并加入 test FPR UCB。",
        "- E6 行为指标不再从 Recall/FPR 机械复制，而是按 harmful/benign 与拒答行为重新计算。",
        "",
        "同时，本轮也保留了负结果：E1 在扩展 official-gold 数据后 `q+y` 未超过 `y_only`；E2 仍缺 official baseline；E6 仍缺独立 guard consensus，因此安全排名不能作为论文强结论。",
        "",
        "## 标签审计",
        csv_md(OUT / "audit_label_integrity" / "high_standard_full" / "dataset_label_audit.csv"),
        "",
        "审计结论：Aegis2.0、Do-Not-Answer、Fraud-R1 均有可计算二分类的正负类；OR-Bench hard-safe 是纯 safe 子集，只能报告 FPR/specificity，不应报告 Recall_unsafe 或 Macro-F1 主结论。重复 prompt hash 数量偏高，后续论文级版本仍需做近重复簇级 split。",
        "",
        "## E1 输入消融",
        csv_md(OUT / "exp1_input_ablation" / "high_standard_full" / "tables" / "main_table.csv"),
        "",
        "E1 分析：在更大 official-gold 混合集上，`y_only` 仍然最强，`q+y` 低于 `y_only`。这说明此前 hard-control 上的 q+y 召回增益不能直接外推到通用 response-level gold 数据。论文中可写 q_only 混淆意图与回答风险；不能写 q+y 总体显著优于 y_only。后续若要主推 q+y，应转向边界子集和 paired fix/FPR 叙事。",
        "",
        "## E2 现有工作对比",
        csv_md(OUT / "exp2_prior_work_comparison" / "high_standard_full" / "tables" / "main_table.csv"),
        "",
        "E2 分析：本轮不再将规则近似称为 official baseline。FraudDistill student proxy 在 Aegis2.0、Do-Not-Answer、Fraud-R1 上均能跑出有效指标，说明标签映射和 pipeline 修复有效；但 official WildGuard/AegisGuard/OR-Bench evaluator 仍不在当前仓库内，因此 E2 不能作为论文对比主表。该实验当前价值是数据审计和 proxy smoke。",
        "",
        "## E3 Agent 与蒸馏",
        "### Student 消融",
        csv_md(OUT / "exp3_agent_distillation_ablation" / "high_standard_full" / "tables" / "student_ablation.csv"),
        "",
        "### Agent 消融",
        csv_md(OUT / "exp3_agent_distillation_ablation" / "high_standard_full" / "tables" / "agent_ablation.csv"),
        "",
        "E3 分析：Full Distill Macro-F1=0.8105，高于 Student-Gold=0.8010，灾难性退化已经修复。各蒸馏项形成梯度：teacher label、soft、type、full 都略有差异，说明不再是完全未生效。但增益只有约 +0.0095，未达到强蒸馏主张。Agent 表仍弱，full learned proxy 没有明显超过 fraud only，不能宣称多 Agent 组件均有效。",
        "",
        "## E4 Unseen 泛化",
        csv_md(OUT / "exp4_unseen" / "high_standard_full" / "tables" / "main_table.csv"),
        "",
        "E4 分析：fake_job_postings 泛化最强，Macro-F1=0.9610；impersonation 次之，Macro-F1=0.8566；phishing_scams FPR=0.1528，是主要弱项。OR-Bench hard-safe FPR=0.0767，已低于 0.10。可写类别迁移存在梯度，不同欺诈类型难度不同；不能写所有 unseen 轴均强通过，因为 source-held-out unsafe 轴仍不足。",
        "",
        "## E5 校准与阈值",
        csv_md(OUT / "exp5_calibration" / "high_standard_full" / "tables" / "calibration_table.csv"),
        "",
        "E5 分析：Platt default 将 FPR 从 0.1626 降至 0.0481，ECE 从 0.1427 降至 0.0157，Brier 从 0.1041 降至 0.0743，校准结论可用。dev-UCB FPR<=0.05 的 test FPR=0.0417，Recall=0.5918，说明低误报可控但召回代价较大。dev-UCB FPR<=0.10 的 Recall=0.7736，但 test UCB=0.103，略超 0.10。论文中应主推 Platt 概率校准，不要夸大低 FPR 部署。",
        "",
        "## E6 多 API",
        "### Student vs prompt reference",
        csv_md(OUT / "exp6_multi_api" / "high_standard_full" / "tables" / "student_vs_prompt_reference.csv"),
        "",
        "### Behavior metrics",
        csv_md(OUT / "exp6_multi_api" / "high_standard_full" / "tables" / "behavior_metrics_by_model.csv"),
        "",
        "E6 分析：重评后四模型 Macro-F1 仍只有 0.53-0.57，说明冻结 Student 跨模型检测不稳。行为指标已拆开：Kimi 的 FAR/UAR 最低、ORR 较高；DeepSeek FAR 较高；但由于缺少独立 guard consensus，这只能写成 detector-dependent observation，不能写成真实安全排名。",
        "",
        "## 当前可用于论文的叙事",
        "",
        "- q_only 明显不是回答级风险检测的充分输入。",
        "- Platt 校准是当前最稳的正面结果。",
        "- E4 类别泛化存在清晰梯度，fake job 最强、phishing 最难。",
        "- E3 蒸馏已从错误实现修复到小幅正迁移，但不是强贡献。",
        "",
        "## 当前不能写的结论",
        "",
        "- 不能写 q+y 在全数据上显著强于 y_only。",
        "- 不能写 FraudDistill 已优于四个现有工作。",
        "- 不能写多 Agent 每个组件都有显著贡献。",
        "- 不能写 E6 的模型安全排名是真实排名。",
    ]
    (OUT / "HIGH_STANDARD_RERUN_SUMMARY.md").write_text("\n".join(sections), encoding="utf-8")


def csv_md(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "_缺失或为空_"
    rows = list(csv.reader(path.open("r", encoding="utf-8")))
    return "\n".join([
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * len(rows[0])) + " |",
        *["| " + " | ".join(row) + " |" for row in rows[1:]],
    ])


if __name__ == "__main__":
    main()
