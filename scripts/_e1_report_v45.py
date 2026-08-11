# -*- coding: utf-8 -*-
"""E1 report v4.5/v4.6 update: new wrong-q control, Macro-F1 bootstrap CI,
clean-anchor sensitivity, B gold agreement, and wording corrections."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments" / "exp1_input_ablation" / "report" / "E1_FINAL_REPORT.md"
TABLES = ROOT / "experiments" / "exp1_input_ablation" / "tables" / "E1_PAPER_TABLES.md"
PROGRESS = ROOT / "experiments" / "exp1_input_ablation" / "PROGRESS.md"


def sub(t: str, old: str, new: str, tag: str) -> str:
    assert old in t, f"MISSING: {tag}"
    return t.replace(old, new)


def main() -> None:
    # ---------------- report ----------------
    t = REPORT.read_text(encoding="utf-8-sig")

    # A. 2.3 descriptive wording
    old = """分层要点：
- 语言维度：英文响应 11/3754（0.29%），中文响应 0/3746（0%）——本数据集中欺诈协助主要集中在英文诱导场景；
- 场景维度：roleplay 场景 9/3218（0.28%）明显高于 assistant 场景 2/4282（0.05%），角色扮演类诱导更易诱发协助行为；
- 类别维度：fake_job_posting 8/980（0.82%）与 network_friendship 3/448（0.67%）最集中，phishing / impersonation / fraudulent_service 为 0——说明欺诈协助并非均匀分布，而是集中于特定诱导类型。"""
    new = """分层要点（描述性观察，central positive 总数仅 11，不作统计推断）：
- 语言维度：11 个 central positive 全部出现在英文响应（11/3754，0.29%），中文响应 0/3746（0%）——在本数据集的诱导分布下，正例集中于英文场景；
- 场景维度：roleplay 9/3218（0.28%）高于 assistant 2/4282（0.05%）——描述性差异，受极小样本限制不宜外推；
- 类别维度：fake_job_posting 8/980（0.82%）与 network_friendship 3/448（0.67%）出现正例，phishing / impersonation / fraudulent_service 为 0——描述性观察显示正例并非均匀分布，而是集中于特定诱导类型。"""
    t = sub(t, old, new, "A 2.3")

    # B. 4.4 B-specific gold agreement
    old = """- Gold 完成率 = 1.0；valid schema 率 = 1.0；全部 disagreement 已 adjudication 清零；
- 双 Judge agreement 面板级 0.9984（E1-A registry 口径）与 B 面板 double-gold 协议一致，质量满足论文要求。"""
    new = """- Gold 完成率 = 1.0；valid schema 率 = 1.0；全部 disagreement 已 adjudication 清零（最终 gold = 双 Judge 共识 + adjudication）；
- **B 面板自身的双 Judge binary agreement = 0.9281**（在 4268/6000 行有双 Judge 投票的面板行上统计；投票以 material_central 口径解析、每行取最后一次投票；≥ 质量门槛 0.90 ✅）；
- 分层 agreement：B1 0.9575（n=2000）/ B2 0.9293（n=1500）/ B3 0.8490（n=768）；B3 双投票覆盖率较低（768/2000）因其大量行复用 v3.2 冻结 gold（source_derived_open_control / aegis / real_target 等）；
- 说明：报告中的 0.9984 仅指 E1-A registry 口径，与 B 面板 0.9281 分开表述，不混用。"""
    t = sub(t, old, new, "B 4.4")

    # C. 5.1 near-dup honest + forward ref
    old = """- near-dup y（跨 split 近似重复 y）= 357，其中同标签 292（同标签近似重复不构成标签泄漏）；"""
    new = """- near-dup y（跨 split 近似重复 y，定义：归一化 y 前 80 字符相同）= 357，其中同标签 292、跨标签 65；其中 Anchor 内与 model_dev/calibration 近重复的行 = 212；
- 说明：同标签近重复虽不构成"标签反转"式泄漏，但生成模板可能在训练/测试之间共享回答风格，存在模板记忆的残余风险；为此新增完全离线的 clean-anchor 敏感性分析（剔除 Anchor 中 212 行近重复 y 后重算 q / y / q+y，见 §7.4），结论不变，判定该风险不改变主结论；"""
    t = sub(t, old, new, "C 5.1")

    # D. 6 status
    old = "| M1 XLM-R joint encoder | 语义联合编码器（learned primary）：xlm-roberta-base，中英混合，5 seeds × 3 views | ⏳ 训练中（0/15） |"
    new = "| M1 XLM-R joint encoder | 语义联合编码器（learned primary）：xlm-roberta-base，中英混合，5 seeds × 3 views | ✅ 完成（15/15，服务器 GPU 训练） |"
    t = sub(t, old, new, "D 6")

    # E. 7.3 table wrong_q_y row
    old = "| wrong_q_y | 0.7609 ± 0.0022 | 0.7315 | 0.6473 | 0.8377 | 0.7281 | 0.3130 |"
    new = "| wrong_q_y | 0.7574 ± 0.0039 | 0.7327 | 0.6632 | 0.8200 | 0.7300 | 0.3033 |"
    t = sub(t, old, new, "E 7.3 row")

    # F. 7.3 stats bullets
    old = "- **Δ_joint = 0.9685 − 0.8017 = +0.1717**（目标 ≥ 0.05 ✅）；bootstrap 95% CI = [0.1213, 0.1498]，p(Δ ≤ 0) = 0.0 ✅；"
    new = "- **Δ_joint = 0.9685 − 0.8017 = +0.1717**（目标 ≥ 0.05 ✅）；family-cluster bootstrap 95% CI（Macro-F1 口径，v4.6 修正）= [0.1511, 0.1928]，p(Δ ≤ 0) = 0.0 ✅；"
    t = sub(t, old, new, "F1")

    old = "q_y vs wrong_q_y b=256 / c=7，p = 2.21e-66；"
    new = "q_y vs wrong_q_y b=257 / c=7，p = 1.13e-66；"
    t = sub(t, old, new, "F2")

    old = "- **关系性负控制**：错误配对 wrong_q_y 仅 0.7609（AUROC 0.7315），比 q_y 低 0.2075，说明模型确实利用 q 与 y 的**关系**而非仅 y 的表面特征；"
    new = "- **关系性负控制（v4.5 重建）**：错误配对 wrong_q_y 仅 0.7574（AUROC 0.7327），比 q_y 低 0.2111，说明模型确实利用 q 与 y 的**关系**而非仅 y 的表面特征；wrong-q map 已按预注册口径重建为**同语言 + 同 fraud category + 不同 family**（1200/1200 全部满足，0 回退），并据此离线重跑 M1 wrong_q_y 推理、重新收集 Qwen/DeepSeek 的 wrong_q_y 投票；"
    t = sub(t, old, new, "F3")

    old = "- 备注：bootstrap CI 来自 family-cluster 重抽样（以 family 为单位、权重与全样本不同），其分布中心与全样本点估计存在小幅差异；CI 下界仍远大于 0，结论稳健。"
    new = "- 备注：v4.6 修正了 bootstrap 实现（原实现只对正类计算 F1，现与点估计一致按 Macro-F1 重抽样），修正后 CI 以点估计为中心；Qwen / DeepSeek 单视图交叉验证的 Δ 95% CI 亦相应更新（Qwen [0.0396, 0.0737] / DeepSeek [0.0605, 0.0969]）。"
    t = sub(t, old, new, "F4")

    # G. insert 7.4 clean-anchor sensitivity before section 8
    sec8 = "\n## 8. E1-C：独立自然低基率迁移"
    block74 = (
        "### 7.4 Clean-Anchor 敏感性分析（跨 split 近重复 y 的离线检验）\n\n"
        "**动机**：§5.1 显示 Anchor 中有 212 行 y 与 model_dev/calibration 共享归一化前 80 字符（模板/前缀近重复），存在\"回答风格模板记忆\"的残余风险。本分析不重建 split、不重训模型：仅从冻结 Anchor 中剔除这 212 行，用既有冻结预测（M1 本地推理 + Qwen/DeepSeek 投票）在 clean subset 上重算指标。\n\n"
        "| 项目 | 数值 |\n|---|---|\n"
        "| Anchor 总行数 | 1200 |\n"
        "| 剔除（近重复 y） | 212（B1 70 / B2 75 / B3 67；正 69 / 负 143） |\n"
        "| Clean subset | 988（正 531 / 负 457） |\n\n"
        "Clean subset 指标（5 seeds 均值 ± sd，M1 XLM-R）：\n\n"
        "| View | Macro-F1 | AUROC | AUPRC | Recall | FPR |\n|---|---|---|---|---|---|\n"
        "| q_only | 0.6713 ± 0.0196 | 0.7303 ± 0.0163 | 0.6919 ± 0.0243 | 0.9122 ± 0.0545 | 0.5492 ± 0.0698 |\n"
        "| y_only | 0.8011 ± 0.0045 | 0.9196 ± 0.0027 | 0.9336 ± 0.0040 | 0.8414 ± 0.0988 | 0.2376 ± 0.1111 |\n"
        "| **q_y** | **0.9700 ± 0.0031** | **0.9943 ± 0.0011** | 0.9940 ± 0.0015 | 0.9849 ± 0.0087 | 0.0468 ± 0.0112 |\n"
        "| wrong_q_y | 0.7509 ± 0.0029 | 0.7166 ± 0.0186 | 0.6800 ± 0.0172 | 0.8249 ± 0.0151 | 0.3256 ± 0.0125 |\n\n"
        "**敏感性统计（seed-0 口径，与 §7.3 一致）**：\n"
        "- **Clean Δ_joint = +0.1784**（q_y 0.9724 vs best-single 0.7941）；family-cluster bootstrap 95% CI = [0.1551, 0.2028]，p(Δ ≤ 0) = 0.0；\n"
        "- 配对 McNemar：q_y vs y_only b=165 / c=2，p = 1.50e-46；q_y vs q_only b=258 / c=2，p = 3.66e-74；q_y vs wrong_q_y b=216 / c=5，p = 2.55e-57；\n"
        "- **敏感性 Gate 全部通过**：q_y > y_only ✅ / Δ ≥ 0.05 ✅ / CI 下界 > 0 ✅ / q_y > wrong_q_y ✅；\n"
        "- LLM 视图在 clean subset 上保持相同模式：Qwen q_y 0.8071 > wrong_q_y 0.6386；DeepSeek q_y 0.8335 > wrong_q_y 0.6977；\n"
        "- **结论**：剔除近重复 y 后主结论不变（联合视图仍显著优于任意单视图，错误配对负控制仍成立），模板记忆风险不改变 E1-B 的因果结论，无需按 y-template family 重建 split。\n\n"
    )
    t = sub(t, sec8, "\n" + block74 + sec8.lstrip("\n"), "G 7.4")

    # H. 8.4 exploratory wording
    old = "- **联合机制迁移成立**：q_y 的 Macro-F1（0.675 vs 0.610）与 AUPRC（0.397 vs 0.298；相对 prevalence 0.96% 的 AUPRC lift ≈ 28–71×）均优于 y_only，说明 B 中学到的 q+y 联合判断在独立自然低基率分布上仍然有效；"
    new = "- **方向性/探索性支持**：在仅 6 个独立阳性的小样本下，q_y 的 Macro-F1（0.675 vs 0.610）与 AUPRC（0.397 vs 0.298；相对 prevalence 0.96% 的 AUPRC lift ≈ 28–71×）均优于 y_only，为 B 中学到的 q+y 联合判断在独立自然低基率分布上的迁移提供方向性支持（不作为 confirmatory 结论）；"
    t = sub(t, old, new, "H 8.4")

    # I. 9 cost
    old = "- 累计 API 成本：**¥86.96**（Qwen ¥54.43 / DeepSeek ¥32.53）；"
    new = "- 累计 API 成本：**¥92.78**（Qwen ¥57.67 / DeepSeek ¥35.11）；其中 v4.5 wrong_q_y 重投票新增 2,384 次调用、+¥5.82；"
    t = sub(t, old, new, "I1")

    old = "- 统计检验、E1-C 回放、报告生成全部离线完成，**无新增 API 成本**。"
    new = "- **统计修正（v4.6）**：cluster bootstrap 改为按 Macro-F1（正类 F1 与负类 F1 均值）重抽样，与点估计口径一致；修正后 Δ_joint 点估计 0.1717、95% CI [0.1511, 0.1928]（原实现仅计算正类 F1，CI 与点估计不可比）；仅影响 CI 数值，不涉及数据与模型；\n- 统计检验、E1-C 回放、clean-anchor 敏感性分析、报告生成全部离线完成，**无新增 API 成本**（GPU 推理复用服务器）。"
    t = sub(t, old, new, "I2")

    # J. 10 file list
    old = "| `logs/m1_shard0.out.log` / `logs/m1_shard1.out.log` | 服务器 GPU 训练日志（每任务 loss 轨迹 + Anchor 指标 JSON） |"
    new = ("| `logs/m1_shard0.out.log` / `logs/m1_shard1.out.log` | 服务器 GPU 训练日志（每任务 loss 轨迹 + Anchor 指标 JSON） |\n"
           "| `E1_V4_CLEAN_ANCHOR_SENSITIVITY.json` | clean-anchor 敏感性分析（剔除 212 行近重复 y 后重算，v4.5 新增） |\n"
           "| `E1_V4_WRONG_Q_MAP_V1_ARCHIVE.jsonl` | v1 wrong-q map 备份（仅按语言匹配，v4.5 已被替换） |")
    t = sub(t, old, new, "J 10")

    # K. Appendix A amendments
    old = "In the same window a load-time bug was fixed: NeuralJointDetector now receives q_cap/y_cap at load time in the anchor-local and c-replay phases so eval input construction matches training (q+y anchor MF1 restored from ~0.54 to 0.97); no data, labels or hyper-parameters were changed."
    new = (old
           + "\n- **amendment_v45_wrong_q_category_control**：v4.5 correction amendment (registered before any post-M1 statistical recomputation): the pre-registered wrong-q control promised \"same split, same language/category, different family\" but the v1 implementation matched language only; the wrong-q map was rebuilt to match same language AND same fraud category (categories resolved for all 1200 anchor rows via the A7500 canonical-case registry), different merged family, with documented fallbacks; verified 1200/1200 same-language + same-category pairs, 0 fallbacks. M1 wrong_q_y predictions were regenerated offline with the frozen q_y models (server GPU); Qwen/DeepSeek wrong_q_y votes were re-collected (2,384 calls, +¥5.82). Registered as an implementation-mismatch correction, not an outcome-driven control adjustment.\n"
           + "- **amendment_v46_bootstrap_macro_f1**：v4.6 statistical fix (offline, no data/model change): the family-cluster bootstrap previously drew positive-class F1 only; corrected to Macro-F1 (mean of positive- and negative-class F1), matching the binary_metrics point estimate. Updated 95% CIs: M1 Δ [0.1511, 0.1928], Qwen [0.0396, 0.0737], DeepSeek [0.0605, 0.0969]; clean-anchor Δ [0.1551, 0.2028]. Conclusions unchanged.")
    t = sub(t, old, new, "K appendix A")

    REPORT.write_text(t, encoding="utf-8-sig")
    print("report updated")

    # ---------------- tables ----------------
    tb = TABLES.read_text(encoding="utf-8-sig")
    old = "| wrong_q_y | 0.7609 ± 0.0022 | 0.7315 | 0.6473 | 0.8377 | 0.7281 | 0.3130 |"
    new = "| wrong_q_y | 0.7574 ± 0.0039 | 0.7327 | 0.6632 | 0.8200 | 0.7300 | 0.3033 |"
    tb = sub(tb, old, new, "tables wrong row")
    old = "Δ_joint = +0.1717 (family-cluster bootstrap 95% CI [0.1213, 0.1498], p_below_zero = 0.0); exact McNemar q_y vs y_only p = 1.10e-55; Holm-adjusted p < 0.05; q_y beats best-single 5/5 seeds."
    new = "Δ_joint = +0.1717 (family-cluster bootstrap 95% CI, Macro-F1 formulation [0.1511, 0.1928], p_below_zero = 0.0); exact McNemar q_y vs y_only p = 1.10e-55; Holm-adjusted p < 0.05; q_y beats best-single 5/5 seeds. Wrong-q control (v4.5): same language + same fraud category + different family (1200/1200); wrong_q_y MF1 0.7574."
    tb = sub(tb, old, new, "tables delta line")
    tables_block = (
        "## Table E1-B-M1-clean: clean-anchor sensitivity (exclude 212 near-dup-y rows, n=988, 5 seeds mean ± sd)\n\n"
        "| View | Macro-F1 | AUROC | AUPRC | Recall | FPR |\n|---|---|---|---|---|---|\n"
        "| q_only | 0.6713 ± 0.0196 | 0.7303 | 0.6919 | 0.9122 | 0.5492 |\n"
        "| y_only | 0.8011 ± 0.0045 | 0.9196 | 0.9336 | 0.8414 | 0.2376 |\n"
        "| q_y | 0.9700 ± 0.0031 | 0.9943 | 0.9940 | 0.9849 | 0.0468 |\n"
        "| wrong_q_y | 0.7509 ± 0.0029 | 0.7166 | 0.6800 | 0.8249 | 0.3256 |\n\n"
        "Clean Δ_joint = +0.1784 (bootstrap 95% CI [0.1551, 0.2028], p_below_zero = 0.0); McNemar q_y vs y_only b=165/c=2 p=1.50e-46; all sensitivity gates pass.\n"
    )
    tb = sub(tb, "## Table E1-C: Natural distribution transfer", tables_block + "\n## Table E1-C: Natural distribution transfer", "tables clean insert")
    TABLES.write_text(tb, encoding="utf-8-sig")
    print("tables updated")

    # ---------------- progress ----------------
    pr = PROGRESS.read_text(encoding="utf-8-sig")
    old = "| E1-B M1 XLM-R | ✅ 完成 | 15/15（GPU）；q_y MF1 0.9685 ± 0.0035 / AUROC 0.9944；Δ=+0.1717，Gate 全过，5/5 seeds |"
    new = "| E1-B M1 XLM-R | ✅ 完成 | 15/15（GPU）；q_y MF1 0.9685 ± 0.0035 / AUROC 0.9944；Δ=+0.1717，CI [0.1511, 0.1928]（v4.6 Macro-F1 修正），Gate 全过，5/5 seeds |"
    pr = sub(pr, old, new, "progress m1")
    old = "| 统计检验 | ✅ 完成 | bootstrap CI [0.1213, 0.1498]；McNemar / Holm 全部显著 |"
    new = "| 统计检验 | ✅ 完成 | bootstrap CI [0.1511, 0.1928]（v4.6 修正）；McNemar / Holm 全部显著；clean-anchor 敏感性 Gate 全过（Δ=0.1784，CI [0.1551, 0.2028]） |"
    pr = sub(pr, old, new, "progress stats")
    old = "| wrong_q_y | 0.7609 ± 0.0022 | 0.7315 | 0.6473 | 0.8377 | 0.7281 | 0.3130 |"
    new = "| wrong_q_y | 0.7574 ± 0.0039 | 0.7327 | 0.6632 | 0.8200 | 0.7300 | 0.3033 |"
    pr = sub(pr, old, new, "progress wrong row")
    old = "- 本机 CPU 训练已停止；旧 CPU 模型与 GPU 暂存文件已归档。"
    new = "- 本机 CPU 训练已停止；旧 CPU 模型与 GPU 暂存文件已归档；wrong-q 控制已按同语言+同类别重建（v4.5，1200/1200），bootstrap 已修正为 Macro-F1（v4.6）。"
    pr = sub(pr, old, new, "progress note")
    PROGRESS.write_text(pr, encoding="utf-8-sig")
    print("progress updated")


if __name__ == "__main__":
    main()
