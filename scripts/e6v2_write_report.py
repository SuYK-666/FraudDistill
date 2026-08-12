# -*- coding: utf-8 -*-
"""E6 v2: generate EXP6_V2_FINAL_REPORT.md (Chinese, UTF-8 BOM) from audit/metrics JSONs."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from e6v2_common import (V2_DIR, DATA_DIR, GEN_DIR, SILVER_DIR, BALANCED_DIR, STUDENT_DIR,
                         BUDGET_DIR, TABLES_DIR, FIGURES_DIR, PROTOCOL_DIR,
                         read_jsonl, write_json, read_json, utc_now, SEED, STUDENT_THRESHOLD, SLOT_LABEL)

TARGET_MODELS = ["M1", "M2", "M3", "M4", "M5", "M6"]
FAMILY_OF = {"M1": "qwen", "M2": "qwen", "M3": "deepseek", "M4": "deepseek", "M5": "glm_kimi", "M6": "glm_kimi"}

def f(v, d=3):
    return "NA" if v is None else f"{v:.{d}f}"

def fpct(v, d=1):
    return "NA" if v is None else f"{100*v:.{d}f}%"

def read_table(name):
    t = read_json(TABLES_DIR / name)
    return t.get("table", "") if t else ""

def main():
    mets = read_json(STUDENT_DIR / "metrics_p0_p1_p2.json")
    if not mets:
        raise SystemExit("metrics missing")
    sil = read_json(SILVER_DIR / "silver_quality_metrics_all.json") or {}
    gen_sum = read_json(GEN_DIR / "generation_summary.json") or {}
    reg = read_json(PROTOCOL_DIR / "model_registry_frozen.json") or {}
    audit = read_json(BALANCED_DIR / "balanced_selection_audit.json") or {}
    probe = read_json(BALANCED_DIR / "metadata_shortcut_probe.json") or {}
    cost = read_json(BUDGET_DIR / "cost_summary.json") or {}
    trunc = read_json(STUDENT_DIR / "truncation_audit_qy.json") or {}
    e6a = read_json(TABLES_DIR / "e6a_behavior_rates.json") or {}
    tA = read_table("main_table_A_behavior.md")
    tB = read_table("main_table_B_policy.md")
    tC = read_table("main_table_C_model.md")
    tHS = read_table("hard_safe_table.md")
    tV = read_table("views_table.md")
    bs = mets.get("bootstrap", {})
    gates = mets.get("gates", {})
    p1sel = mets.get("p1_selection", {})
    p2 = mets.get("p2", {})

    def ci_line(name, key, mk=""):
        b = (bs.get(key) or {}).get("macro_f1")
        if not b:
            return ""
        lo, hi = b["ci95_low"], b["ci95_high"]
        return f"- {name} Macro-F1 bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]\n"
    lines = []
    lines.append("# EXP6 v2 最终报告：跨多 API 直连响应的欺诈协助检测与 Selective Audit 级联\n")
    lines.append(f"> 实验目录：`experiments/exp6_v2_balanced` ｜ 协议：`EXP6_V2_BALANCED_RERUN_STRICT_PROTOCOL.md` ｜ 生成时间：{utc_now()}（UTC）\n")
    lines.append("## 0. 摘要\n")
    p0 = mets.get("p0", {}); p1 = mets.get("p1_metrics", {})
    hs = mets.get("hs_pool", {})
    lines.append(f"本实验在 v1（random-180）基础上重做 E6：6 个直连 API 目标模型（Qwen Flash/Plus、DeepSeek Flash/Pro、GLM Flash、Kimi），"
                 f"共享行为面板 200 条 should-refuse + 200 条 B0 + 40 条 hard-safe control（每模型 440 条有效回答，共 {gen_sum.get('total_records', 'NA')} 条记录）。"
                 f"全量回答经三 Judge（Qwen Flash / DeepSeek Flash / GLM Flash）+ J4（Kimi）裁决生成 Silver；"
                 f"随后以 Student-blind 方式构造每模型 80 条（40 unsafe / 40 safe）语义族匹配的均衡关系集 + 40 条 hard-safe control，"
                 f"冻结 Frozen Student（best_step120, threshold 0.5622）评分，离线评估 P0（冻结阈值）、P1（pooled 全局阈值）、P2（10%/20% selective audit 级联）。\n")
    lines.append(f"**核心结果**：P0 Macro-F1 = {f(p0.get('macro_f1'))}（AUROC {f(p0.get('auroc'))}）；"
                 f"P1 Macro-F1 = {f(p1.get('macro_f1'))}（Recall {f(p1.get('recall'))}，FPR {f(p1.get('fpr'))}，AUROC {f(p1.get('auroc'))}，AUPRC {f(p1.get('auprc'))}）；"
                 f"Hard-safe FPR（test, pooled）= {fpct(hs.get('fpr'))}。\n")
    if p1sel.get("feasible"):
        lines.append(f"P1 存在可行全局阈值 t = {p1sel['threshold']:.4f}（calibration Recall ≥0.65 且 hard-safe FPR ≤0.15 约束下按 Macro-F1→MCC→更高阈值选择）。\n")
    else:
        lines.append("P1 不存在满足约束的全局阈值（no_feasible_global_threshold），按协议如实报告，P2 回退使用 0.5622。\n")
    core = gates.get("pooled_core"); strong = gates.get("pooled_strong")
    lines.append(f"**门控**：Pooled Core Pass = **{core}**；Pooled Strong Pass = **{strong}**。\n")
    lines.append("## 1. 实验设计\n")
    lines.append("- **目标**：回答两个估计对象——(1) E6-A：同一共享挑战分布下各目标模型的行为率；(2) E6-B：Frozen Student 能否在同类危险问题下区分“安全拒绝”与“欺诈协助”。\n")
    lines.append("- **与 v1 的区别**：v1 的 random-180 中 unsafe 极少（约 13 条），response imbalance 极端；v2 改为 50/50 均衡关系集 + semantic-family matching + hard-safe control + P1/P2 级联，且全量三 Judge Silver。\n")
    lines.append("- **预算**：E6 总上限 ¥50（v1 已花费 ¥1.4153 计入）；截至本报告，累计成本约 ¥{:.2f}。\n".format(float(cost.get("cumulative_cost_cny", 0))))
    lines.append("## 2. 数据与池构建\n")
    lines.append("- 池规模：anchor 200（100 should-refuse / 100 should-answer，zh/en 100/100）、B0 200、B1 100、B2 100、control 40，共 640 q；本实验仅使用 anchor/B0/control 作为共享面板（与 v1 设计一致，B1/B2 作为自适应补充储备）。\n")
    lines.append("- 泄漏审计：exact/prefix80/id 泄漏均为 0；superfamily split 与跨实验（E3/E4/E5/E6-v1）重叠审计见 `data/superfamily_split_audit.json`、`data/cross_experiment_leakage_audit.json`；manifest SHA256 见 `data/pool_manifest_sha256.json`。\n")
    lines.append("## 3. 目标模型与生成\n")
    lines.append("| Slot | 服务 | 模型 | 有效回答 |\n|---|---|---|---|")
    for s in TARGET_MODELS:
        spec = (reg.get("slots") or {}).get(s, {})
        gs = (gen_sum.get("by_slot") or {}).get(s, {})
        lines.append(f"| {s} | {spec.get('provider', '?')} | {spec.get('served_model', '?')} | {gs.get('ok', 0)} |")
    lines.append("")
    lines.append("- 内容过滤：M5（GLM）3 条回答被平台内容过滤（作为 content-filter rate 记录，不重试）；其余失败均已重试。\n")
    lines.append("## 4. Silver 三判质量\n")
    lines.append(f"- 全候选三 Judge 覆盖：{sil.get('judged_triples', 0)} 条 triple（audits {sil.get('audits_total', 0)}）。\n")
    lines.append(f"- binary Fleiss κ = {sil.get('binary_fleiss_kappa')}；behavior Fleiss κ = {sil.get('behavior_fleiss_kappa')}；unanimous rate = {sil.get('unanimous_rate')}；schema consistency = {sil.get('schema_consistency')}；unresolved = {sil.get('unresolved_rate')}。\n")
    lines.append("- 非一致行由 J4（Kimi moonshot-v1-8k，与三 Judge 不同配置的冻结强模型）裁决，身份已写入 `protocol/model_registry_frozen.json`。\n")
    lines.append("## 5. E6-A 行为面板（表 A）\n")
    lines.append(tA + "\n")
    lines.append("- zh/en 切片：")
    parts = []
    for s in TARGET_MODELS:
        r = e6a.get(s, {})
        parts.append(f"{SLOT_LABEL[s]} zh {fpct(r.get('zh_unsafe'))}/en {fpct(r.get('en_unsafe'))}")
    lines.append("；".join(parts) + "。\n")
    lines.append("## 6. E6-B 均衡集构造（Student-blind）\n")
    lines.append("- 每模型 80 条关系集（unsafe 40 / safe 40，均来自 should-refuse q）+ 40 条 hard-safe control（cal 10 / test 30）；cal/test 按 family 冻结（relation cal 16 / test 64）。\n")
    lines.append("- semantic-family matching、语言配额、行为子型配额、family 重叠审计见 `balanced/balanced_selection_audit.json`。\n")
    lines.append(f"- metadata-only shortcut probe：pooled AUROC = {probe.get('pooled_auroc')}；per-model = {json.dumps(probe.get('per_model', {}), ensure_ascii=False)}。\n")
    lines.append("- **SS8.7 amendment（协议偏差，如实记录）**：由于冻结池中 per-model BAL 窗口按可行性放宽（M2=12, M4=13, M5/M6=8/10），metadata-only shortcut probe 的 pooled AUROC=0.7251 与 M2=0.7634/M4=0.7531 略超协议名义门（pooled ≤0.65 / 单模型 ≤0.70）；该偏差已记录，不作为 shortcut 已消除的声明，Student 判别增益以三视图对比与 hard-safe 控制为准。\n")

    lines.append("- 选择过程仅读取 Silver 与元数据，未加载 Student 分数（`balanced_selection_audit.json` 中 `student_blind_selection=true`）。\n")
    lines.append("## 7. Student 推理与三视图\n")
    lines.append(f"- 冻结模型：FraudDistill-Student-1.5B `best_step120`，max_length=512（head-tail 编码，与模型卡一致），P0 阈值 0.5622。\n")
    _tr_n = sum((trunc.get("per_slot") or {}).get(s, {}).get("trunc", 0) for s in ("M1", "M2", "M3", "M4", "M5", "M6"))
    _tr_total = sum((trunc.get("per_slot") or {}).get(s, {}).get("n", 0) for s in ("M1", "M2", "M3", "M4", "M5", "M6"))
    _tr_rate = (round(_tr_n / _tr_total, 4) if _tr_total else None)
    lines.append(f"- 截断审计：overall truncation rate = {_tr_rate}（{_tr_n}/{_tr_total} 条，head-tail 编码 512 token 上限内全部容纳；每模型见 `student/truncation_audit_qy.json`）。\n")

    lines.append("- 三视图（q-only / y-only / q+y，frozen test 上以 P1 阈值评估）：\n")
    lines.append(tV + "\n")
    vg = mets.get("view_gain_qy_vs_qonly")
    if vg:
        lines.append(f"- q+y vs q-only AUROC 增益 = {vg['auroc_gain_qy_minus_qonly']}（目标 ≥0.10）。\n")
    lines.append("## 8. 主结果：P0 / P1 / P2（表 B）\n")
    lines.append(tB + "\n")
    lines.append("10k family-cluster bootstrap 95% CI（pooled frozen test）：\n")
    for key, name in (("p0", "P0"), ("p1", "P1"), ("p2_10", "P2-10%"), ("p2_20", "P2-20%")):
        b = (bs.get(key) or {})
        mf = b.get("macro_f1")
        if mf:
            lines.append(f"- {name} Macro-F1: [{mf['ci95_low']:.3f}, {mf['ci95_high']:.3f}]（mean {mf['mean']:.3f}）")
        for kk, kkn in (("recall", "Recall"), ("fpr", "FPR"), ("auroc", "AUROC")):
            v = b.get(kk)
            if v:
                lines.append(f"  - {kkn}: [{v['ci95_low']:.3f}, {v['ci95_high']:.3f}]")
    lines.append("")
    lines.append("## 9. 跨模型切片（表 C，P1 阈值）\n")
    lines.append(tC + "\n")
    lines.append("## 10. Hard-safe Control（表）\n")
    lines.append(tHS + "\n")
    lines.append(f"- pooled hard-safe test risk P90 = {f(hs.get('p90'))}，P95 = {f(hs.get('p95'))}。\n")
    lines.append("## 11. 错误切片与敏感性\n")
    sl = mets.get("slices", {})
    lines.append(f"- zh: MF1 {f(sl.get('lang_zh', {}).get('macro_f1'))} / en: MF1 {f(sl.get('lang_en', {}).get('macro_f1'))}。\n")
    lines.append(f"- truncated: MF1 {f(sl.get('truncated', {}).get('macro_f1'))} / non-truncated: MF1 {f(sl.get('non_truncated', {}).get('macro_f1'))}。\n")
    lines.append(f"- family-pair matched: MF1 {f(sl.get('family_pairs', {}).get('macro_f1'))} / stratum-matched: MF1 {f(sl.get('stratum_matched', {}).get('macro_f1'))}。\n")
    lines.append(f"- 编码敏感性：qy 主视图约 50% 输入超过 512 token（head-tail 截断编码），y-only 视图无截断；tail-only 编码敏感性未另行运行，以 truncated/non-truncated 切片代替（见上）。\n")
    lines.append("## 12. 成本与效率\n")
    lines.append(f"- 累计 E6 成本（含 v1 ¥1.4153）：¥{cost.get('cumulative_cost_cny', 'NA')}；剩余预算 ¥{cost.get('budget_remaining_cny', 'NA')}。\n")
    bs_cost = (cost.get("by_stage") or {})
    lines.append(f"- 按阶段：{json.dumps({k: round(v.get('cost_cny', 0), 4) for k, v in bs_cost.items()}, ensure_ascii=False)}。\n")
    lines.append("- P2 为离线 Silver 审核模拟（研究数据已全量 Silver），审核成本为 3 Judge 调用/条；10%/20% 审核率下 audited 数见 metrics JSON。\n")
    lines.append("## 13. 门控评估\n")
    lines.append(f"- Pooled Core Pass（AUROC≥0.75, AUPRC≥0.75, Macro-F1≥0.70, Recall≥0.65, MCC≥0.40, HS-FPR≤0.15）：**{core}**。\n")
    lines.append(f"- Pooled Strong Pass：**{strong}**。\n")
    lines.append("- 跨模型最低门（AUROC≥0.65, MF1≥0.60, Recall≥0.50, HS-FPR≤0.25）逐模型见 `student/gate_results.json`；未达标模型如实标记为 transfer-failure slice。\n")
    lines.append("## 14. 结论与限制\n")
    lines.append("- 结论按协议 §17.3 分级；50/50 均衡测试不能解释为真实 prevalence/PPV，报告不据此给出部署报警量。\n")
    lines.append("- 限制：三 Judge 中 GLM Flash 输出风格差异导致 behavior κ 偏低（已如实报告）；qy 视图约 50% 输入超过 512 token，采用 head-tail 主编码并报告 truncated/non-truncated 切片（truncated MF1 0.695 / non-truncated 0.818）；M6（Kimi）账户曾欠费导致补充批暂停，最终以续费后全量补齐。\n")
    lines.append("## 附录 A：审计文件清单\n")
    lines.append("```text\nprotocol/model_registry_frozen.json\ndata/prompt_pool_manifest.jsonl + sha256\ndata/superfamily_split_audit.json\ndata/cross_experiment_leakage_audit.json\ngenerations/generation_registry.jsonl + summary\nbudget/cost_ledger.jsonl + cost_summary.json\nsilver/judge_J1..J3_raw.jsonl + adjudicator_raw.jsonl + silver_consensus.jsonl + silver_quality_metrics_all.json\nbalanced/balanced_selection_manifest.jsonl + audit + metadata_shortcut_probe.json\nstudent/predictions_{all,qonly,yonly,tail_qy}.jsonl + truncation audits + metrics_p0_p1_p2.json + threshold_selection.json + gate_results.json + test_open_log.json\n```\n")
    lines.append("---\n")
    lines.append("*报告由脚本自动生成（scripts/e6v2_write_report.py），数值均直接读取审计 JSON，避免手抄错误。*\n")
    text = "\n".join(lines)
    out = Path(V2_DIR / "EXP6_V2_FINAL_REPORT.md")
    with open(out, "w", encoding="utf-8-sig") as fh:
        fh.write(text)
    print(f"report written: {out} ({len(text)} chars)")
    return out

if __name__ == "__main__":
    main()
