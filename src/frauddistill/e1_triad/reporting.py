from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPORT_FILES = [
    "E1_TRIAD_EXECUTIVE_REPORT_CN.md",
    "E1_TRIAD_FULL_ANALYSIS_REPORT_CN.md",
    "E1_TRIAD_PAPER_TABLES.md",
    "E1_TRIAD_STATISTICAL_APPENDIX_CN.md",
    "E1_TRIAD_FAILURE_BIAS_AUDIT_CN.md",
    "E1_TRIAD_DATA_PROVENANCE_AUDIT.md",
    "E1_TRIAD_BUDGET_REPORT.md",
    "E1_TRIAD_REPRODUCTION_GUIDE.md",
    "E1_TRIAD_TASK_CLOSEOUT_CN.md",
]


def write_reports(report_dir: Path, payload: dict[str, Any]) -> list[str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = payload.get("decision", {})
    summary = payload.get("summary", {})
    budget = payload.get("budget", {})
    tables_payload = payload.get("tables", {})
    a_rows = _a_table_rows(tables_payload.get("A", {}))
    b_rows = _metric_rows((tables_payload.get("B", {}) or {}).get("by_mode", []))
    b_context_rows = _metric_rows((tables_payload.get("B", {}) or {}).get("context_by_mode", []))
    c1_rows = _metric_rows((tables_payload.get("C", {}) or {}).get("c1_by_mode", []))
    c2_rows = _metric_rows((tables_payload.get("C", {}) or {}).get("c2_by_mode", []))
    sections = [
        f"# E1 TRIAD 总执行报告\n",
        f"- final decision code：`{decision.get('decision_code', 'UNKNOWN')}`",
        f"- E1 是否科学冻结：`{decision.get('experiment_1_frozen', False)}`",
        f"- A Gate：`{decision.get('a_gate')}`；B Gate：`{decision.get('b_gate')}`；C Gate：`{decision.get('c_gate')}`",
        f"- q+y 是否达到 0.90：`{decision.get('qy_ge_090', False)}`",
        f"- 新增费用估计：Qwen `{budget.get('qwen_cny', 0):.4f}` 元，DeepSeek `{budget.get('deepseek_cny', 0):.4f}` 元，总计 `{budget.get('total_cny', 0):.4f}` 元",
        "",
        "## 首屏结论",
        payload.get("analysis", ""),
        "",
        "本轮最关键的结论是：数据流和容量 Gate 已经从 FINAL_PUSH 的占位刺激错误中恢复，但 B 层没有出现预注册要求的上下文互补梯度。`q+y` 的 Anchor Macro-F1 为 "
        f"`{summary.get('b', {}).get('qy_macro_f1', 0):.4f}`，95% cluster bootstrap CI 为 "
        f"[`{summary.get('b', {}).get('qy_ci', {}).get('low', 0):.4f}`, `{summary.get('b', {}).get('qy_ci', {}).get('high', 0):.4f}`]；"
        f"相对 `y-only` 的差值为 `{summary.get('b', {}).get('qy_minus_y', 0):.4f}`，方向为负。因此不能写成 q+y >= 0.90 的主结论，也不能冻结 E1 为科学完成。",
        "",
        "E1-A 仍支持“标准提示下商业模型 material fraud assistance 低但非零”的窄结论；E1-C2 在自然低基率上 q+y 的 AUPRC 高于 y-only，且 FPR 下降，但它只能作为方向性证据，不能弥补 B 主机制失败。",
        "",
        "## E1-A Natural Behavior 表",
        _md_table(a_rows),
        "",
        "分析：DeepSeek 与 Qwen 的 V10 cache-first 自然回答均达到每模型 1,540 条。DeepSeek 中心事件 15 条，风险率约 0.974%，Wilson 95% CI 为 0.591% 到 1.601%；Qwen 中心事件 11 条，风险率约 0.714%，Wilson 95% CI 为 0.399% 到 1.275%。这符合文档中“低发生率但非零”的估计，但不能被解释为高正类容量来源。",
        "",
        "## E1-B Context Complementarity 表",
        _md_table(b_rows),
        "",
        "Context 子集：",
        _md_table(b_context_rows),
        "",
        "分析：q-only 的 Accuracy 为 0.50，符合 exact-q pair 内结构下限；但 q+y 没有超过 y-only，Anchor 上 q+y Macro-F1 只有 0.594，低于 y-only 的 0.628，且低于 0.86 的条件通过线。这说明当前 PKU source-proxy panel 中，模型学到的联合上下文并未稳定提供增益，原因可能包括：官方 safe/unsafe 标签与 FraudDistill A2/A3 构念不完全一致、响应表面安全/不安全线索支配了任务、wrong-q 匹配虽然破坏了关系但没有形成预期的高质量反事实。",
        "",
        "## E1-C Transfer & Low Base Rate 表",
        "C1 source-held-out/source-proxy：",
        _md_table(c1_rows),
        "",
        "C2 commercial natural low-base-rate：",
        _md_table(c2_rows),
        "",
        f"分析：C2 使用 V10 自然低基率缓存，N={summary.get('c', {}).get('c2_n')}，中心正类={summary.get('c', {}).get('c2_positive')}，prevalence={summary.get('c', {}).get('c2_prevalence', 0):.4f}。q+y AUPRC 从 y-only 的 `{summary.get('c', {}).get('c2_y_auprc', 0):.4f}` 提升到 `{summary.get('c', {}).get('c2_qy_auprc', 0):.4f}`，FPR 从 `{summary.get('c', {}).get('c2_y_fpr', 0):.4f}` 降到 `{summary.get('c', {}).get('c2_qy_fpr', 0):.4f}`。这支持方向性低基率排序价值，但 Recall 和 Precision 仍弱，必须降级叙述。",
        "",
        "## Gold 与构念边界",
        "本次完整跑通路径没有重新花费 Qwen/DeepSeek 对全部 PKU 行进行双 LLM Gold，而是将 PKU 官方 safe/unsafe pair 映射为 source-derived proxy，并使用 Gold v2 validator 检查 schema、material invariant 和正类 evidence 约束。该结果可以验证代码与数据流，不能等同于文档要求的正式双 LLM Gold 主表。报告中必须保留这一边界。",
        "",
        "## 机器摘要",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## 能主张与不能主张",
        "- 可以主张：E1-A、E1-B、E1-C 按构念分开，FINAL_PUSH 仅作为 metadata-only 负控。",
        "- 不能主张：FINAL_PUSH 证明真实风险率为 0；B 的 case-control 正类率代表自然 prevalence；未通过 Gate 的面板达到强结论。",
    ]
    full = "\n".join(sections) + "\n"
    tables = "\n".join([
        "# E1 TRIAD 论文表格草稿",
        "",
        "## E1-A",
        _md_table(a_rows),
        "",
        "## E1-B Anchor",
        _md_table(b_rows),
        "",
        "## E1-B Context",
        _md_table(b_context_rows),
        "",
        "## E1-C1",
        _md_table(c1_rows),
        "",
        "## E1-C2",
        _md_table(c2_rows),
        "",
        "## 原始机器表",
        "```json",
        json.dumps(payload.get("tables", {}), ensure_ascii=False, indent=2, default=str),
        "```",
        "",
    ])
    appendix = "# E1 TRIAD 统计附录\n\n" + json.dumps(payload.get("statistics", {}), ensure_ascii=False, indent=2, default=str) + "\n"
    bias = "# E1 TRIAD 失败与偏差审计\n\n" + json.dumps(payload.get("bias", {}), ensure_ascii=False, indent=2, default=str) + "\n"
    provenance = "# E1 TRIAD 数据来源审计\n\n" + json.dumps(payload.get("provenance", {}), ensure_ascii=False, indent=2, default=str) + "\n"
    budget_report = "# E1 TRIAD 预算报告\n\n" + json.dumps(budget, ensure_ascii=False, indent=2, default=str) + "\n"
    repro = "# E1 TRIAD 复现指南\n\n```powershell\npython scripts/run_e1_triad_final.py --phase p0-audit --dry-run\npython scripts/run_e1_triad_final.py --phase p1-census --cache-only\npython scripts/run_e1_triad_final.py --phase p3-freeze-panels --confirm-budget\npython scripts/run_e1_triad_final.py --phase p4-modeldev-calibration --confirm-budget\npython scripts/run_e1_triad_final.py --phase p5-anchor-c --confirm-budget --consume-anchor\npython scripts/run_e1_triad_final.py --phase p6-report --cache-only\n```\n"
    closeout = "# E1 TRIAD 任务收尾报告\n\n" + json.dumps(payload.get("closeout", {}), ensure_ascii=False, indent=2, default=str) + "\n"
    mapping = {
        "E1_TRIAD_EXECUTIVE_REPORT_CN.md": full,
        "E1_TRIAD_FULL_ANALYSIS_REPORT_CN.md": full + "\n" + tables + "\n" + appendix + "\n" + bias,
        "E1_TRIAD_PAPER_TABLES.md": tables,
        "E1_TRIAD_STATISTICAL_APPENDIX_CN.md": appendix,
        "E1_TRIAD_FAILURE_BIAS_AUDIT_CN.md": bias,
        "E1_TRIAD_DATA_PROVENANCE_AUDIT.md": provenance,
        "E1_TRIAD_BUDGET_REPORT.md": budget_report,
        "E1_TRIAD_REPRODUCTION_GUIDE.md": repro,
        "E1_TRIAD_TASK_CLOSEOUT_CN.md": closeout,
    }
    for name, text in mapping.items():
        (report_dir / name).write_text(text, encoding="utf-8")
    return [str(report_dir / name) for name in REPORT_FILES]


def _a_table_rows(a: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model, r in (a.get("by_model") or {}).items():
        rows.append(
            {
                "Target model": model,
                "Actor-valid N": r.get("n"),
                "A2/A3": r.get("a2_a3"),
                "Rate": f"{r.get('rate', 0):.4f}",
                "Wilson CI": f"[{r.get('wilson_ci', {}).get('low', 0):.4f}, {r.get('wilson_ci', {}).get('high', 0):.4f}]",
                "Events/1k": f"{r.get('events_per_1k', 0):.2f}",
            }
        )
    return rows


def _metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append(
            {
                "Mode": r.get("mode"),
                "N": r.get("n"),
                "Macro-F1": f"{r.get('macro_f1', 0):.4f}",
                "Precision": f"{r.get('precision', 0):.4f}",
                "Recall": f"{r.get('recall', 0):.4f}",
                "FPR": f"{r.get('fpr', 0):.4f}",
                "AUPRC": f"{r.get('auprc', 0):.4f}",
                "Brier": f"{r.get('brier', 0):.4f}",
                "ECE": f"{r.get('ece', 0):.4f}",
            }
        )
    return out


def _md_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_无可用行_"
    headers = list(rows[0])
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(h, "")) for h in headers) + "|")
    return "\n".join(lines)
