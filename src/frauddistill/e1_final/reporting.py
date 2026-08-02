from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def md_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_无数据_"
    cols = list(rows[0])
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(c, "")) for c in cols) + "|")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ")


def block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n```"


def write_reports(report_dir: Path, payload: dict[str, Any]) -> list[str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = payload.get("decision", {})
    budget = payload.get("budget", {})
    executive = "\n".join(
        [
            "# E1 FINAL TRIAD 总报告",
            "",
            "## 结论摘要",
            f"- 最终决策：`{decision.get('decision_code')}`",
            f"- A Gate：`{decision.get('a_gate')}`；B Gate：`{decision.get('b_gate')}`；C Gate：`{decision.get('c_gate')}`；Gold Gate：`{decision.get('gold_gate')}`",
            f"- q+y Anchor Macro-F1：`{decision.get('b_qy_macro_f1', 0):.4f}`",
            f"- q+y - y-only：`{decision.get('b_delta_y', 0):.4f}`；q+y - wrong-q+y：`{decision.get('b_delta_wrong', 0):.4f}`",
            f"- C 层 q+y AUPRC：`{decision.get('c_qy_auprc', 0):.4f}`；y-only AUPRC：`{decision.get('c_y_auprc', 0):.4f}`",
            f"- API 估算费用：Qwen `{budget.get('qwen_cny', 0):.4f}` 元，DeepSeek `{budget.get('deepseek_cny', 0):.4f}` 元，总计 `{budget.get('total_cny', 0):.4f}` 元",
            "",
            "## 核心分析",
            payload.get("analysis", ""),
            "",
            "## A 自然行为测量",
            md_table(payload.get("a", {}).get("main_table", [])),
            "",
            "## B 上下文互补机制",
            md_table(payload.get("b", {}).get("main_table", [])),
            "",
            "## C 自然低基率迁移",
            md_table(payload.get("c", {}).get("main_table", [])),
        ]
    ) + "\n"
    full = executive + "\n## 完整机器可读结果\n" + block(payload) + "\n"
    files = {
        "E1_FINAL_EXECUTIVE_REPORT_CN.md": executive,
        "E1_FINAL_FULL_ANALYSIS_REPORT_CN.md": full,
        "E1_FINAL_PAPER_TABLES.md": "# E1 FINAL 论文表格\n\n## A\n" + md_table(payload.get("a", {}).get("main_table", [])) + "\n\n## B\n" + md_table(payload.get("b", {}).get("main_table", [])) + "\n\n## C\n" + md_table(payload.get("c", {}).get("main_table", [])) + "\n",
        "E1_FINAL_STATISTICAL_APPENDIX_CN.md": "# E1 FINAL 统计附录\n\n" + block(payload.get("statistics", {})) + "\n",
        "E1_FINAL_DATA_PROVENANCE_AUDIT.md": "# E1 FINAL 数据来源审计\n\n" + block(payload.get("provenance", {})) + "\n",
        "E1_FINAL_GOLD_QUALITY_REPORT.md": "# E1 FINAL Gold 质量报告\n\n" + block(payload.get("gold", {})) + "\n",
        "E1_FINAL_FAILURE_BIAS_AUDIT_CN.md": "# E1 FINAL 失败与偏差审计\n\n" + block(payload.get("bias", {})) + "\n",
        "E1_FINAL_BUDGET_REPORT.md": "# E1 FINAL 预算报告\n\n" + block(budget) + "\n",
        "E1_FINAL_REPRODUCTION_GUIDE.md": "# E1 FINAL 复现指南\n\n```powershell\npython scripts/run_e1_final_triad.py --phase p0\npython scripts/run_e1_final_triad.py --phase registry --cache-only\npython scripts/run_e1_final_triad.py --phase build-candidates --cache-only\npython scripts/run_e1_final_triad.py --phase budget-plan --dry-run\npython scripts/run_e1_final_triad.py --phase gold-and-freeze --confirm-budget\npython scripts/run_e1_final_triad.py --phase train-calibrate\npython scripts/run_e1_final_triad.py --phase anchor --consume-anchor\npython scripts/run_e1_final_triad.py --phase c-transfer\npython scripts/run_e1_final_triad.py --phase report\n```\n",
        "E1_FINAL_TASK_CLOSEOUT_CN.md": "# E1 FINAL 任务收尾\n\n" + block(payload.get("closeout", {})) + "\n",
    }
    for name, text in files.items():
        (report_dir / name).write_text(text, encoding="utf-8")
    return [str(report_dir / name) for name in files]
