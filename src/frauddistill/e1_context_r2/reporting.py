from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def md_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_无数据_"
    cols = list(rows[0])
    out = ["|" + "|".join(cols) + "|", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        out.append("|" + "|".join(str(row.get(c, "")) for c in cols) + "|")
    return "\n".join(out)


def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n```"


def write_reports(report_dir: Path, payload: dict[str, Any]) -> list[str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = payload.get("decision", {})
    a = payload.get("a", {})
    b = payload.get("b", {})
    c = payload.get("c", {})
    gold = payload.get("gold", {})
    budget = payload.get("budget", {})
    provenance = payload.get("provenance", {})
    statistics = payload.get("statistics", {})
    closeout = payload.get("closeout", {})

    executive = "\n".join(
        [
            "# E1 Context-Recovery R2 总报告",
            "",
            "## 结论摘要",
            f"- 决策代码：`{decision.get('decision_code')}`",
            f"- A Gate：`{decision.get('a_gate')}`；B Gate：`{decision.get('b_gate')}`；C Gate：`{decision.get('c_gate')}`；Gold Gate：`{decision.get('gold_gate')}`",
            f"- 实验 1 是否冻结：`{decision.get('experiment_1_frozen')}`",
            f"- q+y 是否达到 0.90：`{decision.get('qy_ge_090')}`",
            f"- 本轮 API 估算费用：Qwen `{budget.get('qwen_cny', 0):.4f}` 元，DeepSeek `{budget.get('deepseek_cny', 0):.4f}` 元，总计 `{budget.get('total_cny', 0):.4f}` 元",
            "",
            "## 核心分析",
            payload.get("analysis", ""),
            "",
            "本轮按 R2 文档把正式 B 层从 PKU proxy 和 FINAL_PUSH 补丁数据中剥离，改为只审计历史真实 Qwen/DeepSeek q-y 缓存。结果显示，Gold 质量修复后已经达标，但历史缓存 pilot 的上下文正例容量不足：context_critical_positive 为 0，context_stable_positive 为 0，无法进入正式 B 层训练和 Anchor 消耗。该 STOP 是数据容量层面的严格准入失败，不是运行中断。",
            "",
            "## A 自然发生层",
            md_table(a.get("table", [])),
            "",
            "## B 机制面板",
            md_table(b.get("table", [])),
            "",
            "## C 完整自然低基率层",
            md_table(c.get("table", [])),
            "",
            "## Gold 质量与容量",
            json_block(gold),
            "",
            "## 预算",
            json_block(budget),
        ]
    ) + "\n"

    full = "\n".join(
        [
            executive,
            "## 来源审计",
            json_block(provenance),
            "",
            "## 统计与准入记录",
            json_block(statistics),
            "",
            "## 偏差与边界说明",
            json_block(payload.get("bias", {})),
            "",
            "## 收尾信息",
            json_block(closeout),
        ]
    )

    files = {
        "E1_R2_EXECUTIVE_REPORT_CN.md": executive,
        "E1_R2_FULL_ANALYSIS_REPORT_CN.md": full,
        "E1_R2_PAPER_TABLES.md": "\n".join(
            [
                "# E1 R2 论文表格",
                "",
                "## A 自然发生层",
                md_table(a.get("table", [])),
                "",
                "## B 机制面板",
                md_table(b.get("table", [])),
                "",
                "## C 完整自然低基率层",
                md_table(c.get("table", [])),
                "",
                "## Gold pilot 分层",
                json_block(gold.get("strata", {})),
            ]
        )
        + "\n",
        "E1_R2_STATISTICAL_APPENDIX_CN.md": "# E1 R2 统计附录\n\n" + json_block(statistics) + "\n",
        "E1_R2_GOLD_QUALITY_REPORT.md": "# E1 R2 Gold 质量报告\n\n" + json_block(gold) + "\n",
        "E1_R2_DATA_PROVENANCE_AUDIT.md": "# E1 R2 数据来源审计\n\n" + json_block(provenance) + "\n",
        "E1_R2_FAILURE_BIAS_AUDIT_CN.md": "# E1 R2 失败与偏差审计\n\n" + json_block(payload.get("bias", {})) + "\n",
        "E1_R2_BUDGET_REPORT.md": "# E1 R2 预算报告\n\n" + json_block(budget) + "\n",
        "E1_R2_REPRODUCTION_GUIDE.md": (
            "# E1 R2 复现指南\n\n"
            "```powershell\n"
            "python scripts/run_e1_context_recovery_r2.py --phase p0-audit --dry-run\n"
            "python scripts/run_e1_context_recovery_r2.py --phase p1-census\n"
            "python scripts/run_e1_context_recovery_r2.py --phase p2-freeze-pilot-manifest\n"
            "python scripts/run_e1_context_recovery_r2.py --phase p3-gold-pilot --confirm-budget\n"
            "python scripts/run_e1_context_recovery_r2.py --phase p4-pilot-gate\n"
            "python scripts/run_e1_context_recovery_r2.py --phase p8-report\n"
            "```\n"
        ),
        "E1_R2_TASK_CLOSEOUT_CN.md": "# E1 R2 任务收尾\n\n" + json_block(closeout) + "\n",
    }
    for name, text in files.items():
        (report_dir / name).write_text(text, encoding="utf-8")
    return [str(report_dir / name) for name in files]
