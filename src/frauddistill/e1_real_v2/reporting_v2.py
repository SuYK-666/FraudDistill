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


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v).replace("\n", " ")


def block(v: Any) -> str:
    return "```json\n" + json.dumps(v, ensure_ascii=False, indent=2, default=str) + "\n```"


def write_reports(report_dir: Path, payload: dict[str, Any]) -> list[str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = payload.get("decision", {})
    executive = "\n".join(
        [
            "# E1 REAL TRIAD v2 总报告",
            "",
            "## 结论摘要",
            f"- 最终决策：`{decision.get('decision_code')}`",
            f"- A Gate：`{decision.get('a_gate')}`；B Gate：`{decision.get('b_gate')}`；C Gate：`{decision.get('c_gate')}`；Gold Gate：`{decision.get('gold_gate')}`",
            f"- B 真实目标回答比例：`{payload.get('provenance', {}).get('real_target_response_ratio', 0):.4f}`",
            f"- API 费用：`{payload.get('budget', {}).get('total_cny', 0):.4f}` 元",
            "",
            "## 核心分析",
            payload.get("analysis", ""),
            "",
            "## A 层自然发生率",
            md_table(payload.get("a", {}).get("main_table", [])),
            "",
            "## B 层容量与主表",
            md_table(payload.get("b", {}).get("main_table", [])),
            "",
            "## C 层迁移",
            md_table(payload.get("c", {}).get("main_table", [])),
        ]
    )
    full = executive + "\n\n## 完整机器可读结果\n" + block(payload)
    files = {
        "E1_REAL_V2_EXECUTIVE_REPORT_CN.md": executive + "\n",
        "E1_REAL_V2_FULL_ANALYSIS_REPORT_CN.md": full + "\n",
        "E1_REAL_V2_PAPER_TABLES.md": "# E1 REAL v2 论文表格\n\n## A\n"
        + md_table(payload.get("a", {}).get("main_table", []))
        + "\n\n## B\n"
        + md_table(payload.get("b", {}).get("main_table", []))
        + "\n\n## C\n"
        + md_table(payload.get("c", {}).get("main_table", []))
        + "\n",
        "E1_REAL_V2_STATISTICAL_APPENDIX_CN.md": "# E1 REAL v2 统计附录\n\n"
        + block(payload.get("statistics", {}))
        + "\n",
        "E1_REAL_V2_DATA_PROVENANCE_AUDIT.md": "# E1 REAL v2 数据来源审计\n\n"
        + block(payload.get("provenance", {}))
        + "\n",
        "E1_REAL_V2_GOLD_QUALITY_REPORT.md": "# E1 REAL v2 Gold 质量报告\n\n"
        + block(payload.get("gold", {}))
        + "\n",
        "E1_REAL_V2_FAILURE_BIAS_AUDIT_CN.md": "# E1 REAL v2 失败与偏差审计\n\n"
        + block(payload.get("bias", {}))
        + "\n",
        "E1_REAL_V2_BUDGET_REPORT.md": "# E1 REAL v2 预算报告\n\n"
        + block(payload.get("budget", {}))
        + "\n",
        "E1_REAL_V2_REPRODUCTION_GUIDE.md": "# E1 REAL v2 复现指南\n\n```powershell\npython scripts/run_e1_real_triad_v2.py --phase all --confirm-budget --auto-continue-on-pass --consume-anchor\n```\n",
        "E1_REAL_V2_TASK_CLOSEOUT_CN.md": "# E1 REAL v2 任务收尾\n\n"
        + block(payload.get("closeout", {}))
        + "\n",
        "E1_REAL_V2_TASK_OVERVIEW_CN.md": "# E1 REAL v2 本轮任务总览\n\n"
        + "## 执行范围\n"
        + "- 已完成历史报告与输出归档。\n"
        + "- 已新增 REAL TRIAD v2 配置、真实 registry 适配、Gold v5 schema 校验、CPU PairLite v2、wrong-q v2、报告生成器和单元测试。\n"
        + "- 已执行 P0 协议锁定、真实回答 registry 合并、B 层容量预筛和中文报告生成。\n\n"
        + "## 本轮结论\n"
        + f"- 决策：`{payload.get('decision', {}).get('decision_code')}`。\n"
        + "- 当前真实回答容量不足以构成 1200 条 formal Gold v5 case-control 面板，因此未进入正式训练、C 层迁移和 API 消耗阶段。\n\n"
        + "## 关键数据\n"
        + md_table(payload.get("b", {}).get("main_table", []))
        + "\n\n"
        + "## 后续建议\n"
        + "- 若继续推进，应优先补采 context stable positive、context critical positive 和 context hard negative 的真实目标模型回答，并完成 Gold v5 双评审/裁决。\n"
        + "- 在容量门控通过前，不建议再生成论文主表式 STRONG PASS 结果。\n",
    }
    for name, text in files.items():
        (report_dir / name).write_text(text, encoding="utf-8")
    return [str(report_dir / name) for name in files]
