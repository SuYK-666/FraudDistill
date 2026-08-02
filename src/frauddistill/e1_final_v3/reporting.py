from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def table(rows: list[dict[str, Any]]) -> str:
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


def block(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n```"


def write_reports(report_dir: Path, payload: dict[str, Any]) -> list[str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = payload["decision"]
    summary = "\n".join(
        [
            "# E1 FINAL TRIAD v3 任务总报告",
            "",
            "## 总体结论",
            f"- 最终决策：`{decision['decision_code']}`",
            f"- P0：`{decision['p0_gate']}`；E1-A：`{decision['a_gate']}`；E1-B：`{decision['b_gate']}`；E1-C：`{decision['c_gate']}`",
            f"- 用户请求并发：`{payload['budget']['effective_concurrency']['user_requested_total']}`；实际协议并发：Qwen `{payload['budget']['effective_concurrency']['qwen']}`，DeepSeek `{payload['budget']['effective_concurrency']['deepseek']}`",
            "",
            "## 关键分析",
            payload["analysis"],
            "",
            "## E1-A 配额审计",
            table(payload["a"]["quota_table"]),
            "",
            "## E1-B 容量审计",
            table(payload["b"]["quota_table"]),
            "",
            "## E1-C 准入",
            table(payload["c"]["gate_table"]),
        ]
    )
    files = {
        "E1_V3_TASK_OVERVIEW_CN.md": summary + "\n",
        "E1_V3_FULL_REPORT_CN.md": summary + "\n\n## 机器可读完整结果\n" + block(payload) + "\n",
        "E1_V3_DATA_AUDIT.md": "# E1 v3 数据与来源审计\n\n" + block(payload["data_audit"]) + "\n",
        "E1_V3_BUDGET_REPORT.md": "# E1 v3 预算报告\n\n" + block(payload["budget"]) + "\n",
        "E1_V3_REPRODUCTION_GUIDE.md": "# E1 v3 复现指南\n\n```powershell\npython scripts/run_e1_a7500.py --phase all --confirm-budget\npython scripts/run_e1_b3200.py --phase all --confirm-budget --auto-continue-on-pass --consume-anchor\npython scripts/run_e1_c_real_prevalence.py --phase all\n```\n",
    }
    paths = []
    for name, text in files.items():
        path = report_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))
    return paths
