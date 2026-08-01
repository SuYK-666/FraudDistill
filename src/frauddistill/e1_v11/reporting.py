from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def metric_table(metrics: list[dict[str, Any]]) -> str:
    lines = ["| Mode | n | Macro-F1 | BalAcc | AUPRC | Recall | FPR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row.get('mode')} | {row.get('n', 0)} | {row.get('macro_f1', 0):.4f} | {row.get('balanced_accuracy', 0):.4f} | {row.get('auprc', 0):.4f} | {row.get('recall', 0):.4f} | {row.get('fpr', 0):.4f} |")
    return "\n".join(lines)


def write_report_set(report_dir: Path, payload: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = payload.get("decision", {})
    a = payload.get("a", {})
    b = payload.get("b", {})
    c = payload.get("c", {})
    budget = payload.get("budget", {})
    executive = f"""# E1 V11 Executive Report

## 首屏结论

- 最终 decision code：`{decision.get('decision', 'UNKNOWN')}`
- A：`{decision.get('a_gate', 'NA')}`；B1：`{decision.get('b1_gate', 'NA')}`；B2：`{decision.get('b2_gate', 'NA')}`；B3：`{decision.get('b3_gate', 'NA')}`；C：`{decision.get('c_gate', 'NA')}`
- 新增预算：Qwen {budget.get('qwen_cny', 0):.4f} 元；DeepSeek {budget.get('deepseek_cny', 0):.4f} 元；总计 {budget.get('total_cny', 0):.4f} 元。
- 论文口径：A 自然行为冻结；B1 为风险富集 case-control；B2/B3 为机制辅助；C 为低基率压力 holdout。

## E1-A NATURAL

```json
{json.dumps(a, ensure_ascii=False, indent=2)}
```

## E1-B ENRICHED CASE-CONTROL

```json
{json.dumps(b, ensure_ascii=False, indent=2)}
```

## E1-C PRESSURE HOLDOUT

```json
{json.dumps(c, ensure_ascii=False, indent=2)}
```
"""
    full = executive + "\n## 详细分析\n\n" + payload.get("analysis_text", "本报告由 V11 runner 自动生成，所有机器可读产物保留在 data/prepared/e1_v11_event_pool。") + "\n"
    files = {
        "E1_V11_EXECUTIVE_REPORT_CN.md": executive,
        "E1_V11_FULL_ANALYSIS_REPORT_CN.md": full,
        "E1_V11_TASK_CLOSEOUT_CN.md": "# E1 V11 任务收尾报告\n\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        "E1_V11_FAILURE_BIAS_AUDIT_CN.md": "# E1 V11 失败与偏差审计\n\n" + json.dumps(payload.get("bias", {}), ensure_ascii=False, indent=2) + "\n",
        "E1_V11_STATISTICAL_APPENDIX_CN.md": "# E1 V11 统计附录\n\n" + json.dumps(payload.get("stats", {}), ensure_ascii=False, indent=2) + "\n",
        "E1_V11_BUDGET_REPORT.md": "# E1 V11 预算报告\n\n" + json.dumps(budget, ensure_ascii=False, indent=2) + "\n",
        "E1_V11_REPRODUCTION_GUIDE.md": "# E1 V11 复现指南\n\n```powershell\npython scripts/run_e1_v11_event_pool.py --phase all\n```\n",
        "E1_V11_PAPER_TABLES.md": "# E1 V11 论文表格草稿\n\n" + json.dumps({"a": a, "b": b, "c": c}, ensure_ascii=False, indent=2) + "\n",
    }
    for name, text in files.items():
        (report_dir / name).write_text(text, encoding="utf-8")
