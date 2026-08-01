from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_reports(report_dir: Path, payload: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    decision = payload.get("decision", {})
    budget = payload.get("budget", {})
    executive = f"""# E1 Final Push 执行报告

## 首屏结论

- 最终 decision code：`{decision.get('decision', 'UNKNOWN')}`
- E1-A：`{decision.get('a_gate', 'NA')}`；E1-B：`{decision.get('b_gate', 'NA')}`；E1-C：`{decision.get('c_gate', 'NA')}`
- 本轮新增目标回答数：{payload.get('new_target_responses', 0)}
- 新增 Gold/evaluator 调用数：{payload.get('new_judge_or_eval_calls', 0)}
- Qwen 费用：{budget.get('qwen_cny', 0):.4f} 元；DeepSeek 费用：{budget.get('deepseek_cny', 0):.4f} 元
- 是否达到 q+y ≥0.90：`{decision.get('qy_ge_090', False)}`
- 是否允许进入实验2：`{decision.get('experiment_1_frozen', False)}`

## 不能主张的结论

{chr(10).join('- ' + x for x in payload.get('cannot_claim', []))}
"""
    full = executive + "\n## 完整分析\n\n" + payload.get("analysis", "") + "\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n```\n"
    files = {
        "E1_FP_EXECUTIVE_REPORT_CN.md": executive,
        "E1_FP_FULL_ANALYSIS_REPORT_CN.md": full,
        "E1_FP_PAPER_TABLES.md": "# E1 Final Push 论文表格\n\n" + json.dumps(payload.get("tables", {}), ensure_ascii=False, indent=2, default=str) + "\n",
        "E1_FP_STATISTICAL_APPENDIX_CN.md": "# E1 Final Push 统计附录\n\n" + json.dumps(payload.get("stats", {}), ensure_ascii=False, indent=2, default=str) + "\n",
        "E1_FP_FAILURE_BIAS_AUDIT_CN.md": "# E1 Final Push 失败与偏差审计\n\n" + json.dumps(payload.get("bias", {}), ensure_ascii=False, indent=2, default=str) + "\n",
        "E1_FP_BUDGET_REPORT.md": "# E1 Final Push 预算报告\n\n" + json.dumps(budget, ensure_ascii=False, indent=2, default=str) + "\n",
        "E1_FP_REPRODUCTION_GUIDE.md": "# E1 Final Push 复现指南\n\n```powershell\npython scripts/run_e1_final_push.py --phase p0-code-audit --dry-run\npython scripts/run_e1_final_push.py --phase p0-reuse-audit --cache-only\npython scripts/run_e1_final_push.py --phase p1-build-q-pool --cache-only\npython scripts/run_e1_final_push.py --phase p1-pilot-decision --cache-only\npython scripts/run_e1_final_push.py --phase p6-report --cache-only\n```\n",
        "E1_FP_TASK_CLOSEOUT_CN.md": "# E1 Final Push 任务收尾\n\n" + json.dumps(payload.get("closeout", payload), ensure_ascii=False, indent=2, default=str) + "\n",
    }
    for name, text in files.items():
        (report_dir / name).write_text(text, encoding="utf-8")
