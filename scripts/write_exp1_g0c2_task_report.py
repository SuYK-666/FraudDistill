from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def z(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


def main() -> None:
    run_dir = ROOT / "outputs" / "exp1_ccfa_cpu_v5" / "g0c2"
    data_dir = ROOT / "data" / "prepared" / "exp1_cpu_v5" / "g0c2"
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    data_audit = read_json(data_dir / "G0c2_DATA_AUDIT.json")
    decision = read_json(run_dir / "G0c2_DECISION.json")
    selected = read_json(run_dir / "G0c2_SELECTED_MODELS.json")
    paired = read_json(run_dir / "G0c2_PAIRED_STATS.json")
    resource = read_json(run_dir / "G0c2_RESOURCE_GATE.json")
    modeldev = read_csv(run_dir / "G0c2_MODELDEV_METRICS_BY_SEED.csv")
    p1 = read_csv(run_dir / "G0c2_P1_METRICS_BY_SEED.csv")
    p2 = read_csv(run_dir / "G0c2_P2_METRICS_BY_SEED.csv")
    lines: list[str] = []
    lines.extend(
        [
            "# E1_CPU_v5_G0c2 \u6574\u4f53\u4efb\u52a1\u62a5\u544a",
            "",
            "## \u6267\u884c\u7ed3\u8bba",
            f"- \u6700\u7ec8\u51b3\u7b56\uff1a{decision['decision']}",
            f"- Base data Gate\uff1a{decision['base_data_gate']}",
            f"- Model-dev Gate\uff1a{decision['modeldev_gate']}",
            f"- P1 Gate\uff1a{decision['p1_gate']}",
            f"- P2 data Gate\uff1a{decision['p2_data_gate']}",
            f"- P2 model Gate\uff1a{decision['p2_model_gate']}",
            f"- Resource Gate\uff1a{decision['resource_gate']}",
            "",
            "\u672c\u8f6e\u5df2\u5b8c\u6210 G0c2 \u8981\u6c42\u7684\u4ee3\u7801\u91cd\u6784\u3001\u6570\u636e\u6784\u5efa\u3001model-dev \u4e09\u79cd\u5b50\u8bc4\u4f30\u548c P1 \u4e09\u79cd\u5b50\u8bc4\u4f30\u3002P2-DVM \u5728\u542f\u7528 Do-Not-Answer fallback \u540e\u4ecd\u672a\u8fbe\u5230 300 groups\uff0c\u56e0\u6b64 P2 \u6a21\u578b\u8bc4\u4f30\u6309\u9884\u6ce8\u518c\u72b6\u6001\u673a\u4e0d\u8fd0\u884c\u3002\u4e00\u6b21 ONE_MODEL_FIX \u5df2\u7528\u4e8e S1 relation weight/C grid\uff0c\u4fee\u590d\u540e model-dev \u4ecd\u672a\u8fbe\u5230\u5173\u7cfb\u589e\u76ca Gate\uff0c\u6240\u4ee5\u5f53\u524d\u7ed3\u8bba\u5e94\u6536\u7a84\u4e3a `STOP_OR_NARROW_CLAIM`\u3002",
            "",
            "## \u5931\u8d25\u9879",
        ]
    )
    lines.extend([f"- {item}" for item in decision.get("failed_checks", [])] or ["- \u65e0"])
    lines.extend(["", "## Table A\uff1a\u6570\u636e\u6f0f\u6597\u4e0e\u6837\u672c\u7edf\u8ba1", ""])
    lines.append(markdown_counts(data_audit["counts"]))
    lines.extend(["", "\u6570\u636e Gate \u89e3\u8bfb\uff1aBase data Gate \u5df2\u901a\u8fc7\uff0ctrain/model-dev/threshold-dev/P1/P3 \u7684\u884c\u6570\u3001\u6807\u7b7e\u5e73\u8861\u3001\u6765\u6e90\u4e0a\u9650\u3001\u8de8 split component leakage\u3001\u8fd1\u91cd\u590d\u548c label-token leakage \u5747\u5408\u683c\u3002P1 \u4e3a\u6ee1\u8db3 source cap \u548c 600/600 \u5e73\u8861\uff0c\u5728\u4f18\u5148 fraud-core \u540e\u4f7f\u7528 fraud-adjacent/general safety \u8865\u8db3\uff0c\u56e0\u6b64\u62a5\u544a\u4e2d\u660e\u786e\u4fdd\u7559 fraud-core \u884c\u6570\uff1asafe=304\uff0cunsafe=430\u3002"])
    lines.extend(["", "## Table B\uff1aModel-dev \u4e09\u79cd\u5b50", ""])
    lines.append(markdown_rows(top_modeldev(modeldev), ["comparator", "seed", "macro_f1", "recall", "fpr", "threshold_feasible"]))
    lines.extend(["", f"\u51bb\u7ed3\u9009\u62e9\uff1aS1={selected['best_s1_qy']}\uff0cS0={selected['best_s0_qy']}\uff0cbest single={selected['best_single']}\u3002S1 q+y Macro-F1 \u8fbe\u5230 0.846\uff0cunsafe recall=0.852\uff0cFPR=0.160\uff0c\u4f46\u76f8\u5bf9 best single \u548c S0 q+y \u7684\u589e\u76ca\u90fd\u8fdc\u4f4e\u4e8e +0.02/+0.015 \u95e8\u69db\uff0c\u56e0\u6b64 model-dev Gate \u5931\u8d25\u3002"])
    lines.extend(["", "## Table C\uff1aP1/P2 \u4e3b\u7ed3\u679c", ""])
    lines.append(markdown_rows(p1, ["panel", "comparator", "seed", "macro_f1", "recall", "fpr"]))
    lines.extend(["", "P1 paired stats:", "```json", json.dumps(paired.get("P1", {}), ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "\u5206\u6790\uff1aP1 \u4e0a S1 q+y Macro-F1=0.762\uff0c\u7565\u9ad8\u4e8e best single\uff08+0.0188\uff09\u548c S0\uff08+0.0025\uff09\uff0c\u4f46\u672a\u8fbe P1 PASS \u95e8\u69db\uff1aMacro-F1 <0.78\uff0c\u5bf9 best single \u7684 bootstrap 95% CI \u4e0b\u754c\u4e3a -0.0061\uff0cHolm \u6821\u6b63\u540e\u4e0d\u663e\u8457\u3002\u8fd9\u8bf4\u660e\u5f53\u524d R3 \u5728\u81ea\u7136\u9762\u677f\u4e0a\u6709\u5c0f\u5e45\u6b63\u5411\u8d8b\u52bf\uff0c\u4f46\u8fd8\u4e0d\u80fd\u652f\u6491 CCF-A \u4e3b\u5f20\u3002"])
    if p2:
        lines.append(markdown_rows(p2, ["panel", "comparator", "seed", "macro_f1", "recall", "fpr"]))
    else:
        lines.append("P2-DVM \u56e0 data Gate FAIL \u672a\u8fd0\u884c\u6a21\u578b\u8bc4\u4f30\u3002")
    lines.extend(["", "## Table D\uff1aP2-DVM balance \u4e0e caliper", "```json", json.dumps(data_audit["p2_dvm_audit"], ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "\u5206\u6790\uff1aP2-DVM \u7684 Q/Y selector SMD \u5206\u522b\u4e3a 0.0024/0.0429\uff0cindependent q/y probe AUC \u4e3a 0.512/0.520\uff0c\u8bf4\u660e\u5355\u89c6\u56fe\u53ef\u9884\u6d4b\u6027\u63a7\u5236\u662f\u6709\u6548\u7684\u3002\u4f46 E \u7ea7 caliper \u4e5f\u53ea\u80fd\u5f62\u6210 149 groups\uff0c\u4e14 PKU-SafeRLHF \u5360\u6bd4 84.6%\uff0clog answer length SMD=0.171 \u8d85\u8fc7\u95e8\u69db\u3002\u8fd9\u4e0d\u662f\u6a21\u578b\u5931\u8d25\uff0c\u800c\u662f\u5f53\u524d\u516c\u5f00\u5019\u9009\u6c60\u5728\u4e25\u683c\u53cc\u5355\u89c6\u56fe\u5339\u914d\u4e0b\u4e0d\u8db3\u4ee5\u652f\u6491 300 groups\u3002"])
    lines.extend(["", "## Table E\uff1a\u8d44\u6e90", "```json", json.dumps(resource, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## \u7ed3\u8bba\u4e0e\u540e\u7eed\u5efa\u8bae", "\u5efa\u8bae\u6682\u65f6\u4e0d\u542f\u52a8 24k bridge\u3002\u5f53\u524d\u6700\u7a33\u59a5\u7684\u8bba\u6587\u53d9\u4e8b\u662f\uff1apair-aware/R3 \u5728 P1 \u4e0a\u6709\u6709\u9650\u6b63\u5411\u8d8b\u52bf\uff0c\u4f46\u5c1a\u672a\u8fbe\u5230\u7edf\u8ba1\u663e\u8457\u548c\u9884\u8bbe effect size\uff1b\u4e25\u683c\u7684 P2-DVM challenge set \u5728\u5f53\u524d\u516c\u5f00\u6570\u636e\u4e0b\u4ecd\u4e0d\u8db3 300 groups\uff0c\u5e94\u4f5c\u4e3a\u900f\u660e\u6784\u5efa\u8bca\u65ad\u548c\u9644\u5f55\u7ed3\u679c\uff0c\u800c\u4e0d\u5e94\u901a\u8fc7\u6539\u6d4b\u8bd5\u96c6\u6216\u964d\u4f4e Gate \u5f3a\u884c\u5236\u9020 0.9\u3002"])
    output = report_dir / z("E1_CPU_v5_G0c2_\\u6574\\u4f53\\u4efb\\u52a1\\u62a5\\u544a_\\u4e2d\\u6587.md")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def top_modeldev(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: float(row.get("macro_f1") or 0.0), reverse=True)[:12]


def markdown_counts(counts: dict) -> str:
    rows = []
    for split, item in counts.items():
        rows.append(
            {
                "split": split,
                "rows": item["rows"],
                "components": item["components"],
                "safe": item["by_label"].get("safe", 0),
                "unsafe": item["by_label"].get("unsafe", 0),
                "sources": json.dumps(item["by_source"], ensure_ascii=False),
                "fraud_core_safe": item.get("fraud_core_safe", 0),
                "fraud_core_unsafe": item.get("fraud_core_unsafe", 0),
            }
        )
    return markdown_rows(rows, ["split", "rows", "components", "safe", "unsafe", "sources", "fraud_core_safe", "fraud_core_unsafe"])


def markdown_rows(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
