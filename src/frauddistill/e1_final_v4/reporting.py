# -*- coding: utf-8 -*-
"""v4 report generation (CN executive report + paper tables + appendix)."""
from __future__ import annotations

import collections
import json
from typing import Any

from frauddistill.e1_final_v3.io import read_json, read_jsonl, write_json
from frauddistill.e1_final_v4.judge_views import view_label


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _row(rid: str) -> dict[str, Any]:
    return {"response_id": rid}


def build_report(cfg, out_dir) -> dict[str, Any]:
    a = read_json(out_dir / "E1_V4_A_RECONCILIATION.json", {})
    audit = read_json(out_dir / "E1_V4_PANEL_AUDIT.json", {})
    split = read_json(out_dir / "E1_V4_SPLIT_AUDIT.json", {})
    stats = read_json(out_dir / "E1_V4_STATS.json", {})
    train = read_json(out_dir / "E1_V4_TRAIN_RESULTS.json", {})
    c_res = read_json(out_dir / "E1_V4_C_RESULT.json", {})

    # ---------- tables
    lines: list[str] = []
    lines.append("# E1-FINAL-TRIAD-v4 执行报告（CN）\n")
    lines.append("> 协议：E1-FINAL-TRIAD-v4-Relational-Ablation · 生成时间：见 git commit\n")
    lines.append("## 1. E1-A 自然低基率发生率（A7500 冻结）\n")
    lines.append(f"- registry 行数：{a.get('registry_rows')}；unique response_id：{a.get('unique_response_ids')}；duplicate：{a.get('duplicate_response_ids')}")
    lines.append(f"- central positive：**{a.get('central_positives')} / {a.get('registry_rows')}**（{a.get('central_rate')}）；Wilson 95% CI：{a.get('wilson_ci95')}")
    lines.append(f"- cluster bootstrap（case-level）：{a.get('cluster_bootstrap')}")
    lines.append(f"- 双 Judge agreement：{a.get('judge_agreement')}；Gold 完成率：{a.get('gold_completion')}")
    lines.append(f"- 旧 11 vs 28 reconciliation：{json.dumps(a.get('reconciliation', {}), ensure_ascii=False)}")
    if a.get("strata"):
        for key, groups in a["strata"].items():
            lines.append(f"- 分层 {key}：{json.dumps(groups, ensure_ascii=False)}")
    lines.append("\n## 2. E1-B 面板（6000 行）\n")
    lines.append(f"- 总行数：{audit.get('n_rows')}；label：{audit.get('by_label')}；语言：{audit.get('by_language')}")
    lines.append(f"- stratum：{audit.get('by_stratum')}")
    lines.append(f"- provenance：{audit.get('by_provenance')}")
    lines.append(f"- B1 保留 discordant pairs：{audit.get('b1_kept')}；B2：{audit.get('b2_kept')}；B3 pos/neg：{audit.get('b3_pos')}/{audit.get('b3_neg')}")
    lines.append("\n## 3. Split 与反快捷方式审计\n")
    lines.append(f"- split 行数：{split.get('n')}；label：{split.get('label_counts')}")
    lines.append(f"- 跨 split family：{split.get('cross_split_families')}；exact (q,y) 泄漏：{split.get('cross_split_exact_qy')}；near-dup y：{split.get('cross_split_near_dup_y')}")
    lines.append(f"- shortcut AUC：{split.get('shortcut_auc')}；Gate：{split.get('gate')}")
    lines.append("\n## 4. 主消融结果（Frozen Anchor）\n")
    rows = []
    for key in ["m1_local", "m2_m3_llm"]:
        if key not in stats:
            continue
        src = stats[key]
        if key == "m1_local":
            for view in ["q_only", "y_only", "q_y", "wrong_q_y"]:
                d = src.get(view, {})
                mean = d.get("mean", {})
                rows.append({"model": "M1 XLM-R", "view": view, "macro_f1": mean.get("macro_f1"), "auprc": mean.get("auprc"), "recall": mean.get("recall"), "fpr": mean.get("fpr"), "precision": mean.get("precision")})
        else:
            prov = "Qwen" if "qwen" in key else "DeepSeek"
            for view in ["q_only", "y_only", "q_y", "wrong_q_y"]:
                d = src.get(view, {})
                rows.append({"model": prov, "view": view, "macro_f1": d.get("macro_f1"), "auprc": d.get("auprc"), "recall": d.get("recall"), "fpr": d.get("fpr"), "precision": d.get("precision")})
    lines.append("| Model | View | Macro-F1 | AUPRC | Recall | Precision | FPR |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['model']} | {r['view']} | {_fmt(r['macro_f1'])} | {_fmt(r['auprc'])} | {_fmt(r['recall'])} | {_fmt(r['precision'])} | {_fmt(r['fpr'])} |")
    lines.append("\n### Δ_joint 与统计检验\n")
    for key in ["stats_m1", "stats_llm_qwen", "stats_llm_deepseek"]:
        if key not in stats:
            continue
        s = stats[key]
        lines.append(f"- **{key}**：Δ_joint = {_fmt(s['delta_joint']['delta'])}；bootstrap 95% CI {[round(x,4) for x in s['bootstrap']['ci95']]}；Holm p：{s['holm']['p_adj']}；scientific gate：{s['scientific_gate']}")
        lines.append(f"  - McNemar：{json.dumps(s['mcnemar'], ensure_ascii=False, default=str)}")
    lines.append("\n## 5. E1-C 独立自然回放（A7500）\n")
    agg = c_res.get("aggregate", {})
    for mode, d in agg.items():
        lines.append(f"- {mode}：{json.dumps(d, ensure_ascii=False)}")
    lines.append("\n## 6. 成本\n")
    ledger = read_jsonl(rel_cfg(cfg) / "E1_V4_BUDGET_LEDGER.jsonl")
    total = sum(float(r.get("cost_cny", 0) or 0) for r in ledger)
    by_prov = collections.Counter()
    for r in ledger:
        by_prov[r.get("provider", "?")] += float(r.get("cost_cny", 0) or 0)
    lines.append(f"- 总成本：¥{total:.2f}；by provider：{dict(by_prov)}")
    report = "\n".join(lines) + "\n"

    # ---------- paper tables
    pt: list[str] = []
    pt.append("# E1 Paper Tables (v4)\n")
    pt.append("## Table E1-A: Natural fraud-assistance prevalence\n")
    if a.get("strata"):
        pt.append("| Stratum | N | Positive | Rate |")
        pt.append("|---|---|---|---|")
        for gname, g in a["strata"].get("target_provider", {}).items():
            pt.append(f"| {gname} | {g['n']} | {g['positive']} | {g['rate']} |")
    pt.append("\n## Table E1-B: Input-boundary ablation (Frozen Anchor 1200)\n")
    pt.append("| Model | View | Macro-F1 | AUPRC | Recall | Precision | FPR |")
    pt.append("|---|---|---|---|---|---|---|")
    for r in rows:
        pt.append(f"| {r['model']} | {r['view']} | {_fmt(r['macro_f1'])} | {_fmt(r['auprc'])} | {_fmt(r['recall'])} | {_fmt(r['precision'])} | {_fmt(r['fpr'])} |")
    pt.append("\n## Table E1-C: Natural distribution transfer\n")
    pt.append("| Mode | Macro-F1 | Recall | FPR | AUROC | AUPRC | R@FPR1% | R@FPR5% | P@10 |")
    pt.append("|---|---|---|---|---|---|---|---|---|")
    for mode, d in agg.items():
        pt.append(f"| {mode} | {_fmt(d.get('macro_f1', {}).get('mean', 0))} | {_fmt(d.get('recall', {}).get('mean', 0))} | {_fmt(d.get('fpr', {}).get('mean', 0))} | {_fmt(d.get('auroc', {}).get('mean', 0))} | {_fmt(d.get('auprc', {}).get('mean', 0))} | {_fmt(d.get('recall_at_fpr_1pct', {}).get('mean', 0))} | {_fmt(d.get('recall_at_fpr_5pct', {}).get('mean', 0))} | {_fmt(d.get('precision_at_10', {}).get('mean', 0))} |")
    papers = "\n".join(pt) + "\n"

    (out_dir / "E1_V4_REPORT_CN.md").write_text(report, encoding="utf-8")
    (out_dir / "E1_V4_PAPER_TABLES.md").write_text(papers, encoding="utf-8")
    return {"report": report, "tables": papers}


def rel_cfg(cfg) -> Any:
    from pathlib import Path
    p = Path(cfg["data"]["output_dir"])
    return p if p.is_absolute() else Path(__file__).resolve().parents[3] / p
