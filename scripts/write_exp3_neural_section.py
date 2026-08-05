# -*- coding: utf-8 -*-
"""Generate the neural-student section (16) markdown for the Exp3 report from
canonical eval metrics (guide 3.8: report reads one canonical metrics file)."""
from __future__ import annotations
import json, sys
from pathlib import Path

BASE = Path(r"experiments\exp3_agent_distillation_ablation\outputs\neural_student")
def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None

def m(name):
    return load(BASE / f"eval_{name}" / "neural_student_metrics.json")

gold, soft, full, gold10, zero = m("gold"), m("soft"), m("full"), m("lowlabel_gold10"), m("zero_shot")

def f1(m): return f"{m['macro_f1']:.4f}"
def row(name, m, note):
    return (f"| {name} | {note} | {f1(m)} | {m['recall']:.4f} | {m['fpr']:.4f} | "
            f"{m['auprc']:.4f} | {m['mcc']:.4f} | {m['acc']:.4f} | {m['4class_macro_f1']:.4f} |")

lines = []
lines.append("### 16.3 主表（指南 §27.2；test n=1,262，0.5 阈值）")
lines.append("| Model | 训练数据 | Macro-F1 | Recall | FPR | AUPRC | MCC | Acc | 4类F1 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
if zero: lines.append(row("Neural-ZeroShot（随机分类头下界）", zero, "—"))
if gold: lines.append(row("Neural-Gold", gold, "2,235（基础清单）"))
if soft: lines.append(row("Neural-SoftDistill", soft, "2,235（基础清单）"))
if full: lines.append(row("Neural-FullDistill", full, "6,235（含 4,000 扩展）"))
lines.append("")
if gold and soft:
    lines.append("### 16.4 蒸馏机制（指南 §27.3，test）")
    lines.append("| 设置 | Soft | Pair | 扩展数据 | ΔMacro-F1 (vs Gold) | ΔRecall | ΔFPR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    d_m = soft["macro_f1"] - gold["macro_f1"]; d_r = soft["recall"] - gold["recall"]; d_f = soft["fpr"] - gold["fpr"]
    lines.append(f"| Neural-SoftDistill | ✓ | — | — | **+{d_m:.4f}** | {d_r:+.4f} | **{d_f:+.4f}** |")
    if full:
        d_m = full["macro_f1"] - gold["macro_f1"]; d_r = full["recall"] - gold["recall"]; d_f = full["fpr"] - gold["fpr"]
        lines.append(f"| Neural-FullDistill | ✓ | ✓ | ✓（+4,000） | {d_m:+.4f} | {d_r:+.4f} | {d_f:+.4f} |")
    lines.append("")
lines.append("### 16.5 低标注曲线（指南 §22.3/§27.4；test，0.5 阈值）")
lines.append("| Gold fraction | Neural-Gold | SoftDistill | 说明 |")
lines.append("|---|---:|---:|---|")
lines.append(f"| 10% | {gold10['macro_f1']:.4f}（R {gold10['recall']:.4f} / FPR {gold10['fpr']:.4f}） | 未完成（soft10 中断于 ~1 epoch，resume.pt 已保留） | 后续工作 |" if gold10 else "| 10% | — | — | 未完成 |")
if gold:
    lines.append(f"| 100% | {gold['macro_f1']:.4f} | {soft['macro_f1']:.4f} | SoftDistill +{soft['macro_f1']-gold['macro_f1']:.4f} |")
lines.append("")
lines.append("### 16.6 机制切片（指南 §23.3，test）")
for name, m in [("Neural-Gold", gold), ("Neural-SoftDistill", soft), ("Neural-FullDistill", full), ("Neural-ZeroShot", zero)]:
    if not m: continue
    s = m["slices"]
    lines.append(f"- **{name}**：direct_recall {s.get('direct_recall')}（n={s.get('direct_n')}）、trust_recall {s.get('trust_recall')}（n={s.get('trust_n')}）、leakage_recall {s.get('leakage_recall')}（n={s.get('leakage_n')}）、clean_refusal_fpr {s.get('clean_refusal_fpr')}（n={s.get('clean_refusal_n')}）、hard_safe_fpr {s.get('hard_safe_fpr')}（n={s.get('hard_safe_n')}）、over_refusal_recall {s.get('over_refusal_recall')}（n={s.get('over_refusal_n')}）、context_flip_pair_acc {s.get('context_flip_pair_acc')}（pairs={s.get('context_flip_pairs')}）")
lines.append("")
lines.append("### 16.7 泛化切片（指南 §23.4，test）")
for name, m in [("Neural-Gold", gold), ("Neural-SoftDistill", soft), ("Neural-FullDistill", full)]:
    if not m: continue
    g = m["generalization"]
    parts = []
    for k in ("real_only", "synthetic_only", "zh", "en", "fraudr1_all_source", "e1_context_r2_source"):
        if k in g:
            parts.append(f"{k} n={g[k]['n']} MF1={g[k]['macro_f1']}")
    lines.append(f"- **{name}**：" + "；".join(parts))
lines.append("")
lines.append("### 16.8 部署（指南 §21.4/§23.5，CPU fp32，max_length=384）")
for name, m in [("Neural-Gold", gold), ("Neural-SoftDistill", soft), ("Neural-FullDistill", full)]:
    if not m: continue
    d = m["deployment"]
    lines.append(f"- **{name}**：模型磁盘 {d['model_disk_mb']} MB（LoRA+头，基座另存）、batch=8 P50 {d['batch1_latency_p50_ms']} ms/条、P95 {d['batch1_latency_p95_ms']} ms/条、吞吐 {d['throughput_samples_per_s']} 条/s（CPU）")
lines.append("")
out_path = Path(r"experiments\exp3_agent_distillation_ablation") / "_section16_generated.md"
out_path.write_text("\n".join(lines), encoding="utf-8")
print("written", out_path)
