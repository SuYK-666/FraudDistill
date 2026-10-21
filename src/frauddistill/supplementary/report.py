"""Build the two supplementary tables and draft SVG curves from frozen outputs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import yaml

from .common import REPO, binary_label, binary_metrics, read_json, read_jsonl, row_key, write_json

SUPP_DIR = REPO / "experiments" / "lighteval_supplementary"
RESULT_DIR = SUPP_DIR / "results"
SCALE_DIR = RESULT_DIR / "scale"
TABLES_DIR = RESULT_DIR / "tables"
FIGURES_DIR = RESULT_DIR / "figures"
A0_METRICS = RESULT_DIR / "a0" / "a0_metrics.json"
AUDIT_CACHE = SUPP_DIR / "audit" / "v2" / "audit_cache.jsonl"
AUDIT_USAGE_SUMMARY = RESULT_DIR / "ablation_v2" / "audit_usage_summary.json"
STUDENT_TRAINING_STATE = REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student/training_state.json"
STUDENT_PREDICTIONS = REPO / "experiments/exp6_balanced_multi_api/student/predictions_all.jsonl"
STUDENT_DEPLOYMENT_METRICS = REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/eval_gold/neural_student_metrics.json"


def _scale_cfg() -> dict:
    return yaml.safe_load((REPO / "configs/supplementary/lighteval_scale.yaml").read_text(encoding="utf-8"))


def _ablation_cfg() -> dict:
    return yaml.safe_load((REPO / "configs/supplementary/lighteval_ablation.yaml").read_text(encoding="utf-8"))


def _legacy_deepseek_row(model: dict) -> dict:
    base = REPO / "experiments/exp3_agent_distillation_ablation/outputs/neural_student/final_distilled_student"
    core = read_json(base / "test_metrics.json")
    e6 = read_json(REPO / "experiments/exp6_balanced_multi_api/student/metrics_p0_p1_p2.json")
    online = e6["p1_metrics"]
    hs = e6["hs_pool"]
    training = read_json(base / "training_state.json")
    return {
        "id": model["id"], "label": model["label"], "params_b": model["params_b"],
        "source": "legacy_existing_e6",
        "core_macro_f1": core["macro_f1"], "core_recall": core["recall"], "core_fpr": core["fpr"],
        "online_macro_f1": online["macro_f1"], "online_recall": online["recall"],
        "online_fpr": online["fpr"], "hard_safe_fpr": hs["fpr"],
        "checkpoint": str(base / "best_step120"), "threshold": e6["thresholds"]["p1"],
        "training_seconds": training.get("wall_seconds"),
        "peak_vram_gb": None, "latency_p50_ms": None, "latency_p95_ms": None,
        "rows_per_s": None, "gpu_hours": None, "estimated_gpu_cny": None,
        "seconds_per_1000_rows": None, "train_input_tokens": None, "eval_input_tokens": None,
    }


def _row_from_summary(model: dict, summary: dict) -> dict:
    online = summary["metrics"]["relation_test_online"]
    core = summary["metrics"]["core_test"]
    hs = summary["metrics"]["hard_safe_test"]
    serving = summary["cost"]["evaluation"]["serving"]
    return {
        "id": model["id"], "label": model["label"], "params_b": model["params_b"],
        "source": "new_scale_run",
        "core_macro_f1": core["macro_f1"], "core_recall": core["recall"], "core_fpr": core["fpr"],
        "online_macro_f1": online["macro_f1"], "online_recall": online["recall"],
        "online_fpr": online["fpr"], "hard_safe_fpr": hs["fpr"],
        "checkpoint": summary["checkpoint"], "threshold": summary["threshold_selection"]["threshold"],
        "training_seconds": summary["cost"]["training"]["wall_seconds"],
        "peak_vram_gb": serving.get("peak_vram_gb"),
        "latency_p50_ms": serving.get("latency_p50_ms"),
        "latency_p95_ms": serving.get("latency_p95_ms"),
        "rows_per_s": serving.get("rows_per_s"),
        "seconds_per_1000_rows": serving.get("seconds_per_1000_rows"),
        "train_input_tokens": summary["cost"]["training"].get("input_tokens"),
        "eval_input_tokens": summary["cost"]["evaluation"].get("total_input_tokens"),
        "gpu_hours": summary["cost"]["total_gpu_hours"],
        "estimated_gpu_cny": summary["cost"]["estimated_total_gpu_cny"],
    }


def build_scale_rows() -> tuple[list[dict], list[dict]]:
    scale = _scale_cfg()
    v2_path = RESULT_DIR / "ablation_v2" / "ablation_results.json"
    ablation_path = v2_path if v2_path.exists() else RESULT_DIR / "ablation" / "ablation_results.json"
    ablation = read_json(ablation_path) if ablation_path.exists() else {}
    a6 = ablation.get("policies", {}).get("A6", {})
    a6_ready = a6.get("status") == "ready"
    rows = []
    for model in scale["models"]:
        summary_path = SCALE_DIR / model["id"] / "summary.json"
        if summary_path.exists():
            rows.append(_row_from_summary(model, read_json(summary_path)))
        elif model.get("reuse"):
            rows.append(_legacy_deepseek_row(model))
        else:
            rows.append({
                "id": model["id"], "label": model["label"], "params_b": model["params_b"],
                "source": "pending", "core_macro_f1": None, "core_recall": None, "core_fpr": None,
                "online_macro_f1": None, "online_recall": None, "online_fpr": None,
                "hard_safe_fpr": None, "checkpoint": None, "threshold": None,
                "training_seconds": None, "peak_vram_gb": None, "latency_p50_ms": None,
                "latency_p95_ms": None, "rows_per_s": None, "gpu_hours": None,
                "seconds_per_1000_rows": None, "train_input_tokens": None, "eval_input_tokens": None,
                "estimated_gpu_cny": None,
            })
    audit_comparison = []
    if a6_ready:
        base = next(row for row in rows if row["id"] == "deepseek_r1_qwen_1_5b")
        audit_comparison.append({
            **base,
            "id": "deepseek_r1_qwen_1_5b_audit20",
            "label": "Original 1.5B + 20% real E6 audit",
            "online_macro_f1": a6["metrics"]["macro_f1"],
            "online_recall": a6["metrics"]["recall"],
            "online_fpr": a6["metrics"]["fpr"],
            "audit_k": a6["audited"],
            "audit_cost_cny": a6.get("incremental_api_cost_cny"),
        })
    return rows, audit_comparison


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_scale_table(rows: list[dict], audit_rows: list[dict]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = rows + audit_rows
    header = ("| Model | Params (B) | Core MF1 | Online MF1 | Recall | FPR | Hard-safe FPR | "
              "P50/P95 ms | Rows/s | Sec/1k | Eval input tok | Peak VRAM GB | Train h | GPU h | Est. CNY |")
    lines = [header,
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in all_rows:
        train_h = (row["training_seconds"] / 3600) if row.get("training_seconds") else None
        if row["source"] == "pending":
            pending_values = [row["label"], row["params_b"], "PENDING"] + ["-"] * 12
            lines.append("| " + " | ".join(str(value) for value in pending_values) + " |")
            continue
        lat = (f"{_fmt(row.get('latency_p50_ms'), 2)}/{_fmt(row.get('latency_p95_ms'), 2)}"
               if row.get("latency_p50_ms") is not None else "-")
        lines.append(
            f"| {row['label']} | {row['params_b']} | {_fmt(row['core_macro_f1'])} | "
            f"{_fmt(row['online_macro_f1'])} | {_fmt(row['online_recall'])} | "
            f"{_fmt(row['online_fpr'])} | {_fmt(row['hard_safe_fpr'])} | {lat} | "
            f"{_fmt(row.get('rows_per_s'), 2)} | {_fmt(row.get('seconds_per_1000_rows'), 2)} | "
            f"{_fmt(row.get('eval_input_tokens'), 0)} | {_fmt(row.get('peak_vram_gb'), 2)} | "
            f"{_fmt(train_h, 2)} | {_fmt(row.get('gpu_hours'), 3)} | "
            f"{_fmt(row.get('estimated_gpu_cny'), 2)} |"
        )
    lines.extend([
        "",
        "> Train h is end-to-end wall time including model reload, resume and CUDA-retry overhead; endpoint checkpoint saving is excluded. GPU h counts the newly rented GPU training run and its frozen evaluation; the reused DeepSeek student was trained earlier on CPU. Classifiers emit zero output tokens.",
        ">",
        "> Qwen2.5-7B used 3 training attempts and 2 CUDA retries. Qwen3.5-4B used 9 attempts and 8 CUDA retries, with official evaluation batch=2 after a cuBLAS error. Qwen3.5-2B/4B use the training-best proxy checkpoint because full candidate selection was unavailable. Qwen3.5 timing uses unoptimized reference kernels, so its deployment cost is conservative."
    ])
    (TABLES_DIR / "table_scale_quality_cost.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (TABLES_DIR / "table_scale_quality_cost.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = []
        for table_row in all_rows:
            for key in table_row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(TABLES_DIR / "table_scale_quality_cost.json", all_rows)


def write_ablation_table() -> dict:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    v2_path = RESULT_DIR / "ablation_v2" / "ablation_results.json"
    path = v2_path if v2_path.exists() else RESULT_DIR / "ablation" / "ablation_results.json"
    if not path.exists():
        text = "| Policy | Status |\n|---|---|\n| A0-A6 | PENDING: run scripts/supp_ablation_evaluate.py |\n"
        (TABLES_DIR / "table_ablation_budget.md").write_text(text, encoding="utf-8")
        return {}
    data = read_json(path)
    cfg = _ablation_cfg()
    a0 = read_json(A0_METRICS) if A0_METRICS.exists() else {}
    audit_cache_path = AUDIT_CACHE
    audit_rows = read_jsonl(audit_cache_path) if audit_cache_path.exists() else []
    audit_rows = sorted(audit_rows, key=lambda row: int(row.get("rank", 1_000_000)))
    audit_usage = {}
    for k in (19, 38, 57, 76):
        prefix = audit_rows[:k]
        audit_usage[k] = {
            "input_tokens": sum(int(row.get("input_tokens", 0) or 0) for row in prefix),
            "output_tokens": sum(int(row.get("output_tokens", 0) or 0) for row in prefix),
            "latency_ms": sum(float(row.get("latency_ms", 0.0) or 0.0) for row in prefix),
            "cost_cny": sum(float(row.get("estimated_cost_cny", 0.0) or 0.0) for row in prefix),
        }
    if not audit_rows and AUDIT_USAGE_SUMMARY.exists():
        summary = read_json(AUDIT_USAGE_SUMMARY)
        audit_usage = {
            int(k): {
                "input_tokens": int(item["input_tokens"]),
                "output_tokens": int(item["output_tokens"]),
                "latency_ms": float(item["latency_ms"]),
                "cost_cny": float(item["incremental_cost_cny"]),
            }
            for k, item in summary["budgets"].items()
        }
    training = read_json(STUDENT_TRAINING_STATE) if STUDENT_TRAINING_STATE.exists() else {}
    training_h = float(training.get("wall_seconds") or 0.0) / 3600.0
    model_input_tokens = None
    if STUDENT_PREDICTIONS.exists():
        model_input_tokens = sum(
            int(row.get("input_tokens", 0) or 0) for row in read_jsonl(STUDENT_PREDICTIONS)
            if row.get("relation") == "relation" and row.get("split") == "test"
        )
    deployment = (read_json(STUDENT_DEPLOYMENT_METRICS).get("deployment", {})
                  if STUDENT_DEPLOYMENT_METRICS.exists() else {})
    design_labels = {
        "A0": "未训练 1.5B，直接使用", 
        "A1": "训练后，使用源域工作点",
        "A2": "训练后，目标域校准自动判定",
        "A3": "A2 + 5% 边界审计",
        "A4": "A2 + 10% 边界审计",
        "A5": "A2 + 15% 边界审计",
        "A6": "A2 + 20% 边界审计",
    }
    lines = [
        "### 消融质量指标",
        "",
        "| Policy | 配置 | 训练 | 校准 | 审计 | 审计条数 | 阈值 | Macro-F1 | Recall | FPR | ΔF1 vs A2 | 修复 | 引入错误 | 净修正 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in cfg["ablations"]:
        policy = data["policies"].get(item["id"], {})
        if policy.get("status") != "ready":
            lines.append(
                f"| {item['id']} | {design_labels[item['id']]} | {'Y' if item['training'] else '-'} | "
                f"{'Y' if item['calibration'] else '-'} | {item['audit_rate']:.0%} | - | - | "
                f"{policy.get('status', 'pending').upper()} | - | - | - | - | - | - |"
            )
            continue
        m = policy["metrics"]
        lines.append(
            f"| {item['id']} | {design_labels[item['id']]} | {'Y' if item['training'] else '-'} | "
            f"{'Y' if item['calibration'] else '-'} | {item['audit_rate']:.0%} | "
            f"{policy.get('audited', 0)} | {_fmt(policy.get('threshold'), 4)} | "
            f"{_fmt(m['macro_f1'])} | {_fmt(m['recall'])} | {_fmt(m['fpr'])} | "
            f"{policy.get('delta_macro_f1_vs_A2', 0.0):+.4f} | "
            f"{policy.get('fixed_errors', 0)} | {policy.get('introduced_errors', 0)} | "
            f"{policy.get('net_corrections', 0):+d} |"
        )
    lines.extend([
        "",
        "### 消融 token、耗时与增量成本",
        "",
        "| Policy | 配置 | 训练耗时 h | E6 模型输入 token | E6 模型输出 token | 审计输入 token | 审计输出 token | 审计总 token | 总处理 token | 增量审计耗时 s | 平均单条审计 ms | 增量 API CNY |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in cfg["ablations"]:
        policy = data["policies"].get(item["id"], {})
        if policy.get("status") != "ready":
            continue
        if item["id"] == "A0":
            model_in = int(a0.get("input_tokens", 0) or 0)
            model_out = int(a0.get("output_tokens", 0) or 0)
            train_h = 0.0
            usage = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}
        else:
            model_in = int(model_input_tokens or 0)
            model_out = 0
            train_h = training_h
            usage = audit_usage.get(policy.get("audited", 0), {
                "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0,
            })
        total_tokens = (model_in + model_out + usage["input_tokens"] + usage["output_tokens"])
        lines.append(
            f"| {item['id']} | {design_labels[item['id']]} | {train_h:.3f} | {model_in:,} | {model_out:,} | "
            f"{usage['input_tokens']:,} | {usage['output_tokens']:,} | {usage['input_tokens'] + usage['output_tokens']:,} | {total_tokens:,} | "
            f"{usage['latency_ms'] / 1000:.2f} | {_fmt(policy.get('avg_online_latency_ms'), 2)} | "
            f"{_fmt(policy.get('incremental_api_cost_cny'), 6)} |"
        )
    lines.extend([
        "",
        f"> Token 口径：A0 的 E6 实际输入/输出 token 为 {int(a0.get('input_tokens', 0) or 0):,}/{int(a0.get('output_tokens', 0) or 0):,}，评测墙钟 {_fmt(a0.get('wall_seconds'), 1)} s，P50/P95 为 {_fmt(a0.get('latency_p50_ms'), 2)}/{_fmt(a0.get('latency_p95_ms'), 2)} ms。训练学生的 384 条 E6 输入 token 为 {_fmt(model_input_tokens, 1)}，分类输出 token 为 0；训练墙钟为 {_fmt(training.get('wall_seconds'), 1)} s。",
        f"> A3-A6 的审计 token、在线延迟和 API 费用来自真实 DeepSeek API 缓存；审计输入/输出 token 随嵌套前缀严格累加。",
        f"> 历史部署记录中，训练学生的 CPU batch-1 P50/P95 为 {_fmt(deployment.get('batch1_latency_p50_ms'), 1)}/{_fmt(deployment.get('batch1_latency_p95_ms'), 1)} ms、{_fmt(deployment.get('throughput_samples_per_s'), 2)} rows/s；该次部署基准没有与 E6 单独评测共用同一计时会话，因此仅作参考，不能与 A0 的 GPU 墙钟直接相减。",
        "> 真实 API 消融包含 5%/10%/15%/20% 四档；25%/30% 只存在于 Table 9 的 Silver oracle 扩展模拟，不作为真实 API 成本观测。"
    ])
    (TABLES_DIR / "table_ablation_budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (TABLES_DIR / "table9_real_api_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(TABLES_DIR / "table_ablation_budget.json", data)
    return data


def _svg_panel(title: str, points: list[tuple[float, float]], x_label: str, y_label: str,
               width: int = 520, height: int = 320) -> str:
    left, right, top, bottom = 64, 24, 42, 52
    pw, ph = width - left - right, height - top - bottom
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not points:
        return ""
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    # Log scale is useful for parameter counts, but remains safe for zero costs.
    x_log = min(xs) > 0
    tx = (lambda x: math.log10(x)) if x_log else (lambda x: x)
    x_min_t, x_max_t = tx(x_min), tx(x_max)
    if x_min_t == x_max_t:
        x_min_t, x_max_t = x_min_t - 1, x_max_t + 1
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.15, 0.01)
    y_min, y_max = y_min - pad, y_max + pad
    coords = []
    for x, y in points:
        px = left + (tx(x) - x_min_t) / (x_max_t - x_min_t) * pw
        py = top + (1 - (y - y_min) / (y_max - y_min)) * ph
        coords.append((px, py))
    grid = []
    for i in range(5):
        gy = top + i * ph / 4
        grid.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left+pw}" y2="{gy:.1f}" stroke="#e5e7eb"/>')
        value = y_max - i * (y_max - y_min) / 4
        grid.append(f'<text x="{left-8}" y="{gy+4:.1f}" text-anchor="end" font-size="10" fill="#4b5563">{value:.3f}</text>')
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1d4ed8"/>' for x, y in coords)
    labels = "".join(
        f'<text x="{x:.1f}" y="{top+ph+20}" text-anchor="middle" font-size="10" fill="#374151">'
        f'{orig_x:g}</text>' for (orig_x, _), (x, _) in zip(points, coords)
    )
    return f"""<g>
  <text x="{left}" y="22" font-size="15" font-weight="600" fill="#111827">{title}</text>
  {''.join(grid)}
  <polyline points="{poly}" fill="none" stroke="#1d4ed8" stroke-width="2"/>
  {circles}{labels}
  <text x="{left+pw/2}" y="{height-10}" text-anchor="middle" font-size="11" fill="#374151">{x_label}</text>
  <text x="15" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 15 {top+ph/2})" font-size="11" fill="#374151">{y_label}</text>
</g>"""


def write_scale_figure(rows: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ready = [row for row in rows if row["source"] != "pending" and row.get("online_macro_f1") is not None]
    quality = [(float(r["params_b"]), float(r["online_macro_f1"])) for r in ready]
    gpu_cost = [(float(r["params_b"]), float(r.get("gpu_hours") or 0.0)) for r in ready if r.get("gpu_hours") is not None]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="360" viewBox="0 0 1080 360">',
           '<rect width="1080" height="360" fill="white"/>',
           _svg_panel("Online quality vs parameters", quality, "Parameters (B, log scale)", "E6 Macro-F1", width=530),
           f'<g transform="translate(540,0)">',
           _svg_panel("Total GPU hours vs parameters", gpu_cost, "Parameters (B, log scale)", "GPU hours", width=530),
           '</g>', '</svg>']
    (FIGURES_DIR / "fig_scale_quality_cost.svg").write_text("".join(svg), encoding="utf-8")


def write_audit_figure(ablation: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    policies = ablation.get("policies", {}) if ablation else {}
    quality, cost = [], []
    if policies.get("A2", {}).get("status") == "ready":
        quality.append((0, policies["A2"]["metrics"]["macro_f1"]))
        cost.append((0, 0.0))
    for policy_id in ("A3", "A4", "A5", "A6"):
        policy = policies.get(policy_id, {})
        if policy.get("status") == "ready":
            k = int(policy["audited"])
            quality.append((k, policy["metrics"]["macro_f1"]))
            cost.append((k, float(policy.get("incremental_api_cost_cny") or 0.0)))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="360" viewBox="0 0 1080 360">',
           '<rect width="1080" height="360" fill="white"/>',
           _svg_panel("E6 online quality vs audit budget", quality, "Audited rows (nested prefix)", "Macro-F1", width=530),
           '<g transform="translate(540,0)">',
           _svg_panel("Incremental audit cost", cost, "Audited rows (nested prefix)", "API CNY", width=530),
           '</g>', '</svg>']
    (FIGURES_DIR / "fig_audit_budget_quality_cost.svg").write_text("".join(svg), encoding="utf-8")


def write_threshold_sensitivity() -> list[dict]:
    test_rows = read_jsonl(REPO / "experiments/lighteval_supplementary/prepared/raw/relation_test.jsonl")
    core_rows = read_jsonl(REPO / "experiments/lighteval_supplementary/prepared/raw/core_test.jsonl")
    rel_gold = {row_key(row): binary_label(row) for row in test_rows}
    core_gold = {row_key(row): binary_label(row) for row in core_rows}
    out = []
    model_dirs = sorted(SCALE_DIR.iterdir()) if SCALE_DIR.exists() else []
    for model_dir in model_dirs:
        summary_path = model_dir / "summary.json"
        rel_path = model_dir / "predictions_relation_test.jsonl"
        core_path = model_dir / "predictions_core_test.jsonl"
        if not (summary_path.exists() and rel_path.exists() and core_path.exists()):
            continue
        summary = read_json(summary_path)
        rel = read_jsonl(rel_path)
        core = read_jsonl(core_path)
        selected = float(summary["threshold_selection"]["threshold"])
        row = {"model_id": model_dir.name, "selected_threshold": selected, "source_threshold": 0.5622}
        for set_name, rows, gold in (("relation_test", rel, rel_gold), ("core_test", core, core_gold)):
            for threshold_name, threshold in (("selected", selected), ("source", 0.5622)):
                y = [gold[item["id"]] for item in rows]
                scores = [float(item["risk_score"]) for item in rows]
                row[f"{set_name}_{threshold_name}"] = binary_metrics(y, scores, threshold=threshold)
        out.append(row)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["| Model | Selected T | Source T | E6 F1 selected/source | E6 FPR selected/source | Core F1 selected/source | Core FPR selected/source |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in out:
        e_rel = row["relation_test_selected"]
        e_src = row["relation_test_source"]
        c_rel = row["core_test_selected"]
        c_src = row["core_test_source"]
        lines.append(
            f"| {row['model_id']} | {row['selected_threshold']:.4f} | 0.5622 | "
            f"{_fmt(e_rel['macro_f1'])}/{_fmt(e_src['macro_f1'])} | "
            f"{_fmt(e_rel['fpr'])}/{_fmt(e_src['fpr'])} | "
            f"{_fmt(c_rel['macro_f1'])}/{_fmt(c_src['macro_f1'])} | "
            f"{_fmt(c_rel['fpr'])}/{_fmt(c_src['fpr'])} |"
        )
    (TABLES_DIR / "table_scale_threshold_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(TABLES_DIR / "table_scale_threshold_sensitivity.json", out)
    return out

def write_main_table9() -> dict:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "ablation_oracle" / "budget_sensitivity" / "extended_oracle_budget.json"
    if not path.exists():
        return {}
    data = read_json(path)
    labels = [("A2", "校准后自动判定"), ("5pct", "参考审计5%（19条）"), ("10pct", "参考审计10%（38条）"),
              ("15pct", "参考审计15%（57条）"), ("20pct", "参考审计20%（76条）"),
              ("25pct", "参考审计25%（96条）"), ("30pct", "参考审计30%（115条）")]
    lines = ["# 表9 在线响应的预算审计结果", "", "> Reference-label replacement (Silver) oracle simulation; not an external API audit.", "", "| 运行配置 | Macro-F1 | Recall | FPR | Delta F1 vs A2 |", "|---|---:|---:|---:|---:|"]
    for key, label in labels:
        item = data["policies"].get(key)
        if not item:
            continue
        m = item["metrics"]
        lines.append(f"| {label} | {m['macro_f1']:.4f} | {m['recall']:.4f} | {m['fpr']:.4f} | {item.get('delta_f1_vs_A2') or 0:+.4f} |")
    (TABLES_DIR / "table9_online_budget_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data

def build_report() -> dict:
    rows, audit_rows = build_scale_rows()
    write_scale_table(rows, audit_rows)
    ablation = write_ablation_table()
    write_main_table9()
    write_scale_figure(rows)
    write_audit_figure(ablation)
    sensitivity = write_threshold_sensitivity()
    summary = {
        "scale_rows": rows,
        "audit_comparison": audit_rows,
        "ablation_complete": bool(ablation),
        "threshold_sensitivity_rows": sensitivity,
        "outputs": {
            "scale_table": str(TABLES_DIR / "table_scale_quality_cost.md"),
            "ablation_table": str(TABLES_DIR / "table_ablation_budget.md"),
            "scale_figure": str(FIGURES_DIR / "fig_scale_quality_cost.svg"),
            "audit_figure": str(FIGURES_DIR / "fig_audit_budget_quality_cost.svg"),
            "threshold_sensitivity": str(TABLES_DIR / "table_scale_threshold_sensitivity.md"),
            "table9_main": str(TABLES_DIR / "table9_online_budget_audit.md"),
            "table9_real_api": str(TABLES_DIR / "table9_real_api_audit.md"),
        },
    }
    write_json(RESULT_DIR / "report_index.json", summary)
    return summary


def main() -> int:
    result = build_report()
    print(json.dumps({
        "status": "READY",
        "scale_pending": [row["id"] for row in result["scale_rows"] if row["source"] == "pending"],
        "audit_comparison_ready": bool(result["audit_comparison"]),
        "outputs": result["outputs"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
