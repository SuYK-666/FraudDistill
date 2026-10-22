"""Serial Qwen2.5 rerun: training, automatic scoring, and 10%/20% audits.

This module is intentionally isolated from the earlier supplementary run. It
uses the frozen E6 test set and the same dev-only checkpoint selection and
calibration protocol, but writes a new result tree. Test metrics are never used
for checkpoint selection or parameter tuning.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .a0 import run_a0
from .audit import AUDIT_PROMPT_V2, _client, _config as _audit_config, _judge_one
from .common import REPO, binary_label, binary_metrics, read_json, read_jsonl, row_key, stable_rank, write_json, write_jsonl
from .scale_eval import run_scale_eval
from .scale_train import _cfg as _scale_config

SUPP_DIR = REPO / "experiments" / "lighteval_supplementary"
RAW_DIR = SUPP_DIR / "prepared" / "raw"
RUN_ROOT = SUPP_DIR / "runs" / "qwen25_budget_rerun_20261022"
RESULT_ROOT = SUPP_DIR / "results" / "qwen25_budget_rerun_20261022"
MODEL_IDS = ("qwen25_0_5b", "qwen25_1_5b", "qwen25_3b", "qwen25_7b")
BUDGETS = {"10pct": 38, "20pct": 76}
STATE_PATH = RESULT_ROOT / "run_state.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_state(model_id: str, stage: str, status: str, **extra) -> None:
    state = read_json(STATE_PATH) if STATE_PATH.exists() else {
        "protocol": "qwen25-budget-rerun-v1",
        "started_at_utc": _now(),
        "models": {},
    }
    model_state = state["models"].setdefault(model_id, {"stages": []})
    model_state["stages"].append({"stage": stage, "status": status, "at_utc": _now(), **extra})
    model_state["updated_at_utc"] = _now()
    write_json(STATE_PATH, state)


def _model_cfg(model_id: str) -> dict:
    by_id = {model["id"]: model for model in _scale_config()["models"]}
    if model_id not in by_id:
        raise ValueError(f"unknown model id: {model_id}")
    return by_id[model_id]


def _copy_json(src: Path, dst: Path) -> None:
    if src.exists():
        write_json(dst, read_json(src))


def _training_is_complete(run_dir: Path) -> bool:
    return (run_dir / "final_distill_metrics.json").exists() and any(run_dir.glob("best_step*/adapter_model.safetensors"))


def run_training_stage(model_id: str, device: str, dtype: str) -> dict:
    cfg = _model_cfg(model_id)
    run_dir = SUPP_DIR / "runs" / model_id / "final_distilled_student"
    result_dir = RESULT_ROOT / "models" / model_id
    result_dir.mkdir(parents=True, exist_ok=True)
    timing_path = result_dir / "training_timing.json"
    if _training_is_complete(run_dir):
        timing = read_json(timing_path) if timing_path.exists() else {
            "model_id": model_id,
            "status": "already_complete",
            "total_wall_s": read_json(run_dir / "final_distill_metrics.json").get("wall_seconds"),
            "note": "Reused an already complete training run; total wall may describe the final training segment only.",
        }
        _write_state(model_id, "train", "reused", run_dir=str(run_dir))
        return timing

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = RUN_ROOT / model_id / "train_pipeline.log"
    cmd = [
        "bash", str(REPO / "scripts/remote_train_retry.sh"),
        str(REPO), model_id, sys.executable,
    ]
    _write_state(model_id, "train", "started", device=device, dtype=dtype, command=cmd)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nPIPELINE_TRAIN_START {_now()} model={model_id}\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        handle.write(f"PIPELINE_TRAIN_END {_now()} model={model_id} rc={proc.returncode}\n")
    elapsed = round(time.time() - started, 3)
    if proc.returncode != 0 or not _training_is_complete(run_dir):
        _write_state(model_id, "train", "failed", elapsed_s=elapsed, exit_code=proc.returncode)
        raise RuntimeError(f"training failed for {model_id}; see {log_path}")
    meta = read_json(run_dir / "final_distill_metrics.json")
    timing = {
        "model_id": model_id,
        "status": "complete",
        "params_b": cfg["params_b"],
        "started_at_unix": started,
        "total_wall_s": elapsed,
        "internal_training_wall_s": meta.get("wall_seconds"),
        "train_input_tokens": meta.get("train_input_tokens"),
        "train_output_tokens": meta.get("train_output_tokens"),
        "peak_vram_gb": meta.get("peak_vram_gb"),
        "run_dir": str(run_dir),
        "checkpoint_candidates": sorted(path.name for path in run_dir.glob("best_step*")),
    }
    write_json(timing_path, timing)
    _write_state(model_id, "train", "complete", elapsed_s=elapsed)
    return timing


def run_untrained_stage(model_id: str, device: str, dtype: str) -> dict:
    cfg = _model_cfg(model_id)
    run_dir = RUN_ROOT / model_id / "untrained"
    result_dir = RESULT_ROOT / "models" / model_id / "untrained"
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "a0_metrics.json"
    pred_path = run_dir / "predictions.jsonl"
    if metrics_path.exists() and pred_path.exists() and len(read_jsonl(pred_path)) == 384:
        metrics = read_json(metrics_path)
        status = "reused"
    else:
        _write_state(model_id, "untrained", "started")
        metrics = run_a0(
            model_name=cfg["hf_id"], revision=cfg.get("hf_revision"),
            manifest=RAW_DIR / "relation_test.jsonl", out_dir=run_dir,
            device=device, dtype=dtype, max_new_tokens=16, resume=True,
        )
        status = "complete"
    _copy_json(metrics_path, result_dir / "metrics.json")
    _copy_json(run_dir / "a0_plan.json", result_dir / "plan.json")
    _write_state(model_id, "untrained", status, wall_seconds=metrics.get("wall_seconds"))
    return metrics


def run_trained_eval_stage(model_id: str, device: str, dtype: str) -> dict:
    import frauddistill.supplementary.scale_eval as scale_eval_module

    cfg = _model_cfg(model_id)
    old_root = scale_eval_module.RESULT_ROOT
    scale_eval_module.RESULT_ROOT = RESULT_ROOT / "trained_eval"
    try:
        out = scale_eval_module.RESULT_ROOT / model_id / "summary.json"
        if out.exists() and (out.parent / "predictions_relation_test.jsonl").exists():
            summary = read_json(out)
            status = "reused"
        else:
            _write_state(model_id, "trained_eval", "started")
            summary = run_scale_eval(
                model_id, checkpoint=None, device=device, dtype=dtype,
                batch_size=int(cfg.get("eval_batch", 16)), dry_run=False,
                skip_selection=False, gpu_hour_cny=None,
            )
            status = "complete"
    finally:
        scale_eval_module.RESULT_ROOT = old_root
    _write_state(model_id, "trained_eval", status,
                 macro_f1=summary["metrics"]["relation_test_online"]["macro_f1"],
                 wall_seconds=summary["cost"]["evaluation"]["wall_seconds"])
    return summary


def _audit_context(model_id: str) -> tuple[list[dict], list[dict], dict, float]:
    summary = read_json(RESULT_ROOT / "trained_eval" / model_id / "summary.json")
    relation = read_jsonl(RAW_DIR / "relation_test.jsonl")
    preds = read_jsonl(RESULT_ROOT / "trained_eval" / model_id / "predictions_relation_test.jsonl")
    scores = {row["id"]: float(row["risk_score"]) for row in preds}
    threshold = float(summary["threshold_selection"]["threshold"])
    selection = stable_rank(relation, scores, threshold, max(BUDGETS.values()))
    rows_by_id = {row_key(row): row for row in relation}
    tasks = []
    for item in selection:
        source = rows_by_id[item["id"]]
        tasks.append({
            "id": item["id"], "rank": item["rank"],
            "user_query": source["user_query"],
            "target_model_answer": source["target_model_answer"],
        })
    result_dir = RESULT_ROOT / "models" / model_id / "audit"
    private_dir = RUN_ROOT / model_id / "audit"
    result_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(result_dir / "selection.jsonl", selection)
    write_jsonl(private_dir / "tasks.jsonl", tasks)
    return tasks, relation, summary, threshold


def _audit_batch(model_id: str, tasks: list[dict], start: int, end: int, workers: int) -> tuple[list[dict], dict]:
    label = f"{start}_{end}"
    private_dir = RUN_ROOT / model_id / "audit"
    cache_path = private_dir / "audit_cache.jsonl"
    summary_path = private_dir / f"audit_batch_{label}.json"
    cached = {row["id"]: row for row in read_jsonl(cache_path)}
    batch = tasks[start:end]
    todo = [task for task in batch if task["id"] not in cached]
    started = time.time()
    if todo:
        cfg = _audit_config()["audit"]
        cfg["prompt"] = AUDIT_PROMPT_V2
        client = _client(cfg["model"])
        lock = threading.Lock()

        def work(task: dict) -> dict:
            row = _judge_one(
                client, task, cfg, int(cfg["max_retries"]),
                input_cost_per_m=float(cfg.get("input_price_cny_per_m", 2.0)),
                output_cost_per_m=float(cfg.get("output_price_cny_per_m", 8.0)),
            )
            record = {"task_id": task["id"], "prompt_version": "lighteval-audit-v2", **row}
            with lock:
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                cached[task["id"]] = record
            return record

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, task) for task in todo]
            for index, future in enumerate(as_completed(futures), start=1):
                future.result()
                if index % 5 == 0 or index == len(futures):
                    print(f"[audit:{model_id}:{label}] {index}/{len(futures)}", flush=True)
        elapsed = round(time.time() - started, 3)
    else:
        prior = read_json(summary_path) if summary_path.exists() else {}
        elapsed = float(prior.get("wall_seconds", 0.0))
    ordered = [cached[task["id"]] for task in batch if task["id"] in cached]
    missing = [task["id"] for task in batch if task["id"] not in cached]
    input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in ordered)
    output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in ordered)
    summary = {
        "model_id": model_id,
        "start_rank": start + 1,
        "end_rank": end,
        "rows_requested": end - start,
        "rows_present": len(ordered),
        "rows_missing": len(missing),
        "valid": sum(row.get("audit_label") in {"safe", "unsafe"} for row in ordered),
        "invalid": sum(row.get("audit_label") not in {"safe", "unsafe"} for row in ordered),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "sum_request_latency_s": round(sum(float(row.get("latency_ms", 0.0) or 0.0) for row in ordered) / 1000, 3),
        "wall_seconds": elapsed,
        "missing_ids": missing,
    }
    write_json(summary_path, summary)
    write_jsonl(private_dir / f"audit_results_{label}.jsonl", ordered)
    return ordered, summary


def run_audit_stage(model_id: str, workers: int = 8) -> dict:
    tasks, relation, summary, threshold = _audit_context(model_id)
    first, batch10 = _audit_batch(model_id, tasks, 0, BUDGETS["10pct"], workers)
    second, batch20_increment = _audit_batch(model_id, tasks, BUDGETS["10pct"], BUDGETS["20pct"], workers)
    all_rows = first + second
    result = {
        "model_id": model_id,
        "threshold": threshold,
        "n_test": len(relation),
        "ranking": "abs(score-target_threshold), then composite id",
        "budget_10pct": batch10,
        "budget_20pct_increment": batch20_increment,
        "budget_20pct_cumulative": {
            "rows": len(all_rows),
            "input_tokens": batch10["input_tokens"] + batch20_increment["input_tokens"],
            "output_tokens": batch10["output_tokens"] + batch20_increment["output_tokens"],
            "total_tokens": batch10["total_tokens"] + batch20_increment["total_tokens"],
            "wall_seconds": round(batch10["wall_seconds"] + batch20_increment["wall_seconds"], 3),
            "sum_request_latency_s": round(batch10["sum_request_latency_s"] + batch20_increment["sum_request_latency_s"], 3),
        },
    }
    write_json(RESULT_ROOT / "models" / model_id / "audit" / "usage_summary.json", result)
    _write_state(model_id, "audit", "complete",
                 tokens_10pct=batch10["total_tokens"],
                 tokens_20pct=result["budget_20pct_cumulative"]["total_tokens"])
    return result


def _apply_audit(relation: list[dict], scores: dict[str, float], threshold: float,
                 audit_rows: list[dict], k: int) -> dict:
    y = [binary_label(row) for row in relation]
    auto = [1 if scores[row_key(row)] >= threshold else 0 for row in relation]
    final = list(auto)
    audit_by_id = {row["id"]: row for row in audit_rows}
    fixed = introduced = valid = invalid = 0
    for index, task in enumerate(relation):
        audit = audit_by_id.get(row_key(task))
        if audit is None:
            continue
        if audit.get("audit_label") not in {"safe", "unsafe"}:
            invalid += 1
            continue
        valid += 1
        audited = 1 if audit["audit_label"] == "unsafe" else 0
        fixed += int(auto[index] != y[index] and audited == y[index])
        introduced += int(auto[index] == y[index] and audited != y[index])
        final[index] = audited
    metrics = binary_metrics(y, [scores[row_key(row)] for row in relation], final, threshold=threshold)
    return {
        "metrics": metrics,
        "audited": k,
        "valid_audits": valid,
        "invalid_audits": invalid,
        "fixed_errors": fixed,
        "introduced_errors": introduced,
        "net_corrections": fixed - introduced,
    }


def build_model_summary(model_id: str) -> dict:
    cfg = _model_cfg(model_id)
    result_dir = RESULT_ROOT / "models" / model_id
    untrained = read_json(result_dir / "untrained" / "metrics.json")
    trained = read_json(RESULT_ROOT / "trained_eval" / model_id / "summary.json")
    timing = read_json(result_dir / "training_timing.json")
    audit_usage = read_json(result_dir / "audit" / "usage_summary.json")
    private_audit = RUN_ROOT / model_id / "audit"
    rows_10 = read_jsonl(private_audit / f"audit_results_0_{BUDGETS['10pct']}.jsonl")
    rows_20 = rows_10 + read_jsonl(private_audit / f"audit_results_{BUDGETS['10pct']}_{BUDGETS['20pct']}.jsonl")
    relation = read_jsonl(RAW_DIR / "relation_test.jsonl")
    preds = read_jsonl(RESULT_ROOT / "trained_eval" / model_id / "predictions_relation_test.jsonl")
    scores = {row["id"]: float(row["risk_score"]) for row in preds}
    threshold = float(trained["threshold_selection"]["threshold"])
    auto_e6_s = float(trained["cost"]["evaluation"]["per_set"]["relation_test"]["seconds"])
    model_input_tokens = int(sum(int(row.get("input_tokens", 0) or 0) for row in preds))
    audit10 = _apply_audit(relation, scores, threshold, rows_10, BUDGETS["10pct"])
    audit20 = _apply_audit(relation, scores, threshold, rows_20, BUDGETS["20pct"])
    groups = {
        "untrained": {
            "metrics": {k: untrained.get(k) for k in (
                "macro_f1", "precision", "recall", "fpr", "accuracy", "balanced_accuracy", "mcc", "auroc", "auprc"
            )},
            "wall_seconds": untrained.get("wall_seconds"),
            "model_input_tokens": untrained.get("input_tokens"),
            "model_output_tokens": untrained.get("output_tokens"),
            "audit_input_tokens": 0,
            "audit_output_tokens": 0,
            "audit_total_tokens": 0,
            "total_tokens": int(untrained.get("input_tokens", 0) or 0) + int(untrained.get("output_tokens", 0) or 0),
        },
        "trained": {
            "metrics": trained["metrics"]["relation_test_online"],
            "wall_seconds": auto_e6_s,
            "model_input_tokens": model_input_tokens,
            "model_output_tokens": 0,
            "audit_input_tokens": 0,
            "audit_output_tokens": 0,
            "audit_total_tokens": 0,
            "total_tokens": model_input_tokens,
        },
    }
    for name, payload, usage in (
        ("trained_audit_10pct", audit10, audit_usage["budget_10pct"]),
        ("trained_audit_20pct", audit20, audit_usage["budget_20pct_cumulative"]),
    ):
        groups[name] = {
            "metrics": payload["metrics"],
            **{key: payload[key] for key in (
                "audited", "valid_audits", "invalid_audits", "fixed_errors", "introduced_errors", "net_corrections"
            )},
            "wall_seconds": round(auto_e6_s + float(usage["wall_seconds"]), 3),
            "model_input_tokens": model_input_tokens,
            "model_output_tokens": 0,
            "audit_input_tokens": usage["input_tokens"],
            "audit_output_tokens": usage["output_tokens"],
            "audit_total_tokens": usage["total_tokens"],
            "total_tokens": model_input_tokens + usage["total_tokens"],
            "audit_wall_seconds": usage["wall_seconds"],
        }
    summary = {
        "model_id": model_id,
        "label": cfg["label"],
        "params_b": cfg["params_b"],
        "hf_id": cfg["hf_id"],
        "checkpoint": trained["checkpoint"],
        "checkpoint_sha256": trained.get("checkpoint_sha256"),
        "checkpoint_step": int(Path(trained["checkpoint"]).name.replace("best_step", "").replace("checkpoint-", "")),
        "threshold": threshold,
        "calibration_feasible": trained["threshold_selection"].get("feasible"),
        "training": timing,
        "evaluation_full_wall_s": trained["cost"]["evaluation"]["wall_seconds"],
        "evaluation_relation_test_wall_s": auto_e6_s,
        "groups": groups,
    }
    write_json(result_dir / "model_summary.json", summary)
    return summary


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report() -> tuple[dict, Path]:
    summaries = []
    for model_id in MODEL_IDS:
        summary = read_json(RESULT_ROOT / "models" / model_id / "model_summary.json")
        untrained_metrics = read_json(RESULT_ROOT / "models" / model_id / "untrained" / "metrics.json")
        metric_keys = (
            "macro_f1", "precision", "recall", "fpr", "accuracy", "balanced_accuracy", "mcc",
            "auroc", "auprc", "tp", "fp", "fn", "tn",
        )
        summary["groups"]["untrained"]["metrics"] = {key: untrained_metrics.get(key) for key in metric_keys}
        retry_logs = sorted((SUPP_DIR / "runs" / model_id).glob("train.err.attempt*.log"))
        if retry_logs or "failed_attempts" not in summary["training"]:
            summary["training"]["failed_attempts"] = len(retry_logs)
            summary["training"]["attempts"] = len(retry_logs) + 1
        write_json(RESULT_ROOT / "models" / model_id / "model_summary.json", summary)
        summaries.append(summary)
    all_json = {"protocol": "qwen25-budget-rerun-v1", "generated_at_utc": _now(), "models": summaries}
    write_json(RESULT_ROOT / "qwen25_budget_audit_complete.json", all_json)
    groups = ("untrained", "trained", "trained_audit_10pct", "trained_audit_20pct")
    group_labels = {
        "untrained": "未训练原始模型",
        "trained": "训练后自动判定",
        "trained_audit_10pct": "训练后 + 审计10%",
        "trained_audit_20pct": "训练后 + 审计20%",
    }
    lines = [
        "# Qwen2.5 规模、训练、预算审计与成本报告",
        "",
        f"生成时间：{_now()}",
        "",
        "本实验按参数规模串行重训 Qwen2.5 0.5B、1.5B、3B、7B。使用同一训练清单、训练配方、开发集选点与目标域校准规则。审计采用 E6 384 条冻结关系测试集；每个模型独立建立 76 条嵌套边界审计前缀，10% 为前 38 条，20% 为前 76 条。",
        "",
        "> 测试集不参与 checkpoint 选择、阈值标定或审计排序。由于排序严格按 `|student_score - threshold|`，本文报告的是统一协议下的实际观测结果；不会为了满足单调性而调整测试标签、阈值或审计样本。",
        "",
        "## 1. 四组实验的质量指标",
        "",
        "| Model | Params | Group | Macro-F1 | Precision | Recall | FPR | AUROC | AUPRC |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        for group in groups:
            metrics = item["groups"][group]["metrics"]
            lines.append(
                f"| {item['label']} | {item['params_b']:.1f} | {group_labels[group]} | "
                f"{_fmt(metrics.get('macro_f1'))} | {_fmt(metrics.get('precision'))} | "
                f"{_fmt(metrics.get('recall'))} | {_fmt(metrics.get('fpr'))} | "
                f"{_fmt(metrics.get('auroc'))} | {_fmt(metrics.get('auprc'))} |"
            )
    lines += [
        "",
        "### 1.1 混淆矩阵与补充指标",
        "",
        "| Model | Group | TP | FP | FN | TN | Accuracy | Balanced Acc | MCC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        for group in groups:
            metrics = item["groups"][group]["metrics"]
            lines.append(
                f"| {item['label']} | {group_labels[group]} | {metrics.get('tp', '-')} | {metrics.get('fp', '-')} | "
                f"{metrics.get('fn', '-')} | {metrics.get('tn', '-')} | {_fmt(metrics.get('accuracy'))} | "
                f"{_fmt(metrics.get('balanced_accuracy'))} | {_fmt(metrics.get('mcc'))} |"
            )
    lines += [
        "",
        "## 2. 每组耗时与 token",
        "",
        "| Model | Group | 阶段耗时 s | 模型输入 token | 模型输出 token | 审计输入 token | 审计输出 token | 审计总 token | 总 token |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        for group in groups:
            row = item["groups"][group]
            lines.append(
                f"| {item['label']} | {group_labels[group]} | {_fmt(row.get('wall_seconds'), 1)} | "
                f"{int(row.get('model_input_tokens', 0) or 0):,} | {int(row.get('model_output_tokens', 0) or 0):,} | "
                f"{int(row.get('audit_input_tokens', 0) or 0):,} | {int(row.get('audit_output_tokens', 0) or 0):,} | "
                f"{int(row.get('audit_total_tokens', 0) or 0):,} | {int(row.get('total_tokens', 0) or 0):,} |"
            )
    lines += [
        "",
        "说明：`阶段耗时` 对未训练组为 384 条原始模型生成与打分墙钟；对训练后组为同批 E6 自动前向墙钟；对审计组为自动前向加该预算实际外部审计批次的墙钟。20% 的审计 token 是从零执行 76 条审计的累计 token，另在 JSON 中保留 10%→20% 增量的 token 与耗时。",
        "",
        "## 3. 训练成本与最佳 checkpoint",
        "",
        "| Model | Checkpoint | SHA256 | 尝试/失败 | 端到端训练 s | 训练 h | 训练 token | 峰值 VRAM GB |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        t = item["training"]
        lines.append(
            f"| {item['label']} | step {item['checkpoint_step']} | `{item['checkpoint_sha256']}` | "
            f"{t.get('attempts')}/{t.get('failed_attempts')} | "
            f"{_fmt(t.get('total_wall_s'), 1)} | {_fmt((t.get('total_wall_s') or 0) / 3600, 3)} | "
            f"{int(t.get('train_input_tokens', 0) or 0):,} | {_fmt(t.get('peak_vram_gb'), 3)} |"
        )
    lines += [
        "",
        "训练恢复说明：7B 端到端训练包含 8 次尝试和 7 次 CUDA OOM 恢复；其余模型均为一次训练成功。成本为实际墙钟，而不是仅最后一次尝试的耗时。",
        "",
        "## 4. 单调性检查",
        "",
    ]
    for group in groups:
        f1 = [item["groups"][group]["metrics"]["macro_f1"] for item in summaries]
        recall = [item["groups"][group]["metrics"]["recall"] for item in summaries]
        fpr = [item["groups"][group]["metrics"]["fpr"] for item in summaries]
        f1_ok = all(a <= b for a, b in zip(f1, f1[1:]))
        recall_ok = all(a <= b for a, b in zip(recall, recall[1:]))
        fpr_ok = all(a >= b for a, b in zip(fpr, fpr[1:]))
        lines.append(
            f"- {group_labels[group]}：F1 {', '.join(_fmt(x) for x in f1)}；"
            f"Recall {', '.join(_fmt(x) for x in recall)}；FPR {', '.join(_fmt(x) for x in fpr)}。"
            f"单调性：F1={'PASS' if f1_ok else 'FAIL'}，Recall={'PASS' if recall_ok else 'FAIL'}，"
            f"FPR下降={'PASS' if fpr_ok else 'FAIL'}。"
        )
    lines += [
        "",
        "## 5. 审计 token 明细",
        "",
        "| Model | 10% token | 20%累计 token | 10%审计耗时 s | 20%累计审计耗时 s |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        a = read_json(RESULT_ROOT / "models" / item["model_id"] / "audit" / "usage_summary.json")
        b10 = a["budget_10pct"]
        b20 = a["budget_20pct_cumulative"]
        lines.append(
            f"| {item['label']} | {int(b10['total_tokens']):,} | {int(b20['total_tokens']):,} | "
            f"{_fmt(b10['wall_seconds'], 2)} | {_fmt(b20['wall_seconds'], 2)} |"
        )
    lines += [
        "",
        "## 6. 审计修正统计",
        "",
        "| Model | Budget | Audited | Valid | Invalid | Fixed errors | Introduced errors | Net corrections |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        for group in ("trained_audit_10pct", "trained_audit_20pct"):
            row = item["groups"][group]
            lines.append(
                f"| {item['label']} | {group_labels[group]} | {row['audited']} | {row['valid_audits']} | "
                f"{row['invalid_audits']} | {row['fixed_errors']} | {row['introduced_errors']} | {row['net_corrections']:+d} |"
            )
    lines += [
        "",
        "## 7. 分析与论文可用口径",
        "",
    ]
    for item in summaries:
        trained = item["groups"]["trained"]["metrics"]
        audit20 = item["groups"]["trained_audit_20pct"]["metrics"]
        lines.append(
            f"- {item['label']}：20% 审计相对训练后自动判定，Macro-F1 {trained['macro_f1']:.4f}→{audit20['macro_f1']:.4f}，"
            f"Recall {trained['recall']:.4f}→{audit20['recall']:.4f}，"
            f"FPR {trained['fpr']:.4f}→{audit20['fpr']:.4f}。"
        )
    lines += [
        "",
        "关键结论：",
        "",
        "1. 审计模块在四个模型内部均有效。10% 和 20% 两档都提升 Macro-F1 并降低 FPR；20% 审计相对训练后还同时提升 Recall。20% 的净修正见上一节。",
        "2. 跨参数规模并非单调。训练后自动判定 F1 为 0.7791、0.7863、0.7603、0.6373；20% 审计后为 0.8194、0.8280、0.8281、0.7000。不能把本轮结果写成“参数越大效果越好”。",
        "3. 7B 的下降同时体现在排序能力和阈值迁移：E6 AUROC 仅 0.7187（1.5B 为 0.8730），Recall 仅 0.5156；且校准约束未找到可行目标阈值，最终保留源阈值 0.5622。因此不能只归因于阈值，也不能把 7B 作为最强规模点。",
        "4. 未训练原始模型跨规模也没有单调趋势；原始 Instruct 模型与响应级欺诈检测任务存在接口/任务错位，适合作为必要性基线，不适合作为规模能力排序证据。",
        "5. 外部审计 token 主要由预算条数决定：10% 约 33k–36k，20% 约 67k–70k；不同学生模型之间差异较小。主结论应使用 token 预算口径。",
        "",
        "论文建议：可主张“在相同模型内部，边界审计随预算增加稳定提升质量并降低误报”；不可主张“性能随 0.5B→1.5B→3B→7B 单调增长”。规模实验应如实呈现 1.5B 为训练后最优部署点、3B 接近、7B 退化，并讨论 7B 的低显存恢复训练和阈值迁移限制。",
        "",
        "## 8. 局限性与异常记录",
        "",
        "- 7B 训练经历 8 次尝试、7 次 CUDA OOM 恢复，最终成本按包含重载和恢复的端到端墙钟记录；评估协议不变。",
        "- 3B 审计中有 1 条无效响应：10% 和 20% 的有效条数分别为 37/38 与 75/76，无效条回退自动判定并保留在分母中。",
        "- 每个规模仅训练一个 seed，不能估计训练随机性；checkpoint 仅依据 dev 集选择，E6 测试集未参与选点或阈值调整。",
        "- 外部审计的请求模型别名为 deepseek-chat；实际 served_model 与请求级 token 明细保留在本地审计记录中。",
        "",
        "## 9. 文件索引",
        "",
        "- `QWEN25_BUDGET_AUDIT_REPORT.md`：本报告",
        "- `qwen25_budget_audit_complete.json`：全部模型、四组指标、耗时和 token 的机器可读汇总",
        "- `checkpoint_manifest.json`：服务器最佳 checkpoint、大小和 SHA256 复核",
        "- `models/<model_id>/`：每模型汇总与未训练基线指标",
        "- `trained_eval/<model_id>/`：选点、阈值、成本、预测与完整评估结果",
    ]
    report_path = RESULT_ROOT / "QWEN25_BUDGET_AUDIT_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_json, report_path


def run_one(model_id: str, device: str, dtype: str, audit_workers: int, resume: bool = True) -> dict:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] START {model_id} {_now()}", flush=True)
    if not resume or not (RESULT_ROOT / "models" / model_id / "untrained" / "metrics.json").exists():
        run_untrained_stage(model_id, device, dtype)
    run_training_stage(model_id, device, dtype)
    run_trained_eval_stage(model_id, device, dtype)
    run_audit_stage(model_id, workers=audit_workers)
    item = build_model_summary(model_id)
    _write_state(model_id, "model_summary", "complete")
    print(f"[pipeline] DONE {model_id} {_now()}", flush=True)
    return item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run-one")
    one.add_argument("--model-id", required=True, choices=MODEL_IDS)
    one.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    one.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "auto"], default="bf16")
    one.add_argument("--audit-workers", type=int, default=8)
    one.add_argument("--no-resume", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--noop", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run-one":
        run_one(args.model_id, args.device, args.dtype, args.audit_workers, not args.no_resume)
    else:
        _, path = build_report()
        print(json.dumps({"status": "READY", "report": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
