from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frauddistill.exp1_restored.manifest import build_anchor_manifest
from frauddistill.exp1_restored.stats import paired_stats
from frauddistill.experiments.run_exp1_single_judge_api import INPUT_MODES, run_api_judge
from frauddistill.target_llm.provider_config import get_provider_config
from frauddistill.utils.io import read_jsonl, write_jsonl

CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_restored_v1plus_final.yaml"
PREFIX = "E1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 Prompt-Parity R2")
    parser.add_argument("--stage", choices=["anchor2400", "preflight_api", "anchor_api", "report", "all"], required=True)
    parser.add_argument("--data_dir", default="data/prepared/e1_restored_v1plus_final")
    parser.add_argument("--output_dir", default="outputs/e1_restored_v1plus_final")
    parser.add_argument("--bootstrap_iterations", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = ROOT / args.data_dir
    output_dir = ROOT / args.output_dir

    if args.stage == "all":
        for stage in ("anchor2400", "preflight_api", "anchor_api"):
            payload = run_stage(stage, config, data_dir, output_dir, args.bootstrap_iterations)
            if payload.get("decision") not in {"E1_ANCHOR_READY", "E1_PREFLIGHT_READY", "E1_ANCHOR_API_PASS"}:
                run_stage("report", config, data_dir, output_dir, args.bootstrap_iterations)
                raise SystemExit(f"{stage} stopped with decision={payload.get('decision')}")
        run_stage("report", config, data_dir, output_dir, args.bootstrap_iterations)
    else:
        run_stage(args.stage, config, data_dir, output_dir, args.bootstrap_iterations)


def run_stage(stage: str, config: dict, data_dir: Path, output_dir: Path, bootstrap_iterations: int | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage == "anchor2400":
        audit = build_anchor_manifest(config, data_dir)
        write_smoke_input(data_dir, output_dir, int(config["api"]["preflight"]["sample_rows"]))
        payload = {"decision": "E1_ANCHOR_READY" if audit["rows"] == 2400 and audit["label_balance"] else "E1_STOP", "stage": stage, "audit": audit}
    elif stage == "preflight_api":
        require_decision(output_dir, "anchor2400", {"E1_ANCHOR_READY"})
        payload = api_preflight(config, output_dir)
    elif stage == "anchor_api":
        require_decision(output_dir, "preflight_api", {"E1_PREFLIGHT_READY"})
        payload = run_anchor_api(config, data_dir, output_dir, bootstrap_iterations)
    elif stage == "report":
        payload = write_report(config, data_dir, output_dir)
    else:
        raise ValueError(stage)
    payload = {"protocol": config["experiment"]["protocol"], "git_commit": git_commit(), "git_status": git_status(), **payload}
    write_json(output_dir / stage / f"{PREFIX}_DECISION.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    return payload


def api_preflight(config: dict, output_dir: Path) -> dict:
    api_cfg = config["api"]["preflight"]
    provider = api_cfg["provider"]
    model = api_cfg["judge_model"]
    pc = get_provider_config(provider, model)
    smoke_input = output_dir / "preflight_api" / "E1_SMOKE40_INPUT.jsonl"
    started = time.time()
    try:
        summary = run_api_judge(
            str(smoke_input),
            str(output_dir / "preflight_api"),
            provider=provider,
            model=model,
            concurrency=int(api_cfg["concurrency"]),
            modes=INPUT_MODES,
            suffix="_smoke40",
            temperature=float(api_cfg.get("temperature", 0.0)),
            max_tokens=int(api_cfg.get("max_tokens", 256)),
            enable_thinking=api_cfg.get("enable_thinking"),
        )
        success = min(metric.get("api_success_rate", 0.0) for metric in summary["metrics"].values())
        complete = all(metric.get("status") == "complete" for metric in summary["metrics"].values())
        decision = "E1_PREFLIGHT_READY" if complete and success >= float(api_cfg["success_min"]) else "E1_PREFLIGHT_STOP"
        error = None
    except Exception as exc:  # noqa: BLE001
        summary = {}
        success = 0.0
        decision = "E1_PREFLIGHT_STOP"
        error = {"type": exc.__class__.__name__, "message": str(exc)[:500]}
    payload = {
        "decision": decision,
        "stage": "preflight_api",
        "provider": provider,
        "requested_model": model,
        "base_url": pc.base_url,
        "sample_rows": int(api_cfg["sample_rows"]),
        "planned_api_calls": int(api_cfg["sample_rows"]) * len(INPUT_MODES),
        "api_success_rate_min": success,
        "summary": summary,
        "error": error,
        "wall_seconds": round(time.time() - started, 3),
    }
    write_json(output_dir / "preflight_api" / "E1_API_PREFLIGHT.json", payload)
    return payload


def run_anchor_api(config: dict, data_dir: Path, output_dir: Path, bootstrap_iterations: int | None) -> dict:
    api_cfg = config["api"]["anchor"]
    rows = [row for row in read_jsonl(data_dir / "anchor2400.jsonl") if row.get("split") == "formal_test"]
    rows = rows[: int(api_cfg["sample_rows"])]
    anchor_file = output_dir / "anchor_api" / "E1_ANCHOR480_INPUT.jsonl"
    write_jsonl(anchor_file, rows)
    write_json(output_dir / "anchor_api" / "E1_ANCHOR480_INPUT_HASH.json", {"rows": len(rows), "sha256": sha256(anchor_file)})

    started = time.time()
    summary = run_api_judge(
        str(anchor_file),
        str(output_dir / "anchor_api"),
        provider=api_cfg["provider"],
        model=api_cfg["judge_model"],
        concurrency=int(api_cfg["concurrency"]),
        modes=INPUT_MODES,
        suffix="_anchor480",
        temperature=float(api_cfg.get("temperature", 0.0)),
        max_tokens=int(api_cfg.get("max_tokens", 256)),
        enable_thinking=api_cfg.get("enable_thinking"),
    )
    predictions = load_api_predictions(output_dir / "anchor_api", summary["provider"], "_anchor480")
    completeness = api_completeness(rows, predictions)
    if not completeness["complete"]:
        payload = {"decision": "E1_ANCHOR_API_STOP_STRUCTURAL", "stage": "anchor_api", "summary": summary, "api_completeness": completeness, "wall_seconds": round(time.time() - started, 3)}
        write_json(output_dir / "anchor_api" / "E1_ANCHOR_API_STATS.json", payload)
        write_api_audits(output_dir / "anchor_api", rows, predictions, summary, completeness)
        return payload

    iterations = int(bootstrap_iterations or config["statistics"].get("anchor_bootstrap_iterations", 2000))
    stats = paired_stats(rows, predictions, iterations=iterations, seed=int(config["statistics"]["bootstrap_seed"]))
    decision = decision_from_anchor_stats(stats, config["gates"]["anchor"], float(api_cfg["success_min"]), completeness)
    payload = {
        "decision": decision["decision"],
        "stage": "anchor_api",
        "provider": summary["provider"],
        "requested_model": summary["model"],
        "sample_rows": len(rows),
        "planned_api_calls": len(rows) * len(INPUT_MODES),
        "summary": summary,
        "api_completeness": completeness,
        "checks": decision["checks"],
        "stats": stats,
        "wall_seconds": round(time.time() - started, 3),
    }
    write_json(output_dir / "anchor_api" / "E1_ANCHOR_API_STATS.json", payload)
    write_api_audits(output_dir / "anchor_api", rows, predictions, summary, completeness)
    return payload


def decision_from_anchor_stats(stats: dict, gates: dict, success_min: float, completeness: dict) -> dict:
    m = stats["metrics"]
    delta_qy_y = m["q_y"]["macro_f1"] - m["y_only"]["macro_f1"]
    delta_y_q = m["y_only"]["macro_f1"] - m["q_only"]["macro_f1"]
    checks = {
        "api_success_rate": completeness["success_rate_min"] >= success_min,
        "q_only_min": m["q_only"]["macro_f1"] >= float(gates["q_only_min"]),
        "q_only_max": m["q_only"]["macro_f1"] <= float(gates["q_only_max"]),
        "y_only_min": m["y_only"]["macro_f1"] >= float(gates["y_only_min"]),
        "q_y_min": m["q_y"]["macro_f1"] >= float(gates["q_y_min"]),
        "q_y_minus_y": delta_qy_y >= float(gates["q_y_minus_y_min"]),
        "y_minus_q": delta_y_q >= float(gates["y_minus_q_min"]),
        "q_y_minus_y_ci_lower": stats["bootstrap"]["delta.q_y-y_only"]["low"] > float(gates["q_y_minus_y_ci_lower_min"]),
        "q_y_recall": m["q_y"]["recall"] >= float(gates["q_y_recall_min"]),
        "q_y_precision": m["q_y"]["precision"] >= float(gates["q_y_precision_min"]),
        "q_y_fpr": m["q_y"]["fpr"] <= float(gates["q_y_fpr_max"]),
        "hard_safe_fpr": m["q_y"].get("hard_safe_fpr", 1.0) <= float(gates["hard_safe_fpr_max"]),
        "q_y_auprc": m["q_y"].get("auprc", 0.0) >= float(gates["q_y_auprc_min"]),
    }
    if all(checks.values()):
        decision = "E1_ANCHOR_API_PASS"
    elif m["q_y"]["macro_f1"] >= 0.875:
        decision = "E1_ANCHOR_API_STOP_NEAR"
    else:
        decision = "E1_ANCHOR_API_STOP_STRUCTURAL"
    return {"decision": decision, "checks": checks}


def write_smoke_input(data_dir: Path, output_dir: Path, sample_rows: int) -> None:
    rows = [row for row in read_jsonl(data_dir / "anchor2400.jsonl") if row.get("split") == "model_dev"]
    selected = stratified_rows(rows, sample_rows, "smoke40")
    smoke_input = output_dir / "preflight_api" / "E1_SMOKE40_INPUT.jsonl"
    write_jsonl(smoke_input, selected)
    write_json(output_dir / "preflight_api" / "E1_SMOKE40_INPUT_HASH.json", {"rows": len(selected), "sha256": sha256(smoke_input), "source_split": "anchor2400.model_dev"})


def stratified_rows(rows: list[dict], sample_rows: int, salt: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("bucket", ""))].append(row)
    targets = {
        "unsafe_regular": sample_rows // 4,
        "hard_unsafe": sample_rows // 4,
        "safe_refusal_generated": sample_rows // 4,
        "anti_fraud_education_safe": sample_rows // 8,
        "hard_benign_safe": sample_rows // 8,
    }
    selected = []
    for bucket, target in targets.items():
        candidates = sorted(buckets.get(bucket, []), key=lambda row: hashlib.sha256(f"{salt}:{row['id']}".encode("utf-8")).hexdigest())
        selected.extend(candidates[:target])
    return selected[:sample_rows]


def load_api_predictions(stage_dir: Path, provider: str, suffix: str) -> dict[str, list[dict]]:
    pred_dir = stage_dir / "predictions"
    return {
        mode: [row for row in read_jsonl(pred_dir / f"single_judge_api_{provider}_{mode}{suffix}_predictions.jsonl") if row.get("status") == "ok"]
        for mode in INPUT_MODES
    }


def api_completeness(rows: list[dict], predictions: dict[str, list[dict]]) -> dict:
    ids = {row["id"] for row in rows}
    rates = {}
    missing = {}
    for mode, pred_rows in predictions.items():
        ok_ids = {row["id"] for row in pred_rows}
        rates[mode] = len(ok_ids & ids) / max(len(ids), 1)
        missing[mode] = sorted(ids - ok_ids)[:20]
    return {"complete": all(rate == 1.0 for rate in rates.values()), "success_rate_by_mode": rates, "success_rate_min": min(rates.values()) if rates else 0.0, "missing_examples": missing}


def write_api_audits(stage_dir: Path, rows: list[dict], predictions: dict[str, list[dict]], summary: dict, completeness: dict) -> None:
    write_json(stage_dir / "E1_API_COMPLETENESS.json", completeness)
    write_json(stage_dir / "E1_API_FAILURE_AUDIT.json", {"status_counts": {mode: metric.get("status_counts", {}) for mode, metric in summary.get("metrics", {}).items()}})
    usage_rows = []
    model_rows = []
    cache_rows = []
    input_ids = {row["id"] for row in rows}
    for mode, pred_rows in predictions.items():
        for pred in pred_rows:
            if pred["id"] not in input_ids:
                continue
            usage = pred.get("usage") or {}
            usage_rows.append({"input": mode, "id": pred["id"], "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens")})
            model_rows.append({"input": mode, "id": pred["id"], "requested_model": pred.get("requested_model"), "response_model": pred.get("response_model"), "request_id": pred.get("request_id"), "finish_reason": pred.get("finish_reason"), "base_url_region": pred.get("base_url_region")})
            cache_rows.append({"input": mode, "id": pred["id"], "prompt_revision": pred.get("prompt_revision"), "schema_revision": pred.get("schema_revision"), "normalizer_revision": pred.get("normalizer_revision"), "prompt_sha256": pred.get("prompt_sha256"), "run_fingerprint": pred.get("run_fingerprint")})
    write_csv(stage_dir / "E1_API_USAGE.csv", usage_rows)
    write_csv(stage_dir / "E1_API_MODEL_AUDIT.csv", model_rows)
    write_csv(stage_dir / "E1_CACHE_AUDIT.csv", cache_rows)


def write_report(config: dict, data_dir: Path, output_dir: Path) -> dict:
    latest = latest_decision(output_dir)
    report = output_dir / "report" / "E1_PROMPT_PARITY_R2_REPORT_CN.md"
    lines = [
        "# FraudDistill E1 Prompt-Parity R2 任务报告",
        "",
        f"- 协议：`{config['experiment']['protocol']}`",
        f"- 最新阶段：`{latest.get('stage')}`",
        f"- 最新判定：`{latest.get('decision')}`",
        f"- Git 提交：`{latest.get('git_commit', git_commit())}`",
        "",
        "## 本轮关键修订",
        "",
        "1. 已归档上一轮 `E1_ANCHOR_QWENMAX_POSITIVE_BUT_CONFOUNDED_R0`。",
        "2. 三种输入模式使用同一个 q->y 判定目标和同一套 Rubric，差异只来自字段隐藏。",
        "3. 输出 schema 改为 `p_unsafe`，并强制 label-score-type 语义一致。",
        "4. 缓存绑定 provider/model/base URL/temperature/max_tokens/thinking/prompt/schema/normalizer revision。",
        "5. 本轮 API 并发配置为 80；若完整率不足则 Gate 停止。",
        "",
        "## 阶段结果",
        "",
    ]
    for stage in ("anchor2400", "preflight_api", "anchor_api"):
        path = output_dir / stage / f"{PREFIX}_DECISION.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"### {stage}")
        lines.append(f"- 判定：`{payload.get('decision')}`")
        if stage == "preflight_api":
            lines.append(f"- API 完整率下界：`{payload.get('api_success_rate_min')}`")
        if stage == "anchor_api" and payload.get("stats"):
            lines.extend(anchor_stats_table(payload["stats"]))
        if payload.get("checks"):
            lines.extend(["", "| Check | Pass |", "|---|---:|"])
            for key, value in payload["checks"].items():
                lines.append(f"| `{key}` | {value} |")
        lines.append("")
    lines.extend([
        "## 数据与审计文件",
        "",
        f"- 数据目录：`{data_dir}`",
        f"- 输出目录：`{output_dir}`",
        "- 原始预测：各阶段 `predictions/`。",
        "- API 审计：`E1_API_COMPLETENESS.json`、`E1_API_MODEL_AUDIT.csv`、`E1_API_USAGE.csv`、`E1_API_FAILURE_AUDIT.json`、`E1_CACHE_AUDIT.csv`。",
        "- Anchor 输入 hash：`anchor_api/E1_ANCHOR480_INPUT_HASH.json`。",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    final_report = reports_dir / "E1_PromptParity_R2_任务报告_中文.md"
    final_report.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    return {"decision": latest.get("decision", "E1_STOP"), "stage": "report", "report": str(report), "tracked_report": str(final_report)}


def anchor_stats_table(stats: dict) -> list[str]:
    lines = ["", "| Input | Macro-F1 | Recall | Precision | FPR | Hard-safe FPR | AUPRC | AUROC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, metric in stats.get("metrics", {}).items():
        lines.append(f"| {mode} | {metric.get('macro_f1', 0):.4f} | {metric.get('recall', 0):.4f} | {metric.get('precision', 0):.4f} | {metric.get('fpr', 0):.4f} | {metric.get('hard_safe_fpr', 0):.4f} | {metric.get('auprc', 0):.4f} | {metric.get('auroc', 0):.4f} |")
    return lines


def latest_decision(output_dir: Path) -> dict:
    for stage in ("anchor_api", "preflight_api", "anchor2400"):
        path = output_dir / stage / f"{PREFIX}_DECISION.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"decision": "E1_NOT_STARTED", "stage": "none"}


def require_decision(output_dir: Path, stage: str, allowed: set[str]) -> None:
    path = output_dir / stage / f"{PREFIX}_DECISION.json"
    if not path.exists():
        raise SystemExit(f"missing upstream decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("decision") not in allowed:
        raise SystemExit(f"stage {stage} decision {payload.get('decision')} not in {sorted(allowed)}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_status() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    main()
