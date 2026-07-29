from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frauddistill.eval.metrics import binary_metrics
from frauddistill.exp1_restored.manifest import build_anchor_manifest, build_full_manifest
from frauddistill.exp1_restored.stats import decision_from_stats, paired_stats
from frauddistill.experiments.run_exp1_single_judge_api import INPUT_MODES, run_api_judge
from frauddistill.labelers.refusal_rules import detect_refusal
from frauddistill.student.pair_tfidf import PairTfidfDetector
from frauddistill.target_llm.provider_config import get_provider_config
from frauddistill.utils.io import read_jsonl, write_jsonl


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_restored_v1plus_final.yaml"
TAXONOMY_PATH = ROOT / "configs" / "data" / "FRAUD_TAXONOMY_LOCK.yaml"
PREFIX = "E1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 API empirical restore protocol")
    parser.add_argument(
        "--stage",
        choices=["preflight_api", "anchor2400", "anchor_api", "build_full", "freeze", "formal", "report", "all"],
        required=True,
    )
    parser.add_argument("--data_dir", default="data/prepared/e1_restored_v1plus_final")
    parser.add_argument("--output_dir", default="outputs/e1_restored_v1plus_final")
    parser.add_argument("--bootstrap_iterations", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = ROOT / args.data_dir
    output_dir = ROOT / args.output_dir
    if args.stage == "all":
        for stage in ("preflight_api", "anchor2400", "anchor_api"):
            payload = run_stage(stage, config, data_dir, output_dir, args.bootstrap_iterations)
            if not str(payload.get("decision", "")).endswith(("READY", "PASS")):
                run_stage("report", config, data_dir, output_dir, args.bootstrap_iterations)
                raise SystemExit(f"{stage} stopped with decision={payload.get('decision')}")
        for stage in ("build_full", "freeze", "formal", "report"):
            run_stage(stage, config, data_dir, output_dir, args.bootstrap_iterations)
    else:
        run_stage(args.stage, config, data_dir, output_dir, args.bootstrap_iterations)


def run_stage(stage: str, config: dict, data_dir: Path, output_dir: Path, bootstrap_iterations: int | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage == "preflight_api":
        payload = api_preflight(config, output_dir)
    elif stage == "anchor2400":
        audit = build_anchor_manifest(config, data_dir)
        payload = {"decision": "E1_ANCHOR_READY" if audit["rows"] == 2400 and audit["label_balance"] else "E1_STOP", "stage": stage, "audit": audit}
    elif stage == "anchor_api":
        require_decision(output_dir, "anchor2400", {"E1_ANCHOR_READY"})
        payload = run_anchor_api(config, data_dir, output_dir, bootstrap_iterations)
    elif stage == "build_full":
        require_decision(output_dir, "anchor_api", {"E1_ANCHOR_API_PASS"})
        audit = build_full_manifest(config, data_dir, TAXONOMY_PATH)
        payload = {"decision": "E1_FULL_MANIFEST_READY" if audit["rows"] == 8000 and audit["label_balance"] and audit.get("passed", False) else "E1_STOP", "stage": stage, "audit": audit}
    elif stage == "freeze":
        require_decision(output_dir, "build_full", {"E1_FULL_MANIFEST_READY"})
        payload = freeze_manifest(data_dir)
    elif stage == "formal":
        require_decision(output_dir, "freeze", {"E1_MANIFEST_FROZEN"})
        payload = run_formal(config, data_dir, output_dir, bootstrap_iterations)
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
    started = time.time()
    try:
        summary = run_api_judge(
            config["data"]["anchor_file"],
            str(output_dir / "preflight_api"),
            provider=provider,
            model=model,
            concurrency=int(api_cfg["concurrency"]),
            limit=int(api_cfg["sample_rows"]),
            modes=INPUT_MODES,
            suffix="_preflight",
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
        "failed_fill_policy": "failed rows remain status=failed and pred_label=null; metrics stop if success gate fails",
    }
    write_json(output_dir / "preflight_api" / "E1_API_PREFLIGHT.json", payload)
    return payload


def run_anchor_api(config: dict, data_dir: Path, output_dir: Path, bootstrap_iterations: int | None) -> dict:
    api_cfg = config["api"]["anchor"]
    rows = [row for row in read_jsonl(data_dir / "anchor2400.jsonl") if row.get("split") == "formal_test"]
    rows = rows[: int(api_cfg["sample_rows"])]
    anchor_file = output_dir / "anchor_api" / "E1_ANCHOR480_INPUT.jsonl"
    write_jsonl(anchor_file, rows)
    started = time.time()
    summary = run_api_judge(
        str(anchor_file),
        str(output_dir / "anchor_api"),
        provider=api_cfg["provider"],
        model=api_cfg["judge_model"],
        concurrency=int(api_cfg["concurrency"]),
        modes=INPUT_MODES,
        suffix="_anchor480",
    )
    predictions = load_api_predictions(output_dir / "anchor_api", summary["provider"], "_anchor480")
    completeness = api_completeness(rows, predictions)
    if not completeness["complete"]:
        payload = {
            "decision": "E1_ANCHOR_API_STOP",
            "stage": "anchor_api",
            "summary": summary,
            "api_completeness": completeness,
            "wall_seconds": round(time.time() - started, 3),
        }
        write_json(output_dir / "anchor_api" / "E1_ANCHOR_API_STATS.json", payload)
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
        "hard_safe_fpr": m["q_y"].get("hard_safe_fpr", 1.0) <= float(gates["hard_safe_fpr_max"]),
        "q_y_minus_y_ci_positive": stats["bootstrap"]["delta.q_y-y_only"]["low"] > 0.0,
    }
    return {"decision": "E1_ANCHOR_API_PASS" if all(checks.values()) else "E1_ANCHOR_API_STOP", "checks": checks}


def load_api_predictions(stage_dir: Path, provider: str, suffix: str) -> dict[str, list[dict]]:
    pred_dir = stage_dir / "predictions"
    out = {}
    for mode in INPUT_MODES:
        path = pred_dir / f"single_judge_api_{provider}_{mode}{suffix}_predictions.jsonl"
        out[mode] = [row for row in read_jsonl(path) if row.get("status") == "ok"]
    return out


def api_completeness(rows: list[dict], predictions: dict[str, list[dict]]) -> dict:
    ids = {row["id"] for row in rows}
    rates = {}
    missing = {}
    for mode, pred_rows in predictions.items():
        ok_ids = {row["id"] for row in pred_rows}
        rates[mode] = len(ok_ids & ids) / max(len(ids), 1)
        missing[mode] = sorted(ids - ok_ids)[:20]
    return {"complete": all(rate == 1.0 for rate in rates.values()), "success_rate_by_mode": rates, "success_rate_min": min(rates.values()) if rates else 0.0, "missing_examples": missing}


def freeze_manifest(data_dir: Path) -> dict:
    hashes = {}
    for name in ("train.jsonl", "model_dev.jsonl", "threshold_cal.jsonl", "formal_test.jsonl"):
        path = data_dir / name
        hashes[name] = sha256(path) if path.exists() else None
    passed = all(hashes.values())
    write_json(data_dir / "E1_MANIFEST_HASHES.json", hashes)
    return {"decision": "E1_MANIFEST_FROZEN" if passed else "E1_STOP", "stage": "freeze", "manifest_hashes": hashes}


def run_formal(config: dict, data_dir: Path, output_dir: Path, bootstrap_iterations: int | None) -> dict:
    started = time.time()
    train = list(read_jsonl(data_dir / "train.jsonl"))
    dev = list(read_jsonl(data_dir / "model_dev.jsonl"))
    test = list(read_jsonl(data_dir / "formal_test.jsonl"))
    modes = config["model"]["modes"]
    rows_metrics = []
    pred_dir = output_dir / "formal" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    primary_predictions = {}
    for seed_idx, seed in enumerate(config["model"]["cpu_seeds"]):
        for mode in modes:
            best = select_cpu_model(train, dev, mode, config["model"]["c_grid"], int(seed))
            preds = predict_cpu(best["model"], test, mode, detector=f"CPU PairTfidf seed={seed} C={best['C']}")
            metric = binary_metrics([row["gold_label"] for row in test], [row["pred_label"] for row in preds], [row["pred_score"] for row in preds])
            rows_metrics.append({"detector": "CPU PairTfidf", "input": mode, "seed": seed, "selected_C": best["C"], **metric})
            if seed_idx == 0:
                primary_predictions[mode] = preds
                write_jsonl(pred_dir / f"cpu_pairtfidf_{mode}.jsonl", preds)
    rule_predictions = {}
    for mode in modes:
        preds = predict_rule(test, mode)
        rule_predictions[mode] = preds
        write_jsonl(pred_dir / f"rule_{mode}.jsonl", preds)
        metric = binary_metrics([row["gold_label"] for row in test], [row["pred_label"] for row in preds], [row["pred_score"] for row in preds])
        rows_metrics.append({"detector": "Rule", "input": mode, "seed": "rule", "selected_C": "", **metric})
    iterations = int(bootstrap_iterations or config["statistics"]["bootstrap_iterations"])
    stats = paired_stats(test, primary_predictions, iterations=iterations, seed=int(config["statistics"]["bootstrap_seed"]))
    decision = decision_from_stats(stats, config["gates"]["full"])
    write_csv(output_dir / "formal" / "E1_CPU_METRICS_BY_SEED.csv", rows_metrics)
    write_json(output_dir / "formal" / "E1_PAIRED_STATS.json", stats)
    write_csv(output_dir / "formal" / "E1_CPU_METRICS_BY_SUBGROUP.csv", subgroup_rows(test, primary_predictions))
    write_json(output_dir / "formal" / "E1_FAILURE_AUDIT.json", {"api_predictions_present": False, "failed_call_fill": 0, "note": "Formal CPU stage is allowed only after Anchor API pass; failed API rows are never filled."})
    write_csv(output_dir / "formal" / "E1_COST_PROFILE.csv", [{"stage": "formal_cpu", "wall_seconds": round(time.time() - started, 3), "api_calls": 0}])
    write_json(output_dir / "formal" / "E1_DECISION.json", {**decision, "stats": stats})
    return {"decision": decision["decision"], "stage": "formal", "checks": decision["checks"], "stats": stats}


def select_cpu_model(train: list[dict], dev: list[dict], mode: str, c_grid: list[float], seed: int) -> dict:
    ordered_train = sorted(train, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode("utf-8")).hexdigest())
    best = None
    for c in c_grid:
        model = PairTfidfDetector(C=float(c)).fit(ordered_train, [row["gold_label"] for row in ordered_train], mode=mode)
        scores = model.predict_proba(dev, mode=mode)
        preds = ["unsafe" if score >= 0.5 else "safe" for score in scores]
        metric = binary_metrics([row["gold_label"] for row in dev], preds, scores.tolist())
        if best is None or metric["macro_f1"] > best["metric"]["macro_f1"]:
            best = {"C": float(c), "model": model, "metric": metric}
    return best or {}


def predict_cpu(model: PairTfidfDetector, rows: list[dict], mode: str, detector: str) -> list[dict]:
    scores = model.predict_proba(rows, mode=mode)
    return [
        {"id": row["id"], "gold_label": row["gold_label"], "pred_label": "unsafe" if score >= 0.5 else "safe", "pred_score": float(score), "detector": detector, "input": mode, "bucket": row.get("bucket"), "source": row.get("source"), "language": row.get("language"), "canonical_prompt_cluster": row.get("canonical_prompt_cluster"), "relation_challenge": row.get("relation_challenge", 0)}
        for row, score in zip(rows, scores)
    ]


def predict_rule(rows: list[dict], mode: str) -> list[dict]:
    out = []
    for row in rows:
        q = str(row.get("user_query", "")) if mode in {"q_only", "q_y"} else ""
        y = str(row.get("target_model_answer", "")) if mode in {"y_only", "q_y"} else ""
        text = f"{q} {y}".lower()
        risky = any(term in text for term in ("phishing", "password", "otp", "bank", "transfer", "scam", "credential", "verify", "wallet", "验证码", "密码", "转账", "钓鱼", "诈骗"))
        refusal = bool(detect_refusal(q, y).get("refused"))
        education = any(term in text for term in ("prevent", "report", "recognize", "education", "防范", "举报", "识别"))
        score = 0.75 if risky else 0.25
        if mode != "q_only" and (refusal or education):
            score -= 0.35
        score = max(0.0, min(1.0, score))
        out.append({"id": row["id"], "gold_label": row["gold_label"], "pred_label": "unsafe" if score >= 0.5 else "safe", "pred_score": score, "detector": "Rule", "input": mode, "bucket": row.get("bucket"), "source": row.get("source"), "language": row.get("language"), "canonical_prompt_cluster": row.get("canonical_prompt_cluster"), "relation_challenge": row.get("relation_challenge", 0)})
    return out


def subgroup_rows(rows: list[dict], predictions: dict[str, list[dict]]) -> list[dict]:
    pred_by_mode = {mode: {row["id"]: row for row in pred_rows} for mode, pred_rows in predictions.items()}
    groups = {"ALL": rows}
    for key in ("bucket", "language", "source", "relation_challenge"):
        values = defaultdict(list)
        for row in rows:
            values[str(row.get(key, ""))].append(row)
        for value, group_rows in values.items():
            groups[f"{key}={value}"] = group_rows
    out = []
    for group, group_rows in groups.items():
        for mode in predictions:
            pred = [pred_by_mode[mode][row["id"]]["pred_label"] for row in group_rows]
            score = [pred_by_mode[mode][row["id"]]["pred_score"] for row in group_rows]
            metric = binary_metrics([row["gold_label"] for row in group_rows], pred, score)
            out.append({"subgroup": group, "input": mode, "n": len(group_rows), **metric})
    return out


def write_report(config: dict, data_dir: Path, output_dir: Path) -> dict:
    latest = latest_decision(output_dir)
    report = output_dir / "report" / "E1_API_EMPIRICAL_REPORT_CN.md"
    lines = [
        "# FraudDistill E1 API 实证推进报告",
        "",
        f"- 协议：`{config['experiment']['protocol']}`",
        f"- 最新阶段：`{latest.get('stage')}`",
        f"- 最新判定：`{latest.get('decision')}`",
        f"- Git 提交：`{latest.get('git_commit', git_commit())}`",
        "",
        "## 本轮关键修订",
        "",
        "1. 将上一轮 CPU 结果定位为 `E1_CPU_DIAGNOSTIC_STOP_R0`，不再作为 q->y 假设的最终否定。",
        "2. 新增真实 API 预检与 Anchor API Gate，失败行不回填，所有 API 响应保留模型、usage、request id、finish reason 和响应哈希。",
        "3. API Judge 采用追加式 JSONL 缓存，重复运行只补缺失成功样本，避免重复付费调用。",
        "4. Anchor API 未通过前不进入 full manifest、formal CPU 或全量 API。",
        "",
        "## 阶段结果",
        "",
    ]
    for stage in ("preflight_api", "anchor2400", "anchor_api", "build_full", "freeze", "formal"):
        path = output_dir / stage / f"{PREFIX}_DECISION.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        lines.append(f"### {stage}")
        lines.append(f"- 判定：`{payload.get('decision')}`")
        if stage == "anchor_api" and payload.get("stats"):
            lines.extend(anchor_stats_table(payload["stats"]))
        if stage == "formal" and payload.get("stats"):
            lines.extend(anchor_stats_table(payload["stats"]))
        if payload.get("checks"):
            lines.append("")
            lines.append("| Check | Pass |")
            lines.append("|---|---:|")
            for key, value in payload["checks"].items():
                lines.append(f"| `{key}` | {value} |")
        lines.append("")
    deepseek_path = output_dir / "anchor_api_deepseek_preflight" / "tables" / "single_judge_api_deepseek_deepseek_preflight_metrics.json"
    if deepseek_path.exists():
        deepseek = json.loads(deepseek_path.read_text(encoding="utf-8"))
        lines.extend(["## DeepSeek 预检对照", "", f"- 模型：`{deepseek.get('model')}`", f"- 样本数：`{deepseek.get('sample_count')}`", ""])
        lines.extend(anchor_stats_table({"metrics": deepseek.get("metrics", {})}))
        lines.extend(["", "结论：DeepSeek 40 样本预检中 q+y 未优于 y-only，且 Macro-F1 明显低于 Qwen Anchor，因此本轮未继续执行 DeepSeek 480 全 Anchor。", ""])
    lines.extend([
        "## 数据与原始结果位置",
        "",
        f"- 数据目录：`{data_dir}`",
        f"- 输出目录：`{output_dir}`",
        "- 预测原始 JSONL 保留在各阶段 `predictions/` 子目录。",
        "- 旧输出已归档到 `archive/`，当前正式输出目录只保留本轮数据。",
    ])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    final_report = reports_dir / "E1_API实证推进_任务报告_中文.md"
    final_report.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    return {"decision": latest.get("decision", "E1_STOP"), "stage": "report", "report": str(report), "tracked_report": str(final_report)}


def anchor_stats_table(stats: dict) -> list[str]:
    lines = ["", "| Input | Macro-F1 | Recall | Precision | FPR | Hard-safe FPR | AUPRC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, metric in stats.get("metrics", {}).items():
        lines.append(f"| {mode} | {metric.get('macro_f1', 0):.4f} | {metric.get('recall', 0):.4f} | {metric.get('precision', 0):.4f} | {metric.get('fpr', 0):.4f} | {metric.get('hard_safe_fpr', 0):.4f} | {metric.get('auprc', 0):.4f} |")
    return lines


def latest_decision(output_dir: Path) -> dict:
    anchor_path = output_dir / "anchor_api" / f"{PREFIX}_DECISION.json"
    if anchor_path.exists():
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        if anchor.get("decision") != "E1_ANCHOR_API_PASS":
            return anchor
    for stage in ("formal", "anchor_api", "anchor2400", "preflight_api"):
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
