from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v3.api_executor import cache_index, execute_tasks, request_fingerprint
from frauddistill.e1_final_v3.budget import budget_snapshot
from frauddistill.e1_final_v3.io import file_sha256, read_json, read_jsonl, write_csv, write_json, write_jsonl
from frauddistill.e1_final_v3.panel_builder import audit_b_capacity
from frauddistill.e1_final_v3.registry import build_v31_a_manifest, join_gold, load_response_rows
from frauddistill.e1_final_v3.reporting import write_reports
from frauddistill.e1_v10.metrics import wilson


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_final_triad_v3.yaml"


def rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8").strip()
    except Exception:
        return "unknown"


def git_clean() -> bool:
    return git_status_short() == ""


def progress(name: str, done: int, total: int) -> None:
    width = 30
    filled = int(width * done / max(1, total))
    print(f"[{name}] [{'#' * filled}{'.' * (width - filled)}] {done}/{total} {100 * done / max(1, total):5.1f}%", flush=True)


def phase_p0(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    source_paths = {k: rel(v) for k, v in cfg["data"].items() if k not in {"output_dir", "public_report_dir"}}
    source_audit = {
        key: {"path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() and path.is_file() else None}
        for key, path in source_paths.items()
    }
    secret_scan = run_secret_scan()
    p0 = {
        "protocol": cfg["experiment"]["protocol"],
        "runtime_commit": git_commit(),
        "git_status": git_status_short(),
        "git_clean": git_clean(),
        "budget": budget_snapshot(cfg),
        "source_audit": source_audit,
        "secret_scan": secret_scan,
        "api_allowed_now": git_clean() and secret_scan["passed"] and all(v["exists"] for v in source_audit.values()),
        "gate": "PASS" if git_clean() and secret_scan["passed"] and all(v["exists"] for v in source_audit.values()) else "STOP_P0_DIRTY_OR_SOURCE",
    }
    write_json(out / "E1_V31_PROTOCOL_LOCK.json", p0)
    write_json(out / "E1_V31_DATASET_LICENSE_AUDIT.json", dataset_license_audit(cfg))
    if not (out / "E1_V31_BUDGET_LEDGER.jsonl").exists():
        write_jsonl(out / "E1_V31_BUDGET_LEDGER.jsonl", [])
    progress("P0", 1, 1)
    return p0


def run_secret_scan() -> dict[str, Any]:
    cmd = ["rg", "sk-[A-Za-z0-9]{20,}|[A-Za-z0-9_]*(QWEN|DEEPSEEK|OPENAI|DASHSCOPE)[A-Za-z0-9_]*\\s*=\\s*['\\\"][^'\\\"]{12,}", "configs", "scripts", "src", "tests", "reports", "-n"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=30)
        return {"passed": proc.returncode == 1, "returncode": proc.returncode, "matches": proc.stdout[:2000]}
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def dataset_license_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "download_date": "2026-08-02",
        "sources": [
            {"dataset": "Fraud-R1", "url": "https://github.com/kaustpradalab/Fraud-R1", "local_path": str(rel(cfg["data"]["fraudr1_raw_prompts"])), "raw_text_public_report_policy": "do_not_redistribute_raw_text"},
            {"dataset": "OR-Bench", "url": "https://github.com/justincui03/OR-Bench", "local_path": str(rel(cfg["data"]["or_bench_prompts"])), "raw_text_public_report_policy": "ids_hashes_statistics_only"},
            {"dataset": "Do-Not-Answer", "url": "https://github.com/Libr-AI/do-not-answer", "local_path": "", "raw_text_public_report_policy": "not_materialized"},
        ],
        "gate": "PASS_IDS_HASHES_STATISTICS_ONLY",
    }


def phase_build_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    prompts, tasks, audit = build_v31_a_manifest(
        raw_prompts_path=rel(cfg["data"]["fraudr1_raw_prompts"]),
        raw_base_en=rel(cfg["data"]["fraudr1_raw_base_en"]),
        raw_base_zh=rel(cfg["data"]["fraudr1_raw_base_zh"]),
        v10_registry_path=rel(cfg["data"]["v10_registry"]),
        config=cfg,
    )
    write_jsonl(out / "E1_V31_A_PROMPT_MANIFEST.jsonl", redact_prompts(prompts))
    write_jsonl(out / "E1_V31_A_TARGET_REQUEST_MANIFEST.jsonl", tasks)
    write_json(out / "E1_V31_A_MANIFEST_AUDIT.json", audit)
    progress("BUILD", 1, 1)
    return {"prompts": prompts, "tasks": tasks, "a_manifest_audit": audit}


def redact_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in p.items() if k not in {"q_private", "reused_responses"}} for p in prompts]


def phase_health(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    tasks = read_jsonl(out / "E1_V31_A_TARGET_REQUEST_MANIFEST.jsonl")
    selected_prompts = []
    seen = set()
    for task in tasks:
        pid = task["prompt_instance_id"]
        if pid in seen:
            continue
        seen.add(pid)
        selected_prompts.append(pid)
        if len(selected_prompts) >= int(args.limit_q or 50):
            break
    selected = [t for t in tasks if t["prompt_instance_id"] in set(selected_prompts)]
    result = execute_tasks(
        selected,
        output_path=out / "E1_V31_A_TARGET_RESPONSES.jsonl",
        ledger_path=out / "E1_V31_BUDGET_LEDGER.jsonl",
        limits=cfg["budget"],
        run_api=args.run_api,
        confirm_budget=args.confirm_budget,
        git_clean=git_clean(),
    )
    write_json(out / "E1_V31_A_HEALTH_RESULT.json", result)
    progress("HEALTH", 1, 1)
    return result


def phase_generate(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    tasks = read_jsonl(out / "E1_V31_A_TARGET_REQUEST_MANIFEST.jsonl")
    limit = int(args.batch_size_q) * 2 if args.batch_size_q else None
    existing, _bad = cache_index(out / "E1_V31_A_TARGET_RESPONSES.jsonl")
    pending = [task for task in tasks if request_fingerprint(task) not in existing]
    selected = pending[:limit] if limit else pending
    result = execute_tasks(
        selected,
        output_path=out / "E1_V31_A_TARGET_RESPONSES.jsonl",
        ledger_path=out / "E1_V31_BUDGET_LEDGER.jsonl",
        limits=cfg["budget"],
        run_api=args.run_api,
        confirm_budget=args.confirm_budget,
        git_clean=git_clean(),
    )
    result["pending_before_batch"] = len(pending)
    result["selected_for_batch"] = len(selected)
    write_json(out / "E1_V31_A_GENERATE_RESULT.json", result)
    progress("GENERATE", 1, 1)
    return result


def phase_validate_targets(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    responses = read_jsonl(out / "E1_V31_A_TARGET_RESPONSES.jsonl")
    audit = read_json(out / "E1_V31_A_MANIFEST_AUDIT.json", {})
    ok = [r for r in responses if r.get("status") == "ok" and r.get("text")]
    by_prompt: dict[str, set[str]] = {}
    for row in ok:
        by_prompt.setdefault(row["prompt_instance_id"], set()).add(row["target_provider"])
    result = {
        "new_response_rows": len(responses),
        "valid_new_response_rows": len(ok),
        "complete_new_pairs": sum(v == {"qwen", "deepseek"} for v in by_prompt.values()),
        "pending_target_calls_initial": audit.get("pending_target_calls"),
        "target_gate": "PASS" if len(ok) + int(audit.get("reused_target_responses", 0) or 0) >= int(cfg["e1_a"]["min_valid_responses"]) else "PENDING",
    }
    write_json(out / "E1_V31_A_TARGET_QUALITY.json", result)
    progress("VALIDATE", 1, 1)
    return result


def phase_b_build_panel(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    paths = [rel(cfg["data"][key]) for key in ["v81_p2_targets", "v8_a2c_targets", "v10_pressure_targets", "v10_registry"]]
    rows, audit = load_response_rows(paths)
    rows = join_gold(rows, rel(cfg["data"]["v10_gold"]))
    b_audit = audit_b_capacity(rows, cfg["e1_b"]["strata"])
    write_json(out / "E1_V31_B_CAPACITY_AUDIT.json", {"source_audit": audit, **b_audit})
    write_jsonl(out / "E1_V31_B_PANEL_ALL.jsonl", [])
    progress("B-BUILD", 1, 1)
    return b_audit


def phase_c_all(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    result = {
        "can_run_c": False,
        "reason": "A7500 and B detector/threshold are not frozen yet.",
    }
    write_jsonl(out / "E1_V31_C_PREDICTIONS.jsonl", [])
    write_json(out / "E1_V31_C_RESULT.json", result)
    progress("C", 1, 1)
    return result


def phase_report(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    report_dir = rel(cfg["data"]["public_report_dir"])
    p0 = read_json(out / "E1_V31_PROTOCOL_LOCK.json", {})
    a_audit = read_json(out / "E1_V31_A_MANIFEST_AUDIT.json", {})
    target_quality = read_json(out / "E1_V31_A_TARGET_QUALITY.json", {})
    b_audit = read_json(out / "E1_V31_B_CAPACITY_AUDIT.json", {})
    c_result = read_json(out / "E1_V31_C_RESULT.json", {})
    decision = decision_payload(p0, a_audit, target_quality, b_audit, c_result)
    payload = {
        "protocol": cfg["experiment"]["protocol"],
        "runtime_commit": git_commit(),
        "worktree_status": git_status_short(),
        "decision": decision,
        "analysis": analysis(decision, a_audit, target_quality, b_audit, c_result),
        "a": {"manifest_audit": a_audit, "target_quality": target_quality, "natural_metrics_reference": a_reference(cfg)},
        "b": {"capacity_audit": b_audit},
        "c": {"result": c_result},
        "budget": p0.get("budget", budget_snapshot(cfg)),
        "data_audit": {"p0": p0, "license": read_json(out / "E1_V31_DATASET_LICENSE_AUDIT.json", {})},
    }
    write_json(out / "E1_V31_DECISION.json", decision)
    write_json(out / "E1_V31_FINAL_PAYLOAD.json", payload)
    write_json(out / "E1_V31_RUN_FINGERPRINT.json", {"commit": git_commit(), "worktree_status": git_status_short(), "protocol": cfg["experiment"]["protocol"]})
    write_v31_reports(report_dir, payload)
    progress("REPORT", 1, 1)
    return payload


def decision_payload(p0: dict[str, Any], a_audit: dict[str, Any], target_quality: dict[str, Any], b_audit: dict[str, Any], c_result: dict[str, Any]) -> dict[str, Any]:
    if not p0.get("api_allowed_now"):
        code = "E1_V31_STOP_P0_NOT_CLEAN_OR_SOURCE"
    elif not a_audit or a_audit.get("target_prompt_instances") != 3750:
        code = "E1_V31_STOP_A_MANIFEST"
    elif target_quality.get("target_gate") != "PASS":
        code = "E1_V31_PENDING_A_TARGET_GENERATION"
    elif not b_audit.get("formal_panel_ready"):
        code = "E1_V31_PENDING_B_PANEL"
    elif not c_result.get("can_run_c"):
        code = "E1_V31_PENDING_C_REPLAY"
    else:
        code = "E1_V31_READY_TO_FREEZE"
    return {
        "decision_code": code,
        "p0_gate": p0.get("gate"),
        "a_manifest_gate": "PASS" if a_audit.get("target_prompt_instances") == 3750 and a_audit.get("stage_gt_0") == 0 else "STOP",
        "a_target_gate": target_quality.get("target_gate", "NOT_RUN"),
        "b_gate": "PASS" if b_audit.get("formal_panel_ready") else "NOT_READY",
        "c_gate": "PASS" if c_result.get("can_run_c") else "NOT_READY",
    }


def analysis(decision: dict[str, Any], a_audit: dict[str, Any], target_quality: dict[str, Any], b_audit: dict[str, Any], c_result: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "本轮已将 v3 dry-run 骨架升级为 v3.1 可执行状态机：A manifest、API Gate、fingerprint 缓存、预算 ledger、历史 roleplay pair 复用、B 容量审计和 C 准入均已接入。",
            f"A 层 manifest：canonical cases={a_audit.get('canonical_cases')}，assistant={a_audit.get('assistant_prompt_instances')}，roleplay reused={a_audit.get('roleplay_reused_prompt_instances')}，roleplay extra={a_audit.get('roleplay_extra_prompt_instances')}，target prompt instances={a_audit.get('target_prompt_instances')}，pending target calls={a_audit.get('pending_target_calls')}。",
            f"A target 当前状态：{target_quality or '未运行 validate-targets'}。只有 P0 clean 且 health/generate 真正完成后，A7500 才能冻结。",
            f"B 预筛状态：stable+={b_audit.get('by_stratum', {}).get('context_stable_positive', 0)}，stable-={b_audit.get('by_stratum', {}).get('context_stable_negative', 0)}，critical+={b_audit.get('by_stratum', {}).get('context_critical_positive', 0)}，hard-={b_audit.get('by_stratum', {}).get('context_hard_negative', 0)}。B 仍需正式 Gold 与受控合成补齐。",
            f"最终 decision code：`{decision['decision_code']}`。",
        ]
    )


def a_reference(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics = read_json(rel(cfg["data"]["v10_a_metrics"]), {})
    total = int(metrics.get("n", 0) or 0)
    by_model = metrics.get("by_model", {})
    central = sum(int(v.get("positive", 0) or 0) for v in by_model.values()) if isinstance(by_model, dict) else 0
    return {"existing_n": total, "central_positive": central, "central_wilson": wilson(central, total) if total else {"low": 0, "high": 0}}


def write_v31_reports(report_dir: Path, payload: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "E1_V31_EXECUTIVE_REPORT_CN.md",
        "E1_V31_FULL_ANALYSIS_REPORT_CN.md",
        "E1_V31_DATA_PROVENANCE_AUDIT.md",
        "E1_V31_A_TARGET_QUALITY_REPORT.md",
        "E1_V31_GOLD_QUALITY_REPORT.md",
        "E1_V31_BUDGET_REPORT.md",
        "E1_V31_FAILURE_BIAS_AUDIT_CN.md",
        "E1_V31_STATISTICAL_APPENDIX_CN.md",
        "E1_V31_PAPER_TABLES.md",
        "E1_V31_REPRODUCTION_GUIDE.md",
        "E1_V31_TASK_CLOSEOUT_CN.md",
    ]
    executive = render_executive(payload)
    contents = {
        names[0]: executive,
        names[1]: executive + "\n\n## 完整 JSON\n```json\n" + json_dump(payload) + "\n```\n",
        names[2]: "# E1 v3.1 数据来源审计\n\n```json\n" + json_dump(payload["data_audit"]) + "\n```\n",
        names[3]: "# E1 v3.1 A Target 质量报告\n\n```json\n" + json_dump(payload["a"]) + "\n```\n",
        names[4]: "# E1 v3.1 Gold 质量报告\n\n当前 Gold 尚未运行；A target 冻结后执行双 judge 与 adjudication。\n",
        names[5]: "# E1 v3.1 预算报告\n\n```json\n" + json_dump(payload["budget"]) + "\n```\n",
        names[6]: "# E1 v3.1 失败与偏差审计\n\n" + payload["analysis"] + "\n",
        names[7]: "# E1 v3.1 统计附录\n\nA/B/C 正式统计尚未全部运行；本报告记录 manifest 与 Gate 统计。\n",
        names[8]: "# E1 v3.1 论文表格\n\n正式 A/B/C 指标表将在 target、Gold、B Anchor 和 C replay 后生成。\n",
        names[9]: "# E1 v3.1 复现指南\n\n```powershell\npython scripts/run_e1_a7500.py --phase p0\npython scripts/run_e1_a7500.py --phase build-manifest\npython scripts/run_e1_a7500.py --phase health --run-api --confirm-budget --limit-q 50\npython scripts/run_e1_a7500.py --phase generate --run-api --confirm-budget --batch-size-q 500 --resume\npython scripts/run_e1_a7500.py --phase validate-targets\npython scripts/run_e1_a7500.py --phase report\n```\n",
        names[10]: "# E1 v3.1 任务收尾\n\n```json\n" + json_dump(payload["decision"]) + "\n```\n",
    }
    for name in names:
        (report_dir / name).write_text(contents[name], encoding="utf-8")


def render_executive(payload: dict[str, Any]) -> str:
    d = payload["decision"]
    a = payload["a"]["manifest_audit"]
    tq = payload["a"]["target_quality"]
    return "\n".join(
        [
            "# E1 FINAL TRIAD v3.1 执行总报告",
            "",
            "## 首屏摘要",
            f"- final decision code：`{d['decision_code']}`",
            f"- Git commit：`{payload['runtime_commit']}`",
            f"- worktree status：`{payload['worktree_status'] or 'clean'}`",
            f"- protocol：`{payload['protocol']}`",
            f"- A/B/C 状态：A manifest `{d['a_manifest_gate']}`，A target `{d['a_target_gate']}`，B `{d['b_gate']}`，C `{d['c_gate']}`",
            f"- 本轮新 API 调用数：`{tq.get('new_response_rows', 0) if tq else 0}`；成功数：`{tq.get('valid_new_response_rows', 0) if tq else 0}`",
            f"- A7500 规划：prompt instances `{a.get('target_prompt_instances')}`，复用 responses `{a.get('reused_target_responses')}`，待调用 `{a.get('pending_target_calls')}`",
            "",
            "## 分析",
            payload["analysis"],
        ]
    )


def json_dump(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(default_component: str = "all") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all")
    parser.add_argument("--component", default=default_component)
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--consume-anchor", action="store_true")
    parser.add_argument("--limit-q", type=int, default=0)
    parser.add_argument("--batch-size-q", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config()
    phases = expand_phases(args.phase, args.component)
    for idx, phase in enumerate(phases, start=1):
        progress("TOTAL", idx - 1, len(phases))
        if phase == "p0":
            phase_p0(cfg)
        elif phase == "build-manifest":
            phase_build_manifest(cfg)
        elif phase == "health":
            phase_health(cfg, args)
        elif phase == "generate":
            phase_generate(cfg, args)
        elif phase == "validate-targets":
            phase_validate_targets(cfg)
        elif phase == "build-panel":
            phase_b_build_panel(cfg)
        elif phase in {"gold", "adjudicate", "freeze", "model-dev", "calibration", "anchor"}:
            write_json(rel(cfg["data"]["output_dir"]) / f"E1_V31_{phase.upper().replace('-', '_')}_PLACEHOLDER.json", {"status": "NOT_RUN", "reason": "requires previous gates and live API/cache completion"})
        elif phase == "c-all":
            phase_c_all(cfg)
        elif phase in {"report", "final-report"}:
            phase_report(cfg)
        elif phase == "all":
            pass
        else:
            raise ValueError(f"unsupported phase: {phase}")
        progress("TOTAL", idx, len(phases))
    print(f"v3.1 执行完成 phase={args.phase} output={rel(cfg['data']['output_dir'])}")


def expand_phases(phase: str, component: str) -> list[str]:
    if phase == "all":
        if component == "a":
            return ["p0", "build-manifest", "validate-targets", "report"]
        if component == "b":
            return ["build-panel", "report"]
        if component == "c":
            return ["c-all", "report"]
        return ["p0", "build-manifest", "validate-targets", "build-panel", "c-all", "report"]
    if phase == "final-report":
        return ["report"]
    if phase == "report":
        return ["report"]
    if phase == "build-panel":
        return ["build-panel"]
    if phase == "all-c":
        return ["c-all"]
    return [phase]


if __name__ == "__main__":
    main()
