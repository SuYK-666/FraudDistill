from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v3.budget import budget_snapshot, hard_stop_decision
from frauddistill.e1_final_v3.io import file_sha256, read_json, write_csv, write_json, write_jsonl
from frauddistill.e1_final_v3.panel_builder import audit_b_capacity
from frauddistill.e1_final_v3.registry import join_gold, load_fraudr1_q_manifest, load_response_rows
from frauddistill.e1_final_v3.reporting import write_reports
from frauddistill.e1_v10.metrics import wilson


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_final_triad_v3.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


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


def progress(name: str, done: int, total: int) -> None:
    width = 30
    filled = int(width * done / max(1, total))
    print(f"[{name}] [{'#' * filled}{'.' * (width - filled)}] {done}/{total} {100 * done / max(1, total):5.1f}%", flush=True)


def phase_p0(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    budget = budget_snapshot(cfg)
    source_paths = {k: rel(v) for k, v in cfg["data"].items() if k not in {"output_dir", "public_report_dir"}}
    source_audit = {
        key: {"path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() and path.is_file() else None}
        for key, path in source_paths.items()
    }
    license_audit = {
        "download_date": "2026-08-02",
        "sources": [
            {
                "dataset": "Fraud-R1",
                "url": "https://github.com/kaustpradalab/Fraud-R1",
                "local_path": str(rel(cfg["data"]["fraudr1_prompts"])),
                "redistribution_policy": "final public artifacts should prefer IDs/hashes/statistics unless dataset license snapshot permits raw text redistribution",
            },
            {
                "dataset": "OR-Bench",
                "url": "https://github.com/justincui03/OR-Bench",
                "local_path": str(rel(cfg["data"]["or_bench_prompts"])),
                "redistribution_policy": "open-control only; not natural prevalence",
            },
            {
                "dataset": "Do-Not-Answer",
                "url": "https://github.com/Libr-AI/do-not-answer",
                "local_path": "",
                "redistribution_policy": "not yet materialized in v3 local run",
            },
        ],
        "gate": "PASS_WITH_LOCAL_HASHES",
        "note": "License evidence is recorded as URLs and local file hashes. Raw text redistribution remains restricted in reports.",
    }
    p0 = {
        "protocol": cfg["experiment"]["protocol"],
        "runtime_commit": git_commit(),
        "git_status": git_status_short(),
        "budget": budget,
        "source_audit": source_audit,
        "license_audit": license_audit,
        "gate": "PASS_DRY_RUN" if all(v["exists"] for v in source_audit.values() if "targets" not in v["path"]) else "STOP_SOURCE_MISSING",
        "api_allowed_now": False,
        "api_block_reason": "P0 implementation run must be committed and reviewed before live API expansion. Current command performs reproducible dry-run/audit only.",
    }
    write_json(out / "E1_V3_PROTOCOL_LOCK.json", p0)
    write_json(out / "E1_V3_DATASET_LICENSE_AUDIT.json", license_audit)
    write_jsonl(out / "E1_V3_BUDGET_LEDGER.jsonl", [])
    progress("P0", 1, 1)
    return p0


def phase_a(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    prompts, prompt_audit = load_fraudr1_q_manifest(rel(cfg["data"]["fraudr1_prompts"]))
    existing, reg_audit = load_response_rows([rel(cfg["data"]["v10_registry"])])
    existing = join_gold(existing, rel(cfg["data"]["v10_gold"]))
    existing_counts = existing_a_counts(existing)
    selected, quota_table = build_a_manifest(prompts, existing_counts, cfg)
    manifest = []
    for row in selected:
        for provider in ["qwen", "deepseek"]:
            manifest.append(
                {
                    **row,
                    "target_provider": provider,
                    "requested_target_model": cfg["models"][f"target_{provider}"]["model"],
                    "phase": "E1-A-target-generation",
                    "status": "PENDING_API",
                }
            )
    write_jsonl(out / "E1_V3_A_EXPANSION_Q_MANIFEST.jsonl", selected)
    write_jsonl(out / "E1_V3_A_TARGET_REQUEST_MANIFEST.jsonl", manifest)
    write_jsonl(out / "E1_V3_TARGET_REGISTRY.jsonl", existing)
    write_json(out / "E1_V3_A_QUOTA_AUDIT.json", {"prompt_audit": prompt_audit, "existing_registry_audit": reg_audit, "quota_table": quota_table, "pending_target_calls": len(manifest)})
    progress("E1-A", 1, 1)
    return {"existing_a_rows": existing, "a_quota_table": quota_table, "a_pending_calls": len(manifest)}


def existing_a_counts(existing: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    by_cell: dict[tuple[str, str], set[str]] = {}
    for row in existing:
        if row.get("source_dataset") != "V10-natural-real":
            continue
        if not row.get("q_private"):
            continue
        key = (row.get("language", "unknown"), row.get("fraud_category", "unknown"))
        by_cell.setdefault(key, set()).add(str(row.get("canonical_q_id") or row.get("q_hash_recomputed")))
    return {k: len(v) for k, v in by_cell.items()}


def build_a_manifest(prompts: list[dict[str, Any]], existing_counts: dict[tuple[str, str], int], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prompts:
        by_cell.setdefault((row["language"], row["fraud_category"]), []).append(row)
    selected = []
    quota_table = []
    target_per_cell = int(cfg["e1_a"]["per_language_category_q"])
    for lang in ["en", "zh"]:
        for category in ["fake_job_posting", "fraudulent_service", "impersonation", "network_friendship", "phishing"]:
            rows = by_cell.get((lang, category), [])
            existing_n = int(existing_counts.get((lang, category), 0))
            needed = max(0, target_per_cell - existing_n)
            new_rows = rows[:needed]
            selected.extend(new_rows)
            quota_table.append(
                {
                    "language": lang,
                    "category": category,
                    "existing_unique_q": existing_n,
                    "target_unique_q": target_per_cell,
                    "new_q_needed": needed,
                    "new_q_selected": len(new_rows),
                    "cell_ready": existing_n + len(new_rows) >= target_per_cell,
                }
            )
    return selected, quota_table


def phase_b(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    response_paths = [rel(cfg["data"][key]) for key in ["v81_p2_targets", "v8_a2c_targets", "v10_pressure_targets", "v10_registry"]]
    rows, audit = load_response_rows(response_paths)
    rows = join_gold(rows, rel(cfg["data"]["v10_gold"]))
    b_audit = audit_b_capacity(rows, cfg["e1_b"]["strata"])
    quota_table = [
        {
            "stratum": name,
            "available_known_or_prescreen": int(b_audit["by_stratum"].get(name, 0)),
            "required": int(required),
            "gap": max(0, int(required) - int(b_audit["by_stratum"].get(name, 0))),
            "ready": bool(b_audit["quota_checks"].get(name, False)),
        }
        for name, required in cfg["e1_b"]["strata"].items()
    ]
    write_json(out / "E1_V3_B_CAPACITY_AUDIT.json", b_audit)
    write_csv(out / "E1_V3_B_QUOTA_TABLE.csv", quota_table)
    write_jsonl(out / "E1_V3_B_PANEL_ALL.jsonl", [])
    write_jsonl(out / "E1_V3_SPLIT_MANIFEST.jsonl", [])
    write_jsonl(out / "E1_V3_WRONG_Q_MAP.jsonl", [])
    progress("E1-B", 1, 1)
    return {"b_audit": b_audit, "b_quota_table": quota_table}


def phase_c(cfg: dict[str, Any], b_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    b_audit = b_audit or read_json(out / "E1_V3_B_CAPACITY_AUDIT.json", {})
    a_ready = read_json(out / "E1_V3_A_QUOTA_AUDIT.json", {})
    c_gate = {
        "can_run_c": False,
        "reason": "E1-C requires frozen A7500 Gold and frozen B detector/thresholds. Current run is dry-run/audit and B formal panel is not ready.",
        "a_pending_target_calls": int(a_ready.get("pending_target_calls", 0) or 0),
        "b_formal_panel_ready": bool(b_audit.get("formal_panel_ready", False)),
    }
    write_jsonl(out / "E1_V3_A_PREDICTIONS.jsonl", [])
    write_jsonl(out / "E1_V3_B_ANCHOR_PREDICTIONS.jsonl", [])
    write_jsonl(out / "E1_V3_C_PREDICTIONS.jsonl", [])
    write_csv(out / "E1_V3_METRICS_BY_SEED.csv", [])
    write_json(out / "E1_V3_THRESHOLDS.json", {"status": "NOT_FROZEN", "reason": c_gate["reason"]})
    write_json(out / "E1_V3_PAIRED_STATS.json", {"status": "NOT_RUN", "reason": c_gate["reason"]})
    progress("E1-C", 1, 1)
    return {"c_gate": c_gate}


def phase_report(cfg: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    report_dir = rel(cfg["data"]["public_report_dir"])
    p0 = state.get("p0") or read_json(out / "E1_V3_PROTOCOL_LOCK.json", {})
    a_audit = read_json(out / "E1_V3_A_QUOTA_AUDIT.json", {})
    b_audit = state.get("b_audit") or read_json(out / "E1_V3_B_CAPACITY_AUDIT.json", {})
    b_quota = state.get("b_quota_table") or read_csv_dicts(out / "E1_V3_B_QUOTA_TABLE.csv")
    c_gate = state.get("c_gate") or {"can_run_c": False}
    decision = decide(p0, a_audit, b_audit, c_gate)
    payload = {
        "protocol": cfg["experiment"]["protocol"],
        "runtime_commit": git_commit(),
        "decision": decision,
        "analysis": analysis_text(decision, a_audit, b_audit, c_gate),
        "a": {"quota_table": a_audit.get("quota_table", []), "pending_target_calls": a_audit.get("pending_target_calls", 0), "natural_metrics_reused_reference": summarize_a_metrics(cfg)},
        "b": {"quota_table": b_quota, "capacity_audit": b_audit},
        "c": {"gate_table": [c_gate]},
        "budget": p0.get("budget", budget_snapshot(cfg)),
        "data_audit": {"p0": p0, "a": a_audit, "b": b_audit},
    }
    write_json(out / "E1_V3_DECISION.json", decision)
    write_json(out / "E1_V3_RUN_FINGERPRINT.json", {"runtime_commit": git_commit(), "git_status": git_status_short(), "protocol": cfg["experiment"]["protocol"]})
    write_json(out / "E1_V3_FINAL_PAYLOAD.json", payload)
    write_reports(report_dir, payload)
    progress("REPORT", 1, 1)
    return payload


def summarize_a_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics = read_json(rel(cfg["data"]["v10_a_metrics"]), {})
    total = int(metrics.get("n", 0) or 0)
    by_model = metrics.get("by_model", {})
    lower = sum(int(v.get("lower_positive", 0) or 0) for v in by_model.values()) if isinstance(by_model, dict) else 0
    central = sum(int(v.get("positive", 0) or 0) for v in by_model.values()) if isinstance(by_model, dict) else 0
    upper = sum(int(v.get("upper_positive", 0) or 0) for v in by_model.values()) if isinstance(by_model, dict) else 0
    return {
        "existing_n": total,
        "lower_positive": lower,
        "central_positive": central,
        "upper_positive": upper,
        "central_wilson": wilson(central, total) if total else {"low": 0, "high": 0},
    }


def decide(p0: dict[str, Any], a_audit: dict[str, Any], b_audit: dict[str, Any], c_gate: dict[str, Any]) -> dict[str, Any]:
    p0_gate = p0.get("gate", "UNKNOWN")
    a_ready = all(row.get("cell_ready") for row in a_audit.get("quota_table", [])) and int(a_audit.get("pending_target_calls", 1) or 0) == 0
    b_ready = bool(b_audit.get("formal_panel_ready", False))
    if p0_gate.startswith("STOP"):
        code = "E1_V3_STOP_P0"
    elif not a_ready:
        code = "E1_V3_STOP_A7500_EXPANSION_PENDING"
    elif not b_ready:
        code = "E1_V3_STOP_B3200_PANEL_NOT_READY"
    elif not c_gate.get("can_run_c"):
        code = "E1_V3_STOP_C_NOT_READY"
    else:
        code = "E1_V3_READY_FOR_ANCHOR"
    return {
        "decision_code": code,
        "p0_gate": p0_gate,
        "a_gate": "PASS" if a_ready else "PENDING_API_EXPANSION",
        "b_gate": "PASS" if b_ready else "PENDING_PANEL_GOLD_AND_SYNTHESIS",
        "c_gate": "PASS" if c_gate.get("can_run_c") else "NOT_RUN",
    }


def analysis_text(decision: dict[str, Any], a_audit: dict[str, Any], b_audit: dict[str, Any], c_gate: dict[str, Any]) -> str:
    pending = int(a_audit.get("pending_target_calls", 0) or 0)
    b_counts = b_audit.get("by_stratum", {})
    return "\n\n".join(
        [
            "本轮按照 v3 冻结方案完成代码重构、报告归档、P0 dry-run、E1-A 7500 配额审计、E1-B 3200 容量审计和 E1-C 准入判定。报告不再沿用 v2 的“只能真实回答”假设，已允许 B 层后续进入受控合成，但自然发生率仍只由 E1-A 真实 target response 支撑。",
            f"E1-A 当前仍需补齐目标回答调用 {pending} 次；这些调用必须在 P0 clean commit 和预算硬上限生效后分批执行，不能为了追求结果好看而替换 q 或重复采样。",
            f"E1-B 真实候选预筛 stratum 计数为：stable+={b_counts.get('context_stable_positive', 0)}，stable-={b_counts.get('context_stable_negative', 0)}，critical+={b_counts.get('context_critical_positive', 0)}，hard-={b_counts.get('context_hard_negative', 0)}。该结果用于决定后续 Gold v5 与 counterfactual 合成补齐，不是正式 Anchor 结果。",
            f"E1-C 当前未运行，原因是：{c_gate.get('reason', '未满足 A/B 冻结条件')}。最终决策为 `{decision['decision_code']}`。",
        ]
    )


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(default_component: str = "all") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["p0", "a", "b", "c", "report", "all"], default="all")
    parser.add_argument("--component", choices=["all", "a", "b", "c"], default=default_component)
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--auto-continue-on-pass", action="store_true")
    parser.add_argument("--consume-anchor", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    phases = ["p0", "a", "b", "c", "report"] if args.phase == "all" else [args.phase]
    if args.component == "a" and args.phase == "all":
        phases = ["p0", "a", "report"]
    if args.component == "b" and args.phase == "all":
        phases = ["p0", "b", "report"]
    if args.component == "c" and args.phase == "all":
        phases = ["p0", "c", "report"]
    state: dict[str, Any] = {}
    for idx, phase in enumerate(phases, start=1):
        progress("TOTAL", idx - 1, len(phases))
        if phase == "p0":
            state["p0"] = phase_p0(cfg)
        elif phase == "a":
            state.update(phase_a(cfg))
        elif phase == "b":
            state.update(phase_b(cfg))
        elif phase == "c":
            state.update(phase_c(cfg, state.get("b_audit")))
        elif phase == "report":
            state["payload"] = phase_report(cfg, state)
        progress("TOTAL", idx, len(phases))
    print(f"v3 执行完成：component={args.component} phase={args.phase} output={rel(cfg['data']['output_dir'])}")


if __name__ == "__main__":
    main()
