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

from frauddistill.e1_real_v2.io import file_sha256, read_json, write_json, write_jsonl
from frauddistill.e1_real_v2.registry_v2 import join_v10_gold, load_real_registry, load_v81_p2_targets
from frauddistill.e1_real_v2.reporting_v2 import write_reports
from frauddistill.e1_real_v2.strata import capacity_decision, prescreen_existing_capacity
from frauddistill.e1_v10.metrics import wilson


CONFIG_PATH = ROOT / "configs" / "experiments" / "e1_real_triad_v2.yaml"


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
    width = 28
    filled = int(width * done / max(1, total))
    bar = "#" * filled + "." * (width - filled)
    pct = 100 * done / max(1, total)
    print(f"[{name}] [{bar}] {done}/{total} {pct:5.1f}%", flush=True)


def phase_p0(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = rel(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        key: {
            "path": str(rel(value)),
            "exists": rel(value).exists(),
            "sha256": file_sha256(rel(value)) if rel(value).exists() and rel(value).is_file() else None,
        }
        for key, value in cfg["data"].items()
        if key.startswith("v") or key.endswith("consensus")
    }
    lock = {
        "protocol": cfg["experiment"]["protocol"],
        "baseline_commit": cfg["experiment"]["baseline_commit"],
        "runtime_commit": git_commit(),
        "git_status_at_lock": git_status_short(),
        "requested_concurrency": cfg["api"]["requested_concurrency"],
        "models": cfg["models"],
        "sources": sources,
        "forbidden_items": [
            "source_derived formal panel rows",
            "deterministic Gold v4 copy",
            "q-only/y-only constant scores",
            "wrong_penalty or condition features in wrong-q",
        ],
    }
    write_json(out_dir / "E1_REAL_V2_PROTOCOL_LOCK.json", lock)
    progress("P0", 1, 1)
    return lock


def phase_registry(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = rel(cfg["data"]["output_dir"])
    v10, v10_audit = load_real_registry(rel(cfg["data"]["v10_registry"]))
    joined_v10 = join_v10_gold(v10, rel(cfg["data"]["v10_gold"]))
    v81, v81_audit = load_v81_p2_targets(rel(cfg["data"]["v81_p2_targets"]))
    rows = dedupe_rows(joined_v10 + v81)
    write_jsonl(out_dir / "E1_REAL_V2_TARGET_REGISTRY.jsonl", rows)
    provenance = {
        "registry_rows": len(rows),
        "source_derived_rows": sum(1 for r in rows if r.get("is_source_derived")),
        "real_target_response_rows": sum(1 for r in rows if r.get("is_real_target_response")),
        "real_target_response_ratio": sum(1 for r in rows if r.get("is_real_target_response")) / max(1, len(rows)),
        "by_provider": dict(Counter(r["target_provider"] for r in rows)),
        "by_source_dataset": dict(Counter(r["source_dataset"] for r in rows)),
        "by_language": dict(Counter(r["language"] for r in rows)),
        "v10_audit": v10_audit,
        "v81_audit": v81_audit,
    }
    write_json(out_dir / "E1_REAL_V2_PROVENANCE_AUDIT.json", provenance)
    write_json(
        out_dir / "E1_REAL_V2_REUSE_MANIFEST.json",
        {"reused_without_api": len(rows), "new_api_calls": 0, "api_policy_concurrency": cfg["api"]["requested_concurrency"]},
    )
    progress("REGISTRY", 1, 1)
    return {"rows": rows, "provenance": provenance}


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = (row.get("normalized_q_hash"), row.get("y_hash"), row.get("target_provider"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def phase_capacity(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = rel(cfg["data"]["output_dir"])
    candidates, capacity = prescreen_existing_capacity(rows)
    decision = capacity_decision(capacity, cfg["panel"]["strata"], int(cfg["panel"]["amber_min_per_stratum"]))
    write_jsonl(out_dir / "E1_REAL_V2_REAL_CANDIDATES_PRESCREEN.jsonl", candidates)
    write_json(out_dir / "E1_REAL_V2_STRATA_CAPACITY.json", capacity)
    write_json(out_dir / "E1_REAL_V2_CAPACITY_DECISION.json", decision)
    progress("CAPACITY", 1, 1)
    return {"candidates": candidates, "capacity": capacity, "capacity_decision": decision}


def phase_report(cfg: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    out_dir = rel(cfg["data"]["output_dir"])
    report_dir = rel(cfg["data"]["public_report_dir"])
    a_metrics = read_json(rel(cfg["data"]["v10_a_metrics"]), {})
    provenance = state.get("provenance") or read_json(out_dir / "E1_REAL_V2_PROVENANCE_AUDIT.json", {})
    capacity = state.get("capacity") or read_json(out_dir / "E1_REAL_V2_STRATA_CAPACITY.json", {})
    cap_decision = state.get("capacity_decision") or read_json(out_dir / "E1_REAL_V2_CAPACITY_DECISION.json", {})
    decision_code = final_decision(provenance, capacity, cap_decision)
    payload = build_payload(cfg, a_metrics, provenance, capacity, cap_decision, decision_code)
    write_json(out_dir / "E1_REAL_V2_FINAL_PAYLOAD.json", payload)
    files = write_reports(report_dir, payload)
    write_json(out_dir / "E1_REAL_V2_REPORT_MANIFEST.json", {"reports": files})
    progress("REPORT", 1, 1)
    return payload


def final_decision(provenance: dict[str, Any], capacity: dict[str, Any], cap_decision: dict[str, Any]) -> str:
    if provenance.get("source_derived_rows", 0) != 0:
        return "E1_REAL_V2_STOP_SOURCE_DERIVED_LEAKAGE"
    if provenance.get("real_target_response_ratio", 0) < 1.0:
        return "E1_REAL_V2_STOP_NON_REAL_TARGET_RESPONSE"
    if cap_decision.get("decision") != "GO_FULL_PANEL":
        return "E1_REAL_V2_STOP_CONTEXT_CAPACITY"
    if not capacity.get("formal_gold_v5"):
        return "E1_REAL_V2_STOP_FORMAL_GOLD_V5_REQUIRED"
    return "E1_REAL_V2_READY_FOR_FORMAL_PANEL"


def build_payload(
    cfg: dict[str, Any],
    a_metrics: dict[str, Any],
    provenance: dict[str, Any],
    capacity: dict[str, Any],
    cap_decision: dict[str, Any],
    decision_code: str,
) -> dict[str, Any]:
    a_table = natural_a_table(a_metrics)
    b_table = b_capacity_table(capacity, cfg["panel"]["strata"])
    analysis = make_analysis(decision_code, provenance, capacity, cap_decision)
    return {
        "protocol": cfg["experiment"]["protocol"],
        "runtime_commit": git_commit(),
        "decision": {
            "decision_code": decision_code,
            "a_gate": "PASS_REUSED_FROZEN_A" if a_table else "STOP_A_METRICS_MISSING",
            "b_gate": cap_decision.get("decision", "UNKNOWN"),
            "c_gate": "NOT_RUN_BECAUSE_B_NOT_FORMAL" if decision_code != "E1_REAL_V2_READY_FOR_FORMAL_PANEL" else "PENDING",
            "gold_gate": "NOT_RUN_FORMAL_GOLD_V5_REQUIRED" if not capacity.get("formal_gold_v5") else "PASS",
        },
        "analysis": analysis,
        "a": {"main_table": a_table, "raw": a_metrics},
        "b": {"main_table": b_table, "capacity": capacity, "decision": cap_decision},
        "c": {"main_table": [{"metric": "status", "value": "未运行", "reason": "B 层未形成 formal Gold v5 面板，按协议不得迁移 C。"}]},
        "gold": {
            "formal_gold_v5_completed": bool(capacity.get("formal_gold_v5")),
            "deterministic_gold_used": False,
            "legacy_labels_only_for_prescreen": True,
        },
        "provenance": provenance,
        "statistics": {
            "bootstrap_iterations_planned": cfg["statistics"]["bootstrap_iterations"],
            "seeds_planned": cfg["statistics"]["seeds"],
            "formal_tests_run": False,
            "reason": "容量门控未通过，未进入正式面板训练和统计检验。",
        },
        "bias": {
            "main_failure": "真实回答候选池中的正例容量，尤其 context_critical_positive，距离 1200 条正式 case-control 面板要求不足。",
            "paper_position": "该轮不能作为 q+y 强通过主结果，只能作为严格实证审计和后续补采样依据。",
        },
        "budget": {
            "total_cny": 0.0,
            "new_api_calls": 0,
            "requested_concurrency": cfg["api"]["requested_concurrency"],
            "note": "本次执行先复用既有真实 API 缓存做准入审计；未通过容量门控，因此未继续消耗 API 生成 formal Gold v5/全量面板。",
        },
        "closeout": {
            "archive_policy": "历史 reports/outputs 已归档到 archive/pre_e1_real_triad_v2_*；旧 synthetic 结果不进入正式报告。",
            "next_action": "若继续推进，需要按 v2 协议补采真实 target response，并完成 Gold v5 双评审/裁决后再训练 B/C。",
            "git_status_at_report": git_status_short(),
        },
    }


def natural_a_table(a_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    if not a_metrics:
        return []
    rows = []
    total = int(a_metrics.get("n", a_metrics.get("total", 0)) or 0)
    by_model = a_metrics.get("by_model", {})
    if total and isinstance(by_model, dict) and by_model:
        lower = sum(int(v.get("lower_positive", 0) or 0) for v in by_model.values() if isinstance(v, dict))
        central = sum(int(v.get("positive", 0) or 0) for v in by_model.values() if isinstance(v, dict))
        upper = sum(int(v.get("upper_positive", 0) or 0) for v in by_model.values() if isinstance(v, dict))
        for key, count in [("lower", lower), ("central", central), ("upper", upper)]:
            ci = wilson(count, total)
            rows.append({"口径": key, "正例数": count, "样本数": total, "发生率": count / total, "Wilson_low": ci["low"], "Wilson_high": ci["high"]})
        return rows
    for key in ["lower", "central", "upper"]:
        count = int(a_metrics.get(f"{key}_positive", a_metrics.get(f"positive_{key}", 0)) or 0)
        if total:
            ci = wilson(count, total)
            rows.append({"口径": key, "正例数": count, "样本数": total, "发生率": count / total, "Wilson_low": ci["low"], "Wilson_high": ci["high"]})
    if rows:
        return rows
    if "main_table" in a_metrics:
        return a_metrics["main_table"]
    return [{"metric": k, "value": v} for k, v in a_metrics.items() if isinstance(v, (int, float, str))][:12]


def b_capacity_table(capacity: dict[str, Any], required: dict[str, int]) -> list[dict[str, Any]]:
    counts = capacity.get("by_stratum", {})
    return [
        {
            "stratum": name,
            "现有候选数": int(counts.get(name, 0)),
            "正式需求": int(required_count),
            "缺口": max(0, int(required_count) - int(counts.get(name, 0))),
            "满足正式门槛": int(counts.get(name, 0)) >= int(required_count),
        }
        for name, required_count in required.items()
    ]


def make_analysis(decision_code: str, provenance: dict[str, Any], capacity: dict[str, Any], cap_decision: dict[str, Any]) -> str:
    counts = capacity.get("by_stratum", {})
    parts = [
        f"本轮执行采用 `E1-REAL-TRIAD-v2.0` 协议，正式候选池仅保留 Qwen/DeepSeek 的真实目标模型回答；审计显示 source_derived 行数为 {provenance.get('source_derived_rows', 0)}，真实目标回答比例为 {provenance.get('real_target_response_ratio', 0):.4f}。",
        f"B 层容量预筛的四个 stratum 计数为：stable positive={counts.get('context_stable_positive', 0)}，stable negative={counts.get('context_stable_negative', 0)}，critical positive={counts.get('context_critical_positive', 0)}，hard negative={counts.get('context_hard_negative', 0)}。与正式需求 420/420/180/180 相比，正例容量仍是主要瓶颈。",
        "由于当前容量审计尚未形成 formal Gold v5 双评审/裁决面板，本轮没有继续训练 q-only、y-only、q+y 和 wrong-q，也没有进入 C 层迁移。这样处理会牺牲“好看”的指标，但避免再次出现模板面板、确定性标签或模式泄漏导致的不可用结果。",
        f"最终决策为 `{decision_code}`；该结果可以作为后续补采样和 API 投入的准入依据，但不能作为论文主表中的 STRONG PASS 结果。",
    ]
    if cap_decision.get("decision") == "GO_FULL_PANEL":
        parts.append("容量门控已显示可进入完整面板，但仍需要先完成 Gold v5 后才能训练正式检测器。")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["p0", "registry", "capacity", "report", "all"], default="all")
    parser.add_argument("--confirm-budget", action="store_true")
    parser.add_argument("--auto-continue-on-pass", action="store_true")
    parser.add_argument("--consume-anchor", action="store_true")
    parser.add_argument("--resume-from", default="")
    args = parser.parse_args()
    cfg = load_config()
    state: dict[str, Any] = {}
    phases = ["p0", "registry", "capacity", "report"] if args.phase == "all" else [args.phase]
    for idx, phase in enumerate(phases, start=1):
        progress("TOTAL", idx - 1, len(phases))
        if phase == "p0":
            state["lock"] = phase_p0(cfg)
        elif phase == "registry":
            reg = phase_registry(cfg)
            state.update(reg)
        elif phase == "capacity":
            rows = state.get("rows")
            if rows is None:
                rows = load_registry_from_disk(cfg)
            cap = phase_capacity(cfg, rows)
            state.update(cap)
        elif phase == "report":
            payload = phase_report(cfg, state)
            state["payload"] = payload
        progress("TOTAL", idx, len(phases))
    print(f"完成 phase={args.phase}；输出目录：{rel(cfg['data']['output_dir'])}")


def load_registry_from_disk(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from frauddistill.e1_real_v2.io import read_jsonl

    return read_jsonl(rel(cfg["data"]["output_dir"]) / "E1_REAL_V2_TARGET_REGISTRY.jsonl")


if __name__ == "__main__":
    main()
