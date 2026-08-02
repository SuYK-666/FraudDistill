from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v3.api_executor import (
    cache_index,
    execute_json_tasks,
    execute_tasks,
    request_fingerprint,
)
from frauddistill.e1_final_v3.budget import budget_snapshot
from frauddistill.e1_final_v3.c_replay_v31 import c_block, directional, paired_bootstrap_gain
from frauddistill.e1_final_v3.detector_v31 import (
    STRATA,
    VIEWS,
    aggregate_seeds,
    cluster_bootstrap_macro_f1,
    panel_rows_to_eval,
    paired_mcnemar,
    run_model_dev_cv,
    run_seed,
    shortcut_audits,
)
from frauddistill.e1_final_v3.gold_v31 import (
    adjudication_task,
    a7500_registry,
    consensus_from_votes,
    gold_judge_task,
    needs_adjudication,
    reuse_v10_gold,
    v31_response_id,
    votes_by_response,
)
from frauddistill.e1_final_v3.io import file_sha256, read_json, read_jsonl, write_csv, write_json, write_jsonl
from frauddistill.e1_final_v3.panel_builder_v31 import (
    STRATUM_QUOTAS,
    assemble_panel,
    build_real_pool,
    classify_rows,
    intended_stratum,
    panel_row_from_generated,
    select_real_panel,
    split_by_family,
    synthetic_tasks_for_deficit,
)
from frauddistill.e1_final_v3.registry import build_v31_a_manifest, load_v31_reusable_roleplay
from frauddistill.e1_final_v3.reporting import block, table
from frauddistill.e1_final_v3.stats_v31 import a_behavior_stats
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
        concurrency_by_provider=target_concurrency(cfg),
        pricing=cfg.get("pricing_cny_per_million_tokens"),
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
        concurrency_by_provider=target_concurrency(cfg),
        pricing=cfg.get("pricing_cny_per_million_tokens"),
    )
    result["pending_before_batch"] = len(pending)
    result["selected_for_batch"] = len(selected)
    write_json(out / "E1_V31_A_GENERATE_RESULT.json", result)
    progress("GENERATE", 1, 1)
    return result


def target_concurrency(cfg: dict[str, Any]) -> dict[str, int]:
    api = cfg.get("api", {})
    return {
        "qwen": int(api.get("effective_qwen_concurrency", 1) or 1),
        "deepseek": int(api.get("effective_deepseek_concurrency", 1) or 1),
    }


def gold_concurrency(cfg: dict[str, Any]) -> dict[str, int]:
    api = cfg.get("api", {})
    return {
        "qwen": int(api.get("effective_qwen_concurrency", 1) or 1),
        "deepseek": int(api.get("effective_deepseek_concurrency", 1) or 1),
    }


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


def load_ok_new_responses(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out = rel(cfg["data"]["output_dir"])
    return [r for r in read_jsonl(out / "E1_V31_A_TARGET_RESPONSES.jsonl") if r.get("status") == "ok" and r.get("text")]

def phase_gold(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    responses = load_ok_new_responses(cfg)
    pending = read_jsonl(out / "E1_V31_A_GOLD_REUSED_PENDING.jsonl")
    pending_ids = {v31_response_id(r) for r in pending}
    responses = responses + pending
    limit_q = int(args.limit_q or 0)
    if limit_q:
        responses = responses[: limit_q * 2]
    tasks = []
    for r in responses:
        ph = "E1-A-gold-v31-reused" if v31_response_id(r) in pending_ids else "E1-A-gold-v31"
        tasks.extend(gold_judge_task(r, judge, cfg, phase=ph) for judge in ["judge_a", "judge_b"])
    result = execute_json_tasks(
        tasks,
        output_path=out / "E1_V31_A_GOLD_VOTES.jsonl",
        ledger_path=out / "E1_V31_BUDGET_LEDGER.jsonl",
        limits=cfg["budget"],
        run_api=args.run_api,
        confirm_budget=args.confirm_budget,
        git_clean=git_clean(),
        concurrency_by_provider=gold_concurrency(cfg),
        pricing=cfg.get("pricing_cny_per_million_tokens"),
    )
    result["tasks_built"] = len(tasks)
    write_json(out / "E1_V31_A_GOLD_RESULT.json", result)
    progress("GOLD", 1, 1)
    return result


def phase_adjudicate(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    pending = read_jsonl(out / "E1_V31_A_GOLD_REUSED_PENDING.jsonl")
    pending_ids = {v31_response_id(r) for r in pending}
    responses = load_ok_new_responses(cfg) + pending
    votes = read_jsonl(out / "E1_V31_A_GOLD_VOTES.jsonl")
    by_resp = votes_by_response(votes)
    tasks = []
    for row in responses:
        rid = v31_response_id(row)
        va = by_resp.get(rid, {}).get("judge_a")
        vb = by_resp.get(rid, {}).get("judge_b")
        if va is not None and vb is not None and needs_adjudication(va, vb):
            ph = "E1-A-gold-v31-reused" if rid in pending_ids else "E1-A-gold-v31"
            tasks.append(adjudication_task(row, va, vb, cfg, phase=ph))
    result = execute_json_tasks(
        tasks,
        output_path=out / "E1_V31_A_GOLD_ADJUDICATION.jsonl",
        ledger_path=out / "E1_V31_BUDGET_LEDGER.jsonl",
        limits=cfg["budget"],
        run_api=args.run_api,
        confirm_budget=args.confirm_budget,
        git_clean=git_clean(),
        concurrency_by_provider={"qwen": int(cfg["api"].get("effective_adjudicator_concurrency", 8) or 8)},
        pricing=cfg.get("pricing_cny_per_million_tokens"),
    )
    result["adjudication_tasks_built"] = len(tasks)
    write_json(out / "E1_V31_A_ADJUDICATION_RESULT.json", result)
    progress("ADJUDICATE", 1, 1)
    return result


def phase_freezer(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    responses = load_ok_new_responses(cfg)
    votes = read_jsonl(out / "E1_V31_A_GOLD_VOTES.jsonl")
    adjud = read_jsonl(out / "E1_V31_A_GOLD_ADJUDICATION.jsonl")
    consensus, quality = consensus_from_votes(responses, votes, adjud)
    write_jsonl(out / "E1_V31_A_GOLD_CONSENSUS.jsonl", consensus)
    write_json(out / "E1_V31_A_GOLD_QUALITY.json", quality)
    reused_pairs, _audit = load_v31_reusable_roleplay(rel(cfg["data"]["v10_registry"]))
    reused = [row for pair in reused_pairs.values() for row in pair.values()]
    reused_gold, reuse_stats = reuse_v10_gold(reused, rel(cfg["data"]["v10_gold"]), rel(cfg["data"]["v10_registry"]))
    pending_reuse = [r for r in reused_gold if r["gold_status"] == "REUSE_REJECTED" and r.get("reuse_reason") == "missing_gold"]
    write_jsonl(out / "E1_V31_A_GOLD_REUSED_PENDING.jsonl", pending_reuse)
    if pending_reuse:
        reuse_votes = [v for v in votes if v.get("phase") == "E1-A-gold-v31-reused"]
        reuse_adjud = [a for a in adjud if a.get("phase") == "E1-A-gold-v31-reused"]
        reuse_consensus, _ = consensus_from_votes(pending_reuse, reuse_votes, reuse_adjud)
        reuse_map = {str(c["response_id"]): c for c in reuse_consensus}
        for row in reused_gold:
            c = reuse_map.get(str(row.get("response_id", "")))
            if c and c.get("gold_status") == "KNOWN":
                row["gold_status"] = "KNOWN_REUSED"
                row["gold_lower"] = c["gold_lower"]
                row["gold_central"] = c["gold_central"]
                row["gold_upper"] = c["gold_upper"]
                row["gold_uncertain"] = c["gold_uncertain"]
                row["judge_a_positive"] = c["judge_a_positive"]
                row["judge_b_positive"] = c["judge_b_positive"]
    registry = a7500_registry(responses, reused_gold, consensus)
    write_jsonl(out / "E1_V31_A_REGISTRY_FROZEN.jsonl", registry)
    stats = a_behavior_stats(registry, cfg)
    write_json(out / "E1_V31_A_BEHAVIOR_STATS.json", stats)
    coverage = {
        "new_responses": len(responses),
        "new_gold_known": sum(1 for c in consensus if c["gold_status"] == "KNOWN"),
        "reused_responses": len(reused_gold),
        "reused_gold_known": sum(1 for r in reused_gold if r["gold_status"] == "KNOWN_REUSED"),
        "reuse_stats": reuse_stats,
        "registry_rows": len(registry),
        "registry_with_gold": sum(1 for r in registry if int(r.get("gold_central", -1)) >= 0),
        "gold_quality": quality,
    }
    write_json(out / "E1_V31_A_GOLD_COVERAGE.json", coverage)
    progress("FREEZE", 1, 1)
    return coverage


def safe_prompts(cfg: dict[str, Any]) -> list[str]:
    rows = read_jsonl(rel(cfg["data"]["or_bench_prompts"]))
    return [str(r.get("user_query") or r.get("q") or "") for r in rows if r.get("user_query") or r.get("q")][:300]


def source_derived_rows(cfg: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Open-source control: q = Fraud-R1 official prompt, y = Fraud-R1 raw 'generated text'.
    Explicitly provenance-tagged as source_derived_open_control, never mixed with real y."""
    out = []
    prompts = read_jsonl(rel(cfg["data"]["fraudr1_raw_prompts"]))
    for row in prompts:
        if len(out) >= count:
            break
        if "FP-base" not in str(row.get("source_file", "")):
            continue
        q = row.get("user_query") or ""
        y = row.get("generated_text") or row.get("generated text") or row.get("raw_data") or ""
        if not q or not y:
            continue
        cid = str(row.get("id", "")).lower()
        out.append(
            {
                "response_id": f"source-derived|{cid}",
                "prompt_instance_id": f"source-derived|{cid}|stage0|assistant",
                "canonical_case_id": cid,
                "stage_id": 0,
                "scenario": "assistant",
                "q_private": q,
                "y_private": y,
                "target_provider": "open-control",
                "language": str(row.get("language", "")).lower()[:2],
                "fraud_category": str(row.get("fraud_category", "")),
                "source_dataset": "Fraud-R1",
                "provenance": "source_derived_open_control",
                "source_run": "source-derived-open-control",
                "gold_status": "PENDING_GOLD",
            }
        )
    return out


def phase_b_build_panel(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    seed = int(cfg["experiment"]["seed"])
    registry_path = out / "E1_V31_A_REGISTRY_FROZEN.jsonl"
    if not registry_path.exists():
        return {"status": "STOP_A_NOT_FROZEN", "reason": "run --phase freeze first"}
    real_pool, pool_audit = build_real_pool(registry_path, rel(cfg["data"]["v10_pressure_gold"]))
    classified = classify_rows(real_pool)
    real_selected, real_audit = select_real_panel(classified, seed)
    real_by_stratum = {}
    for r in real_selected:
        real_by_stratum[r["stratum"]] = real_by_stratum.get(r["stratum"], 0) + 1
    deficits = {s: max(0, STRATUM_QUOTAS[s] - real_by_stratum.get(s, 0)) for s in STRATA}
    alloc: dict[str, int] = {}
    alloc["context_critical_positive"] = min(320, deficits["context_critical_positive"])
    alloc["context_hard_negative"] = min(320, deficits["context_hard_negative"])
    remaining = 1200 - alloc["context_critical_positive"] - alloc["context_hard_negative"]
    alloc["context_stable_positive"] = min(remaining, deficits["context_stable_positive"])
    alloc["context_stable_negative"] = min(remaining, deficits["context_stable_negative"])
    tasks, synth_plan = synthetic_tasks_for_deficit(alloc, real_pool, safe_prompts(cfg), cfg, seed)
    result = execute_tasks(
        tasks,
        output_path=out / "E1_V31_B_SYNTHETIC_RESPONSES.jsonl",
        ledger_path=out / "E1_V31_BUDGET_LEDGER.jsonl",
        limits=cfg["budget"],
        run_api=args.run_api,
        confirm_budget=args.confirm_budget,
        git_clean=git_clean(),
        concurrency_by_provider=target_concurrency(cfg),
        pricing=cfg.get("pricing_cny_per_million_tokens"),
    )
    generated = [r for r in read_jsonl(out / "E1_V31_B_SYNTHETIC_RESPONSES.jsonl") if r.get("status") == "ok" and r.get("text")]
    task_map = {str(t["cf_task_id"]): t for t in tasks}
    synth_rows = [panel_row_from_generated(task_map[str(r.get("cf_task_id", ""))], r) for r in generated if str(r.get("cf_task_id", "")) in task_map]
    write_jsonl(out / "E1_V31_B_PANEL_SYNTHETIC.jsonl", synth_rows)
    derived = source_derived_rows(cfg, 400)
    write_jsonl(out / "E1_V31_B_PANEL_SOURCE_DERIVED.jsonl", derived)
    write_jsonl(out / "E1_V31_B_PANEL_SYNTHETIC_TASKS.jsonl", tasks)
    payload = {
        "status": result.get("status", "DONE"),
        "real_pool_audit": pool_audit,
        "real_selection_audit": real_audit,
        "deficits": deficits,
        "synthetic_allocation": alloc,
        "synthetic_plan": synth_plan,
        "synthetic_generation_result": result,
        "generated_rows": len(generated),
        "synthetic_panel_rows": len(synth_rows),
        "source_derived_rows": len(derived),
    }
    write_json(out / "E1_V31_B_PANEL_BUILD.json", payload)
    progress("B-BUILD", 1, 1)
    return payload

def phase_b_gold(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    rows = [r for r in read_jsonl(out / "E1_V31_B_PANEL_SYNTHETIC.jsonl")] + [r for r in read_jsonl(out / "E1_V31_B_PANEL_SOURCE_DERIVED.jsonl")]
    tasks = [gold_judge_task(r, judge, cfg, phase="E1-B-gold-v31") for r in rows for judge in ["judge_a", "judge_b"]]
    result = execute_json_tasks(
        tasks,
        output_path=out / "E1_V31_B_GOLD_VOTES.jsonl",
        ledger_path=out / "E1_V31_BUDGET_LEDGER.jsonl",
        limits=cfg["budget"],
        run_api=args.run_api,
        confirm_budget=args.confirm_budget,
        git_clean=git_clean(),
        concurrency_by_provider=gold_concurrency(cfg),
        pricing=cfg.get("pricing_cny_per_million_tokens"),
    )
    result["tasks_built"] = len(tasks)
    write_json(out / "E1_V31_B_GOLD_RESULT.json", result)
    progress("B-GOLD", 1, 1)
    return result


def phase_b_adjudicate(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    rows = [r for r in read_jsonl(out / "E1_V31_B_PANEL_SYNTHETIC.jsonl")] + [r for r in read_jsonl(out / "E1_V31_B_PANEL_SOURCE_DERIVED.jsonl")]
    votes = read_jsonl(out / "E1_V31_B_GOLD_VOTES.jsonl")
    by_resp = votes_by_response(votes)
    tasks = []
    for row in rows:
        rid = v31_response_id(row)
        va = by_resp.get(rid, {}).get("judge_a")
        vb = by_resp.get(rid, {}).get("judge_b")
        if va is not None and vb is not None and needs_adjudication(va, vb):
            tasks.append(adjudication_task(row, va, vb, cfg, phase="E1-B-gold-v31"))
    result = execute_json_tasks(
        tasks,
        output_path=out / "E1_V31_B_GOLD_ADJUDICATION.jsonl",
        ledger_path=out / "E1_V31_BUDGET_LEDGER.jsonl",
        limits=cfg["budget"],
        run_api=args.run_api,
        confirm_budget=args.confirm_budget,
        git_clean=git_clean(),
        concurrency_by_provider={"qwen": int(cfg["api"].get("effective_adjudicator_concurrency", 8) or 8)},
        pricing=cfg.get("pricing_cny_per_million_tokens"),
    )
    result["adjudication_tasks_built"] = len(tasks)
    write_json(out / "E1_V31_B_ADJUDICATION_RESULT.json", result)
    progress("B-ADJUDICATE", 1, 1)
    return result


def phase_b_consensus(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    rows = [r for r in read_jsonl(out / "E1_V31_B_PANEL_SYNTHETIC.jsonl")] + [r for r in read_jsonl(out / "E1_V31_B_PANEL_SOURCE_DERIVED.jsonl")]
    votes = read_jsonl(out / "E1_V31_B_GOLD_VOTES.jsonl")
    adjud = read_jsonl(out / "E1_V31_B_GOLD_ADJUDICATION.jsonl")
    consensus, quality = consensus_from_votes(rows, votes, adjud)
    write_jsonl(out / "E1_V31_B_GOLD_CONSENSUS.jsonl", consensus)
    write_json(out / "E1_V31_B_GOLD_QUALITY.json", quality)
    by_rid = {str(c["response_id"]): c for c in consensus}
    for row in rows:
        c = by_rid.get(v31_response_id(row), {})
        row["gold_status"] = c.get("gold_status", "MISSING")
        row["gold_lower"] = c.get("gold_lower", -1)
        row["gold_central"] = c.get("gold_central", -1)
        row["gold_upper"] = c.get("gold_upper", -1)
        row["gold_uncertain"] = c.get("gold_uncertain", False)
        row["judge_a_positive"] = c.get("judge_a_positive", -1)
        row["judge_b_positive"] = c.get("judge_b_positive", -1)
    write_jsonl(out / "E1_V31_B_PANEL_SYNTHETIC.jsonl", [r for r in rows if r.get("provenance") == "counterfactual_synthetic"])
    write_jsonl(out / "E1_V31_B_PANEL_SOURCE_DERIVED.jsonl", [r for r in rows if r.get("provenance") == "source_derived_open_control"])
    progress("B-CONSENSUS", 1, 1)
    return {"consensus_rows": len(consensus), "quality": quality}


def phase_validate_panel(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    seed = int(cfg["experiment"]["seed"])
    registry_path = out / "E1_V31_A_REGISTRY_FROZEN.jsonl"
    real_pool, pool_audit = build_real_pool(registry_path, rel(cfg["data"]["v10_pressure_gold"]))
    classified = classify_rows(real_pool)
    real_selected, real_audit = select_real_panel(classified, seed)
    synthetic = read_jsonl(out / "E1_V31_B_PANEL_SYNTHETIC.jsonl")
    derived = read_jsonl(out / "E1_V31_B_PANEL_SOURCE_DERIVED.jsonl")
    accepted_synth = []
    rejects = []
    for row in synthetic:
        if not str(row.get("gold_status", "")).startswith("KNOWN"):
            rejects.append({**row, "reject_reason": "gold_not_known"})
            continue
        stratum = classify_stratum(row)
        row["stratum"] = stratum
        if stratum == intended_stratum(row.get("cf_family", "")):
            accepted_synth.append(row)
        else:
            rejects.append({**row, "reject_reason": f"stratum_mismatch:{stratum}"})
    accepted_derived = []
    for row in derived:
        if str(row.get("gold_status", "")).startswith("KNOWN"):
            row["stratum"] = classify_stratum(row)
            accepted_derived.append(row)
        else:
            rejects.append({**row, "reject_reason": "gold_not_known"})
    panel, panel_audit = assemble_panel(real_selected, accepted_synth, accepted_derived)
    for row in panel:
        row["stratum"] = row.get("stratum") or classify_stratum(row)
    write_jsonl(out / "E1_V31_B_PANEL_ALL.jsonl", panel)
    write_jsonl(out / "E1_V31_B_PANEL_REJECTS.jsonl", rejects)
    splits, split_audit = split_by_family(panel, cfg["e1_b"]["splits"], seed)
    for name, rows in splits.items():
        write_jsonl(out / f"E1_V31_B_PANEL_{name.upper().replace('-', '_')}.jsonl", rows)
    audit = {
        "real_selection": real_audit,
        "synthetic_accepted": len(accepted_synth),
        "synthetic_rejected": sum(1 for r in rejects if r.get("provenance") == "counterfactual_synthetic"),
        "source_derived_accepted": len(accepted_derived),
        "rejects_total": len(rejects),
        "panel": panel_audit,
        "splits": split_audit,
        "formal_panel_ready": bool(panel_audit.get("formal_panel_ready")) and split_audit.get("cross_split_families", 1) == 0,
    }
    write_json(out / "E1_V31_B_PANEL_AUDIT.json", audit)
    progress("B-PANEL", 1, 1)
    return audit


def phase_model_dev(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    dev = read_jsonl(out / "E1_V31_B_PANEL_MODEL_DEV.jsonl")
    seeds = cfg["statistics"]["seeds"]
    folds = int(cfg["e1_b"].get("cv_folds", 5))
    results = []
    for view in VIEWS:
        for seed in seeds:
            results.append(run_model_dev_cv(dev, view, int(seed), folds=folds))
    write_json(out / "E1_V31_B_MODEL_DEV_RESULTS.json", {"results": results})
    progress("MODEL-DEV", 1, 1)
    return {"results": results}


def phase_calibration(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    dev = read_jsonl(out / "E1_V31_B_PANEL_MODEL_DEV.jsonl")
    cal = read_jsonl(out / "E1_V31_B_PANEL_CALIBRATION.jsonl")
    seeds = cfg["statistics"]["seeds"]
    results = []
    for view in VIEWS:
        for seed in seeds:
            res = run_seed(dev, cal, [], view, int(seed), cfg["e1_b"])
            results.append({"mode": view, "seed": seed, "threshold": res["threshold"], "cal_macro_f1": res["cal_macro_f1_at_threshold"]})
    write_json(out / "E1_V31_B_CALIBRATION_RESULTS.json", {"results": results})
    progress("CALIBRATION", 1, 1)
    return {"results": results}


def phase_freeze_b(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import hashlib

    out = rel(cfg["data"]["output_dir"])
    dev = read_jsonl(out / "E1_V31_B_PANEL_MODEL_DEV.jsonl")
    cal = read_jsonl(out / "E1_V31_B_PANEL_CALIBRATION.jsonl")
    main_seed = 13
    thresholds = {}
    freeze = {"protocol": cfg["experiment"]["protocol"], "main_seed": main_seed, "views": {}}
    for view in VIEWS:
        threshold = run_seed(dev, cal, [], view, main_seed, cfg["e1_b"])["threshold"]
        thresholds[view] = threshold
        freeze["views"][view] = {"threshold": threshold}
    freeze["shortcut_audits"] = shortcut_audits(dev, seed=main_seed)
    freeze["thresholds_hash"] = hashlib.sha256(str(thresholds).encode("utf-8")).hexdigest()
    write_json(out / "E1_V31_B_FREEZE.json", freeze)
    from frauddistill.e1_final_v3.detector_v31 import ViewDetector
    import joblib

    d_rows, d_labels = panel_rows_to_eval(dev)
    for view in VIEWS:
        detector = ViewDetector(mode=view, seed=main_seed, C=float(cfg["e1_b"].get("detector_C", 1.0)), max_features=int(cfg["e1_b"].get("max_features", 60000)))
        detector.fit(d_rows, d_labels)
        joblib.dump(detector, out / f"E1_V31_B_MODEL_{view.replace('+', 'p')}.joblib")
    progress("FREEZE-B", 1, 1)
    return freeze


def phase_anchor(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    out = rel(cfg["data"]["output_dir"])
    if not args.consume_anchor:
        return {"status": "STOP_ANCHOR_REQUIRES_CONSUME", "reason": "--consume-anchor required for one-time anchor consumption"}
    dev = read_jsonl(out / "E1_V31_B_PANEL_MODEL_DEV.jsonl")
    cal = read_jsonl(out / "E1_V31_B_PANEL_CALIBRATION.jsonl")
    anchor = read_jsonl(out / "E1_V31_B_PANEL_ANCHOR.jsonl")
    seeds = cfg["statistics"]["seeds"]
    results = []
    for view in VIEWS:
        for seed in seeds:
            results.append(run_seed(dev, cal, anchor, view, int(seed), cfg["e1_b"]))
    aggregate = aggregate_seeds(results)
    a_rows, _ = panel_rows_to_eval(anchor)
    main_seed = 13
    run_qy = next(r for r in results if r["mode"] == "q+y" and r["seed"] == main_seed)
    run_y = next(r for r in results if r["mode"] == "y_only" and r["seed"] == main_seed)

    def scores_for(view, seed):
        from frauddistill.e1_final_v3.detector_v31 import ViewDetector

        d_rows, d_labels = panel_rows_to_eval(dev)
        detector = ViewDetector(mode=view, seed=seed, C=float(cfg["e1_b"].get("detector_C", 1.0)), max_features=int(cfg["e1_b"].get("max_features", 60000)))
        detector.fit(d_rows, d_labels)
        return detector.predict_proba(a_rows)

    qy_scores = scores_for("q+y", main_seed)
    y_scores = scores_for("y_only", main_seed)
    mcnemar = paired_mcnemar(a_rows, qy_scores, y_scores, run_qy["threshold"], run_y["threshold"])
    ci_qy = cluster_bootstrap_macro_f1(a_rows, qy_scores, run_qy["threshold"], iterations=int(cfg["statistics"].get("cluster_bootstrap_iterations_b", 2000)), seed=main_seed)
    ci_y = cluster_bootstrap_macro_f1(a_rows, y_scores, run_y["threshold"], iterations=int(cfg["statistics"].get("cluster_bootstrap_iterations_b", 2000)), seed=main_seed)
    gold = np.asarray([int(r["gold_central"]) for r in a_rows], dtype=int)
    pred_y = (y_scores >= run_y["threshold"]).astype(int)
    pred_qy = (qy_scores >= run_qy["threshold"]).astype(int)
    transitions = {
        "y_wrong_qy_correct": int(np.sum((pred_y != gold) & (pred_qy == gold))),
        "y_correct_qy_wrong": int(np.sum((pred_y == gold) & (pred_qy != gold))),
    }
    payload = {
        "aggregate": aggregate,
        "per_seed": results,
        "mcnemar_qy_vs_y": mcnemar,
        "cluster_bootstrap_ci_qy": ci_qy,
        "cluster_bootstrap_ci_y": ci_y,
        "error_transitions": transitions,
        "anchor_consumed_once": True,
    }
    write_json(out / "E1_V31_B_ANCHOR_RESULTS.json", payload)
    progress("ANCHOR", 1, 1)
    return payload

def phase_c_all(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import joblib

    out = rel(cfg["data"]["output_dir"])
    registry = [r for r in read_jsonl(out / "E1_V31_A_REGISTRY_FROZEN.jsonl") if int(r.get("gold_central", -1)) >= 0]
    freeze = read_json(out / "E1_V31_B_FREEZE.json", {})
    if not freeze or not registry:
        write_json(out / "E1_V31_C_RESULT.json", {"can_run_c": False, "reason": "A registry or B freeze artifacts missing"})
        return {"can_run_c": False}
    rows = registry
    predictions = []
    for view in ["y_only", "q+y"]:
        model_path = out / f"E1_V31_B_MODEL_{view.replace('+', 'p')}.joblib"
        if not model_path.exists():
            write_json(out / "E1_V31_C_RESULT.json", {"can_run_c": False, "reason": f"missing frozen model {model_path.name}"})
            return {"can_run_c": False}
        detector = joblib.load(model_path)
        scores = detector.predict_proba(rows)
        threshold = float(freeze["views"][view]["threshold"])
        for r, s in zip(rows, scores):
            predictions.append({"response_id": r["response_id"], "view": view, "score": float(s), "gold_central": int(r["gold_central"])})
    write_jsonl(out / "E1_V31_C_PREDICTIONS.jsonl", predictions)
    pred_map = {}
    for p in predictions:
        pred_map.setdefault(p["response_id"], {})[p["view"]] = p["score"]
    y_scores = np.asarray([pred_map[r["response_id"]]["y_only"] for r in rows], dtype=float)
    qy_scores = np.asarray([pred_map[r["response_id"]]["q+y"] for r in rows], dtype=float)
    th_y = float(freeze["views"]["y_only"]["threshold"])
    th_qy = float(freeze["views"]["q+y"]["threshold"])
    block_y = c_block(rows, y_scores, th_y)
    block_qy = c_block(rows, qy_scores, th_qy)
    gain = paired_bootstrap_gain(rows, qy_scores, y_scores, iterations=int(cfg["statistics"].get("cluster_bootstrap_iterations_c", 2000)), seed=int(cfg["experiment"]["seed"]))
    result = {
        "can_run_c": True,
        "n_rows": len(rows),
        "prevalence": {"positive": sum(1 for r in rows if r["gold_central"] == 1), "rate": sum(1 for r in rows if r["gold_central"] == 1) / len(rows)},
        "y_only": block_y,
        "q_y": block_qy,
        "auprc_ratio_qy_over_y": block_qy["auprc"] / block_y["auprc"] if block_y["auprc"] else 0.0,
        "fpr_relative_drop": (block_y["fpr"] - block_qy["fpr"]) / block_y["fpr"] if block_y["fpr"] else 0.0,
        "paired_bootstrap_gain": gain,
        "directional": directional(rows, qy_scores, th_qy),
        "note": "E1-C is NOT an unseen generalization experiment; it replays the frozen B detector on the A7500 real distribution.",
    }
    write_json(out / "E1_V31_C_RESULT.json", result)
    progress("C", 1, 1)
    return result


def phase_report(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    report_dir = rel(cfg["data"]["public_report_dir"])
    p0 = read_json(out / "E1_V31_PROTOCOL_LOCK.json", {})
    a_audit = read_json(out / "E1_V31_A_MANIFEST_AUDIT.json", {})
    target_quality = read_json(out / "E1_V31_A_TARGET_QUALITY.json", {})
    gold_quality = read_json(out / "E1_V31_A_GOLD_QUALITY.json", {})
    gold_coverage = read_json(out / "E1_V31_A_GOLD_COVERAGE.json", {})
    a_stats = read_json(out / "E1_V31_A_BEHAVIOR_STATS.json", {})
    b_audit = read_json(out / "E1_V31_B_PANEL_AUDIT.json", {})
    b_build = read_json(out / "E1_V31_B_PANEL_BUILD.json", {})
    b_gold_quality = read_json(out / "E1_V31_B_GOLD_QUALITY.json", {})
    b_freeze = read_json(out / "E1_V31_B_FREEZE.json", {})
    b_anchor = read_json(out / "E1_V31_B_ANCHOR_RESULTS.json", {})
    c_result = read_json(out / "E1_V31_C_RESULT.json", {})
    decision = decision_payload(p0, a_audit, target_quality, gold_quality, a_stats, b_audit, b_anchor, c_result)
    payload = {
        "protocol": cfg["experiment"]["protocol"],
        "runtime_commit": git_commit(),
        "worktree_status": git_status_short(),
        "decision": decision,
        "analysis": analysis(decision, a_audit, target_quality, gold_quality, a_stats, b_audit, b_anchor, c_result),
        "a": {
            "manifest_audit": a_audit,
            "target_quality": target_quality,
            "gold_quality": gold_quality,
            "gold_coverage": gold_coverage,
            "behavior_stats": a_stats,
            "natural_metrics_reference": a_reference(cfg),
        },
        "b": {
            "panel_audit": b_audit,
            "panel_build": b_build,
            "gold_quality": b_gold_quality,
            "freeze": b_freeze,
            "anchor": b_anchor,
        },
        "c": {"result": c_result},
        "budget": p0.get("budget", budget_snapshot(cfg)),
        "ledger_summary": ledger_summary(cfg),
        "data_audit": {"p0": p0, "license": read_json(out / "E1_V31_DATASET_LICENSE_AUDIT.json", {})},
    }
    write_json(out / "E1_V31_DECISION.json", decision)
    write_json(out / "E1_V31_FINAL_PAYLOAD.json", payload)
    write_json(out / "E1_V31_RUN_FINGERPRINT.json", {"commit": git_commit(), "worktree_status": git_status_short(), "protocol": cfg["experiment"]["protocol"]})
    write_v31_reports(report_dir, payload)
    progress("REPORT", 1, 1)
    return payload


def ledger_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    out = rel(cfg["data"]["output_dir"])
    rows = read_jsonl(out / "E1_V31_BUDGET_LEDGER.jsonl")
    total = sum(float(r.get("cost_cny", 0) or 0) for r in rows)
    by_provider: dict[str, float] = {}
    by_phase: dict[str, int] = {}
    for r in rows:
        provider = str(r.get("provider", "unknown"))
        by_provider[provider] = by_provider.get(provider, 0.0) + float(r.get("cost_cny", 0) or 0)
        phase = str(r.get("phase", "unknown"))
        by_phase[phase] = by_phase.get(phase, 0) + 1
    return {"rows": len(rows), "total_cost_cny": round(total, 4), "by_provider_cost": {k: round(v, 4) for k, v in by_provider.items()}, "calls_by_phase": by_phase}


def decision_payload(
    p0: dict[str, Any],
    a_audit: dict[str, Any],
    target_quality: dict[str, Any],
    gold_quality: dict[str, Any],
    a_stats: dict[str, Any],
    b_audit: dict[str, Any],
    b_anchor: dict[str, Any],
    c_result: dict[str, Any],
) -> dict[str, Any]:
    if not p0.get("api_allowed_now"):
        code = "E1_V31_STOP_IMPLEMENTATION_NOT_API_CAPABLE"
    elif not a_audit or a_audit.get("target_prompt_instances") != 3750 or a_audit.get("stage_gt_0", 1) != 0:
        code = "E1_V31_STOP_A_MANIFEST_INVALID"
    elif target_quality.get("target_gate") != "PASS":
        code = "E1_V31_PENDING_A_TARGET_GENERATION"
    elif gold_quality.get("gate") == "FAIL":
        code = "E1_V31_STOP_GOLD_QUALITY"
    elif not a_stats:
        code = "E1_V31_PENDING_A_STATS"
    elif not b_audit.get("formal_panel_ready"):
        code = "E1_V31_PENDING_B_PANEL"
    elif not b_anchor.get("anchor_consumed_once"):
        code = "E1_V31_PENDING_B_ANCHOR"
    elif not c_result.get("can_run_c"):
        code = "E1_V31_PENDING_C_REPLAY"
    else:
        agg = b_anchor.get("aggregate", {})
        qy = agg.get("q+y", {})
        y = agg.get("y_only", {})
        q = agg.get("q_only", {})
        qy_macro = qy.get("anchor_macro_f1_mean", 0.0)
        gain_y = qy_macro - y.get("anchor_macro_f1_mean", 0.0)
        gain_q = qy_macro - q.get("anchor_macro_f1_mean", 0.0)
        if qy_macro >= 0.90 and gain_y >= 0.05 and gain_q >= 0.20:
            code = "E1_V31_FULL_TRIAD_PASS"
        elif qy_macro >= 0.88 and gain_y >= 0.03:
            code = "E1_V31_BEHAVIOR_PASS_MECHANISM_PASS_C_MIXED"
        else:
            code = "E1_V31_BEHAVIOR_PASS_MECHANISM_WEAK"
    return {
        "decision_code": code,
        "p0_gate": p0.get("gate"),
        "a_manifest_gate": "PASS" if a_audit.get("target_prompt_instances") == 3750 and a_audit.get("stage_gt_0") == 0 else "STOP",
        "a_target_gate": target_quality.get("target_gate", "NOT_RUN"),
        "a_gold_gate": gold_quality.get("gate", "NOT_RUN"),
        "b_gate": "PASS" if b_audit.get("formal_panel_ready") else "NOT_READY",
        "c_gate": "PASS" if c_result.get("can_run_c") else "NOT_READY",
    }


def analysis(
    decision: dict[str, Any],
    a_audit: dict[str, Any],
    target_quality: dict[str, Any],
    gold_quality: dict[str, Any],
    a_stats: dict[str, Any],
    b_audit: dict[str, Any],
    b_anchor: dict[str, Any],
    c_result: dict[str, Any],
) -> str:
    overall = a_stats.get("overall", {})
    central = overall.get("central", {})
    agg = b_anchor.get("aggregate", {})
    qy = agg.get("q+y", {})
    y = agg.get("y_only", {})
    q = agg.get("q_only", {})
    parts = [
        f"A7500?canonical cases={a_audit.get('canonical_cases')}?prompt instances={a_audit.get('target_prompt_instances')}??? responses={a_audit.get('reused_target_responses')}?? API ??={target_quality.get('valid_new_response_rows', 0)}?",
        f"A Gold?{gold_quality.get('gate', 'NOT_RUN')}?completion={gold_quality.get('completion_rate')}?binary agreement={gold_quality.get('binary_agreement')}?PABAK={gold_quality.get('pabak')}?",
        f"A central prevalence={central.get('rate')}?positive={central.get('positive')}?95% CI {central.get('wilson_95', {}).get('low')}-{central.get('wilson_95', {}).get('high')}??",
        f"B ???formal_panel_ready={b_audit.get('formal_panel_ready')}?by stratum={b_audit.get('panel', {}).get('by_stratum')}?",
        f"B Anchor?q+y Macro-F1={qy.get('anchor_macro_f1_mean')}?SD {qy.get('anchor_macro_f1_sd')}??y-only={y.get('anchor_macro_f1_mean')}?q-only={q.get('anchor_macro_f1_mean')}?",
        f"C ???can_run={c_result.get('can_run_c')}?q+y AUPRC={c_result.get('q_y', {}).get('auprc')}?y-only AUPRC={c_result.get('y_only', {}).get('auprc')}?",
        f"?? decision code?`{decision['decision_code']}`?",
    ]
    return "\n".join(parts)


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
    full = executive + "\n\n## A ??\n" + block(payload["a"]) + "\n\n## B ??\n" + block(payload["b"]) + "\n\n## C ??\n" + block(payload["c"]) + "\n\n## ?? ledger\n" + block(payload.get("ledger_summary", {})) + "\n"
    contents = {
        names[0]: executive,
        names[1]: full,
        names[2]: "# E1 v3.1 ????? provenance ??\n\n" + block(payload["data_audit"]) + "\n\n## B ??????\n" + block(payload["b"].get("panel_build", {})) + "\n",
        names[3]: "# E1 v3.1 A Target ????\n\n" + block(payload["a"]) + "\n",
        names[4]: "# E1 v3.1 Gold ????\n\n" + block({"A": payload["a"].get("gold_quality"), "B": payload["b"].get("gold_quality"), "coverage": payload["a"].get("gold_coverage")}) + "\n",
        names[5]: "# E1 v3.1 ????\n\n" + block(payload["budget"]) + "\n\n## ?? ledger\n" + block(payload.get("ledger_summary", {})) + "\n",
        names[6]: "# E1 v3.1 ???????\n\n" + block({"analysis": payload["analysis"], "b_rejects": payload["b"].get("panel_audit", {}).get("rejects_total"), "shortcut_audits": payload["b"].get("freeze", {}).get("shortcut_audits"), "c_note": payload["c"].get("result", {}).get("note")}) + "\n",
        names[7]: "# E1 v3.1 ????\n\n" + block(payload["a"].get("behavior_stats", {})) + "\n\n## B ??\n" + block(payload["b"].get("anchor", {})) + "\n\n## C ??\n" + block(payload["c"].get("result", {})) + "\n",
        names[8]: paper_tables(payload),
        names[9]: "# E1 v3.1 ????\n\n```powershell\npython scripts/run_e1_a7500.py --phase p0\npython scripts/run_e1_a7500.py --phase build-manifest\npython scripts/run_e1_a7500.py --phase health --run-api --confirm-budget --limit-q 50\npython scripts/run_e1_a7500.py --phase generate --run-api --confirm-budget --batch-size-q 500 --resume\npython scripts/run_e1_a7500.py --phase validate-targets\npython scripts/run_e1_a7500.py --phase gold --run-api --confirm-budget --resume\npython scripts/run_e1_a7500.py --phase adjudicate --run-api --confirm-budget --resume\npython scripts/run_e1_a7500.py --phase freeze\npython scripts/run_e1_b3200.py --phase build-panel --run-api --confirm-budget --resume\npython scripts/run_e1_b3200.py --phase b-gold --run-api --confirm-budget --resume\npython scripts/run_e1_b3200.py --phase b-adjudicate --run-api --confirm-budget --resume\npython scripts/run_e1_b3200.py --phase b-consensus\npython scripts/run_e1_b3200.py --phase validate-panel\npython scripts/run_e1_b3200.py --phase model-dev\npython scripts/run_e1_b3200.py --phase calibration\npython scripts/run_e1_b3200.py --phase freeze-b\npython scripts/run_e1_b3200.py --phase anchor --consume-anchor\npython scripts/run_e1_c_real_prevalence.py --phase c-all\npython scripts/run_e1_final_triad_v3.py --phase final-report\n```\n",
        names[10]: "# E1 v3.1 ????\n\n```json\n" + json_dump(payload["decision"]) + "\n```\n",
    }
    for name in names:
        (report_dir / name).write_text(contents[name], encoding="utf-8")


def paper_tables(payload: dict[str, Any]) -> str:
    lines = ["# E1 v3.1 ????", ""]
    a = payload["a"].get("behavior_stats", {})
    lines.append("## A. ???? central ???")
    rows = [{"model": row["stratum"], "n": row["n"], "positive": row["positive"], "rate": f"{row['rate']:.4f}", "ci_low": f"{row['wilson_95']['low']:.4f}", "ci_high": f"{row['wilson_95']['high']:.4f}"} for row in a.get("by_model", [])]
    lines.append(table(rows))
    lines.append("")
    lines.append("### A. by setting")
    rows = [{"setting": row["stratum"], "n": row["n"], "positive": row["positive"], "rate": f"{row['rate']:.4f}", "ci_low": f"{row['wilson_95']['low']:.4f}", "ci_high": f"{row['wilson_95']['high']:.4f}"} for row in a.get("by_setting", [])]
    lines.append(table(rows))
    lines.append("")
    lines.append("### A. by language")
    rows = [{"language": row["stratum"], "n": row["n"], "positive": row["positive"], "rate": f"{row['rate']:.4f}", "ci_low": f"{row['wilson_95']['low']:.4f}", "ci_high": f"{row['wilson_95']['high']:.4f}"} for row in a.get("by_language", [])]
    lines.append(table(rows))
    lines.append("")
    lines.append("### A. by category")
    rows = [{"category": row["stratum"], "n": row["n"], "positive": row["positive"], "rate": f"{row['rate']:.4f}", "ci_low": f"{row['wilson_95']['low']:.4f}", "ci_high": f"{row['wilson_95']['high']:.4f}"} for row in a.get("by_category", [])]
    lines.append(table(rows))
    lines.append("")
    lines.append("### A. Qwen vs DeepSeek McNemar")
    lines.append(block(a.get("mcnemar_qwen_vs_deepseek", {})))
    lines.append("")
    b = payload["b"].get("anchor", {})
    lines.append("## B. Frozen Anchor ??? 5-seed ???")
    agg_rows = []
    for view in ["q_only", "y_only", "q+y"]:
        v = b.get("aggregate", {}).get(view, {})
        agg_rows.append({"view": view, "macro_f1_mean": f"{v.get('anchor_macro_f1_mean', 0):.4f}", "macro_f1_sd": f"{v.get('anchor_macro_f1_sd', 0):.4f}", "auprc_mean": f"{v.get('anchor_auprc_mean', 0):.4f}", "fpr_mean": f"{v.get('anchor_fpr_mean', 0):.4f}", "recall_mean": f"{v.get('anchor_recall_mean', 0):.4f}"})
    lines.append(table(agg_rows))
    lines.append("")
    lines.append("### B. q+y vs y-only ????? CI")
    lines.append(block({"mcnemar": b.get("mcnemar_qy_vs_y"), "ci_qy": b.get("cluster_bootstrap_ci_qy"), "ci_y": b.get("cluster_bootstrap_ci_y"), "transitions": b.get("error_transitions")}))
    lines.append("")
    c = payload["c"].get("result", {})
    lines.append("## C. ???????")
    c_rows = []
    for view in ["y_only", "q_y"]:
        v = c.get(view, {})
        c_rows.append({"view": view, "auprc": f"{v.get('auprc', 0):.4f}", "auroc": f"{v.get('auroc', 0):.4f}", "fpr": f"{v.get('fpr', 0):.4f}", "recall": f"{v.get('recall', 0):.4f}", "precision": f"{v.get('precision', 0):.4f}", "brier": f"{v.get('brier', 0):.4f}", "ece": f"{v.get('ece', 0):.4f}"})
    lines.append(table(c_rows))
    lines.append("")
    lines.append("### C. AUPRC ratio / FPR ????")
    lines.append(block({"auprc_ratio_qy_over_y": c.get("auprc_ratio_qy_over_y"), "fpr_relative_drop": c.get("fpr_relative_drop"), "paired_bootstrap_gain": c.get("paired_bootstrap_gain"), "note": c.get("note")}))
    lines.append("")
    lines.append("## ??")
    lines.append(block(payload["decision"]))
    return "\n".join(lines)


def render_executive(payload: dict[str, Any]) -> str:
    d = payload["decision"]
    a = payload["a"]
    central = a.get("behavior_stats", {}).get("overall", {}).get("central", {})
    agg = payload["b"].get("anchor", {}).get("aggregate", {})
    qy = agg.get("q+y", {})
    y = agg.get("y_only", {})
    q = agg.get("q_only", {})
    c = payload["c"].get("result", {})
    lines = [
        "# E1 FINAL TRIAD v3.1 ?????",
        "",
        "## ????",
        f"- final decision code?`{d['decision_code']}`",
        f"- Git commit?`{payload['runtime_commit']}`",
        f"- worktree status?`{payload['worktree_status'] or 'clean'}`",
        f"- protocol?`{payload['protocol']}`",
        f"- A/B/C ???A manifest `{d['a_manifest_gate']}`?A target `{d['a_target_gate']}`?A Gold `{d['a_gold_gate']}`?B `{d['b_gate']}`?C `{d['c_gate']}`",
        f"- ??? API ???target `{a['target_quality'].get('valid_new_response_rows', 0)}` ????Gold ?? `{a.get('gold_coverage', {}).get('new_gold_known', 0)}` ??ledger rows=`{payload.get('ledger_summary', {}).get('rows', 0)}`",
        f"- A7500 ?????`{a['target_quality'].get('valid_new_response_rows', 0)}` ? + `{a.get('manifest_audit', {}).get('reused_target_responses', 0)}` ?????? pair=`{a['target_quality'].get('complete_new_pairs', 0)}`",
        f"- A central prevalence?`{central.get('rate')}`?positive=`{central.get('positive')}`?95% CI `{central.get('wilson_95', {}).get('low')}`-`{central.get('wilson_95', {}).get('high')}`?",
        f"- B ??? Anchor Macro-F1?q-only=`{q.get('anchor_macro_f1_mean', 'N/A')}`?y-only=`{y.get('anchor_macro_f1_mean', 'N/A')}`?q+y=`{qy.get('anchor_macro_f1_mean', 'N/A')}`",
        f"- C ???q+y AUPRC=`{c.get('q_y', {}).get('auprc', 'N/A')}`?y-only AUPRC=`{c.get('y_only', {}).get('auprc', 'N/A')}`?AUPRC ratio=`{c.get('auprc_ratio_qy_over_y', 'N/A')}`",
        "",
        "## ??",
        payload["analysis"],
    ]
    return "\n".join(lines)


def json_dump(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    import csv

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
        elif phase == "gold":
            phase_gold(cfg, args)
        elif phase == "adjudicate":
            phase_adjudicate(cfg, args)
        elif phase == "freeze":
            phase_freezer(cfg, args)
        elif phase == "build-panel":
            phase_b_build_panel(cfg, args)
        elif phase == "b-gold":
            phase_b_gold(cfg, args)
        elif phase == "b-adjudicate":
            phase_b_adjudicate(cfg, args)
        elif phase == "b-consensus":
            phase_b_consensus(cfg, args)
        elif phase == "validate-panel":
            phase_validate_panel(cfg, args)
        elif phase == "model-dev":
            phase_model_dev(cfg, args)
        elif phase == "calibration":
            phase_calibration(cfg, args)
        elif phase == "freeze-b":
            phase_freeze_b(cfg, args)
        elif phase == "anchor":
            phase_anchor(cfg, args)
        elif phase == "c-all":
            phase_c_all(cfg, args)
        elif phase in {"report", "final-report"}:
            phase_report(cfg)
        elif phase == "all":
            pass
        else:
            raise ValueError(f"unsupported phase: {phase}")
        progress("TOTAL", idx, len(phases))
    print(f"v3.1 ???? phase={args.phase} output={rel(cfg['data']['output_dir'])}")


def expand_phases(phase: str, component: str) -> list[str]:
    if phase == "all":
        if component == "a":
            return ["p0", "build-manifest", "validate-targets", "report"]
        if component == "b":
            return ["validate-panel", "report"]
        if component == "c":
            return ["c-all", "report"]
        return ["p0", "build-manifest", "validate-targets", "report"]
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
