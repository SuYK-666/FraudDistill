from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from frauddistill.e1_v8.official_prompt_renderer import official_assistant_prompt, official_roleplay_prompt

from .io import norm, read_jsonl, sha_text


def normalize_language(value: str) -> str:
    text = (value or "").lower()
    if text in {"chinese", "zh", "cn"}:
        return "zh"
    if text in {"english", "en"}:
        return "en"
    return text or "unknown"


def normalize_category(value: str) -> str:
    text = (value or "").lower().replace("-", " ").replace("_", " ")
    if "job" in text:
        return "fake_job_posting"
    if "phish" in text:
        return "phishing"
    if "imperson" in text:
        return "impersonation"
    if "service" in text:
        return "fraudulent_service"
    if "relationship" in text or "network" in text or "friendship" in text or "dating" in text:
        return "network_friendship"
    return text.replace(" ", "_") or "unknown"


def load_fraudr1_q_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    rejects = Counter()
    seen_hashes = set()
    for raw in read_jsonl(path):
        q = raw.get("user_query") or raw.get("q") or ""
        if not q:
            rejects["missing_q"] += 1
            continue
        q_hash = sha_text(norm(q))
        if q_hash in seen_hashes:
            rejects["duplicate_q_hash"] += 1
            continue
        seen_hashes.add(q_hash)
        rows.append(
            {
                "canonical_q_id": raw.get("id") or q_hash[:16],
                "q_private": q,
                "q_hash_recomputed": q_hash,
                "language": normalize_language(raw.get("language", "")),
                "fraud_category": normalize_category(raw.get("fraud_category", raw.get("category", ""))),
                "source_dataset": "Fraud-R1",
                "source_file": str(path),
                "source_prior": raw.get("source_prior", ""),
                "raw_metadata": raw.get("metadata", {}),
            }
        )
    audit = {"source_file": str(path), "accepted": len(rows), "rejects": dict(rejects), "unique_q_hashes": len(seen_hashes)}
    return rows, audit


def load_response_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rejects = Counter()
    by_source = Counter()
    seen = set()
    for path in paths:
        for raw in read_jsonl(path):
            provider = raw.get("target_provider") or raw.get("provider") or raw.get("target_model")
            if provider not in {"qwen", "deepseek"}:
                rejects["non_target_provider"] += 1
                continue
            q = raw.get("q") or raw.get("prompt") or ""
            y = raw.get("y") or raw.get("text") or ""
            if not y:
                rejects["missing_y"] += 1
                continue
            q_hash = sha_text(norm(q)) if q else ""
            y_hash = raw.get("normalized_y_sha256") or raw.get("response_sha256") or sha_text(norm(y))
            key = (q_hash, y_hash, provider, raw.get("request_id", ""))
            if key in seen:
                rejects["duplicate_response"] += 1
                continue
            seen.add(key)
            source_name = source_from_path(path)
            by_source[source_name] += 1
            out.append(normalize_response(raw, path, source_name, provider, q, y, q_hash, y_hash))
    audit = {
        "input_files": [str(p) for p in paths],
        "accepted": len(out),
        "rejects": dict(rejects),
        "by_source_dataset": dict(by_source),
        "by_provider": dict(Counter(r["target_provider"] for r in out)),
        "with_q_text": sum(1 for r in out if r["q_private"]),
        "missing_q_text": sum(1 for r in out if not r["q_private"]),
        "source_derived_rows": 0,
    }
    return out, audit


def normalize_response(raw: dict[str, Any], path: Path, source_name: str, provider: str, q: str, y: str, q_hash: str, y_hash: str) -> dict[str, Any]:
    response_id = raw.get("response_id") or raw.get("task_id") or sha_text("|".join([q_hash, y_hash, provider]))
    return {
        "response_id": response_id,
        "canonical_q_id": raw.get("canonical_id") or raw.get("canonical_q_id") or response_id,
        "q_private": q,
        "q_hash_recomputed": q_hash,
        "y_private": y,
        "y_hash": y_hash,
        "target_provider": provider,
        "requested_target_model": raw.get("requested_target_model") or raw.get("requested_model", ""),
        "resolved_target_model": raw.get("resolved_target_model") or raw.get("response_model", ""),
        "request_id": raw.get("request_id", ""),
        "status": raw.get("status", ""),
        "finish_reason": raw.get("finish_reason", ""),
        "language": normalize_language(raw.get("language", "")),
        "fraud_category": normalize_category(raw.get("category", raw.get("fraud_category", ""))),
        "source_file": str(path),
        "source_dataset": source_name,
        "provenance": "real_target_response",
        "is_real_target_response": True,
        "is_source_derived": False,
        "used_in_prior_report": bool(raw.get("used_in_prior_report", False)),
        "eligible_for_v10_dev": bool(raw.get("eligible_for_v10_dev", False)),
        "eligible_for_v10_anchor": bool(raw.get("eligible_for_v10_anchor", False)),
        "gold_status": "INLINE" if any(k in raw for k in ["material_central", "material_assist"]) else "UNKNOWN",
        "gold_lower": int(raw.get("material_lower", 0) or 0),
        "gold_central": int(raw.get("material_central", raw.get("material_assist", 0)) or 0),
        "gold_upper": int(raw.get("material_upper", 0) or 0),
        "judge_a_positive": int(raw.get("judge_a_positive", 0) or 0),
        "judge_b_positive": int(raw.get("judge_b_positive", 0) or 0),
        "severity_central": int(raw.get("severity_central", 0) or 0),
    }


def source_from_path(path: Path) -> str:
    text = str(path).lower()
    if "v81" in text and "p2" in text:
        return "V8.1-P2-real"
    if "v8_a2c" in text:
        return "V8-A2C-real"
    if "pressure" in text:
        return "V10-pressure-real"
    if "v10" in text:
        return "V10-natural-real"
    return path.stem


def join_gold(rows: list[dict[str, Any]], gold_path: Path) -> list[dict[str, Any]]:
    gold = {r.get("response_id"): r for r in read_jsonl(gold_path)}
    joined = []
    for row in rows:
        g = gold.get(row["response_id"])
        if not g:
            joined.append(row)
            continue
        joined.append(
            {
                **row,
                "gold_status": "KNOWN",
                "gold_lower": int(g.get("material_lower", 0) or 0),
                "gold_central": int(g.get("material_central", g.get("material_assist", 0)) or 0),
                "gold_upper": int(g.get("material_upper", 0) or 0),
                "judge_a_positive": int(g.get("judge_a_positive", 0) or 0),
                "judge_b_positive": int(g.get("judge_b_positive", 0) or 0),
                "severity_central": int(g.get("severity_central", 0) or 0),
            }
        )
    return joined


def load_base_cases(raw_prompts_path: Path, raw_base_en: Path, raw_base_zh: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = [r for r in read_jsonl(raw_prompts_path) if "FP-base" in str(r.get("source_file", ""))]
    raw_entries = raw_entry_map(raw_base_en, raw_base_zh)
    out = []
    seen = set()
    for row in prompts:
        canonical = canonical_case_id(row)
        if canonical in seen:
            continue
        seen.add(canonical)
        raw = raw_entries.get(canonical)
        entry = entry_for_renderer(row, raw)
        out.append({**row, "canonical_case_id": canonical, "renderer_entry": entry})
    audit = {
        "raw_prompt_rows": len(read_jsonl(raw_prompts_path)),
        "base_rows": len(prompts),
        "canonical_cases": len(out),
        "by_language_category": stringify_counter(Counter((normalize_language(r["language"]), normalize_category(r["fraud_category"])) for r in out)),
    }
    return out, audit


def raw_entry_map(raw_base_en: Path, raw_base_zh: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    import json

    for path, lang_prefix in [(raw_base_en, "english"), (raw_base_zh, "chinese")]:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw in data:
            out[f"fraudr1_{lang_prefix}_{raw.get('id')}"] = raw
    return out


def canonical_case_id(row: dict[str, Any]) -> str:
    return str(row.get("id", "")).lower()


def entry_for_renderer(row: dict[str, Any], raw: dict[str, Any] | None) -> dict[str, Any]:
    lang = normalize_language(row.get("language", ""))
    base = raw or {}
    return {
        "language": lang,
        "data_type": row.get("data_type") or base.get("data_type"),
        "role_bg": base.get("role_bg", {}),
        "stages": [{"generated_data": row.get("user_query") or base.get("generated text") or row.get("raw_data", "")}],
    }


def build_v31_a_manifest(
    *,
    raw_prompts_path: Path,
    raw_base_en: Path,
    raw_base_zh: Path,
    v10_registry_path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cases, case_audit = load_base_cases(raw_prompts_path, raw_base_en, raw_base_zh)
    history, history_audit = load_v31_reusable_roleplay(v10_registry_path)
    history_case_ids = set(history)
    assistant = [prompt_instance(row, "assistant", config) for row in cases]
    roleplay_reused = [roleplay_instance_from_history(case_id, pair, config) for case_id, pair in sorted(history.items())]
    remaining = [row for row in cases if row["canonical_case_id"] not in history_case_ids]
    extra_n = int(config["e1_a"]["extra_roleplay_prompt_instances"])
    extra = select_extra_roleplay_cases(remaining, cases, extra_n, int(config["experiment"]["seed"]))
    roleplay_extra = [prompt_instance(row, "roleplay", config) for row in extra]
    prompts = assistant + roleplay_reused + roleplay_extra
    tasks = [target_task(row, provider, config) for row in prompts for provider in ["qwen", "deepseek"] if not has_reused_response(row, provider)]
    audit = {
        "case_audit": case_audit,
        "history_audit": history_audit,
        "canonical_cases": len(cases),
        "assistant_prompt_instances": len(assistant),
        "roleplay_reused_prompt_instances": len(roleplay_reused),
        "roleplay_extra_prompt_instances": len(roleplay_extra),
        "target_prompt_instances": len(prompts),
        "expected_target_responses": len(prompts) * 2,
        "reused_target_responses": sum(len(p.get("reused_responses", {})) for p in roleplay_reused),
        "pending_target_calls": len(tasks),
        "stage_gt_0": sum(1 for p in prompts if int(p["stage_id"]) != 0),
        "prompt_instance_duplicates": len(prompts) - len({p["prompt_instance_id"] for p in prompts}),
        "exact_q_hash_conflicts": exact_q_conflicts(prompts),
    }
    return prompts, tasks, audit


def load_v31_reusable_roleplay(path: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    rejects = Counter()
    for raw in read_jsonl(path):
        if raw.get("source_run") != "v91_frozen":
            rejects["not_v91"] += 1
            continue
        try:
            stage_id = int(raw.get("stage_id", -1))
        except Exception:
            stage_id = -1
        if stage_id != 0:
            rejects["not_stage0"] += 1
            continue
        if raw.get("used_in_prior_report") is not True:
            rejects["not_prior_report"] += 1
            continue
        if raw.get("status") != "ok":
            rejects["bad_status"] += 1
            continue
        provider = raw.get("target_provider")
        if provider not in {"qwen", "deepseek"}:
            rejects["bad_provider"] += 1
            continue
        q = raw.get("q") or ""
        y = raw.get("y") or ""
        if not q or not y:
            rejects["missing_qy"] += 1
            continue
        case_id = str(raw.get("canonical_id", "")).replace("_en_", "_english_").replace("_zh_", "_chinese_")
        row = normalize_response(raw, path, "V9.1-roleplay-reused", provider, q, y, sha_text(norm(q)), raw.get("normalized_y_sha256") or sha_text(norm(y)))
        pairs.setdefault(case_id, {})[provider] = row
    complete = {case: by_provider for case, by_provider in pairs.items() if set(by_provider) == {"qwen", "deepseek"}}
    audit = {
        "complete_prompt_pairs": len(complete),
        "reused_responses": sum(len(v) for v in complete.values()),
        "rejects": dict(rejects),
        "provider_counts": dict(Counter(provider for v in complete.values() for provider in v)),
    }
    return complete, audit


def prompt_instance(row: dict[str, Any], scenario: str, config: dict[str, Any]) -> dict[str, Any]:
    entry = row["renderer_entry"]
    q = official_assistant_prompt(entry, 0) if scenario == "assistant" else official_roleplay_prompt(entry, 0)
    prompt_id = f"{row['canonical_case_id']}|stage0|{scenario}|fraudr1-official-v1"
    return {
        "prompt_instance_id": prompt_id,
        "canonical_case_id": row["canonical_case_id"],
        "stage_id": 0,
        "scenario": scenario,
        "prompt_renderer_version": "fraudr1-official-v1",
        "q_private": q,
        "exact_q_sha256": sha_text(q),
        "normalized_q_sha256": sha_text(norm(q)),
        "language": normalize_language(row.get("language", "")),
        "fraud_category": normalize_category(row.get("fraud_category", "")),
        "source_dataset": "Fraud-R1",
        "source_file": row.get("source_file"),
        "reused_responses": {},
    }


def roleplay_instance_from_history(case_id: str, pair: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    q = next(iter(pair.values()))["q_private"]
    return {
        "prompt_instance_id": f"{case_id}|stage0|roleplay|fraudr1-official-v1",
        "canonical_case_id": case_id,
        "stage_id": 0,
        "scenario": "roleplay",
        "prompt_renderer_version": "fraudr1-official-v1",
        "q_private": q,
        "exact_q_sha256": sha_text(q),
        "normalized_q_sha256": sha_text(norm(q)),
        "language": next(iter(pair.values()))["language"],
        "fraud_category": next(iter(pair.values()))["fraud_category"],
        "source_dataset": "Fraud-R1",
        "source_file": "data/prepared/e1_v10_trilayer/E1_V10_RESPONSE_REGISTRY.jsonl",
        "reused_responses": pair,
    }


def select_extra_roleplay_cases(remaining: list[dict[str, Any]], all_cases: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    target_dist = Counter((normalize_language(r["language"]), normalize_category(r["fraud_category"])) for r in all_cases)
    selected: list[dict[str, Any]] = []
    pool = remaining[:]
    while pool and len(selected) < n:
        best = min(pool, key=lambda r: (l1_after(selected + [r], target_dist), rng.random()))
        selected.append(best)
        pool.remove(best)
    return selected


def l1_after(selected: list[dict[str, Any]], target_dist: Counter) -> float:
    if not selected:
        return 0.0
    selected_dist = Counter((normalize_language(r["language"]), normalize_category(r["fraud_category"])) for r in selected)
    total_s = sum(selected_dist.values())
    total_t = sum(target_dist.values())
    keys = set(selected_dist) | set(target_dist)
    return sum(abs(selected_dist.get(k, 0) / total_s - target_dist.get(k, 0) / total_t) for k in keys)


def target_task(prompt: dict[str, Any], provider: str, config: dict[str, Any]) -> dict[str, Any]:
    model_key = f"target_{provider}_v31"
    model = config["models"][model_key]
    return {
        **{k: v for k, v in prompt.items() if k != "reused_responses"},
        "target_provider": provider,
        "requested_target_model": model["model"],
        "extra_body": model.get("extra_body", {}),
        "temperature": config["generation"]["temperature"],
        "top_p": config["generation"]["top_p"],
        "max_tokens": config["generation"]["max_tokens"],
        "timeout_seconds": config["generation"]["timeout_seconds"],
        "phase": "E1-A-target-generation-v31",
        "status": "PENDING_API",
    }


def has_reused_response(prompt: dict[str, Any], provider: str) -> bool:
    return provider in (prompt.get("reused_responses") or {})


def exact_q_conflicts(prompts: list[dict[str, Any]]) -> int:
    owners: dict[str, set[str]] = {}
    for prompt in prompts:
        owners.setdefault(prompt["exact_q_sha256"], set()).add(prompt["prompt_instance_id"])
    return sum(1 for v in owners.values() if len(v) > 1)


def stringify_counter(counter: Counter) -> dict[str, int]:
    return {"|".join(str(part) for part in key) if isinstance(key, tuple) else str(key): int(value) for key, value in counter.items()}
