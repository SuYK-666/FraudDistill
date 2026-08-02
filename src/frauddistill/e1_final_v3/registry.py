from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

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
