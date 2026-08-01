from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha_text, write_csv, write_json, write_jsonl
from .provenance import canonicalize_row


def scan_history(root: Path, dirs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted: dict[tuple[str, str, str], dict[str, Any]] = {}
    flow = Counter()
    by_version = Counter()
    reject_rows = []
    for d in dirs:
        path = root / d
        if not path.exists():
            flow["missing_dir"] += 1
            continue
        for file in path.rglob("*.jsonl"):
            if "BUDGET_LEDGER" in file.name or "PREDICTIONS" in file.name:
                continue
            rows = read_jsonl(file)
            flow["historical_scanned_rows"] += len(rows)
            for i, row in enumerate(rows):
                canon, reason = canonicalize_row(file, row, i)
                flow[reason] += 1
                if canon is None:
                    if reason in {"placeholder_q", "missing_q", "missing_y"}:
                        reject_rows.append({"file": str(file), "index": i, "reason": reason})
                    continue
                key = (canon["exact_q_hash"], canon["y_hash"], canon["target_model_snapshot"])
                if key not in accepted:
                    accepted[key] = canon
                    by_version[canon["source_version"]] += 1
    rows = list(accepted.values())
    summary = {
        "flow": dict(flow),
        "dedup_unique_rows": len(rows),
        "by_version": dict(by_version),
        "by_model": dict(Counter(r["target_model"] for r in rows)),
        "by_language": dict(Counter(r["language"] for r in rows)),
        "by_category": dict(Counter(r["fraud_category"] for r in rows)),
        "reject_samples": reject_rows[:200],
    }
    return rows, summary


def freeze_pilot(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    eligible = [r for r in rows if r["fraud_category"] != "unknown" and r["actor_role"] in {"PERPETRATOR", "DUAL_USE_AMBIGUOUS", "VICTIM_DEFENDER"}]
    buckets: list[tuple[str, list[dict[str, Any]], int]] = [
        ("old_positive_or_upper", [r for r in eligible if r["old_material"] or r["old_upper"]], 120),
        ("old_uncertain", [r for r in eligible if r["old_uncertain"]], 80),
        ("actionable_hard_negative", [r for r in eligible if not r["old_material"] and actionish(r["y_private"])], 60),
        ("stable_negative", [r for r in eligible if not r["old_material"] and not r["old_uncertain"]], 80),
    ]
    picked: dict[str, dict[str, Any]] = {}
    def add_matching(predicate, limit: int, bucket_name: str) -> None:
        for row in sorted([r for r in eligible if predicate(r)], key=lambda r: sha_text(f"{seed}|{bucket_name}|{r['response_id']}")):
            if len([x for x in picked.values() if x.get("balance_bucket") == bucket_name]) >= limit:
                break
            if row["response_id"] not in picked:
                picked[row["response_id"]] = {**row, "pilot_bucket": row.get("pilot_bucket", "balance"), "balance_bucket": bucket_name, "split_role": "gold_pilot"}

    for model in ("qwen", "deepseek"):
        add_matching(lambda r, m=model: r["target_model"] == m, 120, f"model_{model}")
    for lang in ("en", "zh"):
        need = max(0, 120 - sum(1 for r in picked.values() if r["language"] == lang))
        add_matching(lambda r, l=lang: r["language"] == l, need, f"language_{lang}")
    for cat in ("phishing", "impersonation", "fake_job", "relationship_investment", "fraudulent_service"):
        need = max(0, 30 - sum(1 for r in picked.values() if r["fraud_category"] == cat))
        add_matching(lambda r, c=cat: r["fraud_category"] == c, need, f"category_{cat}")
    for name, bucket, cap in buckets:
        for row in sorted(bucket, key=lambda r: sha_text(f"{seed}|{name}|{r['response_id']}")):
            if len([x for x in picked.values() if x.get("pilot_bucket") == name]) >= cap:
                break
            key = row["response_id"]
            if key not in picked:
                picked[key] = {**row, "pilot_bucket": name, "split_role": "gold_pilot"}
            if len(picked) >= n:
                break
        if len(picked) >= n:
            break
    if len(picked) < n:
        for row in sorted(eligible, key=lambda r: sha_text(f"{seed}|fill|{r['response_id']}")):
            if row["response_id"] not in picked:
                picked[row["response_id"]] = {**row, "pilot_bucket": "fill", "split_role": "gold_pilot"}
            if len(picked) >= n:
                break
    return list(picked.values())[:n]


def pilot_gate(rows: list[dict[str, Any]], req: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "total": len(rows),
        "by_model": dict(Counter(r["target_model"] for r in rows)),
        "by_language": dict(Counter(r["language"] for r in rows)),
        "by_category": dict(Counter(r["fraud_category"] for r in rows)),
        "final_push_rows": sum("e1_final_push" in r["source_file"].lower() for r in rows),
        "pku_proxy_rows": sum("pku" in r["source_file"].lower() for r in rows),
    }
    checks = {
        "total_ge_300": counts["total"] >= int(req["min_total"]),
        "model_min": all(v >= int(req["min_per_model"]) for v in counts["by_model"].values()) and {"qwen", "deepseek"} <= set(counts["by_model"]),
        "language_min": all(counts["by_language"].get(k, 0) >= int(req["min_per_language"]) for k in ("en", "zh")),
        "category_coverage": sum(v >= int(req["min_per_category"]) for v in counts["by_category"].values()) >= 5,
        "no_final_push": counts["final_push_rows"] == 0,
        "no_pku_proxy": counts["pku_proxy_rows"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "counts": counts}


def actionish(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in ["step", "template", "script", "first", "then", "具体", "步骤", "话术", "链接", "账户"])
