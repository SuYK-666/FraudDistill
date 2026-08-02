from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from frauddistill.target_llm.openai_client import OpenAITextClient
from frauddistill.target_llm.provider_config import get_provider_config, require_api_key

from .budget import hard_stop_decision
from .io import norm, read_jsonl, sha_text, write_jsonl


def request_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        "provider": task["target_provider"],
        "model": task["requested_target_model"],
        "q": norm(task["q_private"]),
        "temperature": task.get("temperature"),
        "top_p": task.get("top_p"),
        "max_tokens": task.get("max_tokens"),
        "extra_body": task.get("extra_body") or {},
    }
    return sha_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def cache_index(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ok: dict[str, dict[str, Any]] = {}
    bad: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        fp = row.get("request_fingerprint")
        if not fp or row.get("status") != "ok" or not row.get("text"):
            bad.append(row)
            continue
        ok[str(fp)] = row
    return ok, bad


def execute_tasks(
    tasks: list[dict[str, Any]],
    *,
    output_path: Path,
    ledger_path: Path,
    limits: dict[str, Any],
    run_api: bool,
    confirm_budget: bool,
    git_clean: bool,
    provider_factory: Callable[[str, str], Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if not run_api or not confirm_budget:
        return {"status": "DRY_RUN_NO_API", "created_calls": 0, "reason": "both --run-api and --confirm-budget are required"}
    if not git_clean:
        return {"status": "STOP_DIRTY_WORKTREE", "created_calls": 0, "reason": "git worktree must be clean before API calls"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = read_jsonl(ledger_path)
    budget = hard_stop_decision(ledger, limits)
    if budget["hard_stop"]:
        return {"status": "STOP_BUDGET", "created_calls": 0, "budget": budget}
    existing, bad = cache_index(output_path)
    created = 0
    skipped = 0
    clients: dict[tuple[str, str], Any] = {}
    selected = tasks[:limit] if limit else tasks
    for task in selected:
        fp = request_fingerprint(task)
        if fp in existing:
            skipped += 1
            continue
        if created and created % 200 == 0:
            budget = hard_stop_decision(read_jsonl(ledger_path), limits)
            if budget["hard_stop"]:
                return {"status": "STOP_BUDGET", "created_calls": created, "skipped_cache": skipped, "budget": budget}
        provider = task["target_provider"]
        model = task["requested_target_model"]
        client_key = (provider, model)
        if client_key not in clients:
            clients[client_key] = provider_factory(provider, model) if provider_factory else build_text_client(provider, model, task.get("timeout_seconds", 180))
        row = call_with_retry(clients[client_key], task, fp)
        append_jsonl(output_path, row)
        append_jsonl(ledger_path, ledger_row(row, task))
        created += 1
    return {"status": "DONE", "created_calls": created, "skipped_cache": skipped, "invalid_cache_rows": len(bad)}


def build_text_client(provider: str, model: str, timeout: float) -> OpenAITextClient:
    cfg = get_provider_config(provider, model)
    require_api_key(cfg)
    return OpenAITextClient(cfg.default_model, cfg.api_key, cfg.base_url, timeout=timeout)


def call_with_retry(client: Any, task: dict[str, Any], fp: str) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, 6):
        started = time.time()
        try:
            if hasattr(client, "complete_text_envelope"):
                result = client.complete_text_envelope(task)
            else:
                text = client.complete_text(
                    task["q_private"],
                    max_tokens=int(task.get("max_tokens", 1536)),
                    temperature=float(task.get("temperature", 0.2)),
                    top_p=float(task.get("top_p", 0.9)),
                    extra_body=task.get("extra_body"),
                )
                result = {"text": text, "response_model": getattr(client, "model", task["requested_target_model"]), "request_id": "", "usage": {}, "finish_reason": ""}
            return {
                **task,
                **result,
                "status": "ok" if result.get("text") else "empty",
                "request_fingerprint": fp,
                "latency_ms": int((time.time() - started) * 1000),
                "attempt": attempt,
            }
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            time.sleep(min(30, 2**attempt))
    return {**task, "status": "error", "error": last_error, "request_fingerprint": fp, "text": ""}


def ledger_row(row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    usage = row.get("usage") or {}
    return {
        "request_fingerprint": row.get("request_fingerprint"),
        "provider": task.get("target_provider"),
        "phase": task.get("phase"),
        "status": row.get("status"),
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
        "cost_cny": 0.0,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
