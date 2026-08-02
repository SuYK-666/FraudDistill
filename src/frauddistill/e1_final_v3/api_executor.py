from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from frauddistill.target_llm.openai_client import OpenAIJsonClient, OpenAITextClient
from frauddistill.target_llm.provider_config import get_provider_config, require_api_key

from .budget import hard_stop_decision
from .io import norm, read_jsonl, sha_text, write_jsonl


def request_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        'provider': task['target_provider'],
        'model': task['requested_target_model'],
        'q': norm(task['q_private']),
        'temperature': task.get('temperature'),
        'top_p': task.get('top_p'),
        'max_tokens': task.get('max_tokens'),
        'extra_body': task.get('extra_body') or {},
    }
    return sha_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def json_task_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        'provider': task['target_provider'],
        'model': task['requested_target_model'],
        'prompt': task['judge_prompt'],
        'temperature': task.get('temperature'),
        'max_tokens': task.get('max_tokens'),
        'extra_body': task.get('extra_body') or {},
    }
    return sha_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _is_json_row(row: dict[str, Any]) -> bool:
    return 'content_json' in row and 'judge_prompt' in row


def cache_index(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ok: dict[str, dict[str, Any]] = {}
    bad: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        fp = row.get('request_fingerprint')
        if _is_json_row(row):
            valid = bool(fp and row.get('status') == 'ok' and row.get('content_json'))
        else:
            valid = bool(fp and row.get('status') == 'ok' and row.get('text'))
        if not valid:
            bad.append(row)
            continue
        ok[str(fp)] = row
    return ok, bad


def _gate(run_api: bool, confirm_budget: bool, git_clean: bool) -> tuple[dict[str, Any], bool]:
    if not run_api or not confirm_budget:
        return {'status': 'DRY_RUN_NO_API', 'created_calls': 0, 'reason': 'both --run-api and --confirm-budget are required'}, False
    if not git_clean:
        return {'status': 'STOP_DIRTY_WORKTREE', 'created_calls': 0, 'reason': 'git worktree must be clean before API calls'}, False
    return {}, True


def _pending_tasks(tasks: list[dict[str, Any]], output_path: Path, fingerprint_fn: Callable[[dict[str, Any]], str]) -> tuple[list[tuple[dict[str, Any], str]], int, int]:
    existing, bad = cache_index(output_path)
    pending: list[tuple[dict[str, Any], str]] = []
    skipped = 0
    for task in tasks:
        fp = fingerprint_fn(task)
        if fp in existing:
            skipped += 1
            continue
        pending.append((task, fp))
    return pending, skipped, len(bad)


def _pool_execute(
    pending: list[tuple[dict[str, Any], str]],
    *,
    output_path: Path,
    ledger_path: Path,
    limits: dict[str, Any],
    client_factory: Callable[[dict[str, Any]], Any],
    call_fn: Callable[[Any, dict[str, Any], str], dict[str, Any]],
    pricing: dict[str, Any] | None,
    concurrency_by_provider: dict[str, int] | None,
) -> dict[str, Any]:
    created = 0
    clients: dict[tuple[str, str], Any] = {}
    if concurrency_by_provider:
        provider_limits = {str(k).lower(): max(1, int(v)) for k, v in concurrency_by_provider.items()}
        max_workers = max(1, sum(provider_limits.values()))
        semaphores = {provider: threading.Semaphore(limit) for provider, limit in provider_limits.items()}

        def run_one(task_fp: tuple[dict[str, Any], str]) -> tuple[dict[str, Any], dict[str, Any]]:
            task, fp = task_fp
            provider = task['target_provider']
            model = task['requested_target_model']
            with semaphores.get(str(provider).lower(), threading.Semaphore(1)):
                with threading.Lock():
                    key = (provider, model)
                    if key not in clients:
                        clients[key] = client_factory(task)
                    client = clients[key]
                return task, call_fn(client, task, fp)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_one, item) for item in pending]
            for future in as_completed(futures):
                if created and created % 200 == 0:
                    budget = hard_stop_decision(read_jsonl(ledger_path), limits)
                    if budget['hard_stop']:
                        return {
                            'status': 'STOP_BUDGET',
                            'created_calls': created,
                            'skipped_cache': len(pending) - created,
                            'budget': budget,
                            'concurrency_by_provider': provider_limits,
                        }
                task, row = future.result()
                append_jsonl(output_path, row)
                append_jsonl(ledger_path, ledger_row(row, task, pricing))
                created += 1
        return {
            'status': 'DONE',
            'created_calls': created,
            'skipped_cache': len(pending) - created,
            'concurrency_by_provider': provider_limits,
        }
    for task, fp in pending:
        if created and created % 200 == 0:
            budget = hard_stop_decision(read_jsonl(ledger_path), limits)
            if budget['hard_stop']:
                return {'status': 'STOP_BUDGET', 'created_calls': created, 'skipped_cache': len(pending) - created, 'budget': budget}
        provider = task['target_provider']
        model = task['requested_target_model']
        key = (provider, model)
        if key not in clients:
            clients[key] = client_factory(task)
        row = call_fn(clients[key], task, fp)
        append_jsonl(output_path, row)
        append_jsonl(ledger_path, ledger_row(row, task, pricing))
        created += 1
    return {'status': 'DONE', 'created_calls': created, 'skipped_cache': len(pending) - created}


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
    concurrency_by_provider: dict[str, int] | None = None,
    pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate, ok = _gate(run_api, confirm_budget, git_clean)
    if not ok:
        return gate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = read_jsonl(ledger_path)
    budget = hard_stop_decision(ledger, limits)
    if budget['hard_stop']:
        return {'status': 'STOP_BUDGET', 'created_calls': 0, 'budget': budget}
    selected = tasks[:limit] if limit else tasks
    pending, skipped, bad = _pending_tasks(selected, output_path, request_fingerprint)
    result = _pool_execute(
        pending,
        output_path=output_path,
        ledger_path=ledger_path,
        limits=limits,
        client_factory=lambda task: (provider_factory(task['target_provider'], task['requested_target_model']) if provider_factory else build_text_client(task['target_provider'], task['requested_target_model'], task.get('timeout_seconds', 180))),
        call_fn=call_with_retry,
        pricing=pricing,
        concurrency_by_provider=concurrency_by_provider,
    )
    result['skipped_cache'] = skipped
    result['invalid_cache_rows'] = bad
    return result


def execute_json_tasks(
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
    concurrency_by_provider: dict[str, int] | None = None,
    pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate, ok = _gate(run_api, confirm_budget, git_clean)
    if not ok:
        return gate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = read_jsonl(ledger_path)
    budget = hard_stop_decision(ledger, limits)
    if budget['hard_stop']:
        return {'status': 'STOP_BUDGET', 'created_calls': 0, 'budget': budget}
    selected = tasks[:limit] if limit else tasks
    pending, skipped, bad = _pending_tasks(selected, output_path, json_task_fingerprint)
    result = _pool_execute(
        pending,
        output_path=output_path,
        ledger_path=ledger_path,
        limits=limits,
        client_factory=lambda task: (provider_factory(task['target_provider'], task['requested_target_model']) if provider_factory else build_json_client(task['target_provider'], task['requested_target_model'], task.get('timeout_seconds', 180))),
        call_fn=json_call_with_retry,
        pricing=pricing,
        concurrency_by_provider=concurrency_by_provider,
    )
    result['skipped_cache'] = skipped
    result['invalid_cache_rows'] = bad
    return result


def build_text_client(provider: str, model: str, timeout: float) -> OpenAITextClient:
    cfg = get_provider_config(provider, model)
    require_api_key(cfg)
    return OpenAITextClient(cfg.default_model, cfg.api_key, cfg.base_url, timeout=timeout)


def build_json_client(provider: str, model: str, timeout: float) -> OpenAIJsonClient:
    cfg = get_provider_config(provider, model)
    require_api_key(cfg)
    return OpenAIJsonClient(cfg.default_model, cfg.api_key, cfg.base_url, timeout=timeout)


def call_with_retry(client: Any, task: dict[str, Any], fp: str) -> dict[str, Any]:
    last_error = ''
    for attempt in range(1, 6):
        started = time.time()
        try:
            if hasattr(client, 'complete_text_envelope'):
                result = client.complete_text_envelope(task)
            else:
                text = client.complete_text(
                    task['q_private'],
                    max_tokens=int(task.get('max_tokens', 1536)),
                    temperature=float(task.get('temperature', 0.2)),
                    top_p=float(task.get('top_p', 0.9)),
                    extra_body=task.get('extra_body'),
                )
                result = {'text': text, 'response_model': getattr(client, 'model', task['requested_target_model']), 'request_id': '', 'usage': {}, 'finish_reason': ''}
            return {
                **task,
                **result,
                'status': 'ok' if result.get('text') else 'empty',
                'request_fingerprint': fp,
                'latency_ms': int((time.time() - started) * 1000),
                'attempt': attempt,
            }
        except Exception as exc:
            last_error = f'{exc.__class__.__name__}: {exc}'
            time.sleep(min(30, 2 ** attempt))
    return {**task, 'status': 'error', 'error': last_error, 'request_fingerprint': fp, 'text': ''}


def json_call_with_retry(client: Any, task: dict[str, Any], fp: str) -> dict[str, Any]:
    last_error = ''
    for attempt in range(1, 6):
        started = time.time()
        try:
            result = client.complete_json_envelope(
                task['judge_prompt'],
                max_tokens=int(task.get('max_tokens', 1024)),
                temperature=float(task.get('temperature', 0.0)),
                extra_body=task.get('extra_body'),
            )
            content = result.get('content_json') or {}
            parsed_ok = isinstance(content, dict) and not content.get('parse_error')
            return {
                **task,
                **result,
                'status': 'ok' if parsed_ok else 'bad_json',
                'request_fingerprint': fp,
                'latency_ms': int((time.time() - started) * 1000),
                'attempt': attempt,
            }
        except Exception as exc:
            last_error = f'{exc.__class__.__name__}: {exc}'
            time.sleep(min(30, 2 ** attempt))
    return {**task, 'status': 'error', 'error': last_error, 'request_fingerprint': fp, 'content_json': {}}


def ledger_row(row: dict[str, Any], task: dict[str, Any], pricing: dict[str, Any] | None = None) -> dict[str, Any]:
    usage = row.get('usage') or {}
    provider = task.get('target_provider')
    model = task.get('requested_target_model')
    cost_cny = 0.0
    if pricing and isinstance(usage, dict):
        table = pricing.get(str(provider).lower(), {})
        prices = table.get(str(model), {})
        if prices:
            in_tokens = float(usage.get('prompt_tokens', 0) or 0)
            out_tokens = float(usage.get('completion_tokens', 0) or 0)
            cost_cny = in_tokens / 1_000_000 * float(prices.get('input', 0) or 0) + out_tokens / 1_000_000 * float(prices.get('output', 0) or 0)
    return {
        'request_fingerprint': row.get('request_fingerprint'),
        'provider': provider,
        'phase': task.get('phase'),
        'status': row.get('status'),
        'prompt_tokens': usage.get('prompt_tokens') if isinstance(usage, dict) else None,
        'completion_tokens': usage.get('completion_tokens') if isinstance(usage, dict) else None,
        'cost_cny': round(cost_cny, 6),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + chr(10))