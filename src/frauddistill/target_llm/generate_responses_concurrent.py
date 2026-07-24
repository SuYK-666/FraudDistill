from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import threading
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from frauddistill.target_llm.generate_responses import _build_generation_row
from frauddistill.target_llm.model_registry import enabled_models, load_model_registry
from frauddistill.target_llm.openai_client import OpenAITextClient
from frauddistill.target_llm.provider_config import get_provider_config, require_api_key
from frauddistill.utils.io import read_jsonl, write_jsonl


def generate_responses_concurrent(
    input_file: str | Path,
    output_dir: str | Path,
    registry_file: str | Path,
    *,
    concurrency: int = 80,
    max_retries: int = 3,
    model_ids: set[str] | None = None,
) -> dict[str, Any]:
    registry = load_model_registry(registry_file)
    generation = dict(registry.get("generation") or {})
    models = enabled_models(registry, role="target")
    if model_ids:
        models = [model for model in models if str(model.get("id")) in model_ids]
    if not models:
        raise ValueError("no enabled target models selected")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / "generations.jsonl"
    success_file = out / "generations_success.jsonl"
    failure_file = out / "generation_failures.jsonl"
    summary_file = out / "SUMMARY.json"

    prompts = list(read_jsonl(input_file))
    done_ids = _completed_ids(output_file)
    tasks = []
    for prompt_row in prompts:
        q = str(prompt_row.get("user_query") or prompt_row.get("q") or "")
        if not q.strip():
            raise ValueError(f"prompt row is missing user_query/q: {prompt_row.get('id')}")
        for model in models:
            task_id = _task_id(prompt_row, model)
            if task_id not in done_ids:
                tasks.append((task_id, prompt_row, q, model))

    locks = {"all": threading.Lock(), "success": threading.Lock(), "failure": threading.Lock()}
    summary: dict[str, Any] = {
        "input_file": str(input_file),
        "registry_file": str(registry_file),
        "output_dir": str(out),
        "prompt_rows": len(prompts),
        "model_count": len(models),
        "expected_rows": len(prompts) * len(models),
        "already_completed": len(done_ids),
        "scheduled_rows": len(tasks),
        "concurrency": concurrency,
        "max_retries": max_retries,
        "models": [str(model["id"]) for model in models],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(_run_one, prompt_row, q, model, generation, max_retries): task_id
            for task_id, prompt_row, q, model in tasks
        }
        for future in tqdm(futures.as_completed(future_map), total=len(future_map), desc="full target generation"):
            row = future.result()
            _append_jsonl(output_file, row, locks["all"])
            if row.get("metadata", {}).get("api_error") or not row.get("target_model_answer"):
                _append_jsonl(failure_file, row, locks["failure"])
            else:
                _append_jsonl(success_file, row, locks["success"])

    final = _summarize_outputs(output_file, success_file, failure_file)
    summary.update(final)
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _run_one(prompt_row: dict[str, Any], q: str, model: dict[str, Any], generation: dict[str, Any], max_retries: int) -> dict[str, Any]:
    provider = str(model["provider"])
    model_id = str(model["id"])
    started = time.perf_counter()
    answer = ""
    error = None
    for attempt in range(max_retries + 1):
        try:
            config = get_provider_config(provider, model_id)
            require_api_key(config)
            client = OpenAITextClient(config.default_model, config.api_key, config.base_url)
            answer = client.complete_text(
                q,
                max_tokens=int(generation.get("max_tokens", 768)),
                temperature=float(generation.get("temperature", 0.2)),
                top_p=float(generation.get("top_p", 1.0)),
                system_prompt=str(generation.get("system_prompt", "")),
            )
            error = None
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(min(30.0, 1.5 * (2**attempt)))
    latency = time.perf_counter() - started
    row = _build_generation_row(prompt_row, q, answer, model, generation, latency, error, dry_run=False)
    row["metadata"]["retry_count"] = 0 if error is None else max_retries
    return row


def _task_id(prompt_row: dict[str, Any], model: dict[str, Any]) -> str:
    base_id = str(prompt_row.get("id") or prompt_row.get("sample_id"))
    model_id = str(model["id"]).replace("/", "_")
    return f"{base_id}__{model_id}"


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["id"]) for row in read_jsonl(path)}


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")


def _summarize_outputs(output_file: Path, success_file: Path, failure_file: Path) -> dict[str, Any]:
    all_rows = list(read_jsonl(output_file)) if output_file.exists() else []
    success_rows = list(read_jsonl(success_file)) if success_file.exists() else []
    failure_rows = list(read_jsonl(failure_file)) if failure_file.exists() else []
    by_model: dict[str, int] = {}
    failures_by_model: dict[str, int] = {}
    for row in all_rows:
        model = str(row.get("target_model", "unknown"))
        by_model[model] = by_model.get(model, 0) + 1
    for row in failure_rows:
        model = str(row.get("target_model", "unknown"))
        failures_by_model[model] = failures_by_model.get(model, 0) + 1
    return {
        "written_rows": len(all_rows),
        "success_rows": len(success_rows),
        "failure_rows": len(failure_rows),
        "by_model": by_model,
        "failures_by_model": failures_by_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--registry_file", required=True)
    parser.add_argument("--concurrency", type=int, default=80)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()
    summary = generate_responses_concurrent(
        args.input_file,
        args.output_dir,
        args.registry_file,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        model_ids=set(args.models) if args.models else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
