from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from frauddistill.target_llm.model_registry import enabled_models, load_model_registry
from frauddistill.target_llm.openai_client import OpenAITextClient
from frauddistill.target_llm.provider_config import get_provider_config, require_api_key
from frauddistill.utils.io import read_jsonl, write_jsonl


def generate_responses(
    input_file: str | Path,
    output_file: str | Path,
    registry_file: str | Path = "configs/models.yaml",
    *,
    limit: int | None = None,
    dry_run: bool = False,
    model_ids: set[str] | None = None,
) -> int:
    registry = load_model_registry(registry_file)
    generation = dict(registry.get("generation") or {})
    models = enabled_models(registry, role="target")
    if model_ids:
        models = [model for model in models if str(model.get("id")) in model_ids]
    if not models:
        raise ValueError("no enabled target models selected")

    prompts = list(read_jsonl(input_file))
    if limit is not None:
        prompts = prompts[:limit]
    rows = _iter_generation_rows(prompts, models, generation, dry_run=dry_run)
    return write_jsonl(output_file, rows)


def _iter_generation_rows(
    prompts: list[dict[str, Any]],
    models: list[dict[str, Any]],
    generation: dict[str, Any],
    *,
    dry_run: bool,
) -> Iterable[dict[str, Any]]:
    clients: dict[str, OpenAITextClient] = {}
    for prompt_row in tqdm(prompts, desc="generating target responses"):
        q = str(prompt_row.get("user_query") or prompt_row.get("q") or "")
        if not q.strip():
            raise ValueError(f"prompt row is missing user_query/q: {prompt_row.get('id')}")
        for model in models:
            model_id = str(model["id"])
            provider = str(model["provider"])
            started = time.perf_counter()
            error = None
            if dry_run:
                answer = "[DRY_RUN] target model response placeholder for pipeline validation."
            else:
                try:
                    client_key = f"{provider}:{model_id}"
                    if client_key not in clients:
                        config = get_provider_config(provider, model_id)
                        require_api_key(config)
                        clients[client_key] = OpenAITextClient(config.default_model, config.api_key, config.base_url)
                    answer = clients[client_key].complete_text(
                        q,
                        max_tokens=int(generation.get("max_tokens", 768)),
                        temperature=float(generation.get("temperature", 0.2)),
                        top_p=float(generation.get("top_p", 1.0)),
                        system_prompt=str(generation.get("system_prompt", "")),
                    )
                except Exception as exc:  # noqa: BLE001 - record API failures as data for reruns.
                    answer = ""
                    error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - started
            yield _build_generation_row(prompt_row, q, answer, model, generation, latency, error, dry_run)


def _build_generation_row(
    prompt_row: dict[str, Any],
    q: str,
    answer: str,
    model: dict[str, Any],
    generation: dict[str, Any],
    latency_seconds: float,
    error: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    model_id = str(model["id"])
    base_id = str(prompt_row.get("id") or prompt_row.get("sample_id") or _hash_text(q)[:12])
    metadata = dict(prompt_row.get("metadata") or {})
    metadata.update(
        {
            "source_prompt_id": base_id,
            "api_error": error,
            "dry_run": dry_run,
            "response_hash": _hash_text(answer) if answer else "",
        }
    )
    return {
        "id": f"{base_id}__{model_id.replace('/', '_')}",
        "source": prompt_row.get("source", "synthetic"),
        "source_prior": prompt_row.get("source_prior", "unknown"),
        "fraud_category": prompt_row.get("fraud_category", "unknown"),
        "fraud_subcategory": prompt_row.get("fraud_subcategory", "unknown"),
        "prompt_setting": prompt_row.get("prompt_setting", "unknown"),
        "language": prompt_row.get("language", "unknown"),
        "user_query": q,
        "target_model_answer": answer,
        "prompt_risk_label": prompt_row.get("prompt_risk_label", prompt_row.get("gold_label")),
        "prompt_risk_type": prompt_row.get("gold_risk_type", prompt_row.get("prompt_risk_type", "none")),
        "response_harm_label": None,
        "response_refusal_label": None,
        "pair_fraud_label": None,
        "label_provenance": "unlabeled_generation",
        "split": prompt_row.get("split", "unspecified"),
        "target_model": model_id,
        "target_provider": model["provider"],
        "target_model_version": model.get("version", "record_actual_version"),
        "generation_params": {
            "temperature": float(generation.get("temperature", 0.2)),
            "top_p": float(generation.get("top_p", 1.0)),
            "max_tokens": int(generation.get("max_tokens", 768)),
            "system_prompt": generation.get("system_prompt", ""),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": latency_seconds,
        "metadata": metadata,
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--registry_file", default="configs/models.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    count = generate_responses(
        args.input_file,
        args.output_file,
        args.registry_file,
        limit=args.limit,
        dry_run=args.dry_run,
        model_ids=set(args.models) if args.models else None,
    )
    print(f"wrote {count} generations to {args.output_file}")


if __name__ == "__main__":
    main()
