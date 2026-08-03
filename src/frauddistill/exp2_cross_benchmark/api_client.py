"""Async OpenAI-compatible API helpers with retries, JSON mode and cost tracking."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

from openai import AsyncOpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import api_keys  # noqa: E402

# DeepSeek pricing (RMB per 1M tokens), snapshot 2026-08-03 per experiment doc.
DEEPSEEK_PRICES = {"input_hit": 0.02, "input_miss": 1.00, "output": 2.00}
QWEN_PRICES = {"input_hit": 0.0, "input_miss": 0.4, "output": 1.2}  # qwen-plus approx, snapshot


class ApiConfig:
    def __init__(self, provider: str, model: str, *, base_url: str | None = None, api_key: str | None = None):
        self.provider = provider
        self.model = model
        if provider == "deepseek":
            self.base_url = base_url or api_keys.DEEPSEEK_BASE_URL
            self.api_key = api_key or api_keys.DEEPSEEK_API_KEY
            self.prices = DEEPSEEK_PRICES
        elif provider == "qwen":
            self.base_url = base_url or api_keys.QWEN_BASE_URL
            self.api_key = api_key or api_keys.QWEN_API_KEY
            self.prices = QWEN_PRICES
        else:
            raise ValueError(f"unknown provider {provider}")


class CostLedger:
    """Thread-safe cost/usage ledger."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.input_hit = 0
        self.input_miss = 0
        self.output = 0
        self.calls = 0
        self.retries = 0
        self.errors = 0

    async def add(self, *, input_hit=0, input_miss=0, output=0, retries=0, errors=0, calls=1):
        async with self.lock:
            self.input_hit += input_hit
            self.input_miss += input_miss
            self.output += output
            self.retries += retries
            self.errors += errors
            self.calls += calls

    def rmb(self, prices) -> float:
        return (self.input_hit * prices["input_hit"] + self.input_miss * prices["input_miss"] + self.output * prices["output"]) / 1e6

    def snapshot(self, prices) -> dict:
        return {
            "calls": self.calls,
            "input_hit_tokens": self.input_hit,
            "input_miss_tokens": self.input_miss,
            "output_tokens": self.output,
            "retries": self.retries,
            "errors": self.errors,
            "estimated_cost_rmb": round(self.rmb(prices), 4),
        }


async def complete_json(
    client: AsyncOpenAI,
    cfg: ApiConfig,
    ledger: CostLedger,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 220,
    temperature: float = 0.0,
    json_mode: bool = True,
    thinking_disabled: bool = True,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run one completion and return {ok, parsed, raw, model, usage, latency_ms, retry_count}."""
    started = time.perf_counter()
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if thinking_disabled and cfg.provider == "deepseek":
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = await client.chat.completions.create(**kwargs)
            usage = resp.usage
            content = (resp.choices[0].message.content or "").strip()
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            cached = int(getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
            input_miss = max(0, input_tokens - cached)
            await ledger.add(input_hit=cached, input_miss=input_miss, output=output_tokens, retries=attempt)
            parsed: dict[str, Any] = {}
            parse_ok = False
            if json_mode:
                try:
                    parsed = json.loads(content)
                    parse_ok = True
                except json.JSONDecodeError:
                    parse_ok = False
            else:
                parse_ok = True
                parsed = {"text": content}
            return {
                "ok": True,
                "parse_ok": parse_ok,
                "parsed": parsed,
                "raw": content,
                "model": resp.model,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "retry_count": attempt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached,
                "request_id": getattr(resp, "id", ""),
            }
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                await asyncio.sleep(1.5 * (attempt + 1) + 0.5 * (attempt**2))
    await ledger.add(errors=1)
    return {
        "ok": False,
        "parse_ok": False,
        "parsed": {},
        "raw": last_err,
        "model": cfg.model,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "retry_count": max_retries,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "request_id": "",
    }


def make_client(cfg: ApiConfig) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=120, max_retries=0)
