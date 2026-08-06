"""Async DeepSeek client with JSON mode, request cache, budget checks and usage ledger.

Implements experiment guide sections 12-13: fixed system prefix for cache hits,
content-hash cache with prompt_version in the key, at most one retry, budget
hard cap with stop-before-cap, per-call usage (cache hit/miss tokens) and latency.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

from openai import AsyncOpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import api_keys  # noqa: E402

from frauddistill.runtime.budget import BudgetExceeded, BudgetState, estimate_rmb
from frauddistill.runtime.cache import RequestCache
from frauddistill.runtime.retry import is_retryable
from frauddistill.exp2_static_repair.offline_guard import assert_online_allowed  # noqa: E402

# DeepSeek pricing (RMB per 1M tokens), snapshot 2026-08-03 (guide section 12.4).
DEEPSEEK_PRICES = {"input_hit": 0.02, "input_miss": 1.00, "output": 2.00}
DEEPSEEK_MODEL = "deepseek-v4-flash"


class UsageLedger:
    """Async-safe cumulative usage/cost counters."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.input_hit_tokens = 0
        self.input_miss_tokens = 0
        self.output_tokens = 0
        self.retries = 0
        self.errors = 0
        self.latencies_ms: list[float] = []

    async def add(self, *, calls=0, cache_hit=0, cache_miss=0, input_hit=0, input_miss=0, output=0, retries=0, errors=0, latency_ms=None):
        async with self._lock:
            self.calls += calls
            self.cache_hits += cache_hit
            self.cache_misses += cache_miss
            self.input_hit_tokens += input_hit
            self.input_miss_tokens += input_miss
            self.output_tokens += output
            self.retries += retries
            self.errors += errors
            if latency_ms is not None:
                self.latencies_ms.append(latency_ms)

    def rmb(self, prices: dict[str, float]) -> float:
        return estimate_rmb(self.input_hit_tokens, self.input_miss_tokens, self.output_tokens, prices)

    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = min(len(s) - 1, int(0.95 * len(s)))
        return s[idx]

    def snapshot(self, prices: dict[str, float]) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "input_hit_tokens": self.input_hit_tokens,
            "input_miss_tokens": self.input_miss_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "errors": self.errors,
            "mean_latency_ms": round(sum(self.latencies_ms) / max(len(self.latencies_ms), 1), 1),
            "p95_latency_ms": round(self.p95_latency_ms(), 1),
            "estimated_cost_rmb": round(self.rmb(prices), 4),
        }


class DeepSeekClient:
    """OpenAI-compatible async client for the budgeted cascade."""

    def __init__(
        self,
        model: str = DEEPSEEK_MODEL,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        json_mode: bool = True,
        thinking_disabled: bool = True,
        timeout_seconds: float = 90,
        max_retries: int = 1,
        prices: dict[str, float] | None = None,
        cache: RequestCache | None = None,
        budget: BudgetState | None = None,
        concurrency: int = 120,
    ):
        assert_online_allowed()
        self.model = model
        self.json_mode = json_mode
        self.thinking_disabled = thinking_disabled
        self.max_retries = max_retries
        self.prices = prices or dict(DEEPSEEK_PRICES)
        self.cache = cache
        self.budget = budget or BudgetState()
        self.ledger = UsageLedger()
        self._client = AsyncOpenAI(
            api_key=api_key or api_keys.DEEPSEEK_API_KEY,
            base_url=base_url or api_keys.DEEPSEEK_BASE_URL,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self._sem = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """Robust JSON extraction: direct parse, then substring between first { and last }."""
        import json

        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(raw[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None

    def _token_estimate(self, *texts: str) -> int:
        return sum(len(t) for t in texts) // 2 + 8

    def _check_budget(self, system_prompt: str, user_prompt: str, max_tokens: int) -> None:
        est_miss = self._token_estimate(system_prompt, user_prompt)
        est = estimate_rmb(0, est_miss, max_tokens, self.prices)
        if not self.budget.can_spend(est):
            raise BudgetExceeded(
                f"API budget would exceed cap: used={self.budget.used_rmb:.2f} "
                f"cap={self.budget.effective_cap:.2f} RMB (request est {est:.4f})"
            )

    async def _call_once(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.thinking_disabled:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = await self._client.chat.completions.create(**kwargs)
        usage = resp.usage
        content = (resp.choices[0].message.content or "").strip()
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None) or {}
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        input_miss = max(0, input_tokens - cached)
        return {
            "ok": True,
            "parsed": content,
            "raw": content,
            "model": getattr(resp, "model", self.model),
            "usage": {"input_hit": cached, "input_miss": input_miss, "output": output_tokens},
            "request_id": getattr(resp, "id", ""),
        }

    # ------------------------------------------------------------------ public
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        prompt_version: str,
        max_tokens: int = 180,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> dict:
        """One cache-aware, budget-checked JSON completion.

        Returns dict: {ok, parsed, raw, route, latency_ms, retry_count, usage, error}.
        parsed is a dict on success (json_mode=True), raw is the content string.
        """
        started = time.perf_counter()
        key = None
        if self.cache is not None:
            key = RequestCache.key(self.model, prompt_version, system_prompt, user_prompt)
            hit = self.cache.get(key)
            if hit is not None:
                await self.ledger.add(cache_hit=1, latency_ms=0.0)
                out = dict(hit)
                out["route"] = "cache_hit"
                out["latency_ms"] = 0.0
                out["ok"] = True
                out["retry_count"] = 0
                return out
            if os.environ.get("EXP3_DRY_RUN") == "1":
                # Cache-replay safety mode: never touch the API. Any cache miss is
                # reported as a per-call failure so offline replays cost zero RMB.
                await self.ledger.add(cache_miss=1, errors=1, latency_ms=0.0)
                try:
                    logp = os.environ.get("EXP3_DRY_RUN_LOG") or os.path.join(os.getcwd(), "_dryrun_misses.log")
                    with open(logp, "a", encoding="utf-8") as _f:
                        _f.write(json.dumps({
                            "prompt_version": prompt_version,
                            "system": system_prompt[:120],
                            "user": user_prompt[:160],
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                return {
                    "ok": False,
                    "parse_ok": False,
                    "parsed": {},
                    "raw": "dry_run_cache_miss",
                    "route": "dry_run",
                    "latency_ms": 0.0,
                    "retry_count": 0,
                    "usage": {"input_hit": 0, "input_miss": 0, "output": 0},
                    "error": "dry_run_cache_miss",
                    "prompt_version": prompt_version,
                }

        self._check_budget(system_prompt, user_prompt, max_tokens)
        last_err = ""
        last_resp: dict | None = None
        retry_count = 0
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                async with self._sem:
                    resp = await self._call_once(system_prompt, user_prompt, max_tokens, temperature)
                parsed = self._extract_json(resp["parsed"]) if json_mode else {"text": resp["parsed"]}
                if json_mode and parsed is None:
                    # parse failure: retry once (guide 13.3)
                    if attempt < self.max_retries:
                        retry_count += 1
                        await asyncio.sleep(1.2 * (attempt + 1))
                        continue
                    last_resp = resp
                    break
                last_resp = resp
                break
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries and is_retryable(exc):
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                break

        latency_ms = (time.perf_counter() - started) * 1000
        parse_ok = bool(last_resp is not None and last_resp.get("ok") and (not json_mode or parsed is not None))
        if last_resp is not None and last_resp.get("ok"):
            raw = last_resp["parsed"]
            usage = last_resp.get("usage", {"input_hit": 0, "input_miss": 0, "output": 0})
            await self.ledger.add(
                calls=1,
                cache_miss=1,
                input_hit=usage["input_hit"],
                input_miss=usage["input_miss"],
                output=usage["output"],
                retries=retry_count,
                latency_ms=latency_ms,
            )
            cost = estimate_rmb(usage["input_hit"], usage["input_miss"], usage["output"], self.prices)
            self.budget.record(cost)
            record = {
                "ok": True,
                "parse_ok": parse_ok,
                "parsed": parsed if parse_ok else {},
                "raw": raw,
                "route": "api",
                "latency_ms": latency_ms,
                "retry_count": retry_count,
                "usage": usage,
                "error": "" if parse_ok else "json_parse_failed",
                "prompt_version": prompt_version,
            }
            if key is not None and parse_ok:
                self.cache.put(key, record)
            return record

        # failure
        await self.ledger.add(calls=1, cache_miss=1, errors=1, retries=retry_count, latency_ms=latency_ms)
        return {
            "ok": False,
            "parse_ok": False,
            "parsed": {},
            "raw": last_err,
            "route": "api",
            "latency_ms": latency_ms,
            "retry_count": retry_count,
            "usage": {"input_hit": 0, "input_miss": 0, "output": 0},
            "error": last_err or "empty_content",
            "prompt_version": prompt_version,
        }
