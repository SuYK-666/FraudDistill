"""Retry policy: at most one retry for transport/parse failures (guide 13.3)."""
from __future__ import annotations

import asyncio

RETRYABLE = ("APITimeoutError", "APIStatusError", "APIConnectionError", "RateLimitError", "InternalServerError")


def is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    return any(t in name for t in RETRYABLE) or "429" in str(exc) or "5" == str(getattr(exc, "status_code", ""))[:1]


async def with_retry(coro_factory, attempts: int = 2, delay: float = 1.5):
    last = None
    for i in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < attempts - 1:
                await asyncio.sleep(delay * (i + 1))
    raise last