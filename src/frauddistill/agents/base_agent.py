from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    def complete_json(self, prompt: str, *, max_tokens: int = 768) -> dict[str, Any]:
        ...


@dataclass
class BaseAgent:
    name: str
    client: LLMClient | None = None

    def build_prompt(self, sample: dict[str, Any]) -> str:
        raise NotImplementedError

    def heuristic_output(self, sample: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def parse_output(self, raw: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)

    def run(self, sample: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        if self.client is None:
            parsed = self.heuristic_output(sample)
            return self._envelope("ok", 0, "offline_heuristic", parsed, "", started)
        raw_text = ""
        for attempt in range(3):
            try:
                parsed = self.client.complete_json(self.build_prompt(sample) + ("\nReturn strict JSON only." if attempt else ""))
                if parsed.get("parse_error"):
                    raw_text = str(parsed.get("raw_text", ""))
                    continue
                return self._envelope("ok", attempt, str(getattr(self.client, "model", "api")), parsed, raw_text, started)
            except Exception as exc:
                raw_text = f"{type(exc).__name__}: {exc}"[:2000]
        return self._envelope("abstain", 2, str(getattr(self.client, "model", "api")), {}, raw_text, started)

    @staticmethod
    def _envelope(status: str, retry_count: int, model_id: str, parsed: dict[str, Any], raw_text: str, started: float) -> dict[str, Any]:
        # Parsed fields are mirrored temporarily for compatibility with existing callers;
        # the stable machine-readable contract remains the `parsed` object.
        return {"status": status, "retry_count": retry_count, "raw_text": raw_text, "parsed": parsed, "model_id": model_id, "latency_ms": (time.perf_counter() - started) * 1000, **parsed}
