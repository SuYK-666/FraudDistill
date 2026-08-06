from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .schemas import FraudEvidence, RefusalEvidence, ContextEvidence, TeacherSignal


class LLMClient(Protocol):
    """Sync client protocol kept for legacy callers (offline heuristic mode)."""

    def complete_json(self, prompt: str, *, max_tokens: int = 768) -> dict[str, Any]:
        ...


SCHEMA_BY_AGENT = {
    "fraud_assistance_agent": FraudEvidence,
    "refusal_quality_agent": RefusalEvidence,
    "relevance_agent": ContextEvidence,
    "evidence_arbiter_agent": TeacherSignal,
}


@dataclass
class BaseAgent:
    name: str
    client: Any | None = None
    max_tokens: int = 180
    prompt_version: str = "exp3-v1"

    def build_prompt(self, sample: dict[str, Any]) -> str:
        raise NotImplementedError

    def build_messages(self, sample: dict[str, Any]) -> tuple[str, str]:
        """Return (system, user) messages. System is the fixed task prompt
        (cache-friendly prefix); user contains the dynamic q+y content."""
        prompt = self.build_prompt(sample)
        marker = "\n[USER QUESTION]\n"
        idx = prompt.find(marker)
        if idx > 0:
            return prompt[:idx], prompt[idx + 1 :]
        return prompt, json.dumps(sample, ensure_ascii=False)

    def heuristic_output(self, sample: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def validate(self, parsed: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate parsed JSON against the agent schema. Returns (ok, errors)."""
        schema_cls = SCHEMA_BY_AGENT.get(self.name)
        if schema_cls is None:
            return True, []
        try:
            schema_cls.model_validate(parsed)
            return True, []
        except Exception as exc:  # noqa: BLE001
            return False, [str(exc)]

    def repair_prompt(self, parsed: dict[str, Any], errors: list[str]) -> str:
        return (
            "Your previous JSON output failed schema validation. "
            "Return ONLY corrected JSON with all required fields as numbers in [0,1] "
            "or the allowed literals. Validation errors: "
            + "; ".join(errors[:3])
            + "\nPrevious output: " + json.dumps(parsed, ensure_ascii=False)[:1500]
        )

    def _heuristic_envelope(self, sample: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        parsed = self.heuristic_output(sample)
        ok, errors = self.validate(parsed)
        if not ok:
            parsed = self._fill_defaults(parsed)
        return {
            "status": "ok" if ok else "parse_repaired",
            "retry_count": 0,
            "raw_text": "",
            "parsed": parsed,
            "model_id": "offline_heuristic",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": {"input_hit": 0, "input_miss": 0, "output": 0},
            **parsed,
        }

    def _fill_defaults(self, parsed: dict[str, Any]) -> dict[str, Any]:
        schema_cls = SCHEMA_BY_AGENT.get(self.name)
        if schema_cls is None:
            return parsed
        try:
            merged = {}
            for fname, fdef in schema_cls.model_fields.items():
                if fname not in parsed or parsed[fname] is None:
                    if fdef.default is not None:
                        merged[fname] = fdef.default
            merged.update(parsed)
            return schema_cls.model_validate(merged).model_dump()
        except Exception:  # noqa: BLE001
            return parsed

    def run(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            return self._heuristic_envelope(sample)
        if not hasattr(self.client, "chat"):
            return self._run_sync(sample)
        return asyncio.run(self.run_async(sample, self.client))

    def _run_sync(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Legacy sync path for OpenAI-compatible clients exposing complete_json(prompt)."""
        import time as _time

        started = _time.perf_counter()
        prompt = self.build_prompt(sample)
        raw = ""
        parsed: dict[str, Any] = {}
        try:
            resp = self.client.complete_json(prompt, max_tokens=self.max_tokens)
            if isinstance(resp, dict):
                parsed = resp
            raw = str(resp)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "parse_failed",
                "retry_count": 0,
                "raw_text": f"{type(exc).__name__}: {exc}",
                "parsed": {},
                "model_id": str(getattr(self.client, "model", "legacy_sync")),
                "latency_ms": round((_time.perf_counter() - started) * 1000, 1),
                "usage": {"input_hit": 0, "input_miss": 0, "output": 0},
            }
        ok, errors = self.validate(parsed)
        if not ok:
            try:
                resp2 = self.client.complete_json(
                    self.repair_prompt(parsed, errors), max_tokens=self.max_tokens
                )
                parsed2 = resp2 if isinstance(resp2, dict) else {}
                ok2, _ = self.validate(parsed2)
                if ok2:
                    parsed, ok = parsed2, True
                else:
                    parsed = self._fill_defaults(parsed2) if parsed2 else {}
            except Exception:  # noqa: BLE001
                parsed = self._fill_defaults(parsed)
        return {
            "status": "ok" if ok else "parse_failed",
            "retry_count": 1 if not ok else 0,
            "raw_text": raw[:2000],
            "parsed": parsed,
            "model_id": str(getattr(self.client, "model", "legacy_sync")),
            "latency_ms": round((_time.perf_counter() - started) * 1000, 1),
            "usage": {"input_hit": 0, "input_miss": 0, "output": 0},
            "prompt_version": self.prompt_version,
            **parsed,
        }

    async def run_async(
        self,
        sample: dict[str, Any],
        client: Any | None = None,
        *,
        system_prompt_override: str | None = None,
    ) -> dict[str, Any]:
        """Async execution with the shared DeepSeekClient (cache + budget + retry).

        First JSON/schema failure triggers one repair call; second failure is
        marked parse_failed/abstain (never silently defaulted to safe/unsafe).

        system_prompt_override: when provided (skills runtime), the system
        message is the composed prompt (role + constraints + skills + schema)
        and the user message keeps the q+y serialization. The cache key covers
        the full system prompt, so skill digest changes invalidate the cache.
        """
        client = client or self.client
        started = time.perf_counter()
        if client is None:
            return self._heuristic_envelope(sample)
        if system_prompt_override is not None:
            system = system_prompt_override
            _, user = self.build_messages(sample)
        else:
            system, user = self.build_messages(sample)
        resp = await client.chat(
            system, user, prompt_version=self.prompt_version, max_tokens=self.max_tokens,
        )
        parsed: dict[str, Any] = resp.get("parsed") or {}
        # 2026-08-06 bugfix: the client returns parsed={} when the model output
        # fails JSON extraction (e.g. truncated at max_tokens). Schemas have
        # all-default fields, so validate({}) would silently pass and produce
        # all-zero evidence. Force the repair path and surface parse failures.
        ok, errors = self.validate(parsed)
        if resp.get("parse_ok") is False:
            ok = False
            errors = ["client json parse failed (possibly truncated output)"] + errors
        raw_text = str(resp.get("raw", ""))
        if not ok:
            repair_user = self.repair_prompt(parsed, errors)
            resp2 = await client.chat(
                system, repair_user, prompt_version=self.prompt_version + "-repair",
                max_tokens=self.max_tokens,
            )
            parsed2: dict[str, Any] = resp2.get("parsed") or {}
            ok2, errors2 = self.validate(parsed2)
            if resp2.get("parse_ok") is False:
                ok2 = False
                errors2 = ["client json parse failed (possibly truncated output)"] + errors2
            if ok2:
                parsed, ok = parsed2, True
            else:
                parsed = self._fill_defaults(parsed2) if parsed2 else {}
                raw_text = str(resp2.get("raw", "")) or raw_text
        status = "ok" if ok else "parse_failed"
        usage = resp.get("usage") or {"input_hit": 0, "input_miss": 0, "output": 0}
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "status": status,
            "retry_count": int(resp.get("retry_count", 0)),
            "raw_text": raw_text[:2000],
            "parsed": parsed,
            "model_id": str(resp.get("model", client.model)),
            "latency_ms": round(latency_ms, 1),
            "usage": usage,
            "prompt_version": self.prompt_version,
            **parsed,
        }
