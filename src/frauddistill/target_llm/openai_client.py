from __future__ import annotations

import json
import re
import hashlib
from typing import Any


class OpenAIJsonClient:
    """Small wrapper for official OpenAI-compatible JSON calls."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: float = 60.0):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def complete_json_envelope(self, prompt: str, *, max_tokens: int = 768) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        choice = response.choices[0]
        content = choice.message.content or "{}"
        parsed = parse_json_content(content)
        raw_payload = _model_dump(response)
        raw_text = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str)
        return {
            "content_json": parsed,
            "raw_text": content,
            "requested_model": self.model,
            "response_model": str(getattr(response, "model", self.model) or self.model),
            "request_id": str(getattr(response, "id", "") or ""),
            "created": getattr(response, "created", None),
            "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
            "usage": _model_dump(getattr(response, "usage", None)),
            "raw_response_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        }

    def complete_json(self, prompt: str, *, max_tokens: int = 768) -> dict[str, Any]:
        return self.complete_json_envelope(prompt, max_tokens=max_tokens)["content_json"]


class OpenAITextClient:
    """Small wrapper for OpenAI-compatible text calls."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: float = 60.0):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 1.0,
        system_prompt: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else {"value": value}
            except json.JSONDecodeError:
                pass
    return {"parse_error": True, "raw_text": text[:1000]}


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value
