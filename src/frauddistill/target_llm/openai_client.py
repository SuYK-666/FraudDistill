from __future__ import annotations

import json
import re
from typing import Any


class OpenAIJsonClient:
    """Small wrapper for official OpenAI-compatible JSON calls."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete_json(self, prompt: str, *, max_tokens: int = 768) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return parse_json_content(content)


class OpenAITextClient:
    """Small wrapper for OpenAI-compatible text calls."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete_text(self, prompt: str, *, max_tokens: int = 256) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
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
