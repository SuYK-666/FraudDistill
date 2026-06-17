from __future__ import annotations

import json
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
        if self.client is None:
            return self.heuristic_output(sample)
        return self.client.complete_json(self.build_prompt(sample))
