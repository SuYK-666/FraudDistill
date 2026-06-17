from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    do_sample: bool = False
    repetition_penalty: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)
