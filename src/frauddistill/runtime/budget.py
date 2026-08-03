"""Global API budget manager (guide section 12.4). Hard cap with stop-before-cap."""
from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetState:
    max_rmb: float = 27.0
    reserved_rmb: float = 3.0
    used_rmb: float = 0.0
    hard_stop: bool = True

    @property
    def effective_cap(self) -> float:
        return self.max_rmb - self.reserved_rmb

    def can_spend(self, estimated_rmb: float) -> bool:
        return self.used_rmb + estimated_rmb <= self.effective_cap

    def record(self, actual_rmb: float) -> None:
        self.used_rmb += actual_rmb
        if self.hard_stop and self.used_rmb > self.effective_cap:
            raise BudgetExceeded(
                f"API budget exceeded: used={self.used_rmb:.2f} cap={self.effective_cap:.2f} RMB"
            )


def estimate_rmb(input_hit_tokens: int, input_miss_tokens: int, output_tokens: int, prices: dict[str, float]) -> float:
    return (
        input_hit_tokens / 1_000_000 * prices["input_hit"]
        + input_miss_tokens / 1_000_000 * prices["input_miss"]
        + output_tokens / 1_000_000 * prices["output"]
    )