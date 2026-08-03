"""Missing-answer gate: answers that are absent are invalid_qy, never query-only judged."""
from __future__ import annotations

from dataclasses import dataclass

from frauddistill.data.input_normalizer import InputStatus


@dataclass(frozen=True)
class MissingAnswerResult:
    status: str          # valid | invalid_qy
    risk_label: str | None
    risk_type: str | None
    risk_score: float
    reason: str | None


def run_missing_answer_gate(status: InputStatus) -> MissingAnswerResult:
    if status in (InputStatus.MISSING_ANSWER, InputStatus.EMPTY_ANSWER, InputStatus.MISSING_QUERY, InputStatus.MALFORMED):
        return MissingAnswerResult(
            status="invalid_qy",
            risk_label="safe",
            risk_type="no_response",
            risk_score=0.0,
            reason=f"invalid input: {status.value}",
        )
    return MissingAnswerResult(status="valid", risk_label=None, risk_type=None, risk_score=0.0, reason=None)