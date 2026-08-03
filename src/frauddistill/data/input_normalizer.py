"""Input normalization for q+y safety evaluation (exp2 budgeted cascade).

Per the 30-yuan experiment guide (section 4): answers that are missing,
empty or literal None/null must NOT be treated as regular q+y samples.
They are flagged invalid_qy and excluded from main metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InputStatus(str, Enum):
    VALID = "valid"
    MISSING_QUERY = "missing_query"
    MISSING_ANSWER = "missing_answer"
    EMPTY_ANSWER = "empty_answer"
    MALFORMED = "malformed"


NULL_STRINGS = {"", "none", "null", "nan", "n/a", "<none>", "-"}


@dataclass(frozen=True)
class NormalizedSample:
    sample_id: str
    query: str
    answer: str
    status: InputStatus
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in NULL_STRINGS:
        return ""
    return text


def normalize_record(record: dict[str, Any]) -> NormalizedSample:
    """Normalize one unified row to (query, answer, status)."""
    sample_id = str(record.get("id") or record.get("sample_id") or "").strip()
    query = clean_text(record.get("query") or record.get("prompt"))
    answer = clean_text(record.get("answer") or record.get("response"))
    if not sample_id:
        status = InputStatus.MALFORMED
    elif not query:
        status = InputStatus.MISSING_QUERY
    elif not answer:
        # distinguish empty-string vs literal None for reporting
        raw_answer = record.get("answer")
        status = InputStatus.EMPTY_ANSWER if raw_answer is not None and str(raw_answer).strip() == "" else InputStatus.MISSING_ANSWER
    else:
        status = InputStatus.VALID
    language = str(record.get("language") or "unknown").lower()
    meta = {k: v for k, v in record.items() if k not in {"query", "prompt", "answer", "response"}}
    return NormalizedSample(sample_id=sample_id, query=query, answer=answer, status=status, language=language, metadata=meta)


def normalize_jsonl(path: str) -> tuple[list[NormalizedSample], dict[str, int]]:
    """Normalize a unified jsonl file; returns (samples, status_counts)."""
    import json

    samples: list[NormalizedSample] = []
    counts: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                counts["malformed"] = counts.get("malformed", 0) + 1
                continue
            s = normalize_record(rec)
            samples.append(s)
            counts[s.status.value] = counts.get(s.status.value, 0) + 1
    return samples, counts