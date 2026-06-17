from __future__ import annotations

from typing import Any

from frauddistill.data.schema import Label, RiskType


def map_binary_label(value: Any) -> Label:
    text = str(value).strip().lower()
    safe_values = {"safe", "pass", "faithful", "supported", "non-hallucinated", "non_hallucinated", "0", "false", "no"}
    unsafe_values = {"unsafe", "fail", "hallucinated", "unsupported", "1", "true", "yes"}
    if text in safe_values:
        return Label.SAFE
    if text in unsafe_values:
        return Label.UNSAFE
    raise ValueError(f"cannot map label value to safe/unsafe: {value!r}")


def first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def response_risk_type(label: Label, default_unsafe: RiskType) -> RiskType:
    return RiskType.NONE if label == Label.SAFE else default_unsafe
