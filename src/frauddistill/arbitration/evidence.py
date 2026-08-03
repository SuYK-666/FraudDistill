"""Risk evidence dataclass shared across triage/specialists/arbiter."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskEvidence:
    prompt_harm_intent: float = 0.0
    answer_task_alignment: float = 0.0
    explicit_refusal: float = 0.0
    safe_redirection: float = 0.0
    actionable_assistance: float = 0.0
    leakage_after_refusal: float = 0.0
    trust_facilitation: float = 0.0
    general_harmful_compliance: float = 0.0
    anti_fraud_education: float = 0.0
    # specialist-only fields (default None -> not present)
    direct_actionability: float | None = None
    harmful_detail_density: float | None = None
    intent_alignment: float | None = None
    risk_amplification: float | None = None
    should_refuse: float | None = None
    did_refuse: float | None = None
    refusal_is_clear: float | None = None
    partial_compliance: float | None = None
    leakage_before_refusal: float | None = None
    safe_probability: float | None = None
    unsafe_probability: float | None = None
    primary_risk_type: str | None = None
    risk_domain: str = "unknown"          # fraud | general_safety | benign | unknown
    provisional_label: str = "uncertain"  # safe | unsafe | uncertain
    confidence: float = 0.0
    evidence_spans: list[str] = field(default_factory=list)
    specialist_conflict: bool = False
    privacy_sensitive_query: bool = False


def clamp01(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))