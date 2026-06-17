from __future__ import annotations

from typing import Any

from .arbiter_agent import ArbiterAgent
from .factuality_agent import FactualityAgent
from .fraud_assistance_agent import FraudAssistanceAgent
from .relevance_agent import RelevanceAgent
from .refusal_quality_agent import RefusalQualityAgent


class MultiAgentTeacher:
    def __init__(self, client=None):
        self.relevance_agent = RelevanceAgent(client)
        self.factuality_agent = FactualityAgent(client)
        self.fraud_agent = FraudAssistanceAgent(client)
        self.refusal_agent = RefusalQualityAgent(client)
        self.arbiter_agent = ArbiterAgent()

    def run(self, sample: dict[str, Any]) -> dict[str, Any]:
        relevance = _normalize_relevance(self.relevance_agent.run(sample))
        factuality = _normalize_factuality(self.factuality_agent.run(sample))
        fraud = _normalize_fraud(self.fraud_agent.run(sample))
        refusal = _normalize_refusal(self.refusal_agent.run(sample))
        return self.arbiter_agent.run(sample, relevance, factuality, fraud, refusal)


def _score(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _normalize_relevance(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "relevance_score": _score(row.get("relevance_score"), 0.5),
        "is_answering_user_intent": bool(row.get("is_answering_user_intent", True)),
        "reason": str(row.get("reason", row.get("relevance_reason", ""))),
    }


def _normalize_factuality(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("evidence_status", "not_enough_information"))
    if status not in {"supported", "contradicted", "not_enough_information"}:
        status = "not_enough_information"
    return {
        "factuality_risk": _score(row.get("factuality_risk"), 0.5),
        "unsupported_spans": _strings(row.get("unsupported_spans")),
        "contradicted_spans": _strings(row.get("contradicted_spans")),
        "evidence_status": status,
        "reason": str(row.get("reason", "")),
    }


def _normalize_fraud(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "fraud_assistance_risk": _score(row.get("fraud_assistance_risk", row.get("fraud_risk")), 0.0),
        "fraud_type": str(row.get("fraud_type", "none")),
        "actionability": _score(row.get("actionability"), 0.0),
        "fraud_spans": _strings(row.get("fraud_spans")),
        "reason": str(row.get("reason", "")),
    }


def _normalize_refusal(row: dict[str, Any]) -> dict[str, Any]:
    should_refuse = bool(row.get("should_refuse", False))
    did_refuse = bool(row.get("did_refuse", False))
    return {
        "refusal_quality": _score(row.get("refusal_quality"), 0.5),
        "should_refuse": should_refuse,
        "did_refuse": did_refuse,
        "refusal_failure": bool(row.get("refusal_failure", should_refuse and not did_refuse)),
        "over_refusal": bool(row.get("over_refusal", (not should_refuse) and did_refuse)),
        "reason": str(row.get("reason", "")),
    }
