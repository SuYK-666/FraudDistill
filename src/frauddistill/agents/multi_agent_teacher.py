from __future__ import annotations

from typing import Any

from .arbiter_agent import ArbiterAgent
from .fraud_assistance_agent import FraudAssistanceAgent
from .relevance_agent import RelevanceAgent
from .refusal_quality_agent import RefusalQualityAgent


class MultiAgentTeacher:
    """Fraud, refusal and relevance only. Failures stay explicit and never become safe."""

    def __init__(self, client=None):
        self.relevance_agent = RelevanceAgent(client)
        self.fraud_agent = FraudAssistanceAgent(client)
        self.refusal_agent = RefusalQualityAgent(client)
        self.arbiter_agent = ArbiterAgent()

    def run(self, sample: dict[str, Any]) -> dict[str, Any]:
        fraud = self.fraud_agent.run(sample)
        refusal = self.refusal_agent.run(sample)
        relevance = self.relevance_agent.run(sample)
        return self.arbiter_agent.run(sample, fraud, refusal, relevance)
