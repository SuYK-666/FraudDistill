from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent


class RelevanceAgent(BaseAgent):
    def __init__(self, client=None):
        super().__init__("relevance_agent", client)

    def build_prompt(self, sample: dict[str, Any]) -> str:
        return (
            "Judge whether the target model answer addresses the user question. "
            "Return only valid JSON with exactly these fields: "
            '{"relevance_score": number between 0 and 1, '
            '"is_answering_user_intent": boolean, "reason": string}.\n'
            "Important: judge the answer in the context of the question; do not judge the user alone.\n"
            f"[USER QUESTION]\n{sample['user_query']}\n\n[MODEL ANSWER]\n{sample['target_model_answer']}"
        )

    def heuristic_output(self, sample: dict[str, Any]) -> dict[str, Any]:
        answer = sample["target_model_answer"].lower()
        evasive = any(token in answer for token in ["i don't know", "cannot answer", "不清楚", "无法回答"])
        return {
            "relevance_score": 0.35 if evasive else 0.8,
            "is_answering_user_intent": not evasive,
            "reason": "offline heuristic relevance estimate",
        }
