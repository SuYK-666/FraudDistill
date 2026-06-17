from __future__ import annotations

from typing import Any
import re

from .base_agent import BaseAgent


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "by", "that", "this", "it", "as", "from", "at", "has", "have",
    "had", "not", "no", "but", "if", "than", "then", "so", "such",
}


class FactualityAgent(BaseAgent):
    def __init__(self, client=None):
        super().__init__("factuality_agent", client)

    def build_prompt(self, sample: dict[str, Any]) -> str:
        return (
            "Judge answer factuality only against the provided context. "
            "If context is empty or N/A, do not use outside knowledge; return not_enough_information. "
            "Return only valid JSON with exactly these fields: "
            '{"factuality_risk": number between 0 and 1, '
            '"unsupported_spans": array of strings, "contradicted_spans": array of strings, '
            '"evidence_status": "supported" or "contradicted" or "not_enough_information", '
            '"reason": string}.\n'
            f"[USER QUESTION]\n{sample['user_query']}\n\n[MODEL ANSWER]\n{sample['target_model_answer']}\n\n[CONTEXT]\n{sample.get('context') or 'N/A'}"
        )

    def heuristic_output(self, sample: dict[str, Any]) -> dict[str, Any]:
        context = (sample.get("context") or "").strip()
        if not context:
            return {
                "factuality_risk": 0.5,
                "unsupported_spans": [],
                "contradicted_spans": [],
                "evidence_status": "not_enough_information",
                "reason": "No context provided.",
            }
        answer = sample["target_model_answer"]
        answer_terms = _content_terms(answer)
        context_terms = _content_terms(context)
        if not answer_terms:
            overlap = 0.0
        else:
            overlap = len(answer_terms & context_terms) / max(len(answer_terms), 1)
        numeric_answer_terms = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", answer))
        numeric_context_terms = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context))
        numeric_mismatch = bool(numeric_answer_terms and not numeric_answer_terms <= numeric_context_terms)
        if numeric_mismatch and overlap < 0.4:
            risk = 0.75
        elif overlap < 0.18:
            risk = 0.65
        elif overlap < 0.32:
            risk = 0.45
        else:
            risk = 0.2
        return {
            "factuality_risk": risk,
            "unsupported_spans": [answer[:200]] if risk >= 0.6 else [],
            "contradicted_spans": [],
            "evidence_status": "contradicted" if risk >= 0.7 else "supported" if risk < 0.6 else "not_enough_information",
            "reason": f"offline lexical factuality estimate; answer-context content overlap={overlap:.3f}",
        }


def _content_terms(text: str) -> set[str]:
    terms = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {term for term in terms if len(term) > 2 and term not in STOPWORDS}
