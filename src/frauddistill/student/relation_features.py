from __future__ import annotations

import re

import numpy as np


UNARY_FEATURE_NAMES = [
    "length_log",
    "token_count_log",
    "has_refusal",
    "has_safe_education",
    "has_fraud_terms",
    "has_credential_terms",
    "has_template_terms",
    "imperative_ratio",
]

CROSS_FEATURE_NAMES = [
    "token_overlap_q_share",
    "token_overlap_y_share",
    "length_ratio_minmax",
    "both_have_fraud_terms",
    "q_fraud_y_credentials",
    "q_fraud_y_template",
    "q_fraud_y_refusal",
    "q_safe_y_refusal",
    "refusal_with_prompt_overlap",
    "step_terms_with_q_fraud",
]

_REFUSAL = ("cannot help", "can't help", "i can't", "unable to help", "do not", "will not", "refuse")
_SAFE = ("prevent", "recognize", "report", "stay safe", "law enforcement", "victim", "avoid")
_FRAUD = ("phishing", "impersonat", "credential", "verification code", "transfer", "payment", "account", "scam", "fraud")
_CREDENTIAL = ("password", "credential", "verification", "bank", "payment", "transfer", "account", "otp")
_TEMPLATE = ("template", "script", "message", "email", "sms", "pretend", "spoof")
_IMPERATIVE = re.compile(r"\b(?:send|use|ask|tell|click|provide|pretend|write|create|call|transfer)\b", re.I)
_STEP = re.compile(r"(?:^|\s)(?:\d+[.)]|first|then|step|next)\b", re.I)
_TOKEN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")


def unary_text_features(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        lower = text.lower()
        tokens = _tokens(lower)
        token_count = max(1, len(tokens))
        rows.append(
            [
                np.log1p(len(text)) / 10.0,
                np.log1p(token_count) / 5.0,
                float(_contains(lower, _REFUSAL)),
                float(_contains(lower, _SAFE)),
                float(_contains(lower, _FRAUD)),
                float(_contains(lower, _CREDENTIAL)),
                float(_contains(lower, _TEMPLATE)),
                min(len(_IMPERATIVE.findall(lower)) / token_count, 1.0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def cross_qy_features(queries: list[str], answers: list[str]) -> np.ndarray:
    result = []
    for query, answer in zip(queries, answers):
        q = query.lower()
        y = answer.lower()
        q_tokens = set(_tokens(q))
        y_tokens = set(_tokens(y))
        overlap = q_tokens & y_tokens
        q_len = max(1, len(query))
        y_len = max(1, len(answer))
        q_fraud = _contains(q, _FRAUD)
        q_safe = _contains(q, _SAFE)
        y_refusal = _contains(y, _REFUSAL)
        y_credentials = _contains(y, _CREDENTIAL)
        y_template = _contains(y, _TEMPLATE)
        result.append(
            [
                len(overlap) / max(1, len(q_tokens)),
                len(overlap) / max(1, len(y_tokens)),
                min(q_len, y_len) / max(q_len, y_len),
                float(q_fraud and _contains(y, _FRAUD)),
                float(q_fraud and y_credentials),
                float(q_fraud and y_template),
                float(q_fraud and y_refusal),
                float(q_safe and y_refusal),
                float(y_refusal and len(overlap) > 0),
                float(q_fraud and bool(_STEP.search(y))),
            ]
        )
    return np.asarray(result, dtype=np.float32)


def relation_feature_matrix(queries: list[str], answers: list[str]) -> np.ndarray:
    """Backward-compatible name for strictly cross-side features."""

    return cross_qy_features(queries, answers)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)
