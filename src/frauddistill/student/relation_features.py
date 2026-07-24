from __future__ import annotations

import re

import numpy as np


FEATURE_NAMES = ["refusal", "refusal_with_steps", "step_count", "credential_or_payment", "urgency", "safe_education", "query_answer_overlap", "template_or_script", "imperative_ratio", "query_fraud_context"]

_REFUSAL = ("cannot help", "can't help", "i can't", "无法帮助", "不能帮助", "拒绝")
_SAFE = ("prevent", "recognize", "report", "stay safe", "防范", "识别", "举报", "反诈")
_FRAUD = ("phishing", "impersonat", "credential", "verification code", "诈骗", "钓鱼", "冒充", "验证码", "转账", "押金")


def relation_feature_matrix(queries: list[str], answers: list[str]) -> np.ndarray:
    result = []
    for query, answer in zip(queries, answers):
        q, a = query.lower(), answer.lower()
        refusal = _contains(a, _REFUSAL)
        steps = len(re.findall(r"(?:^|\s)(?:\d+[.)]|first|then|step|首先|然后|第[一二三四五])", a))
        safe = _contains(a, _SAFE)
        q_tokens, a_tokens = set(re.findall(r"[a-z0-9_\u4e00-\u9fff]+", q)), set(re.findall(r"[a-z0-9_\u4e00-\u9fff]+", a))
        overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))
        credentials = _contains(a, ("password", "credential", "verification", "验证码", "账号", "银行卡", "payment", "转账"))
        urgency = _contains(a, ("urgent", "immediately", "freeze", "立即", "马上", "紧急", "冻结"))
        template = _contains(a, ("template", "script", "模板", "话术"))
        imperative = len(re.findall(r"\b(?:send|use|ask|tell|click|provide|pretend)\b|请|使用|发送|提供|伪装", a)) / max(1, len(re.findall(r"\w+|[\u4e00-\u9fff]", a)))
        result.append([float(refusal), float(refusal and steps > 0), min(steps / 5, 1), float(credentials), float(urgency), float(safe), overlap, float(template), imperative, float(_contains(q, _FRAUD))])
    return np.asarray(result, dtype=np.float32)


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)
