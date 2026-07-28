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
    "has_money_transfer_terms",
    "has_impersonation_terms",
    "has_job_scam_terms",
    "has_romance_scam_terms",
    "has_defense_terms",
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
    "q_risk_y_action",
    "q_benign_y_refusal",
    "q_fraud_y_money_transfer",
    "q_fraud_y_impersonation",
    "q_fraud_y_defense",
]

_REFUSAL = ("cannot help", "can't help", "i can't", "unable to help", "do not", "will not", "refuse", "抱歉", "不能帮助", "无法协助", "不应", "不要")
_SAFE = ("prevent", "recognize", "report", "stay safe", "law enforcement", "victim", "avoid", "防范", "识别", "举报", "报警", "受害者", "保护")
_FRAUD = ("phishing", "impersonat", "credential", "verification code", "transfer", "payment", "account", "scam", "fraud", "钓鱼", "冒充", "验证码", "转账", "付款", "账户", "诈骗", "欺诈")
_CREDENTIAL = ("password", "credential", "verification", "bank", "payment", "transfer", "account", "otp", "密码", "凭证", "验证码", "银行", "银行卡", "账户")
_TEMPLATE = ("template", "script", "message", "email", "sms", "pretend", "spoof", "模板", "话术", "短信", "邮件", "伪装", "冒充")
_MONEY = ("wire transfer", "bank transfer", "payment", "gift card", "crypto", "wallet", "转账", "汇款", "付款", "数字货币", "钱包", "银行卡")
_IMPERSONATION = ("impersonate", "pretend", "spoof", "fake identity", "冒充", "伪装", "假扮", "身份")
_JOB = ("job", "recruit", "resume", "interview", "招聘", "求职", "简历", "面试", "兼职")
_ROMANCE = ("romance", "dating", "relationship", "lover", "恋爱", "交友", "感情", "网恋")
_DEFENSE = ("detect", "prevent", "verify", "report", "block", "识别", "防范", "核实", "举报", "拦截")
_IMPERATIVE = re.compile(r"\b(?:send|use|ask|tell|click|provide|pretend|write|create|call|transfer|collect|convince)\b|发送|使用|点击|提供|冒充|编写|创建|转账|收集|说服", re.I)
_STEP = re.compile(r"(?:^|\s)(?:\d+[.)]|first|then|step|next)\b|步骤|首先|然后|接着|下一步", re.I)
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
                float(_contains(lower, _MONEY)),
                float(_contains(lower, _IMPERSONATION)),
                float(_contains(lower, _JOB)),
                float(_contains(lower, _ROMANCE)),
                float(_contains(lower, _DEFENSE)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def cross_qy_features(queries: list[str], answers: list[str]) -> np.ndarray:
    result = []
    for query, answer in zip(queries, answers):
        if not query.strip() or not answer.strip():
            result.append([0.0] * len(CROSS_FEATURE_NAMES))
            continue
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
        y_action = bool(_IMPERATIVE.search(y) or _STEP.search(y))
        y_defense = _contains(y, _DEFENSE) or _contains(y, _SAFE)
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
                float(q_fraud and y_action),
                float(not q_fraud and y_refusal),
                float(q_fraud and _contains(y, _MONEY)),
                float(q_fraud and _contains(y, _IMPERSONATION)),
                float(q_fraud and y_defense and not y_action),
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
