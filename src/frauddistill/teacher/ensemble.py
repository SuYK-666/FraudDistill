from __future__ import annotations

from typing import Any


def teacher_confidence(signal: dict[str, Any]) -> float:
    score = _float(signal.get("teacher_score"), 0.5)
    explicit = signal.get("teacher_confidence")
    if explicit is not None:
        return max(0.0, min(1.0, _float(explicit, abs(score - 0.5) * 2.0)))
    return max(0.0, min(1.0, abs(score - 0.5) * 2.0))


def enrich_teacher_signal(signal: dict[str, Any], sample: dict[str, Any], teacher_name: str) -> dict[str, Any]:
    raw = signal.get("raw_agent_outputs") or {}
    subscores = {
        "relevance_risk": 1.0 - _float((raw.get("relevance") or {}).get("relevance_score"), 0.5),
        "factuality_risk": _float((raw.get("factuality") or {}).get("factuality_risk"), _float(signal.get("teacher_score"), 0.5)),
        "fraud_assistance_risk": _float((raw.get("fraud") or {}).get("fraud_assistance_risk"), 0.0),
        "refusal_failure_risk": 1.0 if (raw.get("refusal") or {}).get("refusal_failure") else 0.0,
    }
    spans = signal.get("teacher_spans") or signal.get("risky_spans") or []
    score = _float(signal.get("teacher_score"), 0.5)
    label = str(signal.get("teacher_label", "unsafe" if score >= 0.5 else "safe"))
    enriched = {
        **signal,
        "id": sample["id"],
        "teacher_name": teacher_name,
        "teacher_label": label,
        "teacher_score": score,
        "teacher_type": str(signal.get("teacher_type", "none")),
        "teacher_confidence": teacher_confidence(signal),
        "teacher_gold_agree": label == sample.get("gold_label"),
        "subscores": subscores,
        "risky_spans": spans,
        "rationale": str(signal.get("teacher_rationale", signal.get("rationale", ""))),
    }
    return enriched


def ensemble_teacher_signals(qwen: dict[str, Any], deepseek: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    q_score = _float(qwen.get("teacher_score"), 0.5)
    d_score = _float(deepseek.get("teacher_score"), 0.5)
    q_conf = teacher_confidence(qwen)
    d_conf = teacher_confidence(deepseek)
    avg_score = (q_score * q_conf + d_score * d_conf) / max(q_conf + d_conf, 1e-6)
    q_label = str(qwen.get("teacher_label", "unsafe" if q_score >= 0.5 else "safe"))
    d_label = str(deepseek.get("teacher_label", "unsafe" if d_score >= 0.5 else "safe"))
    if q_label == d_label:
        label = q_label
        confidence = max(q_conf, d_conf, abs(avg_score - 0.5) * 2.0)
    elif abs(q_conf - d_conf) >= 0.2:
        stronger = qwen if q_conf > d_conf else deepseek
        label = str(stronger.get("teacher_label"))
        confidence = max(q_conf, d_conf) * 0.75
    else:
        label = "unsafe" if avg_score >= 0.5 else "safe"
        confidence = abs(avg_score - 0.5) * 2.0

    subscores = {}
    for key in {"relevance_risk", "factuality_risk", "fraud_assistance_risk", "refusal_failure_risk"}:
        subscores[key] = (
            _float((qwen.get("subscores") or {}).get(key), 0.0) * q_conf
            + _float((deepseek.get("subscores") or {}).get(key), 0.0) * d_conf
        ) / max(q_conf + d_conf, 1e-6)

    signal = {
        "id": sample["id"],
        "teacher_name": "ensemble_qwen_deepseek",
        "teacher_label": label,
        "teacher_score": avg_score,
        "teacher_type": _choose_type(qwen, deepseek, q_conf, d_conf, label),
        "teacher_confidence": max(0.0, min(1.0, confidence)),
        "teacher_gold_agree": label == sample.get("gold_label"),
        "subscores": subscores,
        "risky_spans": (qwen.get("risky_spans") or []) + (deepseek.get("risky_spans") or []),
        "rationale": "Ensemble of Qwen and DeepSeek multi-agent teacher signals.",
        "raw_teacher_outputs": {"qwen": qwen, "deepseek": deepseek},
    }
    return signal


def compute_teacher_weight(gold_label: str, teacher_label: str, confidence: float) -> float:
    agree = gold_label == teacher_label
    if agree and confidence >= 0.75:
        return 1.0
    if agree and confidence >= 0.50:
        return 0.5
    if (not agree) and confidence >= 0.75:
        return 0.1
    return 0.0


def _choose_type(qwen: dict[str, Any], deepseek: dict[str, Any], q_conf: float, d_conf: float, label: str) -> str:
    if label == "safe":
        return "none"
    q_type = str(qwen.get("teacher_type", "none"))
    d_type = str(deepseek.get("teacher_type", "none"))
    if q_type == d_type:
        return q_type
    return q_type if q_conf >= d_conf else d_type


def _float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
