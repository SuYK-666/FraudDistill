# -*- coding: utf-8 -*-
"""Neural-student dataset construction (guide 11, 12, 20.1).

Extends the existing linear student builders with:
- four-class label mapping (safe / fraud_assistance / refusal_failure / over_refusal);
- q+y input template with optional multi-turn context (guide 12.1/12.2);
- head-tail truncation instead of naive tail truncation (guide 12.5);
- teacher soft distribution + gold-priority blend (guide 11.3/11.4);
- sample weights (guide 16.2) and pair metadata.
"""
from __future__ import annotations

from frauddistill.data.schema import FraudDistillSample
from frauddistill.utils.text import build_detector_input
from .model import masked_pair_text

LABEL_TO_ID = {"safe": 0, "fraud_assistance": 1, "refusal_failure": 2, "over_refusal": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

SUBTYPE_TO_TYPE = {
    "direct_fraud": "fraud_assistance",
    "trust_facilitation": "fraud_assistance",
    "harmful_compliance": "fraud_assistance",
    "clean_refusal_to_fraud": "refusal_failure",
    "partial_leakage": "refusal_failure",
    "toxic": "refusal_failure",
    "aegis_harmful": "refusal_failure",
    "over_refusal": "over_refusal",
}


def map_gold_type(gold_label: str, gold_type: str, subtype: str, teacher_type: str = "safe",
                  teacher_unsafe: bool = True) -> str:
    """Best-effort four-class gold mapping (guide 11.1).

    Specific gold_type wins; otherwise subtype table; otherwise the teacher's
    type (high-confidence teacher signal) for unsafe rows; safe stays safe.
    """
    if str(gold_label) == "safe":
        return "safe"
    g = str(gold_type or "")
    if g in LABEL_TO_ID:
        return g
    if subtype in SUBTYPE_TO_TYPE:
        return SUBTYPE_TO_TYPE[subtype]
    if teacher_unsafe and str(teacher_type) in LABEL_TO_ID:
        return str(teacher_type)
    return "fraud_assistance"


def build_student_examples(rows: list[dict], input_mode: str = "q_y_c") -> list[dict]:
    examples = []
    for row in rows:
        sample = FraudDistillSample.model_validate(row)
        examples.append(
            {
                "id": sample.id,
                "text": build_detector_input(sample.user_query, sample.target_model_answer, sample.context, input_mode),
                "label": sample.gold_label.value,
                "risk_type": sample.gold_risk_type.value if sample.gold_risk_type else "none",
                "teacher_score": row.get("teacher_score"),
                "teacher_type": row.get("teacher_type"),
                "teacher_confidence": row.get("teacher_confidence"),
                "teacher_gold_agree": row.get("teacher_label") == sample.gold_label.value if row.get("teacher_label") else None,
                "source": sample.source.value,
                "category": row.get("fraud_category", "unknown"),
                "language": sample.language,
            }
        )
    return examples


def build_masked_student_examples(rows: list[dict], mode: str = "q_y") -> list[dict]:
    return [{**row, "text": masked_pair_text(str(row.get("user_query", "")), str(row.get("target_model_answer", "")), mode)} for row in rows]


# ---------------------------------------------------------------------------
# Neural student (guide 12, 20.1)
# ---------------------------------------------------------------------------
TASK_PREFIX = "Task: Detect whether the answer creates a fraud-related behavior error.\n\n"


def neural_input_text(query: str, answer: str, context: str | None = None) -> str:
    """Guide 12.1/12.2 input template. No benchmark/gold/teacher metadata."""
    parts = [TASK_PREFIX]
    if context and str(context).strip():
        parts.append("[CONTEXT]\n" + str(context).strip() + "\n\n")
    parts.append(f"[QUERY]\n{query}\n\n[ANSWER]\n{answer}")
    return "".join(parts)


def teacher_distribution(teacher_score: float, teacher_type: str) -> list[float]:
    """Guide 11.3: p(safe) = 1 - s; remaining mass on the teacher type."""
    dist = [0.0, 0.0, 0.0, 0.0]
    s = float(teacher_score)
    t = str(teacher_type)
    if t not in LABEL_TO_ID:
        t = "fraud_assistance" if s >= 0.5 else "safe"
    if t == "safe":
        dist[0] = 1.0
    else:
        dist[0] = max(0.0, 1.0 - s)
        dist[LABEL_TO_ID[t]] = min(1.0, s)
    total = sum(dist)
    if total <= 0:
        dist[0] = 1.0
    return [x / max(sum(dist), 1e-9) for x in dist]


def blend_target_distribution(gold_label: str, gold_type_id: int, teacher_score: float,
                              teacher_type: str, gold_source: str, teacher_tier: str) -> list[float]:
    """Guide 11.4: gold-priority blend of one-hot gold and teacher distribution."""
    gold_one_hot = [0.0, 0.0, 0.0, 0.0]
    gold_one_hot[gold_type_id] = 1.0
    tdist = teacher_distribution(teacher_score, teacher_type)
    if str(gold_label) == "safe":
        gold_w, teacher_w = 0.80, 0.20
    elif str(gold_source) == "official":
        gold_w, teacher_w = 0.80, 0.20
    elif str(gold_source) == "audit":
        gold_w, teacher_w = 0.75, 0.25
    else:  # procedural_weak
        gold_w, teacher_w = 0.60, 0.40
    if str(teacher_tier) == "low":
        teacher_w = min(teacher_w, 0.10)
    dist = [gold_w * g + teacher_w * t for g, t in zip(gold_one_hot, tdist)]
    s = sum(dist)
    return [x / max(s, 1e-9) for x in dist]


def sample_weight(gold_source: str, teacher_confidence: float, agent_agreement: float,
                  conflict_flags: list | None, subtype: str) -> float:
    """Guide 16.2 sample weight."""
    w = 1.0
    if gold_source == "official":
        w *= 1.0
    elif gold_source == "audit":
        w *= 0.9
    else:
        w *= 0.65
    w *= 0.5 + 0.5 * float(teacher_confidence)
    w *= 0.75 + 0.25 * float(agent_agreement)
    if conflict_flags:
        w *= 0.6
    if subtype in ("partial_leakage", "context_flip", "over_refusal"):
        w *= 1.4
    elif subtype in ("hard_safe", "trust_facilitation", "quotation_analysis"):
        w *= 1.3
    return float(w)


def build_neural_examples(rows: list[dict], max_length: int = 1536,
                          use_teacher_soft: bool = True, use_pairwise: bool = True) -> list[dict]:
    """Build neural-student examples from manifest rows (guide 20.1 fields)."""
    out = []
    for row in rows:
        gold_label = str(row.get("gold_label", "safe"))
        gold_type = map_gold_type(
            gold_label,
            str(row.get("gold_type", "")),
            str(row.get("subtype", "")),
            str(row.get("teacher_type", "safe")),
            teacher_unsafe=(str(row.get("teacher_label", "safe")) == "unsafe"),
        )
        gold_type_id = LABEL_TO_ID[gold_type]
        ts = float(row.get("teacher_score", 0.5))
        tt = str(row.get("teacher_type", "safe"))
        tier = str(row.get("confidence_tier", "medium"))
        gold_source = str(row.get("gold_source", "procedural_weak"))
        tdist = blend_target_distribution(gold_label, gold_type_id, ts, tt, gold_source, tier) if use_teacher_soft \
            else [0.0, 0.0, 0.0, 0.0]
        if not use_teacher_soft:
            tdist[gold_type_id] = 1.0
        teacher_confidence = float(row.get("teacher_confidence", 0.5) or 0.5)
        teacher_label = str(row.get("teacher_label", "safe"))
        teacher_gold_agree = row.get("teacher_gold_agree")
        if teacher_gold_agree is None:
            teacher_gold_agree = (teacher_label == gold_label) or (teacher_label == "unsafe" and gold_label == "unsafe")
        out.append({
            "id": row["id"],
            "text": neural_input_text(str(row.get("user_query", "")), str(row.get("target_model_answer", "")),
                                      row.get("context")),
            "gold_label": gold_label,
            "gold_type_id": gold_type_id,
            "teacher_distribution": tdist,
            "teacher_score": ts,
            "teacher_type": tt,
            "teacher_confidence": teacher_confidence,
            "teacher_gold_agree": bool(teacher_gold_agree),
            "teacher_only": bool(row.get("teacher_only", False)),
            "source_bucket": str(row.get("source_bucket", "")),
            "sample_weight": (float(row["sample_weight"]) if row.get("sample_weight") is not None
                              else sample_weight(gold_source, teacher_confidence,
                                                 float(row.get("agent_agreement", 0.5)),
                                                 row.get("conflict_flags"), str(row.get("subtype", "")))),
            "pair_id": row.get("pair_id"),
            "group_id": row.get("group_id"),
            "template_family_id": row.get("template_family_id", row.get("group_id")),
            "source": row.get("source", ""),
            "subtype": row.get("subtype", ""),
            "language": row.get("language", ""),
            "max_length": max_length,
            "use_pairwise": bool(use_pairwise and row.get("pair_id")),
        })
    return out
