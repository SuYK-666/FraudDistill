from __future__ import annotations

from frauddistill.data.schema import FraudDistillSample
from frauddistill.utils.text import build_detector_input
from .model import masked_pair_text


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
