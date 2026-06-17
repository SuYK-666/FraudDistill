from __future__ import annotations

from frauddistill.data.schema import FraudDistillSample
from frauddistill.utils.text import build_detector_input


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
            }
        )
    return examples
