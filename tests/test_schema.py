import pytest

from frauddistill.data.schema import FraudDistillSample, TeacherSignal


def test_sample_requires_q_and_answer():
    with pytest.raises(ValueError):
        FraudDistillSample(
            id="x",
            source="HaluBench",
            user_query="",
            target_model_answer="answer",
            gold_label="safe",
        )


def test_teacher_score_range():
    with pytest.raises(ValueError):
        TeacherSignal(id="x", teacher_label="unsafe", teacher_score=1.5, teacher_type="fraud_assistance")
