from frauddistill.student.dataset import build_student_examples
from frauddistill.student.losses import distillation_weights, total_loss_value


def test_student_example_input_modes():
    rows = [{"id": "x", "source": "synthetic", "user_query": "q", "target_model_answer": "a", "gold_label": "safe"}]
    ex = build_student_examples(rows, "q_y")
    assert "[USER QUESTION]" in ex[0]["text"]
    assert "[MODEL ANSWER]" in ex[0]["text"]


def test_loss_weight_helpers():
    assert distillation_weights(True) == 0.1
    assert total_loss_value(1.0, score_loss=1.0, weights={"gold": 1.0, "teacher_score": 0.3}) == 1.3
