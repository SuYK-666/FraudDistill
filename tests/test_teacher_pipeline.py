from frauddistill.teacher.align_teacher_with_gold import align_rows
from frauddistill.teacher.teacher_quality import teacher_quality_report


def test_align_teacher_marks_conflict():
    samples = [{"id": "x", "source": "synthetic", "user_query": "q", "target_model_answer": "a", "gold_label": "safe"}]
    signals = [{"id": "x", "teacher_label": "unsafe", "teacher_score": 0.8, "teacher_type": "fraud_assistance"}]
    aligned = align_rows(samples, signals)
    assert aligned[0]["teacher_gold_conflict"] is True
    assert aligned[0]["teacher_loss_weight"] == 0.1
    assert aligned[0]["teacher_label"] == "unsafe"


def test_teacher_quality_report_agreement():
    samples = [{"id": "x", "gold_label": "safe"}]
    signals = [{"id": "x", "teacher_label": "safe", "teacher_score": 0.2}]
    report = teacher_quality_report(samples, signals)
    assert report["agreement"] == 1.0
