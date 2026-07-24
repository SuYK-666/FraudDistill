from frauddistill.labelers.aggregator import aggregate_votes


def test_aggregate_votes_marks_high_confidence_unsafe():
    result = aggregate_votes(
        [
            {"id": "x", "labeler": "a", "label": "unsafe", "score": 0.9, "risk_type": "fraud_assistance"},
            {"id": "x", "labeler": "b", "label": "unsafe", "score": 0.8, "risk_type": "fraud_assistance"},
            {"id": "x", "labeler": "c", "label": "unsafe", "score": 0.85, "risk_type": "refusal_failure"},
        ]
    )
    assert result["auto_label"]["binary"] == "unsafe"
    assert result["auto_label"]["confidence"] == "silver_high"
    assert result["auto_label"]["risk_type"] == "fraud_assistance"


def test_aggregate_votes_marks_ambiguous_middle_scores():
    result = aggregate_votes(
        [
            {"id": "x", "labeler": "a", "label": "unsafe", "score": 0.55},
            {"id": "x", "labeler": "b", "label": "safe", "score": 0.45},
        ]
    )
    assert result["auto_label"]["binary"] == "ambiguous"
