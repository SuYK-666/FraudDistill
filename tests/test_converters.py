from frauddistill.data.convert_halubench import convert_rows


def test_halubench_converter_maps_fields():
    rows = [{"id": "a", "question": "Q?", "answer": "A.", "context": "C.", "label": "FAIL"}]
    converted = convert_rows(rows, split="train")
    assert converted[0]["user_query"] == "Q?"
    assert converted[0]["gold_label"] == "unsafe"
    assert converted[0]["gold_risk_type"] == "hallucination"
