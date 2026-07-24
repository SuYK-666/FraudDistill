from frauddistill.data.group_split import assert_no_leakage, grouped_train_dev_test_split


def test_group_split_keeps_cases_together():
    rows = [{"id": str(i), "user_query": f"q{i // 2}", "target_model_answer": "a", "gold_label": "unsafe" if i % 3 else "safe", "metadata": {"fraudr1_raw_id": str(i // 2)}} for i in range(30)]
    splits = grouped_train_dev_test_split(rows)
    assert sum(map(len, splits.values())) == len(rows)
    assert_no_leakage(splits)
