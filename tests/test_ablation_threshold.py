from frauddistill.eval.threshold_selection import select_qy_threshold_with_ablation_constraints


def test_qy_threshold_uses_dev_constraint_not_test_data():
    labels = ["safe", "safe", "unsafe", "unsafe"]
    chosen = select_qy_threshold_with_ablation_constraints(
        labels, [0.05, 0.20, 0.62, 0.80], [0.05, 0.10, 0.35, 0.80], 0.5,
        min_macro_gain=0.0, min_recall_gain=0.0, max_fpr_increase=0.2,
    )
    assert chosen["constraint_satisfied_on_dev"] is True
    assert 0.0 <= chosen["threshold"] <= 1.0
