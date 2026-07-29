from frauddistill.eval.metrics import binary_metrics


def test_binary_metrics_smoke():
    metrics = binary_metrics(["safe", "unsafe"], ["safe", "unsafe"], [0.1, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["mcc"] == 1.0
    assert "brier" in metrics
