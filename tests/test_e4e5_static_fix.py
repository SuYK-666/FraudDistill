# -*- coding: utf-8 -*-
"""Unit tests for E4/E5 static-fix metric and statistics repairs."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from frauddistill.e4e5_v2.metrics import binary_metrics_raw
from frauddistill.e4e5_v2.cluster_bootstrap import exact_mcnemar, holm_correct, paired_cluster_bootstrap


def test_macro_f1_is_two_class_macro_average():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    pred = (s >= 0.5).astype(int)
    m = binary_metrics_raw(y, s, pred=pred)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 200
    assert abs(m["macro_f1"] - f1_score(y, pred, average="macro")) < 1e-8
    assert abs(m["f1_unsafe"] - f1_score(y, pred, pos_label=1)) < 1e-8
    assert abs(m["f1_safe"] - f1_score(y, pred, pos_label=0)) < 1e-8


def test_auroc_requires_only_gold_two_classes():
    y = np.array([1, 0, 0, 1, 1, 0])
    s = np.array([0.9, 0.2, 0.1, 0.8, 0.7, 0.3])
    pred = np.ones(6, dtype=int)  # single-class predictions
    m = binary_metrics_raw(y, s, pred=pred)
    assert "auroc" in m and "auprc" in m


def test_mcnemar_exact_p_small_disagreement():
    n = 1200
    pa = np.array([1] * 85 + [0] * 5 + [0] * (n - 90))
    pb = np.array([0] * 85 + [1] * 5 + [0] * (n - 90))
    y = np.array([1] * 90 + [0] * (n - 90))
    mc = exact_mcnemar(pa, pb, y)
    assert mc["b"] == 5 and mc["c"] == 85
    assert mc["p_exact"] > 0 and abs(mc["p_exact"] - 7.532842063484042e-20) < 1e-24


def test_holm_uses_cumulative_max():
    out = holm_correct([0.01, 0.02, 0.05], ["a", "b", "c"])
    assert abs(out["a"]["p_holm"] - 0.03) < 1e-12
    assert abs(out["b"]["p_holm"] - 0.04) < 1e-12
    assert abs(out["c"]["p_holm"] - 0.05) < 1e-12
    # monotone non-decreasing along raw-p order
    raw = [0.1, 0.001]
    o = holm_correct(raw, ["x", "y"])
    assert o["y"]["p_holm"] <= o["x"]["p_holm"] + 1e-12
    assert o["y"]["p_holm"] >= o["y"]["p_raw"]


def test_bootstrap_ci_contains_point_and_no_nan():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 90)
    s = rng.random(90)
    fams = [f"f{i // 3}" for i in range(90)]
    res = paired_cluster_bootstrap(y, s, s + 0.05, (0.5, 0.5), fams, replicates=2000,
                                   metrics=["macro_f1", "f1_unsafe", "recall", "fpr", "mcc"],
                                   include_auroc_auprc=True)
    for m in ("macro_f1", "f1_unsafe", "recall", "fpr", "mcc", "auroc", "auprc"):
        b = res[m]
        assert not np.isnan(b["mean_diff"]) and not np.isnan(b["ci95"][0]) and not np.isnan(b["ci95"][1])
        assert b["ci95"][0] <= b["mean_diff"] <= b["ci95"][1]
