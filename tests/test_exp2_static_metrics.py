# -*- coding: utf-8 -*-
"""Canonical metrics & statistics tests (guide 28.2-28.3, sections 9-14)."""
import numpy as np

from frauddistill.exp2_static_repair.frames import build_frame
from frauddistill.exp2_static_repair.metrics import (
    apply_holm,
    choose_score_direction,
    evaluate_binary,
    exact_mcnemar,
    macro_f1_fn,
    paired_group_bootstrap,
)


def make_frame(y_true, y_pred, y_score=None, groups=None):
    n = len(y_true)
    return build_frame(
        benchmark="test", track="t",
        sample_ids=[f"s{i}" for i in range(n)],
        group_ids=groups or [f"g{i}" for i in range(n)],
        y_true=y_true, y_pred=y_pred, y_score=y_score,
    )


def test_binary_metrics_reconstruct():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    metrics = evaluate_binary(make_frame(y_true, y_pred))
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert abs(metrics["macro_f1"] - 0.5) < 1e-12


def test_macro_f1_identity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.integers(0, 2, 200)
    m = evaluate_binary(make_frame(y, p))
    assert abs(m["macro_f1"] - (m["unsafe_f1"] + m["safe_f1"]) / 2) < 1e-12


def test_n_positive_matches_tp_fn():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([1, 0, 1, 1, 0, 0, 0])
    m = evaluate_binary(make_frame(y, p))
    assert m["n_positive"] == m["tp"] + m["fn"] == 3
    assert m["n"] == m["tn"] + m["fp"] + m["fn"] + m["tp"] == 7


def test_safe_unsafe_f1_reconstruct_macro():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([1, 1, 0, 0, 0, 1])
    m = evaluate_binary(make_frame(y, p))
    assert abs(m["macro_f1"] - (m["unsafe_f1"] + m["safe_f1"]) / 2) < 1e-12


def test_mcnemar_accuracy_delta_identity():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    a = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    b = np.array([1, 1, 0, 0, 0, 1, 1, 0])
    res = exact_mcnemar(y, a, b)
    assert abs(res["accuracy_delta"] - (res["baseline_wrong_teacher_right"] - res["baseline_right_teacher_wrong"]) / len(y)) < 1e-12


def test_mcnemar_known_case():
    # 0/8 discordant split => exact two-sided p = 2*(1/2)^8 * C(8,0) ... = 0.0078125
    y = np.array([1] * 8)
    a = np.array([0] * 8)  # baseline all wrong
    b = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # teacher all right
    res = exact_mcnemar(y, a, b)
    assert res["baseline_wrong_teacher_right"] == 8
    assert res["baseline_right_teacher_wrong"] == 0
    assert abs(res["raw_p"] - 0.0078125) < 1e-9


def test_holm_preserves_tiny_p_values():
    rows = apply_holm([{"raw_p": 1e-20}, {"raw_p": 0.0078125}, {"raw_p": 0.2}])
    assert rows[0]["holm_p"] < 1e-18
    assert rows[1]["holm_p"] < 0.05
    assert rows[2]["reject_h0"] is False


def test_bootstrap_uses_same_metric():
    rng = np.random.default_rng(1)
    n = 120
    groups = [f"g{i % 12}" for i in range(n)]
    y = rng.integers(0, 2, n)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    frame = make_frame(y, b, groups=groups)
    # replicate baseline as a separate prediction array
    res = paired_group_bootstrap(frame, a, b, reps=200, seed=42)
    obs = macro_f1_fn(y, b) - macro_f1_fn(y, a)
    assert abs(res["observed_delta"] - obs) < 1e-12
    assert abs(res["bootstrap_mean_delta"] - obs) < 0.15  # sanity: CI centered near obs


def test_group_bootstrap_preserves_groups():
    # swapping predictions within a group must not affect group-level delta
    y = np.array([1, 0, 1, 0])
    groups = np.array(["g1", "g1", "g2", "g2"])
    a = np.array([1, 0, 0, 1])
    b = np.array([0, 1, 1, 0])
    frame = build_frame(
        benchmark="t", track="t", sample_ids=["a", "b", "c", "d"],
        group_ids=list(groups), y_true=list(y), y_pred=list(b),
    )
    res = paired_group_bootstrap(frame, a, b, reps=100, seed=7)
    # both preds are wrong in both groups -> delta must be 0
    assert abs(res["observed_delta"]) < 1e-12


def test_choose_score_direction_detects_reversal():
    import pytest

    y = np.array([1, 1, 0, 0])
    s = np.array([0.1, 0.2, 0.8, 0.9])  # reversed (safe scores high)
    with pytest.raises(ValueError):
        choose_score_direction(y, s)
    flipped, info = choose_score_direction(y, s, allow_flip=True)
    assert info["flipped"] is True
    assert np.allclose(flipped, 1.0 - s)
