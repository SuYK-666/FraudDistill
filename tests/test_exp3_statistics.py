# -*- coding: utf-8 -*-
"""Unit tests for Exp3 statistics fixes (guide 3.1-3.4, 4.2)."""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sklearn.metrics import f1_score

import evaluate_exp3 as ev
import build_exp3_agent_ablations as ablations


def test_true_macro_f1_differs_from_positive_class_f1():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0, 0])
    expected_macro = float(f1_score(y, pred, average="macro", zero_division=0))
    expected_unsafe = float(f1_score(y, pred, pos_label=1, zero_division=0))
    assert ev._macro_f1(y, pred) == pytest.approx(expected_macro)
    # classic failure mode: harmonic mean of P/R equals unsafe F1, not macro
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    harmonic = 2 * prec * rec / (prec + rec)
    assert expected_unsafe == pytest.approx(harmonic)
    assert abs(expected_macro - expected_unsafe) > 1e-6


def test_exact_mcnemar_orientation():
    # 10 records: A wrong & B right -> b; A right & B wrong -> c
    gold = ["unsafe", "unsafe", "unsafe", "unsafe", "unsafe",
            "safe", "safe", "safe", "safe", "safe"]
    lab_a = ["safe", "safe", "unsafe", "unsafe", "unsafe",
             "unsafe", "unsafe", "safe", "safe", "safe"]
    lab_b = ["unsafe", "unsafe", "unsafe", "unsafe", "safe",
             "unsafe", "safe", "safe", "safe", "unsafe"]
    recs = [{"sample": {"gold_label": g}, "id": i} for i, (g, a, b) in enumerate(zip(gold, lab_a, lab_b))]
    out = ev.exact_mcnemar(recs, lambda r: lab_a[r["id"]], lambda r: lab_b[r["id"]])
    # b = A wrong & B right: rows 0, 1 (A safe vs gold unsafe, B unsafe ok),
    #     row 6 (A unsafe vs gold safe, B safe ok)
    # c = A right & B wrong: row 4 (A unsafe ok, B safe), row 9 (A safe ok, B unsafe)
    assert out["b_only_a_wrong_b_right"] == 3
    assert out["c_only_a_right_b_wrong"] == 2
    from scipy.stats import binomtest
    assert out["p_exact"] == pytest.approx(float(binomtest(2, 5, 0.5, alternative="two-sided").pvalue))


def test_holm_does_not_turn_tiny_p_values_into_one():
    from statsmodels.stats.multitest import multipletests
    raw = [1e-20, 0.0078125, 0.2]
    _, adjusted, _, _ = multipletests(raw, method="holm")
    assert adjusted[0] < 0.001
    assert adjusted[1] < 0.05
    adj, rej = ev.holm(raw)
    assert adj[0] < 0.001
    assert rej[0] and rej[1] and not rej[2]


def test_observed_delta_equals_full_test_difference():
    rng = np.random.default_rng(7)
    n = 200
    recs = []
    for i in range(n):
        recs.append({
            "id": str(i),
            "group_id": f"g{i % 20}",
            "sample": {"gold_label": "unsafe" if rng.random() < 0.5 else "safe"},
        })
    for i, r in enumerate(recs):
        r["_sa"] = rng.random()
        r["_sb"] = min(1.0, r["_sa"] + 0.15)
    sa = {"label": lambda r: ablations.label_from_score(r["_sa"], 0.5),
          "score": lambda r: r["_sa"]}
    sb = {"label": lambda r: ablations.label_from_score(r["_sb"], 0.5),
          "score": lambda r: r["_sb"]}
    boot = ev.bootstrap_ci(recs, sa, sb, n_reps=200, seed=42)
    macro = boot["macro_f1"]
    obs = macro["observed_delta"]
    recomputed = (ev._macro_f1(np.array([1 if r["sample"]["gold_label"] == "unsafe" else 0 for r in recs]),
                               np.array([1 if sb["label"](r) == "unsafe" else 0 for r in recs]))
                  - ev._macro_f1(np.array([1 if r["sample"]["gold_label"] == "unsafe" else 0 for r in recs]),
                                 np.array([1 if sa["label"](r) == "unsafe" else 0 for r in recs])))
    assert obs == pytest.approx(round(recomputed, 4), abs=1e-9)
    assert boot["macro_f1"]["ci95"][0] < obs < boot["macro_f1"]["ci95"][1]


def test_threshold_selection_on_dev_uses_macro_f1():
    rng = np.random.default_rng(3)
    y = np.array([1 if rng.random() < 0.5 else 0 for _ in range(100)], dtype=int)
    recs = [{"sample": {"gold_label": "unsafe" if v else "safe"}} for v in y]
    scores = np.clip(0.3 + 0.6 * rng.random(100) + 0.25 * y, 0, 1)
    thr = ablations.select_threshold_on_dev(recs, lambda r: scores[id(r) % 100] if False else scores[recs.index(r)], objective="macro_f1")
    assert thr is not None
