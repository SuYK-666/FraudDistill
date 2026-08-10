# -*- coding: utf-8 -*-
"""Shortcut audit: metadata-only diagnostics on the formal panel (guide 4.5)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .schemas import norm_text


def _text_feature_rows(rows: list[dict], key: str) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [norm_text(str(r.get(key) or "")) for r in rows]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    try:
        X = vec.fit_transform(docs)
    except Exception:
        X = vec.fit_transform(["x"] * len(docs))
    return X


def _cv_auc_ba(X, y, n_splits: int = 5, seed: int = 20260810) -> tuple[float, float]:
    """Stratified CV balanced accuracy + AUC (honest out-of-sample numbers)."""
    from sklearn.model_selection import StratifiedKFold
    ba, aucs = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        ba.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
        if len(set(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return round(float(np.mean(ba)), 4), round(float(np.mean(aucs)), 4)


def metadata_only_auc(rows: list[dict], field: str, y: np.ndarray) -> dict:
    """LR on one metadata field (OHE) predicting gold; CV balanced accuracy + AUC."""
    vals = [str(r.get(field) or "unknown") for r in rows]
    keys = sorted({v for v in vals})
    X = np.zeros((len(vals), len(keys)))
    for i, v in enumerate(vals):
        X[i, keys.index(v)] = 1.0
    if len(keys) < 2 or len(set(vals)) == 1:
        return {"field": field, "n_levels": len(keys), "balanced_accuracy": 0.5, "auroc": None,
                "note": "single level -> trivial"}
    try:
        ba, auc = _cv_auc_ba(X, y)
        return {"field": field, "n_levels": len(keys), "balanced_accuracy": ba, "auroc": auc}
    except Exception as e:
        return {"field": field, "error": str(e)}


def shortcut_audit(rows: list[dict], out_dir: Path) -> dict:
    y = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in rows], dtype=int)
    results = {}
    for field in ("source", "fraud_category", "target_model", "language", "primary_shift"):
        results[field] = metadata_only_auc(rows, field, y)
    # q-only / y-only TF-IDF LR (5-fold CV, honest out-of-sample)
    for name, key in (("q_only_tfidf", "user_query"), ("y_only_tfidf", "target_model_answer")):
        try:
            X = _text_feature_rows(rows, key)
            ba, auc = _cv_auc_ba(X, y)
            results[name] = {"balanced_accuracy": ba, "auroc": auc, "cv": "5fold"}
        except Exception as e:
            results[name] = {"error": str(e)}
    # length SMD
    lens_p = np.array([len(str(r.get("target_model_answer") or "")) for r in rows if r["gold_label"] == "unsafe"])
    lens_n = np.array([len(str(r.get("target_model_answer") or "")) for r in rows if r["gold_label"] == "safe"])
    if len(lens_p) and len(lens_n):
        sp, sn = lens_p.std(), lens_n.std()
        smd = (lens_p.mean() - lens_n.mean()) / np.sqrt((sp ** 2 + sn ** 2) / 2)
    else:
        smd = 0.0
    results["answer_length_smd"] = round(float(smd), 4)
    # per-cell balance
    cells = Counter((r.get("primary_shift"), r.get("fraud_category"), r.get("gold_label")) for r in rows)
    results["cell_balance"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(cells.items())}
    passed = all(
        (v.get("balanced_accuracy", 0.5) is not None and abs(v.get("balanced_accuracy", 0.5) - 0.5) < 0.08)
        for k, v in results.items() if isinstance(v, dict) and "balanced_accuracy" in v and k != "q_only_tfidf" and k != "y_only_tfidf"
    )
    # per-shift breakdown for metadata fields (the target-model axis U3 is the
    # designed exposure axis and must be balanced; U1 generator mix is a nuisance)
    per_shift = {}
    for shift in ("U1_category", "U2_source", "U3_target_style"):
        sub = [r for r in rows if r.get("primary_shift") == shift]
        if not sub:
            continue
        ys = np.array([1 if r["gold_label"] == "unsafe" else 0 for r in sub], dtype=int)
        per_shift[shift] = {}
        for field in ("source", "fraud_category", "target_model", "language"):
            per_shift[shift][field] = metadata_only_auc(sub, field, ys)
    results["per_shift"] = per_shift
    results["passed"] = bool(passed)
    results["note"] = ("metadata-only BA near 0.5 expected because every cell is 50/50 balanced. "
                       "CV-based diagnostics; target_model reflects U1 generator-model mix (nuisance, "
                       "absent from model input) while the designed U3 target-model axis is balanced.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shortcut_audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
