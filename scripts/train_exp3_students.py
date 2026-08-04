# -*- coding: utf-8 -*-
"""Train the lightweight Student gradient S0-S4 for Exp3 (guide 18).

Students: TF-IDF + Logistic Regression on q+y only (guide 18.1/18.4).
S0 gold, S1 +high-confidence hard teacher labels, S2 +teacher score weights,
S3 +risk type heads, S4 +evidence-aware weights & pair ranking.
5 seeds -> mean +/- std on held-out test.

Usage: python scripts/train_exp3_students.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data/prepared/exp3_agent_distillation/exp3_dataset.jsonl"
OUT_ROOT = Path(os.environ.get("EXP3_OUT_ROOT") or REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs")
AGENT_DIR = OUT_ROOT / "agent_predictions"
SEEDS = [11, 23, 37, 53, 71]
LAMBDA_H = 0.5      # hard teacher label duplication weight
LAMBDA_S = 0.5      # score weight strength
LAMBDA_PAIR = 0.4   # pair ranking boost
HIGH_CONF = 0.80
# The arbiter's self-reported confidence is almost always >= 0.8 (degenerate),
# so "high confidence" is gated on score decisiveness: a decisive band of
# +-0.25 around the frozen operating threshold (0.84): score>=0.95 (strong
# unsafe) or score<=0.60 (strong safe). Determined from train statistics only.
DECISIVE_UNSAFE_SCORE = 0.95
DECISIVE_SAFE_SCORE = 0.60
TYPES = ["fraud_assistance", "refusal_failure", "over_refusal", "safe"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_all() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    dataset = read_jsonl(DATASET)
    teacher = {r["id"]: r for r in read_jsonl(AGENT_DIR / "train.jsonl")}
    dev_t = {r["id"]: r for r in read_jsonl(AGENT_DIR / "dev.jsonl")}
    test_t = {r["id"]: r for r in read_jsonl(AGENT_DIR / "test.jsonl")}
    teacher.update(dev_t)
    teacher.update(test_t)
    return dataset, teacher, teacher


class QyTfidf:
    """TF-IDF on q+y with a binary LR head (student, guide 18.1)."""

    def __init__(self, max_features: int = 60000, C: float = 1.0, seed: int = 11):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=C, solver="liblinear", random_state=seed)

    def fit(self, rows, labels, sample_weight=None):
        X = self.vec.fit_transform([str(r["user_query"]) + " [SEP] " + str(r["target_model_answer"]) for r in rows])
        y = np.array([1 if str(l) == "unsafe" else 0 for l in labels], dtype=int)
        kw = {}
        if sample_weight is not None:
            kw["sample_weight"] = np.asarray(sample_weight, dtype=float)
        self.clf.fit(X, y, **kw)
        return self

    def predict_proba(self, rows):
        X = self.vec.transform([str(r["user_query"]) + " [SEP] " + str(r["target_model_answer"]) for r in rows])
        return self.clf.predict_proba(X)[:, 1]


def teacher_signal(rec: dict) -> dict:
    sig = rec.get("signal") or {}
    score = float(sig.get("teacher_score", 0.5))
    confidence = float(sig.get("confidence", 0.5))
    decisive = score >= DECISIVE_UNSAFE_SCORE or score <= DECISIVE_SAFE_SCORE
    return {
        "label": str(sig.get("teacher_label", "safe")),
        "score": score,
        "type": str(sig.get("teacher_type", "safe")),
        "confidence": confidence,
        "decisive": bool(confidence >= HIGH_CONF and decisive),
        "conflict": bool(rec.get("conflict_flags")),
        "correction_used": bool(sig.get("correction_used")),
    }


def build_teacher_rows(train_rows, teacher_map) -> tuple[list, list, list]:
    """Return (rows, labels, weights) with high-confidence teacher duplication."""
    rows, labels, weights = [], [], []
    for r in train_rows:
        t = teacher_signal(teacher_map.get(r["id"]) or {})
        rows.append(r)
        labels.append(r["gold_label"])
        weights.append(1.0)
        if t["decisive"] and t["label"] in ("safe", "unsafe"):
            rows.append(r)
            labels.append(t["label"])
            weights.append(LAMBDA_H)
    return rows, labels, weights


def full_distill_rows(train_rows, teacher_map):
    """S4: score distill + evidence-aware weights + pair ranking (soft signals).

    Hard teacher-label duplication was net-negative on dev (hard pseudo-label
    error amplification), so S4 keeps soft score weights, evidence weights and
    pair ranking, and drops the hard-label duplicates (documented deviation).
    """
    rows, labels, weights = [], [], []
    base_ew = evidence_weights(train_rows, teacher_map)
    for i, r in enumerate(train_rows):
        t = teacher_signal(teacher_map.get(r["id"]) or {})
        score_w = 1.0 + LAMBDA_S * abs(2 * t["score"] - 1)
        rows.append(r)
        labels.append(r["gold_label"])
        weights.append(score_w * base_ew[i])
    for r in train_rows:
        if r.get("pair_id") and r["gold_label"] == "unsafe":
            t = teacher_signal(teacher_map.get(r["id"]) or {})
            w = 1.0 + LAMBDA_PAIR
            if t["decisive"]:
                w += 0.3
            rows.append(r)
            labels.append("unsafe")
            weights.append(w)
    return rows, labels, weights


def evidence_weights(train_rows, teacher_map) -> list[float]:
    out = []
    for r in train_rows:
        t = teacher_signal(teacher_map.get(r["id"]) or {})
        w = 1.0
        if t["confidence"] >= HIGH_CONF:
            w *= 1.5
        if t["conflict"]:
            w *= 0.5
        if r.get("subtype") in ("hard_safe", "partial_leakage"):
            w *= 1.5
        out.append(w)
    return out


def pair_boosted_rows(train_rows, teacher_map):
    """Duplicate context-flip pair rows with relative-order emphasis (guide 18.2 S4)."""
    rows, labels, weights = [], [], []
    pairs: dict[str, list[dict]] = {}
    for r in train_rows:
        pid = r.get("pair_id")
        if pid:
            pairs.setdefault(pid, []).append(r)
    for r in train_rows:
        rows.append(r)
        labels.append(r["gold_label"])
        weights.append(1.0)
        pid = r.get("pair_id")
        if pid and r["gold_label"] == "unsafe":
            t = teacher_signal(teacher_map.get(r["id"]) or {})
            w = 1.0 + LAMBDA_PAIR
            if t["decisive"]:
                w += 0.3
            rows.append(r)
            labels.append("unsafe")
            weights.append(w)
    return rows, labels, weights


def type_head_fit(train_rows, teacher_map, seed):
    """4 one-vs-rest type heads on gold type (teacher type when high-conf)."""
    rows, labels = [], []
    for r in train_rows:
        gold_type = str(r.get("gold_type") or (r["gold_label"] if r["gold_label"] == "safe" else "fraud_assistance"))
        rows.append(r)
        labels.append(gold_type)
        t = teacher_signal(teacher_map.get(r["id"]) or {})
        if t["confidence"] >= HIGH_CONF and t["type"] in TYPES:
            rows.append(r)
            labels.append(t["type"])
    vec = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([str(r["user_query"]) + " [SEP] " + str(r["target_model_answer"]) for r in rows])
    heads = {}
    for tp in TYPES:
        y = np.array([1 if l == tp else 0 for l in labels], dtype=int)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed)
        clf.fit(X, y)
        heads[tp] = clf
    return vec, heads


def type_proba(vec, heads, rows) -> dict[str, np.ndarray]:
    X = vec.transform([str(r["user_query"]) + " [SEP] " + str(r["target_model_answer"]) for r in rows])
    return {tp: heads[tp].predict_proba(X)[:, 1] for tp in TYPES}


def metrics_binary(labels, scores, threshold=0.5) -> dict:
    y = np.array([1 if l == "unsafe" else 0 for l in labels], dtype=int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr = fp / max(tn + fp, 1)
    acc = (tp + tn) / max(len(y), 1)
    auprc = float(average_precision_score(y, scores)) if len(set(scores)) > 1 else float("nan")
    return {"acc": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "macro_f1": round(f1, 4), "fpr": round(fpr, 4), "auprc": round(auprc, 4)}


def type_macro_f1(labels, proba: dict[str, np.ndarray]) -> float:
    y = np.array([str(l) for l in labels])
    pred = np.array([max(TYPES, key=lambda tp: proba[tp][i]) for i in range(len(y))])
    f1s = []
    for tp in TYPES:
        tp_y = (y == tp).astype(int)
        tp_p = (pred == tp).astype(int)
        t = int((tp_p & tp_y).sum())
        fp = int((tp_p & (tp_y == 0)).sum())
        fn = int(((tp_p == 0) & tp_y).sum())
        prec = t / max(t + fp, 1)
        rec = t / max(t + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9) if t + fp + fn else 1.0)
    return round(float(np.mean(f1s)), 4)


def run_setting(name, train_rows, test_rows, dev_rows, teacher_map, seeds) -> dict:
    results = []
    for seed in seeds:
        if name == "S0_gold":
            rows, labels, weights = train_rows, [r["gold_label"] for r in train_rows], None
            model = QyTfidf(seed=seed).fit(rows, labels, weights)
            scores = model.predict_proba(test_rows)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
        elif name == "S1_hard_teacher":
            rows, labels, weights = build_teacher_rows(train_rows, teacher_map)
            model = QyTfidf(seed=seed).fit(rows, labels, weights)
            scores = model.predict_proba(test_rows)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
        elif name == "S2_score_distill":
            rows, labels, weights = build_teacher_rows(train_rows, teacher_map)
            w = np.asarray(weights, dtype=float)
            # score-based confidence weighting on all rows
            base_w = []
            for r in train_rows:
                t = teacher_signal(teacher_map.get(r["id"]) or {})
                base_w.append(1.0 + LAMBDA_S * abs(2 * t["score"] - 1))
            # duplicated teacher rows keep lambda_h * score weight
            dups = len(train_rows)
            w[dups:] = w[dups:] * np.asarray([1.0 + LAMBDA_S * abs(2 * teacher_signal(teacher_map.get(rows[i]["id"]) or {})["score"] - 1) for i in range(dups, len(rows))])
            w[:dups] = w[:dups] * np.asarray(base_w)
            model = QyTfidf(seed=seed).fit(rows, labels, w)
            scores = model.predict_proba(test_rows)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
        elif name == "S3_type_distill":
            rows, labels, weights = build_teacher_rows(train_rows, teacher_map)
            model = QyTfidf(seed=seed).fit(rows, labels, weights)
            scores = model.predict_proba(test_rows)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
            vec, heads = type_head_fit(train_rows, teacher_map, seed)
            proba = type_proba(vec, heads, test_rows)
            m["type_macro_f1"] = type_macro_f1([r.get("gold_type") or (r["gold_label"] if r["gold_label"] == "safe" else "fraud_assistance") for r in test_rows], proba)
        elif name == "S4_evidence_distill":
            rows, labels, weights = full_distill_rows(train_rows, teacher_map)
            w = np.asarray(weights, dtype=float)
            model = QyTfidf(seed=seed).fit(rows, labels, w)
            scores = model.predict_proba(test_rows)
            m = metrics_binary([r["gold_label"] for r in test_rows], scores)
        else:
            raise ValueError(name)
        results.append(m)
    keys = [k for k in results[0] if k != "n"]
    agg = {"setting": name, "seeds": seeds}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            agg[k + "_mean"] = round(float(np.mean(vals)), 4)
            agg[k + "_std"] = round(float(np.std(vals)), 4)
    return agg


def main() -> None:
    dataset, teacher_map, _ = load_all()
    train_rows = [r for r in dataset if r["split"] == "train"]
    dev_rows = [r for r in dataset if r["split"] == "dev"]
    test_rows = [r for r in dataset if r["split"] == "test"]
    print(f"train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)} teacher_rows={len(teacher_map)}")

    missing = [r["id"] for r in train_rows if r["id"] not in teacher_map]
    if missing:
        print(f"WARNING: {len(missing)} train rows lack teacher signals; run --mode train first")
        sys.exit(2)

    settings = ["S0_gold", "S1_hard_teacher", "S2_score_distill", "S3_type_distill", "S4_evidence_distill"]
    rows_out = []
    for name in settings:
        agg = run_setting(name, train_rows, test_rows, dev_rows, teacher_map, SEEDS)
        rows_out.append(agg)
        print(json.dumps(agg, ensure_ascii=False))

    METRICS = OUT_ROOT / "metrics"
    METRICS.mkdir(parents=True, exist_ok=True)
    cols = ["setting", "seeds"]
    for k in rows_out[0]:
        if k not in cols:
            cols.append(k)
    with (METRICS / "student_gradient.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows_out:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print("wrote", METRICS / "student_gradient.csv")


if __name__ == "__main__":
    main()