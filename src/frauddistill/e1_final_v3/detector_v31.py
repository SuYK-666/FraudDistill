from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from frauddistill.e1_v10.metrics import auprc, binary_metrics, binom_two_sided, ece, groupby, wilson
from frauddistill.student.relation_features import CROSS_FEATURE_NAMES, UNARY_FEATURE_NAMES, cross_qy_features, unary_text_features

VIEWS = ["q_only", "y_only", "q+y"]
STRATA = ["context_stable_positive", "context_stable_negative", "context_critical_positive", "context_hard_negative"]


class ViewDetector:
    """CPU PairLite detector with strict view isolation."""

    def __init__(self, mode: str, max_features: int = 60000, C: float = 1.0, seed: int = 13, use_interaction: bool = True):
        if mode not in VIEWS:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.max_features = max_features
        self.C = C
        self.seed = seed
        self.use_interaction = use_interaction
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.classifier = LogisticRegression(max_iter=3000, class_weight="balanced", C=C, solver="liblinear", random_state=seed)

    def _corpus(self, rows: list[dict[str, Any]]) -> list[str]:
        if self.mode == "q_only":
            return [str(r.get("q_private", "")) for r in rows]
        if self.mode == "y_only":
            return [str(r.get("y_private", "")) for r in rows]
        return [str(r.get("q_private", "")) for r in rows] + [str(r.get("y_private", "")) for r in rows]

    def fit(self, rows: list[dict[str, Any]], labels: list[int]) -> "ViewDetector":
        self.vectorizer.fit(self._corpus(rows))
        X = self.features(rows)
        y = np.asarray(labels, dtype=int)
        self.classifier.fit(X, y)
        return self

    def features(self, rows: list[dict[str, Any]]):
        queries = [str(r.get("q_private", "")) for r in rows]
        answers = [str(r.get("y_private", "")) for r in rows]
        q_tfidf = self.vectorizer.transform(queries) if self.mode in {"q_only", "q+y"} else self._zeros(queries)
        y_tfidf = self.vectorizer.transform(answers) if self.mode in {"y_only", "q+y"} else self._zeros(answers)
        q_unary = unary_text_features(queries) if self.mode in {"q_only", "q+y"} else self._zeros_2d(len(rows), len(UNARY_FEATURE_NAMES))
        y_unary = unary_text_features(answers) if self.mode in {"y_only", "q+y"} else self._zeros_2d(len(rows), len(UNARY_FEATURE_NAMES))
        parts: list[Any] = [q_tfidf, q_unary, y_tfidf, y_unary]
        if self.mode == "q+y":
            cross = []
            if self.use_interaction:
                cross = cross_qy_features(queries, answers)
            else:
                cross = self._zeros_2d(len(rows), len(CROSS_FEATURE_NAMES))
            parts.append(np.asarray(cross))
        return hstack(parts, format="csr")

    def _zeros(self, items: list[str]):
        import scipy.sparse as sp

        return sp.csr_matrix((len(items), self.vectorizer.max_features if hasattr(self.vectorizer, "max_features") else 1))

    def _zeros_2d(self, n: int, m: int):
        import numpy as np

        return np.zeros((n, m))

    def predict_proba(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.classifier.predict_proba(self.features(rows))[:, 1]


def panel_rows_to_eval(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    valid = [r for r in rows if int(r.get("gold_central", -1)) >= 0]
    labels = [int(r["gold_central"]) for r in valid]
    return valid, labels


def metrics_at_threshold(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (scores >= threshold).astype(int)
    evals = [{**r, "gold": int(r["gold_central"]), "pred": int(p), "score": float(s)} for r, p, s in zip(rows, preds, scores)]
    m = binary_metrics(evals)
    per_stratum = {}
    for stratum in STRATA:
        sub = [e for e in evals if e.get("stratum") == stratum]
        per_stratum[stratum] = binary_metrics(sub) if sub else {"n": 0}
    return {**m, "threshold": threshold, "per_stratum": per_stratum}


def choose_threshold(cal_rows: list[dict[str, Any]], scores: np.ndarray, metric: str = "macro_f1") -> float:
    best_t = 0.5
    best_v = -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        m = metrics_at_threshold(cal_rows, scores, float(t))
        if m[metric] > best_v:
            best_v = m[metric]
            best_t = float(t)
    return best_t


def run_seed(dev: list[dict[str, Any]], cal: list[dict[str, Any]], anchor: list[dict[str, Any]], mode: str, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    d_rows, d_labels = panel_rows_to_eval(dev)
    c_rows, _ = panel_rows_to_eval(cal)
    a_rows, _ = panel_rows_to_eval(anchor)
    if not d_rows or not c_rows or not a_rows:
        return {"mode": mode, "seed": seed, "error": "empty split"}
    detector = ViewDetector(mode=mode, seed=seed, C=float(cfg.get("detector_C", 1.0)), max_features=int(cfg.get("max_features", 60000)))
    detector.fit(d_rows, d_labels)
    cal_scores = detector.predict_proba(c_rows)
    threshold = choose_threshold(c_rows, cal_scores)
    anchor_scores = detector.predict_proba(a_rows)
    return {
        "mode": mode,
        "seed": seed,
        "threshold": threshold,
        "anchor": metrics_at_threshold(a_rows, anchor_scores, threshold),
        "anchor_auprc": auprc([int(r["gold_central"]) for r in a_rows], anchor_scores),
        "anchor_auroc": roc_auc_score([int(r["gold_central"]) for r in a_rows], anchor_scores) if len(set(r["gold_central"] for r in a_rows)) > 1 else 0.0,
        "cal_macro_f1_at_threshold": metrics_at_threshold(c_rows, cal_scores, threshold)["macro_f1"],
    }


def run_model_dev_cv(dev: list[dict[str, Any]], mode: str, seed: int, folds: int = 5) -> dict[str, Any]:
    rows, labels = panel_rows_to_eval(dev)
    if len(rows) < folds:
        return {"mode": mode, "seed": seed, "cv_macro_f1": 0.0, "cv_auprc": 0.0}
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    fold_size = len(idx) // folds
    macro = []
    auprc_scores = []
    for f in range(folds):
        test_idx = set(idx[f * fold_size : (f + 1) * fold_size if f < folds - 1 else len(idx)])
        tr = [rows[i] for i in range(len(rows)) if i not in test_idx]
        te = [rows[i] for i in range(len(rows)) if i in test_idx]
        detector = ViewDetector(mode=mode, seed=seed)
        detector.fit(tr, [int(r["gold_central"]) for r in tr])
        scores = detector.predict_proba(te)
        m = metrics_at_threshold(te, scores, 0.5)
        macro.append(m["macro_f1"])
        auprc_scores.append(auprc([int(r["gold_central"]) for r in te], scores))
    return {"mode": mode, "seed": seed, "cv_macro_f1": float(np.mean(macro)), "cv_macro_f1_sd": float(np.std(macro)), "cv_auprc": float(np.mean(auprc_scores))}


def aggregate_seeds(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        by_mode[r["mode"]].append(r)
    out = {}
    for mode, runs in by_mode.items():
        macro = [r["anchor"]["macro_f1"] for r in runs]
        fpr = [r["anchor"]["fpr"] for r in runs]
        recall = [r["anchor"]["recall"] for r in runs]
        auprc_vals = [r["anchor_auprc"] for r in runs]
        out[mode] = {
            "n_seeds": len(runs),
            "anchor_macro_f1_mean": float(np.mean(macro)),
            "anchor_macro_f1_sd": float(np.std(macro)),
            "anchor_fpr_mean": float(np.mean(fpr)),
            "anchor_recall_mean": float(np.mean(recall)),
            "anchor_auprc_mean": float(np.mean(auprc_vals)),
            "anchor_auprc_sd": float(np.std(auprc_vals)),
            "seed_runs": runs,
        }
    return out


def paired_mcnemar(anchor_rows: list[dict[str, Any]], scores_qy: np.ndarray, scores_y: np.ndarray, threshold_qy: float, threshold_y: float) -> dict[str, Any]:
    pred_qy = (scores_qy >= threshold_qy).astype(int)
    pred_y = (scores_y >= threshold_y).astype(int)
    gold = np.asarray([int(r["gold_central"]) for r in anchor_rows], dtype=int)
    b = int(np.sum((gold == 1) & (pred_qy == 1) & (pred_y == 0)))
    c = int(np.sum((gold == 1) & (pred_qy == 0) & (pred_y == 1)))
    qy_correct = int(np.sum(pred_qy == gold))
    y_correct = int(np.sum(pred_y == gold))
    return {
        "n": len(gold),
        "qy_correct": qy_correct,
        "y_correct": y_correct,
        "qy_better_positive_cases": b,
        "y_better_positive_cases": c,
        "p_exact_mcnemar": binom_two_sided(b, c),
    }


def cluster_bootstrap_macro_f1(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float, cluster_key: str = "canonical_case_id", iterations: int = 2000, seed: int = 20260802) -> dict[str, Any]:
    rng = random.Random(seed)
    clusters = list(groupby(rows, cluster_key).values())
    if not clusters:
        return {"point": 0.0, "low": 0.0, "high": 0.0}
    preds = (scores >= threshold).astype(int)
    index = {id(r): i for i, r in enumerate(rows)}

    def macro(ids: list[int]) -> float:
        evals = [{**rows[i], "gold": int(rows[i]["gold_central"]), "pred": int(preds[i]), "score": float(scores[i])} for i in ids]
        return binary_metrics(evals)["macro_f1"]

    vals = []
    for _ in range(max(1, iterations)):
        ids = [index[id(r)] for c in (rng.choice(clusters) for _ in clusters) for r in c]
        vals.append(macro(ids))
    vals.sort()
    return {"point": macro(list(range(len(rows)))), "low": vals[int(0.025 * (len(vals) - 1))], "high": vals[int(0.975 * (len(vals) - 1))]}


def shortcut_audits(dev: list[dict[str, Any]], seed: int = 13) -> dict[str, Any]:
    rows, labels = panel_rows_to_eval(dev)
    if not rows:
        return {}
    audit: dict[str, Any] = {}
    # provenance shortcut: classifier on provenance one-hot
    provenance_values = sorted({str(r.get("provenance", "unknown")) for r in rows})
    X = np.zeros((len(rows), len(provenance_values)))
    for i, r in enumerate(rows):
        X[i, provenance_values.index(str(r.get("provenance", "unknown")))] = 1.0
    if len(set(labels)) > 1:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X, labels)
        audit["provenance_shortcut_auc"] = float(roc_auc_score(labels, clf.predict_proba(X)[:, 1]))
    else:
        audit["provenance_shortcut_auc"] = 0.0
    # nuisance baseline: length/punctuation/language
    nuis = []
    for r in rows:
        q = str(r.get("q_private", ""))
        y = str(r.get("y_private", ""))
        nuis.append([len(q), len(y), sum(c.isdigit() for c in y), sum(c in "!????" for c in y), 1 if r.get("language") == "zh" else 0])
    if len(set(labels)) > 1:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(np.asarray(nuis), labels)
        audit["nuisance_baseline_auc"] = float(roc_auc_score(labels, clf.predict_proba(np.asarray(nuis))[:, 1]))
    else:
        audit["nuisance_baseline_auc"] = 0.0
    # wrong-q permutation: q+y detector trained on dev, evaluated with q permuted
    detector = ViewDetector(mode="q+y", seed=seed)
    detector.fit(rows, labels)
    scores_orig = detector.predict_proba(rows)
    perm = list(range(len(rows)))
    rng = random.Random(seed + 1)
    rng.shuffle(perm)
    perm_rows = [{**rows[i], "q_private": rows[perm[i]]["q_private"]} for i in range(len(rows))]
    scores_perm = detector.predict_proba(perm_rows)
    audit["wrong_q_perm_orig_macro_f1"] = metrics_at_threshold(rows, scores_orig, 0.5)["macro_f1"]
    audit["wrong_q_perm_permuted_macro_f1"] = metrics_at_threshold(perm_rows, scores_perm, 0.5)["macro_f1"]
    audit["wrong_q_perm_drop"] = audit["wrong_q_perm_orig_macro_f1"] - audit["wrong_q_perm_permuted_macro_f1"]
    return audit


from collections import defaultdict
