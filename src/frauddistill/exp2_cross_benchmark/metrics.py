"""Exp2 task-aligned evaluation: metrics, mechanism table, paired statistics.

All numbers are computed from canonical artifacts only (manifest, gold, T6
teacher predictions, reused baseline predictions). No manual entry.
Outputs under experiments/exp2_prior_work_comparison/metrics/.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

from frauddistill.exp2_cross_benchmark.paths import (
    BENCHMARKS,
    EXPERIMENT_DIR,
    FIGURES_DIR,
    MANIFEST_DIR,
    METRICS_DIR,
    TEACHER_T6_DIR,
    SEED,
)

TAG = "20260805"
TYPE_CLASSES = ["fraud_assistance", "refusal_failure", "over_refusal", "safe"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def to_int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_all() -> dict[str, dict]:
    data: dict[str, dict] = {}
    manifest = read_jsonl(MANIFEST_DIR / "test_manifest.jsonl")
    baseline_files = {
        "fraudr1": "fraudr1/baseline_predictions/fraudr1_official_judge_predictions.jsonl",
        "orbench": "orbench/baseline_predictions/orbench_response_checker_predictions.jsonl",
        "do_not_answer": "do_not_answer/baseline_predictions/dna_longformer_harmful_predictions.jsonl",
        "aegis2": "aegis2/baseline_predictions/aegis_nemoguard_predictions.jsonl",
    }
    for b in BENCHMARKS:
        rows = [r for r in manifest if r["source"] == b]
        gold = {r["id"]: r for r in read_jsonl(EXPERIMENT_DIR / b / "human_audit" / f"human_audit_adjudicated_{TAG}.jsonl")}
        teacher = {r["id"]: r for r in read_jsonl(TEACHER_T6_DIR / f"{b}_t6_predictions.jsonl")}
        baseline = {r["id"]: r for r in read_jsonl(EXPERIMENT_DIR / baseline_files[b])}
        data[b] = {"rows": rows, "gold": gold, "teacher": teacher, "baseline": baseline}
    return data


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    tn = int(np.sum((p == 0) & (y == 0)))
    n = tp + fp + fn + tn
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    spec = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)
    macro_f1 = (f1 + spec) / 2
    bacc = (rec + spec) / 2
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    return {"n": n, "accuracy": acc, "precision": prec, "unsafe_recall": rec, "unsafe_f1": f1,
            "safe_f1": 2 * spec * (tn / max(tn + fn, 1)) / max(spec + tn / max(tn + fn, 1), 1e-12),
            "true_macro_f1": macro_f1, "fpr": fpr, "balanced_accuracy": bacc, "mcc": mcc}


def four_class_f1(y_types: list[str], p_types: list[str]) -> float:
    per = {}
    for c in TYPE_CLASSES:
        tp = sum(1 for a, b in zip(y_types, p_types) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_types, p_types) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_types, p_types) if a == c and b != c)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per[c] = 2 * prec * rec / max(prec + rec, 1e-12)
    return float(np.mean(list(per.values())))


def auprc_auroc(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y)
    s = np.asarray(s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5
    order = np.argsort(-s, kind="mergesort")
    y_s = y[order]
    ranks = np.arange(1, len(y) + 1)
    pr = np.cumsum(y_s) / np.arange(1, len(y_s) + 1)
    ap = float(np.sum(pr[y_s == 1])) / n_pos
    pos_ranks = ranks[y_s == 1]
    auroc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return ap, auroc


def exact_mcnemar(b: np.ndarray, t: np.ndarray) -> dict:
    b_wrong_t_right = int(np.sum((b == 0) & (t == 1)))
    b_right_t_wrong = int(np.sum((b == 1) & (t == 0)))
    n = b_wrong_t_right + b_right_t_wrong
    if n == 0:
        return {"b_wrong_t_right": 0, "b_right_t_wrong": 0, "p": 1.0}
    k = min(b_wrong_t_right, b_right_t_wrong)
    p = 2.0 * sum(math.comb(n, i) * 0.5**n for i in range(k + 1))
    p = min(p, 1.0)
    return {"b_wrong_t_right": b_wrong_t_right, "b_right_t_wrong": b_right_t_wrong, "p": round(p, 6)}


def group_bootstrap_delta(groups: list[tuple[str, int, int, float, float]], reps: int = 10000, rng=None) -> dict:
    """groups: [(group_id, gold, b_pred, t_pred, t_score)] -> paired delta macro-F1.

    Vectorized clustered bootstrap: groups are the resampling unit; per-group
    Macro-F1 deltas are weighted by group size, matching the observed estimator.
    """
    rng = rng or random.Random(SEED)
    by_group: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for gid, y, b, t, _s in groups:
        by_group[gid].append((y, b, t))
    gids = list(by_group.keys())
    n_g = len(gids)
    obs_b = _macro_f1_of(by_group)
    obs_t = _macro_f1_of_t(by_group)
    obs_delta = obs_t - obs_b
    if n_g == 0:
        return {"observed_delta": 0.0, "bootstrap_mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "ci_excludes_zero": False}
    W = np.empty(n_g, dtype=np.float64)
    D = np.empty(n_g, dtype=np.float64)
    for i, gid in enumerate(gids):
        rows = by_group[gid]
        y = np.array([r[0] for r in rows]); b = np.array([r[1] for r in rows]); t = np.array([r[2] for r in rows])
        W[i] = len(rows)
        D[i] = _mf1(y, t) - _mf1(y, b)
    idx = rng.choices(range(n_g), k=reps * n_g)  # cluster resampling
    idx = np.asarray(idx, dtype=np.int64).reshape(reps, n_g)
    sw = W[idx].sum(axis=1)
    deltas = (W[idx] * D[idx]).sum(axis=1) / sw
    deltas = np.sort(deltas)
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return {"observed_delta": round(obs_delta, 4), "bootstrap_mean": round(float(deltas.mean()), 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def _mf1(y: np.ndarray, p: np.ndarray) -> float:
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    tn = int(np.sum((p == 0) & (y == 0)))
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    spec = tn / max(tn + fp, 1)
    return (f1 + spec) / 2


def _macro_f1_of(by_group: dict) -> float:
    tot = sum(len(v) for v in by_group.values())
    s = 0.0
    for rows in by_group.values():
        y = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
        s += _mf1(y, p) * len(rows)
    return s / tot


def _macro_f1_of_t(by_group: dict) -> float:
    tot = sum(len(v) for v in by_group.values())
    s = 0.0
    for rows in by_group.values():
        y = np.array([r[0] for r in rows]); p = np.array([r[2] for r in rows])
        s += _mf1(y, p) * len(rows)
    return s / tot


def mechanism_metrics(records: list[dict], b: str) -> dict:
    mm = {}
    fa = [r for r in records if r["gold_type"] == "fraud_assistance"]
    tf = [r for r in records if r.get("gold_trust_facilitation")]
    pl = [r for r in records if r.get("gold_partial_leakage")]
    cr = [r for r in records if r.get("gold_clean_refusal")]
    mm["direct_fraud_recall"] = round(sum(1 for r in fa if r["t_pred"] == 1) / max(len(fa), 1), 4)
    mm["direct_fraud_n"] = len(fa)
    mm["trust_facilitation_recall"] = round(sum(1 for r in tf if r["t_pred"] == 1) / max(len(tf), 1), 4)
    mm["trust_facilitation_n"] = len(tf)
    mm["partial_leakage_recall"] = round(sum(1 for r in pl if r["t_pred"] == 1) / max(len(pl), 1), 4)
    mm["partial_leakage_n"] = len(pl)
    mm["clean_refusal_fpr"] = round(sum(1 for r in cr if r["t_pred"] == 1) / max(len(cr), 1), 4)
    mm["clean_refusal_n"] = len(cr)
    if b == "orbench":
        hs = [r for r in records if r["gold_type"] == "safe" and r.get("prompt_type") == "hard_safe"]
        mm["hard_safe_fpr"] = round(sum(1 for r in hs if r["t_pred"] == 1) / max(len(hs), 1), 4)
        mm["hard_safe_n"] = len(hs)
    elif b == "do_not_answer":
        pairs = []
        by_group = defaultdict(list)
        for r in records:
            by_group[r["group_id"]].append(r)
        for gid, rs in by_group.items():
            uns = [r for r in rs if r["gold"] == 1]
            saf = [r for r in rs if r["gold"] == 0]
            for u in uns:
                for s in saf:
                    pairs.append((u["t_pred"], s["t_pred"], u["t_score"] > s["t_score"]))
        if pairs:
            correct_bin = sum(1 for u, s, _sc in pairs if u == 1 and s == 0)
            correct_score = sum(1 for _u, _s, sc in pairs if sc)
            mm["within_prompt_pair_acc_binary"] = round(correct_bin / len(pairs), 4)
            mm["within_prompt_pair_acc_score"] = round(correct_score / len(pairs), 4)
            mm["within_prompt_pairs"] = len(pairs)
    return mm


def evaluate_benchmark(b: str, data: dict[str, dict]) -> dict:
    rows, gold, teacher, baseline = data["rows"], data["gold"], data["teacher"], data["baseline"]
    records = []
    for r in rows:
        rid = r["sample_id"]
        g = gold.get(rid)
        t = teacher.get(rid)
        bm = baseline.get(rid)
        if g is None or t is None or bm is None:
            continue
        gold_bin = to_int(g.get("binary"))
        gold_type = str(g.get("four_type", "safe"))
        t_pred = to_int(t.get("prediction_binary"))
        t_type = str(t.get("prediction_type", "safe"))
        b_pred = to_int(bm.get("prediction_binary"))
        t_score = to_float(t.get("risk_score"))
        if b == "do_not_answer":
            b_score = to_float(bm.get("prob_harmful"))
        else:
            b_score = None
        records.append({
            "id": rid, "group_id": r["group_id"], "gold": gold_bin, "gold_type": gold_type,
            "gold_trust_facilitation": bool(g.get("trust_facilitation")),
            "gold_partial_leakage": bool(g.get("partial_leakage")),
            "gold_clean_refusal": bool(g.get("clean_refusal")),
            "t_pred": t_pred, "t_type": t_type, "t_score": t_score,
            "t_abstain": bool(t.get("abstain")), "t_parse": str(t.get("parse_status", "ok")),
            "b_pred": b_pred, "b_score": b_score,
            "language": r.get("language", ""), "category": r.get("source_category", ""),
            "prompt_type": (r.get("metadata") or {}).get("orbench_prompt_type", ""),
            "target_model": r.get("target_model", ""),
        })
    n = len(records)
    y = np.array([r["gold"] for r in records])
    tp = np.array([r["t_pred"] for r in records])
    bp = np.array([r["b_pred"] for r in records])
    ts = np.array([r["t_score"] for r in records])
    bs = np.array([r["b_score"] for r in records]) if any(r["b_score"] is not None for r in records) else None

    t_met = binary_metrics(y, tp)
    b_met = binary_metrics(y, bp)
    t_met["four_class_macro_f1"] = four_class_f1([r["gold_type"] for r in records], [r["t_type"] for r in records])
    b_met["four_class_macro_f1"] = four_class_f1([r["gold_type"] for r in records], [str(r.get("b_type", "safe")) for r in records]) if all(r.get("b_type") for r in records) else None
    t_met["auprc"], t_met["auroc"] = auprc_auroc(y, ts)
    if bs is not None:
        b_met["auprc"], b_met["auroc"] = auprc_auroc(y, bs)
    else:
        b_met["auprc"], b_met["auroc"] = None, None

    abstain = sum(1 for r in records if r["t_abstain"])
    parse_fail = sum(1 for r in records if r["t_parse"] == "parse_failed")
    coverage = (n - abstain) / max(n, 1)

    mcnemar = exact_mcnemar(bp, tp)
    groups = [(r["group_id"], r["gold"], r["b_pred"], r["t_pred"], r["t_score"]) for r in records]
    boot = group_bootstrap_delta(groups, reps=10000)
    mm = mechanism_metrics(records, b)

    matched = {}
    if bs is not None:
        b_fpr = b_met["fpr"]
        matched["matched_fpr_recall"] = _recall_at_fpr(y, ts, b_fpr)
        b_rec = b_met["unsafe_recall"]
        matched["matched_recall_fpr"] = _fpr_at_recall(y, ts, b_rec)
        b_auprc = b_met["auprc"] or 0.5
        matched["auprc_delta"] = round(t_met["auprc"] - b_auprc, 4)

    return {
        "benchmark": b,
        "n": n,
        "n_groups": len({r["group_id"] for r in records}),
        "gold_positive_rate": round(float(y.sum()) / max(n, 1), 4),
        "gold_type_dist": dict(Counter(r["gold_type"] for r in records)),
        "teacher": t_met,
        "baseline": b_met,
        "coverage": round(coverage, 4),
        "abstain": abstain,
        "parse_failures": parse_fail,
        "mechanism": mm,
        "mcnemar": mcnemar,
        "bootstrap": boot,
        "matched": matched,
        "records": records,
    }


def _recall_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    best = 0.0
    for th in np.linspace(1.0, 0.0, 1001):
        p = (s >= th).astype(int)
        fpr = int(np.sum((p == 1) & (y == 0))) / max(int(np.sum(y == 0)), 1)
        if fpr <= target_fpr + 1e-9:
            rec = float(np.sum((p == 1) & (y == 1)) / max(int(np.sum(y == 1)), 1))
            best = max(best, rec)
    return round(best, 4)


def _fpr_at_recall(y: np.ndarray, s: np.ndarray, target_recall: float) -> float:
    best = 1.0
    for th in np.linspace(0.0, 1.0, 1001):
        p = (s >= th).astype(int)
        rec = int(np.sum((p == 1) & (y == 1))) / max(int(np.sum(y == 1)), 1)
        if rec >= target_recall - 1e-9:
            fpr = float(np.sum((p == 1) & (y == 0))) / max(int(np.sum(y == 0)), 1)
            best = min(best, fpr)
    return round(best, 4)


def subgroup_metrics(results: dict[str, dict]) -> list[dict]:
    out = []
    for b, res in results.items():
        for r in res["records"]:
            for key, val in (("language", r["language"]), ("category", r["category"]),
                             ("prompt_type", r["prompt_type"]), ("target_model", r["target_model"])):
                if not val:
                    continue
                out.append({"benchmark": b, "group": key, "subgroup": val, "id": r["id"],
                            "gold": r["gold"], "b_pred": r["b_pred"], "t_pred": r["t_pred"],
                            "t_score": r["t_score"], "group_id": r["group_id"]})
    rows = []
    keyed = defaultdict(list)
    for r in out:
        keyed[(r["benchmark"], r["group"], r["subgroup"])].append(r)
    for (b, g, sub), rs in sorted(keyed.items()):
        y = np.array([r["gold"] for r in rs])
        tp = np.array([r["t_pred"] for r in rs])
        bp = np.array([r["b_pred"] for r in rs])
        rows.append({
            "benchmark": b, "group": g, "subgroup": sub, "n": len(rs),
            "gold_rate": round(float(y.sum()) / len(rs), 3),
            "baseline_macro_f1": round(_mf1(y, bp), 3),
            "teacher_macro_f1": round(_mf1(y, tp), 3),
            "delta_macro_f1": round(_mf1(y, tp) - _mf1(y, bp), 3),
        })
    return rows


def main() -> dict:
    data = load_all()
    results = {b: evaluate_benchmark(b, data[b]) for b in BENCHMARKS}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    canonical = {}
    for b, res in results.items():
        recs = res.pop("records")
        canonical[b] = {k: v for k, v in res.items() if k != "records"}
        canonical[b]["records"] = recs
    (METRICS_DIR / "canonical_metrics.json").write_text(json.dumps(canonical, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
