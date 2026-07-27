from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class GlobalPairScorer:
    vectorizer: TfidfVectorizer

    def score_pairs(self, left: list[str], right: list[str]) -> np.ndarray:
        left_matrix = self.vectorizer.transform(left)
        right_matrix = self.vectorizer.transform(right)
        scores = left_matrix.multiply(right_matrix).sum(axis=1)
        return np.asarray(scores).reshape(-1).astype(float)


def fit_global_word_char_scorer(texts: list[str]) -> GlobalPairScorer:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        norm="l2",
        sublinear_tf=True,
        max_features=120000,
        dtype=np.float32,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        norm="l2",
        sublinear_tf=True,
        max_features=120000,
        dtype=np.float32,
    )
    char_matrix = vectorizer.fit_transform(texts)
    word_matrix = word.fit_transform(texts)
    merged = sparse.hstack([word_matrix * 0.45, char_matrix * 0.55], format="csr", dtype=np.float32)
    frozen = TfidfVectorizer()
    frozen.vocabulary_ = {f"f{i}": i for i in range(merged.shape[1])}
    scorer = GlobalPairScorer.__new__(GlobalPairScorer)
    scorer.vectorizer = _FrozenMatrixVectorizer(texts, merged)
    return scorer


class _FrozenMatrixVectorizer:
    def __init__(self, texts: list[str], matrix):
        self.index = {str(text): idx for idx, text in enumerate(texts)}
        self.matrix = matrix

    def transform(self, texts: list[str]):
        return self.matrix[[self.index.get(str(text), 0) for text in texts]]


def calibrate_p2(rows: list[dict], candidate_pool: list[dict], controls_per_group: int, seed: int, output_dir: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("context_collision_group_id") or row.get("p2_group_id")), []).append(row)
    safe_pool = [row for row in candidate_pool if row.get("exp1_label") == "safe"]
    unsafe_pool = [row for row in candidate_pool if row.get("exp1_label") == "unsafe"]
    all_texts = []
    for row in [*rows, *safe_pool, *unsafe_pool]:
        all_texts.extend([str(row.get("user_query", "")), str(row.get("target_model_answer", ""))])
    scorer = fit_global_word_char_scorer(sorted(set(all_texts)))
    y_percentiles: list[float] = []
    q_percentiles: list[float] = []
    gaps: list[float] = []
    y_beats_control = 0
    y_gt_q = 0
    control_records = []
    for group_id, members in sorted(by_group.items()):
        if len(members) != 2:
            continue
        safe = next(row for row in members if row["exp1_label"] == "safe")
        unsafe = next(row for row in members if row["exp1_label"] == "unsafe")
        controls = matched_controls(unsafe, safe, safe_pool, unsafe_pool, controls_per_group, rng)
        y_score = float(scorer.score_pairs([unsafe["target_model_answer"]], [safe["target_model_answer"]])[0])
        q_score = float(scorer.score_pairs([unsafe["user_query"]], [safe["user_query"]])[0])
        control_y = scorer.score_pairs([u["target_model_answer"] for u, _ in controls], [s["target_model_answer"] for _, s in controls]).tolist() if controls else []
        control_q = scorer.score_pairs([u["user_query"] for u, _ in controls], [s["user_query"] for _, s in controls]).tolist() if controls else []
        yp = percentile(y_score, control_y)
        qp = percentile(q_score, control_q)
        y_percentiles.append(yp)
        q_percentiles.append(qp)
        gaps.append(yp - qp)
        y_beats_control += int(y_score > float(np.mean(control_y)) if control_y else False)
        y_gt_q += int(y_score > q_score)
        control_records.append({"group_id": group_id, "source_pair": source_pair_key(unsafe, safe), "y_score": y_score, "q_score": q_score, "y_percentile": yp, "q_percentile": qp})
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "p2_calibration_controls.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in control_records), encoding="utf-8")
    groups = len(y_percentiles)
    source_pairs = [source_pair_key(members[0], members[1]) for members in by_group.values() if len(members) == 2]
    source_pair_counts = {key: source_pairs.count(key) for key in sorted(set(source_pairs))}
    return {
        "groups": groups,
        "mean_y_percentile": mean(y_percentiles),
        "p10_y_percentile": quantile(y_percentiles, 0.10),
        "mean_q_percentile": mean(q_percentiles),
        "mean_collision_gap": mean(gaps),
        "p10_collision_gap": quantile(gaps, 0.10),
        "independent_y_beats_control_rate": y_beats_control / max(groups, 1),
        "independent_y_gt_q_rate": y_gt_q / max(groups, 1),
        "source_pair_counts": source_pair_counts,
        "largest_source_pair_rate": max(source_pair_counts.values(), default=0) / max(groups, 1),
        "source_pair_types": len(source_pair_counts),
        "scorer": "global_word_char_tfidf_shared_vector_space",
    }


def p2_calibration_passed(summary: dict, policy: dict) -> dict:
    checks = {
        "C5_mean_y_percentile": summary["mean_y_percentile"] >= float(policy["mean_y_percentile_min"]),
        "C6_p10_y_percentile": summary["p10_y_percentile"] >= float(policy["p10_y_percentile_min"]),
        "C7_mean_q_percentile": summary["mean_q_percentile"] <= float(policy["mean_q_percentile_max"]),
        "C8_mean_collision_gap": summary["mean_collision_gap"] >= float(policy["mean_collision_gap_min"]),
        "C9_p10_collision_gap": summary["p10_collision_gap"] > float(policy["p10_collision_gap_min"]),
        "C10_y_beats_control": summary["independent_y_beats_control_rate"] >= float(policy["independent_y_beats_control_min"]),
        "C11_y_gt_q": summary["independent_y_gt_q_rate"] >= float(policy["independent_y_gt_q_min"]),
        "C15_largest_source_pair": summary["largest_source_pair_rate"] <= float(policy["largest_source_pair_max"]),
        "C16_source_pair_types": summary["source_pair_types"] >= int(policy["source_pair_types_min"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def matched_controls(unsafe: dict, safe: dict, safe_pool: list[dict], unsafe_pool: list[dict], count: int, rng: np.random.Generator) -> list[tuple[dict, dict]]:
    ratio = len(unsafe["target_model_answer"]) / max(len(safe["target_model_answer"]), 1)
    controls = []
    unsafe_candidates = [row for row in unsafe_pool if row["semantic_component_id"] != unsafe["semantic_component_id"]]
    safe_candidates = [row for row in safe_pool if row["semantic_component_id"] != safe["semantic_component_id"]]
    for _ in range(count * 20):
        if len(controls) >= count or not unsafe_candidates or not safe_candidates:
            break
        u = unsafe_candidates[int(rng.integers(0, len(unsafe_candidates)))]
        s = safe_candidates[int(rng.integers(0, len(safe_candidates)))]
        candidate_ratio = len(u["target_model_answer"]) / max(len(s["target_model_answer"]), 1)
        if abs(candidate_ratio - ratio) <= 0.35:
            controls.append((u, s))
    return controls


def scorer(a: str, b: str) -> float:
    raise RuntimeError("pairwise TF-IDF scorer removed; use fit_global_word_char_scorer(...).score_pairs(...)")


def percentile(value: float, controls: list[float]) -> float:
    if not controls:
        return 0.0
    return float((np.asarray(controls) <= value).mean())


def source_pair_key(a: dict, b: dict) -> str:
    return "|".join(sorted([str(a.get("source")), str(b.get("source"))]))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else 0.0


def calibration_fingerprint(summary: dict) -> str:
    return hashlib.sha256(json.dumps(summary, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
