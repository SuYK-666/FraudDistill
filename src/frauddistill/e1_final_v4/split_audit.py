# -*- coding: utf-8 -*-
"""v4 split + anti-shortcut audit + freeze (family-level, leak-free).

v4.1 amendment:
- families are merged by union-find over (a) original family_id, (b) exact
  normalized (q,y), (c) exact normalized q, so identical queries/responses can
  never straddle splits.
- provenance shortcut audit uses the coarse real-vs-generated grouping: fine
  provenance encodes the matched-pair construction (e.g. generated_y_counter-
  factual_qreal vs generated_y_generated_q is the B1 mechanism itself) and is
  therefore a label synonym by design; the coarse text-origin grouping is the
  registered style shortcut feature.
- length audit: panel-level Tfidf-on-length LR (registered); numeric AUC
  reported as diagnostic.
"""
from __future__ import annotations

import collections
import hashlib
import random
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

from frauddistill.e1_final_v3.io import norm, read_jsonl, sha_text, write_json, write_jsonl

PROV_COARSE = {
    "generated_y": "generated",
    "generated_y_counterfactual_qreal": "generated",
    "generated_y_generated_q": "generated",
    "generated_refusal": "generated",
    "generated_defensive": "generated",
    "source_derived_open_control": "open_source",
    "aegis_refusal": "open_source",
    "real_matched_v32": "open_source",
    "real_target_v32": "open_source",
}

PROV_COARSE_BIN = {"open_source": 1.0, "generated": 0.0}


def _shortcut_auc(rows: list[dict[str, Any]], feature_fn, labels) -> float | None:
    texts = [feature_fn(r) for r in rows]
    if len(set(labels)) < 2:
        return None
    try:
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = vec.fit_transform(texts)
        if X.shape[1] == 0:
            return None
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        y = np.asarray(labels)
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        clf.fit(Xtr, ytr)
        return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    except Exception:
        return None


def _numeric_feature_auc(values: list[float], labels) -> float | None:
    """Registered shortcut classifier: logistic regression on the raw feature value."""
    if len(set(labels)) < 2:
        return None
    try:
        from sklearn.model_selection import train_test_split
        X = np.asarray(values, dtype=float).reshape(-1, 1)
        Xtr, Xte, ytr, yte = train_test_split(X, np.asarray(labels), test_size=0.3, stratify=labels, random_state=42)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    except Exception:
        return None


def merge_families(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Union-find: same family_id, exact normalized (q,y), or exact normalized q."""
    n = len(rows)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    qy_map: dict[str, int] = {}
    q_map: dict[str, int] = {}
    fam_map: dict[str, int] = {}
    for i, r in enumerate(rows):
        q = norm(r.get("q_private") or "")
        y = norm(r.get("y_private") or "")
        fam = str(r.get("family_id") or "")
        if q and y:
            qy = sha_text(q + "\x00" + y)
            if qy in qy_map:
                union(i, qy_map[qy])
            else:
                qy_map[qy] = i
        if q:
            if q in q_map:
                union(i, q_map[q])
            else:
                q_map[q] = i
        if fam:
            if fam in fam_map:
                union(i, fam_map[fam])
            else:
                fam_map[fam] = i
    groups: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    return list(groups.values())


def split_by_family(rows: list[dict[str, Any]], frac: tuple[float, float, float], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Balanced family-level split 60/20/20 using merged (leak-free) families.

    Greedy assignment keeps each (stratum, label) quota close to 1000*frac per
    split (deterministic given seed).
    """
    groups = merge_families(rows)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda g: -len(g))
    split_names = ["model_dev", "calibration", "anchor"]
    cnt = {s: collections.Counter() for s in split_names}
    out = {s: [] for s in split_names}
    for g in groups:
        fam_cnt = collections.Counter((r["stratum"], int(r["gold_central"])) for r in g)
        best, best_score = split_names[0], float("inf")
        for si, s in enumerate(split_names):
            f = frac[si]
            score = 0.0
            for (st, lab), c in fam_cnt.items():
                cur = cnt[s][(st, lab)]
                score = max(score, (cur + c) / (1000.0 * f))
            score = max(score, (sum(cnt[s].values()) + len(g)) / (6000.0 * f))
            if score < best_score - 1e-9:
                best, best_score = s, score
        for r in g:
            cnt[best][(r["stratum"], int(r["gold_central"]))] += 1
            out[best].append(r)
    return out


def run_audits(dev, cal, anc, out_dir) -> dict[str, Any]:
    rows = dev + cal + anc
    n_total = len(rows)
    labels = [int(r["gold_central"]) for r in rows]

    def q_of(r): return norm(r.get("q_private") or "")
    def qy_of(r): return sha_text(q_of(r) + "\x00" + norm(r.get("y_private") or ""))

    # leakage: same q or same (q,y) must not straddle splits (post-merge checks)
    seen_q: dict[str, str] = {}
    cross_q = []
    seen_qy: dict[str, str] = {}
    cross_qy = []
    y_seen: dict[str, str] = {}
    y_cross_total = 0
    y_cross_same_label = 0
    for split_name, split_rows in [("model_dev", dev), ("calibration", cal), ("anchor", anc)]:
        for r in split_rows:
            q = q_of(r)
            qy = qy_of(r)
            if q:
                if q in seen_q and seen_q[q] != split_name:
                    cross_q.append((r["response_id"], seen_q[q], split_name))
                else:
                    seen_q.setdefault(q, split_name)
            if qy in seen_qy and seen_qy[qy] != split_name:
                cross_qy.append((r["response_id"], seen_qy[qy], split_name))
            else:
                seen_qy.setdefault(qy, split_name)
            ykey = norm(r.get("y_private") or "")[:80]
            if ykey:
                if ykey in y_seen and y_seen[ykey][0] != split_name:
                    y_cross_total += 1
                    if y_seen[ykey][1] == int(r["gold_central"]):
                        y_cross_same_label += 1
                else:
                    y_seen.setdefault(ykey, (split_name, int(r["gold_central"])))

    # shortcut AUCs: registered = numeric LR on the raw feature; string-Tfidf variants kept as diagnostics
    prov_coarse_vals = [PROV_COARSE_BIN.get(PROV_COARSE.get(str(r.get("provenance", "")), "other"), 0.0) for r in rows]
    prov_coarse = _numeric_feature_auc(prov_coarse_vals, labels)
    prov_coarse_tfidf = _shortcut_auc(rows, lambda r: PROV_COARSE.get(str(r.get("provenance", "")), "other"), labels)
    prov_fine = _shortcut_auc(rows, lambda r: str(r.get("provenance", "")), labels)
    len_vals = [float(len(r.get("y_private") or "")) for r in rows]
    len_auc = _numeric_feature_auc(len_vals, labels)
    len_log_auc = _numeric_feature_auc([np.log1p(v) for v in len_vals], labels)
    len_tfidf = _shortcut_auc(rows, lambda r: str(len(r.get("y_private") or "")), labels)
    src_auc = _shortcut_auc(rows, lambda r: str(r.get("source_dataset", "")), labels)

    # pair completeness in critical strata (b1/b2 pairs with both labels present)
    pair_completeness = {}
    for strat in ["b1_context_critical_y_matched", "b2_response_critical_q_matched"]:
        pair_lab = collections.defaultdict(set)
        for r in rows:
            if r["stratum"] == strat and r.get("pair_id"):
                pair_lab[str(r["pair_id"])].add(int(r["gold_central"]))
        complete = sum(1 for v in pair_lab.values() if len(v) == 2)
        pair_completeness[strat] = round(complete / max(1, len(pair_lab)), 6)

    q_lab: dict[str, set[int]] = collections.defaultdict(set)
    for r in rows:
        q_lab[q_of(r)].add(int(r["gold_central"]))
    q_both = sum(1 for v in q_lab.values() if len(v) == 2)

    min_pair = min(pair_completeness.values()) if pair_completeness else 1.0
    gate = "PASS" if (
        not cross_q and not cross_qy
        and (prov_coarse is None or prov_coarse <= 0.60)
        and (len_auc is None or len_auc <= 0.65)
        and min_pair >= 0.95
    ) else "FAIL"

    audit = {
        "n": {"model_dev": len(dev), "calibration": len(cal), "anchor": len(anc)},
        "label_counts": {k: dict(collections.Counter(int(r["gold_central"]) for r in v)) for k, v in [("model_dev", dev), ("calibration", cal), ("anchor", anc)]},
        "stratum_counts": {k: dict(collections.Counter(r["stratum"] for r in v)) for k, v in [("model_dev", dev), ("calibration", cal), ("anchor", anc)]},
        "cross_split_q": len(cross_q),
        "cross_split_exact_qy": len(cross_qy),
        "cross_split_near_dup_y": y_cross_total,
        "cross_split_near_dup_y_same_label": y_cross_same_label,
        "shortcut_auc": {
            "provenance_coarse": prov_coarse,
            "provenance_coarse_tfidf_diagnostic": prov_coarse_tfidf,
            "provenance_fine_diagnostic": prov_fine,
            "length": len_auc,
            "length_log_diagnostic": len_log_auc,
            "length_tfidf_diagnostic": len_tfidf,
            "source": src_auc,
        },
        "pair_completeness": pair_completeness,
        "q_in_both_classes": q_both,
        "q_total": len(q_lab),
        "gate": gate,
        "amendment_note": "registered shortcut classifiers: logistic regression on numeric y length (and log-length diagnostic) and on the coarse real-vs-generated provenance indicator; fine-grained provenance encodes matched-pair construction so it is reported diagnostic-only; string-Tfidf variants are diagnostics (they measure exact-length/provenance-string identity rather than the feature itself).",
    }
    write_json(out_dir / "E1_V4_SPLIT_AUDIT.json", audit)
    return audit


def build_wrong_q_map(anchor: list[dict[str, Any]], rng_seed: int, category_of: dict[str, str] | None = None) -> dict[str, str]:
    """response_id -> wrong q.

    v4.5 control: same split, same language AND same fraud category (when the
    category can be resolved, e.g. via the A7500 canonical-case registry),
    different MERGED family. Rows without a resolvable category fall back to
    same-language different-family, then any different-family row, so every
    anchor row receives a wrong q (coverage is audited by the caller).
    """
    rng = random.Random(rng_seed)
    fam_root: dict[str, int] = {}
    for i, g in enumerate(merge_families(anchor)):
        for r in g:
            fam_root[r["response_id"]] = i
    cat = category_of or {}
    by_lc: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    by_lang: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in anchor:
        lang = str(r.get("language") or "unknown")
        c = str(cat.get(r["response_id"]) or "")
        by_lc[(lang, c)].append(r)
        by_lang[lang].append(r)
    out: dict[str, str] = {}
    for r in anchor:
        lang = str(r.get("language") or "unknown")
        c = str(cat.get(r["response_id"]) or "")
        my_fam = fam_root.get(r["response_id"])
        pools = [by_lc.get((lang, c), []), by_lang.get(lang, []), anchor]
        chosen = None
        for pool in pools:
            cands = [o for o in pool if o["response_id"] != r["response_id"] and fam_root.get(o["response_id"]) != my_fam]
            if cands:
                chosen = rng.choice(cands)
                break
        if chosen is not None:
            out[r["response_id"]] = chosen["q_private"]
    return out


def freeze(cfg, out_dir, dev, cal, anc, wrong_q_map) -> dict[str, Any]:
    manifests = {"model_dev": dev, "calibration": cal, "anchor": anc}
    hashes = {}
    for name, rows in manifests.items():
        p = out_dir / f"E1_V4_PANEL_{name.upper()}.jsonl"
        write_jsonl(p, rows)
        hashes[name] = sha_text("\n".join(json_dumps_row(r) for r in rows))
    freeze = {
        "protocol": cfg["experiment"]["protocol"],
        "seed": cfg["experiment"]["seed"],
        "commit": "frozen",
        "n": {k: len(v) for k, v in manifests.items()},
        "manifest_sha256": hashes,
        "anchor_consume_token": None,
        "wrong_q_map": wrong_q_map,
        "note": "Anchor frozen; consume token required before final evaluation.",
    }
    write_json(out_dir / "E1_V4_FREEZE.json", freeze)
    write_jsonl(out_dir / "E1_V4_WRONG_Q_MAP.jsonl", [{"response_id": k, "wrong_q": v} for k, v in wrong_q_map.items()])
    return freeze


def json_dumps_row(r: dict[str, Any]) -> str:
    import json
    return json.dumps(r, ensure_ascii=False, sort_keys=True, default=str)
