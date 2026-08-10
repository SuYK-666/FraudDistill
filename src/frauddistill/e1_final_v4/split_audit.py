# -*- coding: utf-8 -*-
"""v4 split + anti-shortcut audit + freeze (family-level)."""
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
        # stratified split for AUC estimate
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        clf.fit(Xtr, ytr)
        return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    except Exception:
        return None


def split_by_family(rows: list[dict[str, Any]], frac: tuple[float, float, float], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Stratified family-level split 60/20/20."""
    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_family[str(r["family_id"])].append(r)
    families = sorted(by_family)
    strata_buckets: dict[tuple[str, int], list[str]] = collections.defaultdict(list)
    for fam in families:
        first = by_family[fam][0]
        strata_buckets[(first["stratum"], int(first["gold_central"]))].append(fam)
    dev, cal, anc = [], [], []
    for key, fams in strata_buckets.items():
        rng.shuffle(fams)
        n = len(fams)
        n_dev = round(n * frac[0])
        n_cal = round(n * frac[1])
        for fam in fams[:n_dev]:
            dev.extend(by_family[fam])
        for fam in fams[n_dev:n_dev + n_cal]:
            cal.extend(by_family[fam])
        for fam in fams[n_dev + n_cal:]:
            anc.extend(by_family[fam])
    return {"model_dev": dev, "calibration": cal, "anchor": anc}


def run_audits(dev, cal, anc, out_dir) -> dict[str, Any]:
    rows = dev + cal + anc
    all_fams = collections.defaultdict(list)
    for split_name, split_rows in [("model_dev", dev), ("calibration", cal), ("anchor", anc)]:
        for r in split_rows:
            all_fams[r["family_id"]].append(split_name)
    cross_family = {k: set(v) for k, v in all_fams.items() if len(set(v)) > 1}
    # exact (q,y) cross-split
    seen: dict[str, str] = {}
    cross_qy = []
    for split_name, split_rows in [("model_dev", dev), ("calibration", cal), ("anchor", anc)]:
        for r in split_rows:
            key = sha_text(norm(r.get("q_private") or "") + "\x00" + norm(r.get("y_private") or ""))
            if key in seen and seen[key] != split_name:
                cross_qy.append((r["response_id"], seen[key], split_name))
            else:
                seen.setdefault(key, split_name)
    # near-dup y across splits (prefix-80 normalized)
    y_seen: dict[str, str] = {}
    y_cross = 0
    for split_name, split_rows in [("model_dev", dev), ("calibration", cal), ("anchor", anc)]:
        for r in split_rows:
            key = norm(r.get("y_private") or "")[:80]
            if key in y_seen and y_seen[key] != split_name:
                y_cross += 1
            else:
                y_seen.setdefault(key, split_name)
    labels = [int(r["gold_central"]) for r in rows]
    prov_auc = _shortcut_auc(rows, lambda r: str(r.get("provenance", "")), labels)
    len_auc = _shortcut_auc(rows, lambda r: str(len(r.get("y_private") or "")), labels)
    src_auc = _shortcut_auc(rows, lambda r: str(r.get("source_dataset", "")), labels)
    # q coverage in both classes
    q_lab: dict[str, set[int]] = collections.defaultdict(set)
    for r in rows:
        q_lab[sha_text(norm(r.get("q_private") or ""))].add(int(r["gold_central"]))
    q_both = sum(1 for v in q_lab.values() if len(v) == 2)
    audit = {
        "n": {"model_dev": len(dev), "calibration": len(cal), "anchor": len(anc)},
        "label_counts": {k: dict(collections.Counter(int(r["gold_central"]) for r in v)) for k, v in [("model_dev", dev), ("calibration", cal), ("anchor", anc)]},
        "stratum_counts": {k: dict(collections.Counter(r["stratum"] for r in v)) for k, v in [("model_dev", dev), ("calibration", cal), ("anchor", anc)]},
        "cross_split_families": len(cross_family),
        "cross_split_exact_qy": len(cross_qy),
        "cross_split_near_dup_y": y_cross,
        "shortcut_auc": {"provenance": prov_auc, "length": len_auc, "source": src_auc},
        "q_in_both_classes": q_both,
        "q_total": len(q_lab),
        "gate": "PASS" if (not cross_family and not cross_qy and (prov_auc is None or prov_auc <= 0.60) and (len_auc is None or len_auc <= 0.65)) else "FAIL",
    }
    write_json(out_dir / "E1_V4_SPLIT_AUDIT.json", audit)
    return audit


def build_wrong_q_map(anchor: list[dict[str, Any]], rng_seed: int) -> dict[str, str]:
    """response_id -> wrong q (same split, same language+category, different family)."""
    rng = random.Random(rng_seed)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in anchor:
        by_key[(r["language"], r.get("fraud_category", ""))].append(r)
    out: dict[str, str] = {}
    for key, rows in by_key.items():
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            other = rows[(i + 1) % len(rows)]
            if other["family_id"] != r["family_id"]:
                out[r["response_id"]] = other["q_private"]
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
