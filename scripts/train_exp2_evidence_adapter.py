# -*- coding: utf-8 -*-
"""Train the shared multi-head Evidence Adapter (guide section 23).

Zero-API. Trains one LogisticRegression per task head on Exp3 train/dev
samples that do NOT overlap the Exp2 full pool. C is selected on Exp3 dev
only. The adapter only recombines saved specialist evidence; it adds no LLM
calls.

Outputs: experiments/exp2_prior_work_comparison/offline_rescore/shared_adapter/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_static_repair.adapter_model import (  # noqa: E402
    HEADS,
    predict_multihead,
    save_models,
    train_multihead,
)
from frauddistill.exp2_static_repair.evidence import row_features  # noqa: E402
from frauddistill.exp2_static_repair.offline_guard import require_offline  # noqa: E402

EXP3_AGENT_DIR = REPO / "experiments" / "exp3_agent_distillation_ablation" / "outputs" / "agent_predictions"
EXPERIMENT_DIR = REPO / "experiments" / "exp2_prior_work_comparison"
OUT_DIR = EXPERIMENT_DIR / "offline_rescore" / "shared_adapter"

C_GRID = [0.01, 0.1, 1.0, 10.0]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def exp2_overlap_ids() -> set[str]:
    ids = set()
    p = EXPERIMENT_DIR / "manifests" / "full_manifest.jsonl"
    if p.exists():
        for l in p.open(encoding="utf-8"):
            r = json.loads(l)
            ids.add(r.get("sample_id", ""))
            ids.add(r.get("qy_hash", ""))
    return ids


def load_exp3(split: str):
    p = EXP3_AGENT_DIR / f"{split}.jsonl"
    rows = read_jsonl(p)
    out = []
    for r in rows:
        rec = {
            "id": r.get("id", ""),
            "gold_label": (r.get("sample") or {}).get("gold_label"),
            "gold_type": (r.get("sample") or {}).get("gold_type"),
            "fraud": r.get("fraud") or {},
            "refusal": r.get("refusal") or {},
            "context": r.get("context") or {},
            "evidence_table": r.get("evidence_table") or {},
        }
        out.append(rec)
    return out


def make_targets(rows: list[dict], overlap: set[str]) -> tuple[list[dict], list[dict]]:
    kept, dropped = [], []
    for r in rows:
        if r["id"] in overlap:
            dropped.append(r)
            continue
        kept.append(r)
    return kept, dropped


def head_targets(rows: list[dict]) -> dict[str, np.ndarray]:
    """Multi-head labels (guide 23.4).

    Exp3 uses coarse gold_type="unsafe" for most positive rows after the
    Exp2-overlap exclusion drops fine-grained refusal_failure subtypes, so
    harmful_compliance/refusal_detection treat gold_label=="unsafe" as the
    positive class. The fraud head stays precise (fraud_assistance only).
    """
    y = {}
    y["fraud"] = np.array([1 if (r["gold_type"] == "fraud_assistance") else 0 for r in rows], dtype=int)
    y["harmful_compliance"] = np.array([
        1 if (r["gold_label"] == "unsafe" or r["gold_type"] in {"fraud_assistance", "refusal_failure"}) else 0
        for r in rows
    ], dtype=int)
    y["over_refusal"] = np.array([1 if (r["gold_type"] == "over_refusal") else 0 for r in rows], dtype=int)
    y["refusal_detection"] = np.array([
        1 if (r["gold_label"] == "unsafe" or r["gold_type"] in {"refusal_failure", "over_refusal"}) else 0
        for r in rows
    ], dtype=int)
    return y


def features(rows: list[dict]) -> np.ndarray:
    X = np.vstack([row_features({"agent_fraud_json": r["fraud"], "agent_refusal_json": r["refusal"], "agent_context_json": r["context"]}) for r in rows])
    return X


def auprc(y, s) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def select_C(Xtr, ytr, Xdv, ydv, head: str) -> float:
    best_c, best_ap = C_GRID[0], -1.0
    for c in C_GRID:
        from frauddistill.exp2_static_repair.adapter_model import train_adapter

        m = train_adapter(Xtr, ytr, C=c)
        ap = auprc(ydv, m.predict_proba(Xdv)[:, 1])
        if ap > best_ap:
            best_ap, best_c = ap, c
    return best_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-exp2-overlap", action="store_true", required=True)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        os.environ["FRAUDDISTILL_OFFLINE"] = "1"
    require_offline()

    overlap = exp2_overlap_ids()
    train_rows = load_exp3("train")
    dev_rows = load_exp3("dev")
    print(f"exp3 train={len(train_rows)} dev={len(dev_rows)} exp2_overlap_ids={len(overlap)}")

    train_rows, dropped_tr = make_targets(train_rows, overlap)
    dev_rows, dropped_dv = make_targets(dev_rows, overlap)
    print(f"after overlap exclusion: train={len(train_rows)} (dropped {len(dropped_tr)}) dev={len(dev_rows)} (dropped {len(dropped_dv)})")

    if len(train_rows) < 100 or len(dev_rows) < 50:
        raise SystemExit("insufficient non-overlapping training data")

    Xtr = features(train_rows)
    Xdv = features(dev_rows)
    ytr = head_targets(train_rows)
    ydv = head_targets(dev_rows)

    models = {}
    dev_report = {}
    for head in HEADS:
        if len(np.unique(ytr[head])) < 2:
            print(f"{head}: single class in train, skipping")
            models[head] = None
            dev_report[head] = {"skipped": True}
            continue
        c = select_C(Xtr, ytr[head], Xdv, ydv[head], head)
        models[head] = train_multihead(Xtr, {head: ytr[head]}, C=c)[head]
        s = models[head].predict_proba(Xdv)[:, 1]
        dev_report[head] = {
            "C": c,
            "dev_auprc": round(auprc(ydv[head], s), 4),
            "dev_n_positive": int(ydv[head].sum()),
            "dev_n": len(ydv[head]),
        }
        print(f"{head}: C={c} dev_auprc={dev_report[head]['dev_auprc']} n_pos={dev_report[head]['dev_n_positive']}")

    meta = {
        "guide": "exp2-static-fix-20260806-section23",
        "training_source": "exp3_agent_predictions train/dev",
        "excluded_exp2_overlap": True,
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "dev_report": dev_report,
        "model": "LogisticRegression(C=per-head, class_weight=balanced, liblinear, max_iter=5000)",
        "frozen": True,
    }
    save_models(models, OUT_DIR, meta)
    print("outputs ->", OUT_DIR)


if __name__ == "__main__":
    main()
