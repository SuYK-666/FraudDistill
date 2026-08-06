"""Local multi-head Evidence Adapter (guide section 23).

Trains one LogisticRegression per task head on Exp3 train/dev samples that do
NOT overlap the Exp2 full pool; C is selected on Exp3 dev only. No API calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from frauddistill.exp2_static_repair.evidence import feature_names

HEADS = {
    "fraud": "FraudEvidenceAdapter",
    "harmful_compliance": "HarmfulComplianceAdapter",
    "over_refusal": "OverRefusalAdapter",
    "refusal_detection": "RefusalDetectionAdapter",
}


def train_adapter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    seed: int = 20260806,
) -> LogisticRegression:
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=5000,
        solver="liblinear",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def train_multihead(
    X_train: np.ndarray,
    y_heads: dict[str, np.ndarray],
    C: float = 1.0,
    seed: int = 20260806,
) -> dict[str, LogisticRegression]:
    models = {}
    for head, y in y_heads.items():
        if len(np.unique(y)) < 2:
            models[head] = None
            continue
        models[head] = train_adapter(X_train, y, C=C, seed=seed)
    return models


def predict_multihead(
    models: dict[str, LogisticRegression],
    X: np.ndarray,
) -> dict[str, np.ndarray]:
    out = {}
    for head, model in models.items():
        if model is None:
            out[head] = np.full(len(X), 0.5)
        else:
            out[head] = model.predict_proba(X)[:, 1]
    return out


def save_models(models: dict[str, LogisticRegression], out_dir: Path, meta: dict) -> None:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    for head, model in models.items():
        if model is not None:
            joblib.dump(model, out_dir / f"{head}_adapter.joblib")
    meta = {
        **meta,
        "feature_names": feature_names(),
        "heads": {h: (HEADS[h] if models.get(h) is not None else None) for h in HEADS},
        "frozen": True,
    }
    (out_dir / "adapter_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_models(out_dir: Path) -> dict[str, LogisticRegression]:
    import joblib

    models = {}
    for head in HEADS:
        p = out_dir / f"{head}_adapter.joblib"
        if p.exists():
            models[head] = joblib.load(p)
        else:
            models[head] = None
    return models
