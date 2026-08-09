# -*- coding: utf-8 -*-
"""MODEL_LOCK / LABEL_POLICY_LOCK / PROTOCOL_LOCK creation."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .schemas import file_sha256


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty(repo: Path) -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo), capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return True


def build_model_lock(repo: Path, cfg: dict, protocol_id: str, out_dir: Path) -> dict:
    ckpt = repo / cfg["primary_model"]["checkpoint"]
    ckpt_dir = ckpt if (ckpt / "adapter_model.safetensors").exists() else ckpt
    adapter = ckpt_dir / "adapter_model.safetensors"
    manifest = repo / "data/prepared/exp3_neural_student/final_train_manifest.jsonl"
    lock = {
        "protocol_version": "e4e5_v1",
        "protocol_id": protocol_id,
        "primary_model": cfg["primary_model"]["name"],
        "checkpoint": str(ckpt_dir),
        "checkpoint_sha256": file_sha256(adapter) if adapter.exists() else "MISSING",
        "base_model": cfg["primary_model"]["base_model"],
        "training_manifest_sha256": file_sha256(manifest),
        "max_length": cfg["primary_model"]["max_length"],
        "risk_score": cfg["primary_model"]["risk_score"],
        "frozen_threshold": cfg["primary_model"]["threshold"],
        "primary": True,
        "created_before_test_labels_visible": True,
        "git_commit": git_commit(repo),
        "git_dirty": git_dirty(repo),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MODEL_LOCK.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    return lock


def build_label_policy_lock(repo: Path, out_dir: Path) -> dict:
    lock = {
        "protocol_version": "e4e5_v1",
        "binary_label": {"safe": 0, "unsafe": 1},
        "unsafe_types": ["fraud_assistance", "refusal_failure", "over_refusal"],
        "risk_score": "1 - P(type=safe)",
        "four_class": ["safe", "fraud_assistance", "refusal_failure", "over_refusal"],
        "strict_fraud_positive": "fraud_assistance",
        "mapping_changes": [],
        "note": "Canonical mapping identical to Exp3 final student; no re-interpretation allowed.",
        "git_commit": git_commit(repo),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "LABEL_POLICY_LOCK.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    return lock


def build_protocol_lock(repo: Path, cfg: dict, protocol_id: str, out_dir: Path,
                        model_lock: dict, registry_summary: dict) -> dict:
    lock = {
        "protocol_version": "e4e5_v1",
        "protocol_id": protocol_id,
        "research_questions": ["RQ4-1..4", "RQ5-1..4"],
        "model_lock": {"primary": cfg["primary_model"]["name"], "sha256": model_lock["checkpoint_sha256"]},
        "label_policy": "LABEL_POLICY_LOCK.json",
        "panel": cfg["panel"],
        "overlap": cfg["overlap"],
        "statistics": cfg["statistics"],
        "api": cfg["api"],
        "exposure_registry": registry_summary,
        "git_commit": git_commit(repo),
        "created_before_test_consume": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PROTOCOL_LOCK.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    return lock
