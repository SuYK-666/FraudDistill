from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pku_saferlhf(cache_dir: Path, relevant_categories: set[str], limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", cache_dir=str(cache_dir))
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(ds):
        safe0 = bool(rec["is_response_0_safe"])
        safe1 = bool(rec["is_response_1_safe"])
        cats0 = {k for k, v in rec["response_0_harm_category"].items() if v}
        cats1 = {k for k, v in rec["response_1_harm_category"].items() if v}
        rel = (cats0 | cats1) & relevant_categories
        if safe0 == safe1 or not rel:
            continue
        prompt = str(rec["prompt"])
        category = map_fraud_category(rel)
        for ridx, safe in [(0, safe0), (1, safe1)]:
            y = str(rec[f"response_{ridx}"])
            qid = f"pku_train_{idx:06d}"
            rows.append(
                {
                    "canonical_q_id": qid,
                    "source_dataset": "PKU-SafeRLHF",
                    "source_revision": "hf_default_snapshot_cached_2026-08-01",
                    "source_record_id": str(idx),
                    "source_split": "train",
                    "source_license": "CC-BY-NC-4.0",
                    "q_private": prompt,
                    "language": "zh" if has_cjk(prompt) else "en",
                    "language_origin": "native",
                    "actor_role": "PERPETRATOR",
                    "fraud_category": category,
                    "behavior_cluster_id": f"pku_{idx:06d}",
                    "response_id": f"{qid}|response_{ridx}",
                    "target_model_key": "public_artifact_model",
                    "target_model_snapshot": str(rec.get(f"response_{ridx}_source") or "unknown"),
                    "generation_mode": "public_pressure_artifact",
                    "replicate_id": 0,
                    "y_private": y,
                    "y_sha256": sha_text(y),
                    "status": "ok",
                    "source_response_id": str(rec.get(f"response_{ridx}_sha256") or f"response_{ridx}"),
                    "source_material_label": 0 if safe else 1,
                    "source_safety_label": "safe" if safe else "unsafe",
                    "source_harm_categories": sorted(rel),
                    "split_role": "",
                }
            )
        if limit and len(rows) >= limit * 2:
            break
    manifest = {
        "source_dataset": "PKU-SafeRLHF",
        "rows": len(rows),
        "exact_q_groups": len({r["canonical_q_id"] for r in rows}),
        "license": "CC-BY-NC-4.0",
        "revision": "hf_default_snapshot_cached_2026-08-01",
        "cache_dir": str(cache_dir),
    }
    return rows, manifest


def load_harmbench_behaviors(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for rec in csv.DictReader(handle):
            text = rec.get("Behavior", "")
            if fraudish(text) or rec.get("SemanticCategory") in {"illegal", "cybercrime_intrusion"}:
                rows.append(
                    {
                        "canonical_q_id": f"harmbench_{rec.get('BehaviorID')}",
                        "source_dataset": "HarmBench",
                        "source_revision": "local_third_party_snapshot",
                        "source_record_id": rec.get("BehaviorID"),
                        "source_split": "text_all",
                        "source_license": "MIT code; dataset terms recorded from upstream",
                        "q_private": text,
                        "language": "en",
                        "language_origin": "native",
                        "actor_role": "PERPETRATOR",
                        "fraud_category": infer_category(text),
                        "behavior_cluster_id": rec.get("BehaviorID"),
                    }
                )
    return rows, {"source_dataset": "HarmBench", "rows": len(rows), "file": str(path), "sha256": file_sha256(path)}


def load_fraudr1(path: Path, limit: int = 500) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = str(rec.get("prompt") or rec.get("user_query") or rec.get("q") or rec.get("text") or rec.get("messages") or "")[:4000]
        if not text:
            continue
        rows.append(
            {
                "canonical_q_id": f"fraudr1_{len(rows):06d}",
                "source_dataset": "Fraud-R1",
                "source_revision": "local_raw_snapshot",
                "source_record_id": str(rec.get("id") or len(rows)),
                "source_split": "raw_prompts",
                "source_license": "research/education per upstream",
                "q_private": text,
                "language": "zh" if has_cjk(text) else "en",
                "language_origin": "native",
                "actor_role": "VICTIM_DEFENDER",
                "fraud_category": infer_category(text),
                "behavior_cluster_id": f"fraudr1_{len(rows):06d}",
            }
        )
        if len(rows) >= limit:
            break
    return rows, {"source_dataset": "Fraud-R1", "rows": len(rows), "file": str(path), "sha256": file_sha256(path)}


def load_jbb_behaviors(revision: str = "not_downloaded") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {"source_dataset": "JailbreakBench", "rows": 0, "revision": revision, "status": "not_available_locally"}


def load_jbb_artifacts(method: str, model: str, revision: str = "not_downloaded") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {"source_dataset": "JailbreakBenchArtifacts", "rows": 0, "method": method, "model": model, "revision": revision, "status": "not_available_locally"}


def load_strongreject(revision: str = "not_downloaded") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {"source_dataset": "StrongREJECT", "rows": 0, "revision": revision, "status": "not_available_locally"}


def source_record_resolves(row: dict[str, Any]) -> bool:
    return bool(row.get("source_dataset") and row.get("source_record_id") not in {None, ""})


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")


def sha_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def fraudish(text: str) -> bool:
    low = (text or "").lower()
    keys = ["fraud", "scam", "phishing", "credential", "bank", "cash", "money", "fake", "impersonat", "login", "identity", "email", "romance"]
    return any(k in low for k in keys)


def infer_category(text: str) -> str:
    low = (text or "").lower()
    if "phish" in low or "login" in low or "credential" in low:
        return "phishing"
    if "impersonat" in low or "identity" in low:
        return "impersonation"
    if "job" in low or "recruit" in low:
        return "fake_job"
    if "romance" in low or "dating" in low or "relationship" in low:
        return "relationship_investment"
    return "fraudulent_service"


def map_fraud_category(categories: set[str]) -> str:
    if "Privacy Violation" in categories or "Cybercrime" in categories:
        return "phishing"
    if "Mental Manipulation" in categories:
        return "relationship_investment"
    if "White-Collar Crime" in categories:
        return "fraudulent_service"
    return "fraudulent_service"
