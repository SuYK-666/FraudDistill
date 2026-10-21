"""Freeze the supplementary data bundle and audit-selection plan offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .common import (REPO, budget_prefixes, read_json, read_jsonl, resolve_repo,
                     row_key, select_target_threshold, sha256_file, stable_rank,
                     write_json, write_jsonl)

SUPP_DIR = REPO / "experiments" / "lighteval_supplementary"
PREP_DIR = SUPP_DIR / "prepared"
RAW_DIR = PREP_DIR / "raw"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_config(path: str | Path) -> tuple[dict, Path]:
    p = resolve_repo(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")), p


def _partition(rows: list[dict], field: str, value: str) -> list[dict]:
    return [row for row in rows if str(row.get(field)) == value]


def _asset(path: Path, rows: int | None = None) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    item = {"path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if rows is not None:
        item["rows"] = rows
    return item


def _write_raw_sets(scale: dict) -> dict:
    paths = scale["paths"]
    expected = scale["expected_counts"]
    core_path = resolve_repo(paths["core_dataset"])
    core = read_jsonl(core_path)
    dev = _partition(core, "split", "dev")
    test = _partition(core, "split", "test")
    train_manifest = read_jsonl(resolve_repo(paths["train_manifest"]))
    calibration_all = read_jsonl(resolve_repo(paths["relation_calibration"]))
    relation_cal = _partition(calibration_all, "relation", "relation")
    hs_cal = _partition(calibration_all, "relation", "hard_safe")
    relation_test = read_jsonl(resolve_repo(paths["relation_test"]))
    hs_all = read_jsonl(resolve_repo(paths["hard_safe"]))
    hs_test = _partition(hs_all, "split", "test")
    hs_cal_direct = _partition(hs_all, "split", "cal")
    if {row_key(row) for row in hs_cal} != {row_key(row) for row in hs_cal_direct}:
        raise ValueError("hard-safe calibration rows disagree between calibration_manifest and hard_safe_manifest")

    checks = {
        "train": (len(train_manifest), expected["train"]),
        "core_dev": (len(dev), expected["core_dev"]),
        "core_test": (len(test), expected["core_test"]),
        "relation_calibration": (len(relation_cal), expected["relation_calibration"]),
        "hard_safe_calibration": (len(hs_cal), expected["hard_safe_calibration"]),
        "relation_test": (len(relation_test), expected["relation_test"]),
        "hard_safe_test": (len(hs_test), expected["hard_safe_test"]),
    }
    bad = {key: {"actual": actual, "expected": wanted}
           for key, (actual, wanted) in checks.items() if actual != wanted}
    if bad:
        raise ValueError(f"asset count mismatch: {bad}")
    if sum(row.get("binary_label") == "unsafe" for row in relation_test) != 192:
        raise ValueError("E6 relation test is not 192 unsafe / 192 safe")
    if sum(row.get("binary_label") == "unsafe" for row in relation_cal) != 48:
        raise ValueError("E6 relation calibration is not 48 unsafe / 48 safe")
    if any(row.get("binary_label") != "safe" for row in hs_test + hs_cal):
        raise ValueError("hard-safe controls contain a non-safe label")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(RAW_DIR / "core_dev.jsonl", dev)
    write_jsonl(RAW_DIR / "core_test.jsonl", test)
    write_jsonl(RAW_DIR / "relation_calibration.jsonl", relation_cal)
    write_jsonl(RAW_DIR / "hard_safe_calibration.jsonl", hs_cal)
    write_jsonl(RAW_DIR / "relation_test.jsonl", relation_test)
    write_jsonl(RAW_DIR / "hard_safe_test.jsonl", hs_test)

    ids = {
        "core_dev": {row_key(row) for row in dev},
        "core_test": {row_key(row) for row in test},
        "relation_calibration": {row_key(row) for row in relation_cal},
        "relation_test": {row_key(row) for row in relation_test},
        "hard_safe_calibration": {row_key(row) for row in hs_cal},
        "hard_safe_test": {row_key(row) for row in hs_test},
    }
    overlaps = {}
    names = list(ids)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = ids[left] & ids[right]
            if overlap:
                overlaps[f"{left}__{right}"] = len(overlap)
    if overlaps:
        raise ValueError(f"unexpected evaluation-set ID overlap: {overlaps}")

    return {
        "counts": {key: actual for key, (actual, _) in checks.items()},
        "raw_files": {
            "core_dev": _asset(RAW_DIR / "core_dev.jsonl", len(dev)),
            "core_test": _asset(RAW_DIR / "core_test.jsonl", len(test)),
            "relation_calibration": _asset(RAW_DIR / "relation_calibration.jsonl", len(relation_cal)),
            "hard_safe_calibration": _asset(RAW_DIR / "hard_safe_calibration.jsonl", len(hs_cal)),
            "relation_test": _asset(RAW_DIR / "relation_test.jsonl", len(relation_test)),
            "hard_safe_test": _asset(RAW_DIR / "hard_safe_test.jsonl", len(hs_test)),
        },
        "sources": {
            "train_manifest": _asset(resolve_repo(paths["train_manifest"]), len(train_manifest)),
            "core_dataset": _asset(core_path, len(core)),
            "relation_calibration_all": _asset(resolve_repo(paths["relation_calibration"]), len(calibration_all)),
            "relation_test": _asset(resolve_repo(paths["relation_test"]), len(relation_test)),
            "hard_safe_all": _asset(resolve_repo(paths["hard_safe"]), len(hs_all)),
        },
        "id_overlaps": {},
    }


def _write_upload_manifest(scale: dict, ablation: dict, prepared: dict) -> dict:
    paths = [
        resolve_repo(scale["paths"]["train_manifest"]),
        resolve_repo(scale["paths"]["core_dataset"]),
        resolve_repo(scale["paths"]["relation_calibration"]),
        resolve_repo(scale["paths"]["relation_test"]),
        resolve_repo(scale["paths"]["hard_safe"]),
        resolve_repo(ablation["paths"]["student_predictions"]),
        resolve_repo(ablation["paths"]["source_threshold"]),
        resolve_repo(ablation["paths"]["e5_audit_cache"]),
    ]
    checkpoint = resolve_repo(scale["paths"]["deepseek_checkpoint"])
    if checkpoint.exists():
        paths.extend(path for path in checkpoint.iterdir() if path.is_file())
    files = []
    for path in paths:
        if path.exists() and path.is_file():
            row_count = sum(1 for _ in path.open("rb")) if path.suffix == ".jsonl" else None
            files.append(_asset(path, row_count))
    manifest = {
        "generated_at_utc": utc_now(),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "model_downloads_on_gpu_only": [model["hf_id"] for model in scale["models"] if model.get("train")],
        "prepared_counts": prepared["counts"],
    }
    write_json(PREP_DIR / "upload_manifest.json", manifest)
    return manifest

def _prepare_ablation(ablation: dict, prepared_rel: dict) -> dict:
    paths = ablation["paths"]
    expected = ablation["expected_counts"]
    cal = read_jsonl(resolve_repo(paths["calibration_manifest"]))
    rel_cal = _partition(cal, "relation", "relation")
    hs_cal = _partition(cal, "relation", "hard_safe")
    test = read_jsonl(resolve_repo(paths["test_manifest"]))
    hs_all = read_jsonl(resolve_repo(paths["hard_safe_manifest"]))
    hs_test = _partition(hs_all, "split", "test")
    checks = {
        "relation_test": (len(test), expected["relation_test"]),
        "relation_calibration": (len(rel_cal), expected["relation_calibration"]),
        "hard_safe_calibration": (len(hs_cal), expected["hard_safe_calibration"]),
        "hard_safe_test": (len(hs_test), expected["hard_safe_test"]),
    }
    bad = {key: {"actual": actual, "expected": wanted}
           for key, (actual, wanted) in checks.items() if actual != wanted}
    if bad:
        raise ValueError(f"ablation count mismatch: {bad}")

    prediction_path = resolve_repo(paths["student_predictions"])
    predictions = read_jsonl(prediction_path)
    pred_map = {row_key(row): row for row in predictions}
    source_threshold = float(read_json(paths["source_threshold"]).get("p0", ablation["threshold"]["source"]))
    target_threshold_info = read_json(paths["source_threshold"]).get("p1") or {}
    target_threshold = target_threshold_info.get("threshold") or ablation["threshold"]["target"]
    target_threshold = float(target_threshold)

    score_of = {key: float(row["risk_score"]) for key, row in pred_map.items()}
    missing = [row_key(row) for row in rel_cal + hs_cal + test + hs_test if row_key(row) not in score_of]
    if missing:
        raise ValueError(f"student predictions missing {len(missing)} calibration/test rows")

    selected = select_target_threshold(
        rel_cal, score_of, hs_cal, score_of,
        recall_min=float(ablation["threshold"]["target_recall_min"]),
        fpr_max=float(ablation["threshold"]["target_hard_safe_fpr_max"]),
        frozen_threshold=source_threshold,
    )
    # The current frozen protocol records 0.2323. Refuse silently changing it.
    if target_threshold_info.get("feasible") and target_threshold_info.get("threshold") is not None:
        recorded = float(target_threshold_info["threshold"])
        if abs(recorded - float(target_threshold)) > 1e-12:
            raise ValueError("inconsistent target threshold in threshold_selection.json")
        if not selected.get("feasible") or abs(selected["threshold"] - recorded) > 1e-6:
            raise ValueError(
                "recomputed target threshold does not match frozen E6 record; "
                "do not silently change the student's operating point"
            )

    audit = ablation["audit"]
    prefix_info = budget_prefixes(len(test), audit["rates"], int(audit["max_items"]))
    if prefix_info[-1]["prefix"] != int(audit["max_items"]):
        raise ValueError(f"expected max audit prefix {audit['max_items']}, got {prefix_info[-1]}")
    ranked = stable_rank(test, score_of, target_threshold, int(audit["max_items"]))
    by_key = {row_key(row): row for row in test}

    selection_rows = []
    task_rows = []
    for item in ranked:
        src = by_key[item["id"]]
        selection_rows.append({
            **item,
            "family_id": src.get("family_id"),
            "slot": src.get("slot"),
            "prompt_id": src.get("prompt_id"),
            "language": src.get("language"),
            "student_pred": "unsafe" if item["risk_score"] >= target_threshold else "safe",
        })
        # Reviewer-visible payload: q+y only. Scores, source labels, and family
        # metadata remain in the separate selection manifest.
        task_rows.append({
            "id": item["id"],
            "rank": item["rank"],
            "user_query": src["user_query"],
            "target_model_answer": src["target_model_answer"],
        })

    # Cache accounting is deliberately conservative: only count a cache hit if
    # its prompt/model metadata matches, or if it came from the explicit legacy
    # E5 cache whose prompt fingerprint is recorded in the protocol.
    e5_cache_path = resolve_repo(paths["e5_audit_cache"])
    e5_cache = read_jsonl(e5_cache_path)
    from frauddistill.e4e5_v2.schemas import qy_hash
    from frauddistill.e4e5_v2.deepseek_fallback import FALLBACK_PROMPT

    task_hash = hashlib.sha256(FALLBACK_PROMPT.encode("utf-8")).hexdigest()[:16]
    e5_by_hash = {row.get("qy_hash"): row for row in e5_cache}
    exact_hits = 0
    for item in selection_rows:
        src = by_key[item["id"]]
        item["qy_hash"] = qy_hash(src["user_query"], src["target_model_answer"])
        if item["qy_hash"] in e5_by_hash:
            exact_hits += 1

    out = {
        "protocol": ablation["protocol"],
        "generated_at_utc": utc_now(),
        "counts": {key: actual for key, (actual, _) in checks.items()},
        "source_threshold": source_threshold,
        "target_threshold": target_threshold,
        "threshold_recomputed": selected,
        "threshold_matches_frozen_record": abs(target_threshold - 0.2323) <= 1e-12,
        "budget_prefixes": prefix_info,
        "audit_max_items": len(selection_rows),
        "ranking": audit["ranking"],
        "source_assets": {
            "calibration_manifest": _asset(resolve_repo(paths["calibration_manifest"]), len(cal)),
            "test_manifest": _asset(resolve_repo(paths["test_manifest"]), len(test)),
            "hard_safe_manifest": _asset(resolve_repo(paths["hard_safe_manifest"]), len(hs_all)),
            "student_predictions": _asset(prediction_path, len(predictions)),
            "threshold_selection": _asset(resolve_repo(paths["source_threshold"])),
            "e5_audit_cache": _asset(e5_cache_path, len(e5_cache)) if e5_cache_path.exists() else None,
        },
        "e5_cache_plan": {
            "rows": len(e5_cache),
            "selected_rows_with_exact_qy_match": exact_hits,
            "compatible_for_reuse": exact_hits,
            "known_prompt_version": audit["e5_cache_compatibility"]["known_e5_prompt"],
            "known_prompt_sha256_prefix": task_hash,
            "note": "No E5 row is reused unless the selected q+y hash matches and prompt/model compatibility is enabled.",
        },
        "prepared_assets": prepared_rel,
    }
    write_jsonl(PREP_DIR / "audit_selection_manifest.jsonl", selection_rows)
    write_jsonl(RAW_DIR / "audit_inputs.jsonl", task_rows)
    out["audit_artifacts"] = {
        "selection_manifest": _asset(PREP_DIR / "audit_selection_manifest.jsonl", len(selection_rows)),
        "reviewer_inputs": _asset(RAW_DIR / "audit_inputs.jsonl", len(task_rows)),
    }
    write_json(PREP_DIR / "ablation_plan.json", out)
    return out


def prepare(scale_config: str | Path, ablation_config: str | Path) -> dict:
    scale, scale_path = _load_config(scale_config)
    ablation, ablation_path = _load_config(ablation_config)
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    prepared = _write_raw_sets(scale)
    model_specs = {
        "protocol": scale["protocol"],
        "seed": int(scale["seed"]),
        "training": scale["training"],
        "calibration": scale["calibration"],
        "cost": scale["cost"],
        "models": scale["models"],
    }
    write_json(PREP_DIR / "scale_model_specs.json", model_specs)
    ablation_plan = _prepare_ablation(ablation, {
        "raw_files": prepared["raw_files"],
        "counts": prepared["counts"],
    })

    upload_manifest = _write_upload_manifest(scale, ablation, prepared)
    lock = {
        "generated_at_utc": utc_now(),
        "python": platform.python_version(),
        "protocols": {"scale": scale["protocol"], "ablation": ablation["protocol"]},
        "configs": {
            "scale": _asset(scale_path),
            "ablation": _asset(ablation_path),
        },
        "prepared": prepared,
        "ablation": ablation_plan,
        "upload_manifest": {
            "path": str((PREP_DIR / "upload_manifest.json").relative_to(REPO)),
            "file_count": len(upload_manifest["files"]),
            "total_bytes": upload_manifest["total_bytes"],
        },
        "scale_models": [{
            "id": model["id"], "hf_id": model["hf_id"], "params_b": model["params_b"],
            "train": bool(model.get("train")), "reuse": bool(model.get("reuse")),
        } for model in scale["models"]],
    }
    write_json(PREP_DIR / "asset_lock.json", lock)
    return lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-config", default="configs/supplementary/lighteval_scale.yaml")
    parser.add_argument("--ablation-config", default="configs/supplementary/lighteval_ablation.yaml")
    args = parser.parse_args(argv)
    lock = prepare(args.scale_config, args.ablation_config)
    print(json.dumps({
        "status": "READY",
        "output": str(PREP_DIR / "asset_lock.json"),
        "counts": lock["prepared"]["counts"],
        "target_threshold": lock["ablation"]["target_threshold"],
        "audit_prefixes": lock["ablation"]["budget_prefixes"],
        "e5_cache_plan": lock["ablation"]["e5_cache_plan"],
        "model_ids": [m["id"] for m in lock["scale_models"]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
