from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STAGE_NAMES = {0: "BASE", 1: "CREDIBILITY", 2: "URGENCY", 3: "EMOTION"}


def normalize_language(value: Any) -> str:
    text = str(value or "").lower()
    return "zh" if "chinese" in text or text == "zh" or "_zh" in text else "en"


def canonical_id_for_item(item: dict[str, Any], file_key: str) -> str:
    language = normalize_language(item.get("language") or file_key)
    return f"fraudr1_{language}_{str(item.get('id') or '').strip()}"


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split())


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_canonical_cases(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    map_rows: list[dict[str, Any]] = []
    for file_key, rel in config["data"]["fraudr1_files"].items():
        path = Path(rel)
        rows = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        for item in rows:
            cid = canonical_id_for_item(item, file_key)
            language = normalize_language(item.get("language") or file_key)
            stages = _extract_stages(item)
            q_hash = sha_text(normalize_ws(stages[0]["generated_data"]) if stages else "")
            map_rows.append(
                {
                    "canonical_id": cid,
                    "file_key": file_key,
                    "source_id": item.get("id"),
                    "language": language,
                    "category": str(item.get("category") or "unknown"),
                    "data_type": str(item.get("data_type") or "unknown"),
                    "q_hash": q_hash,
                }
            )
            current = by_id.setdefault(
                cid,
                {
                    "canonical_id": cid,
                    "language": language,
                    "category": str(item.get("category") or "unknown"),
                    "subcategory": str(item.get("subcategory") or ""),
                    "data_type": str(item.get("data_type") or "unknown"),
                    "role_background": normalize_role_background(item.get("role_bg")),
                    "dataset_revision": config["experiment"]["fraudr1_revision"],
                    "source_ids": [],
                    "source_files": [],
                    "raw_record_sha256": sha_text(json.dumps(item, ensure_ascii=False, sort_keys=True)),
                },
            )
            if "levelup" in file_key:
                current["stages"] = stages
            elif "stages" not in current:
                current["stages"] = stages
            current["source_ids"].append(item.get("id"))
            current["source_files"].append(file_key)
    cases = sorted(by_id.values(), key=lambda row: (row["language"], row["category"], int(str(row["canonical_id"]).split("_")[-1])))
    audit = audit_cases(cases, map_rows)
    return cases, audit


def _extract_stages(item: dict[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    multi = item.get("multi-rounds fraud")
    if isinstance(multi, list) and len(multi) >= 4:
        for idx, row in enumerate(multi[:4]):
            generated = str(row.get("generated_data") or "")
            stages.append({"stage_id": idx, "stage_name": STAGE_NAMES[idx], "generated_data": generated, "generated_data_sha256": sha_text(normalize_ws(generated))})
    else:
        generated = str(item.get("generated text") or item.get("raw_data") or "")
        stages.append({"stage_id": 0, "stage_name": "BASE", "generated_data": generated, "generated_data_sha256": sha_text(normalize_ws(generated))})
    return stages


def normalize_role_background(value: Any) -> dict[str, Any]:
    bg = value if isinstance(value, dict) else {}
    occupations = bg.get("Receiver_occupation") or bg.get("receiver_occupation") or []
    if isinstance(occupations, str):
        occupations = [occupations]
    return {
        "receiver": bg.get("Receiver") or bg.get("receiver") or "unknown",
        "receiver_gender": bg.get("Receiver_gender") or bg.get("receiver_gender") or "unknown",
        "receiver_occupation": [str(x) for x in occupations if str(x).strip()],
    }


def audit_cases(cases: list[dict[str, Any]], map_rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = set(row["category"] for row in cases)
    languages = set(row["language"] for row in cases)
    valid_data_types = {"message", "email", "job posting"}
    checks = {
        "canonical_count_2141": len(cases) == 2141,
        "each_case_four_stages": all(len(row.get("stages", [])) == 4 for row in cases),
        "stage_ids_0_1_2_3": all([s.get("stage_id") for s in row.get("stages", [])] == [0, 1, 2, 3] for row in cases),
        "base_stage0_hash_present": all(row.get("stages", [{}])[0].get("generated_data_sha256") for row in cases),
        "languages_en_zh_only": languages == {"en", "zh"},
        "five_categories": len(categories) == 5 and "unknown" not in categories,
        "valid_data_types": all(row["data_type"] in valid_data_types for row in cases),
        "role_background_complete": all(row["role_background"]["receiver"] and row["role_background"]["receiver_occupation"] for row in cases),
        "q_hash_missing_zero": all(row.get("stages", [{}])[0].get("generated_data_sha256") for row in cases),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "raw_rows": len(map_rows),
        "canonical_rows": len(cases),
        "language_counts": dict(Counter(row["language"] for row in cases)),
        "category_counts": dict(Counter(row["category"] for row in cases)),
        "data_type_counts": dict(Counter(row["data_type"] for row in cases)),
        "map_rows": map_rows,
    }


def stable_hash(seed: int, *parts: Any) -> str:
    text = "||".join([str(seed), *[str(p) for p in parts]])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_split_manifest(cases: list[dict[str, Any]], seed: int, pilot_per_cell: int, dev_per_cell: int) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        buckets[(row["language"], row["category"])].append(row)
    pilot: list[dict[str, Any]] = []
    model_dev: list[dict[str, Any]] = []
    frozen: list[dict[str, Any]] = []
    for key in sorted(buckets):
        ordered = sorted(buckets[key], key=lambda row: stable_hash(seed, row["canonical_id"]))
        pilot.extend(dict(row, split_tag="PILOT_ONLY") for row in ordered[:pilot_per_cell])
        model_dev.extend(dict(row, split_tag="MODEL_DEV") for row in ordered[pilot_per_cell : pilot_per_cell + dev_per_cell])
        frozen.extend(dict(row, split_tag="FROZEN_ANCHOR") for row in ordered[pilot_per_cell + dev_per_cell :])
    all_ids = [r["canonical_id"] for r in [*pilot, *model_dev, *frozen]]
    split_by_type = {name: dict(Counter(r["data_type"] for r in rows)) for name, rows in {"pilot": pilot, "model_dev": model_dev, "frozen": frozen}.items()}
    single_split_types = []
    for data_type in set().union(*(set(v) for v in split_by_type.values())):
        present = [name for name, counts in split_by_type.items() if counts.get(data_type, 0) > 0]
        if len(present) == 1:
            single_split_types.append({"data_type": data_type, "split": present[0]})
    return {
        "seed": seed,
        "pilot_cases": pilot,
        "model_dev_cases": model_dev,
        "frozen_anchor_cases": frozen,
        "audit": {
            "pilot_count": len(pilot),
            "model_dev_count": len(model_dev),
            "frozen_anchor_count": len(frozen),
            "canonical_cross_split_overlap": len(all_ids) - len(set(all_ids)),
            "pilot_by_language_category": dict(Counter(f"{r['language']}|{r['category']}" for r in pilot)),
            "model_dev_by_language_category": dict(Counter(f"{r['language']}|{r['category']}" for r in model_dev)),
            "data_type_by_split": split_by_type,
            "data_type_single_split": single_split_types,
        },
    }
