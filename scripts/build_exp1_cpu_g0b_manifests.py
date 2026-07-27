from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frauddistill.exp1_ccfa.duplicate_audit import duplicate_audit
from frauddistill.exp1_ccfa.public_gold import build_p3_v1
from frauddistill.exp1_ccfa.semantic_components import attach_semantic_components, explicit_label_token_audit, leakage_audit
from frauddistill.utils.io import read_jsonl, write_jsonl


FRAUD_ADJACENT = {
    "financial_crime,property_crime,theft",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "misinformation_regarding_ethics,laws_and_safety",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed E1-CPU-v5 G0b manifests")
    parser.add_argument("--output_dir", default="data/prepared/exp1_cpu_v5/g0b")
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_manifests(output_dir, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_manifests(output_dir: Path, seed: int) -> dict:
    p3_rows, p3_audit = build_p3_v1(ROOT / "data" / "raw" / "aegis" / "test.json")
    used_components = {row["semantic_component_id"] for row in p3_rows}
    p3_polyguard_base_ids = {str(row.get("metadata", {}).get("source_base_id")) for row in p3_rows if row.get("source") == "PolyGuardPrompts"}

    aegis_train = _aegis_rows("train", "train")
    aegis_validation = _aegis_rows("validation", "validation")
    beaver_train = _beavertails_rows("330k_train", "train")
    beaver_test = _beavertails_rows("30k_test", "test")
    beaver_large_test = _beavertails_rows("330k_test", "test")
    polyguard_unused = _polyguard_unused_rows(p3_polyguard_base_ids)
    dna = _jsonl_source(ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "do_not_answer_qy.jsonl", "Do-Not-Answer")
    aegis_qy_extra = [
        row
        for row in _jsonl_source(ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "aegis_qy.jsonl", "Aegis-extra")
        if "test" not in str(row.get("metadata", {}).get("official_split", "")).lower()
    ]
    silver = _silver_rows()

    blocked_keys = _text_key_set(p3_rows)
    train = _take_mixed(
        [
            ("Aegis/Nemotron-V2", aegis_train, 5200),
            ("BeaverTails", beaver_train, 4300),
            ("project_silver", silver, 500),
        ],
        used_components,
        blocked_keys,
        seed,
    )
    model_dev = _take_mixed(
        [
            ("Aegis/Nemotron-V2", aegis_train, 520),
            ("BeaverTails", beaver_train, 430),
            ("project_silver", silver, 50),
        ],
        used_components | {row["semantic_component_id"] for row in train},
        blocked_keys,
        seed + 1,
    )
    threshold_dev = _take_mixed(
        [
            ("Aegis/Nemotron-V2", aegis_train, 520),
            ("BeaverTails", beaver_train, 430),
            ("project_silver", silver, 50),
        ],
        used_components | {row["semantic_component_id"] for row in train + model_dev},
        blocked_keys,
        seed + 2,
    )

    occupied = used_components | {row["semantic_component_id"] for row in train + model_dev + threshold_dev}
    p1_pool = [
        row
        for row in [*beaver_test, *beaver_large_test, *aegis_validation, *polyguard_unused, *aegis_qy_extra, *beaver_train, *aegis_train]
        if row["semantic_component_id"] not in occupied and not _text_key_hit(row, blocked_keys)
    ]
    p1 = _balanced_sample(p1_pool, 1200, seed + 3)
    p1 = _unique_component_rows(p1, 1200)
    blocked_keys.update(_text_key_set(p1))
    occupied |= {row["semantic_component_id"] for row in p1}

    p2_pool = [
        row
        for row in [*beaver_test, *aegis_validation, *dna, *beaver_train]
        if row["semantic_component_id"] not in occupied and not _text_key_hit(row, blocked_keys)
    ]
    p2, p2_audit = _build_p2_collision(p2_pool, 300, seed + 4)
    blocked_keys.update(_text_key_set(p2))

    manifests = {
        "g0_train": train,
        "g0_model_dev": model_dev,
        "g0_threshold_dev": threshold_dev,
        "g0_p1_mini": p1,
        "g0_p2_mini": p2,
        "p3_v1": p3_rows,
    }
    for name, rows in manifests.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)

    audit = _audit_manifests(manifests, p3_audit, p2_audit)
    (output_dir / "g0b_data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _aegis_rows(split: str, use: str) -> list[dict]:
    data = json.loads((ROOT / "data" / "raw" / "aegis" / f"{split}.json").read_text(encoding="utf-8"))
    rows = []
    for item in data:
        label = item.get("response_label")
        query = str(item.get("prompt") or "").strip()
        answer = str(item.get("response") or "").strip()
        if label not in {"safe", "unsafe"} or not query or not answer:
            continue
        row_id = str(item.get("id"))
        rows.append(
            {
                "id": f"g0b_aegis_{split}_{row_id}",
                "source": "Aegis/Nemotron-V2",
                "language": "English",
                "user_query": query,
                "target_model_answer": answer,
                "pair_fraud_label": label,
                "gold_label": label,
                "label_provenance": "external_public_aegis_response_label",
                "source_prompt_id": f"aegis_{split}_{row_id}",
                "metadata": {
                    "official_split": split,
                    "g0b_use": use,
                    "source_dataset": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                    "prompt_label": item.get("prompt_label"),
                    "response_label": item.get("response_label"),
                    "violated_categories": item.get("violated_categories"),
                    "license": "CC-BY-4.0",
                },
            }
        )
    return attach_semantic_components(rows)


def _beavertails_rows(split: str, use: str) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("PKU-Alignment/BeaverTails", split=split)
    rows = []
    for index, item in enumerate(dataset):
        label = "safe" if bool(item.get("is_safe")) else "unsafe"
        query = str(item.get("prompt") or "").strip()
        answer = str(item.get("response") or "").strip()
        if not query or not answer:
            continue
        category = item.get("category") or {}
        rows.append(
            {
                "id": f"g0b_beavertails_{split}_{index}",
                "source": "BeaverTails",
                "language": "English",
                "user_query": query,
                "target_model_answer": answer,
                "pair_fraud_label": label,
                "gold_label": label,
                "label_provenance": "external_public_beavertails_is_safe",
                "source_prompt_id": f"beavertails_{split}_{_hash(query)[:24]}",
                "metadata": {
                    "official_split": split,
                    "g0b_use": use,
                    "source_dataset": "PKU-Alignment/BeaverTails",
                    "category": category,
                    "fraud_adjacent": _fraud_adjacent(category),
                    "license": "CC-BY-NC-4.0",
                },
            }
        )
    return attach_semantic_components(rows)


def _jsonl_source(path: Path, source: str) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for raw in read_jsonl(path):
        label = raw.get("pair_fraud_label") or raw.get("gold_label")
        query = str(raw.get("user_query") or "").strip()
        answer = str(raw.get("target_model_answer") or "").strip()
        if label not in {"safe", "unsafe"} or not query or not answer:
            continue
        metadata = dict(raw.get("metadata") or {})
        prompt_id = metadata.get("source_prompt_id") or raw.get("source_prompt_id") or raw.get("id")
        rows.append(
            {
                "id": f"g0b_{source.lower().replace('-', '_')}_{raw.get('id')}",
                "source": source,
                "language": raw.get("language", "English"),
                "user_query": query,
                "target_model_answer": answer,
                "pair_fraud_label": label,
                "gold_label": label,
                "label_provenance": raw.get("label_provenance") or "external_or_project_existing_label",
                "source_prompt_id": f"{source}_{prompt_id}",
                "metadata": {**metadata, "g0b_use": "p2_candidate"},
            }
        )
    return attach_semantic_components(rows)


def _polyguard_unused_rows(blocked_base_ids: set[str]) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test")
    rows = []
    seen_base_ids = set()
    for item in dataset:
        base_id = str(item.get("id"))
        if base_id in blocked_base_ids or base_id in seen_base_ids:
            continue
        label = _safe_unsafe_label(item.get("response_label") or item.get("response_harm_label"))
        query = str(item.get("prompt") or "").strip()
        answer = str(item.get("response") or "").strip()
        if label not in {"safe", "unsafe"} or not query or not answer:
            continue
        seen_base_ids.add(base_id)
        rows.append(
            {
                "id": f"g0b_polyguard_unused_{base_id}_{item.get('language', 'unknown')}",
                "source": "PolyGuardPrompts-unused",
                "language": item.get("language", "unknown"),
                "user_query": query,
                "target_model_answer": answer,
                "pair_fraud_label": label,
                "gold_label": label,
                "label_provenance": "external_public_polyguard_unused",
                "source_prompt_id": f"polyguard_unused_{base_id}",
                "metadata": {
                    "source_dataset": "ToxicityPrompts/PolyGuardPrompts",
                    "source_base_id": base_id,
                    "response_label": item.get("response_label"),
                    "response_harm_label": item.get("response_harm_label"),
                    "license": "CC-BY-4.0",
                    "g0b_use": "p1_candidate",
                },
            }
        )
    return attach_semantic_components(rows)


def _silver_rows() -> list[dict]:
    gen_path = ROOT / "data" / "generations" / "full" / "full_qwen_deepseek_20260706_211621" / "generations_success.jsonl"
    silver_path = ROOT / "data" / "labels" / "silver" / "full" / "full_qwen_deepseek_20260706_211621" / "silver_success.jsonl"
    if not gen_path.exists() or not silver_path.exists():
        return []
    generations = {str(row["id"]): row for row in read_jsonl(gen_path)}
    rows = []
    for item in read_jsonl(silver_path):
        auto = item.get("auto_label") or {}
        label = auto.get("binary")
        if label not in {"safe", "unsafe"} or auto.get("confidence") != "silver_medium":
            continue
        generation = generations.get(str(item.get("id")))
        if not generation:
            continue
        metadata = dict(generation.get("metadata") or {})
        rows.append(
            {
                "id": f"g0b_silver_{generation['id']}",
                "source": generation.get("source") or "project_silver",
                "language": generation.get("language", "unknown"),
                "user_query": generation.get("user_query"),
                "target_model_answer": generation.get("target_model_answer"),
                "pair_fraud_label": label,
                "gold_label": label,
                "label_provenance": "project_silver_medium_train_only",
                "source_prompt_id": metadata.get("source_prompt_id") or generation["id"],
                "metadata": {**metadata, "g0b_use": "train_auxiliary", "silver_risk_score": auto.get("risk_score")},
            }
        )
    return attach_semantic_components(rows)


def _take_mixed(sources: list[tuple[str, list[dict], int]], blocked: set[str], blocked_keys: set[tuple[str, str]], seed: int) -> list[dict]:
    result = []
    used = set(blocked)
    for source_name, rows, count in sources:
        candidates = [row for row in rows if row["semantic_component_id"] not in used and not _text_key_hit(row, blocked_keys)]
        selected = []
        for row in _balanced_sample(candidates, count * 2, seed + len(result)):
            if row["semantic_component_id"] in used or _text_key_hit(row, blocked_keys):
                continue
            selected.append(row)
            used.add(row["semantic_component_id"])
            blocked_keys.update(_text_key_set([row]))
            if len(selected) >= count:
                break
        for row in selected:
            row = dict(row)
            row["metadata"] = {**dict(row.get("metadata") or {}), "g0b_manifest_source_quota": source_name}
            result.append(row)
    return sorted(result, key=lambda row: _hash(f"{seed}:manifest:{row['id']}"))


def _balanced_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    by_label = {"safe": [], "unsafe": []}
    for row in rows:
        by_label[row["exp1_label"]].append(row)
    target_unsafe = min(len(by_label["unsafe"]), count // 2)
    target_safe = min(len(by_label["safe"]), count - target_unsafe)
    selected = _stable_sample(by_label["unsafe"], target_unsafe, f"{seed}:unsafe")
    selected.extend(_stable_sample(by_label["safe"], target_safe, f"{seed}:safe"))
    if len(selected) < count:
        remaining = [row for row in rows if row not in selected]
        selected.extend(_stable_sample(remaining, count - len(selected), f"{seed}:fill"))
    return selected[:count]


def _build_p2_collision(rows: list[dict], target_groups: int, seed: int) -> tuple[list[dict], dict]:
    safe_rows = _stable_sample([row for row in rows if row["exp1_label"] == "safe"], 5000, f"{seed}:safe_pool")
    unsafe_rows = _stable_sample([row for row in rows if row["exp1_label"] == "unsafe"], 5000, f"{seed}:unsafe_pool")
    if not safe_rows or not unsafe_rows:
        return [], {"passed": False, "reason": "empty safe or unsafe pool"}
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    safe_y = [row["target_model_answer"] for row in safe_rows]
    unsafe_y = [row["target_model_answer"] for row in unsafe_rows]
    vectorizer.fit([*safe_y, *unsafe_y])
    safe_vec = vectorizer.transform(safe_y)
    unsafe_vec = vectorizer.transform(unsafe_y)
    nn = NearestNeighbors(n_neighbors=min(20, len(safe_rows)), metric="cosine")
    nn.fit(safe_vec)
    distances, indices = nn.kneighbors(unsafe_vec)
    selected: list[dict] = []
    used_components: set[str] = set()
    groups = 0
    similarities = []
    for unsafe_idx in np.argsort([_hash(row["id"]) for row in unsafe_rows]):
        unsafe = unsafe_rows[int(unsafe_idx)]
        if unsafe["semantic_component_id"] in used_components:
            continue
        for distance, safe_idx in zip(distances[int(unsafe_idx)], indices[int(unsafe_idx)]):
            similarity = 1.0 - float(distance)
            if similarity < 0.20:
                continue
            safe = safe_rows[int(safe_idx)]
            if safe["semantic_component_id"] in used_components:
                continue
            if not _length_ratio_ok(unsafe["target_model_answer"], safe["target_model_answer"]):
                continue
            group_id = f"p2_collision_{groups:04d}"
            for row in (unsafe, safe):
                item = dict(row)
                item["context_collision_group_id"] = group_id
                item["source_prompt_id"] = group_id
                item["metadata"] = {
                    **dict(item.get("metadata") or {}),
                    "p2_y_similarity": similarity,
                    "p2_group_source": "unsupervised_answer_neighbor",
                }
                selected.append(item)
            used_components.add(unsafe["semantic_component_id"])
            used_components.add(safe["semantic_component_id"])
            similarities.append(similarity)
            groups += 1
            break
        if groups >= target_groups:
            break
    selected = attach_semantic_components(selected)
    audit = {
        "groups": groups,
        "rows": len(selected),
        "target_groups": target_groups,
        "mixed_label_rate": 1.0 if groups else 0.0,
        "mean_y_similarity": float(np.mean(similarities)) if similarities else 0.0,
        "min_y_similarity": float(np.min(similarities)) if similarities else 0.0,
        "passed": groups >= target_groups,
    }
    return selected, audit


def _audit_manifests(manifests: dict[str, list[dict]], p3_audit: dict, p2_audit: dict) -> dict:
    split_audit = leakage_audit({name: rows for name, rows in manifests.items()})
    duplicate = duplicate_audit({name: rows for name, rows in manifests.items()})
    label_tokens = {name: explicit_label_token_audit(rows) for name, rows in manifests.items()}
    counts = {
        name: {
            "rows": len(rows),
            "components": len({row["semantic_component_id"] for row in rows}),
            "by_label": _count_by(rows, "exp1_label"),
            "by_source": _count_by(rows, "source"),
            "project_silver_rows": sum(1 for row in rows if str(row.get("label_provenance")) == "project_silver_medium_train_only"),
        }
        for name, rows in manifests.items()
    }
    train_rows = counts["g0_train"]["rows"]
    train_silver_ratio = counts["g0_train"]["project_silver_rows"] / max(train_rows, 1)
    gate = {
        "train_rows_exact": train_rows == 10000,
        "model_dev_rows_exact": counts["g0_model_dev"]["rows"] == 1000,
        "threshold_dev_rows_exact": counts["g0_threshold_dev"]["rows"] == 1000,
        "p1_components_min": counts["g0_p1_mini"]["components"] >= 1200,
        "p2_groups_min": p2_audit.get("groups", 0) >= 300,
        "leakage_passed": split_audit["passed"],
        "duplicates_passed": duplicate["passed"],
        "explicit_label_tokens_passed": all(row["passed"] for row in label_tokens.values()),
        "p3_passed": bool(p3_audit.get("passed")),
        "project_silver_train_lte_10pct": train_silver_ratio <= 0.10,
    }
    return {
        "gate": gate,
        "passed": all(gate.values()),
        "counts": counts,
        "split_audit": split_audit,
        "duplicate_audit": duplicate,
        "explicit_label_token_audit": label_tokens,
        "p2_audit": p2_audit,
        "p3_audit": p3_audit,
        "manifest_files": {name: f"{name}.jsonl" for name in manifests},
    }


def _stable_sample(rows: list[dict], count: int, salt: str) -> list[dict]:
    return sorted(rows, key=lambda row: _hash(f"{salt}:{row['id']}"))[: min(count, len(rows))]


def _unique_component_rows(rows: list[dict], count: int) -> list[dict]:
    selected = []
    seen = set()
    for row in rows:
        component = row["semantic_component_id"]
        if component in seen:
            continue
        selected.append(row)
        seen.add(component)
        if len(selected) >= count:
            break
    return selected


def _text_key_set(rows: list[dict]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        query = _norm(row.get("user_query", ""))
        answer = _norm(row.get("target_model_answer", ""))
        pair = f"{query}\n{answer}"
        keys.add(("q", query))
        keys.add(("y", answer))
        keys.add(("pair", pair))
    return keys


def _text_key_hit(row: dict, keys: set[tuple[str, str]]) -> bool:
    query = _norm(row.get("user_query", ""))
    answer = _norm(row.get("target_model_answer", ""))
    pair = f"{query}\n{answer}"
    return ("q", query) in keys or ("y", answer) in keys or ("pair", pair) in keys


def _norm(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _fraud_adjacent(category: dict) -> bool:
    return any(bool(category.get(key)) for key in FRAUD_ADJACENT)


def _length_ratio_ok(a: str, b: str) -> bool:
    ratio = len(a) / max(len(b), 1)
    return 0.5 <= ratio <= 2.0


def _count_by(rows: list[dict], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field))
        result[key] = result.get(key, 0) + 1
    return result


def _safe_unsafe_label(value: object) -> str | None:
    text = str(value or "").lower()
    if text in {"safe", "unharmful", "benign"}:
        return "safe"
    if text in {"unsafe", "harmful"}:
        return "unsafe"
    return None


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
