from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frauddistill.exp1_ccfa.semantic_components import attach_semantic_components
from frauddistill.utils.io import read_jsonl, write_jsonl


STATIC_TRAIN_SOURCES = [
    ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "fraudr1_all_categories_qy.jsonl",
    ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "exp1_fraudr1_full.jsonl",
    ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "v2_hard_control_full.jsonl",
    ROOT / "data" / "processed" / "qy_v3" / "judged_pairs_v3.jsonl",
]

GENERATION_FILE = ROOT / "data" / "generations" / "full" / "full_qwen_deepseek_20260706_211621" / "generations_success.jsonl"
SILVER_FILE = ROOT / "data" / "labels" / "silver" / "full" / "full_qwen_deepseek_20260706_211621" / "silver_success.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare E1-CPU-v5 train candidate pool")
    parser.add_argument("--output", default="data/prepared/exp1_cpu_v5/e1_cpu_v5_train_pool.jsonl")
    parser.add_argument("--audit", default="data/prepared/exp1_cpu_v5/e1_cpu_v5_train_pool_audit.json")
    parser.add_argument("--include_silver", action="store_true")
    parser.add_argument("--include_beavertails", action="store_true")
    parser.add_argument("--beavertails_rows", type=int, default=24000)
    args = parser.parse_args()

    rows, audit = build_train_pool(
        include_silver=args.include_silver,
        include_beavertails=args.include_beavertails,
        beavertails_rows=args.beavertails_rows,
    )
    output = ROOT / args.output
    audit_path = ROOT / args.audit
    write_jsonl(output, rows)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({**audit, "output": str(output)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**audit, "output": str(output), "audit": str(audit_path)}, ensure_ascii=False, indent=2))


def build_train_pool(include_silver: bool = True, include_beavertails: bool = False, beavertails_rows: int = 24000) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    source_counts: dict[str, int] = {}
    for path in STATIC_TRAIN_SOURCES:
        if not path.exists():
            continue
        added = 0
        for raw in read_jsonl(path):
            row = _normalize_static_row(raw, path.stem)
            if row:
                rows.append(row)
                added += 1
        source_counts[path.stem] = added
    if include_silver and GENERATION_FILE.exists() and SILVER_FILE.exists():
        silver_rows = _silver_rows()
        rows.extend(silver_rows)
        source_counts["silver_generation_confident"] = len(silver_rows)
    if include_beavertails:
        beaver_rows = _beavertails_rows(beavertails_rows)
        rows.extend(beaver_rows)
        source_counts["beavertails_train_stratified"] = len(beaver_rows)

    deduped, duplicates = _dedupe(rows)
    attached = sorted(attach_semantic_components(deduped), key=lambda row: _stable_hash(f"cpu_v5_order:{row['id']}"))
    label_counts: dict[str, int] = {}
    for row in attached:
        label_counts[row["exp1_label"]] = label_counts.get(row["exp1_label"], 0) + 1
    audit = {
        "raw_rows": len(rows),
        "rows": len(attached),
        "duplicates_removed": duplicates,
        "components": len({row["semantic_component_id"] for row in attached}),
        "by_source": source_counts,
        "by_label": label_counts,
        "include_silver": include_silver,
        "include_beavertails": include_beavertails,
        "p3_external_public_gold_excluded": True,
    }
    return attached, audit


def _normalize_static_row(raw: dict, source_name: str) -> dict | None:
    label = raw.get("pair_fraud_label") or raw.get("gold_label")
    if label not in {"safe", "unsafe"}:
        return None
    query = str(raw.get("user_query") or "").strip()
    answer = str(raw.get("target_model_answer") or "").strip()
    if not query or not answer:
        return None
    metadata = dict(raw.get("metadata") or {})
    source_prompt_id = _source_prompt_id(raw, metadata)
    return {
        "id": f"cpuv5_{source_name}_{raw.get('id')}",
        "source": raw.get("source") or source_name,
        "language": raw.get("language", "unknown"),
        "user_query": query,
        "target_model_answer": answer,
        "pair_fraud_label": label,
        "gold_label": label,
        "label_provenance": raw.get("label_provenance") or "train_candidate_existing_gold_or_proxy",
        "source_prompt_id": source_prompt_id,
        "metadata": {**metadata, "cpu_v5_train_source": source_name},
    }


def _silver_rows() -> list[dict]:
    generations = {str(row["id"]): row for row in read_jsonl(GENERATION_FILE)}
    rows: list[dict] = []
    for silver in read_jsonl(SILVER_FILE):
        auto = silver.get("auto_label") or {}
        label = auto.get("binary")
        if label not in {"safe", "unsafe"} or auto.get("confidence") != "silver_medium":
            continue
        generation = generations.get(str(silver.get("id")))
        if not generation:
            continue
        query = str(generation.get("user_query") or "").strip()
        answer = str(generation.get("target_model_answer") or "").strip()
        if not query or not answer:
            continue
        metadata = dict(generation.get("metadata") or {})
        source_prompt_id = metadata.get("source_prompt_id") or generation.get("id")
        rows.append(
            {
                "id": f"cpuv5_silver_{generation['id']}",
                "source": generation.get("source") or "api_generation_silver",
                "language": generation.get("language", "unknown"),
                "user_query": query,
                "target_model_answer": answer,
                "pair_fraud_label": label,
                "gold_label": label,
                "label_provenance": "silver_medium_train_only",
                "source_prompt_id": source_prompt_id,
                "target_model": generation.get("target_model"),
                "metadata": {
                    **metadata,
                    "cpu_v5_train_source": "silver_generation_confident",
                    "silver_risk_type": auto.get("risk_type"),
                    "silver_risk_score": auto.get("risk_score"),
                    "silver_vote_count": auto.get("vote_count"),
                },
            }
        )
    return rows


def _beavertails_rows(target_rows: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    candidates: dict[str, list[dict]] = {"safe": [], "unsafe": []}
    for index, item in enumerate(dataset):
        label = "safe" if bool(item.get("is_safe")) else "unsafe"
        category = item.get("category") or {}
        fraud_adjacent = _fraud_adjacent(category)
        if label == "unsafe" and not fraud_adjacent:
            continue
        if label == "safe" and not fraud_adjacent and index % 5 != 0:
            continue
        query = str(item.get("prompt") or "").strip()
        answer = str(item.get("response") or "").strip()
        if not query or not answer:
            continue
        component = f"beavertails_{_stable_hash(query)[:24]}"
        candidates[label].append(
            {
                "id": f"cpuv5_beavertails_{index}",
                "source": "BeaverTails",
                "language": "English",
                "user_query": query,
                "target_model_answer": answer,
                "pair_fraud_label": label,
                "gold_label": label,
                "label_provenance": "external_public_beavertails_is_safe_train",
                "source_prompt_id": component,
                "metadata": {
                    "cpu_v5_train_source": "beavertails_train_stratified",
                    "source_dataset": "PKU-Alignment/BeaverTails",
                    "official_split": "330k_train",
                    "category": category,
                    "fraud_adjacent": fraud_adjacent,
                },
            }
        )
    half = target_rows // 2
    selected = _stable_sample(candidates["unsafe"], half, "beaver_unsafe")
    selected.extend(_stable_sample(candidates["safe"], target_rows - len(selected), "beaver_safe"))
    return selected


def _dedupe(rows: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple[str, str]] = set()
    output: list[dict] = []
    duplicates = 0
    for row in rows:
        key = (_norm(row["user_query"]), _norm(row["target_model_answer"]))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        output.append(row)
    return output, duplicates


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _source_prompt_id(raw: dict, metadata: dict) -> str:
    if raw.get("source_prompt_id") or metadata.get("source_prompt_id"):
        return str(raw.get("source_prompt_id") or metadata.get("source_prompt_id"))
    if metadata.get("fraudr1_raw_id") is not None:
        parts = [
            raw.get("source") or "Fraud-R1",
            metadata.get("fraud_category"),
            metadata.get("fraud_subcategory"),
            metadata.get("fraudr1_variant"),
            metadata.get("fraudr1_raw_id"),
            raw.get("language"),
        ]
        return "fraudr1_component_" + "_".join(str(part) for part in parts if part not in {None, ""})
    row_id = str(raw.get("id") or "")
    for suffix in ("_unsafe", "_safe", "_unsafe_regular", "_safe_control"):
        if row_id.endswith(suffix):
            return row_id[: -len(suffix)]
    return row_id


def _fraud_adjacent(category: dict) -> bool:
    keys = {
        "financial_crime,property_crime,theft",
        "non_violent_unethical_behavior",
        "privacy_violation",
        "misinformation_regarding_ethics,laws_and_safety",
    }
    return any(bool(category.get(key)) for key in keys)


def _stable_sample(rows: list[dict], count: int, salt: str) -> list[dict]:
    return sorted(rows, key=lambda row: _stable_hash(f"{salt}:{row['id']}"))[: min(count, len(rows))]


def _stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
