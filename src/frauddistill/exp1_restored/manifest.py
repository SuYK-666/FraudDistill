from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from frauddistill.exp1_ccfa.fraud_taxonomy import load_taxonomy
from frauddistill.exp1_ccfa.relation_manifest import load_public_sources_v6r2, refusal_marker, stable_hash
from frauddistill.utils.io import read_jsonl, write_jsonl


BUCKET_ALIASES = {
    "hard_unsafe_qwen_fn_or_phishing": "hard_unsafe",
    "or_bench_hard_safe": "hard_benign_safe",
}


def build_anchor_manifest(config: dict, output_dir: Path) -> dict:
    rows = [normalize_restored_row(row, "anchor2400") for row in read_jsonl(config["data"]["anchor_file"])]
    rows = assign_stratified_splits(rows, {"train": 1440, "model_dev": 480, "formal_test": 480}, int(config["data"]["seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "anchor2400.jsonl", rows)
    audit = manifest_audit(rows, config["data"]["anchor_buckets"])
    write_json(output_dir / "E1_ANCHOR2400_AUDIT.json", audit)
    write_label_provenance(output_dir / "E1_LABEL_PROVENANCE.csv", rows)
    return audit


def build_full_manifest(config: dict, output_dir: Path, taxonomy_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor = [normalize_restored_row(row, "anchor2400") for row in read_jsonl(config["data"]["anchor_file"])]
    used_ids = {row["id"] for row in anchor}
    selected: list[dict] = []
    selected.extend(anchor)
    relation_rows = load_relation_challenge(int(config["data"]["relation_challenge_groups"]), int(config["data"]["seed"]))
    for row in relation_rows:
        if row["id"] not in used_ids:
            selected.append(row)
            used_ids.add(row["id"])
    selected = fill_from_public(selected, used_ids, config, taxonomy_path)
    selected = exact_bucket_trim(selected, config["data"]["buckets"], int(config["data"]["seed"]))
    splits = assign_formal_aware_splits(selected, config, int(config["data"]["seed"]))
    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    all_rows = [row for rows in splits.values() for row in rows]
    write_json(output_dir / "E1_SPLIT_AUDIT.json", split_audit(splits, config))
    write_json(output_dir / "E1_MANIFEST_HASHES.json", {f"{split}.jsonl": sha256(output_dir / f"{split}.jsonl") for split in splits})
    write_json(output_dir / "E1_PROTOCOL_LOCK.json", {"protocol": config["experiment"]["protocol"], "manifest_hashes": {f"{split}.jsonl": sha256(output_dir / f"{split}.jsonl") for split in splits}})
    write_json(output_dir / "E1_DATASET_REVISION_LOCK.json", {"anchor": config["data"]["anchor_file"], "public_sources": ["PKU-SafeRLHF", "Aegis/Nemotron-V2", "BeaverTails"], "relation_challenge": "latest local v6r3 archive if available"})
    write_label_provenance(output_dir / "E1_LABEL_PROVENANCE.csv", all_rows)
    audit = manifest_audit(all_rows, config["data"]["buckets"])
    split = split_audit(splits, config)
    return {**audit, "split_audit": split, "passed": audit["bucket_targets_met"] and audit["label_balance"] and split["passed"]}


def normalize_restored_row(row: dict, provenance: str) -> dict:
    item = dict(row)
    metadata = dict(item.get("metadata") or {})
    bucket = BUCKET_ALIASES.get(str(metadata.get("data_role") or item.get("bucket") or ""), str(metadata.get("data_role") or item.get("bucket") or "unsafe_regular"))
    item["bucket"] = bucket
    item["gold_label"] = item.get("gold_label") or item.get("exp1_label")
    item["exp1_label"] = item["gold_label"]
    item["label_provenance"] = item.get("label_provenance") or provenance
    item["canonical_prompt_cluster"] = canonical_cluster(item)
    item["relation_challenge"] = int(item.get("relation_challenge", 0))
    return item


def load_relation_challenge(groups: int, seed: int) -> list[dict]:
    candidates = sorted(Path("archive").glob("**/E1_V6R3_R2_SELECTED_GROUPS.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return []
    rows = [normalize_restored_row(row, "v6r3_relation_challenge") for row in read_jsonl(candidates[0])]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("relation_group_id"))].append(row)
    out = []
    for gid, values in sorted(grouped.items(), key=lambda item: stable_hash(f"{seed}:relation:{item[0]}")):
        if len(values) != 2:
            continue
        for row in values:
            item = dict(row)
            item["bucket"] = "hard_unsafe" if item.get("gold_label") == "unsafe" else "safe_refusal_generated"
            item["relation_challenge"] = 1
            item["label_provenance"] = "v6r3_component_matched_public_label"
            item["canonical_prompt_cluster"] = f"relation_{gid}"
            out.append(item)
        if len(out) // 2 >= groups:
            break
    return out


def fill_from_public(rows: list[dict], used_ids: set[str], config: dict, taxonomy_path: Path) -> list[dict]:
    taxonomy = load_taxonomy(taxonomy_path)
    sources = load_public_sources_v6r2(config, taxonomy)
    pool = []
    for source_rows in sources.values():
        for row in source_rows:
            item = public_to_bucket(row)
            if item and item["id"] not in used_ids:
                pool.append(item)
    counts = Counter(row["bucket"] for row in rows)
    targets = config["data"]["buckets"]
    for bucket, target in targets.items():
        need = int(target) - counts[bucket]
        if need <= 0:
            continue
        choices = [row for row in pool if row["bucket"] == bucket and row["id"] not in used_ids]
        for row in sorted(choices, key=lambda item: stable_hash(f"{config['data']['seed']}:{bucket}:{item['id']}"))[:need]:
            rows.append(row)
            used_ids.add(row["id"])
    return rows


def public_to_bucket(row: dict) -> dict | None:
    label = row.get("exp1_label") or row.get("gold_label")
    item = normalize_restored_row(
        {
            **row,
            "id": f"public_{row.get('source','source')}_{row.get('row_uid') or row.get('id')}",
            "gold_label": label,
            "label_provenance": row.get("label_provenance") or f"{row.get('source')} response-level public label",
        },
        "public_response_label",
    )
    if label == "unsafe":
        item["bucket"] = "hard_unsafe" if len(str(row.get("target_model_answer", ""))) < 220 else "unsafe_regular"
        return item
    if label == "safe":
        answer = str(row.get("target_model_answer", ""))
        if refusal_marker(answer):
            item["bucket"] = "safe_refusal_generated"
        elif any(term in f"{row.get('user_query','')} {answer}".lower() for term in ("fraud", "scam", "phishing", "report", "prevent", "education")):
            item["bucket"] = "anti_fraud_education_safe"
        else:
            item["bucket"] = "hard_benign_safe"
        return item
    return None


def exact_bucket_trim(rows: list[dict], targets: dict, seed: int) -> list[dict]:
    out = []
    for bucket, target in targets.items():
        candidates = [row for row in rows if row["bucket"] == bucket]
        out.extend(sorted(candidates, key=lambda row: stable_hash(f"{seed}:trim:{bucket}:{row['id']}"))[: int(target)])
    return out


def assign_formal_aware_splits(rows: list[dict], config: dict, seed: int) -> dict[str, list[dict]]:
    formal_targets = config["data"]["formal_test_buckets"]
    remaining = list(rows)
    formal = []
    for bucket, target in formal_targets.items():
        rel = [row for row in remaining if row["bucket"] == bucket and row.get("relation_challenge")]
        ordinary = [row for row in remaining if row["bucket"] == bucket and not row.get("relation_challenge")]
        chosen = sorted(rel, key=lambda row: stable_hash(f"{seed}:formal_rel:{row['id']}"))[: min(len(rel), int(target))]
        chosen_ids = {row["id"] for row in chosen}
        need = int(target) - len(chosen)
        chosen.extend(sorted([row for row in ordinary if row["id"] not in chosen_ids], key=lambda row: stable_hash(f"{seed}:formal:{bucket}:{row['id']}"))[:need])
        formal.extend(chosen)
        chosen_ids = {row["id"] for row in chosen}
        remaining = [row for row in remaining if row["id"] not in chosen_ids]
    splits = {"formal_test": [dict(row, split="formal_test") for row in formal]}
    split_targets = {"train": 4800, "model_dev": 800, "threshold_cal": 800}
    for split, target in split_targets.items():
        selected = balanced_take(remaining, int(target), seed, split)
        ids = {row["id"] for row in selected}
        splits[split] = [dict(row, split=split) for row in selected]
        remaining = [row for row in remaining if row["id"] not in ids]
    return splits


def assign_stratified_splits(rows: list[dict], split_targets: dict[str, int], seed: int) -> list[dict]:
    remaining = list(rows)
    out = []
    for split, target in split_targets.items():
        selected = balanced_take(remaining, int(target), seed, split)
        ids = {row["id"] for row in selected}
        out.extend([dict(row, split=split) for row in selected])
        remaining = [row for row in remaining if row["id"] not in ids]
    return out


def balanced_take(rows: list[dict], n: int, seed: int, salt: str) -> list[dict]:
    per_label = n // 2
    out = []
    for label in ("safe", "unsafe"):
        candidates = [row for row in rows if row.get("gold_label") == label]
        out.extend(sorted(candidates, key=lambda row: stable_hash(f"{seed}:{salt}:{label}:{row['id']}"))[:per_label])
    return out


def canonical_cluster(row: dict) -> str:
    if row.get("relation_group_id"):
        return f"relation_{row.get('relation_group_id')}"
    base = row.get("source_prompt_id") or row.get("id") or row.get("user_query")
    source = row.get("source") or row.get("metadata", {}).get("source_dataset") or ""
    original = row.get("metadata", {}).get("original_id") or row.get("metadata", {}).get("fraudr1_raw_id") or row.get("id") or base
    return hashlib.sha256(f"{source}\n{original}\n{row.get('id')}".encode("utf-8")).hexdigest()[:24]


def manifest_audit(rows: list[dict], bucket_targets: dict) -> dict:
    buckets = Counter(row["bucket"] for row in rows)
    labels = Counter(row["gold_label"] for row in rows)
    missing_provenance = sum(1 for row in rows if not row.get("label_provenance"))
    return {
        "rows": len(rows),
        "bucket_counts": dict(buckets),
        "label_counts": dict(labels),
        "missing_label_provenance": missing_provenance,
        "bucket_targets_met": all(buckets.get(bucket, 0) == int(target) for bucket, target in bucket_targets.items()),
        "label_balance": labels.get("safe", 0) == labels.get("unsafe", 0),
    }


def split_audit(splits: dict[str, list[dict]], config: dict) -> dict:
    checks = {}
    for split, target in config["data"]["splits"].items():
        rows = splits.get(split, [])
        labels = Counter(row["gold_label"] for row in rows)
        checks[f"{split}_rows"] = len(rows) == int(target)
        checks[f"{split}_balance"] = labels.get("safe", 0) == labels.get("unsafe", 0) == int(target) // 2
    clusters = {}
    cross = 0
    for split, rows in splits.items():
        for row in rows:
            cluster = row["canonical_prompt_cluster"]
            prior = clusters.get(cluster)
            if prior and prior != split:
                cross += 1
            clusters[cluster] = split
    checks["cross_split_cluster_leakage"] = cross == 0
    checks["missing_label_provenance"] = all(row.get("label_provenance") for rows in splits.values() for row in rows)
    return {"passed": all(checks.values()), "checks": checks, "split_counts": {split: manifest_audit(rows, {}) for split, rows in splits.items()}, "cross_split_cluster_hits": cross}


def write_label_provenance(path: Path, rows: list[dict]) -> None:
    counts = Counter((row.get("source"), row.get("label_provenance"), row.get("gold_label")) for row in rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "label_provenance", "label", "count"])
        writer.writeheader()
        for (source, provenance, label), count in sorted(counts.items()):
            writer.writerow({"source": source, "label_provenance": provenance, "label": label, "count": count})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
