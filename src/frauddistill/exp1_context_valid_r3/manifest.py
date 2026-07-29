from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from frauddistill.utils.io import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[3]


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed: int, *parts: Any) -> str:
    return sha_text(":".join([str(seed), *[str(p) for p in parts]]))


def build_g0_anchor(config: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["data"]["seed"])
    relation_path = ROOT / config["data"]["relation_groups_file"]
    panel_a = select_panel_a(config, seed)
    used_qy = {sha_text(str(row["user_query"]) + "\n" + str(row["target_model_answer"])) for row in panel_a}
    panel_b = select_panel_b(config, relation_path, seed, used_qy)
    panel_c = load_or_generate_panel_c_placeholder(config, seed)
    rows = panel_a + panel_b + panel_c
    rows = sorted(rows, key=lambda row: stable_key(seed, "anchor", row["id"]))
    write_jsonl(output_dir / "anchor480.jsonl", rows)
    model_dev = select_model_dev(config, rows, seed)
    write_jsonl(output_dir / "model_dev_pool.jsonl", model_dev)

    source_lock = {
        "protocol": config["experiment"]["protocol"],
        "relation_groups_file": str(relation_path),
        "relation_groups_sha256": sha_file(relation_path) if relation_path.exists() else None,
        "panel_a_sources": config["data"]["panel_a_sources"],
        "panel_c_target_source": config["data"]["fraudr1_raw_files"],
    }
    audits = {
        "source_lock": source_lock,
        "panel_census": panel_census(rows),
        "construct_audit": construct_audit(rows),
        "duplicate_audit": duplicate_audit(rows),
        "context_probe_audit": context_probe_audit(panel_b),
        "fraudr1_hash_leakage_audit": fraudr1_hash_leakage_audit(rows),
        "protocol_lock": protocol_lock(config, output_dir / "anchor480.jsonl", output_dir / "model_dev_pool.jsonl"),
    }
    write_json(output_dir / "E1_R3_SOURCE_LOCK.json", audits["source_lock"])
    write_panel_census(output_dir / "E1_R3_PANEL_CENSUS.csv", rows)
    write_label_provenance(output_dir / "E1_R3_LABEL_PROVENANCE.csv", rows)
    write_json(output_dir / "E1_R3_CONSTRUCT_AUDIT.json", audits["construct_audit"])
    write_json(output_dir / "E1_R3_DUPLICATE_AUDIT.json", audits["duplicate_audit"])
    write_json(output_dir / "E1_R3_CONTEXT_PROBE_AUDIT.json", audits["context_probe_audit"])
    write_json(output_dir / "E1_R3_FRAUDR1_HASH_LEAKAGE_AUDIT.json", audits["fraudr1_hash_leakage_audit"])
    write_json(output_dir / "E1_R3_PROTOCOL_LOCK.json", audits["protocol_lock"])
    passed = (
        len(rows) == 480
        and Counter(row["gold_label"] for row in rows) == Counter({"safe": 240, "unsafe": 240})
        and audits["construct_audit"]["passed"]
        and audits["duplicate_audit"]["passed"]
        and audits["context_probe_audit"]["passed"]
        and audits["fraudr1_hash_leakage_audit"]["passed"]
    )
    return {"passed": passed, "rows": len(rows), **audits}


def select_panel_a(config: dict, seed: int, exclude_qy: set[str] | None = None) -> list[dict]:
    targets = config["data"]["panel_a_sources"]
    label_targets = config["data"].get("panel_a_source_label_targets", {})
    source_files = config["data"]["public_files"]
    out: list[dict] = []
    for source, target in targets.items():
        rows = []
        for path_text in source_files.get(source, []):
            path = ROOT / path_text
            if not path.exists():
                continue
            for raw in read_jsonl(path):
                raw_source = str(raw.get("source") or raw.get("metadata", {}).get("source_dataset") or "")
                if source not in raw_source and raw_source not in source:
                    continue
                if provider_block_risk(raw):
                    continue
                row = normalize_public(raw, source, "natural_public")
                if row and sha_text(str(row["user_query"]) + "\n" + str(row["target_model_answer"])) not in (exclude_qy or set()):
                    rows.append(row)
        quotas = label_targets.get(source)
        if quotas:
            out.extend(take_by_label_quota(rows, {"safe": int(quotas["safe"]), "unsafe": int(quotas["unsafe"])}, seed, f"panel_a:{source}"))
        else:
            out.extend(take_balanced(rows, int(target), seed, f"panel_a:{source}"))
    return out


def select_panel_b(config: dict, relation_path: Path, seed: int, exclude_qy: set[str] | None = None) -> list[dict]:
    if not relation_path.exists():
        return []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for raw in read_jsonl(relation_path):
        gid = str(raw.get("relation_group_id") or raw.get("context_collision_group_id") or "")
        row = normalize_public(raw, str(raw.get("source") or "relation_public"), "context_critical_public")
        if not gid or not row:
            continue
        row["relation_group_id"] = gid
        row["canonical_prompt_cluster"] = f"relation_{gid}"
        row["target_model_answer_origin"] = "public_response"
        grouped[gid].append(row)
    groups = []
    for gid, values in grouped.items():
        labels = Counter(row["gold_label"] for row in values)
        if labels.get("safe", 0) >= 1 and labels.get("unsafe", 0) >= 1:
            safe = sorted([r for r in values if r["gold_label"] == "safe"], key=lambda r: stable_key(seed, gid, r["id"]))[0]
            unsafe = sorted([r for r in values if r["gold_label"] == "unsafe"], key=lambda r: stable_key(seed, gid, r["id"]))[0]
            groups.append([safe, unsafe])
    chosen = []
    used_qy: set[str] = set(exclude_qy or set())
    for group in sorted(groups, key=lambda group: stable_key(seed, "panel_b", group[0]["relation_group_id"])):
        hashes = {sha_text(str(row["user_query"]) + "\n" + str(row["target_model_answer"])) for row in group}
        if hashes & used_qy:
            continue
        chosen.append(group)
        used_qy.update(hashes)
        if len(chosen) >= 45:
            break
    return [row for group in chosen for row in group]


def load_or_generate_panel_c_placeholder(config: dict, seed: int) -> list[dict]:
    path = ROOT / config["data"]["panel_c_file"]
    if path.exists():
        return [normalize_panel_c(row) for row in read_jsonl(path)]
    return []


def normalize_public(raw: dict, source: str, panel: str) -> dict | None:
    label = str(raw.get("gold_label") or raw.get("exp1_label") or raw.get("pair_fraud_label") or "").lower()
    if label not in {"safe", "unsafe"}:
        return None
    q = str(raw.get("user_query") or raw.get("prompt") or "")
    y = str(raw.get("target_model_answer") or raw.get("response") or "")
    if not q.strip() or not y.strip() or y.strip().lower() in {"none", "null", "redacted"}:
        return None
    meta = dict(raw.get("metadata") or {})
    row_id = str(raw.get("id") or raw.get("row_uid") or sha_text(q + "\n" + y)[:20])
    qy_hash = sha_text(q + "\n" + y)[:16]
    return {
        "id": f"r3_{panel}_{source}_{row_id}_{qy_hash}",
        "source": source,
        "panel": panel,
        "language": raw.get("language") or meta.get("language") or "unknown",
        "user_query": q,
        "target_model_answer": y,
        "gold_label": label,
        "gold_risk_type": raw.get("gold_risk_type") or ("safe" if label == "safe" else "fraud_assistance"),
        "label_provenance": raw.get("label_provenance") or meta.get("response_label_source") or "external_public_response_level_label",
        "target_model_answer_origin": "public_response",
        "canonical_prompt_cluster": raw.get("canonical_prompt_cluster") or raw.get("semantic_component_id") or sha_text(q.lower().strip())[:24],
        "metadata": meta | {"source_original_id": row_id},
    }


def provider_block_risk(raw: dict) -> bool:
    meta = raw.get("metadata") or {}
    categories = str(meta.get("violated_categories") or raw.get("violated_categories") or "").lower()
    text = f"{raw.get('user_query','')} {raw.get('target_model_answer','')}".lower()
    if any(term in categories for term in ("sexual", "profanity", "self-harm")):
        return True
    if any(term in text for term in ("sexual abuse", "child sexual", "explicit sexual")):
        return True
    return False


def normalize_panel_c(raw: dict) -> dict:
    row = dict(raw)
    row["panel"] = "fraudr1_target_response"
    row["source"] = "Fraud-R1"
    row["target_model_answer_origin"] = "api_target_response"
    row["canonical_prompt_cluster"] = row.get("canonical_prompt_cluster") or sha_text(str(row.get("user_query", "")).lower().strip())[:24]
    return row


def select_model_dev(config: dict, anchor_rows: list[dict], seed: int) -> list[dict]:
    used = {row["id"] for row in anchor_rows}
    pool = []
    for source in ("Aegis",):
        for path_text in config["data"]["public_files"].get(source, []):
            path = ROOT / path_text
            if path.exists():
                pool.extend(row for raw in read_jsonl(path) if (row := normalize_public(raw, source, "model_dev_public")))
    pool = [row for row in pool if row["id"] not in used]
    return take_balanced(pool, 120, seed, "model_dev")


def take_balanced(rows: list[dict], n: int, seed: int, salt: str) -> list[dict]:
    return take_by_label_quota(rows, {"safe": n // 2, "unsafe": n // 2}, seed, salt)


def take_by_label_quota(rows: list[dict], quotas: dict[str, int], seed: int, salt: str) -> list[dict]:
    out = []
    used_qy: set[str] = set()
    for label, target in quotas.items():
        choices = []
        for row in sorted([row for row in rows if row["gold_label"] == label], key=lambda row: stable_key(seed, salt, label, row["id"])):
            qy = sha_text(str(row["user_query"]) + "\n" + str(row["target_model_answer"]))
            if qy in used_qy:
                continue
            choices.append(row)
            used_qy.add(qy)
            if len(choices) >= target:
                break
        out.extend(choices)
    return out


def panel_census(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "label_counts": dict(Counter(row["gold_label"] for row in rows)),
        "panel_counts": {k: dict(v) for k, v in nested_counts(rows, "panel", "gold_label").items()},
        "source_counts": {k: dict(v) for k, v in nested_counts(rows, "source", "gold_label").items()},
    }


def nested_counts(rows: list[dict], outer: str, inner: str) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[str(row.get(outer))][str(row.get(inner))] += 1
    return counts


def construct_audit(rows: list[dict]) -> dict:
    allowed = {"public_response", "api_target_response"}
    checks = {
        "rows_480": len(rows) == 480,
        "balanced_labels": Counter(row["gold_label"] for row in rows) == Counter({"safe": 240, "unsafe": 240}),
        "allowed_answer_origin": all(row.get("target_model_answer_origin") in allowed for row in rows),
        "no_legacy_buckets": all(str(row.get("bucket", "")) not in {"qwen_generated_safe_control", "hard_unsafe_qwen_fn_or_phishing"} for row in rows),
        "panel_a_rows": sum(row.get("panel") == "natural_public" for row in rows) == 294,
        "panel_b_rows": sum(row.get("panel") == "context_critical_public" for row in rows) == 90,
        "panel_c_rows": sum(row.get("panel") == "fraudr1_target_response" for row in rows) == 96,
    }
    return {"passed": all(checks.values()), "checks": checks}


def duplicate_audit(rows: list[dict]) -> dict:
    ids = Counter(row["id"] for row in rows)
    qy = Counter(sha_text(str(row["user_query"]) + "\n" + str(row["target_model_answer"])) for row in rows)
    checks = {
        "unique_ids": all(v == 1 for v in ids.values()),
        "unique_qy_hash": all(v == 1 for v in qy.values()),
    }
    return {"passed": all(checks.values()), "checks": checks, "duplicate_ids": [k for k, v in ids.items() if v > 1][:20], "duplicate_qy": [k for k, v in qy.items() if v > 1][:20]}


def context_probe_audit(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("relation_group_id"))].append(row)
    labels_ok = all(Counter(row["gold_label"] for row in vals) == Counter({"safe": 1, "unsafe": 1}) for vals in groups.values())
    checks = {"groups_45": len(groups) == 45, "paired_safe_unsafe": labels_ok}
    return {"passed": all(checks.values()), "checks": checks, "group_count": len(groups)}


def fraudr1_hash_leakage_audit(rows: list[dict]) -> dict:
    hits = []
    generated_hashes = set()
    for path_text in [
        "data/raw/fraudr1/repo/dataset/FP-base-full/FP-base-English.json",
        "data/raw/fraudr1/repo/dataset/FP-base-full/FP-base-Chinese.json",
    ]:
        path = ROOT / path_text
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        for item in data if isinstance(data, list) else []:
            generated_hashes.add(sha_text(str(item.get("generated text") or "").strip()))
    for row in rows:
        if row.get("panel") == "fraudr1_target_response" and sha_text(str(row.get("target_model_answer") or "").strip()) in generated_hashes:
            hits.append(row["id"])
    return {"passed": not hits, "hits": hits[:20], "fraudr1_generated_hash_count": len(generated_hashes)}


def protocol_lock(config: dict, anchor_file: Path, dev_file: Path) -> dict:
    return {
        "protocol": config["experiment"]["protocol"],
        "anchor_sha256": sha_file(anchor_file) if anchor_file.exists() else None,
        "model_dev_sha256": sha_file(dev_file) if dev_file.exists() else None,
        "seed": config["data"]["seed"],
    }


def write_panel_census(path: Path, rows: list[dict]) -> None:
    write_csv(path, [
        {"panel": panel, "label": label, "count": count}
        for panel, counter in nested_counts(rows, "panel", "gold_label").items()
        for label, count in counter.items()
    ])


def write_label_provenance(path: Path, rows: list[dict]) -> None:
    counts = Counter((row.get("panel"), row.get("source"), row.get("label_provenance"), row.get("gold_label"), row.get("target_model_answer_origin")) for row in rows)
    write_csv(path, [
        {"panel": p, "source": s, "label_provenance": lp, "label": l, "answer_origin": o, "count": c}
        for (p, s, lp, l, o), c in sorted(counts.items())
    ])


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
