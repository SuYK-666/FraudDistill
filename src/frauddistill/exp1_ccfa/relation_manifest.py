from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from frauddistill.exp1_ccfa.fraud_taxonomy import annotate_risk_type, load_taxonomy
from frauddistill.exp1_ccfa.public_gold import aegis_test_rows
from frauddistill.exp1_ccfa.saferlhf_public import saferlhf_rows
from frauddistill.exp1_ccfa.semantic_components import attach_semantic_components, leakage_audit
from frauddistill.utils.io import write_jsonl
from tqdm import tqdm


FRAUD_FAMILIES = ("phishing", "impersonation", "credential_theft", "fake_job", "romance", "financial_scam")
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.I)
_REFUSAL_RE = re.compile(r"cannot|can't|unable|sorry|refuse|not able|抱歉|不能|无法|拒绝", re.I)


def stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def write_relation_manifests(output_dir: Path, config: dict, taxonomy_path: Path, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(taxonomy_path)
    sources = load_public_sources(config, taxonomy)
    pools = build_relation_pools(sources, seed)
    splits = build_pilot_splits(pools, config, seed)
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    census = relation_census(sources, pools, splits, config)
    (output_dir / "E1_G0_DATA_CENSUS.json").write_text(json.dumps(census, ensure_ascii=False, indent=2), encoding="utf-8")
    write_label_provenance(output_dir / "E1_G0_LABEL_PROVENANCE.csv", sources)
    (output_dir / "E1_G0_COMPONENT_AUDIT.json").write_text(json.dumps(component_audit(splits), ensure_ascii=False, indent=2), encoding="utf-8")
    write_relation_funnel(output_dir / "E1_G0_RELATION_CANDIDATE_FUNNEL.csv", pools)
    (output_dir / "E1_G0_LICENSE_AND_REVISION_LOCK.json").write_text(json.dumps(license_lock(config), ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "E1_SPLIT_COMPONENT_MANIFEST.tsv", component_manifest_rows(splits))
    return census


def load_public_sources(config: dict, taxonomy: dict) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    safe_train = saferlhf_rows("train")
    safe_test = saferlhf_rows("test")
    rows["PKU-SafeRLHF"] = normalize_rows([*safe_train, *safe_test], taxonomy)
    rows["Aegis/Nemotron-V2"] = normalize_rows(aegis_test_rows("data/raw/aegis/test.json"), taxonomy)
    rows["BeaverTails"] = normalize_rows(beavertails_rows(), taxonomy)
    return rows


def beavertails_rows() -> list[dict]:
    from scripts.build_exp1_cpu_g0b_manifests import _beavertails_rows

    loaded: list[dict] = []
    for split, use in (("330k_train", "train"), ("30k_test", "test"), ("330k_test", "test")):
        try:
            loaded.extend(_beavertails_rows(split, use))
        except Exception:
            continue
    return loaded


def normalize_rows(rows: list[dict], taxonomy: dict) -> list[dict]:
    out = []
    for raw in rows:
        label = raw.get("exp1_label") or raw.get("gold_label") or raw.get("pair_fraud_label")
        query = str(raw.get("user_query") or "").strip()
        answer = str(raw.get("target_model_answer") or "").strip()
        if label not in {"safe", "unsafe"} or not query or not answer:
            continue
        row = dict(raw)
        row["exp1_label"] = label
        row["gold_label"] = label
        row["pair_fraud_label"] = label
        row["fraud_family"] = fraud_family(query, answer)
        row = annotate_risk_type(row, taxonomy)
        row["semantic_component_id"] = row.get("semantic_component_id") or f"component_{stable_hash(str(row.get('source_prompt_id')) or query)[:24]}"
        out.append(row)
    return attach_semantic_components(out)


def build_relation_pools(sources: dict[str, list[dict]], seed: int) -> dict[str, list[dict]]:
    safer = [row for row in sources.get("PKU-SafeRLHF", []) if row.get("prompt_risk_domain") == "fraud_core"]
    by_prompt: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"safe": [], "unsafe": []})
    for row in safer:
        by_prompt[str(row.get("source_prompt_id") or row.get("semantic_component_id"))][row["exp1_label"]].append(row)
    r1 = []
    for prompt_id, labels in sorted(by_prompt.items(), key=lambda item: stable_hash(f"{seed}:r1:{item[0]}")):
        if not labels["safe"] or not labels["unsafe"]:
            continue
        group_id = f"r1_{stable_hash(prompt_id)[:20]}"
        for label in ("safe", "unsafe"):
            chosen = sorted(labels[label], key=lambda row: stable_hash(f"{seed}:r1:{label}:{row['id']}"))[0]
            r1.append(as_subset_row(chosen, "R1", group_id))
    natural_pool = [
        row
        for rows in sources.values()
        for row in rows
        if row.get("prompt_risk_domain") == "fraud_core"
        and row.get("label_provenance")
        and "project" not in str(row.get("label_provenance")).lower()
    ]
    r2 = build_r2_groups(natural_pool, seed)
    r3 = select_balanced(natural_pool, seed, "R3", per_label=1500)
    return {"R1": r1, "R2": r2, "R3": r3, "natural_pool": natural_pool}


def as_subset_row(row: dict, subset: str, group_id: str) -> dict:
    item = dict(row)
    item["e1_subset"] = subset
    item["relation_group_id"] = group_id
    item["cluster_id"] = group_id if subset in {"R1", "R2"} else str(item.get("semantic_component_id"))
    return item


def build_r2_groups(rows: list[dict], seed: int) -> list[dict]:
    safe = sorted([r for r in rows if r["exp1_label"] == "safe"], key=lambda r: stable_hash(f"{seed}:r2s:{r['id']}"))
    unsafe = sorted([r for r in rows if r["exp1_label"] == "unsafe"], key=lambda r: stable_hash(f"{seed}:r2u:{r['id']}"))
    by_family_safe: dict[tuple[str, bool], list[tuple[int, dict]]] = defaultdict(list)
    for row in safe:
        by_family_safe[(str(row.get("fraud_family")), refusal_marker(row["target_model_answer"]))].append((len(row["target_model_answer"]), row))
        by_family_safe[("__any__", refusal_marker(row["target_model_answer"]))].append((len(row["target_model_answer"]), row))
    for values in by_family_safe.values():
        values.sort(key=lambda item: (item[0], stable_hash(str(item[1].get("id")))))
    selected = []
    used = set()
    group_index = 0
    for unsafe_row in tqdm(unsafe, desc="build R2 y-hard groups", leave=False):
        if unsafe_row["semantic_component_id"] in used:
            continue
        safe_row = find_r2_safe_match(unsafe_row, by_family_safe, used)
        if safe_row is None:
            continue
        group_id = f"r2_{group_index:04d}"
        selected.extend([as_subset_row(unsafe_row, "R2", group_id), as_subset_row(safe_row, "R2", group_id)])
        used.add(unsafe_row["semantic_component_id"])
        used.add(safe_row["semantic_component_id"])
        group_index += 1
        if group_index >= 1500:
            break
    return selected


def find_r2_safe_match(unsafe_row: dict, by_family_safe: dict[tuple[str, bool], list[tuple[int, dict]]], used: set[str]) -> dict | None:
    unsafe_len = max(len(unsafe_row["target_model_answer"]), 1)
    low = int(unsafe_len * 0.75)
    high = int(unsafe_len * 1.33) + 1
    target_len = unsafe_len
    refusal = refusal_marker(unsafe_row["target_model_answer"])
    keys = [(str(unsafe_row.get("fraud_family")), not refusal), ("__any__", not refusal), (str(unsafe_row.get("fraud_family")), refusal)]
    for key in keys:
        values = by_family_safe.get(key, [])
        if not values:
            continue
        lengths = [item[0] for item in values]
        start = max(0, bisect_left(lengths, target_len) - 50)
        stop = min(len(values), bisect_left(lengths, target_len) + 50)
        best = None
        best_gap = None
        for length, row in values[start:stop]:
            if length < low or length > high or row["semantic_component_id"] in used:
                continue
            gap = abs(length - target_len)
            if best is None or gap < best_gap:
                best = row
                best_gap = gap
        if best is not None:
            return best
    return None


def select_balanced(rows: list[dict], seed: int, subset: str, per_label: int) -> list[dict]:
    selected = []
    used = set()
    for label in ("safe", "unsafe"):
        candidates = sorted([row for row in rows if row["exp1_label"] == label], key=lambda row: stable_hash(f"{seed}:{subset}:{label}:{row['id']}"))
        count = 0
        for row in candidates:
            component = str(row.get("semantic_component_id"))
            if component in used:
                continue
            selected.append(as_subset_row(row, subset, component))
            used.add(component)
            count += 1
            if count >= per_label:
                break
    return selected


def build_pilot_splits(pools: dict[str, list[dict]], config: dict, seed: int) -> dict[str, list[dict]]:
    pilot_cfg = config["data"]["pilot"]
    blocked = set()
    pilot = []
    pilot.extend(take_grouped(pools["R1"], int(pilot_cfg["pilot_rows_per_subset"]), blocked, seed, "pilot:R1", "relation_group_id"))
    pilot.extend(take_grouped(pools["R2"], int(pilot_cfg["pilot_rows_per_subset"]), blocked, seed, "pilot:R2", "relation_group_id"))
    pilot.extend(take_unique(pools["R3"], int(pilot_cfg["pilot_rows_per_subset"]), blocked, seed, "pilot:R3"))
    train_pool = [*pools["natural_pool"], *pools["R1"], *pools["R2"], *pools["R3"]]
    train = take_balanced_from_pool(train_pool, int(pilot_cfg["train_rows"]), blocked, seed, "train")
    model_dev = take_balanced_from_pool(train_pool, int(pilot_cfg["model_dev_rows"]), blocked, seed + 1, "model_dev")
    calibration = take_balanced_from_pool(train_pool, int(pilot_cfg["calibration_dev_rows"]), blocked, seed + 2, "calibration_dev")
    return {"train": train, "model_dev": model_dev, "calibration_dev": calibration, "pilot_test": pilot}


def unique_component_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        component = str(row.get("semantic_component_id"))
        if component in seen:
            continue
        seen.add(component)
        out.append(row)
    return out


def take_grouped(rows: list[dict], n_rows: int, blocked: set[str], seed: int, salt: str, group_field: str) -> list[dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get(group_field) or row.get("semantic_component_id"))].append(row)
    out = []
    for group_id, group_rows in sorted(by_group.items(), key=lambda item: stable_hash(f"{seed}:{salt}:{item[0]}")):
        components = {str(row.get("semantic_component_id")) for row in group_rows}
        if components & blocked:
            continue
        out.extend(group_rows)
        blocked.update(components)
        if len(out) >= n_rows:
            break
    return out[:n_rows]


def take_unique(rows: list[dict], n: int, blocked: set[str], seed: int, salt: str) -> list[dict]:
    out = []
    for row in sorted(rows, key=lambda item: stable_hash(f"{seed}:{salt}:{item['id']}")):
        component = str(row.get("semantic_component_id"))
        if component in blocked:
            continue
        out.append(row)
        blocked.add(component)
        if len(out) >= n:
            break
    return out


def take_balanced_from_pool(rows: list[dict], n: int, blocked: set[str], seed: int, split: str) -> list[dict]:
    out = []
    local_components: set[str] = set()
    per_label = n // 2
    for label in ("safe", "unsafe"):
        count = 0
        candidates = sorted([row for row in rows if row["exp1_label"] == label], key=lambda item: stable_hash(f"{seed}:{split}:{item['id']}"))
        for row in candidates:
            component = str(row.get("semantic_component_id"))
            if component in blocked and component not in local_components:
                continue
            item = dict(row)
            item["e1_split"] = split
            out.append(item)
            local_components.add(component)
            count += 1
            if count >= per_label:
                break
    blocked.update(local_components)
    return out


def relation_census(sources: dict[str, list[dict]], pools: dict[str, list[dict]], splits: dict[str, list[dict]], config: dict) -> dict:
    source_counts = {name: label_counts(rows) for name, rows in sources.items()}
    r1_pairs = len(pools["R1"]) // 2
    r2_groups = len({row["relation_group_id"] for row in pools["R2"]})
    r3_rows = len(pools["R3"])
    fallback_count = sum(1 for rows in sources.values() for row in rows if row.get("metadata", {}).get("p3_label_source") == "prompt_label_fallback")
    empty_response = sum(1 for rows in sources.values() for row in rows if not str(row.get("target_model_answer") or "").strip())
    leakage = leakage_audit(splits)
    checks = {
        "R1_min_pairs": r1_pairs >= int(config["gates"]["g0"]["r1_min_pairs"]),
        "R2_min_groups": r2_groups >= int(config["gates"]["g0"]["r2_min_groups"]),
        "R3_min_rows": r3_rows >= int(config["gates"]["g0"]["r3_min_rows"]),
        "prompt_label_fallback_zero": fallback_count == 0,
        "empty_response_zero": empty_response == 0,
        "cross_split_component_overlap_zero": leakage["passed"],
        "pilot_train_available": len(splits["train"]) >= int(config["data"]["pilot"]["train_rows"]),
        "pilot_test_available": len(splits["pilot_test"]) >= int(config["data"]["pilot"]["pilot_rows_per_subset"]) * 3,
    }
    return {
        "protocol": config["experiment"]["protocol"],
        "source_counts": source_counts,
        "r1_pairs": r1_pairs,
        "r2_groups": r2_groups,
        "r3_rows": r3_rows,
        "prompt_label_fallback": fallback_count,
        "empty_response": empty_response,
        "split_counts": {name: label_counts(rows) for name, rows in splits.items()},
        "fraud_family_counts": dict(Counter(str(row.get("fraud_family")) for rows in sources.values() for row in rows)),
        "leakage": leakage,
        "checks": checks,
        "passed": all(checks.values()),
    }


def label_counts(rows: list[dict]) -> dict:
    labels = Counter(str(row.get("exp1_label") or row.get("gold_label")) for row in rows)
    return {"rows": len(rows), "safe": labels.get("safe", 0), "unsafe": labels.get("unsafe", 0), "components": len({row.get("semantic_component_id") for row in rows})}


def component_audit(splits: dict[str, list[dict]]) -> dict:
    leakage = leakage_audit(splits)
    return {"passed": leakage["passed"], "leakage": leakage}


def component_manifest_rows(splits: dict[str, list[dict]]) -> list[dict]:
    return [
        {"split": split, "id": row.get("id"), "semantic_component_id": row.get("semantic_component_id"), "source_prompt_id": row.get("source_prompt_id"), "subset": row.get("e1_subset")}
        for split, rows in splits.items()
        for row in rows
    ]


def write_label_provenance(path: Path, sources: dict[str, list[dict]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "label_provenance", "label", "count"])
        writer.writeheader()
        counts = Counter((source, str(row.get("label_provenance")), str(row.get("exp1_label"))) for source, rows in sources.items() for row in rows)
        for (source, provenance, label), count in sorted(counts.items()):
            writer.writerow({"source": source, "label_provenance": provenance, "label": label, "count": count})


def write_relation_funnel(path: Path, pools: dict[str, list[dict]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "rows", "groups"])
        writer.writeheader()
        writer.writerow({"stage": "R1_same_q_paired", "rows": len(pools["R1"]), "groups": len(pools["R1"]) // 2})
        writer.writerow({"stage": "R2_y_hard_matched", "rows": len(pools["R2"]), "groups": len({row["relation_group_id"] for row in pools["R2"]})})
        writer.writerow({"stage": "R3_natural_fraud_qy", "rows": len(pools["R3"]), "groups": len({row["semantic_component_id"] for row in pools["R3"]})})


def license_lock(config: dict) -> dict:
    return {"required_sources": config["data"]["required_sources"], "optional_sources": config["data"].get("optional_sources", []), "locked_by": config["experiment"]["protocol"]}


def fraud_family(query: str, answer: str) -> str:
    text = f"{query} {answer}".lower()
    if any(term in text for term in ("phishing", "钓鱼")):
        return "phishing"
    if any(term in text for term in ("impersonat", "pretend", "冒充", "伪装")):
        return "impersonation"
    if any(term in text for term in ("password", "credential", "otp", "验证码", "密码", "凭证")):
        return "credential_theft"
    if any(term in text for term in ("job", "recruit", "招聘", "兼职")):
        return "fake_job"
    if any(term in text for term in ("romance", "dating", "relationship", "网恋", "交友")):
        return "romance"
    return "financial_scam"


def refusal_marker(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text))
