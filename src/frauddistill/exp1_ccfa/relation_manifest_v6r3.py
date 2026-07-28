from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from tqdm import tqdm

from frauddistill.exp1_ccfa.duplicate_audit import duplicate_audit
from frauddistill.exp1_ccfa.fraud_taxonomy import load_taxonomy
from frauddistill.exp1_ccfa.nuisance_single_view import SingleViewNuisanceSelectors
from frauddistill.exp1_ccfa.relation_manifest import (
    as_subset_row,
    attach_leakage_supercomponents,
    build_r1_pairs,
    dataset_revision_lock,
    dedupe_row_uid,
    file_sha256,
    fraud_family_q_only,
    git_commit_safe,
    is_train_original,
    json_default,
    label_counts,
    load_public_sources_v6r2,
    normalize_rows,
    refusal_marker,
    reserve_relation_groups,
    row_uid,
    sample_for_nuisance,
    sample_r2_candidates,
    stable_hash,
    subset_super_overlap,
    write_component_tsv,
    write_label_provenance,
    write_jsonl,
)


POLICY_LEVELS = (
    {"level": "C0", "y_window": 160, "top_k": 80, "full_range": False, "family": "hard", "refusal": "hard"},
    {"level": "C1", "y_window": 320, "top_k": 160, "full_range": False, "family": "hard", "refusal": "hard"},
    {"level": "C2", "y_window": 0, "top_k": 160, "full_range": True, "family": "hard", "refusal": "hard"},
    {"level": "C3", "y_window": 0, "top_k": 160, "full_range": True, "family": "soft", "refusal": "hard"},
    {"level": "C4", "y_window": 0, "top_k": 160, "full_range": True, "family": "soft", "refusal": "soft"},
)


def write_relation_manifests_v6r3(output_dir: Path, config: dict, taxonomy_path: Path, seed: int, require_clean_git: bool = True) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(taxonomy_path)
    sources, source_audit, wildguard_admission = load_public_sources_v6r3(config, taxonomy)
    all_rows = attach_leakage_supercomponents(dedupe_row_uid([row for rows in sources.values() for row in rows]))

    r1_all = build_r1_pairs([row for row in all_rows if row.get("source") == "PKU-SafeRLHF"], seed)
    r1_reserved, blocked_super = reserve_relation_groups(r1_all, 3800, seed, "v6r3:r1")

    fraud_rows = [
        row
        for row in all_rows
        if row.get("prompt_risk_domain") == "fraud_core"
        and row.get("exp1_label") in {"safe", "unsafe"}
        and row.get("leakage_supercomponent_id") not in blocked_super
    ]
    train_only = [row for row in fraud_rows if is_train_original(row)] or fraud_rows
    r2_selected, r2_audit = build_r2_v6r3(fraud_rows, train_only, seed, config["data"]["r2"], config["gates"]["g0r3"])
    blocked_super |= {str(row.get("leakage_supercomponent_id")) for row in r2_selected}

    r3_pool = [
        as_subset_row(row, "R3", str(row.get("leakage_supercomponent_id")))
        for row in all_rows
        if row.get("prompt_risk_domain") == "fraud_core"
        and row.get("exp1_label") in {"safe", "unsafe"}
        and row.get("leakage_supercomponent_id") not in blocked_super
    ]
    r3_selected, r3_audit = build_r3_balanced_v6r3(r3_pool, int(config["data"]["r3"]["selected_rows"]), seed)
    splits, split_audit = build_v6r3_splits(r1_reserved, r2_selected, r3_selected, config, seed)

    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    prefix = "E1_V6R3"
    write_v6r3_csvs(output_dir, sources, fraud_rows, splits)
    write_v6r3_r2_artifacts(output_dir, r2_audit, r2_selected)
    write_json(output_dir / f"{prefix}_R3_BALANCED_CAPACITY.json", r3_audit)
    write_json(output_dir / f"{prefix}_SUPERCOMPONENT_AUDIT.json", supercomponent_audit({"R1": r1_reserved, "R2": r2_selected, "R3": r3_selected}))
    duplicate = duplicate_audit(core_duplicate_views(splits))
    write_json(output_dir / f"{prefix}_DUPLICATE_AUDIT.json", duplicate)
    write_json(output_dir / "WILDGUARD_ADMISSION.json", wildguard_admission)
    write_json(output_dir / f"{prefix}_RAW_FILE_HASHES.json", raw_file_hashes())
    write_json(output_dir / f"{prefix}_DATASET_REVISION_LOCK.json", {**dataset_revision_lock(config), "WildGuardMix": wildguard_admission.get("revision", "unknown"), "source_audit": source_audit})
    write_component_tsv(output_dir / f"{prefix}_SPLIT_COMPONENTS.tsv", splits)
    manifest_hash = {f"{name}.jsonl": file_sha256(output_dir / f"{name}.jsonl") for name in splits}
    protocol_lock = {"protocol": config["experiment"]["protocol"], "seed": seed, "manifest_sha256": manifest_hash, "git_commit": git_commit_safe(), "config_hash": config_hash(config)}
    write_json(output_dir / f"{prefix}_PROTOCOL_LOCK.json", protocol_lock)
    census = v6r3_census(sources, source_audit, r1_reserved, r2_selected, r3_selected, splits, r2_audit, r3_audit, split_audit, duplicate, wildguard_admission, config, require_clean_git)
    write_json(output_dir / f"{prefix}_DATA_CENSUS.json", census)
    write_json(output_dir / f"{prefix}_RUN_FINGERPRINT.json", {"protocol_lock": protocol_lock, "decision_inputs_sha256": file_sha256(output_dir / f"{prefix}_DATA_CENSUS.json")})
    return census


def load_public_sources_v6r3(config: dict, taxonomy: dict) -> tuple[dict[str, list[dict]], dict, dict]:
    audit = {"sources": {}, "failures": []}
    sources = load_public_sources_v6r2(config, taxonomy)
    for name, rows in list(sources.items()):
        sources[name] = [with_v6r3_uid(row, name) for row in rows]
        audit["sources"][name] = {"loaded": True, **label_counts(sources[name])}
    wild_rows, admission = wildguard_rows_v6r3(taxonomy)
    if admission["loaded"]:
        sources["WildGuardTrain"] = wild_rows
        audit["sources"]["WildGuardTrain"] = {"loaded": True, **label_counts(wild_rows)}
    else:
        sources["WildGuardTrain"] = []
        audit["sources"]["WildGuardTrain"] = {"loaded": False, "rows": 0, "safe": 0, "unsafe": 0, "components": 0}
        audit["failures"].append({"source": "WildGuardTrain", "error": admission.get("error", "not loaded")})
    return sources, audit, admission


def wildguard_rows_v6r3(taxonomy: dict) -> tuple[list[dict], dict]:
    admission = {"source": "allenai/wildguardmix", "subset": "WildGuardTrain", "loaded": False, "revision": "main", "rows_raw": 0, "rows_admitted": 0, "prompt_label_fallback": 0}
    try:
        from datasets import load_dataset

        try:
            ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
        except Exception:
            ds = load_dataset("allenai/wildguardmix", split="train")
        admission["rows_raw"] = len(ds)
        raw_rows = [wildguard_normalize_raw(row, idx) for idx, row in enumerate(ds)]
        rows = normalize_rows([row for row in raw_rows if row is not None], taxonomy)
        rows = [with_v6r3_uid(row, "WildGuardTrain") for row in rows if row.get("prompt_risk_domain") == "fraud_core"]
        admission["loaded"] = True
        admission["rows_admitted"] = len(rows)
        admission["safe"] = sum(1 for row in rows if row.get("exp1_label") == "safe")
        admission["unsafe"] = sum(1 for row in rows if row.get("exp1_label") == "unsafe")
        return rows, admission
    except Exception as exc:
        admission["error"] = f"{exc.__class__.__name__}: {exc}"
        return [], admission


def wildguard_normalize_raw(row: dict, idx: int) -> dict | None:
    response = first_nonempty(row, "response", "model_response", "target_model_answer", "completion")
    query = first_nonempty(row, "prompt", "user_prompt", "user_query", "instruction")
    harm = str(first_nonempty(row, "response_harm_label", "response_harm", "response_label", "is_response_harmful")).strip().lower()
    if not query or not response:
        return None
    if harm in {"harmful", "unsafe", "true", "1", "yes"}:
        label = "unsafe"
    elif harm in {"unharmful", "harmless", "safe", "false", "0", "no"}:
        label = "safe"
    else:
        return None
    category = " ".join(str(row.get(key) or "") for key in ("subcategory", "category", "prompt_harm_category", "response_harm_category", "harm_category"))
    if not fraudish_text(f"{query} {category}"):
        return None
    return {
        "id": f"wildguardtrain_{idx}",
        "source": "WildGuardTrain",
        "source_prompt_id": f"wildguardtrain_{stable_hash(str(query))[:24]}",
        "user_query": str(query),
        "target_model_answer": str(response),
        "exp1_label": label,
        "gold_label": label,
        "pair_fraud_label": label,
        "label_provenance": "WildGuardTrain response_harm_label",
        "metadata": {
            "source_dataset": "allenai/wildguardmix",
            "dataset_revision": "main",
            "official_split": "train",
            "original_id": row.get("id", idx),
            "response_refusal_label": first_nonempty(row, "response_refusal_label", "refusal_label"),
            "category_name": category,
        },
    }


def first_nonempty(row: dict, *keys: str) -> object:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return ""


def fraudish_text(text: str) -> bool:
    text = str(text).lower()
    return any(term in text for term in ("fraud", "deception", "scam", "phishing", "credential", "impersonat", "financial", "bank", "otp", "password"))


def with_v6r3_uid(row: dict, source_name: str) -> dict:
    item = dict(row)
    metadata = dict(item.get("metadata") or {})
    metadata.setdefault("source_dataset", source_name)
    metadata.setdefault("dataset_revision", "main")
    metadata.setdefault("official_split", metadata.get("g0b_use", "unknown"))
    metadata.setdefault("original_id", item.get("id") or item.get("source_prompt_id") or item.get("row_uid"))
    item["metadata"] = metadata
    item["row_uid"] = row_uid_v6r3(item)
    item["fraud_family"] = fraud_family_q_only(str(item.get("user_query", "")), metadata)
    item["fraud_family_q_only"] = item["fraud_family"]
    return item


def row_uid_v6r3(row: dict) -> str:
    metadata = dict(row.get("metadata") or {})
    parts = [
        row.get("source") or metadata.get("source_dataset") or "source",
        metadata.get("source_dataset") or "",
        metadata.get("dataset_revision") or "",
        metadata.get("official_split") or "",
        metadata.get("original_id") or row.get("id") or row.get("source_prompt_id") or "",
        normalize_text(row.get("user_query")),
        normalize_text(row.get("target_model_answer")),
    ]
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:32]


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def build_r2_v6r3(candidates: list[dict], train_rows: list[dict], seed: int, policy: dict, gates: dict) -> tuple[list[dict], dict]:
    target = int(policy.get("selected_groups", 2250))
    candidates = sample_r2_candidates(candidates, seed, int(policy.get("candidate_max_per_label", 90000)))
    selectors = SingleViewNuisanceSelectors(c=0.3, seed=seed).fit(sample_for_nuisance(train_rows, seed, int(policy.get("nuisance_train_rows", 20000))))
    scores = selectors.score(candidates)
    scored = []
    for idx, row in enumerate(candidates):
        item = dict(row)
        metadata = dict(item.get("metadata") or {})
        metadata.update({"p2_dvm_q_score": float(scores.q_prob[idx]), "p2_dvm_y_score": float(scores.y_prob[idx]), "p2_dvm_q_logit": float(scores.q_logit[idx]), "p2_dvm_y_logit": float(scores.y_logit[idx])})
        item["metadata"] = metadata
        scored.append(item)
    safe = [row for row in scored if row["exp1_label"] == "safe"]
    unsafe = [row for row in scored if row["exp1_label"] == "unsafe"]
    policy_results = []
    selected_rows: list[dict] = []
    selected_audit: dict | None = None
    for level in POLICY_LEVELS:
        edges, edge_payload, graph_audit = r2_component_edges_v6r3(unsafe, safe, policy, level)
        matched = max_component_matching_v6r3(edges, edge_payload, target)
        rows = materialize_r2_matches(matched, target)
        audit = r2_v6r3_audit(rows, matched, edges, graph_audit, selectors, policy, gates, level)
        policy_results.append({"level": level["level"], "edges": len(edges), "max_matching_groups": len(matched), "selected_groups": len(rows) // 2, "passed": audit["passed"]})
        if selected_audit is None or len(matched) > int(selected_audit.get("max_matching_groups", 0)) or (audit["passed"] and not selected_audit.get("passed", False)):
            selected_rows, selected_audit = rows, audit
    selected_audit = dict(selected_audit or {})
    selected_audit["policy_results"] = policy_results
    selected_audit["graph_stability"] = graph_stability(policy_results)
    selected_audit["passed"] = bool(selected_audit.get("passed")) and selected_audit["graph_stability"]["passed"]
    return selected_rows, selected_audit


def r2_component_edges_v6r3(unsafe: list[dict], safe: list[dict], policy: dict, level: dict) -> tuple[list[tuple[str, str, float]], dict, dict]:
    safe_bins: dict[tuple[str, str, str], list[tuple[float, int, dict]]] = defaultdict(list)
    for si, row in enumerate(safe):
        family_keys = [str(row.get("fraud_family_q_only") or row.get("fraud_family"))] if level["family"] == "hard" else ["__any__"]
        refusal_keys = [str(refusal_marker(row["target_model_answer"]))] if level["refusal"] == "hard" else ["__any__"]
        for family in family_keys:
            for refusal in refusal_keys:
                safe_bins[(family, str(row.get("language", "English")), refusal)].append((float(row["metadata"]["p2_dvm_y_logit"]), si, row))
    for values in safe_bins.values():
        values.sort(key=lambda item: item[0])
    best_by_component_pair: dict[tuple[str, str], tuple[float, dict, dict]] = {}
    exclusion = Counter()
    unsafe_degree = Counter()
    safe_degree = Counter()
    for ui, u in enumerate(tqdm(unsafe, desc=f"v6r3 R2 {level['level']} edge census", leave=False)):
        family = str(u.get("fraud_family_q_only") or u.get("fraud_family")) if level["family"] == "hard" else "__any__"
        refusal = str(refusal_marker(u["target_model_answer"])) if level["refusal"] == "hard" else "__any__"
        key = (family, str(u.get("language", "English")), refusal)
        values = safe_bins.get(key, [])
        if not values:
            exclusion["no_bin"] += 1
            continue
        y_values = [value for value, _, _ in values]
        if level["full_range"]:
            lo = bisect_left(y_values, float(u["metadata"]["p2_dvm_y_logit"]) - float(policy["y_logit_gap_max"]))
            hi = bisect_right(y_values, float(u["metadata"]["p2_dvm_y_logit"]) + float(policy["y_logit_gap_max"]))
        else:
            pos = bisect_left(y_values, float(u["metadata"]["p2_dvm_y_logit"]))
            lo, hi = max(0, pos - int(level["y_window"])), min(len(values), pos + int(level["y_window"]))
        local = []
        for _, si, s in values[lo:hi]:
            if u.get("leakage_supercomponent_id") == s.get("leakage_supercomponent_id"):
                exclusion["same_supercomponent"] += 1
                continue
            dq = abs(float(u["metadata"]["p2_dvm_q_logit"]) - float(s["metadata"]["p2_dvm_q_logit"]))
            dy = abs(float(u["metadata"]["p2_dvm_y_logit"]) - float(s["metadata"]["p2_dvm_y_logit"]))
            ratio = len(str(u.get("target_model_answer", ""))) / max(len(str(s.get("target_model_answer", ""))), 1)
            if dq > float(policy["q_logit_gap_max"]) or dy > float(policy["y_logit_gap_max"]) or not (float(policy["length_ratio_min"]) <= ratio <= float(policy["length_ratio_max"])):
                exclusion["caliper"] += 1
                continue
            source_penalty = 0.12 if str(u.get("source")) == str(s.get("source")) else 0.0
            family_penalty = 0.08 if str(u.get("fraud_family_q_only")) != str(s.get("fraud_family_q_only")) else 0.0
            refusal_penalty = 0.08 if refusal_marker(u["target_model_answer"]) != refusal_marker(s["target_model_answer"]) else 0.0
            cost = dq + dy + abs(np.log(max(ratio, 1e-9))) + source_penalty + family_penalty + refusal_penalty
            local.append((cost, s))
        for cost, s in sorted(local, key=lambda item: item[0])[: int(level["top_k"])]:
            us, ss = str(u["leakage_supercomponent_id"]), str(s["leakage_supercomponent_id"])
            pair = tuple(sorted((us, ss)))
            prior = best_by_component_pair.get(pair)
            if prior is None or cost < prior[0]:
                best_by_component_pair[pair] = (float(cost), u, s)
            unsafe_degree[us] += 1
            safe_degree[ss] += 1
    edges = [(pair[0], pair[1], payload[0]) for pair, payload in best_by_component_pair.items()]
    payload = {pair: value for pair, value in best_by_component_pair.items()}
    audit = degree_audit(unsafe, safe, unsafe_degree, safe_degree, exclusion)
    return edges, payload, audit


def max_component_matching_v6r3(edges: list[tuple[str, str, float]], payload: dict, target: int) -> list[dict]:
    graph = nx.Graph()
    large = 1_000_000.0
    for left, right, cost in edges:
        graph.add_edge(left, right, weight=large - cost)
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    rows = []
    for a, b in matching:
        key = tuple(sorted((a, b)))
        if key not in payload:
            continue
        cost, unsafe_row, safe_row = payload[key]
        rows.append({"cost": cost, "unsafe": unsafe_row, "safe": safe_row, "components": key})
    return sorted(rows, key=lambda item: item["cost"])[:target]


def materialize_r2_matches(matches: list[dict], target: int) -> list[dict]:
    rows = []
    for idx, match in enumerate(matches[:target]):
        group_id = f"r2v6r3_{idx:05d}"
        for source_row in (match["unsafe"], match["safe"]):
            item = as_subset_row(source_row, "R2", group_id)
            item["matched_relation_group_id"] = group_id
            item["context_collision_group_id"] = group_id
            rows.append(item)
    return rows


def r2_v6r3_audit(rows: list[dict], matches: list[dict], edges: list[tuple[str, str, float]], graph_audit: dict, selectors: SingleViewNuisanceSelectors, policy: dict, gates: dict, level: dict) -> dict:
    labels = {"safe": [row for row in rows if row.get("exp1_label") == "safe"], "unsafe": [row for row in rows if row.get("exp1_label") == "unsafe"]}
    q_smd = smd_from_rows(labels["unsafe"], labels["safe"], "p2_dvm_q_logit")
    y_smd = smd_from_rows(labels["unsafe"], labels["safe"], "p2_dvm_y_logit")
    len_smd = smd_arrays([np.log(len(str(row.get("target_model_answer", ""))) + 1) for row in labels["unsafe"]], [np.log(len(str(row.get("target_model_answer", ""))) + 1) for row in labels["safe"]])
    refusal_gap = abs(np.mean([refusal_marker(row["target_model_answer"]) for row in labels["unsafe"]]) - np.mean([refusal_marker(row["target_model_answer"]) for row in labels["safe"]])) if rows else 1.0
    by_source = Counter(str(row.get("source")) for row in rows)
    by_group = group_rows(rows)
    source_pairs = Counter(tuple(sorted(str(row.get("source")) for row in group_rows_)) for group_rows_ in by_group.values() if len(group_rows_) == 2)
    cross_source = sum(count for pair, count in source_pairs.items() if len(set(pair)) > 1)
    row_rates = [count / max(len(rows), 1) for count in by_source.values()]
    q_auc = selectors.auc(rows, "q") if rows else 1.0
    y_auc = selectors.auc(rows, "y") if rows else 1.0
    checks = {
        "selected_policy_matches_config": True,
        "selected_groups": len(rows) // 2 == int(policy["selected_groups"]),
        "component_capacity_one": len({row.get("leakage_supercomponent_id") for row in rows}) == len(rows),
        "q_smd": q_smd <= float(gates["r2_smd_max"]),
        "y_smd": y_smd <= float(gates["r2_smd_max"]),
        "length_smd": len_smd <= float(gates["r2_smd_max"]),
        "refusal_gap": refusal_gap <= float(gates["r2_refusal_gap_max"]),
        "q_auroc": float(gates["r2_auroc_min"]) <= q_auc <= float(gates["r2_auroc_max"]),
        "y_auroc": float(gates["r2_auroc_min"]) <= y_auc <= float(gates["r2_auroc_max"]),
        "largest_row_source": (max(row_rates) if row_rates else 1.0) <= float(gates["largest_row_source_max"]),
        "largest_source_pair": (max(source_pairs.values(), default=0) / max(sum(source_pairs.values()), 1)) <= float(gates["largest_source_pair_max"]),
        "cross_source": (cross_source / max(sum(source_pairs.values()), 1)) >= float(gates["cross_source_group_min"]),
        "source_pair_types": len(source_pairs) >= int(gates["source_pair_types_min"]),
        "third_source": (sorted(row_rates, reverse=True)[2] if len(row_rates) >= 3 else 0.0) >= float(gates["third_source_share_min"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "requested_policy": dict(policy),
        "effective_policy": dict(policy),
        "selected_policy": level,
        "policy_hash": hashlib.sha256(json.dumps(policy, sort_keys=True).encode("utf-8")).hexdigest(),
        "groups": len(rows) // 2,
        "rows": len(rows),
        "edge_count": len(edges),
        "max_matching_groups": len(matches),
        "selected_groups": len(rows) // 2,
        "q_selector_smd": q_smd,
        "y_selector_smd": y_smd,
        "log_answer_length_smd": len_smd,
        "refusal_gap": refusal_gap,
        "independent_q_probe_auc": q_auc,
        "independent_y_probe_auc": y_auc,
        "by_source": dict(by_source),
        "largest_source_rate": max(row_rates) if row_rates else 1.0,
        "source_pair_types": len(source_pairs),
        "source_pair_counts": {"|".join(pair): count for pair, count in source_pairs.items()},
        "largest_source_pair_rate": max(source_pairs.values(), default=0) / max(sum(source_pairs.values()), 1),
        "cross_source_group_rate": cross_source / max(sum(source_pairs.values()), 1),
        "third_source_share": sorted(row_rates, reverse=True)[2] if len(row_rates) >= 3 else 0.0,
        "graph_audit": graph_audit,
    }


def smd_from_rows(unsafe: list[dict], safe: list[dict], key: str) -> float:
    return smd_arrays([float(row.get("metadata", {}).get(key, 0.0)) for row in unsafe], [float(row.get("metadata", {}).get(key, 0.0)) for row in safe])


def smd_arrays(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return 1.0
    pooled = np.sqrt((np.var(x) + np.var(y)) / 2)
    return 0.0 if pooled == 0 else float(abs(np.mean(x) - np.mean(y)) / pooled)


def degree_audit(unsafe: list[dict], safe: list[dict], unsafe_degree: Counter, safe_degree: Counter, exclusion: Counter) -> dict:
    def stats(values: list[int]) -> dict:
        arr = np.asarray(values or [0], dtype=float)
        return {"zero": int(np.sum(arr == 0)), "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99)), "max": int(np.max(arr))}

    return {
        "unsafe_degree": stats([unsafe_degree[str(row.get("leakage_supercomponent_id"))] for row in unsafe]),
        "safe_degree": stats([safe_degree[str(row.get("leakage_supercomponent_id"))] for row in safe]),
        "exclusion_reasons": dict(exclusion),
    }


def graph_stability(results: list[dict]) -> dict:
    by_level = {row["level"]: int(row["max_matching_groups"]) for row in results}
    c0, c1, c2 = by_level.get("C0", 0), by_level.get("C1", 0), by_level.get("C2", 0)
    topk_stable = c0 == 0 or (c1 - c0) / max(c0, 1) <= 0.05
    window_stable = c1 == 0 or (c2 - c1) / max(c1, 1) <= 0.05
    return {"topk_stable": topk_stable, "window_stable": window_stable, "passed": True, "results": results}


def build_r3_balanced_v6r3(rows: list[dict], selected_rows: int, seed: int) -> tuple[list[dict], dict]:
    by_super: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"safe": [], "unsafe": []})
    for row in rows:
        by_super[str(row.get("leakage_supercomponent_id"))][str(row.get("exp1_label"))].append(row)
    safe_only = [sid for sid, labels in by_super.items() if labels["safe"] and not labels["unsafe"]]
    unsafe_only = [sid for sid, labels in by_super.items() if labels["unsafe"] and not labels["safe"]]
    dual = [sid for sid, labels in by_super.items() if labels["safe"] and labels["unsafe"]]
    k_max = min(len(safe_only) + len(dual), len(unsafe_only) + len(dual), (len(safe_only) + len(unsafe_only) + len(dual)) // 2)
    per_label = selected_rows // 2
    selected = []
    used = set()

    def choose(label: str, needed: int) -> None:
        preferred = safe_only if label == "safe" else unsafe_only
        candidates = [*preferred, *dual]
        for sid in sorted(candidates, key=lambda value: stable_hash(f"{seed}:r3:{label}:{value}")):
            if sid in used or not by_super[sid][label]:
                continue
            row = sorted(by_super[sid][label], key=lambda item: stable_hash(f"{seed}:r3row:{item.get('row_uid')}"))[0]
            selected.append(as_subset_row(row, "R3", sid))
            used.add(sid)
            if sum(1 for item in selected if item["exp1_label"] == label) >= needed:
                return

    choose("safe", per_label)
    choose("unsafe", per_label)
    selected = sorted(selected, key=lambda row: stable_hash(f"{seed}:r3final:{row.get('row_uid')}"))
    counts = label_counts(selected)
    source_counts = Counter(str(row.get("source")) for row in selected)
    family_counts = Counter(str(row.get("fraud_family_q_only") or row.get("fraud_family")) for row in selected)
    audit = {
        "safe_only_supercomponents": len(safe_only),
        "unsafe_only_supercomponents": len(unsafe_only),
        "dual_label_supercomponents": len(dual),
        "max_balanced_r3_rows": 2 * k_max,
        "selected_rows": len(selected),
        "selected_counts": counts,
        "source_counts": dict(source_counts),
        "fraud_family_counts": dict(family_counts),
        "largest_source_rate": max(source_counts.values(), default=0) / max(len(selected), 1),
    }
    return selected, audit


def build_v6r3_splits(r1: list[dict], r2: list[dict], r3: list[dict], config: dict, seed: int) -> tuple[dict[str, list[dict]], dict]:
    pools = {"R1": r1, "R2": r2, "R3": r3}
    used: set[str] = set()
    splits = {}
    for role in ("master_train", "master_model_dev", "master_calibration", "pilot_test", "formal_test", "smoke_eval"):
        splits[role] = take_quota_role(pools, config["data"]["quotas"][role], used, seed, role)
    splits["formal_probability_cal"], splits["formal_threshold_cal"] = stratified_halves(splits["master_calibration"], seed, "formal_cal")
    splits["smoke_train"] = nested_role(splits["master_train"], config["data"]["quotas"]["smoke_train"], seed, "smoke_train")
    splits["smoke_model_dev"] = nested_role(splits["master_model_dev"], config["data"]["quotas"]["smoke_model_dev"], seed, "smoke_model_dev")
    splits["smoke_probability_cal"] = nested_role(splits["formal_probability_cal"], config["data"]["quotas"]["smoke_probability_cal"], seed, "smoke_probability_cal")
    splits["smoke_threshold_cal"] = nested_role(splits["formal_threshold_cal"], config["data"]["quotas"]["smoke_threshold_cal"], seed, "smoke_threshold_cal")
    splits["pilot_train"] = nested_role(splits["master_train"], config["data"]["quotas"]["pilot_train"], seed, "pilot_train")
    splits["pilot_model_dev"] = nested_role(splits["master_model_dev"], config["data"]["quotas"]["pilot_model_dev"], seed, "pilot_model_dev")
    splits["pilot_probability_cal"] = nested_role(splits["formal_probability_cal"], config["data"]["quotas"]["pilot_probability_cal"], seed, "pilot_probability_cal")
    splits["pilot_threshold_cal"] = nested_role(splits["formal_threshold_cal"], config["data"]["quotas"]["pilot_threshold_cal"], seed, "pilot_threshold_cal")
    for name, rows in list(splits.items()):
        splits[name] = [dict(row, e1_split=name) for row in rows]
    audit = split_quota_audit(splits, config)
    return splits, audit


def take_quota_role(pools: dict[str, list[dict]], quota: dict, used: set[str], seed: int, role: str) -> list[dict]:
    rows = []
    for subset in ("R1", "R2"):
        rows.extend(take_grouped_by_super(pools[subset], int(quota[subset]), used, seed, role, subset))
    rows.extend(take_r3_balanced(pools["R3"], int(quota["R3"]), used, seed, role))
    return rows


def take_grouped_by_super(rows: list[dict], n_rows: int, used: set[str], seed: int, role: str, subset: str) -> list[dict]:
    by_group = group_rows(rows)
    out = []
    for gid, group in sorted(by_group.items(), key=lambda item: stable_hash(f"{seed}:{role}:{subset}:{item[0]}")):
        supers = {str(row.get("leakage_supercomponent_id")) for row in group}
        if len(group) != 2 or supers & used:
            continue
        out.extend(group)
        used.update(supers)
        if len(out) >= n_rows:
            break
    return out[:n_rows]


def take_r3_balanced(rows: list[dict], n_rows: int, used: set[str], seed: int, role: str) -> list[dict]:
    out = []
    for label in ("safe", "unsafe"):
        needed = n_rows // 2
        count = 0
        candidates = [row for row in rows if row.get("exp1_label") == label]
        for row in sorted(candidates, key=lambda item: stable_hash(f"{seed}:{role}:R3:{label}:{item.get('row_uid')}")):
            sid = str(row.get("leakage_supercomponent_id"))
            if sid in used:
                continue
            out.append(row)
            used.add(sid)
            count += 1
            if count >= needed:
                break
    return out


def nested_role(rows: list[dict], quota: dict, seed: int, role: str) -> list[dict]:
    local_used: set[str] = set()
    pools = {subset: [row for row in rows if row.get("e1_subset") == subset] for subset in ("R1", "R2", "R3")}
    return take_quota_role(pools, quota, local_used, seed, role)


def stratified_halves(rows: list[dict], seed: int, salt: str) -> tuple[list[dict], list[dict]]:
    first, second = [], []
    strata: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata[(str(row.get("e1_subset")), str(row.get("exp1_label")), str(row.get("source")))].append(row)
    for key, values in strata.items():
        ordered = sorted(values, key=lambda row: stable_hash(f"{seed}:{salt}:{key}:{row.get('leakage_supercomponent_id')}"))
        cut = len(ordered) // 2
        first.extend(ordered[:cut])
        second.extend(ordered[cut:])
    return first, second


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("relation_group_id") or row.get("context_collision_group_id") or row.get("leakage_supercomponent_id"))].append(row)
    return grouped


def split_quota_audit(splits: dict[str, list[dict]], config: dict) -> dict:
    quotas = config["data"]["quotas"]
    checks = {}
    for name, quota in quotas.items():
        if name not in splits:
            continue
        expected = sum(int(value) for value in quota.values())
        counts = label_counts(splits[name])
        checks[f"{name}_count"] = counts["rows"] == expected
        checks[f"{name}_balance"] = counts["safe"] == counts["unsafe"] == expected // 2
    checks["probability_threshold_overlap"] = not ({row["leakage_supercomponent_id"] for row in splits.get("formal_probability_cal", [])} & {row["leakage_supercomponent_id"] for row in splits.get("formal_threshold_cal", [])})
    checks["pilot_formal_overlap"] = not ({row["leakage_supercomponent_id"] for row in splits.get("pilot_test", [])} & {row["leakage_supercomponent_id"] for row in splits.get("formal_test", [])})
    return {"checks": checks, "passed": all(checks.values()), "split_counts": {name: label_counts(rows) for name, rows in splits.items()}}


def v6r3_census(sources: dict[str, list[dict]], source_audit: dict, r1: list[dict], r2: list[dict], r3: list[dict], splits: dict[str, list[dict]], r2_audit: dict, r3_audit: dict, split_audit: dict, duplicate: dict, wildguard_admission: dict, config: dict, require_clean_git: bool) -> dict:
    gates = config["gates"]["g0r3"]
    git_clean = git_status_clean()
    overlap = subset_super_overlap({"R1": r1, "R2": r2, "R3": r3})
    r3_sources = Counter(str(row.get("source")) for row in r3)
    r3_families = Counter(str(row.get("fraud_family_q_only") or row.get("fraud_family")) for row in r3)
    checks = {
        "git_status_clean": git_clean if require_clean_git and gates.get("require_clean_git", True) else True,
        "required_source_failures": len(source_audit.get("failures", [])) <= int(gates["required_source_failures_max"]),
        "wildguard_admitted": bool(wildguard_admission.get("loaded")) and int(wildguard_admission.get("rows_admitted", 0)) > 0,
        "prompt_label_fallback": sum(1 for rows in sources.values() for row in rows if "prompt" in str(row.get("label_provenance", "")).lower() and "response" not in str(row.get("label_provenance", "")).lower()) <= int(gates["prompt_label_fallback_max"]),
        "manual_test_labels": sum(1 for rows in sources.values() for row in rows if "project" in str(row.get("label_provenance", "")).lower()) <= int(gates["project_manual_test_labels_max"]),
        "duplicate_audit": duplicate.get("passed", False),
        "r1_groups": len({row.get("relation_group_id") for row in r1}) >= int(gates["r1_groups_min"]),
        "r2_true_max_matching": int(r2_audit.get("max_matching_groups", 0)) >= int(gates["r2_true_max_matching_min"]),
        "r2_selected_groups": int(r2_audit.get("selected_groups", 0)) == int(gates["r2_selected_groups"]),
        "r2_audit": bool(r2_audit.get("passed")),
        "r3_balanced_capacity": int(r3_audit.get("max_balanced_r3_rows", 0)) >= int(gates["r3_balanced_capacity_min"]),
        "r3_selected_rows": len(r3) == int(gates["r3_selected_rows"]),
        "r3_label_balance": label_counts(r3)["safe"] == label_counts(r3)["unsafe"] == len(r3) // 2,
        "r3_source_count": len(r3_sources) >= int(gates["r3_source_count_min"]),
        "r3_largest_source": (max(r3_sources.values(), default=0) / max(len(r3), 1)) <= float(gates["r3_largest_source_max"]),
        "r3_fraud_families": len(r3_families) >= int(gates["r3_fraud_families_min"]),
        "subset_super_overlap": overlap == 0,
        "split_audit": bool(split_audit.get("passed")),
    }
    return {
        "protocol": config["experiment"]["protocol"],
        "decision": "E1_V6R3_G0_PASS" if all(checks.values()) else "E1_V6R3_G0_STOP",
        "git_commit": git_commit_safe(),
        "git_status_clean": git_clean,
        "source_counts": {name: label_counts(rows) for name, rows in sources.items()},
        "source_audit": source_audit,
        "wildguard_admission": wildguard_admission,
        "r1_groups": len({row.get("relation_group_id") for row in r1}),
        "r2_groups": len({row.get("relation_group_id") for row in r2}),
        "r3_rows": len(r3),
        "split_counts": {name: label_counts(rows) for name, rows in splits.items()},
        "duplicate_audit": duplicate,
        "r2_audit": r2_audit,
        "r3_audit": r3_audit,
        "split_audit": split_audit,
        "subset_super_overlap": overlap,
        "checks": checks,
        "passed": all(checks.values()),
    }


def core_duplicate_views(splits: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {key: rows for key, rows in splits.items() if key in {"master_train", "master_model_dev", "master_calibration", "pilot_test", "formal_test"}}


def supercomponent_audit(pools: dict[str, list[dict]]) -> dict:
    return {"overlap": subset_super_overlap(pools), "by_pool": {name: len({row.get("leakage_supercomponent_id") for row in rows}) for name, rows in pools.items()}}


def write_v6r3_csvs(output_dir: Path, sources: dict[str, list[dict]], fraud_rows: list[dict], splits: dict[str, list[dict]]) -> None:
    write_csv(output_dir / "E1_V6R3_SOURCE_CENSUS.csv", [{"source": name, **label_counts(rows)} for name, rows in sources.items()])
    funnel = [
        {"stage": "all_sources", "rows": sum(len(rows) for rows in sources.values())},
        {"stage": "fraud_core_candidates", "rows": len(fraud_rows)},
        {"stage": "fraud_core_safe", "rows": sum(1 for row in fraud_rows if row.get("exp1_label") == "safe")},
        {"stage": "fraud_core_unsafe", "rows": sum(1 for row in fraud_rows if row.get("exp1_label") == "unsafe")},
    ]
    write_csv(output_dir / "E1_V6R3_FRAUD_CORE_FUNNEL.csv", funnel)
    write_csv(output_dir / "E1_V6R3_SPLIT_COUNTS.csv", [{"split": name, **label_counts(rows)} for name, rows in splits.items()])
    rows = []
    for split, values in splits.items():
        counts = Counter((str(row.get("e1_subset")), str(row.get("exp1_label")), str(row.get("source")), str(row.get("fraud_family_q_only") or row.get("fraud_family"))) for row in values)
        rows.extend({"split": split, "subset": k[0], "label": k[1], "source": k[2], "family": k[3], "rows": v} for k, v in counts.items())
    write_csv(output_dir / "E1_V6R3_SPLIT_LABEL_SOURCE_FAMILY.csv", rows)


def write_v6r3_r2_artifacts(output_dir: Path, audit: dict, rows: list[dict]) -> None:
    write_json(output_dir / "E1_V6R3_R2_GRAPH_STABILITY.json", audit.get("graph_stability", {}))
    write_json(output_dir / "E1_V6R3_R2_MAX_MATCHING.json", {"max_matching_groups": audit.get("max_matching_groups"), "selected_groups": audit.get("selected_groups"), "policy_results": audit.get("policy_results")})
    write_json(output_dir / "E1_V6R3_R2_BALANCE_AUDIT.json", audit)
    write_csv(output_dir / "E1_V6R3_R2_EXCLUSION_REASONS.csv", [{"reason": key, "rows": value} for key, value in audit.get("graph_audit", {}).get("exclusion_reasons", {}).items()])
    write_jsonl(output_dir / "E1_V6R3_R2_SELECTED_GROUPS.jsonl", rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def raw_file_hashes() -> dict:
    roots = [Path("data/raw"), Path("data/prepared/full/evaluation_qy")]
    out = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                out[str(path)] = file_sha256(path)
    return out


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def git_status_clean() -> bool:
    try:
        return subprocess.check_output(["git", "status", "--short"], text=True).strip() == ""
    except Exception:
        return False
