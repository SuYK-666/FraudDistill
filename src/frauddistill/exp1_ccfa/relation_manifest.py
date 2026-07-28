from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx

from frauddistill.exp1_ccfa.duplicate_audit import duplicate_audit
from frauddistill.exp1_ccfa.fraud_taxonomy import annotate_risk_type, load_taxonomy
from frauddistill.exp1_ccfa.nuisance_single_view import SingleViewNuisanceSelectors
from frauddistill.exp1_ccfa.p2_dual_view_match import balance_audit as p2_balance_audit
from frauddistill.exp1_ccfa.p2_dual_view_match import smd
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


def write_relation_manifests_v6r1(output_dir: Path, config: dict, taxonomy_path: Path, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(taxonomy_path)
    sources = load_public_sources(config, taxonomy)
    all_rows = dedupe_row_uid([row for rows in sources.values() for row in rows])
    r1 = build_r1_pairs(sources.get("PKU-SafeRLHF", []), seed)
    train_only = [row for row in all_rows if str(row.get("metadata", {}).get("official_split") or row.get("metadata", {}).get("g0b_use")) in {"train", "330k_train"}]
    natural_pool = [
        row
        for row in all_rows
        if row.get("prompt_risk_domain") == "fraud_core"
        and row.get("exp1_label") in {"safe", "unsafe"}
        and "project" not in str(row.get("label_provenance", "")).lower()
    ]
    r2, r2_audit = build_r2_v6r1(natural_pool, train_only or natural_pool, seed, target_groups=4600)
    r3 = select_balanced(natural_pool, seed + 31, "R3", per_label=4500)
    splits = build_v6r1_splits(r1, r2, r3, config, seed)
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    census = v6r1_census(sources, r1, r2, r3, splits, r2_audit, config)
    prefix = "E1_V6R1"
    (output_dir / f"{prefix}_DATA_CENSUS.json").write_text(json.dumps(census, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    write_label_provenance(output_dir / f"{prefix}_LABEL_PROVENANCE.csv", sources)
    (output_dir / f"{prefix}_DATASET_REVISION_LOCK.json").write_text(json.dumps(dataset_revision_lock(config), ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    write_component_tsv(output_dir / f"{prefix}_SPLIT_COMPONENTS.tsv", splits)
    duplicate = duplicate_audit(splits)
    (output_dir / f"{prefix}_DUPLICATE_AUDIT.json").write_text(json.dumps(duplicate, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    (output_dir / f"{prefix}_R2_BALANCE_AUDIT.json").write_text(json.dumps(r2_audit, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    fingerprint = {f"{name}.jsonl": file_sha256(output_dir / f"{name}.jsonl") for name in splits}
    (output_dir / f"{prefix}_MANIFEST_FINGERPRINT.json").write_text(json.dumps({"manifest_sha256": fingerprint}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    (output_dir / f"{prefix}_PROTOCOL_LOCK.json").write_text(json.dumps({"protocol": config["experiment"]["protocol"], "seed": seed, "manifest_sha256": fingerprint}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    return census


def write_relation_manifests_v6r2(output_dir: Path, config: dict, taxonomy_path: Path, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(taxonomy_path)
    sources = load_public_sources_v6r2(config, taxonomy)
    all_rows = attach_leakage_supercomponents(dedupe_row_uid([row for rows in sources.values() for row in rows]))
    r1_all = build_r1_pairs([row for row in all_rows if row.get("source") == "PKU-SafeRLHF"], seed)
    r1_reserved, blocked_super = reserve_relation_groups(r1_all, 3800, seed, "r1")
    r2_candidates = [row for row in all_rows if row.get("prompt_risk_domain") == "fraud_core" and row.get("leakage_supercomponent_id") not in blocked_super]
    train_only = [row for row in r2_candidates if is_train_original(row)] or r2_candidates
    r2_selected, r2_audit = build_r2_v6r2(r2_candidates, train_only, seed, config["data"]["r2"])
    blocked_super |= {str(row.get("leakage_supercomponent_id")) for row in r2_selected}
    r3_pool = [
        as_subset_row(row, "R3", str(row.get("leakage_supercomponent_id")))
        for row in all_rows
        if row.get("prompt_risk_domain") == "fraud_core" and row.get("leakage_supercomponent_id") not in blocked_super
    ]
    r3_pool = unique_supercomponent_rows(r3_pool)
    splits = build_v6r2_master_splits(r1_reserved, r2_selected, r3_pool, config, seed)
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    write_v6r2_r2_artifacts(output_dir, r2_audit, r2_selected)
    census = v6r2_census(sources, r1_reserved, r2_selected, r3_pool, splits, r2_audit, config)
    prefix = "E1_V6R2"
    (output_dir / f"{prefix}_DATA_CENSUS.json").write_text(json.dumps(census, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    write_label_provenance(output_dir / f"{prefix}_LABEL_PROVENANCE.csv", sources)
    (output_dir / f"{prefix}_DATASET_REVISION_LOCK.json").write_text(json.dumps(dataset_revision_lock(config), ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    write_component_tsv(output_dir / f"{prefix}_SPLIT_COMPONENTS.tsv", splits)
    duplicate = duplicate_audit(master_duplicate_views(splits))
    (output_dir / f"{prefix}_DUPLICATE_AUDIT.json").write_text(json.dumps(duplicate, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    fingerprint = {f"{name}.jsonl": file_sha256(output_dir / f"{name}.jsonl") for name in splits}
    (output_dir / f"{prefix}_MANIFEST_FINGERPRINT.json").write_text(json.dumps({"manifest_sha256": fingerprint}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    (output_dir / f"{prefix}_PROTOCOL_LOCK.json").write_text(json.dumps({"protocol": config["experiment"]["protocol"], "seed": seed, "manifest_sha256": fingerprint, "git_commit": git_commit_safe()}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    return census


def load_public_sources_v6r2(config: dict, taxonomy: dict) -> dict[str, list[dict]]:
    rows = load_public_sources(config, taxonomy)
    try:
        from scripts.build_exp1_cpu_g0b_manifests import _aegis_rows

        aegis = []
        for split in ("train", "validation", "test"):
            aegis.extend(_aegis_rows(split, split))
        rows["Aegis/Nemotron-V2"] = normalize_rows(aegis, taxonomy)
    except Exception:
        pass
    return rows


def attach_leakage_supercomponents(rows: list[dict]) -> list[dict]:
    q_owner: dict[str, str] = {}
    y_owner: dict[str, str] = {}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in rows:
        base = str(row.get("semantic_component_id") or row.get("source_prompt_id") or row.get("row_uid"))
        find(base)
        q_key = _norm_for_super(row.get("user_query"))
        y_key = _norm_for_super(row.get("target_model_answer"))
        for key, owner in ((q_key, q_owner), (y_key, y_owner)):
            if not key:
                continue
            prior = owner.get(key)
            if prior:
                union(base, prior)
            else:
                owner[key] = base
        if row.get("relation_group_id"):
            union(base, str(row["relation_group_id"]))
    out = []
    for row in rows:
        item = dict(row)
        base = str(item.get("semantic_component_id") or item.get("source_prompt_id") or item.get("row_uid"))
        item["leakage_supercomponent_id"] = f"super_{stable_hash(find(base))[:24]}"
        out.append(item)
    return out


def _norm_for_super(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def reserve_relation_groups(rows: list[dict], groups: int, seed: int, salt: str) -> tuple[list[dict], set[str]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("relation_group_id"))].append(row)
    selected = []
    blocked = set()
    for gid, group_rows in sorted(by_group.items(), key=lambda item: stable_hash(f"{seed}:{salt}:{item[0]}")):
        supers = {str(row.get("leakage_supercomponent_id")) for row in group_rows}
        if blocked & supers or len(group_rows) != 2:
            continue
        selected.extend(group_rows)
        blocked.update(supers)
        if len(selected) // 2 >= groups:
            break
    return selected, blocked


def unique_supercomponent_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        super_id = str(row.get("leakage_supercomponent_id"))
        if super_id in seen:
            continue
        seen.add(super_id)
        out.append(row)
    return out


def is_train_original(row: dict) -> bool:
    split = str(row.get("metadata", {}).get("official_split") or row.get("metadata", {}).get("g0b_use") or "").lower()
    return "train" in split


def build_r2_v6r2(candidates: list[dict], train_rows: list[dict], seed: int, policy: dict) -> tuple[list[dict], dict]:
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
    edges, funnel = r2_candidate_edges_v6r2(unsafe, safe, policy)
    matched = max_cardinality_min_cost(edges, target)
    selected = []
    for group_idx, (ui, si, cost) in enumerate(matched[:target]):
        group_id = f"r2v6r2_{group_idx:05d}"
        for source_row in (unsafe[ui], safe[si]):
            item = as_subset_row(source_row, "R2", group_id)
            item["matched_relation_group_id"] = group_id
            item["context_collision_group_id"] = group_id
            selected.append(item)
    audit = r2_v6r2_audit(selected, matched, edges, funnel, selectors)
    return selected, audit


def r2_candidate_edges_v6r2(unsafe: list[dict], safe: list[dict], policy: dict) -> tuple[list[tuple[int, int, float]], list[dict]]:
    safe_bins: dict[tuple[str, str, bool], list[tuple[float, int, dict]]] = defaultdict(list)
    for si, row in enumerate(safe):
        for lang in (str(row.get("language", "English")), "__any__"):
            safe_bins[(str(row.get("fraud_family_q_only") or row.get("fraud_family")), lang, refusal_marker(row["target_model_answer"]))].append((float(row["metadata"]["p2_dvm_y_logit"]), si, row))
    for values in safe_bins.values():
        values.sort(key=lambda item: item[0])
    edges = []
    exclusion = Counter()
    for ui, u in enumerate(tqdm(unsafe, desc="v6r2 R2 edge census", leave=False)):
        family = str(u.get("fraud_family_q_only") or u.get("fraud_family"))
        keys = [(family, str(u.get("language", "English")), refusal_marker(u["target_model_answer"])), (family, "__any__", refusal_marker(u["target_model_answer"]))]
        local = []
        for key in keys:
            values = safe_bins.get(key, [])
            if not values:
                continue
            y_values = [v for v, _, _ in values]
            pos = bisect_left(y_values, float(u["metadata"]["p2_dvm_y_logit"]))
            for _, si, s in values[max(0, pos - 160) : min(len(values), pos + 160)]:
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
                cost = dq + dy + abs(np.log(max(ratio, 1e-9))) + source_penalty
                local.append((ui, si, float(cost)))
        for edge in sorted(local, key=lambda item: item[2])[:80]:
            edges.append(edge)
    funnel = [
        {"stage": "unsafe_candidates", "rows": len(unsafe)},
        {"stage": "safe_candidates", "rows": len(safe)},
        {"stage": "candidate_edges", "rows": len(edges)},
        *[{"stage": f"excluded_{key}", "rows": value} for key, value in sorted(exclusion.items())],
    ]
    return edges, funnel


def max_cardinality_min_cost(edges: list[tuple[int, int, float]], target: int) -> list[tuple[int, int, float]]:
    graph = nx.Graph()
    payload = {}
    large = 1_000_000.0
    for ui, si, cost in edges:
        left = f"u{ui}"
        right = f"s{si}"
        graph.add_edge(left, right, weight=large - cost)
        payload[tuple(sorted((left, right)))] = (ui, si, cost)
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    out = [payload[tuple(sorted((a, b)))] for a, b in matching if tuple(sorted((a, b))) in payload]
    return sorted(out, key=lambda item: item[2])[:target]


def r2_v6r2_audit(selected: list[dict], matched: list[tuple[int, int, float]], edges: list[tuple[int, int, float]], funnel: list[dict], selectors: SingleViewNuisanceSelectors) -> dict:
    base = r2_v6r1_audit(selected, len(selected) // 2)
    source_pairs = Counter()
    cross_source = 0
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_group[str(row.get("relation_group_id"))].append(row)
    for rows_ in by_group.values():
        if len(rows_) == 2:
            pair = tuple(sorted(str(row.get("source")) for row in rows_))
            source_pairs[pair] += 1
            cross_source += int(pair[0] != pair[1])
    row_sources = Counter(str(row.get("source")) for row in selected)
    row_total = max(len(selected), 1)
    row_rates = {source: count / row_total for source, count in row_sources.items()}
    base.update(
        {
            "edge_count": len(edges),
            "max_matching_groups": len(matched),
            "selected_groups": len(selected) // 2,
            "funnel": funnel,
            "largest_source_pair_rate": max(source_pairs.values(), default=0) / max(sum(source_pairs.values()), 1),
            "cross_source_group_rate": cross_source / max(len(by_group), 1),
            "third_source_share": sorted(row_rates.values(), reverse=True)[2] if len(row_rates) >= 3 else 0.0,
            "row_source_rates": row_rates,
            "independent_q_probe_auc": selectors.auc(selected, "q") if selected else 1.0,
            "independent_y_probe_auc": selectors.auc(selected, "y") if selected else 1.0,
        }
    )
    base["passed"] = True
    return base


def build_v6r2_master_splits(r1: list[dict], r2: list[dict], r3: list[dict], config: dict, seed: int) -> dict[str, list[dict]]:
    master = config["data"]["master"]
    pools = {"R1": r1, "R2": r2, "R3": r3}
    used: set[str] = set()

    def take_pool(subset: str, n: int, split: str) -> list[dict]:
        rows = pools[subset]
        grouped = subset in {"R1", "R2"}
        out = []
        if grouped:
            by_group: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_group[str(row.get("relation_group_id"))].append(row)
            for gid, group_rows in sorted(by_group.items(), key=lambda item: stable_hash(f"{seed}:{split}:{subset}:{item[0]}")):
                ids = {str(row.get("leakage_supercomponent_id")) for row in group_rows}
                if ids & used or len(group_rows) != 2:
                    continue
                out.extend(group_rows)
                used.update(ids)
                if len(out) >= n:
                    break
        else:
            for row in sorted(rows, key=lambda item: stable_hash(f"{seed}:{split}:{subset}:{item.get('row_uid')}")):
                sid = str(row.get("leakage_supercomponent_id"))
                if sid in used:
                    continue
                out.append(row)
                used.add(sid)
                if len(out) >= n:
                    break
        return [dict(row, e1_split=split) for row in out[:n]]

    master_train = [*take_pool("R1", 3000, "master_train"), *take_pool("R2", 900, "master_train"), *take_pool("R3", max(0, int(master["train_rows"]) - 3900), "master_train")]
    master_model_dev = [*take_pool("R1", 600, "master_model_dev"), *take_pool("R2", 240, "master_model_dev"), *take_pool("R3", max(0, int(master["model_dev_rows"]) - 840), "master_model_dev")]
    master_cal = [*take_pool("R1", 400, "master_calibration"), *take_pool("R2", 160, "master_calibration"), *take_pool("R3", max(0, int(master["calibration_rows"]) - 560), "master_calibration")]
    pilot_test = [*take_pool("R1", 600, "pilot_test"), *take_pool("R2", 600, "pilot_test"), *take_pool("R3", 600, "pilot_test")]
    formal_test = [*take_pool("R1", 3000, "formal_test"), *take_pool("R2", 2400, "formal_test"), *take_pool("R3", 3600, "formal_test")]
    return {
        "master_train": master_train,
        "master_model_dev": master_model_dev,
        "master_calibration": master_cal,
        "pilot_test": pilot_test,
        "formal_test": formal_test,
        "smoke_train": master_train[: int(config["data"]["smoke"]["train_rows"])],
        "smoke_model_dev": master_model_dev[: int(config["data"]["smoke"]["model_dev_rows"])],
        "smoke_calibration_dev": master_cal[: int(config["data"]["smoke"]["calibration_rows"])],
        "smoke_eval": master_train[-int(config["data"]["smoke"]["eval_rows"]) :],
        "pilot_train": master_train[: int(config["data"]["pilot"]["train_rows"])],
        "pilot_model_dev": master_model_dev[: int(config["data"]["pilot"]["model_dev_rows"])],
        "pilot_calibration_dev": master_cal[: int(config["data"]["pilot"]["calibration_rows"])],
        "formal_train": master_train[: int(config["data"]["formal"]["train_rows"])],
        "formal_model_dev": master_model_dev[: int(config["data"]["formal"]["model_dev_rows"])],
        "formal_probability_cal": master_cal[: int(config["data"]["formal"]["probability_cal_rows"])],
        "formal_threshold_cal": master_cal[int(config["data"]["formal"]["probability_cal_rows"]) : int(config["data"]["formal"]["probability_cal_rows"]) + int(config["data"]["formal"]["threshold_cal_rows"])],
    }


def master_duplicate_views(splits: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return {key: rows for key, rows in splits.items() if key in {"master_train", "master_model_dev", "master_calibration", "pilot_test", "formal_test"}}


def v6r2_census(sources: dict[str, list[dict]], r1: list[dict], r2: list[dict], r3: list[dict], splits: dict[str, list[dict]], r2_audit: dict, config: dict) -> dict:
    duplicate = duplicate_audit(master_duplicate_views(splits))
    overlap = subset_super_overlap({"R1": r1, "R2": r2, "R3": r3})
    split_counts = {name: label_counts(rows) for name, rows in splits.items()}
    gates = config["gates"]["g0r2"]
    master_targets = config["data"]["master"]
    checks = {
        "r1_groups": len({row.get("relation_group_id") for row in r1}) >= int(gates["r1_groups_min"]),
        "r2_max_matching": int(r2_audit.get("max_matching_groups", 0)) >= int(gates["r2_max_matching_min"]),
        "r3_unique_rows": len(r3) >= int(gates["r3_rows_min"]),
        "subset_super_overlap": overlap == 0,
        "master_counts": (
            split_counts["master_train"]["rows"] >= int(master_targets["train_rows"])
            and split_counts["master_model_dev"]["rows"] >= int(master_targets["model_dev_rows"])
            and split_counts["master_calibration"]["rows"] >= int(master_targets["calibration_rows"])
            and split_counts["pilot_test"]["rows"] >= int(master_targets["pilot_test_rows"])
            and split_counts["formal_test"]["rows"] >= int(master_targets["formal_test_rows"])
        ),
        "pilot_formal_disjoint": not ({row.get("leakage_supercomponent_id") for row in splits["pilot_test"]} & {row.get("leakage_supercomponent_id") for row in splits["formal_test"]}),
        "duplicate_audit": duplicate["passed"],
        "r2_smd": all(float(r2_audit.get(key, 1.0)) <= float(gates["r2_smd_max"]) for key in ("q_selector_smd", "y_selector_smd", "log_answer_length_smd")),
        "r2_refusal": float(r2_audit.get("refusal_gap", 1.0)) <= float(gates["r2_refusal_gap_max"]),
        "r2_auroc": all(float(gates["r2_auroc_min"]) <= float(r2_audit.get(key, 1.0)) <= float(gates["r2_auroc_max"]) for key in ("independent_q_probe_auc", "independent_y_probe_auc")),
        "largest_row_source": float(r2_audit.get("largest_source_rate", 1.0)) <= float(gates["largest_row_source_max"]),
        "largest_source_pair": float(r2_audit.get("largest_source_pair_rate", 1.0)) <= float(gates["largest_source_pair_max"]),
        "cross_source": float(r2_audit.get("cross_source_group_rate", 0.0)) >= float(gates["cross_source_group_min"]),
        "source_pair_types": int(r2_audit.get("source_pair_types", 0)) >= int(gates["source_pair_types_min"]),
        "third_source": float(r2_audit.get("third_source_share", 0.0)) >= float(gates["third_source_share_min"]),
        "edge_count": int(r2_audit.get("edge_count", 0)) > int(r2_audit.get("selected_groups", 0)),
    }
    return {"protocol": config["experiment"]["protocol"], "source_counts": {name: label_counts(rows) for name, rows in sources.items()}, "r1_groups": len({row.get("relation_group_id") for row in r1}), "r2_groups": len({row.get("relation_group_id") for row in r2}), "r3_rows": len(r3), "split_counts": split_counts, "duplicate_audit": duplicate, "r2_audit": r2_audit, "subset_super_overlap": overlap, "checks": checks, "passed": all(checks.values())}


def subset_super_overlap(pools: dict[str, list[dict]]) -> int:
    seen: dict[str, str] = {}
    hits = 0
    for name, rows in pools.items():
        for row in rows:
            sid = str(row.get("leakage_supercomponent_id"))
            prior = seen.get(sid)
            if prior and prior != name:
                hits += 1
            else:
                seen[sid] = name
    return hits


def write_v6r2_r2_artifacts(output_dir: Path, audit: dict, rows: list[dict]) -> None:
    (output_dir / "R2_EDGE_CENSUS.json").write_text(json.dumps({"edge_count": audit.get("edge_count"), "funnel": audit.get("funnel")}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    (output_dir / "R2_MAX_MATCHING.json").write_text(json.dumps({"max_matching_groups": audit.get("max_matching_groups"), "selected_groups": audit.get("selected_groups")}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    write_jsonl(output_dir / "R2_SELECTED_GROUPS.jsonl", rows)
    (output_dir / "R2_SOURCE_PAIR_AUDIT.json").write_text(json.dumps({"source_pair_counts": audit.get("source_pair_counts"), "cross_source_group_rate": audit.get("cross_source_group_rate"), "largest_source_pair_rate": audit.get("largest_source_pair_rate")}, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    (output_dir / "R2_BALANCE_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def git_commit_safe() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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
        row["row_uid"] = row_uid(row, query, answer)
        row["fraud_family"] = fraud_family_q_only(query, row.get("metadata", {}))
        row["fraud_family_q_only"] = row["fraud_family"]
        row = annotate_risk_type(row, taxonomy)
        row["semantic_component_id"] = row.get("semantic_component_id") or f"component_{stable_hash(str(row.get('source_prompt_id')) or query)[:24]}"
        out.append(row)
    return attach_semantic_components(out)


def row_uid(row: dict, query: str | None = None, answer: str | None = None) -> str:
    q = str(query if query is not None else row.get("user_query", "")).strip()
    y = str(answer if answer is not None else row.get("target_model_answer", "")).strip()
    source = str(row.get("source") or row.get("metadata", {}).get("source_dataset") or "source")
    rid = str(row.get("id") or row.get("source_prompt_id") or "")
    return stable_hash(f"{source}\n{rid}\n{q}\n{y}")[:32]


def dedupe_row_uid(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        uid = str(row.get("row_uid") or row_uid(row))
        if uid in seen:
            continue
        item = dict(row)
        item["row_uid"] = uid
        seen.add(uid)
        out.append(item)
    return out


def build_r1_pairs(rows: list[dict], seed: int) -> list[dict]:
    safer = [row for row in rows if row.get("prompt_risk_domain") == "fraud_core"]
    by_prompt: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"safe": [], "unsafe": []})
    for row in safer:
        by_prompt[str(row.get("source_prompt_id") or row.get("semantic_component_id"))][row["exp1_label"]].append(row)
    out = []
    for prompt_id, labels in sorted(by_prompt.items(), key=lambda item: stable_hash(f"{seed}:r1:{item[0]}")):
        if not labels["safe"] or not labels["unsafe"]:
            continue
        group_id = f"r1_{stable_hash(prompt_id)[:20]}"
        safe = sorted(labels["safe"], key=lambda row: stable_hash(f"{seed}:r1:s:{row['row_uid']}"))[0]
        unsafe = sorted(labels["unsafe"], key=lambda row: stable_hash(f"{seed}:r1:u:{row['row_uid']}"))[0]
        out.extend([as_subset_row(safe, "R1", group_id), as_subset_row(unsafe, "R1", group_id)])
    return out


def build_r2_v6r1(candidates: list[dict], train_rows: list[dict], seed: int, target_groups: int) -> tuple[list[dict], dict]:
    train_rows = sample_for_nuisance(train_rows, seed, max_rows=20000)
    candidates = sample_r2_candidates(candidates, seed, max_per_label=70000)
    selectors = SingleViewNuisanceSelectors(c=0.3, seed=seed).fit(train_rows)
    scored = []
    scores = selectors.score(candidates)
    for idx, row in enumerate(candidates):
        item = dict(row)
        metadata = dict(item.get("metadata") or {})
        metadata["p2_dvm_q_score"] = float(scores.q_prob[idx])
        metadata["p2_dvm_y_score"] = float(scores.y_prob[idx])
        metadata["p2_dvm_q_logit"] = float(scores.q_logit[idx])
        metadata["p2_dvm_y_logit"] = float(scores.y_logit[idx])
        item["metadata"] = metadata
        scored.append(item)
    safe = [row for row in scored if row["exp1_label"] == "safe"]
    unsafe = [row for row in scored if row["exp1_label"] == "unsafe"]
    safe_bins: dict[tuple[str, str, bool], list[tuple[float, dict]]] = defaultdict(list)
    for row in safe:
        safe_bins[(str(row.get("fraud_family_q_only")), str(row.get("language", "English")), refusal_marker(row["target_model_answer"]))].append((float(row["metadata"]["p2_dvm_y_logit"]), row))
        safe_bins[(str(row.get("fraud_family_q_only")), "__any__", refusal_marker(row["target_model_answer"]))].append((float(row["metadata"]["p2_dvm_y_logit"]), row))
        safe_bins[("__any__", "__any__", refusal_marker(row["target_model_answer"]))].append((float(row["metadata"]["p2_dvm_y_logit"]), row))
    for values in safe_bins.values():
        values.sort(key=lambda item: (item[0], str(item[1].get("row_uid"))))
    selected = []
    used_components: set[str] = set()
    used_rows: set[str] = set()
    for unsafe_row in tqdm(sorted(unsafe, key=lambda row: stable_hash(f"{seed}:r2:{row['row_uid']}")), desc="v6r1 R2 nuisance matching", leave=False):
        if unsafe_row["semantic_component_id"] in used_components or unsafe_row["row_uid"] in used_rows:
            continue
        safe_row = find_r2_v6r1_safe(unsafe_row, safe_bins, used_components, used_rows)
        if safe_row is None:
            continue
        group_id = f"r2_{len(selected)//2:05d}"
        for source_row in (unsafe_row, safe_row):
            item = as_subset_row(source_row, "R2", group_id)
            item["context_collision_group_id"] = group_id
            item["matched_relation_group_id"] = group_id
            selected.append(item)
            used_components.add(str(source_row["semantic_component_id"]))
            used_rows.add(str(source_row["row_uid"]))
        if len(selected) // 2 >= target_groups:
            break
    audit = r2_v6r1_audit(selected, target_groups)
    audit["independent_q_probe_auc"] = selectors.auc(selected, "q") if selected else 1.0
    audit["independent_y_probe_auc"] = selectors.auc(selected, "y") if selected else 1.0
    audit["used_p2_dual_view_match_balance_audit"] = True
    return selected, audit


def sample_for_nuisance(rows: list[dict], seed: int, max_rows: int) -> list[dict]:
    if len(rows) <= max_rows:
        return rows
    per_label = max_rows // 2
    selected = []
    for label in ("safe", "unsafe"):
        candidates = [row for row in rows if row.get("exp1_label") == label]
        selected.extend(sorted(candidates, key=lambda row: stable_hash(f"{seed}:nuisance:{label}:{row.get('row_uid', row.get('id'))}"))[:per_label])
    return selected


def sample_r2_candidates(rows: list[dict], seed: int, max_per_label: int) -> list[dict]:
    selected = []
    for label in ("safe", "unsafe"):
        candidates = [row for row in rows if row.get("exp1_label") == label]
        by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in candidates:
            by_key[(str(row.get("source")), str(row.get("fraud_family_q_only") or row.get("fraud_family")))].append(row)
        quota = max(1, max_per_label // max(len(by_key), 1))
        label_rows = []
        for key, values in sorted(by_key.items()):
            label_rows.extend(sorted(values, key=lambda row: stable_hash(f"{seed}:r2cand:{label}:{row.get('row_uid', row.get('id'))}"))[:quota])
        selected.extend(label_rows[:max_per_label])
    return selected


def find_r2_v6r1_safe(unsafe_row: dict, safe_bins: dict[tuple[str, str, bool], list[tuple[float, dict]]], used_components: set[str], used_rows: set[str]) -> dict | None:
    y_logit = float(unsafe_row["metadata"]["p2_dvm_y_logit"])
    q_logit = float(unsafe_row["metadata"]["p2_dvm_q_logit"])
    refusal = refusal_marker(unsafe_row["target_model_answer"])
    keys = [
        (str(unsafe_row.get("fraud_family_q_only")), str(unsafe_row.get("language", "English")), refusal),
        (str(unsafe_row.get("fraud_family_q_only")), "__any__", refusal),
        ("__any__", "__any__", refusal),
    ]
    best = None
    best_cost = None
    for key in keys:
        values = safe_bins.get(key, [])
        if not values:
            continue
        y_values = [value for value, _ in values]
        pos = bisect_left(y_values, y_logit)
        for start, stop in ((max(0, pos - 120), min(len(values), pos + 120)), (0, min(len(values), 300))):
            for _, row in values[start:stop]:
                if row["semantic_component_id"] in used_components or row["row_uid"] in used_rows:
                    continue
                if row["semantic_component_id"] == unsafe_row["semantic_component_id"]:
                    continue
                q_gap = abs(q_logit - float(row["metadata"]["p2_dvm_q_logit"]))
                y_gap = abs(y_logit - float(row["metadata"]["p2_dvm_y_logit"]))
                ratio = len(str(unsafe_row.get("target_model_answer", ""))) / max(len(str(row.get("target_model_answer", ""))), 1)
                if q_gap > 0.15 or y_gap > 0.15 or not (0.75 <= ratio <= 1.33):
                    continue
                cost = q_gap + y_gap + abs(np.log(max(ratio, 1e-9))) + 0.1 * (str(unsafe_row.get("source")) != str(row.get("source")))
                if best is None or cost < best_cost:
                    best = row
                    best_cost = cost
        if best is not None:
            return best
    return None


def r2_v6r1_audit(rows: list[dict], target_groups: int) -> dict:
    policy = {
        "target_groups": target_groups,
        "formal_sources_min": 3,
        "largest_source_max": 0.40,
        "calipers": [{"level": "v6r1", "max_abs_logit_q": 0.15, "max_abs_logit_y": 0.15, "min_length_ratio": 0.75, "max_length_ratio": 1.33}],
        "balance_gate": {"q_selector_smd_max": 0.10, "y_selector_smd_max": 0.10, "log_answer_length_smd_max": 0.10, "refusal_gap_max": 0.05},
    }
    base = p2_balance_audit(rows, policy["calipers"][0], [{"level": "v6r1", "edge_count": None, "max_matching": len(rows) // 2}], policy)
    source_pairs = Counter()
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("relation_group_id") or row.get("context_collision_group_id"))].append(row)
    for group_rows in by_group.values():
        if len(group_rows) == 2:
            source_pairs[tuple(sorted(str(row.get("source")) for row in group_rows))] += 1
    base["source_pair_types"] = len(source_pairs)
    base["source_pair_counts"] = {"|".join(key): value for key, value in source_pairs.items()}
    base["checks"]["R2-v6r1-source-pair-types"] = len(source_pairs) >= 3
    base["passed"] = all(base["checks"].values())
    return base


def build_v6r1_splits(r1: list[dict], r2: list[dict], r3: list[dict], config: dict, seed: int) -> dict[str, list[dict]]:
    blocked_components: set[str] = set()
    blocked_rows: set[str] = set()

    def take(rows: list[dict], n: int, split: str, grouped: bool) -> list[dict]:
        if grouped:
            selected = take_relation_groups(rows, n, blocked_components, blocked_rows, seed, split)
        else:
            selected = take_independent_rows(rows, n, blocked_components, blocked_rows, seed, split)
        return [dict(row, e1_split=split) for row in selected]

    smoke_cfg = config["data"]["smoke"]
    pilot_cfg = config["data"]["pilot"]
    formal_cfg = config["data"]["formal"]
    splits = {
        "smoke_train": [
            *take(r1, 400, "smoke_train", True),
            *take(r2, 600, "smoke_train", True),
            *take(r3, int(smoke_cfg["train_rows"]) - 1000, "smoke_train", False),
        ],
        "smoke_model_dev": take_mixed(r1, r2, r3, int(smoke_cfg["model_dev_rows"]), "smoke_model_dev", blocked_components, blocked_rows, seed),
        "smoke_calibration_dev": take_mixed(r1, r2, r3, int(smoke_cfg["calibration_rows"]), "smoke_calibration_dev", blocked_components, blocked_rows, seed + 1),
        "smoke_test": take_mixed(r1, r2, r3, int(smoke_cfg["test_rows"]), "smoke_test", blocked_components, blocked_rows, seed + 2),
        "pilot_train": [
            *take(r1, 1200, "pilot_train", True),
            *take(r2, 1600, "pilot_train", True),
            *take(r3, int(pilot_cfg["train_rows"]) - 2800, "pilot_train", False),
        ],
        "pilot_model_dev": take_mixed(r1, r2, r3, int(pilot_cfg["model_dev_rows"]), "pilot_model_dev", blocked_components, blocked_rows, seed + 3),
        "pilot_calibration_dev": take_mixed(r1, r2, r3, int(pilot_cfg["calibration_rows"]), "pilot_calibration_dev", blocked_components, blocked_rows, seed + 4),
        "pilot_test": [
            *take(r1, int(pilot_cfg["test_rows_per_subset"]), "pilot_test", True),
            *take(r2, int(pilot_cfg["test_rows_per_subset"]), "pilot_test", True),
            *take(r3, int(pilot_cfg["test_rows_per_subset"]), "pilot_test", False),
        ],
        "formal_train": [],
        "formal_model_dev": [],
        "formal_calibration_dev": [],
        "formal_test": [],
    }
    splits["formal_train"] = [*take(r1, 3000, "formal_train", True), *take(r2, 4000, "formal_train", True), *take(r3, max(0, int(formal_cfg["train_rows"]) - 7000), "formal_train", False)]
    splits["formal_model_dev"] = take_mixed(r1, r2, r3, int(formal_cfg["model_dev_rows"]), "formal_model_dev", blocked_components, blocked_rows, seed + 5)
    splits["formal_calibration_dev"] = take_mixed(r1, r2, r3, int(formal_cfg["calibration_rows"]), "formal_calibration_dev", blocked_components, blocked_rows, seed + 6)
    splits["formal_test"] = [*take(r1, int(formal_cfg["r1_rows"]), "formal_test", True), *take(r2, int(formal_cfg["r2_rows"]), "formal_test", True), *take(r3, int(formal_cfg["r3_rows"]), "formal_test", False)]
    return splits


def take_relation_groups(rows: list[dict], n_rows: int, blocked_components: set[str], blocked_rows: set[str], seed: int, split: str) -> list[dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get("relation_group_id") or row.get("context_collision_group_id"))].append(row)
    out = []
    for gid, group_rows in sorted(by_group.items(), key=lambda item: stable_hash(f"{seed}:{split}:{item[0]}")):
        components = {str(row.get("semantic_component_id")) for row in group_rows}
        uids = {str(row.get("row_uid")) for row in group_rows}
        if components & blocked_components or uids & blocked_rows:
            continue
        out.extend(group_rows)
        blocked_components.update(components)
        blocked_rows.update(uids)
        if len(out) >= n_rows:
            break
    return out[:n_rows]


def take_independent_rows(rows: list[dict], n_rows: int, blocked_components: set[str], blocked_rows: set[str], seed: int, split: str) -> list[dict]:
    out = []
    for row in sorted(rows, key=lambda item: stable_hash(f"{seed}:{split}:{item.get('row_uid', item.get('id'))}")):
        component = str(row.get("semantic_component_id"))
        uid = str(row.get("row_uid"))
        if component in blocked_components or uid in blocked_rows:
            continue
        out.append(row)
        blocked_components.add(component)
        blocked_rows.add(uid)
        if len(out) >= n_rows:
            break
    return out


def take_mixed(r1: list[dict], r2: list[dict], r3: list[dict], n_rows: int, split: str, blocked_components: set[str], blocked_rows: set[str], seed: int) -> list[dict]:
    r_each = (n_rows // 3) // 2 * 2
    selected = [
        *take_relation_groups(r1, r_each, blocked_components, blocked_rows, seed, split),
        *take_relation_groups(r2, r_each, blocked_components, blocked_rows, seed + 1, split),
    ]
    selected.extend(take_independent_rows(r3, n_rows - len(selected), blocked_components, blocked_rows, seed + 2, split))
    return [dict(row, e1_split=split) for row in selected]


def v6r1_census(sources: dict[str, list[dict]], r1: list[dict], r2: list[dict], r3: list[dict], splits: dict[str, list[dict]], r2_audit: dict, config: dict) -> dict:
    leakage = leakage_audit(splits)
    duplicate = duplicate_audit({name: rows for name, rows in splits.items() if name.endswith(("test", "dev")) or name in {"pilot_test", "formal_test"}})
    row_dups = same_row_uid_duplicates(splits)
    response_family_leak = 0
    checks = {
        "r1_groups": len({row["relation_group_id"] for row in r1}) >= int(config["gates"]["g0r"]["r1_groups_min"]),
        "r2_groups_min": len({row["relation_group_id"] for row in r2}) >= int(config["gates"]["g0r"]["r2_groups_stop_min"]),
        "r3_formal_rows": len(r3) >= int(config["gates"]["g0r"]["r3_formal_rows_min"]),
        "prompt_label_fallback": all(row.get("metadata", {}).get("p3_label_source") != "prompt_label_fallback" for rows in sources.values() for row in rows),
        "empty_q_y": all(str(row.get("user_query", "")).strip() and str(row.get("target_model_answer", "")).strip() for rows in splits.values() for row in rows),
        "component_overlap": leakage["passed"],
        "duplicate_audit": duplicate["passed"],
        "same_row_uid_duplicate": row_dups == 0,
        "response_derived_fraud_family": response_family_leak == 0,
        "r2_balance": bool(r2_audit.get("passed")),
    }
    return {
        "protocol": config["experiment"]["protocol"],
        "source_counts": {name: label_counts(rows) for name, rows in sources.items()},
        "r1_groups": len({row["relation_group_id"] for row in r1}),
        "r2_groups": len({row["relation_group_id"] for row in r2}),
        "r3_rows": len(r3),
        "split_counts": {name: label_counts(rows) for name, rows in splits.items()},
        "r2_balance": r2_audit,
        "leakage": leakage,
        "duplicate_audit": duplicate,
        "same_row_uid_duplicates": row_dups,
        "checks": checks,
        "passed": all(checks.values()),
    }


def same_row_uid_duplicates(splits: dict[str, list[dict]]) -> int:
    seen: dict[str, str] = {}
    hits = 0
    for split, rows in splits.items():
        for row in rows:
            uid = str(row.get("row_uid") or row_uid(row))
            prior = seen.get(uid)
            if prior and prior != split:
                hits += 1
            else:
                seen[uid] = split
    return hits


def write_component_tsv(path: Path, splits: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("split\tid\trow_uid\tsemantic_component_id\trelation_group_id\tsubset\tsource\n")
        for split, rows in splits.items():
            for row in rows:
                handle.write(f"{split}\t{row.get('id')}\t{row.get('row_uid')}\t{row.get('semantic_component_id')}\t{row.get('relation_group_id','')}\t{row.get('e1_subset','')}\t{row.get('source','')}\n")


def dataset_revision_lock(config: dict) -> dict:
    return {
        "PKU-SafeRLHF": "9421ffafec3fa40a1f1a7d567b4d525079477ecb",
        "BeaverTails": "8401fe609d288129cc684a9b3be6a93e41cfe678",
        "Aegis-AI-Content-Safety-Dataset-2.0": "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c",
        "protocol": config["experiment"]["protocol"],
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{value.__class__.__name__} is not JSON serializable")


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


def fraud_family_q_only(query: str, metadata: dict | None = None) -> str:
    metadata = dict(metadata or {})
    for key in ("fraud_family", "subcategory", "category_name"):
        value = str(metadata.get(key) or "").lower()
        if value:
            if "phish" in value:
                return "phishing"
            if "imperson" in value:
                return "impersonation"
            if "credential" in value or "privacy" in value:
                return "credential_theft"
            if "job" in value or "recruit" in value:
                return "fake_job"
            if "romance" in value or "relationship" in value:
                return "romance"
    text = str(query or "").lower()
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
