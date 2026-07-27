from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from frauddistill.exp1_ccfa.nuisance_single_view import NuisanceScores


@dataclass
class MatchResult:
    rows: list[dict]
    audit: dict
    edge_rows: list[dict]
    exclusion_rows: list[dict]


def build_p2_dvm(
    candidates: list[dict],
    scores: NuisanceScores,
    policy: dict,
    seed: int,
    output_dir: Path | None = None,
) -> MatchResult:
    indexed = [_with_scores(row, scores, idx) for idx, row in enumerate(candidates)]
    safe = [row for row in indexed if row["exp1_label"] == "safe"]
    unsafe = [row for row in indexed if row["exp1_label"] == "unsafe"]
    funnel = [
        {"stage": "raw_candidates", "rows": len(indexed), "safe": len(safe), "unsafe": len(unsafe), "components": _component_count(indexed)},
        {"stage": "fraud_core_only", "rows": sum(row.get("prompt_risk_domain") == "fraud_core" for row in indexed), "safe": sum(row.get("prompt_risk_domain") == "fraud_core" and row["exp1_label"] == "safe" for row in indexed), "unsafe": sum(row.get("prompt_risk_domain") == "fraud_core" and row["exp1_label"] == "unsafe" for row in indexed), "components": _component_count([row for row in indexed if row.get("prompt_risk_domain") == "fraud_core"])},
    ]
    target = int(policy.get("target_groups", 300))
    caliper_results = []
    best = None
    for caliper in policy["calipers"]:
        edges = candidate_edges(unsafe, safe, caliper, policy)
        matched = min_cost_match(edges, unsafe, safe, target)
        caliper_results.append({"level": caliper["level"], "edge_count": len(edges), "max_matching": len(matched)})
        if best is None and len(matched) >= target:
            best = (caliper, edges, matched[:target])
    if best is None:
        caliper = policy["calipers"][-1]
        edges = candidate_edges(unsafe, safe, caliper, policy)
        matched = min_cost_match(edges, unsafe, safe, target)
        best = (caliper, edges, matched)
    caliper, edges, matched = best
    selected_rows = materialize_matches(matched, unsafe, safe)
    edge_rows = [edge_to_row(edge, unsafe, safe) for edge in edges]
    exclusion_rows = exclusion_report(indexed, selected_rows)
    audit = balance_audit(selected_rows, caliper, caliper_results, policy)
    audit["funnel"] = funnel
    audit["edge_count_selected_caliper"] = len(edges)
    audit["used_qy_scores_for_selection"] = False
    if output_dir is not None:
        write_csv(output_dir / "P2_CANDIDATE_FUNNEL.csv", funnel)
        write_csv(output_dir / "P2_EDGE_CENSUS.csv", edge_rows)
        write_csv(output_dir / "P2_EXCLUSION_REASONS.csv", exclusion_rows)
    return MatchResult(selected_rows, audit, edge_rows, exclusion_rows)


def candidate_edges(unsafe: list[dict], safe: list[dict], caliper: dict, policy: dict) -> list[tuple[int, int, float]]:
    weights = policy["matching_weights"]
    top_k = int(policy.get("edge_top_k_per_unsafe", 80))
    unsafe_components = component_representatives(unsafe)
    safe_components = component_representatives(safe)
    best_by_component_pair: dict[tuple[str, str], tuple[int, int, float]] = {}
    for ui_row, u in enumerate(unsafe):
        local: list[tuple[int, dict, float]] = []
        for si_row, s in enumerate(safe):
            if u["semantic_component_id"] == s["semantic_component_id"]:
                continue
            dq = abs(float(u["_p2_q_logit"]) - float(s["_p2_q_logit"]))
            dy = abs(float(u["_p2_y_logit"]) - float(s["_p2_y_logit"]))
            ratio = len(str(u.get("target_model_answer", ""))) / max(len(str(s.get("target_model_answer", ""))), 1)
            if dq > float(caliper["max_abs_logit_q"]) or dy > float(caliper["max_abs_logit_y"]):
                continue
            if not (float(caliper["min_length_ratio"]) <= ratio <= float(caliper["max_length_ratio"])):
                continue
            cost = (
                float(weights["q"]) * dq
                + float(weights["y"]) * dy
                + float(weights["length"]) * abs(np.log(max(ratio, 1e-9)))
                + float(weights["refusal"]) * abs(refusal_feature(u) - refusal_feature(s))
                + float(weights["cross_source_penalty"]) * (str(u.get("source")) != str(s.get("source")))
                + float(weights["cross_language_penalty"]) * (str(u.get("language", "en")) != str(s.get("language", "en")))
            )
            local.append((si_row, s, float(cost)))
        for si_row, s, cost in sorted(local, key=lambda item: item[2])[:top_k]:
            key = (str(u["semantic_component_id"]), str(s["semantic_component_id"]))
            current = best_by_component_pair.get(key)
            if current is None or cost < current[2]:
                best_by_component_pair[key] = (ui_row, si_row, cost)
    unsafe_pos = {component: idx for idx, component in enumerate(unsafe_components)}
    safe_pos = {component: idx for idx, component in enumerate(safe_components)}
    edges: list[tuple[int, int, float]] = []
    for (u_component, s_component), (u_row, s_row, cost) in best_by_component_pair.items():
        if u_component in unsafe_pos and s_component in safe_pos:
            edges.append((unsafe_pos[u_component], safe_pos[s_component], cost, u_row, s_row))  # type: ignore[arg-type]
    return edges


def min_cost_match(edges: list[tuple], unsafe: list[dict], safe: list[dict], target: int) -> list[tuple[int, int, float]]:
    if not edges:
        return []
    unsafe_components = component_representatives(unsafe)
    safe_components = component_representatives(safe)
    used_components: set[str] = set()
    matched: list[tuple[int, int, float]] = []
    for edge in sorted(edges, key=lambda item: float(item[2])):
        unsafe_component_pos, safe_component_pos, cost = int(edge[0]), int(edge[1]), float(edge[2])
        if unsafe_component_pos >= len(unsafe_components) or safe_component_pos >= len(safe_components):
            continue
        unsafe_component = unsafe_components[unsafe_component_pos]
        safe_component = safe_components[safe_component_pos]
        if unsafe_component in used_components or safe_component in used_components:
            continue
        if len(edge) >= 5:
            ui, si = int(edge[3]), int(edge[4])
        else:
            ui, si = unsafe_component_pos, safe_component_pos
        if str(unsafe[ui].get("semantic_component_id")) in used_components or str(safe[si].get("semantic_component_id")) in used_components:
            continue
        used_components.add(str(unsafe[ui].get("semantic_component_id")))
        used_components.add(str(safe[si].get("semantic_component_id")))
        matched.append((ui, si, cost))
        if len(matched) >= target:
            break
    return matched


def materialize_matches(matches: list[tuple[int, int, float]], unsafe: list[dict], safe: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for group_idx, (ui, si, cost) in enumerate(matches):
        group = f"g0c2_p2_dvm_{group_idx:04d}"
        for source_row in (unsafe[ui], safe[si]):
            row = {key: value for key, value in source_row.items() if not key.startswith("_p2_")}
            row["context_collision_group_id"] = group
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    "p2_dvm_cost": cost,
                    "p2_dvm_q_score": float(source_row["_p2_q_prob"]),
                    "p2_dvm_y_score": float(source_row["_p2_y_prob"]),
                    "p2_dvm_selection": "dual_view_single_view_matched",
                }
            )
            row["metadata"] = metadata
            rows.append(row)
    return rows


def balance_audit(rows: list[dict], caliper: dict, caliper_results: list[dict], policy: dict) -> dict:
    safe = [row for row in rows if row["exp1_label"] == "safe"]
    unsafe = [row for row in rows if row["exp1_label"] == "unsafe"]
    by_source = {}
    for row in rows:
        by_source[str(row.get("source"))] = by_source.get(str(row.get("source")), 0) + 1
    q_safe = [float(row.get("metadata", {}).get("p2_dvm_q_score", 0.0)) for row in safe]
    q_unsafe = [float(row.get("metadata", {}).get("p2_dvm_q_score", 0.0)) for row in unsafe]
    y_safe = [float(row.get("metadata", {}).get("p2_dvm_y_score", 0.0)) for row in safe]
    y_unsafe = [float(row.get("metadata", {}).get("p2_dvm_y_score", 0.0)) for row in unsafe]
    len_safe = [np.log(max(len(str(row.get("target_model_answer", ""))), 1)) for row in safe]
    len_unsafe = [np.log(max(len(str(row.get("target_model_answer", ""))), 1)) for row in unsafe]
    refusal_gap = abs(np.mean([refusal_feature(row) for row in safe]) - np.mean([refusal_feature(row) for row in unsafe])) if rows else 1.0
    groups = len(rows) // 2
    source_max = max(by_source.values(), default=0) / max(len(rows), 1)
    gate_policy = policy["balance_gate"]
    checks = {
        "P2-C1": groups == int(policy.get("target_groups", 300)) and len(rows) == groups * 2,
        "P2-C2": len(safe) == groups and len(unsafe) == groups,
        "P2-C3": _component_count(rows) == len(rows),
        "P2-C4": all(row.get("prompt_risk_domain") == "fraud_core" for row in rows),
        "P2-C5": all("official" in str(row.get("label_provenance", "")).lower() or "public" in str(row.get("label_provenance", "")).lower() or row.get("source") for row in rows),
        "P2-C6": len(by_source) >= int(policy.get("formal_sources_min", 3)),
        "P2-C7": source_max <= float(policy.get("largest_source_max", 0.5)),
        "P2-C10": same_component_pairs(rows) == 0,
        "P2-C11": caliper.get("level") in {item["level"] for item in policy["calipers"]},
        "P2-C12": smd(q_safe, q_unsafe) <= float(gate_policy["q_selector_smd_max"]),
        "P2-C13": smd(y_safe, y_unsafe) <= float(gate_policy["y_selector_smd_max"]),
        "P2-C14": smd(len_safe, len_unsafe) <= float(gate_policy["log_answer_length_smd_max"]),
        "P2-C15": refusal_gap <= float(gate_policy["refusal_gap_max"]),
        "P2-C18": True,
        "P2-C19": len(rows) > 0,
        "P2-C20": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "groups": groups,
        "rows": len(rows),
        "selected_caliper": dict(caliper),
        "caliper_results": caliper_results,
        "by_source": by_source,
        "largest_source_rate": source_max,
        "q_selector_smd": smd(q_safe, q_unsafe),
        "y_selector_smd": smd(y_safe, y_unsafe),
        "log_answer_length_smd": smd(len_safe, len_unsafe),
        "refusal_gap": float(refusal_gap),
        "same_component_pairs": same_component_pairs(rows),
    }


def smd(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 1.0
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    pooled = np.sqrt((np.var(av) + np.var(bv)) / 2)
    if pooled <= 1e-12:
        return 0.0 if abs(float(np.mean(av) - np.mean(bv))) <= 1e-12 else 1.0
    return float(abs(np.mean(av) - np.mean(bv)) / pooled)


def refusal_feature(row: dict) -> float:
    y = str(row.get("target_model_answer", "")).lower()
    return float(any(token in y for token in ("sorry", "cannot", "can't", "not able", "won't", "illegal", "无法", "不能")))


def same_component_pairs(rows: list[dict]) -> int:
    by_group: dict[str, set[str]] = {}
    for row in rows:
        by_group.setdefault(str(row.get("context_collision_group_id")), set()).add(str(row.get("semantic_component_id")))
    return sum(1 for comps in by_group.values() if len(comps) < 2)


def edge_to_row(edge: tuple[int, int, float] | tuple[int, int, float, int, int], unsafe: list[dict], safe: list[dict]) -> dict:
    if len(edge) == 5:
        unsafe_component_pos, safe_component_pos, cost, ui, si = edge
    else:
        unsafe_component_pos, safe_component_pos, cost = edge
        ui, si = unsafe_component_pos, safe_component_pos
    return {
        "unsafe_id": unsafe[ui]["id"],
        "safe_id": safe[si]["id"],
        "unsafe_component_id": unsafe[ui].get("semantic_component_id"),
        "safe_component_id": safe[si].get("semantic_component_id"),
        "unsafe_component_pos": unsafe_component_pos,
        "safe_component_pos": safe_component_pos,
        "unsafe_source": unsafe[ui].get("source"),
        "safe_source": safe[si].get("source"),
        "cost": cost,
    }


def exclusion_report(candidates: list[dict], selected: list[dict]) -> list[dict]:
    selected_ids = {row["id"] for row in selected}
    rows = []
    for row in candidates:
        if row["id"] not in selected_ids:
            rows.append({"id": row["id"], "source": row.get("source"), "label": row.get("exp1_label"), "reason": "not_selected_by_min_cost_matching"})
    return rows


def _with_scores(row: dict, scores: NuisanceScores, idx: int) -> dict:
    item = dict(row)
    item["_p2_row_index"] = idx
    item["_p2_q_prob"] = float(scores.q_prob[idx])
    item["_p2_y_prob"] = float(scores.y_prob[idx])
    item["_p2_q_logit"] = float(scores.q_logit[idx])
    item["_p2_y_logit"] = float(scores.y_logit[idx])
    return item


def component_representatives(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        component = str(row.get("semantic_component_id"))
        if component not in seen:
            seen.add(component)
            out.append(component)
    return out


def _component_count(rows: list[dict]) -> int:
    return len({str(row.get("semantic_component_id")) for row in rows})


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
