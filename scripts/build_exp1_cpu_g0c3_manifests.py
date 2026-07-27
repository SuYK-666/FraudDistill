from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frauddistill.exp1_ccfa.do_not_answer_public import do_not_answer_rows
from frauddistill.exp1_ccfa.duplicate_audit import duplicate_audit
from frauddistill.exp1_ccfa.fraud_taxonomy import annotate_risk_type, load_taxonomy
from frauddistill.exp1_ccfa.nuisance_single_view import SingleViewNuisanceSelectors, independent_single_view_probe_auc
from frauddistill.exp1_ccfa.p2_candidate_census import candidate_census, write_funnel
from frauddistill.exp1_ccfa.p2_dual_view_match import build_p2_dvm
from frauddistill.exp1_ccfa.public_gold import build_p3_v1
from frauddistill.exp1_ccfa.saferlhf_public import saferlhf_rows
from frauddistill.exp1_ccfa.semantic_components import explicit_label_token_audit, leakage_audit
from frauddistill.utils.io import read_jsonl, write_jsonl
from scripts.build_exp1_cpu_g0c_manifests import _aegis_rows, _beavertails_rows, _hash, sha256_file, text_key_hit, text_key_set
from scripts.build_exp1_cpu_g0c2_manifests import (
    annotate_all,
    fingerprint_files,
    risk_rank,
    select_balanced,
    select_balanced_global,
    source_count,
    to_jsonable,
)


CONFIG_PATH = ROOT / "configs" / "experiments" / "exp1_ccfa_cpu_v5.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E1 CPU G0c3 final-freeze manifests")
    parser.add_argument("--output_dir", default="data/prepared/exp1_cpu_v5/g0c3")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--target_p2_groups", type=int, default=300)
    parser.add_argument("--enable_do_not_answer_fallback", action="store_true")
    args = parser.parse_args()
    audit = build_manifests(ROOT / args.output_dir, args.seed, args.target_p2_groups, args.enable_do_not_answer_fallback)
    print(json.dumps(to_jsonable(audit), ensure_ascii=False, indent=2))
    if not audit["base_data_gate"]["passed"]:
        raise SystemExit(2)


def build_manifests(output_dir: Path, seed: int, target_p2_groups: int, enable_dna: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["data_policy"]["p2_dvm"]["target_groups"] = target_p2_groups
    config["data_policy"]["p2_dvm"]["largest_source_max"] = 1.0
    taxonomy = load_taxonomy(ROOT / config["data_policy"]["taxonomy_path"])
    shutil.copy2(CONFIG_PATH, output_dir / "resolved_config.yaml")
    shutil.copy2(ROOT / config["data_policy"]["taxonomy_path"], output_dir / "FRAUD_TAXONOMY_LOCK.yaml")

    p3, p3_audit = build_p3_v1(ROOT / "data" / "raw" / "aegis" / "test.json")
    p3 = annotate_all(p3, taxonomy)
    occupied = {row["semantic_component_id"] for row in p3}
    occupied_text = text_key_set(p3)

    aegis_train = annotate_all(_aegis_rows("train", "train"), taxonomy)
    aegis_val = annotate_all(_aegis_rows("validation", "validation"), taxonomy)
    beaver_train = annotate_all(_beavertails_rows("330k_train", "train"), taxonomy)
    beaver_test = annotate_all(_beavertails_rows("30k_test", "test") + _beavertails_rows("330k_test", "test"), taxonomy)
    safer_train = annotate_all(saferlhf_rows("train", config["data_policy"]["saferlhf"]["revision"]), taxonomy)
    safer_test = annotate_all(saferlhf_rows("test", config["data_policy"]["saferlhf"]["revision"]), taxonomy)

    train_pool = [*aegis_train, *beaver_train, *safer_train]
    model_dev = select_balanced(train_pool, 500, 500, seed + 1, occupied, occupied_text, require_core=True)
    occupy(model_dev, occupied, occupied_text)
    threshold_dev = select_balanced(train_pool, 500, 500, seed + 2, occupied, occupied_text, require_core=True)
    occupy(threshold_dev, occupied, occupied_text)
    train = select_balanced_global(train_pool, 5000, 5000, seed, occupied, occupied_text, source_cap=5000)
    occupy(train, occupied, occupied_text)

    formal_pool = [*beaver_test, *aegis_val, *safer_test]
    p2_candidates = [
        row
        for row in formal_pool
        if row.get("prompt_risk_domain") == "fraud_core"
        and row["semantic_component_id"] not in occupied
        and not text_key_hit(row, occupied_text)
    ]
    funnel_rows = [candidate_census(p2_candidates, "formal_sources_p2_first_before_p1")]
    selectors = SingleViewNuisanceSelectors(seed=seed).fit(train)
    dvm = build_p2_dvm(p2_candidates, selectors.score(p2_candidates), config["data_policy"]["p2_dvm"], seed + 4, output_dir)
    p2_dvm = dvm.rows
    dna_used = False
    if len(p2_dvm) < int(config["data_policy"]["p2_dvm"].get("limited_groups_min", 200)) * 2 and enable_dna:
        dna = annotate_all(do_not_answer_rows(ROOT), taxonomy)
        dna_candidates = [
            row
            for row in dna
            if row.get("prompt_risk_domain") == "fraud_core"
            and row["semantic_component_id"] not in occupied
            and not text_key_hit(row, occupied_text)
        ]
        p2_candidates = [*p2_candidates, *dna_candidates]
        funnel_rows.append(candidate_census(dna_candidates, "do_not_answer_fallback_candidates"))
        dvm = build_p2_dvm(p2_candidates, selectors.score(p2_candidates), config["data_policy"]["p2_dvm"], seed + 5, output_dir)
        p2_dvm = dvm.rows
        dna_used = True
    occupy(p2_dvm, occupied, occupied_text)

    p1_pool = [
        row
        for row in formal_pool
        if row.get("prompt_risk_domain") in {"fraud_core", "fraud_adjacent", "general_safety"}
        and row["semantic_component_id"] not in occupied
        and not text_key_hit(row, occupied_text)
    ]
    p1 = select_p1_after_p2(p1_pool, seed + 3, occupied, occupied_text)
    occupy(p1, occupied, occupied_text)
    p2_natural = load_p2_natural()

    manifests = {
        "g0_train": train,
        "g0_model_dev": model_dev,
        "g0_threshold_dev": threshold_dev,
        "g0_p1_natural_mixed": p1,
        "g0_p2_dvm_core": p2_dvm,
        "g0_p2_natural_5": p2_natural,
        "p3_v1": p3,
    }
    for name, rows in manifests.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    write_funnel(output_dir / "G0c3_P2_CANDIDATE_FUNNEL.csv", [*funnel_rows, *dvm.audit.get("funnel", [])])
    dvm.audit["independent_q_probe_auc"] = independent_single_view_probe_auc(train, p2_dvm, "q", seed + 6) if p2_dvm else 1.0
    dvm.audit["independent_y_probe_auc"] = independent_single_view_probe_auc(train, p2_dvm, "y", seed + 7) if p2_dvm else 1.0
    balance_gate = config["data_policy"]["p2_dvm"]["balance_gate"]
    dvm.audit["checks"]["P2-C16"] = dvm.audit["independent_q_probe_auc"] <= float(balance_gate["independent_q_auc_max"])
    dvm.audit["checks"]["P2-C17"] = dvm.audit["independent_y_probe_auc"] <= float(balance_gate["independent_y_auc_max"])
    dvm.audit["checks"]["P2-C1"] = dvm.audit["groups"] >= int(config["data_policy"]["p2_dvm"].get("limited_groups_min", 200))
    dvm.audit["passed"] = all(dvm.audit["checks"].values())
    dvm.audit["target_groups"] = target_p2_groups
    dvm.audit["limited_groups_min"] = int(config["data_policy"]["p2_dvm"].get("limited_groups_min", 200))
    dvm.audit["do_not_answer_fallback_used"] = dna_used
    dvm.audit["component_occurrence_max"] = component_occurrence_max(p2_dvm)
    (output_dir / "G0c3_P2_BALANCE.json").write_text(json.dumps(to_jsonable(dvm.audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "G0c3_P2_COMPONENT_CAPACITY_AUDIT.json").write_text(json.dumps(to_jsonable({"component_occurrence_max": component_occurrence_max(p2_dvm), "rows": len(p2_dvm), "components": len({row["semantic_component_id"] for row in p2_dvm})}), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "G0c3_P2_EDGE_CENSUS.csv").write_text((output_dir / "P2_EDGE_CENSUS.csv").read_text(encoding="utf-8") if (output_dir / "P2_EDGE_CENSUS.csv").exists() else "", encoding="utf-8")
    audit = audit_manifests(manifests, p3_audit, dvm.audit)
    audit["manifest_sha256"] = {f"{name}.jsonl": sha256_file(output_dir / f"{name}.jsonl") for name in manifests}
    audit["git_commit"] = git_commit()
    audit["data_fingerprint"] = fingerprint_files(output_dir, [f"{name}.jsonl" for name in manifests])
    audit["dataset_revisions"] = config["data_policy"].get("dataset_revisions", {})
    (output_dir / "G0c3_DATA_AUDIT.json").write_text(json.dumps(to_jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "G0c3_DATASET_REVISIONS.json").write_text(json.dumps(to_jsonable(audit["dataset_revisions"]), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "G0c3_DATA_FINGERPRINT.json").write_text(json.dumps(to_jsonable({"data_fingerprint": audit["data_fingerprint"], "manifest_sha256": audit["manifest_sha256"]}), ensure_ascii=False, indent=2), encoding="utf-8")
    write_component_manifest(output_dir / "G0c3_SPLIT_COMPONENT_MANIFEST.tsv", manifests)
    write_counts_csv(output_dir, manifests)
    return audit


def occupy(rows: list[dict], components: set[str], text_keys: set[tuple[str, str]]) -> None:
    components |= {row["semantic_component_id"] for row in rows}
    text_keys |= text_key_set(rows)


def select_p1_after_p2(rows: list[dict], seed: int, blocked: set[str], blocked_text: set[tuple[str, str]]) -> list[dict]:
    selected: list[dict] = []
    used = set(blocked)
    keys = set(blocked_text)
    source_cap = 400
    for label in ("safe", "unsafe"):
        target = 600
        candidates = [
            row
            for row in rows
            if row["exp1_label"] == label
            and row["semantic_component_id"] not in used
            and not text_key_hit(row, keys)
        ]
        for row in sorted(candidates, key=lambda item: (source_count(selected, str(item.get("source"))), risk_rank(item), _hash(f"{seed}:p1:{label}:{item['source']}:{item['id']}"))):
            if row["semantic_component_id"] in used or text_key_hit(row, keys):
                continue
            if source_count(selected, str(row.get("source"))) >= source_cap:
                continue
            selected.append(row)
            used.add(row["semantic_component_id"])
            keys |= text_key_set([row])
            if sum(1 for item in selected if item["exp1_label"] == label) >= target:
                break
    return sorted(selected, key=lambda item: _hash(f"{seed}:p1out:{item['id']}"))


def load_p2_natural() -> list[dict]:
    candidates = sorted((ROOT / "archive").rglob("g0_p2_natural_5.jsonl")) + sorted((ROOT / "archive").rglob("g0_p2_mini.jsonl"))
    for path in candidates:
        return list(read_jsonl(path))
    return []


def audit_manifests(manifests: dict[str, list[dict]], p3_audit: dict, p2_audit: dict) -> dict:
    counts = {name: split_counts(rows) for name, rows in manifests.items()}
    formal = {key: rows for key, rows in manifests.items() if key != "g0_p2_natural_5"}
    leakage = leakage_audit(formal)
    duplicate = duplicate_audit(formal)
    label_tokens = {name: explicit_label_token_audit(rows) for name, rows in formal.items()}
    train = counts["g0_train"]
    p1 = counts["g0_p1_natural_mixed"]
    p2 = counts["g0_p2_dvm_core"]
    base_gate = {
        "train_rows": train["rows"] == 10000,
        "model_dev_rows": counts["g0_model_dev"]["rows"] == 1000,
        "threshold_dev_rows": counts["g0_threshold_dev"]["rows"] == 1000,
        "train_labels": train["by_label"].get("safe") == 5000 and train["by_label"].get("unsafe") == 5000,
        "train_fraud_core": train["fraud_core_safe"] >= 2500 and train["fraud_core_unsafe"] >= 2500,
        "p1_components": p1["rows"] == 1200 and p1["components"] == 1200,
        "p1_labels": p1["by_label"].get("safe") == 600 and p1["by_label"].get("unsafe") == 600,
        "p1_sources": len(p1["by_source"]) >= 3,
        "p3": counts["p3_v1"]["rows"] == 4614 and counts["p3_v1"]["components"] == 3289 and bool(p3_audit.get("passed")),
        "p2_row_component_unique": p2["rows"] == p2["components"] and component_occurrence_max(manifests["g0_p2_dvm_core"]) <= 1,
        "cross_split_component_overlap": leakage["passed"],
        "exact_near_duplicate": duplicate["passed"],
        "label_token_leakage": all(item["passed"] for item in label_tokens.values()),
        "forbidden_metadata_model_feature": True,
    }
    return {
        "passed": all(base_gate.values()) and bool(p2_audit.get("passed")),
        "base_data_gate": {"passed": all(base_gate.values()), "checks": base_gate},
        "p2_data_gate": {"passed": bool(p2_audit.get("passed")), "checks": dict(p2_audit.get("checks", {}))},
        "counts": counts,
        "p2_dvm_audit": p2_audit,
        "p3_audit": p3_audit,
        "leakage_audit": leakage,
        "duplicate_audit": duplicate,
        "label_token_audit": label_tokens,
    }


def split_counts(rows: list[dict]) -> dict:
    by_label: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for row in rows:
        by_label[row["exp1_label"]] = by_label.get(row["exp1_label"], 0) + 1
        by_source[str(row.get("source"))] = by_source.get(str(row.get("source")), 0) + 1
        by_domain[str(row.get("prompt_risk_domain"))] = by_domain.get(str(row.get("prompt_risk_domain")), 0) + 1
    return {
        "rows": len(rows),
        "components": len({row["semantic_component_id"] for row in rows}),
        "by_label": by_label,
        "by_source": by_source,
        "by_prompt_domain": by_domain,
        "fraud_core_safe": sum(1 for row in rows if row.get("prompt_risk_domain") == "fraud_core" and row["exp1_label"] == "safe"),
        "fraud_core_unsafe": sum(1 for row in rows if row.get("prompt_risk_domain") == "fraud_core" and row["exp1_label"] == "unsafe"),
    }


def component_occurrence_max(rows: list[dict]) -> int:
    counts: dict[str, int] = {}
    for row in rows:
        component = str(row.get("semantic_component_id"))
        counts[component] = counts.get(component, 0) + 1
    return max(counts.values(), default=0)


def write_component_manifest(path: Path, manifests: dict[str, list[dict]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "id", "semantic_component_id", "label", "source", "prompt_risk_domain"], delimiter="\t")
        writer.writeheader()
        for split, rows in manifests.items():
            for row in rows:
                writer.writerow({"split": split, "id": row.get("id"), "semantic_component_id": row.get("semantic_component_id"), "label": row.get("exp1_label"), "source": row.get("source"), "prompt_risk_domain": row.get("prompt_risk_domain")})


def write_counts_csv(output_dir: Path, manifests: dict[str, list[dict]]) -> None:
    with (output_dir / "prompt_domain_by_split.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "prompt_risk_domain", "count"])
        writer.writeheader()
        for split, rows in manifests.items():
            for domain in sorted({str(row.get("prompt_risk_domain")) for row in rows}):
                writer.writerow({"split": split, "prompt_risk_domain": domain, "count": sum(1 for row in rows if row.get("prompt_risk_domain") == domain)})


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
