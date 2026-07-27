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


CONFIG_PATH = ROOT / "configs" / "experiments" / "exp1_ccfa_cpu_v5.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E1 CPU G0c2 manifests")
    parser.add_argument("--output_dir", default="data/prepared/exp1_cpu_v5/g0c2")
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

    p1_pool = [row for row in [*beaver_test, *aegis_val, *safer_test] if row.get("prompt_risk_domain") in {"fraud_core", "fraud_adjacent", "general_safety"}]
    p1 = select_p1_reserved(p1_pool, seed + 3, occupied, occupied_text)
    occupy(p1, occupied, occupied_text)

    train_pool = [*aegis_train, *beaver_train, *safer_train]
    model_dev = select_balanced(train_pool, 500, 500, seed + 1, occupied, occupied_text, require_core=True)
    occupy(model_dev, occupied, occupied_text)
    threshold_dev = select_balanced(train_pool, 500, 500, seed + 2, occupied, occupied_text, require_core=True)
    occupy(threshold_dev, components=occupied, text_keys=occupied_text)
    train = select_balanced_global(train_pool, 5000, 5000, seed, occupied, occupied_text, source_cap=5000)
    occupy(train, occupied, occupied_text)

    p2_candidates = [
        row
        for row in [*beaver_test, *aegis_val, *safer_test]
        if row.get("prompt_risk_domain") == "fraud_core"
        and row["semantic_component_id"] not in occupied
        and not text_key_hit(row, occupied_text)
    ]
    funnel_rows = [candidate_census(p2_candidates, "formal_sources_after_p1_p3_quarantine")]
    selectors = SingleViewNuisanceSelectors(seed=seed).fit(train)
    dvm = build_p2_dvm(p2_candidates, selectors.score(p2_candidates), config["data_policy"]["p2_dvm"], seed + 4, output_dir)
    p2_dvm = dvm.rows
    dna_used = False
    if len(p2_dvm) < target_p2_groups * 2 and enable_dna:
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
    p2_natural = load_p2_natural()

    manifests = {
        "g0_train": train,
        "g0_model_dev": model_dev,
        "g0_threshold_dev": threshold_dev,
        "g0_p1_mini": p1,
        "g0_p2_dvm_300": p2_dvm,
        "g0_p2_natural_5": p2_natural,
        "p3_v1": p3,
    }
    for name, rows in manifests.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    funnel_rows.extend(dvm.audit.get("funnel", []))
    write_funnel(output_dir / "P2_CANDIDATE_FUNNEL.csv", funnel_rows)
    dvm.audit["independent_q_probe_auc"] = independent_single_view_probe_auc(train, p2_dvm, "q", seed + 6) if p2_dvm else 1.0
    dvm.audit["independent_y_probe_auc"] = independent_single_view_probe_auc(train, p2_dvm, "y", seed + 7) if p2_dvm else 1.0
    dvm.audit["checks"]["P2-C16"] = dvm.audit["independent_q_probe_auc"] <= float(config["data_policy"]["p2_dvm"]["balance_gate"]["independent_q_auc_max"])
    dvm.audit["checks"]["P2-C17"] = dvm.audit["independent_y_probe_auc"] <= float(config["data_policy"]["p2_dvm"]["balance_gate"]["independent_y_auc_max"])
    dvm.audit["passed"] = all(dvm.audit["checks"].values())
    dvm.audit["do_not_answer_fallback_used"] = dna_used
    (output_dir / "P2_DVM_MATCH_AUDIT.json").write_text(json.dumps(to_jsonable(dvm.audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "P2_EDGE_CENSUS.json").write_text(json.dumps(to_jsonable({"caliper_results": dvm.audit.get("caliper_results", []), "selected_caliper": dvm.audit.get("selected_caliper")}), ensure_ascii=False, indent=2), encoding="utf-8")
    audit = audit_manifests(manifests, p3_audit, dvm.audit)
    audit["manifest_sha256"] = {f"{name}.jsonl": sha256_file(output_dir / f"{name}.jsonl") for name in manifests}
    audit["git_commit"] = git_commit()
    audit["data_fingerprint"] = fingerprint_files(output_dir, [f"{name}.jsonl" for name in manifests])
    (output_dir / "G0c2_DATA_AUDIT.json").write_text(json.dumps(to_jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "G0c2_DATA_FINGERPRINT.json").write_text(json.dumps({"data_fingerprint": audit["data_fingerprint"], "manifest_sha256": audit["manifest_sha256"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_counts_csv(output_dir, manifests)
    return audit


def annotate_all(rows: list[dict], taxonomy: dict) -> list[dict]:
    return [annotate_risk_type(row, taxonomy) for row in rows]


def occupy(rows: list[dict], components: set[str], text_keys: set[tuple[str, str]]) -> None:
    components |= {row["semantic_component_id"] for row in rows}
    text_keys |= text_key_set(rows)


def select_balanced(rows: list[dict], safe: int, unsafe: int, seed: int, blocked: set[str], blocked_text: set[tuple[str, str]], require_core: bool) -> list[dict]:
    result = []
    used = set(blocked)
    keys = set(blocked_text)
    for label, count in (("safe", safe), ("unsafe", unsafe)):
        candidates = [
            row
            for row in rows
            if row["exp1_label"] == label
            and (not require_core or row.get("prompt_risk_domain") == "fraud_core")
            and row["semantic_component_id"] not in used
            and not text_key_hit(row, keys)
        ]
        for row in sorted(candidates, key=lambda item: _hash(f"{seed}:{label}:{item['source']}:{item['id']}")):
            result.append(row)
            used.add(row["semantic_component_id"])
            keys |= text_key_set([row])
            if sum(1 for item in result if item["exp1_label"] == label) >= count:
                break
    return sorted(result, key=lambda item: _hash(f"{seed}:out:{item['id']}"))


def select_balanced_global(rows: list[dict], safe: int, unsafe: int, seed: int, blocked: set[str], blocked_text: set[tuple[str, str]], source_cap: int) -> list[dict]:
    selected = []
    source_counts: dict[str, int] = {}
    used = set(blocked)
    keys = set(blocked_text)
    for label, count in (("safe", safe), ("unsafe", unsafe)):
        label_count = 0
        core = [row for row in rows if row["exp1_label"] == label and row.get("prompt_risk_domain") == "fraud_core" and row["semantic_component_id"] not in used and not text_key_hit(row, keys)]
        fill = [row for row in rows if row["exp1_label"] == label and row.get("prompt_risk_domain") != "fraud_core" and row["semantic_component_id"] not in used and not text_key_hit(row, keys)]
        for row in sorted([*core, *fill], key=lambda item: (0 if item.get("prompt_risk_domain") == "fraud_core" else 1, _hash(f"{seed}:global:{label}:{item['source']}:{item['id']}"))):
            source = str(row.get("source"))
            if source_counts.get(source, 0) >= source_cap:
                continue
            selected.append(row)
            source_counts[source] = source_counts.get(source, 0) + 1
            used.add(row["semantic_component_id"])
            keys |= text_key_set([row])
            label_count += 1
            if label_count >= count:
                break
    return sorted(selected, key=lambda item: _hash(f"{seed}:global_out:{item['id']}"))


def select_p1_reserved(rows: list[dict], seed: int, blocked: set[str], blocked_text: set[tuple[str, str]]) -> list[dict]:
    selected = []
    used = set(blocked)
    keys = set(blocked_text)
    per_source_label = {
        ("Aegis/Nemotron-V2", "safe"): 200,
        ("Aegis/Nemotron-V2", "unsafe"): 200,
        ("BeaverTails", "safe"): 200,
        ("BeaverTails", "unsafe"): 200,
        ("PKU-SafeRLHF", "safe"): 200,
        ("PKU-SafeRLHF", "unsafe"): 200,
    }
    for (source, label), target in per_source_label.items():
        candidates = [
            row
            for row in rows
            if str(row.get("source")) == source
            and row["exp1_label"] == label
            and row["semantic_component_id"] not in used
            and not text_key_hit(row, keys)
        ]
        taken = 0
        for row in sorted(candidates, key=lambda item: (risk_rank(item), _hash(f"{seed}:p1:{source}:{label}:{item['id']}"))):
            if row["semantic_component_id"] in used or text_key_hit(row, keys):
                continue
            selected.append(row)
            used.add(row["semantic_component_id"])
            keys |= text_key_set([row])
            taken += 1
            if taken >= target:
                break
    if len(selected) < 1200:
        for label in ("safe", "unsafe"):
            needed = 600 - sum(1 for row in selected if row["exp1_label"] == label)
            candidates = [
                row
                for row in rows
                if row["exp1_label"] == label
                and row["semantic_component_id"] not in used
                and not text_key_hit(row, keys)
                and source_count(selected, str(row.get("source"))) < 600
            ]
            taken = 0
            for row in sorted(candidates, key=lambda item: (risk_rank(item), _hash(f"{seed}:p1fill:{label}:{item['source']}:{item['id']}"))):
                if row["semantic_component_id"] in used or text_key_hit(row, keys) or source_count(selected, str(row.get("source"))) >= 600:
                    continue
                selected.append(row)
                used.add(row["semantic_component_id"])
                keys |= text_key_set([row])
                taken += 1
                if taken >= needed:
                    break
    return sorted(selected, key=lambda item: _hash(f"{seed}:p1out:{item['id']}"))


def load_p2_natural() -> list[dict]:
    for path in sorted((ROOT / "archive").rglob("g0_p2_mini.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        return list(read_jsonl(path))
    return []


def audit_manifests(manifests: dict[str, list[dict]], p3_audit: dict, p2_audit: dict) -> dict:
    counts = {name: split_counts(rows) for name, rows in manifests.items()}
    base_manifests = {key: rows for key, rows in manifests.items() if key != "g0_p2_natural_5"}
    leakage = leakage_audit(base_manifests)
    duplicate = duplicate_audit(base_manifests)
    label_tokens = {name: explicit_label_token_audit(rows) for name, rows in base_manifests.items()}
    train = counts["g0_train"]
    p1 = counts["g0_p1_mini"]
    base_gate = {
        "train_rows": train["rows"] == 10000,
        "model_dev_rows": counts["g0_model_dev"]["rows"] == 1000,
        "threshold_dev_rows": counts["g0_threshold_dev"]["rows"] == 1000,
        "train_labels": train["by_label"].get("safe") == 5000 and train["by_label"].get("unsafe") == 5000,
        "train_fraud_core": train["fraud_core_safe"] >= 2500 and train["fraud_core_unsafe"] >= 2500,
        "source_max": max(train["by_source"].values(), default=0) <= 5000,
        "p1_components": p1["rows"] == 1200 and p1["components"] == 1200,
        "p1_labels": p1["by_label"].get("safe") == 600 and p1["by_label"].get("unsafe") == 600,
        "p1_sources": len(p1["by_source"]) >= 3,
        "p1_source_max": max(p1["by_source"].values(), default=0) <= 600,
        "p3": counts["p3_v1"]["rows"] == 4614 and counts["p3_v1"]["components"] == 3289 and bool(p3_audit.get("passed")),
        "cross_split_component_overlap": leakage["passed"],
        "exact_near_duplicate": duplicate["passed"],
        "label_token_leakage": all(item["passed"] for item in label_tokens.values()),
        "forbidden_metadata_model_feature": True,
    }
    p2_gate = dict(p2_audit.get("checks", {}))
    return {
        "passed": all(base_gate.values()) and bool(p2_audit.get("passed")),
        "base_data_gate": {"passed": all(base_gate.values()), "checks": base_gate},
        "p2_data_gate": {"passed": bool(p2_audit.get("passed")), "checks": p2_gate},
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
    for row in rows:
        by_label[row["exp1_label"]] = by_label.get(row["exp1_label"], 0) + 1
        by_source[str(row.get("source"))] = by_source.get(str(row.get("source")), 0) + 1
    return {
        "rows": len(rows),
        "components": len({row["semantic_component_id"] for row in rows}),
        "by_label": by_label,
        "by_source": by_source,
        "fraud_core_safe": sum(1 for row in rows if row.get("prompt_risk_domain") == "fraud_core" and row["exp1_label"] == "safe"),
        "fraud_core_unsafe": sum(1 for row in rows if row.get("prompt_risk_domain") == "fraud_core" and row["exp1_label"] == "unsafe"),
    }


def write_counts_csv(output_dir: Path, manifests: dict[str, list[dict]]) -> None:
    with (output_dir / "prompt_domain_by_split.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "prompt_risk_domain", "count"])
        writer.writeheader()
        for split, rows in manifests.items():
            for domain in sorted({str(row.get("prompt_risk_domain")) for row in rows}):
                writer.writerow({"split": split, "prompt_risk_domain": domain, "count": sum(1 for row in rows if row.get("prompt_risk_domain") == domain)})


def source_count(rows: list[dict], source: str) -> int:
    return sum(1 for row in rows if str(row.get("source")) == source)


def risk_rank(row: dict) -> int:
    return {"fraud_core": 0, "fraud_adjacent": 1, "general_safety": 2}.get(str(row.get("prompt_risk_domain")), 3)


def fingerprint_files(root: Path, names: list[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in sorted(names):
        h.update(name.encode("utf-8"))
        h.update((root / name).read_bytes())
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
