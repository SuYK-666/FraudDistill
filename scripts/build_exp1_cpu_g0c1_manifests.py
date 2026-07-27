from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frauddistill.exp1_ccfa.duplicate_audit import duplicate_audit
from frauddistill.exp1_ccfa.fraud_taxonomy import annotate_risk_type, load_taxonomy
from frauddistill.exp1_ccfa.p2_calibration import calibrate_p2, p2_calibration_passed
from frauddistill.exp1_ccfa.public_gold import build_p3_v1
from frauddistill.exp1_ccfa.saferlhf_public import saferlhf_rows
from frauddistill.exp1_ccfa.semantic_components import explicit_label_token_audit, leakage_audit
from frauddistill.utils.io import write_jsonl
from scripts.build_exp1_cpu_g0c_manifests import (
    _aegis_rows,
    _beavertails_rows,
    _hash,
    build_p2_collision,
    official_p1_source,
    sha256_file,
    text_key_hit,
    text_key_set,
)


CONFIG_PATH = ROOT / "configs" / "experiments" / "exp1_ccfa_cpu_v5.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E1-CPU-v5 G0c1 manifests")
    parser.add_argument("--output_dir", default="data/prepared/exp1_cpu_v5/g0c1")
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    audit = build_manifests(ROOT / args.output_dir, args.seed)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise SystemExit(2)


def build_manifests(output_dir: Path, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    taxonomy = load_taxonomy(ROOT / config["data_policy"]["taxonomy_path"])
    shutil.copy2(CONFIG_PATH, output_dir / "resolved_config.yaml")
    shutil.copy2(ROOT / config["data_policy"]["taxonomy_path"], output_dir / "FRAUD_TAXONOMY_LOCK.yaml")
    p3, p3_audit = build_p3_v1(ROOT / "data" / "raw" / "aegis" / "test.json")
    p3 = annotate_all(p3, taxonomy)
    blocked = {row["semantic_component_id"] for row in p3}
    blocked_text = text_key_set(p3)

    aegis_train = annotate_all(_aegis_rows("train", "train"), taxonomy)
    aegis_val = annotate_all(_aegis_rows("validation", "validation"), taxonomy)
    beaver_train = annotate_all(_beavertails_rows("330k_train", "train"), taxonomy)
    beaver_test = annotate_all(_beavertails_rows("30k_test", "test") + _beavertails_rows("330k_test", "test"), taxonomy)
    safer_train = annotate_all(saferlhf_rows("train", config["data_policy"]["saferlhf"]["revision"]), taxonomy)
    safer_test = annotate_all(saferlhf_rows("test", config["data_policy"]["saferlhf"]["revision"]), taxonomy)

    occupied = set(blocked)
    occupied_text = set(blocked_text)

    p1_pool = [row for row in [*beaver_test, *aegis_val, *safer_test] if row["prompt_risk_domain"] == "fraud_core"]
    p1 = select_balanced_with_source_cap(p1_pool, 600, 600, seed + 3, occupied, occupied_text, max_source_rows=600)
    occupied |= {row["semantic_component_id"] for row in p1}
    occupied_text |= text_key_set(p1)

    dev_pool = [*aegis_train, *beaver_train, *safer_train]
    model_dev = select_balanced(dev_pool, 500, 500, seed + 1, occupied, occupied_text, require_core=True)
    occupied |= {row["semantic_component_id"] for row in model_dev}
    occupied_text |= text_key_set(model_dev)
    threshold_dev = select_balanced(dev_pool, 500, 500, seed + 2, occupied, occupied_text, require_core=True)
    occupied |= {row["semantic_component_id"] for row in threshold_dev}
    occupied_text |= text_key_set(threshold_dev)

    train = select_balanced_global([*aegis_train, *beaver_train, *safer_train], 5000, 5000, seed, occupied, occupied_text, source_cap=5000)
    occupied |= {row["semantic_component_id"] for row in train}
    occupied_text |= text_key_set(train)

    p2_pool = [
        row
        for row in [*beaver_test, *aegis_val, *safer_test]
        if row["prompt_risk_domain"] == "fraud_core" and row["semantic_component_id"] not in occupied and not text_key_hit(row, occupied_text)
    ]
    p2_policy = dict(config["data_policy"]["p2_collision"])
    p2_policy["y_similarity_min_min"] = 0.30
    p2_policy["q_similarity_p90_max"] = 0.75
    p2, p2_similarity = build_p2_collision(p2_pool, p2_policy, seed + 4)
    calibration = calibrate_p2(p2, p2_pool, int(config["data_policy"]["p2_calibration"]["controls_per_group"]), seed + 5, output_dir)
    calibration_gate = p2_calibration_passed(calibration, config["data_policy"]["p2_calibration"])
    p2_audit = {**p2_similarity, "calibration": calibration, "calibration_gate": calibration_gate}
    manifests = {
        "g0_train": train,
        "g0_model_dev": model_dev,
        "g0_threshold_dev": threshold_dev,
        "g0_p1_mini": p1,
        "g0_p2_mini": p2,
        "p3_v1": p3,
    }
    for name, rows in manifests.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    audit = audit_manifests(manifests, p3_audit, p2_audit, config)
    audit["manifest_sha256"] = {f"{name}.jsonl": sha256_file(output_dir / f"{name}.jsonl") for name in manifests}
    (output_dir / "G0c1_DATA_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "G0c1_P2_CALIBRATION.json").write_text(json.dumps(p2_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_counts_csv(output_dir, manifests)
    return audit


def annotate_all(rows: list[dict], taxonomy: dict) -> list[dict]:
    return [annotate_risk_type(row, taxonomy) for row in rows]


def select_balanced_global(rows: list[dict], safe: int, unsafe: int, seed: int, blocked: set[str], blocked_text: set[tuple[str, str]], source_cap: int) -> list[dict]:
    selected = []
    source_counts: dict[str, int] = {}
    for label, count in (("safe", safe), ("unsafe", unsafe)):
        label_count = 0
        core = [row for row in rows if row["exp1_label"] == label and row.get("prompt_risk_domain") == "fraud_core" and row["semantic_component_id"] not in blocked and not text_key_hit(row, blocked_text)]
        fill = [row for row in rows if row["exp1_label"] == label and row.get("prompt_risk_domain") != "fraud_core" and row["semantic_component_id"] not in blocked and not text_key_hit(row, blocked_text)]
        candidates = [*core, *fill]
        for row in sorted(candidates, key=lambda item: (0 if item.get("prompt_risk_domain") == "fraud_core" else 1, _hash(f"{seed}:global:{label}:{item['source']}:{item['id']}"))):
            source = str(row.get("source"))
            if source_counts.get(source, 0) >= source_cap:
                continue
            selected.append(row)
            source_counts[source] = source_counts.get(source, 0) + 1
            label_count += 1
            if label_count >= count:
                break
    return sorted(selected, key=lambda item: _hash(f"{seed}:global_out:{item['id']}"))


def select_balanced(rows: list[dict], safe: int, unsafe: int, seed: int, blocked: set[str], blocked_text: set[tuple[str, str]], require_core: bool) -> list[dict]:
    result = []
    for label, count in (("safe", safe), ("unsafe", unsafe)):
        candidates = [
            row
            for row in rows
            if row["exp1_label"] == label
            and (not require_core or row.get("prompt_risk_domain") == "fraud_core")
            and row["semantic_component_id"] not in blocked
            and not text_key_hit(row, blocked_text)
        ]
        for row in sorted(candidates, key=lambda item: _hash(f"{seed}:{label}:{item['id']}")):
            result.append(row)
            if sum(1 for item in result if item["exp1_label"] == label) >= count:
                break
    return sorted(result, key=lambda item: _hash(f"{seed}:out:{item['id']}"))


def select_balanced_with_source_cap(rows: list[dict], safe: int, unsafe: int, seed: int, blocked: set[str], blocked_text: set[tuple[str, str]], max_source_rows: int) -> list[dict]:
    result = []
    by_source: dict[str, int] = {}
    used = set(blocked)
    keys = set(blocked_text)
    for label, count in (("safe", safe), ("unsafe", unsafe)):
        label_count = 0
        candidates = [row for row in rows if row["exp1_label"] == label and row["semantic_component_id"] not in used and not text_key_hit(row, keys)]
        for row in sorted(candidates, key=lambda item: _hash(f"{seed}:{label}:{item['source']}:{item['id']}")):
            source = str(row.get("source"))
            if row["semantic_component_id"] in used or text_key_hit(row, keys):
                continue
            if by_source.get(source, 0) >= max_source_rows:
                continue
            result.append(row)
            by_source[source] = by_source.get(source, 0) + 1
            used.add(row["semantic_component_id"])
            keys |= text_key_set([row])
            label_count += 1
            if label_count >= count:
                break
    return result


def rebalance(rows: list[dict], safe: int, unsafe: int, seed: int) -> list[dict]:
    safe_rows = [row for row in rows if row["exp1_label"] == "safe"][:safe]
    unsafe_rows = [row for row in rows if row["exp1_label"] == "unsafe"][:unsafe]
    return sorted([*safe_rows, *unsafe_rows], key=lambda item: _hash(f"{seed}:rebalance:{item['id']}"))


def audit_manifests(manifests: dict[str, list[dict]], p3_audit: dict, p2: dict, config: dict) -> dict:
    counts = {name: split_counts(rows) for name, rows in manifests.items()}
    leakage = leakage_audit(manifests)
    duplicate = duplicate_audit(manifests)
    label_tokens = {name: explicit_label_token_audit(rows) for name, rows in manifests.items()}
    train = manifests["g0_train"]
    p1 = manifests["g0_p1_mini"]
    p2_rows = manifests["g0_p2_mini"]
    gate = {
        "train_model_threshold_exact": counts["g0_train"]["rows"] == 10000 and counts["g0_model_dev"]["rows"] == 1000 and counts["g0_threshold_dev"]["rows"] == 1000,
        "train_balance": counts["g0_train"]["by_label"].get("safe") == 5000 and counts["g0_train"]["by_label"].get("unsafe") == 5000,
        "train_fraud_core_safe_unsafe": counts["g0_train"]["fraud_core_safe"] >= 2500 and counts["g0_train"]["fraud_core_unsafe"] >= 2500,
        "train_source_max_lte_50pct": max(counts["g0_train"]["by_source"].values(), default=0) <= 5000,
        "p1_1200_600_600": counts["g0_p1_mini"]["rows"] == 1200 and counts["g0_p1_mini"]["components"] == 1200 and counts["g0_p1_mini"]["by_label"].get("safe") == 600 and counts["g0_p1_mini"]["by_label"].get("unsafe") == 600,
        "p1_fraud_core_100pct": all(row.get("prompt_risk_domain") == "fraud_core" for row in p1),
        "p1_source_count_ge_3": len(counts["g0_p1_mini"]["by_source"]) >= 3,
        "p1_source_max_lte_50pct": max(counts["g0_p1_mini"]["by_source"].values(), default=0) <= 600,
        "p2_c1_groups_rows": p2.get("groups") == 300 and p2.get("rows") == 600,
        "p2_c4_fraud_core": all(row.get("prompt_risk_domain") == "fraud_core" for row in p2_rows),
        "p2_c12_length_ratio": p2.get("answer_length_ratio_min", 0) >= 0.67 and p2.get("answer_length_ratio_max", 99) <= 1.50,
        "p2_c13_unique_components": counts["g0_p2_mini"]["components"] == 600,
        "p2_calibration_passed": bool(p2.get("calibration_gate", {}).get("passed")),
        "p3_passed": bool(p3_audit.get("passed")),
        "overlap_duplicate_zero": leakage["passed"] and duplicate["passed"],
        "label_token_zero": all(item["passed"] for item in label_tokens.values()),
    }
    return {"passed": all(gate.values()), "gate": gate, "counts": counts, "p2_audit": p2, "p3_audit": p3_audit, "leakage_audit": leakage, "duplicate_audit": duplicate}


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
            domains = sorted({str(row.get("prompt_risk_domain")) for row in rows})
            for domain in domains:
                writer.writerow({"split": split, "prompt_risk_domain": domain, "count": sum(1 for row in rows if row.get("prompt_risk_domain") == domain)})


if __name__ == "__main__":
    main()
