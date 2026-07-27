from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frauddistill.exp1_ccfa.duplicate_audit import duplicate_audit
from frauddistill.exp1_ccfa.fraud_taxonomy import annotate_risk_type, fraud_primary_rate, load_taxonomy
from frauddistill.exp1_ccfa.public_gold import build_p3_v1
from frauddistill.exp1_ccfa.semantic_components import attach_semantic_components, explicit_label_token_audit, leakage_audit
from frauddistill.utils.io import write_jsonl

from scripts.build_exp1_cpu_g0b_manifests import _aegis_rows, _beavertails_rows, _hash, _jsonl_source


CONFIG_PATH = ROOT / "configs" / "experiments" / "exp1_ccfa_cpu_v5.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E1-CPU-v5 G0c manifests with strict data gates")
    parser.add_argument("--output_dir", default="data/prepared/exp1_cpu_v5/g0c")
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

    p3_rows, p3_audit = build_p3_v1(ROOT / "data" / "raw" / "aegis" / "test.json")
    p3_rows = annotate_all(p3_rows, taxonomy)
    blocked_components = {row["semantic_component_id"] for row in p3_rows}
    blocked_text = text_key_set(p3_rows)
    polyguard_base_ids = {str(row.get("metadata", {}).get("source_base_id")) for row in p3_rows if row.get("source") == "PolyGuardPrompts"}

    aegis_train = annotate_all(_aegis_rows("train", "train"), taxonomy)
    aegis_validation = annotate_all(_aegis_rows("validation", "validation"), taxonomy)
    beaver_train = annotate_all(_beavertails_rows("330k_train", "train"), taxonomy)
    beaver_test = annotate_all(_beavertails_rows("30k_test", "test"), taxonomy)
    beaver_large_test = annotate_all(_beavertails_rows("330k_test", "test"), taxonomy)
    dna = annotate_all(_jsonl_source(ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "do_not_answer_qy.jsonl", "Do-Not-Answer"), taxonomy)
    fraudr1 = annotate_all(_jsonl_source(ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "fraudr1_all_categories_qy.jsonl", "Fraud-R1"), taxonomy)
    aegis_qy = annotate_all(_jsonl_source(ROOT / "data" / "prepared" / "full" / "evaluation_qy" / "aegis_qy.jsonl", "Aegis-extra"), taxonomy)

    train_pool = [*aegis_train, *beaver_train]
    train = select_balanced_unique(
        train_pool,
        safe_count=5000,
        unsafe_count=5000,
        seed=seed,
        blocked_components=blocked_components,
        blocked_text=blocked_text,
        prefer_fraud_primary=True,
    )
    occupied = blocked_components | {row["semantic_component_id"] for row in train}
    occupied_text = blocked_text | text_key_set(train)

    dev_pool = [*aegis_train, *beaver_train]
    model_dev = select_balanced_unique(dev_pool, 500, 500, seed + 1, occupied, occupied_text, prefer_fraud_primary=True)
    occupied |= {row["semantic_component_id"] for row in model_dev}
    occupied_text |= text_key_set(model_dev)
    threshold_dev = select_balanced_unique(dev_pool, 500, 500, seed + 2, occupied, occupied_text, prefer_fraud_primary=True)
    occupied |= {row["semantic_component_id"] for row in threshold_dev}
    occupied_text |= text_key_set(threshold_dev)

    p1_pool = [
        row
        for row in [*beaver_test, *beaver_large_test, *aegis_validation]
        if official_p1_source(row)
        and row["semantic_component_id"] not in occupied
        and not text_key_hit(row, occupied_text)
        and (row["exp1_label"] == "safe" or row.get("gold_risk_type") == "fraud_primary")
    ]
    p1 = select_balanced_unique(p1_pool, 600, 600, seed + 3, occupied, occupied_text, prefer_fraud_primary=True)
    occupied |= {row["semantic_component_id"] for row in p1}
    occupied_text |= text_key_set(p1)

    p2_pool = [
        row
        for row in [*beaver_test, *beaver_large_test, *aegis_validation, *dna, *fraudr1, *aegis_qy]
        if row["semantic_component_id"] not in occupied
        and not text_key_hit(row, occupied_text)
        and "PolyGuard" not in str(row.get("source"))
        and (row["exp1_label"] == "safe" or row.get("gold_risk_type") == "fraud_primary")
    ]
    p2, p2_audit = build_p2_collision(p2_pool, config["data_policy"]["p2_collision"], seed + 4)

    manifests = {
        "g0_train": train,
        "g0_model_dev": model_dev,
        "g0_threshold_dev": threshold_dev,
        "g0_p1_mini": p1,
        "g0_p2_mini": p2,
        "p3_v1": p3_rows,
    }
    for name, rows in manifests.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    audit = audit_manifests(manifests, p3_audit, p2_audit, config, polyguard_base_ids)
    audit["manifest_sha256"] = {f"{name}.jsonl": sha256_file(output_dir / f"{name}.jsonl") for name in manifests}
    audit["git_commit"] = git_commit()
    (output_dir / "g0c_data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def annotate_all(rows: list[dict], taxonomy: dict) -> list[dict]:
    return attach_semantic_components([annotate_risk_type(row, taxonomy) for row in rows])


def select_balanced_unique(
    rows: list[dict],
    safe_count: int,
    unsafe_count: int,
    seed: int,
    blocked_components: set[str],
    blocked_text: set[tuple[str, str]],
    prefer_fraud_primary: bool,
) -> list[dict]:
    selected: list[dict] = []
    used = set(blocked_components)
    local_text = set(blocked_text)
    for label, target in (("unsafe", unsafe_count), ("safe", safe_count)):
        candidates = [
            row
            for row in rows
            if row["exp1_label"] == label and row["semantic_component_id"] not in used and not text_key_hit(row, local_text)
        ]
        if prefer_fraud_primary and label == "unsafe":
            candidates = sorted(
                candidates,
                key=lambda row: (0 if row.get("gold_risk_type") == "fraud_primary" else 1, _hash(f"{seed}:{row['id']}")),
            )
        else:
            candidates = sorted(candidates, key=lambda row: _hash(f"{seed}:{row['id']}"))
        label_rows: list[dict] = []
        for row in candidates:
            component = row["semantic_component_id"]
            if component in used or text_key_hit(row, local_text):
                continue
            item = dict(row)
            metadata = dict(item.get("metadata") or {})
            metadata["g0c_manifest_seed"] = seed
            item["metadata"] = metadata
            selected.append(item)
            label_rows.append(item)
            used.add(component)
            local_text |= text_key_set([item])
            if len(label_rows) >= target:
                break
    return sorted(selected, key=lambda row: _hash(f"{seed}:out:{row['id']}"))


def build_p2_collision(rows: list[dict], policy: dict, seed: int) -> tuple[list[dict], dict]:
    target_groups = int(policy["groups"])
    safe_rows = sorted([row for row in rows if row["exp1_label"] == "safe"], key=lambda row: _hash(f"{seed}:safe:{row['id']}"))[:12000]
    unsafe_rows = sorted(
        [row for row in rows if row["exp1_label"] == "unsafe" and row.get("gold_risk_type") == "fraud_primary"],
        key=lambda row: _hash(f"{seed}:unsafe:{row['id']}"),
    )[:12000]
    if not safe_rows or not unsafe_rows:
        return [], {"passed": False, "reason": "empty safe or fraud-primary unsafe pool", "groups": 0, "rows": 0}
    safe_y, unsafe_y = [r["target_model_answer"] for r in safe_rows], [r["target_model_answer"] for r in unsafe_rows]
    safe_q, unsafe_q = [r["user_query"] for r in safe_rows], [r["user_query"] for r in unsafe_rows]
    y_sim = hybrid_similarity(unsafe_y, safe_y)
    q_sim = hybrid_similarity(unsafe_q, safe_q)
    selected, stats = select_p2_pairs(unsafe_rows, safe_rows, y_sim, q_sim, policy, target_groups)
    backend = "word_char_tfidf"
    if len(stats["similarities_y"]) < target_groups:
        y_sim = frozen_minilm_similarity(unsafe_y, safe_y)
        q_sim = frozen_minilm_similarity(unsafe_q, safe_q)
        selected, stats = select_p2_pairs(unsafe_rows, safe_rows, y_sim, q_sim, policy, target_groups)
        backend = "frozen_minilm"
    audit = p2_audit(stats["similarities_y"], stats["similarities_q"], stats["length_ratios"], stats["mutual_count"], target_groups, policy)
    audit["mining_backend"] = backend
    return selected, audit


def select_p2_pairs(
    unsafe_rows: list[dict],
    safe_rows: list[dict],
    y_sim: np.ndarray,
    q_sim: np.ndarray,
    policy: dict,
    target_groups: int,
) -> tuple[list[dict], dict]:
    forward = top_neighbors(y_sim, n_neighbors=min(50, len(safe_rows)))
    reverse = top_neighbors(y_sim.T, n_neighbors=1)
    selected: list[dict] = []
    used_components: set[str] = set()
    similarities_y: list[float] = []
    similarities_q: list[float] = []
    length_ratios: list[float] = []
    mutual_count = 0
    group_index = 0
    order = np.argsort(-np.max(y_sim, axis=1))
    for unsafe_idx in tqdm(order, desc="mine P2 collisions", leave=False):
        unsafe = unsafe_rows[unsafe_idx]
        if unsafe["semantic_component_id"] in used_components:
            continue
        for safe_idx in forward[unsafe_idx]:
            safe = safe_rows[int(safe_idx)]
            if safe["semantic_component_id"] in used_components:
                continue
            sim_y = float(y_sim[unsafe_idx, int(safe_idx)])
            sim_q = float(q_sim[unsafe_idx, int(safe_idx)])
            ratio = len(unsafe["target_model_answer"]) / max(len(safe["target_model_answer"]), 1)
            if sim_y < float(policy["y_similarity_min_min"]):
                continue
            if sim_q > float(policy["q_similarity_p90_max"]):
                continue
            if not (float(policy["length_ratio_min"]) <= ratio <= float(policy["length_ratio_max"])):
                continue
            is_mutual = bool(int(reverse[int(safe_idx)][0]) == int(unsafe_idx))
            if not is_mutual:
                continue
            group_id = f"g0c_p2_collision_{group_index:04d}"
            for row in (unsafe, safe):
                item = dict(row)
                item["context_collision_group_id"] = group_id
                metadata = dict(item.get("metadata") or {})
                metadata.update(
                    {
                        "p2_y_similarity": sim_y,
                        "p2_q_similarity": sim_q,
                        "p2_answer_length_ratio": ratio,
                        "p2_mutual_nearest": is_mutual,
                    }
                )
                item["metadata"] = metadata
                selected.append(item)
            used_components.add(unsafe["semantic_component_id"])
            used_components.add(safe["semantic_component_id"])
            similarities_y.append(sim_y)
            similarities_q.append(sim_q)
            length_ratios.append(ratio)
            mutual_count += int(is_mutual)
            group_index += 1
            break
        if group_index >= target_groups:
            break
    selected = [dict(row) for row in selected]
    stats = {
        "similarities_y": similarities_y,
        "similarities_q": similarities_q,
        "length_ratios": length_ratios,
        "mutual_count": mutual_count,
    }
    return selected, stats


def hybrid_similarity(left: list[str], right: list[str]) -> np.ndarray:
    word = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=1, sublinear_tf=True, norm="l2")
    char = TfidfVectorizer(analyzer="char_wb", max_features=40000, ngram_range=(3, 5), min_df=1, sublinear_tf=True, norm="l2")
    word_matrix = word.fit_transform([*left, *right])
    char_matrix = char.fit_transform([*left, *right])
    left_word, right_word = word_matrix[: len(left)], word_matrix[len(left) :]
    left_char, right_char = char_matrix[: len(left)], char_matrix[len(left) :]
    sim = 0.55 * (left_word @ right_word.T) + 0.45 * (left_char @ right_char.T)
    if sparse.issparse(sim):
        sim = sim.toarray()
    return np.asarray(sim, dtype=np.float32)


def frozen_minilm_similarity(left: list[str], right: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    left_emb = encode_minilm(left, tokenizer, model, device)
    right_emb = encode_minilm(right, tokenizer, model, device)
    return np.asarray(left_emb @ right_emb.T, dtype=np.float32)


def encode_minilm(texts: list[str], tokenizer, model, device, batch_size: int = 128) -> np.ndarray:
    import torch
    from tqdm import tqdm

    cache_dir = ROOT / "data" / "cache" / "exp1_cpu_v5_g0c_minilm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{hash_texts(texts)}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    vectors = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="encode MiniLM", leave=False):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
            pooled = torch.sum(output.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
    result = np.vstack(vectors)
    np.save(cache_path, result)
    return result


def hash_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(str(text).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def top_neighbors(sim: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_neighbors = max(1, min(n_neighbors, sim.shape[1]))
    if n_neighbors == sim.shape[1]:
        return np.argsort(-sim, axis=1)
    indices = np.argpartition(-sim, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
    scores = np.take_along_axis(sim, indices, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(indices, order, axis=1)


def p2_audit(y: list[float], q: list[float], ratios: list[float], mutual_count: int, target_groups: int, policy: dict) -> dict:
    arr_y = np.asarray(y, dtype=float)
    arr_q = np.asarray(q, dtype=float)
    groups = len(y)
    result = {
        "groups": groups,
        "rows": groups * 2,
        "target_groups": target_groups,
        "mixed_label_rate": 1.0 if groups else 0.0,
        "unique_component_rate": 1.0 if groups else 0.0,
        "mean_y_similarity": float(np.mean(arr_y)) if groups else 0.0,
        "median_y_similarity": float(np.median(arr_y)) if groups else 0.0,
        "p10_y_similarity": float(np.quantile(arr_y, 0.10)) if groups else 0.0,
        "min_y_similarity": float(np.min(arr_y)) if groups else 0.0,
        "mean_q_similarity": float(np.mean(arr_q)) if groups else 0.0,
        "p90_q_similarity": float(np.quantile(arr_q, 0.90)) if groups else 0.0,
        "mutual_nearest_rate": mutual_count / max(groups, 1),
        "answer_length_ratio_min": float(min(ratios)) if ratios else 0.0,
        "answer_length_ratio_max": float(max(ratios)) if ratios else 0.0,
    }
    checks = {
        "groups_exact": groups == target_groups,
        "rows_exact": result["rows"] == int(policy["rows"]),
        "mean_y_ge_gate": result["mean_y_similarity"] >= float(policy["y_similarity_mean_min"]),
        "median_y_ge_gate": result["median_y_similarity"] >= float(policy["y_similarity_median_min"]),
        "p10_y_ge_gate": result["p10_y_similarity"] >= float(policy["y_similarity_p10_min"]),
        "min_y_ge_gate": result["min_y_similarity"] >= float(policy["y_similarity_min_min"]),
        "mean_q_le_gate": result["mean_q_similarity"] <= float(policy["q_similarity_mean_max"]),
        "p90_q_le_gate": result["p90_q_similarity"] <= float(policy["q_similarity_p90_max"]),
        "mutual_nearest_rate_ge_gate": result["mutual_nearest_rate"] >= float(policy["mutual_nearest_rate_min"]),
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result


def audit_manifests(manifests: dict[str, list[dict]], p3_audit: dict, p2: dict, config: dict, polyguard_base_ids: set[str]) -> dict:
    duplicate = duplicate_audit(manifests)
    leakage = leakage_audit(manifests)
    label_tokens = {name: explicit_label_token_audit(rows) for name, rows in manifests.items()}
    counts = {name: split_counts(rows) for name, rows in manifests.items()}
    train = manifests["g0_train"]
    p1 = manifests["g0_p1_mini"]
    silver_non_train = sum(
        1
        for name, rows in manifests.items()
        if name != "g0_train"
        for row in rows
        if "silver" in str(row.get("label_provenance") or "").lower()
    )
    p1_bad_sources = [row["id"] for row in p1 if not official_p1_source(row)]
    p1_polyguard = [row["id"] for row in p1 if "PolyGuard" in str(row.get("source"))]
    p1_unknown_split = [row["id"] for row in p1 if not row.get("metadata", {}).get("official_split")]
    train_silver_ratio = counts["g0_train"]["project_silver_rows"] / max(counts["g0_train"]["rows"], 1)
    gate = {
        "D1_train_model_threshold_exact": counts["g0_train"]["rows"] == 10000
        and counts["g0_model_dev"]["rows"] == 1000
        and counts["g0_threshold_dev"]["rows"] == 1000,
        "D2_train_label_balance_5000_5000": counts["g0_train"]["by_label"].get("safe") == 5000
        and counts["g0_train"]["by_label"].get("unsafe") == 5000,
        "D3_p1_1200_rows_components_600_600": counts["g0_p1_mini"]["rows"] == 1200
        and counts["g0_p1_mini"]["components"] == 1200
        and counts["g0_p1_mini"]["by_label"].get("safe") == 600
        and counts["g0_p1_mini"]["by_label"].get("unsafe") == 600,
        "D4_p2_300_groups_600_rows": p2.get("groups") == 300 and p2.get("rows") == 600,
        "D5_p2_y_similarity": p2.get("checks", {}).get("mean_y_ge_gate", False)
        and p2.get("checks", {}).get("p10_y_ge_gate", False),
        "D6_p2_q_similarity": p2.get("checks", {}).get("mean_q_le_gate", False)
        and p2.get("checks", {}).get("p90_q_le_gate", False),
        "D7_leakage_and_near_duplicate_zero": leakage["passed"] and duplicate["passed"],
        "D8_polyguard_quarantine_violation_zero": not p1_polyguard and len(polyguard_base_ids) == p3_audit.get("polyguard_base_ids"),
        "D9_p1_no_missing_or_unknown_official_split": not p1_bad_sources and not p1_unknown_split,
        "D10_project_silver_train_only_lte_5pct": silver_non_train == 0
        and train_silver_ratio <= float(config["data_policy"]["project_silver_train_ratio_max"]),
        "D11_fraud_primary_p1_100_train_ge_80": fraud_primary_rate(p1) >= 1.0
        and fraud_primary_rate(train) >= float(config["data_policy"]["train_fraud_primary_min"]),
        "D12_label_token_leakage_zero": all(item["passed"] for item in label_tokens.values()),
        "P3_public_gold_passed": bool(p3_audit.get("passed")),
    }
    return {
        "passed": all(gate.values()),
        "gate": gate,
        "counts": counts,
        "fraud_primary_rates": {"train": fraud_primary_rate(train), "p1": fraud_primary_rate(p1)},
        "p2_audit": p2,
        "p3_audit": p3_audit,
        "leakage_audit": leakage,
        "duplicate_audit": duplicate,
        "explicit_label_token_audit": label_tokens,
        "p1_bad_source_examples": p1_bad_sources[:20],
        "p1_polyguard_examples": p1_polyguard[:20],
        "p1_unknown_split_examples": p1_unknown_split[:20],
    }


def split_counts(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "components": len({row["semantic_component_id"] for row in rows}),
        "by_label": count_by(rows, "exp1_label"),
        "by_source": count_by(rows, "source"),
        "by_risk_type": count_by(rows, "gold_risk_type"),
        "project_silver_rows": sum(1 for row in rows if "silver" in str(row.get("label_provenance") or "").lower()),
    }


def official_p1_source(row: dict) -> bool:
    split = str(row.get("metadata", {}).get("official_split") or "")
    source = str(row.get("source") or "")
    return ("BeaverTails" in source and split in {"30k_test", "330k_test"}) or ("Aegis" in source and split == "validation")


def text_key_set(rows: list[dict]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        query = norm(row.get("user_query", ""))
        answer = norm(row.get("target_model_answer", ""))
        keys.add(("q", query))
        keys.add(("y", answer))
        keys.add(("pair", f"{query}\n{answer}"))
    return keys


def text_key_hit(row: dict, keys: set[tuple[str, str]]) -> bool:
    query = norm(row.get("user_query", ""))
    answer = norm(row.get("target_model_answer", ""))
    return ("q", query) in keys or ("y", answer) in keys or ("pair", f"{query}\n{answer}") in keys


def count_by(rows: list[dict], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field))
        result[value] = result.get(value, 0) + 1
    return result


def norm(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
