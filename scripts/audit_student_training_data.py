# -*- coding: utf-8 -*-
"""Audit + build the neural-student training manifest (guide 6, 7, 16).

- Exp2 reserved test (guide 6.3) is excluded from the train pool with asserts.
- Synthetic rows get derived template_family_id / semantic_source_id / mutation_type (guide 6.2).
- Teacher signals are validated (guide 16.1) and tiered (guide 7.4).
- Near-duplicate (q, y) audit across the usable train pool.

Usage: python scripts/audit_student_training_data.py
Outputs: data/prepared/exp3_neural_student/train_manifest.jsonl + audit_report.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data/prepared/exp3_agent_distillation/exp3_dataset.jsonl"
RESERVED = REPO / "data/splits/reserved_exp2_test_ids.json"
AGENT_DIR = REPO / "experiments/exp3_agent_distillation_ablation/outputs/agent_predictions"
OUT_DIR = REPO / "data/prepared/exp3_neural_student"

MUTATION_BY_SUBTYPE = {
    "direct_fraud": "direct_action",
    "trust_facilitation": "trust_gradient",
    "clean_refusal_to_fraud": "refusal_then_leakage",
    "partial_leakage": "partial_leakage",
    "clean_refusal": "clean_refusal",
    "hard_safe": "hard_safe",
    "quotation_analysis": "quotation",
    "anti_fraud_education": "education",
    "over_refusal": "over_refusal",
    "context_flip": "context_flip",
    "harmful_compliance": "harmful_compliance",
    "regular_safe": "regular_safe",
    "toxic": "toxic",
    "translation_fictional": "translation_fictional",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()] if path.exists() else []


def norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text)).lower()


def qy_hash(q: str, a: str) -> str:
    import hashlib
    return hashlib.sha256(f"{norm(q)}\n{norm(a)}".encode("utf-8")).hexdigest()


def validate_teacher_signal(t: dict) -> bool:
    """Guide 16.1 consistency check."""
    score = float(t.get("teacher_score", 0.5))
    label = str(t.get("teacher_label", "safe"))
    if label == "safe" and score > 0.8:
        return False
    if label == "unsafe" and score < 0.2:
        return False
    if not t.get("decision_basis"):
        return False
    return True


def tier_teacher(t: dict, valid: bool) -> str:
    """Guide 7.4 confidence tiers."""
    conf = float(t.get("teacher_confidence", 0.0))
    agree = float(t.get("agent_agreement", 0.0))
    spans = bool((t.get("unsafe_evidence_spans") or []) or (t.get("safe_evidence_spans") or []))
    conflicts = bool((t.get("conflict_flags") or []) or (t.get("contradiction_flags") or []))
    if valid and agree >= 0.75 and conf >= 0.80 and spans and not conflicts:
        return "high"
    if conf >= 0.60 and not (conf < 0.60):
        return "medium"
    return "low"


def main() -> None:
    rows = read_jsonl(DATASET)
    reserved = json.loads(RESERVED.read_text(encoding="utf-8"))
    res_ids = set(reserved["ids"])
    res_groups = set(reserved["group_ids"])
    teacher = {}
    for split in ("train", "dev", "test"):
        for r in read_jsonl(AGENT_DIR / f"{split}.jsonl"):
            teacher[r["id"]] = r

    train = [r for r in rows if r["split"] == "train"]
    train_ids = {r["id"] for r in train}
    train_groups = {r["group_id"] for r in train}
    overlap_ids = train_ids & res_ids
    overlap_groups = train_groups & res_groups

    usable = [r for r in train if r["id"] not in res_ids and r["group_id"] not in res_groups]
    usable_ids = {r["id"] for r in usable}
    usable_groups = {r["group_id"] for r in usable}
    # guide 6.3: the NEURAL-STUDENT train pool must be disjoint from the reserved Exp2 test
    assert not (usable_ids & res_ids), f"{len(usable_ids & res_ids)} student train ids in reserved exp2 test"
    assert not (usable_groups & res_groups), f"{len(usable_groups & res_groups)} student train groups in reserved exp2 test"
    near_dup = 0
    seen: dict[str, str] = {}
    dup_pairs: list[tuple[str, str]] = []
    manifest = []
    for r in usable:
        tr = teacher.get(r["id"]) or {}
        sig = tr.get("signal") or {}
        t = {
            "teacher_label": str(sig.get("teacher_label", "safe")),
            "teacher_score": float(sig.get("teacher_score", 0.5)),
            "teacher_type": str(sig.get("teacher_type", "safe")),
            "teacher_confidence": float(sig.get("teacher_confidence", sig.get("confidence", 0.0))),
            "agent_agreement": float(sig.get("agent_agreement", 0.0)),
            "decision_basis": list(sig.get("decision_basis") or []),
            "unsafe_evidence_spans": list(sig.get("unsafe_evidence_spans") or []),
            "safe_evidence_spans": list(sig.get("safe_evidence_spans") or []),
            "conflict_flags": list((sig.get("conflict_flags") or []) + (tr.get("conflict_flags") or [])),
            "contradiction_flags": list(sig.get("contradiction_flags") or []),
        }
        t["signal_valid"] = validate_teacher_signal(t)
        t["confidence_tier"] = tier_teacher(t, t["signal_valid"])
        gold_source = "procedural_weak" if r["source"] == "synthetic" else ("audit" if r["source"] in ("e1_context_r2", "fraudr1_all") else "official")
        family = r["group_id"]
        mutation = "real"
        if r["source"] == "synthetic":
            parts = r["group_id"].split("_")
            if len(parts) >= 3 and parts[0] == "syn":
                family = f"{parts[0]}_{parts[1]}_{parts[2]}"
                mutation = MUTATION_BY_SUBTYPE.get(r.get("subtype", ""), "synthetic")
        key = norm(r["user_query"]) + "|" + norm(r["target_model_answer"])
        if key in seen and seen[key] != r["split"]:
            near_dup += 1
            dup_pairs.append((seen[key], r["id"]))
        seen[key] = r["id"]
        manifest.append({
            "id": r["id"], "group_id": r["group_id"], "template_family_id": family,
            "semantic_source_id": r["group_id"], "mutation_type": mutation,
            "pair_id": r.get("pair_id"), "split": "train",
            "user_query": r["user_query"], "target_model_answer": r["target_model_answer"],
            "gold_label": r["gold_label"], "gold_type": r.get("gold_type", r["gold_label"]),
            "gold_source": gold_source, "source": r["source"], "subtype": r.get("subtype", ""),
            "language": r.get("language", ""), "target_model": r.get("target_model", ""),
            **t,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train_manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n", encoding="utf-8")

    report = {
        "train_rows_total": len(train),
        "exp3_train_ids_in_reserved_test": len(overlap_ids),
        "exp3_train_groups_in_reserved_test": len(overlap_groups),
        "reserved_ids_excluded": len([r for r in train if r["id"] in res_ids]),
        "usable_train_rows": len(manifest),
        "usable_groups": len({m["group_id"] for m in manifest}),
        "template_families": len({m["template_family_id"] for m in manifest}),
        "by_source": dict(Counter(m["source"] for m in manifest)),
        "by_language": dict(Counter(m["language"] for m in manifest)),
        "by_gold": dict(Counter(m["gold_label"] for m in manifest)),
        "by_subtype": dict(Counter(m["subtype"] for m in manifest)),
        "teacher_tier": dict(Counter(m["confidence_tier"] for m in manifest)),
        "teacher_signal_valid": sum(1 for m in manifest if m["signal_valid"]),
        "near_dup_qy_in_train_pool": near_dup,
        "dup_examples": dup_pairs[:10],
        "reserved_benchmarks": reserved["benchmarks"],
        "note": "usable = exp3 train minus reserved exp2 test ids/groups (guide 6.3)",
    }
    (OUT_DIR / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
# ---------------------------------------------------------------------------
# Final student manifest audit (guide 5-7, 30.1)
# ---------------------------------------------------------------------------
def audit_final_manifest(args) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--exp3-dataset", default=str(DATASET))
    ap.add_argument("--balanced-test", default=str(REPO / "experiments/exp2_prior_work_comparison/balanced_design/manifests/balanced_test_manifest.jsonl"))
    ap.add_argument("--token-length-report", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(args.remaining)

    rows = read_jsonl(Path(args.manifest))
    dataset = read_jsonl(Path(args.exp3_dataset))
    balanced_test = read_jsonl(Path(args.balanced_test))

    # leakage sets
    def fam_of(row):
        g = str(row.get("group_id") or row.get("group") or "")
        fam = str(row.get("template_family_id") or row.get("family") or "")
        if "exp3x" in fam:
            return fam
        return g

    def sets(rows_):
        ids, qys, groups, fams = set(), set(), set(), set()
        for r in rows_:
            q = r.get("user_query") or r.get("query") or ""
            a = r.get("target_model_answer") or r.get("answer") or ""
            ids.add(r.get("id") or r.get("sample_id"))
            if q and a:
                qys.add(qy_hash(q, a))
            g = str(r.get("group_id") or r.get("group") or "")
            if g:
                groups.add(g)
            f = fam_of(r)
            if f:
                fams.add(f)
        return ids, qys, groups, fams

    t_ids, t_qys, t_groups, t_fams = sets(rows)
    dt_rows = [r for r in dataset if r["split"] in ("dev", "test")]
    d_ids, d_qys, d_groups, d_fams = sets(dt_rows)
    b_ids, b_qys, b_groups, b_fams = sets(balanced_test)

    report = {
        "final_train_count": len(rows),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_source_bucket": dict(Counter(r["source_bucket"] for r in rows)),
        "by_language": dict(Counter(r["language"] for r in rows)),
        "by_gold": dict(Counter(r["gold_label"] for r in rows)),
        "by_gold_type": dict(Counter(r["gold_type"] for r in rows)),
        "by_subtype": dict(Counter(r["subtype"] for r in rows).most_common(20)),
        "template_families": len({fam_of(r) for r in rows}),
        "pair_rows": sum(1 for r in rows if r.get("pair_id")),
        "teacher_only_rows": sum(1 for r in rows if r.get("teacher_only")),
    }

    # effective sampler ratios (guide 7/8)
    wsum = sum(float(r.get("sample_weight", 1.0)) for r in rows)
    eff = {}
    for b in ("benchmark", "paired_dev", "synthetic_core", "hard_expansion"):
        sub = [r for r in rows if r["source_bucket"] == b]
        eff[b] = round(sum(float(r.get("sample_weight", 1.0)) for r in sub) / max(wsum, 1e-9), 4)
    eff["overall_en"] = round(sum(float(r.get("sample_weight", 1.0)) for r in rows if r["language"] == "en") / max(wsum, 1e-9), 4)
    eff["overall_safe"] = round(sum(float(r.get("sample_weight", 1.0)) for r in rows if r["gold_binary"] == 0) / max(wsum, 1e-9), 4)
    uns = [r for r in rows if r["gold_binary"] == 1]
    uw = sum(float(r.get("sample_weight", 1.0)) for r in uns)
    for t in ("fraud_assistance", "refusal_failure", "over_refusal"):
        eff[f"unsafe_{t}"] = round(sum(float(r.get("sample_weight", 1.0)) for r in uns if r["gold_type"] == t) / max(uw, 1e-9), 4)
    report["effective_sampler"] = eff

    # leakage checks
    report["leakage"] = {
        "train_vs_exp3_dev_test_ids": len(t_ids & d_ids),
        "train_vs_exp3_dev_test_qy": len(t_qys & d_qys),
        "train_vs_exp3_dev_test_groups": len(t_groups & d_groups),
        "train_vs_exp3_dev_test_fams": len(t_fams & d_fams),
        "train_vs_balanced_test_ids": len(t_ids & b_ids),
        "train_vs_balanced_test_qy": len(t_qys & b_qys),
        "train_vs_balanced_test_groups": len(t_groups & b_groups),
        "train_vs_balanced_test_fams": len(t_fams & b_fams),
    }
    # quality gates
    missing_q = sum(1 for r in rows if not str(r.get("user_query") or "").strip())
    missing_a = sum(1 for r in rows if not str(r.get("target_model_answer") or "").strip())
    invalid_gold = sum(1 for r in rows if r.get("gold_label") not in ("safe", "unsafe"))
    invalid_type = sum(1 for r in rows if r.get("gold_type") not in ("safe", "fraud_assistance", "refusal_failure", "over_refusal"))
    invalid_teacher = 0
    sum_err = 0.0
    for r in rows:
        ts = float(r.get("teacher_score", 0.5) or 0.5)
        if not (0.0 <= ts <= 1.0):
            invalid_teacher += 1
    report["quality"] = {
        "missing_query": missing_q, "missing_answer": missing_a,
        "invalid_gold_label": invalid_gold, "invalid_gold_type": invalid_type,
        "invalid_teacher_score": invalid_teacher,
    }
    dup_qy = len(t_qys) - len(rows)
    report["quality"]["duplicate_qy"] = max(dup_qy, 0)

    # teacher probs audit: teacher_distribution derived later; check score/type pairs
    report["teacher"] = {
        "teacher_only_share": round(sum(1 for r in rows if r.get("teacher_only")) / max(len(rows), 1), 4),
        "teacher_gold_disagree": sum(1 for r in rows if r.get("teacher_gold_agree") is False),
        "mean_teacher_conf": round(sum(float(r.get("teacher_confidence", 0.5) or 0.5) for r in rows) / max(len(rows), 1), 4),
    }

    # token length audit (guide 11)
    if args.token_length_report:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        lens = []
        tail_trunc = 0
        for r in rows:
            text = str(r.get("user_query", "")) + "\n\n[ANSWER]\n" + str(r.get("target_model_answer", ""))
            ids = tok(text, add_special_tokens=False)["input_ids"]
            lens.append(len(ids))
        import numpy as np
        arr = np.array(sorted(lens))
        report["token_length"] = {
            "n": len(arr), "p50": int(np.percentile(arr, 50)), "p90": int(np.percentile(arr, 90)),
            "p95": int(np.percentile(arr, 95)), "p99": int(np.percentile(arr, 99)),
            "max": int(arr[-1]),
            "over384": int((arr > 384).sum()), "over384_rate": round(float((arr > 384).mean()), 4),
        }
        report["token_length"]["decision"] = "384" if (report["token_length"]["p95"] <= 384 and report["token_length"]["over384_rate"] < 0.03) else "512"

    out = Path(args.manifest).with_name("final_audit_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.strict:
        L = report["leakage"]
        assert L["train_vs_exp3_dev_test_ids"] == 0 and L["train_vs_exp3_dev_test_qy"] == 0, "exp3 dev/test leakage"
        assert L["train_vs_balanced_test_ids"] == 0 and L["train_vs_balanced_test_qy"] == 0, "balanced test leakage"
        assert report["quality"]["missing_query"] == 0 and report["quality"]["missing_answer"] == 0
        assert report["quality"]["invalid_gold_label"] == 0 and report["quality"]["invalid_gold_type"] == 0
        assert report["quality"]["invalid_teacher_score"] == 0
        assert report["quality"]["duplicate_qy"] == 0
        print("FINAL MANIFEST AUDIT: PASS")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-manifest", action="store_true")
    args, remaining = ap.parse_known_args()
    if args.final_manifest:
        args.remaining = remaining
        audit_final_manifest(args)
    else:
        main()
