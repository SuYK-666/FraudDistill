# -*- coding: utf-8 -*-
"""Merge annotated expansion pool into the neural-student train manifest (guide 7.2).

Inputs: train_manifest.jsonl (existing audited rows) + expansion_annotated.jsonl
(T6-annotated 4,000-row pool).  Output: train_manifest_expanded.jsonl with the
same schema as the audited manifest; teacher fields flattened from the signal.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data/prepared/exp3_neural_student/train_manifest.jsonl"
EXPANSION = REPO / "data/prepared/exp3_neural_student/expansion_annotated.jsonl"
POOL = REPO / "data/prepared/exp3_neural_student/expansion_pool.jsonl"
RESERVED = REPO / "data/splits/reserved_exp2_test_ids.json"
OUT = REPO / "data/prepared/exp3_neural_student/train_manifest_expanded.jsonl"
REPORT = REPO / "data/prepared/exp3_neural_student/expansion_merge_report.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()] if path.exists() else []


def norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text)).lower()


def validate_teacher_signal(t: dict) -> bool:
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
    conf = float(t.get("teacher_confidence", 0.0))
    agree = float(t.get("agent_agreement", 0.0))
    spans = bool((t.get("unsafe_evidence_spans") or []) or (t.get("safe_evidence_spans") or []))
    conflicts = bool((t.get("conflict_flags") or []) or (t.get("contradiction_flags") or []))
    if valid and agree >= 0.75 and conf >= 0.80 and spans and not conflicts:
        return "high"
    if conf >= 0.60:
        return "medium"
    return "low"


def flatten_teacher(r: dict, sig: dict, fallback_tier: str | None = None) -> dict:
    t = {
        "teacher_label": str(sig.get("teacher_label", "safe")),
        "teacher_score": float(sig.get("teacher_score", 0.5)),
        "teacher_type": str(sig.get("teacher_type", "safe")),
        "teacher_confidence": float(sig.get("teacher_confidence", sig.get("confidence", 0.0))),
        "agent_agreement": float(sig.get("agent_agreement", 0.0)),
        "decision_basis": list(sig.get("decision_basis") or []),
        "unsafe_evidence_spans": list(sig.get("unsafe_evidence_spans") or []),
        "safe_evidence_spans": list(sig.get("safe_evidence_spans") or []),
        "conflict_flags": list(sig.get("conflict_flags") or []),
        "contradiction_flags": list(sig.get("contradiction_flags") or []),
    }
    t["signal_valid"] = validate_teacher_signal(t)
    t["confidence_tier"] = str(r.get("confidence_tier") or fallback_tier) if r.get("confidence_tier") else tier_teacher(t, t["signal_valid"])
    return t


def main() -> None:
    base = read_jsonl(MANIFEST)
    exp = read_jsonl(EXPANSION)
    pool = {r["id"]: r for r in read_jsonl(POOL)}
    reserved = json.loads(RESERVED.read_text(encoding="utf-8"))
    res_ids = set(reserved["ids"])
    res_groups = set(reserved["group_ids"])

    base_ids = {r["id"] for r in base}
    exp_ids = {r["id"] for r in exp}
    overlap = base_ids & exp_ids
    if overlap:
        print(f"WARNING: {len(overlap)} ids already in base manifest; skipping them")
        exp = [r for r in exp if r["id"] not in overlap]

    merged = list(base)
    skipped_no_signal = 0
    for r in exp:
        sig = r.get("signal") or {}
        if not sig.get("teacher_label") and not sig.get("teacher_score"):
            skipped_no_signal += 1
            continue
        smp = r.get("sample") or {}
        po = pool.get(r["id"]) or {}
        t = flatten_teacher(r, sig)
        row = {
            "id": r["id"], "group_id": r.get("group_id") or po.get("group_id") or r["id"],
            "template_family_id": po.get("template_family_id") or po.get("group_id") or r["id"],
            "semantic_source_id": po.get("semantic_source_id") or po.get("group_id") or r["id"],
            "mutation_type": po.get("mutation_type", "expansion"),
            "pair_id": po.get("pair_id", smp.get("pair_id")), "split": "train",
            "user_query": smp.get("user_query") or po.get("user_query", ""),
            "target_model_answer": smp.get("target_model_answer") or po.get("target_model_answer", ""),
            "gold_label": smp.get("gold_label") or po.get("gold_label", "safe"),
            "gold_type": smp.get("gold_type") or po.get("gold_type", "safe"),
            "gold_source": po.get("gold_source", "procedural_weak"),
            "source": po.get("source", "exp3_expansion"),
            "subtype": smp.get("subtype") or po.get("subtype", ""),
            "language": smp.get("language") or po.get("language", ""),
            "target_model": smp.get("target_model") or po.get("target_model", ""),
            **t,
        }
        merged.append(row)

    all_ids = {r["id"] for r in merged}
    all_groups = {r["group_id"] for r in merged}
    assert len(all_ids) == len(merged), "duplicate ids in merged manifest"
    assert not (all_ids & res_ids), "merged manifest collides with reserved exp2 test ids"
    assert not (all_groups & res_groups), "merged manifest collides with reserved exp2 test groups"

    near_dup = 0
    seen: dict[str, str] = {}
    for r in merged:
        key = norm(r["user_query"]) + "|" + norm(r["target_model_answer"])
        if key in seen:
            near_dup += 1
        seen[key] = r["id"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in merged) + "\n", encoding="utf-8")

    report = {
        "base_rows": len(base),
        "expansion_rows": len(exp),
        "expansion_skipped_no_signal": skipped_no_signal,
        "merged_rows": len(merged),
        "groups": len(all_groups),
        "template_families": len({r["template_family_id"] for r in merged}),
        "near_duplicate_pairs": near_dup,
        "by_source": dict(Counter(r["source"] for r in merged)),
        "by_language": dict(Counter(r["language"] for r in merged)),
        "by_gold": dict(Counter(r["gold_label"] for r in merged)),
        "by_subtype": dict(Counter(r["subtype"] for r in merged)),
        "by_tier": dict(Counter(r["confidence_tier"] for r in merged)),
        "pairs": len({r["pair_id"] for r in merged if r["pair_id"]}),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
