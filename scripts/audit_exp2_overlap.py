# -*- coding: utf-8 -*-
"""Exp2 overlap audit (guide 7.1-7.2).

Builds the Exp3 exposure qy_hash set (exp3 train/dev/test incl. pilot, neural
student train manifests) and reports per-benchmark overlap against the unified
pools. Zero API. Outputs:
  experiments/exp2_prior_work_comparison/audit/exp3_exposure_qy_hashes.json
  experiments/exp2_prior_work_comparison/audit/overlap_summary.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_cross_benchmark.paths import (  # noqa: E402
    BENCHMARKS,
    EXP3_DATASET,
    EXP3_STUDENT_MANIFESTS,
    EXPERIMENT_DIR,
    EXPECTED_POOL,
)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def qy_hash(query: str, answer: str) -> str:
    payload = normalize_text(query) + "\0" + normalize_text(answer)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_q(row: dict) -> str:
    return str(row.get("user_query", row.get("query", "")) or "")


def get_a(row: dict) -> str:
    return str(row.get("target_model_answer", row.get("answer", "")) or "")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    exposure: dict[str, int] = {}
    hashes: set[str] = set()

    rows = read_jsonl(EXP3_DATASET)
    for r in rows:
        hashes.add(qy_hash(get_q(r), get_a(r)))
    exposure["exp3_dataset_train_dev_test"] = len(rows)

    for m in EXP3_STUDENT_MANIFESTS:
        n = 0
        for r in read_jsonl(m):
            hashes.add(qy_hash(get_q(r), get_a(r)))
            n += 1
        exposure[f"student_manifest:{m.name}"] = n

    out_dir = EXPERIMENT_DIR / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "exp3_exposure_qy_hashes.json"
    out_file.write_text(
        json.dumps(
            {
                "purpose": "Exp3 exposure qy_hash set (guide 7.2); new exp2 main-test samples prefer outside this set",
                "created": "2026-08-05",
                "hash_fn": "sha256(normalize(q)+chr(0)+normalize(a))",
                "sources": exposure,
                "n_unique_hashes": len(hashes),
                "hashes": sorted(hashes),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {}
    for b in BENCHMARKS:
        unified = EXPERIMENT_DIR / b / "unified" / f"{b}_eval.jsonl"
        rows = read_jsonl(unified)
        n_valid = 0
        n_overlap = 0
        groups: dict[str, list[str]] = {}
        for r in rows:
            if r.get("answer_status") != "frozen" or not (r.get("answer") or "").strip():
                continue
            n_valid += 1
            h = qy_hash(r.get("query"), r.get("answer"))
            ov = h in hashes
            if ov:
                n_overlap += 1
            groups.setdefault(r["group_id"], []).append(1 if ov else 0)
        clean_groups = sum(1 for vals in groups.values() if not any(vals))
        summary[b] = {
            "pool_n": len(rows),
            "expected_n": EXPECTED_POOL[b],
            "valid_qy": n_valid,
            "overlap_qy": n_overlap,
            "overlap_groups": sum(1 for vals in groups.values() if any(vals)),
            "clean_groups": clean_groups,
        }
        print(f"[{b}] pool={len(rows)} valid={n_valid} overlap_qy={n_overlap} clean_groups={clean_groups}")

    summary_file = out_dir / "overlap_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"exposure hashes: {len(hashes)} -> {out_file}")
    print(f"summary -> {summary_file}")


if __name__ == "__main__":
    main()
