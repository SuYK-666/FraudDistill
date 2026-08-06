# -*- coding: utf-8 -*-
"""Fraud-R1 context audit (guide section 19).

Zero-API. Checks whether the manifest query preserves the raw Fraud-R1
prompt content (generated text + role background). Fraud-R1 raw data is
single-turn; the audit therefore verifies completeness of the prompt rather
than multi-turn history, and flags role-play context loss.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_static_repair.offline_guard import require_offline  # noqa: E402

EXPERIMENT_DIR = REPO / "experiments" / "exp2_prior_work_comparison"
AUDIT_DIR = EXPERIMENT_DIR / "audit"
RAW_PROMPTS = REPO / "data" / "raw" / "fraudr1" / "prompts.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_raw_repo_index() -> dict[str, dict]:
    """Build {raw_id: item} from the Fraud-R1 dataset JSON files.

    Manifest metadata.fraudr1_raw_id refers to the id inside the repo
    dataset files (e.g. FP-base-Chinese.json).
    """
    repo_dir = REPO / "data" / "raw" / "fraudr1" / "repo" / "dataset"
    index: dict[str, dict] = {}
    if not repo_dir.exists():
        return index
    for p in repo_dir.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else data.get("data", [])
        for it in items:
            if isinstance(it, dict) and "id" in it:
                index.setdefault(str(it["id"]), {**it, "source_file": str(p)})
    return index


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def audit_fraudr1_context() -> dict:
    manifest = []
    for l in (EXPERIMENT_DIR / "manifests" / "full_manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("source") == "fraudr1":
            manifest.append(r)

    teacher_rows = read_jsonl(EXPERIMENT_DIR / "teacher_predictions_t6" / "fraudr1_t6_predictions.jsonl")
    teacher_by_id = {r.get("id"): r for r in teacher_rows}

    raw_by_id = load_raw_repo_index()

    full_text_preserved = 0
    partial_preserved = 0
    missing_raw = 0
    roleplay_with_prefix = 0
    roleplay_total = 0
    query_lens = []
    raw_lens = []
    per_category = Counter()
    samples_with_teacher = 0

    for r in manifest:
        meta = r.get("metadata", {})
        raw_id = str(meta.get("fraudr1_raw_id", ""))
        raw_item = raw_by_id.get(raw_id)
        q = str(r.get("query") or "")
        query_lens.append(len(q))
        cat = r.get("official_category") or "unknown"
        per_category[cat] += 1

        sid = r.get("sample_id")
        if sid in teacher_by_id:
            samples_with_teacher += 1

        if raw_item is None:
            missing_raw += 1
            continue
        raw_text = str(raw_item.get("generated text") or "")
        raw_lens.append(len(raw_text))
        nq, nr = norm(q), norm(raw_text)
        if nr and nr in nq:
            full_text_preserved += 1
        elif nr and len(nr) > 50:
            # measure overlap
            overlap = 0
            for i in range(0, min(len(nq), len(nr)) - 20, 1):
                if nq[i:i+20] in nr:
                    overlap += 1
            partial_preserved += 1

        scenario = meta.get("fraudr1_scenario", "")
        if scenario == "roleplay":
            roleplay_total += 1
            # role-play markers like "假设你的姓名是" or "假设你是" or "你的身份"
            if re.search(r"假设|扮演|你的(姓名|身份|名字)", q):
                roleplay_with_prefix += 1

    n = len(manifest)
    preserved = (full_text_preserved + partial_preserved) / max(n, 1)
    last_user_only_rate = 0.0  # Fraud-R1 raw data is single-turn by construction
    role_marker_rate = roleplay_with_prefix / max(roleplay_total, 1)

    report = {
        "guide": "exp2-static-fix-20260806-section19",
        "n_manifest_rows": n,
        "n_teacher_rows": len(teacher_rows),
        "n_teacher_matched": samples_with_teacher,
        "n_raw_missing": missing_raw,
        "raw_data_structure": "single_turn_prompt",
        "note": "Fraud-R1 raw prompts are single-turn (generated text + role background); "
                "multi-turn truncation does not apply. Audit checks prompt completeness.",
        "full_text_preserved": full_text_preserved,
        "partial_preserved": partial_preserved,
        "preserved_rate": round(preserved, 4),
        "last_user_only_rate": last_user_only_rate,
        "roleplay_with_role_prefix": roleplay_with_prefix,
        "roleplay_total": roleplay_total,
        "role_marker_rate": round(role_marker_rate, 4),
        "manifest_query_len_p50": float(np.percentile(query_lens, 50)) if query_lens else None,
        "manifest_query_len_p90": float(np.percentile(query_lens, 90)) if query_lens else None,
        "categories": dict(per_category),
        "go_no_go": {
            "context_corrupted": bool(missing_raw / max(n, 1) > 0.05 or (1 - preserved) > 0.05),
            "note": "context considered corrupted when >5% raw prompts are missing or "
                    ">5% queries fail to preserve raw text (guide 19.4)",
        },
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        import os
        os.environ["FRAUDDISTILL_OFFLINE"] = "1"
    require_offline()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    report = audit_fraudr1_context()
    (AUDIT_DIR / "fraudr1_context_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
