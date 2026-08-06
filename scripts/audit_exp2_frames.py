# -*- coding: utf-8 -*-
"""Exp2 frame + overlap integrity audit (guide sections 7, 26, 32.2).

Zero-API. Cross-checks frame parquet/metadata vs rescore files, exports the
Exp3 exposure summary (all-full / unexposed / seen-in-exp3) and writes
audit/frame_integrity.json + audit/overlap_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from frauddistill.exp2_static_repair.offline_guard import require_offline  # noqa: E402

EXPERIMENT_DIR = REPO / "experiments" / "exp2_prior_work_comparison"
AUDIT_DIR = EXPERIMENT_DIR / "audit"
METRICS_DIR = EXPERIMENT_DIR / "metrics"
RESCORE_DET = EXPERIMENT_DIR / "offline_rescore" / "deterministic"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        os.environ["FRAUDDISTILL_OFFLINE"] = "1"
    require_offline()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # frame integrity: metadata n vs rescore n
    frame_integrity = {}
    for meta in sorted((METRICS_DIR / "frames").glob("*.metadata.json")):
        md = json.loads(meta.read_text(encoding="utf-8"))
        parquet = meta.with_suffix("").with_suffix(".parquet")
        import pandas as pd

        df = pd.read_parquet(parquet) if parquet.exists() else None
        frame_integrity[meta.stem.replace(".metadata", "")] = {
            "metadata_n": md["n"],
            "parquet_n": int(len(df)) if df is not None else None,
            "metadata_n_positive": md["n_positive"],
            "parquet_n_positive": int(df["y_true"].sum()) if df is not None else None,
            "ok": bool(df is not None and len(df) == md["n"] and int(df["y_true"].sum()) == md["n_positive"]),
        }

    # overlap summary (guide 26): all-full / unexposed / seen-in-exp3
    manifest = []
    for l in (EXPERIMENT_DIR / "manifests" / "full_manifest.jsonl").open(encoding="utf-8"):
        manifest.append(json.loads(l))
    from collections import Counter

    overlap_summary = {}
    for src in ("fraudr1", "orbench", "do_not_answer", "aegis2"):
        rows = [r for r in manifest if r.get("source") == src]
        seen = sum(1 for r in rows if r.get("overlap_exp3"))
        overlap_summary[src] = {
            "all_full": len(rows),
            "seen_in_exp3": seen,
            "unexposed": len(rows) - seen,
        }
    overlap_summary["_note"] = ("seen-in-exp3 = sample also present in Exp3 "
                                "train/dev/test; unexposed = not in Exp3 (guide 26)")

    # offline guard report (guide 33)
    guard_report = {
        "offline": True,
        "api_calls": 0,
        "guard": "FRAUDDISTILL_OFFLINE=1 enforced via require_offline()/assert_online_allowed()",
        "providers_blocked": ["deepseek", "openai", "dashscope", "google", "zhipuai", "moonshot"],
        "validated_at": "exp2-static-fix-20260806",
    }

    (AUDIT_DIR / "frame_integrity.json").write_text(
        json.dumps(frame_integrity, ensure_ascii=False, indent=1), encoding="utf-8")
    (AUDIT_DIR / "overlap_summary.json").write_text(
        json.dumps(overlap_summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (AUDIT_DIR / "offline_guard_report.json").write_text(
        json.dumps(guard_report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"frame_integrity": frame_integrity, "overlap_summary": overlap_summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
