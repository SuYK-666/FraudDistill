# -*- coding: utf-8 -*-
"""Build the historical Silver-label replacement audit replay (oracle simulation)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from frauddistill.supplementary.common import REPO, binary_label, read_jsonl, row_key, write_json, write_jsonl

ROOT = REPO / "experiments" / "lighteval_supplementary"
SELECTION = ROOT / "prepared" / "audit_selection_manifest.jsonl"
TEST = ROOT / "prepared" / "raw" / "relation_test.jsonl"
OUT = ROOT / "audit" / "oracle"


def main() -> int:
    selection = read_jsonl(SELECTION)
    rows = {row_key(row): row for row in read_jsonl(TEST)}
    results = []
    for selected in selection:
        row = rows[selected["id"]]
        gold = "unsafe" if binary_label(row) else "safe"
        results.append({
            "id": selected["id"],
            "rank": selected["rank"],
            "audit_status": "ok",
            "audit_label": gold,
            "audit_type": row.get("gold_type") or ("fraud_assistance" if gold == "unsafe" else "safe"),
            "confidence": 1.0,
            "evidence_span": "__REFERENCE_LABEL_REPLACEMENT__",
            "estimated_cost_cny": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "retry_count": 0,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "audit_results.jsonl", results)
    write_json(OUT / "metadata.json", {
        "method": "oracle_reference_label_replacement",
        "scope": "E6 relation test 384; audited rows replaced with Silver/reference binary labels",
        "n_items": len(results),
        "budget_sizes": [19, 38, 57, 76],
        "historical_k77_note": "The old 20% table used K=77; the current frozen protocol uses K=76.",
        "warning": "This is an oracle simulation, not an external API audit.",
    })
    print(json.dumps({"status": "READY", "n_items": len(results), "path": str(OUT / "audit_results.jsonl")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
