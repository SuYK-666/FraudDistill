from __future__ import annotations

import csv
from pathlib import Path


def candidate_census(rows: list[dict], stage: str) -> dict:
    safe = [row for row in rows if row.get("exp1_label") == "safe"]
    unsafe = [row for row in rows if row.get("exp1_label") == "unsafe"]
    by_source: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source"))
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "stage": stage,
        "rows": len(rows),
        "components": len({str(row.get("semantic_component_id")) for row in rows}),
        "safe": len(safe),
        "unsafe": len(unsafe),
        "fraud_core_safe": sum(row.get("prompt_risk_domain") == "fraud_core" for row in safe),
        "fraud_core_unsafe": sum(row.get("prompt_risk_domain") == "fraud_core" for row in unsafe),
        "sources": len(by_source),
        "largest_source_rows": max(by_source.values(), default=0),
    }


def write_funnel(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["stage"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
