from __future__ import annotations

from frauddistill.exp1_ccfa.semantic_components import explicit_label_token_audit


def validate_context_collision_rows(rows: list[dict]) -> dict:
    audit = explicit_label_token_audit(rows)
    group_ids = {row.get("context_collision_group_id") for row in rows if row.get("context_collision_group_id")}
    return {
        "passed": audit["passed"] and len(group_ids) > 0,
        "explicit_label_token_audit": audit,
        "group_count": len(group_ids),
        "row_count": len(rows),
    }

