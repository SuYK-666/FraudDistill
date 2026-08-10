# -*- coding: utf-8 -*-
"""E4 v2 FINAL: replace duplicate-id rows in frozen_test with fresh pool rows.

Duplicates found: 56 rows share ids with another row (identical q+y content,
only family metadata differs). We replace each duplicate row with a fresh
candidate from the same (primary_shift, source) cell that:
  - was G2-judged via API (metadata.g2.method == "judge") or official label
  - exposure_level == L3
  - its qy_hash / q_hash / y_hash do NOT collide with any kept test row
This keeps cell quotas (U1 400 / U2 400 / U3 400) and 1200 unique ids.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    test_path = BASE / "manifests/frozen_test.jsonl"
    rows = read_jsonl(test_path)
    ids = [r["id"] for r in rows]
    dup_ids = [i for i, n in Counter(ids).items() if n > 1]
    print(f"[fix-dup] manifest rows={len(rows)} dup_ids={len(dup_ids)}")

    # find the rows to replace: keep first occurrence, replace later ones
    seen = set()
    replace_idx = []
    for i, r in enumerate(rows):
        if r["id"] in seen:
            replace_idx.append(i)
        else:
            seen.add(r["id"])
    print(f"[fix-dup] rows to replace: {len(replace_idx)}")

    pool = read_jsonl(BASE / "candidate_pool.jsonl")
    used_ids = set(ids)
    # hashes of kept rows
    kept = [r for i, r in enumerate(rows) if i not in replace_idx]
    kept_qy = {r.get("qy_hash") for r in kept}
    kept_q = {r.get("q_hash") for r in kept}
    kept_y = {r.get("y_hash") for r in kept}

    # cell quota targets for frozen_test
    quota = Counter((r.get("primary_shift"), r.get("source")) for r in kept)

    # candidate pool rows: unused, L3, g2-judged (or official), no hash collision
    cand = []
    for r in pool:
        if r["id"] in used_ids:
            continue
        if r.get("exposure_level") != "L3":
            continue
        g2 = r.get("metadata", {}).get("g2", {})
        ok_gold = (g2.get("method") == "judge" and g2.get("gold_label")) or \
                  r.get("gold_source") in ("official_response_label", "blind_judge_deepseek")
        if not ok_gold:
            continue
        if r.get("qy_hash") in kept_qy or r.get("q_hash") in kept_q or r.get("y_hash") in kept_y:
            continue
        cand.append(r)
    print(f"[fix-dup] usable candidates: {len(cand)}")

    # assign by cell, filling cells with the most remaining dup rows first
    need = Counter((rows[i]["primary_shift"], rows[i]["source"]) for i in replace_idx)
    print(f"[fix-dup] need by cell: {dict(need)}")
    by_cell = {}
    for r in cand:
        by_cell.setdefault((r.get("primary_shift"), r.get("source")), []).append(r)

    chosen = []
    for cell, n_need in sorted(need.items(), key=lambda kv: -kv[1]):
        cell_cands = by_cell.get(cell, [])
        if len(cell_cands) < n_need:
            print(f"[fix-dup] ERROR: cell {cell} needs {n_need} but only {len(cell_cands)} available")
            sys.exit(2)
        chosen.extend(cell_cands[:n_need])
    print(f"[fix-dup] chosen replacements: {len(chosen)}")

    # apply: replace in original positions
    new_rows = list(rows)
    for idx, r in zip(replace_idx, chosen):
        new_rows[idx] = r

    # validate
    new_ids = [r["id"] for r in new_rows]
    assert len(set(new_ids)) == 1200, f"unique ids {len(set(new_ids))}"
    assert len(new_ids) == 1200
    cq = Counter((r.get("primary_shift"), r.get("source")) for r in new_rows)
    gold = Counter(r.get("gold_label") for r in new_rows)
    print(f"[fix-dup] new quota: {dict(cq)}")
    print(f"[fix-dup] new gold: {dict(gold)}")
    assert all(v == 200 for v in cq.values()) and sum(cq.values()) == 1200
    assert gold["safe"] == 600 and gold["unsafe"] == 600, gold
    assert all(r.get("exposure_level") == "L3" for r in new_rows)

    # backup + write
    bak = test_path.with_name("frozen_test.predupfix.jsonl")
    if not bak.exists():
        test_path.rename(bak)
    with open(test_path, "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # replacement manifest for inference
    out = BASE / "manifests/frozen_test_replacements.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_replaced": len(chosen),
        "backup": str(bak),
        "replacement_manifest": str(out),
        "old_ids": [rows[i]["id"] for i in replace_idx],
        "new_ids": [r["id"] for r in chosen],
    }
    (BASE / "manifests/fix_duplicate_ids.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fix-dup] DONE: {len(chosen)} rows replaced, manifest updated")


if __name__ == "__main__":
    main()
