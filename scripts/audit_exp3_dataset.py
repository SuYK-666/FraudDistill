# -*- coding: utf-8 -*-
"""Audit the Exp3 dataset: group-disjoint splits, dedup, pairs, balance.

Usage: python scripts/audit_exp3_dataset.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / "data/prepared/exp3_agent_distillation/exp3_dataset.jsonl"


def norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).lower()


def audit(path: Path) -> dict:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    issues: list[str] = []
    warns: list[str] = []

    total = len(rows)
    splits = Counter(r["split"] for r in rows)
    gold = Counter(r["gold_label"] for r in rows)
    blocks = Counter(r["block"] for r in rows)
    subtypes = Counter(r["subtype"] for r in rows)
    langs = Counter(r["language"] for r in rows)
    sources = Counter(r["source"] for r in rows)

    # 1) exact duplicate (q, y) removal
    seen_qy = set()
    dup_qy = 0
    for r in rows:
        k = (r["user_query"], r["target_model_answer"])
        if k in seen_qy:
            dup_qy += 1
        seen_qy.add(k)
    if dup_qy:
        issues.append(f"{dup_qy} exact (q,y) duplicates found")

    # 2) same query different gold (allowed only as paired responses) - just report
    q_gold = defaultdict(set)
    for r in rows:
        q_gold[r["user_query"]].add(r["gold_label"])
    multi_gold_q = sum(1 for v in q_gold.values() if len(v) > 1)

    # 3) group-disjoint: no group spans multiple splits
    g_split = defaultdict(set)
    for r in rows:
        g_split[r["group_id"]].add(r["split"])
    cross = {g: s for g, s in g_split.items() if len(s) > 1}
    if cross:
        issues.append(f"{len(cross)} groups span multiple splits")

    # 4) context-flip pair integrity: pair_id rows in same split, both sides present
    pair_split = defaultdict(set)
    pair_sides = defaultdict(set)
    for r in rows:
        if r.get("pair_id"):
            pair_split[r["pair_id"]].add(r["split"])
            pair_sides[r["pair_id"]].add(r["gold_label"])
    bad_pairs = [p for p, s in pair_split.items() if len(s) > 1]
    if bad_pairs:
        issues.append(f"{len(bad_pairs)} flip pairs split across splits")
    incomplete = [p for p, s in pair_sides.items() if s != {"safe", "unsafe"}]
    if incomplete:
        warns.append(f"{len(incomplete)} flip pairs missing one side")

    # 5) missing fields
    for idx, r in enumerate(rows):
        for k in ("id", "group_id", "user_query", "target_model_answer", "gold_label", "split", "block", "subtype", "language", "target_model"):
            if not str(r.get(k, "")).strip():
                issues.append(f"row {idx} missing {k}")
                break

    # 6) empty q / y
    empty = sum(1 for r in rows if not str(r.get("user_query") or "").strip() or not str(r.get("target_model_answer") or "").strip())
    if empty:
        issues.append(f"{empty} rows with empty query/answer")

    # 7) split ratios
    expected = {"train": 0.64, "dev": 0.16, "test": 0.20}
    for s, ratio in expected.items():
        actual = splits.get(s, 0) / max(total, 1)
        if abs(actual - ratio) > 0.03:
            warns.append(f"split {s} ratio {actual:.3f} deviates from {ratio}")

    # 8) dev/test positive presence per subtype
    for st, n in subtypes.items():
        if n < 20:
            continue
        for s in ("dev", "test"):
            pos = sum(1 for r in rows if r["subtype"] == st and r["split"] == s and r["gold_label"] == "unsafe")
            neg = sum(1 for r in rows if r["subtype"] == st and r["split"] == s and r["gold_label"] == "safe")
            if pos == 0 and neg == 0:
                warns.append(f"subtype {st} absent from {s}")

    # 9) cross-split near-dup (normalized q+y) check
    norm_seen = defaultdict(set)
    cross_dup = 0
    for r in rows:
        k = norm(r["user_query"]) + "|" + norm(r["target_model_answer"])
        for s in norm_seen[k]:
            if s != r["split"]:
                cross_dup += 1
                break
        norm_seen[k].add(r["split"])
    if cross_dup:
        warns.append(f"{cross_dup} normalized (q,y) pairs appear in >1 split")

    summary = {
        "total": total, "splits": dict(splits), "gold": dict(gold),
        "blocks": {k: dict(Counter(r["gold_label"] for r in rows if r["block"] == k)) for k in sorted(blocks)},
        "subtypes": dict(subtypes), "languages": dict(langs), "sources": dict(sources),
        "groups": len(g_split), "flip_pairs": len(pair_split),
        "same_query_multi_gold": multi_gold_q,
        "issues": issues, "warnings": warns,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT))
    args = ap.parse_args()
    summary = audit(Path(args.dataset))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["issues"]:
        print("\nISSUES:", summary["issues"])
        sys.exit(1)
    print("\nOK: no blocking issues")


if __name__ == "__main__":
    main()