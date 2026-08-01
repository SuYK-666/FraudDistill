from __future__ import annotations

from collections import defaultdict
from typing import Any

from .gold import is_context_high
from .registry import sha_text


def stable_hash(*parts: Any) -> str:
    return sha_text("|".join(str(p) for p in parts))


def panel_row(row: dict[str, Any], pair_id: str, gold: int, panel: str) -> dict[str, Any]:
    return {
        "panel": panel,
        "pair_id": pair_id,
        "probe_id": f"{panel}|{pair_id}|{gold}|{stable_hash(row['response_id'])[:10]}",
        "response_id": row["response_id"],
        "semantic_cluster_id": row.get("semantic_cluster_id"),
        "canonical_q_hash": row.get("canonical_q_hash"),
        "target_provider": row.get("target_provider"),
        "language": row.get("language"),
        "fraud_category": row.get("fraud_category"),
        "stage": int(row.get("stage") or 0),
        "q": row.get("q", ""),
        "y": row.get("y", ""),
        "gold": int(gold),
        "context_necessity": row.get("context_necessity", "LOW"),
        "gold_source": row.get("gold_source", "v11_consensus_core"),
    }


def match_b1(consensus_rows: list[dict[str, Any]], target_pos: int, seed: int) -> list[dict[str, Any]]:
    positives = [r for r in consensus_rows if r.get("gold") == 1 and r.get("gold_source") == "v11_consensus_core"]
    negatives = [r for r in consensus_rows if r.get("gold") == 0 and r.get("gold_source") == "v11_consensus_core"]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for n in negatives:
        buckets[(n.get("target_provider"), n.get("language"), n.get("fraud_category"), n.get("stage"))].append(n)
    for key in buckets:
        buckets[key].sort(key=lambda r: stable_hash(seed, "neg", r["response_id"]))
    pairs: list[dict[str, Any]] = []
    used_neg: set[str] = set()
    for p in sorted(positives, key=lambda r: stable_hash(seed, "pos", -int(r.get("candidate_score", 0)), r["response_id"])):
        key = (p.get("target_provider"), p.get("language"), p.get("fraud_category"), p.get("stage"))
        pool = [n for n in buckets.get(key, []) if n["response_id"] not in used_neg]
        if not pool:
            continue
        n = pool[0]
        used_neg.add(n["response_id"])
        pid = f"b1_{len(pairs)//2+1:04d}"
        pairs.extend([panel_row(p, pid, 1, "B1"), panel_row(n, pid, 0, "B1")])
        if len(pairs) // 2 >= target_pos:
            break
    return pairs


def exact_q_groups(consensus_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in consensus_rows:
        if row.get("gold_source") == "v11_consensus_core":
            groups[(row.get("canonical_q_hash"), row.get("target_provider"))].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda kv: stable_hash(seed, *kv[0])):
        pos = [r for r in group if r.get("gold") == 1]
        neg = [r for r in group if r.get("gold") == 0]
        if not pos or not neg:
            continue
        pid = f"b2_{len(out)//2+1:04d}"
        out.append(panel_row(sorted(pos, key=lambda r: stable_hash(seed, r["response_id"]))[0], pid, 1, "B2"))
        out.append(panel_row(sorted(neg, key=lambda r: stable_hash(seed, r["response_id"]))[0], pid, 0, "B2"))
    return out


def context_panel(consensus_rows: list[dict[str, Any]], seed: int, target_pos: int = 80) -> list[dict[str, Any]]:
    high = [r for r in consensus_rows if r.get("gold_source") == "v11_consensus_core" and is_context_high(r)]
    return match_b1(high, target_pos, seed)


def assign_split(panel: list[dict[str, Any]], seed: int, dev_frac: float, cal_frac: float) -> list[dict[str, Any]]:
    pair_ids = sorted({r["pair_id"] for r in panel}, key=lambda p: stable_hash(seed, "split", p))
    n = len(pair_ids)
    dev_n = int(round(n * dev_frac))
    cal_n = int(round(n * cal_frac))
    split = {p: "delta_dev" for p in pair_ids[:dev_n]}
    split.update({p: "calibration" for p in pair_ids[dev_n : dev_n + cal_n]})
    split.update({p: "test" for p in pair_ids[dev_n + cal_n :]})
    return [{**r, "split_role": split[r["pair_id"]]} for r in panel]


def wrong_q_map(panel: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in panel:
        candidates = [
            r for r in panel
            if r["probe_id"] != row["probe_id"]
            and r.get("target_provider") == row.get("target_provider")
            and r.get("language") == row.get("language")
            and r.get("fraud_category") == row.get("fraud_category")
            and r.get("stage") == row.get("stage")
            and r.get("canonical_q_hash") != row.get("canonical_q_hash")
        ]
        if not candidates:
            candidates = [r for r in panel if r.get("language") == row.get("language") and r.get("canonical_q_hash") != row.get("canonical_q_hash")]
        if candidates:
            picked = sorted(candidates, key=lambda r: stable_hash(seed, "wrong", row["probe_id"], r["probe_id"]))[0]
            rows.append({"probe_id": row["probe_id"], "wrong_q_probe_id": picked["probe_id"], "wrong_q": picked["q"], "wrong_q_hash": picked["canonical_q_hash"]})
    return rows
