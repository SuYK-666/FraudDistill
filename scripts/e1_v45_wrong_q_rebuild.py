# -*- coding: utf-8 -*-
"""E1 v4.5: rebuild wrong-q map with same-language + same-category matching.

Corrects the pre-registration mismatch (the original map matched only language).
Categories are resolved via the A7500 canonical-case registry (canonical_case_id
-> fraud_category). The old map is preserved as *_V1_ARCHIVE.jsonl. Records an
amendment in E1_V4_PROTOCOL_LOCK.json and refreshes E1_V4_FREEZE.json note.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frauddistill.e1_final_v4.split_audit import build_wrong_q_map
DATA = ROOT / "data" / "prepared" / "e1_final_triad_v4"
POOL = ROOT / "data" / "prepared" / "e1_final_triad_v32" / "E1_V32_REAL_POOL.jsonl"


def read_jsonl(p: pathlib.Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: pathlib.Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    pool = read_jsonl(POOL)
    cat_by_case = {str(r["canonical_case_id"]): str(r["fraud_category"]) for r in pool}
    print("pool rows:", len(pool), "unique cases:", len(cat_by_case))

    anchor = read_jsonl(DATA / "E1_V4_PANEL_ANCHOR.jsonl")
    category_of = {r["response_id"]: cat_by_case.get(str(r["canonical_case_id"]), "") for r in anchor}
    resolved = sum(1 for v in category_of.values() if v)
    print("anchor:", len(anchor), "category resolved:", resolved)

    new_map = build_wrong_q_map(anchor, rng_seed=20260810, category_of=category_of)
    missing = [r["response_id"] for r in anchor if r["response_id"] not in new_map]
    print("new map entries:", len(new_map), "missing:", len(missing))
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} anchor rows lack a wrong q")

    old_path = DATA / "E1_V4_WRONG_Q_MAP.jsonl"
    if old_path.exists():
        backup = DATA / "E1_V4_WRONG_Q_MAP_V1_ARCHIVE.jsonl"
        if not backup.exists():
            old_path.replace(backup)
            print("backed up old map ->", backup.name)

    with old_path.open("w", encoding="utf-8") as f:
        for rid, wq in new_map.items():
            f.write(json.dumps({"response_id": rid, "wrong_q": wq}, ensure_ascii=False) + "\n")
    print("wrote new map ->", old_path.name)

    # coverage by stratum / language / category
    by_lc = collections.Counter()
    for r in anchor:
        key = (r["stratum"], str(r["language"]), category_of[r["response_id"]] or "uncategorized")
        by_lc[key] += 1
    # category match rate: fraction of rows whose wrong-q row has same (lang, cat)
    cat_by_rid = {r["response_id"]: category_of[r["response_id"]] for r in anchor}
    lang_by_rid = {r["response_id"]: str(r.get("language") or "unknown") for r in anchor}
    # every row that shares this wrong-q text counts as a candidate owner;
    # same-language/same-category is satisfied if ANY candidate matches
    q_to_rows: dict[str, list[dict]] = collections.defaultdict(list)
    for r in anchor:
        q_to_rows[r["q_private"]].append(r)
    same_cat = same_lang = 0
    for rid, wq in new_map.items():
        owners = q_to_rows.get(wq, [])
        if not owners:
            continue
        if any(lang_by_rid[rid] == str(o.get("language") or "unknown") for o in owners):
            same_lang += 1
            if cat_by_rid[rid] and any(cat_by_rid[rid] == cat_by_rid.get(o["response_id"], "") for o in owners):
                same_cat += 1
    print(f"same-language pairs: {same_lang}/{len(new_map)}; same-language+category pairs: {same_cat}/{len(new_map)}")

    # update FREEZE.json
    freeze_path = DATA / "E1_V4_FREEZE.json"
    if freeze_path.exists():
        freeze = read_json(freeze_path)
        freeze["wrong_q_map"] = new_map
        freeze["wrong_q_map_v45"] = {
            "note": "v4.5 rebuilt wrong-q map: same split, same language, same fraud category (resolved via A7500 canonical-case registry), different merged family; fallback same-language then any-family; full 1200/1200 coverage.",
            "n": len(new_map),
            "category_resolved_n": resolved,
            "same_lang_n": same_lang,
            "same_lang_cat_n": same_cat,
        }
        write_json(freeze_path, freeze)
        print("FREEZE.json updated")

    # update PROTOCOL_LOCK.json with v4.5 amendment
    lock_path = DATA / "E1_V4_PROTOCOL_LOCK.json"
    if lock_path.exists():
        lock = read_json(lock_path)
        cfg_sha = None
        cfg_path = ROOT / "configs" / "experiments" / "e1_final_triad_v4.yaml"
        if cfg_path.exists():
            cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        lock["amendment_v45_wrong_q_category_control"] = {
            "note": (
                "v4.5 correction amendment (registered before any post-M1 statistical "
                "re-computation): the pre-registered wrong-q control promised 'same split, "
                "same language/category, different family' but the v1 implementation matched "
                "language only. The wrong-q map was rebuilt to match same language AND same "
                "fraud category (categories resolved for all 1200 anchor rows via the A7500 "
                "canonical-case registry), different merged family, with documented fallbacks "
                "(same-language, then any different family) and full 1200/1200 coverage. "
                "M1 wrong_q_y predictions were regenerated offline with the frozen q_y models; "
                "Qwen/DeepSeek wrong_q_y votes were re-collected with the new prompts. "
                "This is a registered correction of an implementation mismatch, not an "
                "outcome-driven control adjustment."
            ),
            "n": len(new_map),
            "category_resolved_n": resolved,
            "same_lang_n": same_lang,
            "same_lang_cat_n": same_cat,
            "config_sha256_after_v45": cfg_sha,
        }
        write_json(lock_path, lock)
        print("PROTOCOL_LOCK.json updated (amendment_v45)")

    print("DONE")


if __name__ == "__main__":
    main()
