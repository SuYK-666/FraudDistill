# -*- coding: utf-8 -*-
"""E4 v2 FINAL: rebuild manifests with G2 gold (v5).
- pool merge with duplicate-id handling (same qy -> drop, diff qy -> suffix)
- rows without G2 labels are exported for a follow-up judge pass
- v5: dedupe by canonical content tuple (context, q, y); recompute qy_hash;
      only load current manifests (ignore archive dirs).
"""
from __future__ import annotations
import json, shutil, sys, time, hashlib
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

from frauddistill.e4e5_v2.exposure_registry import build_registry
from frauddistill.e4e5_v2.schemas import manifest_sha256, read_jsonl, write_jsonl

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def content_key(r):
    return (str(r.get("context") or ""), str(r.get("user_query") or ""), str(r.get("target_model_answer") or ""))

def canonical_qy_hash(r):
    q = str(r.get("user_query") or "")
    y = str(r.get("target_model_answer") or "")
    return hashlib.sha256((q + "\x01" + y).encode("utf-8")).hexdigest()

def g2_map(paths):
    m = {}
    for p in paths:
        fp = BASE / p
        if not fp.exists():
            print("[g2] missing", p); continue
        for l in open(fp, encoding="utf-8"):
            r = json.loads(l)
            if r.get("status") == "ok" and r.get("judge_label") in ("safe", "unsafe"):
                m[r["id"]] = r
    return m

def cell_of(r):
    shift = r["primary_shift"]
    if shift == "U1_category":
        return r.get("fraud_category") or ""
    if shift == "U2_source":
        sv = r.get("source_version") or ""
        if sv == "Alpaca-7B_test.jsonl": sv = "Alpaca-7B_test"
        if sv == "Alpaca2-7B_test.jsonl": sv = "Alpaca2-7B_test"
        return sv
    if shift == "U3_target_style":
        return r.get("target_model") or r.get("source_version") or ""
    return ""

SHIFTS = {"U1_category": ["elder_health_product", "naked_chat_sextortion"],
          "U2_source": ["Alpaca-7B_test", "Alpaca2-7B_test"],
          "U3_target_style": ["phi3_5_mini", "smollm2_1_7b"]}

def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    g2 = {}
    for p in ("audits/g2_full_rejudge.jsonl", "audits/g2_u2_candidates.jsonl",
              "audits/g2_u3_safe.jsonl", "audits/g2_u3_refuse_remaining.jsonl",
              "audits/g2_u1_leftover.jsonl", "audits/g2_u3_leftover.jsonl",
              "audits/g2_u2_pku_leftover.jsonl", "audits/g2_retry_results.jsonl",
              "audits/g2_retry_batch.jsonl", "audits/g2_u1_generated.jsonl",
              "audits/g2_pending_judge.jsonl", "audits/g2_u3v2.jsonl",
              "audits/g2_u1_safe_generated.jsonl"):
        g2.update(g2_map([p]))
    print(f"[rebuild] G2 rows loaded: {len(g2)}")

    rows = {}
    dup_seq = Counter()
    def add_pool(path, tag, skip_archive=True):
        fp = BASE / path
        if not fp.exists():
            return
        for r in load(fp):
            rid = str(r.get("id") or "")
            if not rid:
                continue
            if rid in rows:
                # same content -> true duplicate, drop
                if content_key(r) == content_key(rows[rid]):
                    continue
                # different content -> keep with suffixed id
                dup_seq[rid] += 1
                rid = f"{rid}__dup{dup_seq[rid]}"
                r = dict(r); r["id"] = rid
                r["metadata"] = dict(r.get("metadata") or {})
                r["metadata"]["dup_of"] = rid.rsplit("__dup", 1)[0]
            rows[rid] = r; rows[rid]["_pool"] = tag
    # v5: current manifests only (no archives)
    add_pool("manifests/frozen_test.jsonl", "panel_test")
    add_pool("manifests/calibration.jsonl", "panel_cal")
    add_pool("manifests/u2_candidates_v2.jsonl", "u2_cand")
    add_pool("manifests/u3_safe_candidates_v2.jsonl", "u3_refuse")
    add_pool("manifests/u1_leftover_candidates_v2.jsonl", "u1_left")
    add_pool("manifests/u3_leftover_candidates_v2.jsonl", "u3_left")
    add_pool("manifests/u2_pku_leftover_candidates_v2.jsonl", "u2_left")
    add_pool("manifests/u3_refuse_remaining_v2.jsonl", "u3_refuse2")
    add_pool("manifests/u1_generated_v2.jsonl", "u1_gen")
    add_pool("manifests/u3v2_candidates_v2.jsonl", "u3v2")
    add_pool("manifests/u1_generated_safe_v2.jsonl", "u1_gen_safe")
    print(f"[rebuild] row pool: {len(rows)} (dup-suffixed: {sum(dup_seq.values())})")

    labeled, need_judge = [], []
    for rid, r in rows.items():
        g = g2.get(rid)
        if not g:
            need_judge.append(r)
            continue
        r["gold_label"] = g["judge_label"]
        r["gold_type"] = g.get("judge_type") or ("safe" if g["judge_label"] == "safe" else "fraud_assistance")
        r["gold_binary"] = 1 if g["judge_label"] == "unsafe" else 0
        r["gold_source"] = "blind_judge_deepseek"
        r["exposure_level"] = "L3"
        labeled.append(r)
    print(f"[rebuild] labeled={len(labeled)} need_judge={len(need_judge)}")

    if need_judge:
        jp = BASE / "manifests" / "g2_pending_judge.jsonl"
        write_jsonl(jp, need_judge)
        print(f"[rebuild] pending-judge manifest written: {jp}")

    # v5: canonical dedupe on content tuple (context, q, y)
    seen_key, uniq = set(), []
    for r in labeled:
        k = content_key(r)
        if k in seen_key:
            continue
        seen_key.add(k)
        if r.get("user_query") is not None and r.get("target_model_answer") is not None:
            r["qy_hash"] = canonical_qy_hash(r)
        uniq.append(r)
    print(f"[rebuild] after content dedupe: {len(uniq)} (dropped {len(labeled)-len(uniq)})")

    registry, loaded = build_registry(REPO)
    kept = []
    for r in uniq:
        if registry.audit_candidate(r)["passed"]:
            kept.append(r)
    print(f"[rebuild] exposure audit kept {len(kept)}/{len(uniq)}")

    predicted_ids = set()
    for fp in ("predictions/final_student.jsonl", "predictions/neural_gold.jsonl",
               "predictions/neural_softdistill.jsonl", "predictions/final_student_calibration.jsonl"):
        if (BASE / fp).exists():
            for l in open(BASE / fp, encoding="utf-8"):
                predicted_ids.add(json.loads(l)["id"])

    QUOTA = {"cal": {"safe": 50, "unsafe": 50}, "test": {"safe": 100, "unsafe": 100}}
    cal_rows, test_rows = [], []
    all_ok = True
    for shift, cells in SHIFTS.items():
        srows = [r for r in kept if r["primary_shift"] == shift and cell_of(r) in cells]
        if shift == "U1_category":
            # single-generator U1: SmolLM2-1.7B-Instruct only (shortcut gate:
            # target_model must not separate labels; generator is a nuisance)
            srows = [r for r in srows if r.get("target_model") == "SmolLM2-1.7B-Instruct"]
        fams = defaultdict(list)
        for r in srows:
            fams[r.get("family_id") or r["id"]].append(r)
        def fam_priority(f):
            fr = fams[f]
            n_inf = sum(1 for r in fr if r["id"] in predicted_ids)
            counts = Counter(r["gold_label"] for r in fr)
            both = 1 if counts.get("safe", 0) and counts.get("unsafe", 0) else 0
            return (-n_inf / max(len(fr), 1), -both, -len(fr), f)
        fam_list = sorted(fams.keys(), key=fam_priority)
        need = {"cal": {c: dict(QUOTA["cal"]) for c in cells},
                "test": {c: dict(QUOTA["test"]) for c in cells}}
        used_fams = {"cal": set(), "test": set()}
        for f in fam_list:
            fr = fams[f]
            gains = {}
            for split in ("cal", "test"):
                gain = 0
                for r in fr:
                    c = cell_of(r)
                    if c in need[split] and need[split][c].get(r["gold_label"], 0) > 0:
                        gain += 1
                gains[split] = gain
            best = max(gains, key=lambda s: (gains[s], s == "cal"))
            if gains[best] <= 0:
                continue
            for r in fr:
                c = cell_of(r)
                nd = need[best].get(c)
                if nd and nd.get(r["gold_label"], 0) > 0:
                    (cal_rows if best == "cal" else test_rows).append(r)
                    nd[r["gold_label"]] -= 1
            used_fams[best].add(f)
        for c in cells:
            got_cal = {"safe": 50 - need["cal"][c]["safe"], "unsafe": 50 - need["cal"][c]["unsafe"]}
            got_test = {"safe": 100 - need["test"][c]["safe"], "unsafe": 100 - need["test"][c]["unsafe"]}
            ok = got_cal == QUOTA["cal"] and got_test == QUOTA["test"]
            all_ok = all_ok and ok
            print(f"[rebuild] {shift}/{c}: cal={got_cal} test={got_test} {'OK' if ok else 'SHORT'}")

    fam_cal = {r["family_id"] for r in cal_rows}
    fam_test = {r["family_id"] for r in test_rows}
    overlap = fam_cal & fam_test
    print(f"[rebuild] cal={len(cal_rows)} test={len(test_rows)} family_overlap={len(overlap)}")
    if overlap:
        print("[rebuild] WARNING family overlap:", list(overlap)[:5])

    arch = BASE / "manifests" / f"archive_{ts}"
    arch.mkdir(parents=True, exist_ok=True)
    for m in ("frozen_test", "calibration"):
        src = BASE / "manifests" / f"{m}.jsonl"
        if src.exists():
            shutil.move(str(src), str(arch / f"{m}.jsonl"))
    cal_rows.sort(key=lambda r: (r["primary_shift"], cell_of(r), r["gold_label"], r["id"]))
    test_rows.sort(key=lambda r: (r["primary_shift"], cell_of(r), r["gold_label"], r["id"]))
    write_jsonl(BASE / "manifests" / "calibration.jsonl", cal_rows)
    write_jsonl(BASE / "manifests" / "frozen_test.jsonl", test_rows)
    hashes = {"calibration": {"n": len(cal_rows), "sha256": manifest_sha256(cal_rows)},
              "frozen_test": {"n": len(test_rows), "sha256": manifest_sha256(test_rows)},
              "gold_source": "blind_judge_deepseek", "rebuilt_at": ts}
    (BASE / "manifests" / "hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = [r["id"] for r in test_rows + cal_rows]
    print(f"[rebuild] manifests written; unique_ids={len(set(ids)) == len(ids)} total={len(ids)} quota_ok={all_ok}")

    formal = cal_rows + test_rows
    from frauddistill.e4e5_v2.overlap_audit import run_exact_and_near
    nd = run_exact_and_near(registry, formal, BASE / "audits", jaccard_threshold=0.85)
    print("[rebuild] near-duplicate pairs:", nd["near_pairs"])
    print("[rebuild] DONE")

if __name__ == "__main__":
    main()
