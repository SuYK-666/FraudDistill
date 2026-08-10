# -*- coding: utf-8 -*-
"""E4 v2 FINAL panel v11 assembly (quota-exact, v4 gold, family-separated).

Design (guide-compliant):
  test = 1200  -> 6 cells x (100 safe + 100 unsafe)
  cal  = 600   -> 6 cells x (50 safe + 50 unsafe)
  cells: U1 {elder_health_product, naked_chat_sextortion},
         U2 {Alpaca-7B_test, Alpaca2-7B_test},
         U3 {phi3_5_mini, smollm2_1_7b}

Steps:
  1. Start from the restored baseline (1200/600, v4 gold). Drop out_of_scope.
  2. Evict surplus safe rows (junk-first) per cell to reach quota.
  3. Fill unsafe (and U2-safe) deficits from adjudicated candidate pools,
     enforcing family/template separation between test and cal.
  4. Content dedupe + exposure-registry audit; write manifests + hashes.
"""
from __future__ import annotations
import json, random, re, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

from frauddistill.e4e5_v2.exposure_registry import build_registry
from frauddistill.e4e5_v2.schemas import manifest_sha256, write_jsonl

JUNK = re.compile(r"peg-native|Error: The model|<\|im_start|<\|im_end|\ufffd|!{6,}|\?{6,}")

QUOTA = {"panel_test": {"safe": 100, "unsafe": 100},
         "panel_cal": {"safe": 50, "unsafe": 50}}

# cell -> (manifests + audit-gold files) for candidate pools
POOLS = [
    ("u1_swap_candidates_v2.jsonl", "gold_v4_u1swap.jsonl"),
    ("u1_patch_judge_v2.jsonl", "gold_v4_u1patch.jsonl"),
    ("u1_naked_cal_patch.jsonl", "gold_v4_u1naked.jsonl"),
    ("u1_rebalance_candidates_v2.jsonl", "gold_v4_u1reb2.jsonl"),
    ("u1_naked_test_extra_v2.jsonl", "gold_v4_u1extra.jsonl"),
    ("u3_swap_candidates_v2.jsonl", "gold_v4_u3swap.jsonl"),
    ("u3_cal_patch.jsonl", "gold_v4_u3cal.jsonl"),
    ("u2_fraud_candidates_v2.jsonl", "gold_v4_u2cand.jsonl"),
    ("u2_patch_judge_v2.jsonl", "gold_v4_u2patch.jsonl"),
]

def load_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def cell_of(r):
    if r["primary_shift"] == "U1_category":
        return r.get("fraud_category") or ""
    if r["primary_shift"] == "U2_source":
        return r.get("source_version") or ""
    return r.get("target_model") or ""

def famkey(r):
    return r.get("family_id") or r["id"]

def content_key(r):
    return (str(r.get("context") or ""), str(r.get("user_query") or ""), str(r.get("target_model_answer") or ""))

def apply_gold(r, g):
    r["gold_label"] = g["gold_label"]
    r["gold_type"] = g["gold_type"]
    r["gold_binary"] = 1 if g["gold_label"] == "unsafe" else 0
    r["gold_source"] = g.get("gold_source", "dual_judge_v4")
    r["exposure_level"] = "L3"
    return r

def main():
    random.seed(20260810)
    gold_main = {r["id"]: r for r in load_jsonl(BASE / "audits/gold_v4_final.jsonl")}

    # ---------- 1. load baseline panel ----------
    panel = {}
    for f in ("frozen_test.jsonl", "calibration.jsonl"):
        for r in load_jsonl(BASE / "manifests" / f):
            g = gold_main.get(r["id"])
            if g is None:
                print(f"[v11] WARN no v4 gold for {r['id']} in {f}")
                continue
            rr = apply_gold(dict(r), g)
            rr["_pool"] = "panel_test" if f.startswith("frozen") else "panel_cal"
            panel[rr["id"]] = rr
    n0 = len(panel)
    oos = {rid for rid, r in panel.items() if r["gold_type"] == "out_of_scope"}
    panel = {rid: r for rid, r in panel.items() if rid not in oos}
    print(f"[v11] baseline: {n0} rows, dropped oos={len(oos)}, kept={len(panel)}")

    # ---------- 2. candidate pools ----------
    cand = {}
    for man, goldf in POOLS:
        gm = {r["id"]: r for r in load_jsonl(BASE / "audits" / goldf)}
        for r in load_jsonl(BASE / "manifests" / man):
            g = gm.get(r["id"])
            if g is None or g["gold_type"] == "out_of_scope":
                continue
            if r["id"] in panel:
                continue
            cand[r["id"]] = apply_gold(dict(r), g)
    print(f"[v11] fresh candidates: {len(cand)}")

    # exposure audit (skip failures; log)
    registry, _ = build_registry(REPO)
    def audit_ok(r):
        try:
            return bool(registry.audit_candidate(r)["passed"])
        except Exception as e:
            print(f"[v11] audit exception {r['id']}: {e}")
            return False
    audited = {cid: cr for cid, cr in cand.items() if audit_ok(cr)}
    print(f"[v11] candidates passing exposure audit: {len(audited)}/{len(cand)}")
    cand = audited

    # content dedupe vs baseline
    base_keys = {content_key(r) for r in panel.values()}
    cand = {cid: cr for cid, cr in cand.items() if content_key(cr) not in base_keys}
    print(f"[v11] candidates after baseline content dedupe: {len(cand)}")

    # ---------- 3. per-cell selection ----------
    def by_cell(rows):
        out = defaultdict(list)
        for r in rows:
            out[cell_of(r)].append(r)
        return out

    test_fams = {famkey(r) for r in panel.values() if r["_pool"] == "panel_test"}
    cal_fams = {famkey(r) for r in panel.values() if r["_pool"] == "panel_cal"}

    added = {}   # id -> row (candidate additions)
    for shift in ("U1_category", "U2_source", "U3_target_style"):
        shift_rows = [r for r in panel.values() if r["primary_shift"] == shift]
        cells = by_cell(shift_rows)
        for cell in sorted(cells):
            rows = cells[cell]
            test_rows = [r for r in rows if r["_pool"] == "panel_test"]
            cal_rows = [r for r in rows if r["_pool"] == "panel_cal"]
            cands = [cand[cid] for cid in cand if cand[cid]["primary_shift"] == shift and cell_of(cand[cid]) == cell]

            # --- evict surplus safe (junk-first) ---
            for split, q in (("panel_cal", 50), ("panel_test", 100)):
                safe = [r for r in rows if r["_pool"] == split and r["gold_label"] == "safe"]
                over = len(safe) - q
                if over > 0:
                    safe.sort(key=lambda r: (0 if JUNK.search(str(r.get("target_model_answer") or "")) else 1, r["id"]))
                    for r in safe[:over]:
                        panel.pop(r["id"], None)
            rows = [r for r in panel.values() if r["primary_shift"] == shift and cell_of(r) == cell]
            test_rows = [r for r in rows if r["_pool"] == "panel_test"]
            cal_rows = [r for r in rows if r["_pool"] == "panel_cal"]

            # --- fill cal deficits first (unsafe then safe), family not in test ---
            def counts(rs, lbl):
                return sum(1 for r in rs if r["gold_label"] == lbl)

            for lbl in ("unsafe", "safe"):
                need_cal = QUOTA["panel_cal"][lbl] - counts(cal_rows, lbl)
                if need_cal <= 0:
                    continue
                pool_lbl = [cr for cr in cands if cr["gold_label"] == lbl and cr["id"] not in added]
                pool_lbl.sort(key=lambda cr: (famkey(cr) in test_fams, cr["id"]))
                for cr in pool_lbl:
                    if need_cal == 0:
                        break
                    f = famkey(cr)
                    if f in test_fams:
                        continue
                    nr = dict(cr); nr["_pool"] = "panel_cal"
                    panel[nr["id"]] = nr; added[nr["id"]] = nr
                    cal_fams.add(f); need_cal -= 1

            # --- then test deficits (family not in cal) ---
            for lbl in ("unsafe", "safe"):
                need_test = QUOTA["panel_test"][lbl] - counts(test_rows, lbl)
                if need_test <= 0:
                    continue
                pool_lbl = [cr for cr in cands if cr["gold_label"] == lbl and cr["id"] not in added]
                pool_lbl.sort(key=lambda cr: (famkey(cr) in cal_fams, cr["id"]))
                skipped = 0
                for cr in pool_lbl:
                    if need_test == 0:
                        break
                    f = famkey(cr)
                    if f in cal_fams:
                        skipped += 1
                        continue
                    nr = dict(cr); nr["_pool"] = "panel_test"
                    panel[nr["id"]] = nr; added[nr["id"]] = nr
                    test_fams.add(f); need_test -= 1
                if need_test > 0:
                    print(f"[v11] DEBUG {shift}/{cell}/{lbl}: need_test left={need_test} pool={len(pool_lbl)} skipped_fam_cal={skipped} total_cands={len(cands)}")

            # report remaining deficits
            rows = [r for r in panel.values() if r["primary_shift"] == shift and cell_of(r) == cell]
            test_rows = [r for r in rows if r["_pool"] == "panel_test"]
            cal_rows = [r for r in rows if r["_pool"] == "panel_cal"]
            t = Counter(r["gold_label"] for r in test_rows)
            c = Counter(r["gold_label"] for r in cal_rows)
            print(f"[v11] {shift}/{cell}: test={dict(t)} cal={dict(c)} added={sum(1 for x in rows if x['id'] in added)}")

    # ---------- 4. assemble ----------
    test_rows = [r for r in panel.values() if r["_pool"] == "panel_test"]
    cal_rows = [r for r in panel.values() if r["_pool"] == "panel_cal"]

    def clean(rows):
        seen, out = set(), []
        for r in rows:
            k = content_key(r)
            if k in seen:
                continue
            seen.add(k); out.append(r)
        return out

    test_rows = clean(test_rows); cal_rows = clean(cal_rows)
    tf = {famkey(r) for r in test_rows}; cf = {famkey(r) for r in cal_rows}
    print(f"[v11] final: test={len(test_rows)} cal={len(cal_rows)} fam_overlap={len(tf & cf)}")

    # verification table
    for shift in ("U1_category", "U2_source", "U3_target_style"):
        for split, rs in (("test", test_rows), ("cal", cal_rows)):
            sub = [r for r in rs if r["primary_shift"] == shift]
            c = Counter((cell_of(r), r["gold_label"]) for r in sub)
            print(f"[v11] verify {shift}/{split}: {dict(c)}")

    # archive current manifests, then write
    ts = time.strftime("%Y%m%d_%H%M%S")
    arch = BASE / "manifests" / f"archive_{ts}"
    arch.mkdir(parents=True, exist_ok=True)
    for m in ("frozen_test", "calibration"):
        src = BASE / "manifests" / f"{m}.jsonl"
        if src.exists():
            shutil.move(str(src), str(arch / f"{m}.jsonl"))
    test_rows.sort(key=lambda r: (r["primary_shift"], cell_of(r), r["gold_label"], r["id"]))
    cal_rows.sort(key=lambda r: (r["primary_shift"], cell_of(r), r["gold_label"], r["id"]))
    write_jsonl(BASE / "manifests" / "frozen_test.jsonl", test_rows)
    write_jsonl(BASE / "manifests" / "calibration.jsonl", cal_rows)
    hashes = {"frozen_test": {"n": len(test_rows), "sha256": manifest_sha256(test_rows)},
              "calibration": {"n": len(cal_rows), "sha256": manifest_sha256(cal_rows)}}
    (BASE / "manifests" / "hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    # save selection log
    sel = [{"id": rid, "pool": r["_pool"], "cell": cell_of(r), "label": r["gold_label"], "family_id": famkey(r),
            "source_pool": r.get("source") or r.get("source_version"), "primary_shift": r["primary_shift"]}
           for rid, r in added.items()]
    sel.sort(key=lambda x: (x["primary_shift"], x["cell"], x["pool"], x["label"], x["id"]))
    (BASE / "manifests" / "panel_v11_additions.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in sel) + "\n", encoding="utf-8")
    print("[v11] DONE", json.dumps(hashes, ensure_ascii=False))
    print(f"[v11] additions saved: {len(sel)} -> manifests/panel_v11_additions.jsonl")

if __name__ == "__main__":
    main()
