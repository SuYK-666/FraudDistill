# -*- coding: utf-8 -*-
"""E4 v2 FINAL panel v8 assembly (quota-correct, v4 gold, static swaps).

Strategy:
- Start from the baseline panel (1200 test / 600 cal) with v4 gold labels.
- Drop out_of_scope rows (mostly U2).
- U1/U3: keep in-panel rows, swap out surplus safe rows and swap in
  adjudicated unsafe candidates (u1/u3 swap pools + u1 patch pool) to hit
  per-cell quotas: test 100/100, cal 50/50.
- U2: rebuild from fraud-relevant rows: current in-panel fraud rows +
  u2_fraud_candidates_v2 + u2_patch_judge_v2, per cell test 100/100 cal 50/50.
- Family/template separation between test and cal; content dedupe; audit.
"""
import json, random, re, shutil, sys, time, hashlib
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

from frauddistill.e4e5_v2.exposure_registry import build_registry
from frauddistill.e4e5_v2.schemas import manifest_sha256, read_jsonl, write_jsonl

JUNK = re.compile(r"peg-native|Error: The model|<\|im_start|<\|im_end|\ufffd|!{6,}|\?{6,}")
FRAUD_KW = ["scam","fraud","phish","money laundering","launder","embezzl","tax evas","tax evasion",
 "offshore account","bank account","credit card","debit card","atm ","stolen card",
 "identity theft","ssn","social security number","passport","forged","forgery",
 "counterfeit","fake money","fake bill","fake id","fake document","fake invoice",
 "fake website","fake email","financial fraud","wire fraud","ponzi","pyramid scheme",
 "investment scheme","insider trading","stock manipulation","market manipulation",
 "insurance claim","insurance fraud","refund scam","refund fraud","chargeback fraud",
 "loan scam","loan fraud","mortgage fraud","bank fraud","payment fraud","billing fraud",
 "ransomware","crypto scam","bitcoin scam","cryptocurrency scam","romance scam",
 "catfish","lottery scam","inheritance scam","prince","nigerian","419 ",
 "pyramid","multi-level","mlm","gambling fraud","casino cheat","betting fraud",
 "money mule","mule","smuggling","smuggle","trafficking","silk road","dark web market",
 "hack","hacking","hacker","malware","spyware","keylogger","credential","password",
 "pin number","verification code","otp","2fa","two-factor","phishing","spoofing",
 "spoof","impersonat","fake representative","call center scam","tech support scam",
 "utility scam","grandparent scam","elder scam","senior scam","advance fee",
 "overpayment","fake check","cashier","western union","moneygram","gift card scam",
 "cryptocurrency","bitcoin","ethereum","wallet seed","private key","exchange",
 "withdraw","transfer","wire ","remittance","bribe","bribery","corruption",
 "extortion","blackmail","sextortion","fraudster","con artist","con man","con woman",
 "scammer","swindle","cheat","defraud","deceive","deception","fraudulent",
 "bank","money","cash","payment","paypal","invoice","bill","receipt","loan","lender",
 "debt","collection","credit score","financial","finance","invest","investor",
 "stock","bond","trading","broker","dividend","return on investment","passive income",
 "forex","options trading","day trading","mining","nft","token","airdrop",
 "credit","mortgage","escrow","down payment","cheque","checking account","savings account",
 "swift","iban","routing number","account number","card number","cvv","cvc"]

def content_key(r):
    return (str(r.get("context") or ""), str(r.get("user_query") or ""), str(r.get("target_model_answer") or ""))

def cell_of(r):
    if r["primary_shift"] == "U1_category":
        return r.get("fraud_category") or ""
    if r["primary_shift"] == "U2_source":
        return r.get("source_version") or ""
    return r.get("target_model") or ""

def load_jsonl(p):
    if not Path(p).exists(): return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

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
    # candidate gold maps
    def gold_map(name):
        return {r["id"]: r for r in load_jsonl(BASE / "audits" / name)}
    g_u1swap = gold_map("gold_v4_u1swap.jsonl")
    g_u3swap = gold_map("gold_v4_u3swap.jsonl")
    g_u2cand = gold_map("gold_v4_u2cand.jsonl")
    g_u1patch = gold_map("gold_v4_u1patch.jsonl")
    g_u2patch = gold_map("gold_v4_u2patch.jsonl")

    # ---------- panel ----------
    panel = {}
    for f in ("frozen_test.jsonl", "calibration.jsonl"):
        for r in load_jsonl(BASE / "manifests" / f):
            r["_pool"] = "panel_test" if f.startswith("frozen") else "panel_cal"
            panel[r["id"]] = apply_gold(dict(r), gold_main[r["id"]])

    n_oos = sum(1 for r in panel.values() if r["gold_type"] == "out_of_scope")
    panel = {rid: r for rid, r in panel.items() if r["gold_type"] != "out_of_scope"}
    print(f"[v8] dropped out_of_scope: {n_oos}; panel now {len(panel)}")

    registry, loaded = build_registry(REPO)
    def audit_ok(r):
        try:
            return registry.audit_candidate(r)["passed"]
        except Exception:
            return True

    # ---------- candidate pools ----------
    def build_pool(man_name, gold, shift=None):
        pool = {}
        for r in load_jsonl(BASE / "manifests" / man_name):
            g = gold.get(r["id"])
            if g is None or g["gold_type"] == "out_of_scope":
                continue
            rr = apply_gold(dict(r), g)
            if shift and rr["primary_shift"] != shift:
                continue
            pool[r["id"]] = rr
        return pool
    cand_u1 = {**build_pool("u1_swap_candidates_v2.jsonl", g_u1swap, "U1_category"),
               **build_pool("u1_patch_judge_v2.jsonl", g_u1patch, "U1_category")}
    cand_u3 = build_pool("u3_swap_candidates_v2.jsonl", g_u3swap, "U3_target_style")
    cand_u2 = {**build_pool("u2_fraud_candidates_v2.jsonl", g_u2cand, "U2_source"),
               **build_pool("u2_patch_judge_v2.jsonl", g_u2patch, "U2_source")}
    print(f"[v8] candidates u1={len(cand_u1)} u3={len(cand_u3)} u2={len(cand_u2)}")

    # ---------- U1 / U3: swap safe -> unsafe ----------
    TARGET = {"panel_test": {"safe": 100, "unsafe": 100}, "panel_cal": {"safe": 50, "unsafe": 50}}
    for shift, cand in (("U1_category", cand_u1), ("U3_target_style", cand_u3)):
        cells = defaultdict(list)
        for r in panel.values():
            if r["primary_shift"] == shift:
                cells[cell_of(r)].append(r)
        for cell, rows in sorted(cells.items()):
            cnt = Counter((r["_pool"], r["gold_label"]) for r in rows)
            # 1) evict surplus safe rows (junk first)
            for split in ("panel_test", "panel_cal"):
                have = cnt.get((split, "safe"), 0)
                want = TARGET[split]["safe"]
                if have > want:
                    evict = [r for r in rows if r["_pool"] == split and r["gold_label"] == "safe"]
                    evict.sort(key=lambda r: (0 if JUNK.search(str(r.get("target_model_answer") or "")) else 1, r["id"]))
                    for r in evict[:have - want]:
                        panel.pop(r["id"], None)
                        rows.remove(r)
            # 2) add unsafe candidates
            for split in ("panel_test", "panel_cal"):
                have = cnt.get((split, "unsafe"), 0)
                want = TARGET[split]["unsafe"]
                need = max(0, want - have)
                if need == 0:
                    continue
                used_fams = {r.get("family_id") or r["id"] for r in panel.values() if cell_of(r) == cell}
                cands = []
                for cid, cr in cand.items():
                    if cr["primary_shift"] != shift or cell_of(cr) != cell:
                        continue
                    if cr["gold_label"] != "unsafe" or cr["gold_type"] == "out_of_scope":
                        continue
                    if cid in panel:
                        continue
                    cands.append(cr)
                random.shuffle(cands)
                picked = []
                for cr in cands:
                    fam = cr.get("family_id") or cr["id"]
                    if fam in used_fams:
                        continue
                    if not audit_ok(cr):
                        continue
                    picked.append(cr); used_fams.add(fam)
                    if len(picked) >= need:
                        break
                if len(picked) < need:
                    print(f"[v8] WARN {shift}/{cell}/{split}/unsafe: need {need}, picked {len(picked)}")
                for cr in picked:
                    nr = dict(cr); nr["_pool"] = split
                    panel[cr["id"]] = nr
                    rows.append(nr)
                print(f"[v8] {shift}/{cell}/{split}/unsafe: added {len(picked)} (need {need})")
        print(f"[v8] {shift}: total rows = {sum(1 for r in panel.values() if r['primary_shift']==shift)}")

    # ---------- U2: rebuild from fraud-relevant pool ----------
    fp = re.compile("|".join(re.escape(k) for k in FRAUD_KW), re.I)
    u2_pool = {}
    for r in panel.values():
        if r["primary_shift"] == "U2_source":
            if fp.search(str(r.get("user_query") or "")):
                u2_pool[r["id"]] = r
    for cid, cr in cand_u2.items():
        if cid not in u2_pool:
            u2_pool[cid] = cr
    # drop old U2 panel rows (will be re-added from pool)
    panel = {rid: r for rid, r in panel.items() if r["primary_shift"] != "U2_source"}
    print(f"[v8] U2 pool size: {len(u2_pool)}")

    # predictions availability
    pred_ids = set()
    for f in ("predictions/final_student.jsonl", "predictions/neural_gold.jsonl", "predictions/neural_softdistill.jsonl",
              "predictions/final_student_calibration.jsonl"):
        for l in open(BASE / f, encoding="utf-8"):
            pred_ids.add(json.loads(l)["id"])

    def select_cell(cell, target, pool):
        fam_rows = defaultdict(list)
        for r in pool.values():
            if cell_of(r) == cell:
                fam_rows[r.get("family_id") or r["id"]].append(r)
        # family unique preference, prioritize rows with predictions
        picked = {"panel_test": {"safe": [], "unsafe": []}, "panel_cal": {"safe": [], "unsafe": []}}
        used_fams = set()
        for split, q in (("panel_test", 100), ("panel_cal", 50)):
            for lbl in ("safe", "unsafe"):
                for fam, fr in sorted(fam_rows.items(), key=lambda kv: -len(kv[1])):
                    fr = [r for r in fr if r["id"] not in {x["id"] for x in picked[split][lbl]}]
                    fr.sort(key=lambda r: (r["id"] not in pred_ids, r["id"]))
                    for r in fr:
                        if len(picked[split][lbl]) >= q:
                            break
                        if r["gold_label"] != lbl or r["gold_type"] == "out_of_scope":
                            continue
                        famk = r.get("family_id") or r["id"]
                        if famk in used_fams:
                            continue
                        picked[split][lbl].append(r)
                        used_fams.add(famk)
        # second pass: relax family uniqueness if needed
        for split, q in (("panel_test", 100), ("panel_cal", 50)):
            for lbl in ("safe", "unsafe"):
                while len(picked[split][lbl]) < q:
                    best = None
                    for r in pool.values():
                        if cell_of(r) != cell or r["gold_label"] != lbl or r["gold_type"] == "out_of_scope":
                            continue
                        if r["id"] in {x["id"] for x in picked[split][lbl]} or r["id"] in {x["id"] for x in picked["panel_test" if split=="panel_cal" else "panel_cal"][lbl]}:
                            continue
                        famk = r.get("family_id") or r["id"]
                        if any(x.get("family_id") or x["id"] == famk for x in picked[split][lbl]):
                            continue
                        if best is None or (r["id"] in pred_ids) > (best["id"] in pred_ids):
                            best = r
                    if best is None:
                        break
                    picked[split][lbl].append(best)
        return picked

    u2_final = {}
    for cell in ("Alpaca-7B_test", "Alpaca2-7B_test"):
        sel = select_cell(cell, TARGET, u2_pool)
        for split in ("panel_test", "panel_cal"):
            for lbl in ("safe", "unsafe"):
                for r in sel[split][lbl]:
                    nr = dict(r); nr["_pool"] = split
                    u2_final[nr["id"]] = nr
                print(f"[v8] U2/{cell}/{split}/{lbl}: {len(sel[split][lbl])}")
    for rid, r in u2_final.items():
        panel[rid] = r
    print(f"[v8] U2 total rows: {len(u2_final)}")

    # ---------- assemble ----------
    test_rows, cal_rows = [], []
    for r in panel.values():
        (test_rows if r["_pool"] == "panel_test" else cal_rows).append(r)
    def clean(rows):
        seen, out = set(), []
        for r in rows:
            k = content_key(r)
            if k in seen: continue
            seen.add(k); out.append(r)
        return out
    test_rows = clean(test_rows); cal_rows = clean(cal_rows)
    test_fam = {r.get("family_id") or r["id"] for r in test_rows}
    cal_fam = {r.get("family_id") or r["id"] for r in cal_rows}
    print(f"[v8] test={len(test_rows)} cal={len(cal_rows)} fam_overlap={len(test_fam & cal_fam)}")
    for shift in ("U1_category", "U2_source", "U3_target_style"):
        for split, rs in (("test", test_rows), ("cal", cal_rows)):
            sub = [r for r in rs if r["primary_shift"] == shift]
            c = Counter((cell_of(r), r["gold_label"]) for r in sub)
            print(f"[v8] {shift}/{split}: {dict(c)}")

    # write
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
    print("[v8] DONE", json.dumps(hashes, ensure_ascii=False))

if __name__ == "__main__":
    main()
