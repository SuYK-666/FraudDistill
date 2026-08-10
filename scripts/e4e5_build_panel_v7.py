# -*- coding: utf-8 -*-
"""E4 v2 FINAL panel v7 assembly (correct quota rebuild on v4 gold).

Fixes vs v6:
- uses adjudicated v4 gold for candidates (gold_v4_u1swap/u3swap/u2cand)
- loads candidate content from the manifest files (swap_man bug fixed)
- drops all out_of_scope rows
- fills per-cell quotas: test 100/100, cal 50/50 (U1/U2/U3)
- family/template split checks + content dedupe + exposure audit
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
    ts = time.strftime("%Y%m%d_%H%M%S")
    random.seed(20260810)
    gold_main = {r["id"]: r for r in load_jsonl(BASE / "audits/gold_v4_final.jsonl")}
    gold_u1swap = {r["id"]: r for r in load_jsonl(BASE / "audits/gold_v4_u1swap.jsonl")}
    gold_u3swap = {r["id"]: r for r in load_jsonl(BASE / "audits/gold_v4_u3swap.jsonl")}
    gold_u2cand = {r["id"]: r for r in load_jsonl(BASE / "audits/gold_v4_u2cand.jsonl")}

    panel = {}
    for f in ("frozen_test.jsonl", "calibration.jsonl"):
        for r in load_jsonl(BASE / "manifests" / f):
            r["_pool"] = "panel_test" if f.startswith("frozen") else "panel_cal"
            panel[r["id"]] = apply_gold(dict(r), gold_main[r["id"]])

    # drop out_of_scope rows (they are not fraud-relevant and must not enter the formal test)
    oos = [rid for rid, r in panel.items() if r["gold_type"] == "out_of_scope"]
    for rid in oos:
        del panel[rid]
    print(f"[v7] dropped out_of_scope: {len(oos)}; panel now {len(panel)}")

    registry, loaded = build_registry(REPO)
    def audit_ok(r):
        try:
            return registry.audit_candidate(r)["passed"]
        except Exception:
            return True

    # ---------- candidate pool (content from manifests + gold labels) ----------
    def build_pool(man_name, gold):
        pool = {}
        for r in load_jsonl(BASE / "manifests" / man_name):
            g = gold.get(r["id"])
            if g is None or g["gold_type"] == "out_of_scope":
                continue
            rr = apply_gold(dict(r), g)
            pool[r["id"]] = rr
        return pool
    cand_u1 = build_pool("u1_swap_candidates_v2.jsonl", gold_u1swap)
    cand_u3 = build_pool("u3_swap_candidates_v2.jsonl", gold_u3swap)
    cand_u2 = build_pool("u2_fraud_candidates_v2.jsonl", gold_u2cand)
    print(f"[v7] candidates u1={len(cand_u1)} u3={len(cand_u3)} u2={len(cand_u2)}")

    # ---------- U1 / U3: keep in-panel rows, swap in unsafe to reach quota ----------
    TARGET = {"panel_test": {"safe": 100, "unsafe": 100}, "panel_cal": {"safe": 50, "unsafe": 50}}
    for shift, cand in (("U1_category", cand_u1), ("U3_target_style", cand_u3)):
        cur = {rid: r for rid, r in panel.items() if r["primary_shift"] == shift}
        # group by cell
        cells = defaultdict(list)
        for rid, r in cur.items():
            cells[cell_of(r)].append(r)
        for cell, rows in sorted(cells.items()):
            cnt = Counter((r["_pool"], r["gold_label"]) for r in rows)
            for split in ("panel_test", "panel_cal"):
                for lbl in ("safe", "unsafe"):
                    have = cnt.get((split, lbl), 0)
                    want = TARGET[split][lbl]
                    if have > want:
                        # evict surplus, prefer junk answers then random
                        to_evict = [r for r in rows if r["_pool"] == split and r["gold_label"] == lbl]
                        to_evict.sort(key=lambda r: (0 if JUNK.search(str(r.get("target_model_answer") or "")) else 1, r["id"]))
                        for r in to_evict[have - want:]:
                            del panel[r["id"]]
                            rows.remove(r)
                    elif have < want:
                        need = want - have
                        cands = []
                        for cid, cr in cand.items():
                            if cr["primary_shift"] != shift or cell_of(cr) != cell:
                                continue
                            if cr["gold_label"] != lbl or cr["gold_type"] == "out_of_scope":
                                continue
                            if cid in panel:
                                continue
                            cands.append(cr)
                        random.shuffle(cands)
                        picked = []
                        used_fams = {r.get("family_id") or r["id"] for r in panel.values() if cell_of(r) == cell}
                        for cr in cands:
                            fam = cr.get("family_id") or cr["id"]
                            if fam in used_fams:
                                continue
                            if not audit_ok(cr):
                                continue
                            picked.append(cr)
                            used_fams.add(fam)
                            if len(picked) >= need:
                                break
                        if len(picked) < need:
                            print(f"[v7] WARN {shift}/{cell}/{split}/{lbl}: need {need}, picked {len(picked)}")
                        for cr in picked:
                            nr = dict(cr)
                            nr["_pool"] = split
                            panel[cr["id"]] = nr
                            rows.append(nr)
                        print(f"[v7] {shift}/{cell}/{split}/{lbl}: added {len(picked)} (needed {need})")
        print(f"[v7] {shift}: panel rows = {sum(1 for r in panel.values() if r['primary_shift']==shift)}")

    # ---------- U2: rebuild from fraud-relevant pool ----------
    u2_rows = [r for r in panel.values() if r["primary_shift"] == "U2_source"]
    for rid, r in list(panel.items()):
        if r["primary_shift"] == "U2_source":
            del panel[rid]
    fp = re.compile("|".join(re.escape(k) for k in FRAUD_KW), re.I)
    # in-panel U2 rows that are fraud-relevant stay; others dropped
    kept = [r for r in u2_rows if fp.search(str(r.get("user_query") or ""))]
    print(f"[v7] U2 in-panel fraud-relevant: {len(kept)}/{len(u2_rows)}")
    u2_pool = {r["id"]: r for r in kept}
    for cid, cr in cand_u2.items():
        if cid not in u2_pool:
            u2_pool[cid] = cr
    for cell in ("Alpaca-7B_test", "Alpaca2-7B_test"):
        sel = {"panel_test": {"safe": [], "unsafe": []}, "panel_cal": {"safe": [], "unsafe": []}}
        used_fams = set()
        # first keep already-inferred rows (they have predictions)
        pool_cell = [r for r in u2_pool.values() if cell_of(r) == cell]
        # family-unique first pass
        fam_rows = defaultdict(list)
        for r in pool_cell:
            fam_rows[r.get("family_id") or r["id"]].append(r)
        # prefer rows with predictions
        pred_ids = set()
        for f in ("predictions/final_student.jsonl", "predictions/neural_gold.jsonl", "predictions/neural_softdistill.jsonl"):
            for l in open(BASE / f, encoding="utf-8"):
                pred_ids.add(json.loads(l)["id"])
        for fam, fr in fam_rows.items():
            fr.sort(key=lambda r: (r["id"] not in pred_ids, r["id"]))
        for split, q in (("panel_test", 100), ("panel_cal", 50)):
            for lbl in ("safe", "unsafe"):
                picked = []
                for fam, fr in sorted(fam_rows.items(), key=lambda kv: -len(kv[1])):
                    for r in fr:
                        if r["gold_label"] != lbl:
                            continue
                        if len(picked) >= q:
                            break
                        if r["id"] in {x["id"] for x in picked}:
                            continue
                        picked.append(r)
                    if len(picked) >= q:
                        break
                # relax family-unique if needed
                if len(picked) < q:
                    for fam, fr in fam_rows.items():
                        for r in fr:
                            if r["gold_label"] != lbl or len(picked) >= q:
                                continue
                            if r["id"] in {x["id"] for x in picked}:
                                continue
                            if any(x.get("family_id") == r.get("family_id") for x in picked):
                                continue
                            picked.append(r)
                sel[split][lbl] = picked[:q]
                print(f"[v7] U2/{cell}/{split}/{lbl}: picked {len(picked)}")
        for split in ("panel_test", "panel_cal"):
            for lbl in ("safe", "unsafe"):
                for r in sel[split][lbl]:
                    nr = dict(r)
                    nr["_pool"] = split
                    panel[nr["id"]] = nr
    print(f"[v7] U2 panel rows = {sum(1 for r in panel.values() if r['primary_shift']=='U2_source')}")

    # ---------- split assignment + final checks ----------
    test_rows, cal_rows = [], []
    for r in panel.values():
        (test_rows if r["_pool"] == "panel_test" else cal_rows).append(r)

    def clean(rows):
        seen, out = set(), []
        for r in rows:
            k = content_key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out
    test_rows = clean(test_rows)
    cal_rows = clean(cal_rows)
    test_fam = {r.get("family_id") or r["id"] for r in test_rows}
    cal_fam = {r.get("family_id") or r["id"] for r in cal_rows}
    print(f"[v7] test={len(test_rows)} cal={len(cal_rows)} fam_overlap={len(test_fam & cal_fam)}")
    for shift in ("U1_category", "U2_source", "U3_target_style"):
        for split, rs in (("test", test_rows), ("cal", cal_rows)):
            sub = [r for r in rs if r["primary_shift"] == shift]
            c = Counter((cell_of(r), r["gold_label"]) for r in sub)
            print(f"[v7] {shift}/{split}: {dict(c)}")

    # archive current manifests and write new ones
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
    print("[v7] DONE", json.dumps(hashes, ensure_ascii=False))

if __name__ == "__main__":
    main()
