# -*- coding: utf-8 -*-
"""E4 v2 FINAL panel v6 assembly.

- U1/U3: keep in-panel rows, swap out safe rows for adjudicated unsafe
  candidates to restore 50/50 per cell (per guide 5.4).
- U2: rebuild from fraud-relevant pku rows only (current panel in-scope rows
  + adjudicated candidates), 50/50 per source cell.
- Exposure audit + family/template checks + exact dedupe; archive old panel.
"""
from __future__ import annotations
import json, random, re, shutil, sys, time, hashlib
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

from frauddistill.e4e5_v2.exposure_registry import build_registry  # noqa
from frauddistill.e4e5_v2.schemas import manifest_sha256, read_jsonl, write_jsonl  # noqa

JUNK = re.compile(r"peg-native|Error: The model|<\|im_start|<\|im_end|\ufffd|!{6,}|\?{6,}")

def content_key(r):
    return (str(r.get("context") or ""), str(r.get("user_query") or ""), str(r.get("target_model_answer") or ""))

def cell_of(r):
    if r["primary_shift"] == "U1_category":
        return r.get("fraud_category") or ""
    if r["primary_shift"] == "U2_source":
        sv = str(r.get("source_version") or "")
        if sv.endswith(".jsonl"): sv = sv[:-7]
        return sv
    if r["primary_shift"] == "U3_target_style":
        return r.get("target_model") or ""
    return ""

def load_jsonl(p):
    if not Path(p).exists(): return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def load_gold(p):
    return {r["id"]: r for r in load_jsonl(p)}

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
    gold_main = load_gold(BASE / "audits/gold_v4_final.jsonl")
    gold_u1swap = load_gold(BASE / "audits/gold_v4_u1swap.jsonl")
    gold_u3swap = load_gold(BASE / "audits/gold_v4_u3swap.jsonl")
    gold_u2cand = load_gold(BASE / "audits/gold_v4_u2cand.jsonl")

    panel = {}
    for f in ("frozen_test.jsonl", "calibration.jsonl"):
        for r in load_jsonl(BASE / "manifests" / f):
            r["_pool"] = "panel_test" if f.startswith("frozen") else "panel_cal"
            panel[r["id"]] = r
    print("[v6] panel loaded:", len(panel))

    registry, loaded = build_registry(REPO)
    def audit_ok(r):
        try:
            return registry.audit_candidate(r)["passed"]
        except Exception:
            return True

    # ---------- U1 / U3: swap safe -> unsafe ----------
    swap_man = {}
    for name, key in (("u1_swap_candidates_v2.jsonl", "fraud_category"), ("u3_swap_candidates_v2.jsonl", "target_model")):
        for r in load_jsonl(BASE / "manifests" / name):
            swap_man[r["id"]] = r
    swaps = {"U1_category": (gold_u1swap, "fraud_category"),
             "U3_target_style": (gold_u3swap, "target_model")}
    for shift, (gpool, cellkey) in swaps.items():
        cur = [r for r in panel.values() if r["primary_shift"] == shift]
        need = {}
        for r in cur:
            c = cell_of(r)
            need.setdefault(c, {"safe": 0, "unsafe": 0})
            need[c][apply_gold(dict(r), gold_main[r["id"]])["gold_label"]] += 1
        # per-cell target: 150 safe + 150 unsafe (test 100/100 + cal 50/50)
        candidates = {c: [] for c in need}
        for i, g in gpool.items():
            r = swap_man.get(i) or {}
            c = str(r.get(cellkey) or "") if r else ""
            if c in candidates and g["gold_label"] == "unsafe" and g["gold_type"] != "out_of_scope":
                candidates[c].append(i)
        remove = set()
        for c, cnt in need.items():
            short = 150 - cnt["unsafe"]
            if short <= 0:
                continue
            cand = candidates.get(c) or []
            # prefer rows not already in panel; audit pass
            cand_ok = []
            for i in cand:
                if i in panel: continue
                row = dict(gpool[i]); row["id"] = i
                if audit_ok(row):
                    cand_ok.append(i)
            if len(cand_ok) < short:
                print(f"[v6] WARN {shift}/{c}: need {short} unsafe swaps, have {len(cand_ok)}")
            # choose safe rows to remove: prefer junk answers first, then families with duplicates, then random
            safe_rows = [r for r in cur if apply_gold(dict(r), gold_main[r["id"]])["gold_label"] == "safe"]
            safe_rows.sort(key=lambda r: (0 if JUNK.search(str(r.get("target_model_answer") or "")) else 1,
                                          r["id"]))
            picked = cand_ok[:short]
            for i in picked:
                # find a safe row to evict (same cell, preferably junk)
                evict = None
                for r in safe_rows:
                    if cell_of(r) == c and r["id"] not in remove:
                        evict = r
                        break
                if evict is None:
                    print(f"[v6] WARN no evictable safe row for {c}")
                    continue
                remove.add(evict["id"])
                nrow = dict(gpool[i])
                nrow["id"] = i
                nrow["primary_shift"] = shift
                if shift == "U1_category":
                    nrow["fraud_category"] = c
                    nrow["target_model"] = "SmolLM2-1.7B-Instruct"
                else:
                    nrow["target_model"] = c
                nrow["source"] = "u3_leftover_candidates_v2" if shift == "U3_target_style" else "u1_leftover_candidates_v2"
                panel[i] = apply_gold(nrow, gpool[i])
            print(f"[v6] {shift}/{c}: swapped {len(picked)}; need was {short}")
        for rid in remove:
            del panel[rid]
        print(f"[v6] {shift}: panel now {len([1 for r in panel.values() if r['primary_shift']==shift])}")

    # ---------- U2: rebuild from fraud-relevant rows ----------
    cur_u2 = [apply_gold(dict(r), gold_main[r["id"]]) for r in panel.values() if r["primary_shift"] == "U2_source"]
    cand_u2 = []
    for i, g in gold_u2cand.items():
        r = dict(g); r["id"] = i
        r["primary_shift"] = "U2_source"
        r["source"] = "pku_saferlhf"
        cand_u2.append(apply_gold(r, g))
    # fraud-relevance filter (same lexicon as pool builder)
    _fraud_kw = ["scam","fraud","phish","money laundering","launder","embezzl","tax evas","tax evasion",
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
    _fp = re.compile("|".join(re.escape(k) for k in _fraud_kw), re.I)
    in_scope = lambda r: r.get("gold_type") != "out_of_scope" and bool(_fp.search(str(r.get("user_query") or "")))
    cur_u2 = [r for r in cur_u2 if in_scope(r)]
    cand_u2 = [r for r in cand_u2 if in_scope(r) and r["id"] not in panel]
    print(f"[v6] U2 in-scope: current={len(cur_u2)} candidates={len(cand_u2)}")
    for cell in ("Alpaca-7B_test", "Alpaca2-7B_test"):
        n_cur = sum(1 for r in cur_u2 if cell_of(r) == cell)
        n_cand = sum(1 for r in cand_u2 if cell_of(r) == cell)
        print(f"  {cell}: current={n_cur} cand={n_cand}")
    # select: current first (no re-inference), then candidates; per cell 150/150
    selected = {}
    for cell in ("Alpaca-7B_test", "Alpaca2-7B_test"):
        pool_cell = [r for r in cur_u2 + cand_u2 if cell_of(r) == cell]
        # dedupe by content
        seen_c, uniq = set(), []
        for r in pool_cell:
            k = content_key(r)
            if k in seen_c: continue
            seen_c.add(k); uniq.append(r)
        # family-unique preference + audit
        cur_ids = {r["id"] for r in cur_u2}
        ok_rows = [r for r in uniq if r["id"] in cur_ids or r["id"] not in panel]
        fams = defaultdict(list)
        for r in ok_rows:
            fams[str(r.get("family_id") or "")].append(r)
        sel, used_fam = [], set()
        # first pass: one row per family
        for f, fr in sorted(fams.items(), key=lambda kv: len(kv[1])):
            for r in fr:
                if r["id"] in [s["id"] for s in sel]: continue
                sel.append(r); used_fam.add(f)
                break
        # second pass: fill quotas
        need = {"safe": 150, "unsafe": 150}
        quota_sel = []
        for r in sel:
            if need.get(r["gold_label"], 0) > 0:
                quota_sel.append(r); need[r["gold_label"]] -= 1
        for f, fr in fams.items():
            if f in used_fam: continue
            for r in fr:
                if need.get(r["gold_label"], 0) <= 0: continue
                quota_sel.append(r); need[r["gold_label"]] -= 1
                used_fam.add(f)
                break
        # random fill if still short
        random.shuffle(fams)
        for f, fr in fams.items():
            for r in fr:
                if need.get(r["gold_label"], 0) <= 0: continue
                if r["id"] in [s["id"] for s in quota_sel]: continue
                quota_sel.append(r); need[r["gold_label"]] -= 1
        print(f"[v6] U2 {cell}: selected {len(quota_sel)} remaining need {need}")
        selected[cell] = quota_sel
    # assemble U2 with split assignment: per cell 100/100 test, 50/50 cal
    u2_final = []
    for cell, sel in selected.items():
        safe = [r for r in sel if r["gold_label"] == "safe"]
        unsafe = [r for r in sel if r["gold_label"] == "unsafe"]
        # prefer already-predicted rows for test
        pred_ids = set()
        for f in ("predictions/final_student.jsonl", "predictions/neural_gold.jsonl", "predictions/neural_softdistill.jsonl"):
            for l in open(BASE / f, encoding="utf-8"):
                pred_ids.add(json.loads(l)["id"])
        safe.sort(key=lambda r: (r["id"] not in pred_ids, r["id"]))
        unsafe.sort(key=lambda r: (r["id"] not in pred_ids, r["id"]))
        test = safe[:100] + unsafe[:100]
        cal = safe[100:150] + unsafe[100:150]
        for r in test + cal:
            r["_split"] = "test" if r in test else "cal"
        u2_final.extend(test + cal)
    print(f"[v6] U2 final: {len(u2_final)}")

    # ---------- assemble final test/cal ----------
    test_rows, cal_rows = [], []
    for r in panel.values():
        if r["primary_shift"] == "U2_source":
            continue  # replaced above
        g = gold_main.get(r["id"])
        if g is None: continue
        rr = apply_gold(dict(r), g)
        if rr["gold_type"] == "out_of_scope":
            continue
        (test_rows if r["_pool"] == "panel_test" else cal_rows).append(rr)
    for r in u2_final:
        (test_rows if r["_split"] == "test" else cal_rows).append(r)
    # content dedupe + audit
    def clean(rows):
        seen, out = set(), []
        for r in rows:
            k = content_key(r)
            if k in seen: continue
            seen.add(k)
            if audit_ok(r):
                out.append(r)
        return out
    test_rows = clean(test_rows)
    cal_rows = clean(cal_rows)
    # split overlap check
    test_fam = {r.get("family_id") or r["id"] for r in test_rows}
    cal_fam = {r.get("family_id") or r["id"] for r in cal_rows}
    print(f"[v6] test={len(test_rows)} cal={len(cal_rows)} fam_overlap={len(test_fam & cal_fam)}")
    # quotas check
    for shift in ("U1_category", "U2_source", "U3_target_style"):
        for split, rs in (("test", test_rows), ("cal", cal_rows)):
            sub = [r for r in rs if r["primary_shift"] == shift]
            c = Counter((cell_of(r), r["gold_label"]) for r in sub)
            print(f"[v6] {shift}/{split}: {dict(c)}")
    ts2 = time.strftime("%Y%m%d_%H%M%S")
    arch = BASE / "manifests" / f"archive_{ts2}"
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
    with open(BASE / "manifests" / "hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)
    print("[v6] DONE", json.dumps(hashes, ensure_ascii=False))

if __name__ == "__main__":
    main()
