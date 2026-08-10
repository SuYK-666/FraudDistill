# -*- coding: utf-8 -*-
"""Build U2 fraud-relevant candidate pool (pku_saferlhf) for G2-v2 re-judging.

The v5 U2 panel accidentally mixed in non-fraud general harm (DDoS, weapons,
drugs, stalking, ...) because pku_saferlhf is a general safety benchmark.  E4's
task is fraud-assistance detection, so U2 must contain fraud/scam-related
queries only.  This script:
  1. merges u2_candidates_v2 + u2_pku_leftover_candidates_v2 (content-dedup);
  2. keeps rows whose query matches a fraud/scam lexicon;
  3. excludes rows already in the current panel (they are re-judged in the
     full 1800 pass);
  4. writes manifests/u2_fraud_candidates_v2.jsonl for G2-v2 judging.
"""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"

FRAUD_LEXICON = [
    "scam", "fraud", "phish", "money laundering", "launder", "embezzl", "tax evas", "tax evasion",
    "offshore account", "bank account", "credit card", "debit card", "atm ", "stolen card",
    "identity theft", "ssn", "social security number", "passport", "forged", "forgery",
    "counterfeit", "fake money", "fake bill", "fake id", "fake document", "fake invoice",
    "fake website", "fake email", "financial fraud", "wire fraud", "ponzi", "pyramid scheme",
    "investment scheme", "insider trading", "stock manipulation", "market manipulation",
    "insurance claim", "insurance fraud", "refund scam", "refund fraud", "chargeback fraud",
    "loan scam", "loan fraud", "mortgage fraud", "bank fraud", "payment fraud", "billing fraud",
    "ransomware", "crypto scam", "bitcoin scam", "cryptocurrency scam", "romance scam",
    "catfish", "lottery scam", "inheritance scam", "prince", "nigerian", "419 ",
    "pyramid", "multi-level", "mlm", "gambling fraud", "casino cheat", "betting fraud",
    "money mule", "mule", "smuggling", "smuggle", "trafficking", "silk road", "dark web market",
    "hack", "hacking", "hacker", "malware", "spyware", "keylogger", "credential", "password",
    "pin number", "verification code", "otp", "2fa", "two-factor", "phishing", "spoofing",
    "spoof", "impersonat", "fake representative", "call center scam", "tech support scam",
    "utility scam", "grandparent scam", "elder scam", "senior scam", "advance fee",
    "overpayment", "fake check", "cashier", "western union", "moneygram", "gift card scam",
    "cryptocurrency", "bitcoin", "ethereum", "wallet seed", "private key", "exchange",
    "withdraw", "transfer", "wire ", "remittance", "bribe", "bribery", "corruption",
    "extortion", "blackmail", "sextortion", "fraudster", "con artist", "con man", "con woman",
    "scammer", "swindle", "cheat", "defraud", "deceive", "deception", "fraudulent",
    "bank", "money", "cash", "payment", "paypal", "invoice", "bill", "receipt", "loan", "lender",
    "debt", "collection", "credit score", "financial", "finance", "invest", "investor",
    "stock", "bond", "trading", "broker", "dividend", "return on investment", "passive income",
    "forex", "options trading", "day trading", "mining", "nft", "token", "airdrop",
    "credit", "mortgage", "escrow", "down payment", "cheque", "checking account", "savings account",
    "swift", "iban", "routing number", "account number", "card number", "cvv", "cvc",
]
_pat = re.compile("|".join(re.escape(k) for k in FRAUD_LEXICON), re.I)


def is_fraud_query(q: str) -> bool:
    return bool(_pat.search(q or ""))


def content_key(r):
    return (str(r.get("context") or ""), str(r.get("user_query") or ""), str(r.get("target_model_answer") or ""))


def main():
    pool = {}
    for name in ("u2_candidates_v2.jsonl", "u2_pku_leftover_candidates_v2.jsonl"):
        p = BASE / "manifests" / name
        if not p.exists():
            print("missing", name); continue
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            k = content_key(r)
            if k not in pool:
                pool[k] = r
    print("merged pool:", len(pool))
    # current panel ids (already being re-judged in the 1800 pass)
    panel_ids = set()
    for name in ("frozen_test.jsonl", "calibration.jsonl"):
        for l in open(BASE / "manifests" / name, encoding="utf-8"):
            panel_ids.add(json.loads(l)["id"])
    print("current panel ids:", len(panel_ids))
    cand, seen_q = [], set()
    for k, r in pool.items():
        q = str(r.get("user_query") or "")
        if r["id"] in panel_ids:
            continue
        if not is_fraud_query(q):
            continue
        # dedupe by normalized query (avoid near-identical variants inflating pool)
        nq = re.sub(r"[\s\W_]+", "", q).lower()
        if nq in seen_q:
            continue
        seen_q.add(nq)
        cand.append(r)
    print("fraud candidates (new):", len(cand))
    c = Counter((str(r.get("source_version")), str(r.get("gold_label"))) for r in cand)
    for k, v in sorted(c.items()):
        print("  ", k, v)
    out = BASE / "manifests" / "u2_fraud_candidates_v2.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in cand:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("written:", out, len(cand))


if __name__ == "__main__":
    main()
