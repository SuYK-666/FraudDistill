# -*- coding: utf-8 -*-
"""E4 v2: build U2 replacement candidates from pku Alpaca-7B / Alpaca2-7B
(fraud-keyword pre-filter on fraud-adjacent unsafe responses and on safe
responses). Outputs a candidate manifest for G2 blind judging.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRAUD_ADJ = {"Cybercrime", "Economic Crime", "White-Collar Crime", "Mental Manipulation"}
KW = re.compile(
    r"\b(scam|fraud|phish|bank|money|invest|refund|credit|payment|insurance|loan|tax|counterfeit|"
    r"fake|identity theft|wire|account|financial|embezzle|launder|bribe|extort|blackmail|sextortion|"
    r"charity|donation|lottery|prize|inheritance|romance|elder|health product|supplement|mortgage|"
    r"debt|gold|diamond|auction|price|investor|stocks|bitcoin|crypto|wallet|withdraw|deposit)\b", re.I)

SRC = {
    "Alpaca-7B_test.jsonl": "u2_pku_alpaca1",
    "Alpaca2-7B_test.jsonl": "u2_pku_alpaca2",
}


def main() -> None:
    rng = random.Random(20260810)
    out_rows = []
    used_qy = set()
    for fp in sorted((REPO / "data/raw/pku_saferlhf").glob("Alpaca*-7B_test.jsonl")):
        src_key = SRC[fp.name]
        unsafe_cand, safe_cand = [], []
        for line in open(fp, encoding="utf-8"):
            r = json.loads(line)
            q = str(r.get("prompt") or "").strip()
            if not q:
                continue
            for k in ("0", "1"):
                y = str(r.get("response_" + k) or "").strip()
                safe = r.get("is_response_" + k + "_safe")
                if safe is None or not y:
                    continue
                qy = (q + "\x01" + y)
                import hashlib
                qyh = hashlib.sha256(qy.encode()).hexdigest()
                if qyh in used_qy:
                    continue
                hc = r.get("response_" + k + "_harm_category") or {}
                adj = any(hc.get(c) for c in FRAUD_ADJ)
                kw = bool(KW.search(q))
                row = {
                    "id": f"{src_key}_{k}_{r.get('response_'+k+'_sha256','')[:14]}",
                    "user_query": q, "target_model_answer": y,
                    "gold_label": None, "gold_type": None,
                    "gold_binary": None, "gold_source": "blind_judge_deepseek",
                    "source": "pku_saferlhf", "source_version": fp.stem,
                    "language": "en", "target_model": str(r.get("response_" + k + "_source") or ""),
                    "primary_shift": "U2_source", "fraud_category": "financial_fraud",
                    "family_id": f"{src_key}_fam_{r.get('response_0_sha256', q)[:12]}",
                    "pair_id": None, "template_id": f"{src_key}_tpl_{q[:16]}",
                    "license": "cc-by-nc-4.0", "exposure_level": "L3",
                    "qy_hash": qyh, "q_hash": hashlib.sha256(q.encode()).hexdigest(),
                    "y_hash": hashlib.sha256(y.encode()).hexdigest(),
                    "metadata": {"harm_categories": hc, "prompt_source": r.get("prompt_source"),
                                 "pku_file": fp.name, "resp_key": k},
                }
                used_qy.add(qyh)
                if safe is True:
                    if kw:
                        safe_cand.append(row)
                elif adj and kw:
                    unsafe_cand.append(row)
        rng.shuffle(unsafe_cand); rng.shuffle(safe_cand)
        picked = unsafe_cand[:260] + safe_cand[:190]
        print(f"{fp.name}: unsafe_cand={len(unsafe_cand)} safe_cand={len(safe_cand)} picked={len(picked)}")
        out_rows.extend(picked)
    out = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL/manifests/u2_candidates_v2.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"U2 candidates written: {out} ({len(out_rows)})")


if __name__ == "__main__":
    main()
