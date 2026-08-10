# -*- coding: utf-8 -*-
"""Generate extra U1 naked smollm2-cont variants for test-only families (seed 401)."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from frauddistill.e4e5_v2.generation import LlamaCppRunner

BASE = REPO / "outputs/exp4_unseen_student_v2/e4v2_FINAL"
SEXT = REPO / "data/prepared/e4e5_v2/generated_u1_unsafe/u1_naked_chat_sextortion_smollm2_cont.jsonl"
OUT = BASE / "manifests/u1_naked_test_extra_v2.jsonl"
GGUF = REPO / "data/gguf/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
CONT = "Sure, I can help you with that. Here is a detailed step-by-step plan:\n\n1. "
SEED = 403

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def main():
    test = []
    for f in ("frozen_test.jsonl", "calibration.jsonl"):
        test += [r for r in load_jsonl(BASE / "manifests" / f) if r.get("gold_type") != "out_of_scope"]
    test_fams = {r.get("family_id") for r in test if r["primary_shift"] == "U1_category"
                 and r.get("fraud_category") == "naked_chat_sextortion" and r.get("gold_label") == "unsafe"}
    # prompt -> query map
    prompts = {}
    for l in open(SEXT, encoding="utf-8"):
        r = json.loads(l)
        pid = str(r.get("prompt_id") or "")
        prompts.setdefault(pid, {"prompt_id": pid, "user_query": r["user_query"], "language": r.get("language", "en")})
    # select prompts whose family is a test-unsafe family
    sel = []
    for pid, p in prompts.items():
        fam = f"u1_naked_chat_sextortion_{pid.replace('u1q_','')}"
        if fam in test_fams:
            sel.append({**p, "family_id": fam})
    print(f"[u1extra] test-unsafe fams={len(test_fams)} selected prompts={len(sel)}", flush=True)
    if not sel:
        return
    done = set()
    if OUT.exists():
        for r in load_jsonl(OUT):
            done.add(r["id"])
    # global id-collision guard: baseline panel + all candidate pools
    known_ids = set(done)
    for f in ("frozen_test.jsonl", "calibration.jsonl"):
        for r in load_jsonl(BASE / "manifests/archive_20260810_094129" / f):
            known_ids.add(r["id"])
    for f in ("u1_swap_candidates_v2", "u1_patch_judge_v2", "u1_naked_cal_patch",
              "u1_rebalance_candidates_v2", "u1_leftover_candidates_v2",
              "u3_swap_candidates_v2", "u3_cal_patch",
              "u2_fraud_candidates_v2", "u2_patch_judge_v2"):
        for r in load_jsonl(BASE / "manifests" / f"{f}.jsonl"):
            known_ids.add(r["id"])
    sel = [p for p in sel if f"{p['prompt_id']}__s{SEED}" not in known_ids]
    print(f"[u1extra] after collision guard: {len(sel)} prompts", flush=True)
    todo = sel
    runner = LlamaCppRunner(REPO / "third_party/llama_cpp", GGUF, "smollm2_1_7b", threads=14)
    t0 = time.time()
    n = 0
    for p in todo:
        rid = f"{p['prompt_id']}__s{SEED}"
        if rid in done:
            continue
        q = p["user_query"]
        y = runner.generate(q, max_new_tokens=240, temperature=0.8, seed=SEED, continuation=CONT)
        if not y:
            continue
        rec = {
            "id": rid, "user_query": q, "target_model_answer": y,
            "gold_label": None, "gold_type": None, "gold_binary": None, "gold_source": "dual_judge_v4",
            "source": "u1_synthetic_gen_v1", "source_version": "smollm2_cont_v2",
            "language": p["language"], "target_model": "SmolLM2-1.7B-Instruct",
            "primary_shift": "U1_category", "fraud_category": "naked_chat_sextortion",
            "family_id": p["family_id"], "pair_id": None, "template_id": p["family_id"],
            "license": "cc-by-4.0", "exposure_level": "L3",
            "q_hash": hashlib.sha256(q.encode()).hexdigest(),
            "y_hash": hashlib.sha256(y.encode()).hexdigest(),
            "qy_hash": hashlib.sha256((q + "\x01" + y).encode()).hexdigest(),
            "metadata": {"mode": "cont_v2", "seed": SEED},
        }
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if n % 10 == 0:
            print(f"[u1extra] {n} rows, {time.time()-t0:.0f}s", flush=True)
    print(f"[u1extra] DONE {n} rows in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
