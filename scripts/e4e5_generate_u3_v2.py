# -*- coding: utf-8 -*-
"""E4 v2: targeted U3 refuse_chat supplement for smollm2/phi3.

Generates NEW seeds for prompt families with proven G2-safe refuse history.
Resumable; output appended to generated_u3/<model>_refuse_v2.jsonl.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from frauddistill.e4e5_v2.generation import LlamaCppRunner

MODELS = {"smollm2_1_7b": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
          "phi3_5_mini": "Phi-3.5-mini-instruct-Q4_K_M.gguf"}
REFUSE_WRAPPER = ("\n\nReply as a responsible assistant to the message above. "
                  "Is it fraudulent or legitimate? What should the recipient do? "
                  "If it is fraudulent, do not help the sender and explain why.")

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--families", required=True, help="path to jsonl/JSON list of family ids")
    ap.add_argument("--seeds", default="111,211", help="new seeds")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    fam_file = Path(args.families)
    fams = [json.loads(l)["family_id"] for l in open(fam_file, encoding="utf-8")] if fam_file.suffix == ".jsonl" else json.load(open(fam_file))
    fams = set(fams)
    norm = {f.replace("u3prompt_", "") for f in fams} | fams
    prompts = [r for r in load(REPO / "data/prepared/e4e5_v2/fraudr1_unseen_prompts.jsonl")
               if r.get("id") in norm or r.get("family_id") in norm]
    print(f"[u3v2] model={args.model} target families={len(fams)} prompts matched={len(prompts)}", flush=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out = REPO / f"data/prepared/e4e5_v2/generated_u3/{args.model}_refuse_v2.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out, encoding="utf-8"):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    runner = LlamaCppRunner(REPO / "third_party/llama_cpp", REPO / "data/gguf" / MODELS[args.model],
                            args.model, threads=args.threads)
    t0 = time.time(); total = 0
    for seed in seeds:
        for r in prompts:
            pid = str(r.get("id") or "")
            rid = f"u3v2_{args.model}_{pid}_{seed}"
            if rid in done: continue
            q = str(r.get("user_query") or "")[:1000] + REFUSE_WRAPPER
            y = runner.generate(q, max_new_tokens=220, temperature=0.8, seed=seed)
            rec = {"id": rid, "prompt_id": pid, "user_query": q, "target_model_answer": y,
                   "target_model": args.model, "family_id": pid, "seed": seed,
                   "mode": "refuse_chat_v2", "language": r.get("language", "en"),
                   "fraud_category": r.get("fraud_category", ""), "perspective": None,
                   "generation_status": "ok" if y else "empty"}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add(rid); total += 1
            if total % 10 == 0:
                el = time.time() - t0
                print(f"[u3v2] {total} rows elapsed={el:.0f}s rate={total/max(el,1e-6):.2f}/s", flush=True)
    print(f"[u3v2] DONE total={total} -> {out}", flush=True)

if __name__ == "__main__":
    main()
