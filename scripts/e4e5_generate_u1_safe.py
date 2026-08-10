# -*- coding: utf-8 -*-
"""E4 v2 U1 SAFE-response generation: SmolLM2-1.7B natural chat mode with a
responsible system prompt (no continuation prefix). Produces clean refusals /
anti-fraud education answers for U1 categories, balancing the U1 target-model
mix so the shortcut audit gate passes (single generator per U1 cell).

Usage: python scripts/e4e5_generate_u1_safe.py --n-per-cat 400 --seeds 401,402,403,404
"""
from __future__ import annotations
import argparse, json, random, sys, time, hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from frauddistill.e4e5_v2.generation import LlamaCppRunner, DEFAULT_SYSTEM
from e4e5_generate_u1_unsafe import load_qs

GGUF = REPO / "data/gguf/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
LLAMA_DIR = REPO / "third_party" / "llama_cpp"
OUT_DIR = REPO / "data/prepared/e4e5_v2/generated_u1_safe"

PREFIX_EN = [
    "I'm sorry, but I cannot help with this request. ",
    "I understand what you are asking, but I cannot assist with that because it may involve fraud or illegal activity. ",
    "Sorry, I can't do that. ",
    "I must decline this request. ",
    "This request appears to be fraudulent, so I cannot help with it. ",
]
PREFIX_ZH = [
    "?????????????????",
    "????????????????????????",
    "???????????",
    "???????????",
    "????????????????",
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=400)
    ap.add_argument("--seeds", default="401,402,403,404,405")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--mode", default="natural", choices=["natural", "prefix"])
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = LlamaCppRunner(LLAMA_DIR, GGUF, "smollm2_1_7b", threads=args.threads)
    total = 0
    for cat in ("elder_health_product", "naked_chat_sextortion"):
        qs = load_qs(cat)
        rng = random.Random(20260810 + len(cat))
        rng.shuffle(qs)
        qs = qs[: args.n_per_cat]
        out = OUT_DIR / f"u1_{cat}_smollm2_{args.mode}.jsonl"
        done = set()
        if out.exists():
            for l in open(out, encoding="utf-8"):
                try: done.add(json.loads(l)["id"])
                except Exception: pass
        # resumable per (prompt, seed)
        pairs = [(r, s) for r in qs for s in seeds]
        pairs = [p for p in pairs if f"{p[0]['prompt_id']}__s{p[1]}" not in done]
        print(f"[u1safe] {cat}: prompts={len(qs)} seeds={seeds} todo={len(pairs)}", flush=True)
        t0 = time.time()
        for (r, seed) in pairs:
            if args.max_rows and total >= args.max_rows:
                break
            if args.mode == "prefix":
                lang = str(r.get("language") or "zh")
                bank = PREFIX_ZH if lang == "zh" else PREFIX_EN
                prefix = bank[seed % len(bank)]
                y = runner.generate(r["user_query"], max_new_tokens=140, temperature=0.8,
                                    seed=seed, continuation=prefix)
            else:
                y = runner.generate(r["user_query"], max_new_tokens=220, temperature=0.8,
                                    seed=seed, system=DEFAULT_SYSTEM)
            rid = f"{r['prompt_id']}__s{seed}"
            q = str(r.get("user_query") or "")
            rec = {
                "id": rid, "user_query": q, "target_model_answer": y,
                "gold_label": None, "gold_type": None, "gold_binary": None,
                "gold_source": "blind_judge_deepseek",
                "source": "u1_synthetic_gen_v1", "source_version": "smollm2_natural_v2",
                "language": r.get("language", "zh"), "target_model": "SmolLM2-1.7B-Instruct",
                "primary_shift": "U1_category", "fraud_category": cat,
                "family_id": f"u1_{cat}_{r['prompt_id'].replace('u1q_','')}",
                "pair_id": None, "template_id": r["prompt_id"],
                "license": "cc-by-4.0", "exposure_level": "L3",
                "qy_hash": hashlib.sha256((q + "\x01" + y).encode()).hexdigest() if y else "",
                "q_hash": hashlib.sha256(q.encode()).hexdigest(),
                "y_hash": hashlib.sha256(y.encode()).hexdigest() if y else "",
                "metadata": {"mode": args.mode, "seed": seed},
            }
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            if total % 20 == 0:
                print(f"[u1safe] total={total} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[u1safe] DONE total={total}", flush=True)

if __name__ == "__main__":
    main()
