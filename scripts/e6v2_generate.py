# -*- coding: utf-8 -*-
"""E6 v2 generation: probe -> pilot -> anchor/b0/b1/b2/control panels.
Resumable per-model jsonl; budget gate before every call; per-model concurrency."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from e6v2_common import (V2_DIR, PROTOCOL_DIR, DATA_DIR, GEN_DIR, BUDGET_DIR, read_jsonl, write_jsonl,
                         write_json, read_json, utc_now, sha256_text, norm_query, est_cost,
                         CostLedger, manifest_sha256, SLOT_MODEL, SLOT_PROVIDER, SLOT_LABEL, SEED)
from e6_api import call_model, call_with_rl_retry, PROBE_PROMPTS

PROBE_SLOTS = [
    {"slot": "M1", "provider": "qwen", "candidate_ids": ["qwen-flash"]},
    {"slot": "M2", "provider": "qwen", "candidate_ids": ["qwen-plus"]},
    {"slot": "M3", "provider": "deepseek", "candidate_ids": ["deepseek-v4-flash"],
     "extra_body": {"model": "deepseek-v4-flash", "thinking": {"type": "disabled"}}},
    {"slot": "M4", "provider": "deepseek", "candidate_ids": ["deepseek-v4-pro"],
     "extra_body": {"model": "deepseek-v4-pro", "thinking": {"type": "disabled"}}},
    {"slot": "M5", "provider": "glm", "candidate_ids": ["glm-4-flash"]},
    {"slot": "M6", "provider": "kimi", "candidate_ids": ["moonshot-v1-8k"]},
]
CONCURRENCY = {"M1": 120, "M2": 120, "M3": 120, "M4": 120, "M5": 16, "M6": 8}
SYSTEM_PROMPT = "You are a helpful assistant."

def probe() -> dict:
    ledger = CostLedger()
    results, registry = [], {}
    for slot in PROBE_SLOTS:
        for mid in slot["candidate_ids"]:
            ok, served, err = 0, None, None
            for pid, risk, lang, q in PROBE_PROMPTS:
                try:
                    ledger.gate_ok(est_cost(mid, 200, 64), stage="probe")
                    env = call_model(slot["provider"], mid, q, system_prompt=SYSTEM_PROMPT,
                                     extra_body=slot.get("extra_body"))
                    success = bool(env["text"] and len(env["text"]) >= 2)
                    ledger.append({"stage": "v2_probe", "provider": slot["provider"], "model": mid, "prompt_id": pid,
                                   "input_tokens": env["input_tokens"], "output_tokens": env["output_tokens"],
                                   "reasoning_tokens": env["reasoning_tokens"], "cost_cny": env["estimated_cost_cny"],
                                   "success": success})
                    results.append({"slot": slot["slot"], "requested": mid, "served": env["served_model"],
                                    "prompt": pid, "ok": success, "finish_reason": env["finish_reason"],
                                    "cost_cny": env["estimated_cost_cny"], "latency_seconds": env["latency_seconds"]})
                    if success:
                        ok += 1; served = env["served_model"]
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    results.append({"slot": slot["slot"], "requested": mid, "prompt": pid, "ok": False, "error": err})
            if ok >= 3:
                registry[slot["slot"]] = {"slot": slot["slot"], "provider": slot["provider"],
                                          "requested_model": mid, "served_model": served,
                                          "probe_ok": ok, "probe_total": len(PROBE_PROMPTS),
                                          "extra_body": slot.get("extra_body")}
                break
        if slot["slot"] not in registry:
            print(f"[probe] WARN {slot['slot']} no candidate (err={err})")
    write_jsonl(PROTOCOL_DIR / "probe_results.jsonl", results)
    write_json(PROTOCOL_DIR / "model_registry_frozen.json", {"slots": registry, "frozen_at_utc": utc_now(),
                                                             "probe_prompt_set": [p[0] for p in PROBE_PROMPTS]})
    ledger.write_summary()
    print(f"[probe] registry={sorted(registry)} cum=CNY{ledger.cumulative:.4f}")
    return registry

def cache_path(slot: str) -> Path:
    return GEN_DIR / "per_model" / f"{slot}.jsonl"

def load_cache(slot: str) -> dict:
    out = {}
    for l in read_jsonl(cache_path(slot)):
        out[l["prompt_id"]] = l
    return out

def run_panel(registry: dict, manifest: list[dict], panel: str, max_retries: int = 1, limit: int | None = None, skip_slots: set | None = None) -> None:
    ledger = CostLedger()
    mhash = manifest_sha256(manifest)
    rows = [r for r in manifest if r["panel"] == panel]
    if limit: rows = rows[:limit]
    skip_slots = skip_slots or set()
    for slot_key, spec in registry["slots"].items():
        if slot_key in skip_slots:
            print(f"[{panel}] {slot_key}: SKIPPED (--skip-slots)", flush=True)
            continue
        cache = load_cache(slot_key)
        tasks = [r for r in rows if (cache.get(r["prompt_id"]) or {}).get("generation_success") is not True]
        print(f"[{panel}] {slot_key}: cached={sum(1 for r in rows if (cache.get(r['prompt_id']) or {}).get('generation_success'))} todo={len(tasks)}", flush=True)
        if not tasks:
            continue
        concurrency = min(CONCURRENCY.get(slot_key, 40), 120)
        def work(row):
            pid = row["prompt_id"]
            try:
                ledger.gate_ok(est_cost(spec["requested_model"], 1500, 300), stage=f"gen_{panel}")
                env = call_with_rl_retry(spec["provider"], spec["requested_model"], row["user_query"],
                                         system_prompt=SYSTEM_PROMPT, max_tokens=256, temperature=0.0,
                                         extra_body=spec.get("extra_body"))
                rec = {"run_id": f"v2_{panel}_{slot_key}_{pid}", "prompt_id": pid, "panel": panel,
                       "slot": slot_key, "target_provider": spec["provider"], "requested_model": spec["requested_model"],
                       "served_model": env["served_model"], "target_model_answer": env["text"],
                       "input_tokens": env["input_tokens"], "output_tokens": env["output_tokens"],
                       "reasoning_tokens": env["reasoning_tokens"], "latency_seconds": env["latency_seconds"],
                       "finish_reason": env["finish_reason"], "estimated_cost_cny": env["estimated_cost_cny"],
                       "attempt": 1, "generation_success": bool(env["text"] and len(env["text"]) >= 2),
                       "timestamp_utc": utc_now(), "prompt_manifest_hash": mhash, "stage": f"v2_gen_{panel}"}
                ledger.append({"stage": f"v2_gen_{panel}", "provider": spec["provider"], "model": spec["requested_model"],
                               "prompt_id": pid, "input_tokens": env["input_tokens"], "output_tokens": env["output_tokens"],
                               "reasoning_tokens": env["reasoning_tokens"], "cost_cny": env["estimated_cost_cny"],
                               "success": rec["generation_success"]})
                return rec
            except Exception as e:
                return {"run_id": f"v2_{panel}_{slot_key}_{pid}", "prompt_id": pid, "panel": panel,
                        "slot": slot_key, "target_provider": spec["provider"], "requested_model": spec["requested_model"],
                        "generation_success": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                        "attempt": 1, "timestamp_utc": utc_now(), "prompt_manifest_hash": mhash, "stage": f"v2_gen_{panel}"}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(work, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                cache[rec["prompt_id"]] = rec
                cache_path(slot_key).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path(slot_key), "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += 1
                if done % 100 == 0 or not rec["generation_success"]:
                    print(f"[{panel}] {slot_key} done={done}/{len(tasks)} cum=CNY{ledger.cumulative:.3f}", flush=True)
        # one retry pass for failures
        if max_retries >= 1:
            fails = [r for r in rows if (cache.get(r["prompt_id"]) or {}).get("generation_success") is False]
            for row in fails:
                prev = cache.get(row["prompt_id"], {})
                if prev.get("attempt", 0) >= max_retries + 1:
                    continue
                try:
                    env = call_with_rl_retry(spec["provider"], spec["requested_model"], row["user_query"],
                                             system_prompt=SYSTEM_PROMPT, max_tokens=256, temperature=0.0,
                                             extra_body=spec.get("extra_body"))
                    rec = {"run_id": f"v2_{panel}_{slot_key}_{row['prompt_id']}_r2", "prompt_id": row["prompt_id"],
                           "panel": panel, "slot": slot_key, "target_provider": spec["provider"],
                           "requested_model": spec["requested_model"], "served_model": env["served_model"],
                           "target_model_answer": env["text"], "input_tokens": env["input_tokens"],
                           "output_tokens": env["output_tokens"], "reasoning_tokens": env["reasoning_tokens"],
                           "latency_seconds": env["latency_seconds"], "finish_reason": env["finish_reason"],
                           "estimated_cost_cny": env["estimated_cost_cny"], "attempt": prev.get("attempt", 0) + 1,
                           "generation_success": bool(env["text"] and len(env["text"]) >= 2),
                           "timestamp_utc": utc_now(), "prompt_manifest_hash": mhash, "stage": f"v2_gen_{panel}",
                           "supersedes": prev.get("run_id")}
                    ledger.append({"stage": "v2_retry", "provider": spec["provider"], "model": spec["requested_model"],
                                   "prompt_id": row["prompt_id"], "input_tokens": env["input_tokens"],
                                   "output_tokens": env["output_tokens"], "reasoning_tokens": env["reasoning_tokens"],
                                   "cost_cny": env["estimated_cost_cny"], "success": rec["generation_success"]})
                    if rec["generation_success"]:
                        cache[row["prompt_id"]] = rec
                        cache_path(slot_key).parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path(slot_key), "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"[retry] {slot_key} {row['prompt_id']} ok={rec['generation_success']}", flush=True)
                except Exception as e:
                    print(f"[retry] {slot_key} {row['prompt_id']} error {type(e).__name__}", flush=True)
    ledger.write_summary()

def merged_registry(registry: dict) -> None:
    rows = []
    for slot in registry["slots"]:
        rows += read_jsonl(cache_path(slot))
    write_jsonl(GEN_DIR / "generation_registry.jsonl", rows)
    write_json(GEN_DIR / "generation_summary.json", {
        "total_records": len(rows),
        "by_slot": {s: {"ok": sum(1 for r in rows if r.get("slot") == s and r.get("generation_success")),
                        "fail": sum(1 for r in rows if r.get("slot") == s and not r.get("generation_success"))}
                    for s in registry["slots"]},
        "by_panel": {p: {"ok": sum(1 for r in rows if r.get("panel") == p and r.get("generation_success")),
                         "fail": sum(1 for r in rows if r.get("panel") == p and not r.get("generation_success"))}
                     for p in ("anchor", "b0", "b1", "b2", "control")},
    })

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "pilot", "anchor", "b0", "b1", "b2", "control", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-slots", type=str, default="")
    args = ap.parse_args()
    skip_slots = set(x.strip() for x in args.skip_slots.split(",") if x.strip())
    manifest = read_jsonl(DATA_DIR / "prompt_pool_manifest.jsonl")
    if args.stage == "probe":
        registry = probe()
    else:
        reg_file = PROTOCOL_DIR / "model_registry_frozen.json"
        if not reg_file.exists():
            registry = probe()
        else:
            registry = read_json(reg_file)
        if args.stage == "pilot":
            # first 10 refuse + first 10 answer anchor q's (deterministic)
            refuse = [r for r in manifest if r["panel"] == "anchor" and r["should_refuse"]][:10]
            answer = [r for r in manifest if r["panel"] == "anchor" and not r["should_refuse"]][:10]
            run_panel(registry, refuse + answer, "anchor", limit=20, skip_slots=skip_slots)
        elif args.stage == "all":
            for p in ("anchor", "b0", "control"):
                run_panel(registry, manifest, p, skip_slots=skip_slots)
            run_panel(registry, manifest, "b1", skip_slots=skip_slots)
            run_panel(registry, manifest, "b2", skip_slots=skip_slots)
        else:
            run_panel(registry, manifest, args.stage, limit=args.limit, skip_slots=skip_slots)
        merged_registry(registry)
