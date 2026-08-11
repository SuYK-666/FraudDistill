# -*- coding: utf-8 -*-
"""E6 S4: score all successful generations with frozen Final Student (offline)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from e6_common import (DATA_DIR, GEN_DIR, STUDENT_DIR, STUDENT_CKPT, STUDENT_THRESHOLD, STUDENT_MAX_LENGTH,
                       read_jsonl, write_jsonl, write_json, read_json, manifest_sha256)
from frauddistill.e4e5_v2.student_inference import load_checkpoint, predict_scores
from frauddistill.student.dataset import neural_input_text

def main(threads: int = 12, micro_batch: int = 8):
    if threads:
        torch.set_num_threads(threads)
    manifest = read_jsonl(DATA_DIR / "exp6_prompt_manifest.jsonl")
    mrow = {r["prompt_id"]: r for r in manifest}
    reg = read_json(ROOT / "experiments/exp6_multi_api/protocol/model_registry_frozen.json")
    jobs = []
    for slot in ["M1", "M2", "M3", "M4", "M5", "M6"]:
        cache = {}
        for l in read_jsonl(GEN_DIR / "per_model" / f"{slot}.jsonl"):
            cache[l["prompt_id"]] = l
        for pid, rec in cache.items():
            if not rec.get("generation_success"):
                continue
            q = mrow[pid]["user_query"]
            y = rec["target_model_answer"]
            jobs.append({"id": f"{slot}::{pid}", "slot": slot, "prompt_id": pid,
                         "user_query": q, "target_model_answer": y,
                         "provider": rec.get("target_provider"), "model": rec.get("requested_model"),
                         "served_model": rec.get("served_model")})
    print(f"[score] jobs={len(jobs)}")
    t0 = time.time()
    model, tok = load_checkpoint(STUDENT_CKPT, max_length=STUDENT_MAX_LENGTH)
    print(f"[score] model loaded in {time.time()-t0:.1f}s")
    progress = STUDENT_DIR / "score_progress.jsonl"
    preds, _ = predict_scores(model, tok, jobs, max_length=STUDENT_MAX_LENGTH, micro_batch=micro_batch,
                              progress_path=progress, tag="e6-student")
    del model
    import gc; gc.collect()
    # token lengths + truncation audit
    out_rows = []
    trunc_stats = {}
    for p in preds:
        slot, pid = p["id"].split("::")
        m = mrow[pid]
        text = neural_input_text(m["user_query"], next(j["target_model_answer"] for j in jobs if j["id"] == p["id"]))
        enc = tok(text, add_special_tokens=True)
        n_tok = len(enc["input_ids"])
        truncated = n_tok > STUDENT_MAX_LENGTH
        out_rows.append({
            "prompt_id": pid, "slot": slot, "prompt_stratum": m["prompt_stratum"],
            "prompt_risk_label": m["prompt_risk_label"], "language": m["language"],
            "risk_score": p["risk_score"], "pred_label": "unsafe" if p["risk_score"] >= STUDENT_THRESHOLD else "safe",
            "threshold": STUDENT_THRESHOLD, "input_tokens_qy": n_tok, "truncated": truncated,
            "truncation_tokens": max(0, n_tok - STUDENT_MAX_LENGTH),
            "type_probabilities": p["type_probabilities"],
        })
        trunc_stats.setdefault(slot, {"n": 0, "trunc": 0, "max_tok": 0})
        trunc_stats[slot]["n"] += 1
        trunc_stats[slot]["trunc"] += int(truncated)
        trunc_stats[slot]["max_tok"] = max(trunc_stats[slot]["max_tok"], n_tok)
    write_jsonl(STUDENT_DIR / "predictions_all.jsonl", out_rows)
    write_json(STUDENT_DIR / "truncation_audit.json", {"max_length": STUDENT_MAX_LENGTH, "per_model": trunc_stats,
                                                        "overall_truncation_rate": round(sum(v["trunc"] for v in trunc_stats.values()) / max(sum(v["n"] for v in trunc_stats.values()), 1), 4)})
    print("[score] done:", len(out_rows), "in", round(time.time()-t0, 1), "s")
    for k, v in trunc_stats.items():
        print(f"  {k}: n={v['n']} trunc={v['trunc']} ({v['trunc']/max(v['n'],1):.1%}) max_tok={v['max_tok']}")

if __name__ == "__main__":
    main()
