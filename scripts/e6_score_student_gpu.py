# -*- coding: utf-8 -*-
"""E6 S4 GPU: score all successful generations with frozen Final Student (server RTX 4090)."""
from __future__ import annotations
import json, sys, time, gc
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import torch
from torch.utils.data import DataLoader, Dataset
from frauddistill.e4e5_v2.student_inference import load_checkpoint
from frauddistill.student.collator import neural_collate
from frauddistill.student.dataset import ID_TO_LABEL, build_neural_examples
from frauddistill.student.dataset import neural_input_text
from e6_common import (DATA_DIR, GEN_DIR, STUDENT_DIR, STUDENT_CKPT, STUDENT_THRESHOLD, STUDENT_MAX_LENGTH,
                       read_jsonl, write_jsonl, write_json, read_json)

def main():
    manifest = read_jsonl(DATA_DIR / "exp6_prompt_manifest.jsonl")
    mrow = {r["prompt_id"]: r for r in manifest}
    reg = read_json(REPO / "experiments/exp6_multi_api/protocol/model_registry_frozen.json")
    jobs = []
    for slot in ["M1", "M2", "M3", "M4", "M5", "M6"]:
        cache = {}
        for l in read_jsonl(GEN_DIR / "per_model" / f"{slot}.jsonl"):
            cache[l["prompt_id"]] = l
        for pid, rec in cache.items():
            if not rec.get("generation_success"):
                continue
            jobs.append({"id": f"{slot}::{pid}", "slot": slot, "prompt_id": pid,
                         "user_query": mrow[pid]["user_query"], "target_model_answer": rec["target_model_answer"],
                         "provider": rec.get("target_provider"), "model": rec.get("requested_model"),
                         "served_model": rec.get("served_model")})
    print(f"[score-gpu] jobs={len(jobs)}", flush=True)
    t0 = time.time()
    model, tok = load_checkpoint(STUDENT_CKPT, max_length=STUDENT_MAX_LENGTH)
    model = model.to("cuda").eval()
    print(f"[score-gpu] model loaded+to cuda in {time.time()-t0:.1f}s", flush=True)
    exs = build_neural_examples(jobs, max_length=STUDENT_MAX_LENGTH, use_teacher_soft=True, use_pairwise=False)
    class D(Dataset):
        def __len__(self): return len(exs)
        def __getitem__(self, i): return exs[i]
    loader = DataLoader(D(), batch_size=16, shuffle=False,
                        collate_fn=lambda b: neural_collate(b, tok, max_length=STUDENT_MAX_LENGTH, architecture="standard"))
    preds = []
    n_batches = len(loader)
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            inp = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            kw = {}
            if inp.get("query_mask") is not None:
                kw["query_mask"] = inp["query_mask"]; kw["answer_mask"] = inp["answer_mask"]
            out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], **kw)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            for i, rid in enumerate(inp["ids"]):
                type_probs = {ID_TO_LABEL[j]: round(float(probs[i, j]), 4) for j in range(4)}
                risk = float(1.0 - probs[i, 0])
                preds.append({"id": rid, "risk_score": round(risk, 4), "type_probabilities": type_probs})
            if (bi + 1) % 20 == 0 or bi == n_batches - 1:
                print(f"[score-gpu] {bi+1}/{n_batches} batches ({len(preds)}/{len(jobs)}) {time.time()-t0:.0f}s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # token lengths + truncation audit (cpu tokenizer)
    out_rows = []
    trunc_stats = {}
    jmap = {j["id"]: j for j in jobs}
    for p in preds:
        slot, pid = p["id"].split("::")
        j = jmap[p["id"]]
        m = mrow[pid]
        text = neural_input_text(m["user_query"], j["target_model_answer"])
        enc = tok(text, add_special_tokens=True)
        n_tok = len(enc["input_ids"])
        out_rows.append({
            "prompt_id": pid, "slot": slot, "prompt_stratum": m["prompt_stratum"],
            "prompt_risk_label": m["prompt_risk_label"], "language": m["language"],
            "risk_score": p["risk_score"], "pred_label": "unsafe" if p["risk_score"] >= STUDENT_THRESHOLD else "safe",
            "threshold": STUDENT_THRESHOLD, "input_tokens_qy": n_tok, "truncated": n_tok > STUDENT_MAX_LENGTH,
            "truncation_tokens": max(0, n_tok - STUDENT_MAX_LENGTH),
            "type_probabilities": p["type_probabilities"],
        })
        trunc_stats.setdefault(slot, {"n": 0, "trunc": 0, "max_tok": 0})
        trunc_stats[slot]["n"] += 1
        trunc_stats[slot]["trunc"] += int(n_tok > STUDENT_MAX_LENGTH)
        trunc_stats[slot]["max_tok"] = max(trunc_stats[slot]["max_tok"], n_tok)
    write_jsonl(STUDENT_DIR / "predictions_all.jsonl", out_rows)
    write_json(STUDENT_DIR / "truncation_audit.json", {"max_length": STUDENT_MAX_LENGTH, "per_model": trunc_stats,
                                                        "overall_truncation_rate": round(sum(v["trunc"] for v in trunc_stats.values()) / max(sum(v["n"] for v in trunc_stats.values()), 1), 4)})
    print(f"[score-gpu] DONE {len(out_rows)} rows in {time.time()-t0:.0f}s", flush=True)
    for k, v in trunc_stats.items():
        print(f"  {k}: n={v['n']} trunc={v['trunc']} ({v['trunc']/max(v['n'],1):.1%}) max_tok={v['max_tok']}", flush=True)

if __name__ == "__main__":
    main()
