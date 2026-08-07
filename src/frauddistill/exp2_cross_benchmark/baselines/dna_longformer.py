"""Do-Not-Answer official baseline: LibrAI/longformer-harmful-ro.

Input format follows the official repo (question + " [SEP] " + response),
max_length=512 (official training setting). CPU inference.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "LibrAI/longformer-harmful-ro"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tok, model


@torch.no_grad()
def predict(tok, model, texts: list[str], batch_size: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        out.append(probs.numpy())
    return np.concatenate(out, axis=0)


def sanity(tok, model):
    rows = [json.loads(l) for l in open(os.path.join(out_path("unified"), "do_not_answer_eval.jsonl"), encoding="utf-8")][:20]
    texts = [f"{r['query']} [SEP] {r['answer']}" for r in rows]
    probs = predict(tok, model, texts)
    for r, p in zip(rows, probs):
        print(f"gold={r['gold_binary']} p(harm)={p[1]:.3f} label={int(np.argmax(p))} | {r['query'][:60]!r}")


def out_path(*parts):
    d = os.path.join("experiments", "exp2_prior_work_comparison", "do_not_answer", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def generate_overrides(input_path: str, out_path_override: str, batch_size: int = 16):
    """Same local longformer baseline, custom input/output paths."""
    unified = input_path
    rows = [json.loads(l) for l in open(unified, encoding="utf-8")]
    tok, model = load_model()
    texts = [f"{r['query']} [SEP] {r['answer']}" for r in rows]
    t0 = time.time()
    probs = predict(tok, model, texts, batch_size=batch_size)
    preds = np.argmax(probs, axis=1).tolist()
    out = out_path_override
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    with open(out, "a", encoding="utf-8") as f:
        n = 0
        for r, p_, pred in zip(rows, probs, preds):
            if r["id"] in done:
                continue
            f.write(json.dumps({
                "id": r["id"],
                "benchmark": "do_not_answer",
                "group_id": r["group_id"],
                "query": r["query"],
                "answer": r["answer"],
                "method": "dna_longformer_harmful_ro",
                "prediction_binary": int(pred),
                "prediction_type": "refusal_failure" if pred == 1 else "safe",
                "prob_harmful": float(p_[1]),
                "prob_safe": float(p_[0]),
                "parse_status": "ok",
                "input_format": "question_[SEP]_response",
                "max_length": 512,
                "latency_ms": round((time.time() - t0) / max(len(rows), 1) * 1000, 1),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[dna_longformer] done {n} rows in {time.time()-t0:.0f}s")


def generate(limit: int | None = None, batch_size: int = 16):
    unified = os.path.join(out_path("unified"), "do_not_answer_eval.jsonl")
    rows = [json.loads(l) for l in open(unified, encoding="utf-8")]
    if limit:
        rows = rows[:limit]
    tok, model = load_model()
    texts = [f"{r['query']} [SEP] {r['answer']}" for r in rows]
    t0 = time.time()
    probs = predict(tok, model, texts, batch_size=batch_size)
    preds = np.argmax(probs, axis=1).tolist()
    out = os.path.join(out_path("baseline_predictions"), "dna_longformer_harmful_predictions.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    with open(out, "a", encoding="utf-8") as f:
        n = 0
        for r, p, pred in zip(rows, probs, preds):
            if r["id"] in done:
                continue
            f.write(json.dumps({
                "id": r["id"],
                "benchmark": "do_not_answer",
                "group_id": r["group_id"],
                "query": r["query"],
                "answer": r["answer"],
                "method": "dna_longformer_harmful_ro",
                "prediction_binary": int(pred),
                "prediction_type": "refusal_failure" if pred == 1 else "safe",
                "prob_harmful": float(p[1]),
                "prob_safe": float(p[0]),
                "parse_status": "ok",
                "input_format": "question_[SEP]_response",
                "max_length": 512,
                "latency_ms": round((time.time() - t0) / max(len(rows), 1) * 1000, 1),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[dna_longformer] done {n} rows in {time.time()-t0:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    tok, model = load_model()
    if args.sanity:
        sanity(tok, model)
    else:
        generate(args.limit, args.batch_size)


if __name__ == "__main__":
    main()
