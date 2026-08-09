# -*- coding: utf-8 -*-
"""Local inference for Final Student / Neural-Gold / Neural-SoftDistill and Base zero-shot."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from frauddistill.student.collator import neural_collate  # noqa: E402
from frauddistill.student.dataset import ID_TO_LABEL, build_neural_examples  # noqa: E402
from frauddistill.student.model import NeuralStudentConfig, build_neural_student  # noqa: E402

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def load_checkpoint(ckpt_dir: Path, architecture: str = "standard", max_length: int = 512):
    from transformers import AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = NeuralStudentConfig(model_name=BASE_MODEL, architecture=architecture,
                              max_length=max_length, lora_r=32, lora_alpha=64, use_lora=False)
    model = build_neural_student(cfg, freeze_base=True)
    if (ckpt_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
    model.eval()
    return model, tokenizer


def predict_scores(model, tokenizer, rows: list[dict], max_length: int = 512,
                   architecture: str = "standard", micro_batch: int = 8, with_logits: bool = False):
    """Rows need user_query/target_model_answer/id. Returns preds + logits."""
    exs = build_neural_examples(rows, max_length=max_length, use_teacher_soft=True, use_pairwise=False)

    class D(Dataset):
        def __len__(self): return len(exs)
        def __getitem__(self, i): return exs[i]

    loader = DataLoader(D(), batch_size=micro_batch, shuffle=False,
                        collate_fn=lambda b: neural_collate(b, tokenizer, max_length=max_length, architecture=architecture))
    preds, logits_list = [], []
    with torch.no_grad():
        for batch in loader:
            kw = {}
            if batch.get("query_mask") is not None:
                kw["query_mask"] = batch["query_mask"]
                kw["answer_mask"] = batch["answer_mask"]
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **kw)
            probs = torch.softmax(out.logits, dim=-1).numpy()
            if with_logits:
                logits_list.append(out.logits.numpy())
            for i, rid in enumerate(batch["ids"]):
                type_probs = {ID_TO_LABEL[j]: round(float(probs[i, j]), 4) for j in range(4)}
                risk = float(1.0 - probs[i, 0])
                preds.append({"id": rid, "label": "unsafe" if risk >= 0.5 else "safe",
                              "risk_type": max(type_probs, key=type_probs.get),
                              "risk_score": round(risk, 4), "type_probabilities": type_probs})
    return preds, logits_list


def run_inference(rows: list[dict], ckpt_dir: Path, out_path: Path, threshold: float,
                  max_length: int = 512, micro_batch: int = 8, tag: str = "") -> dict:
    t0 = time.time()
    model, tokenizer = load_checkpoint(ckpt_dir, max_length=max_length)
    preds, _ = predict_scores(model, tokenizer, rows, max_length=max_length, micro_batch=micro_batch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in preds:
            p = dict(p)
            p["threshold"] = threshold
            p["model"] = tag
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    del model, tokenizer
    import gc
    gc.collect()
    return {"n": len(preds), "seconds": round(time.time() - t0, 1), "path": str(out_path)}


def load_predictions(path: Path) -> dict[str, dict]:
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["id"]] = r
    return out
